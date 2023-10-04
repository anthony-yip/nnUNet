#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
import shutil
from _warnings import warn
from collections import OrderedDict
from typing import Tuple
from time import time
from tqdm import trange


import numpy as np
import torch
from nnunet.training.data_augmentation.data_augmentation_moreDA import get_moreDA_augmentation_train, \
    get_moreDA_augmentation_val
from nnunet.training.loss_functions.deep_supervision import MultipleOutputLoss2
from nnunet.utilities.to_torch import maybe_to_torch, to_cuda
from nnunet.network_architecture.generic_UNet import Generic_UNet
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.network_architecture.neural_network import SegmentationNetwork
from nnunet.training.data_augmentation.default_data_augmentation import default_2D_augmentation_params, \
    get_patch_size, default_3D_augmentation_params
from nnunet.training.dataloading.dataset_loading import load_dataset, SS_DataLoader3D, SS_DataLoader2D, unpack_dataset
from nnunet.training.network_training.nnUNetTrainer import nnUNetTrainer
from nnunet.utilities.nd_softmax import softmax_helper
import torch.backends.cudnn as cudnn
from torch import nn
from torch.cuda.amp import autocast
from nnunet.training.learning_rate.poly_lr import poly_lr
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.preprocessing.preprocessing import get_lowres_axis, get_do_separate_z, resample_data_or_seg

import sys
import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt


class SS_nnUNetTrainer_tri(nnUNetTrainer):
    """
    nnUNetTrainer for semi-supervised training (self-training) (so-far)
    """

    def __init__(self, plans_file, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False, learning_rate=1e-2):
        super().__init__(plans_file, 'all', output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.max_num_epochs = 1000
        self.initial_lr = learning_rate
        self.deep_supervision_scales = None
        self.ds_loss_weights = None
        self.pin_memory = True
        # mine
        self.init_args = (plans_file, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                          deterministic, fp16, learning_rate)
        self.subfolders = ["imagesTrL", "imagesTrUL", "imagesVal"]
        self.dataset_trul = None
        self.datasets = OrderedDict()
        self.supervised_mode = None
        self.save_every = 100

    def initialize(self, training=True, force_load_plans=False):
        """
        - replaced get_default_augmentation with get_moreDA_augmentation
        - enforce to only run this code once
        - loss function wrapper for deep supervision

        :param training:
        :param force_load_plans:
        :return:
        """
        if not self.was_initialized:
            maybe_mkdir_p(self.output_folder)

            if force_load_plans or (self.plans is None):
                self.load_plans_file()
            self.process_plans(self.plans)

            # setup parameters for data augmentation in self.data_aug_params
            self.setup_DA_params()

            ################# Here we wrap the loss for deep supervision ############
            # we need to know the number of outputs of the network
            net_numpool = len(self.net_num_pool_op_kernel_sizes)

            # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
            # this gives higher resolution outputs more weight in the loss
            weights = np.array([1 / (2 ** i) for i in range(net_numpool)])

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            mask = np.array([True] + [True if i < net_numpool - 1 else False for i in range(1, net_numpool)])
            weights[~mask] = 0
            weights = weights / weights.sum()
            self.ds_loss_weights = weights
            # now wrap the loss
            self.loss = MultipleOutputLoss2(self.loss, self.ds_loss_weights)
            ################# END ###################

            # prepare the dataloaders and augmenters
            if training:
                self.dl_tr, self.dl_val = self.get_basic_generators()
                # time how long it takes to do this
                self.tr_gen = get_moreDA_augmentation_train(
                    self.dl_tr, self.data_aug_params['patch_size_for_spatialtransform'], self.data_aug_params,
                    deep_supervision_scales=self.deep_supervision_scales,
                    pin_memory=self.pin_memory,
                    use_nondetMultiThreadedAugmenter=False
                )
                self.val_gen = get_moreDA_augmentation_val(
                    self.dl_val, self.data_aug_params, deep_supervision_scales=self.deep_supervision_scales,
                    pin_memory=self.pin_memory, use_nondetMultiThreadedAugmenter=False)
                self.print_to_log_file("TRAINING KEYS:\n %s" % (str(self.dataset_tr.keys())),
                                       also_print_to_console=False)
                self.print_to_log_file("VALIDATION KEYS:\n %s" % (str(self.dataset_val.keys())),
                                       also_print_to_console=False)
            else:
                pass

            self.initialize_network()
            self.initialize_optimizer_and_scheduler()

            assert isinstance(self.network, (SegmentationNetwork, nn.DataParallel))
        else:
            self.print_to_log_file('self.was_initialized is True, not running self.initialize again')
        self.was_initialized = True

    def initialize_network(self):
        """
        - momentum 0.99
        - SGD instead of Adam
        - self.lr_scheduler = None because we do poly_lr
        - deep supervision = True
        - i am sure I forgot something here

        Known issue: forgot to set neg_slope=0 in InitWeights_He; should not make a difference though
        :return:
        """
        if self.threeD:
            conv_op = nn.Conv3d
            dropout_op = nn.Dropout3d
            norm_op = nn.InstanceNorm3d

        else:
            conv_op = nn.Conv2d
            dropout_op = nn.Dropout2d
            norm_op = nn.InstanceNorm2d

        norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        dropout_op_kwargs = {'p': 0, 'inplace': True}
        net_nonlin = nn.LeakyReLU
        net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        # self.net_num_pool_op_kernel_sizes = [[1, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]]
        # self.net_conv_kernel_sizes = [[1, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]]
        # self.conv_pad_sizes = [[0, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]]
        # self.conv_per_stage = 2
        self.network = Generic_UNet(self.num_input_channels, self.base_num_features, self.num_classes,
                                    len(self.net_num_pool_op_kernel_sizes),
                                    self.conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
                                    dropout_op_kwargs,
                                    net_nonlin, net_nonlin_kwargs, True, False, lambda x: x, InitWeights_He(1e-2),
                                    self.net_num_pool_op_kernel_sizes, self.net_conv_kernel_sizes, False, True, True)
        if torch.cuda.is_available():
            self.network.cuda()
        self.network.inference_apply_nonlin = softmax_helper

    def initialize_optimizer_and_scheduler(self):
        assert self.network is not None, "self.initialize_network must be called first"
        self.optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,
                                         momentum=0.99, nesterov=True)
        self.lr_scheduler = None

    def get_basic_generators(self):
        for subfolder in self.subfolders:
            folder_with_preprocessed_data = join(self.dataset_directory, subfolder, self.plans['data_identifier'] +
                                                 "_stage%d" % self.stage)
            # load_dataset just adds the paths of the npz files into the self.datasets dictionary
            self.datasets[subfolder] = load_dataset(folder_with_preprocessed_data)
            if self.unpack_data:
                print("unpacking subfolder: ", subfolder)
                unpack_dataset(folder_with_preprocessed_data)
                print("done")
        # this is for the first phase of training with purely labeled data. We will amend the data loader after each epoch.
        self.dataset_tr = self.datasets["imagesTrL"]
        self.dataset_val = self.datasets["imagesVal"]
        assert isinstance(self.dataset_tr, OrderedDict)
        assert isinstance(self.dataset_val, OrderedDict)

        if self.threeD:
            dl_tr = SS_DataLoader3D(self.dataset_tr, self.basic_generator_patch_size, self.patch_size, self.batch_size,
                                    False, oversample_foreground_percent=self.oversample_foreground_percent,
                                    pad_mode="constant", pad_sides=self.pad_all_sides, memmap_mode='r')
            dl_val = SS_DataLoader3D(self.dataset_val, self.patch_size, self.patch_size, self.batch_size, False,
                                     oversample_foreground_percent=self.oversample_foreground_percent,
                                     pad_mode="constant", pad_sides=self.pad_all_sides, memmap_mode='r')
        else:
            dl_tr = SS_DataLoader2D(self.dataset_tr, self.basic_generator_patch_size, self.patch_size, self.batch_size,
                                    oversample_foreground_percent=self.oversample_foreground_percent,
                                    pad_mode="constant", pad_sides=self.pad_all_sides, memmap_mode='r')
            dl_val = SS_DataLoader2D(self.dataset_val, self.patch_size, self.patch_size, self.batch_size,
                                     oversample_foreground_percent=self.oversample_foreground_percent,
                                     pad_mode="constant", pad_sides=self.pad_all_sides, memmap_mode='r')
        return dl_tr, dl_val

    def run_online_evaluation(self, output, target):
        """
        due to deep supervision the return value and the reference are now lists of tensors. We only need the full
        resolution output because this is what we are interested in in the end. The others are ignored
        :param output:
        :param target:
        :return:
        """
        target = target[0]
        output = output[0]
        return super().run_online_evaluation(output, target)

    def validate(self, do_mirroring: bool = True, use_sliding_window: bool = True,
                 step_size: float = 0.5, save_softmax: bool = True, use_gaussian: bool = True, overwrite: bool = True,
                 validation_folder_name: str = 'validation_raw', debug: bool = False, all_in_gpu: bool = False,
                 segmentation_export_kwargs: dict = None, run_postprocessing_on_folds: bool = True):
        """
        We need to wrap this because we need to enforce self.network.do_ds = False for prediction
        """
        ds = self.network.do_ds
        self.network.do_ds = False
        ret = super().validate(do_mirroring=do_mirroring, use_sliding_window=use_sliding_window, step_size=step_size,
                               save_softmax=save_softmax, use_gaussian=use_gaussian,
                               overwrite=overwrite, validation_folder_name=validation_folder_name, debug=debug,
                               all_in_gpu=all_in_gpu, segmentation_export_kwargs=segmentation_export_kwargs,
                               run_postprocessing_on_folds=run_postprocessing_on_folds)

        self.network.do_ds = ds
        return ret

    def predict_preprocessed_data_return_seg_and_softmax(self, data: np.ndarray, do_mirroring: bool = True,
                                                         mirror_axes: Tuple[int] = None,
                                                         use_sliding_window: bool = True, step_size: float = 0.5,
                                                         use_gaussian: bool = True, pad_border_mode: str = 'constant',
                                                         pad_kwargs: dict = None, all_in_gpu: bool = False,
                                                         verbose: bool = True, mixed_precision=True) -> Tuple[
        np.ndarray, np.ndarray]:
        """
        We need to wrap this because we need to enforce self.network.do_ds = False for prediction
        """
        ds = self.network.do_ds
        self.network.do_ds = False
        ret = super().predict_preprocessed_data_return_seg_and_softmax(data,
                                                                       do_mirroring=do_mirroring,
                                                                       mirror_axes=mirror_axes,
                                                                       use_sliding_window=use_sliding_window,
                                                                       step_size=step_size, use_gaussian=use_gaussian,
                                                                       pad_border_mode=pad_border_mode,
                                                                       pad_kwargs=pad_kwargs, all_in_gpu=all_in_gpu,
                                                                       verbose=verbose,
                                                                       mixed_precision=mixed_precision)
        self.network.do_ds = ds
        return ret

    def run_iteration(self, data_generator, do_backprop=True, run_online_evaluation=False):
        """
        gradient clipping improves training stability

        :param data_generator:
        :param do_backprop:
        :param run_online_evaluation:
        :return:
        """
        data_dict = next(data_generator)
        data = data_dict['data']
        target = data_dict['target']

        data = maybe_to_torch(data)
        target = maybe_to_torch(target)

        if torch.cuda.is_available():
            data = to_cuda(data)
            target = to_cuda(target)

        self.optimizer.zero_grad()

        if self.fp16:
            with autocast():
                output = self.network(data)
                del data
                # output is the segmentation output for the minibatch (one-hot encoding)
                # target is the ground truth segmentation output (no one-hot encoding)
                # since it's deep supervision, both output and target are actually tuples of these ^^ at various resolutions
                l = self.loss(output, target)

            if do_backprop:
                self.amp_grad_scaler.scale(l).backward()
                self.amp_grad_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.amp_grad_scaler.step(self.optimizer)
                self.amp_grad_scaler.update()
        else:
            output = self.network(data)
            del data
            l = self.loss(output, target)

            if do_backprop:
                l.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.optimizer.step()

        if run_online_evaluation:
            self.run_online_evaluation(output, target)

        del target

        return l.detach().cpu().numpy()

    def setup_DA_params(self):
        """
        - we increase roation angle from [-15, 15] to [-30, 30]
        - scale range is now (0.7, 1.4), was (0.85, 1.25)
        - we don't do elastic deformation anymore

        :return:
        """

        self.deep_supervision_scales = [[1, 1, 1]] + list(list(i) for i in 1 / np.cumprod(
            np.vstack(self.net_num_pool_op_kernel_sizes), axis=0))[:-1]

        if self.threeD:
            self.data_aug_params = default_3D_augmentation_params
            self.data_aug_params['rotation_x'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
            self.data_aug_params['rotation_y'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
            self.data_aug_params['rotation_z'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
            if self.do_dummy_2D_aug:
                self.data_aug_params["dummy_2D"] = True
                self.print_to_log_file("Using dummy2d data augmentation")
                self.data_aug_params["elastic_deform_alpha"] = \
                    default_2D_augmentation_params["elastic_deform_alpha"]
                self.data_aug_params["elastic_deform_sigma"] = \
                    default_2D_augmentation_params["elastic_deform_sigma"]
                self.data_aug_params["rotation_x"] = default_2D_augmentation_params["rotation_x"]
        else:
            self.do_dummy_2D_aug = False
            if max(self.patch_size) / min(self.patch_size) > 1.5:
                default_2D_augmentation_params['rotation_x'] = (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi)
            self.data_aug_params = default_2D_augmentation_params
        self.data_aug_params["mask_was_used_for_normalization"] = self.use_mask_for_norm

        if self.do_dummy_2D_aug:
            self.basic_generator_patch_size = get_patch_size(self.patch_size[1:],
                                                             self.data_aug_params['rotation_x'],
                                                             self.data_aug_params['rotation_y'],
                                                             self.data_aug_params['rotation_z'],
                                                             self.data_aug_params['scale_range'])
            self.basic_generator_patch_size = np.array([self.patch_size[0]] + list(self.basic_generator_patch_size))
        else:
            self.basic_generator_patch_size = get_patch_size(self.patch_size, self.data_aug_params['rotation_x'],
                                                             self.data_aug_params['rotation_y'],
                                                             self.data_aug_params['rotation_z'],
                                                             self.data_aug_params['scale_range'])

        self.data_aug_params["scale_range"] = (0.7, 1.4)
        self.data_aug_params["do_elastic"] = False
        self.data_aug_params['selected_seg_channels'] = [0]
        self.data_aug_params['patch_size_for_spatialtransform'] = self.patch_size

        self.data_aug_params["num_cached_per_thread"] = 2

    def plot_progress(self):
        """
        Should probably by improved
        :return:
        """
        try:
            font = {'weight': 'normal',
                    'size': 18}

            matplotlib.rc('font', **font)

            fig = plt.figure(figsize=(30, 24))
            ax = fig.add_subplot(111)
            ax2 = ax.twinx()

            # since epoch is reset but all_tr_losses keeps growing
            overall_epoch = len(self.all_tr_losses)
            assert overall_epoch < 2001
            x_values = list(range(overall_epoch))

            ax.plot(x_values, self.all_tr_losses, color='b', ls='-', label="loss_tr")

            ax.plot(x_values, self.all_val_losses, color='r', ls='-', label="loss_val, train=False")

            if len(self.all_val_losses_tr_mode) > 0:
                ax.plot(x_values, self.all_val_losses_tr_mode, color='g', ls='-', label="loss_val, train=True")
            if len(self.all_val_eval_metrics) == len(x_values):
                ax2.plot(x_values, self.all_val_eval_metrics, color='g', ls='--', label="evaluation metric")

            ax.set_xlabel("epoch")
            ax.set_ylabel("loss")
            ax2.set_ylabel("evaluation metric")
            ax.legend()
            ax2.legend(loc=9)

            fig.savefig(join(self.output_folder, "progress.png"))
            plt.close()
        except IOError:
            self.print_to_log_file("failed to plot: ", sys.exc_info())

    def maybe_save_checkpoint(self):
        """
        Saves a checkpoint every save_ever epochs.
        :return:
        """

        if self.save_intermediate_checkpoints and (self.epoch % self.save_every == (self.save_every - 1)):
            self.print_to_log_file("saving scheduled checkpoint file...")
            if not self.save_latest_only:
                self.save_checkpoint(join(self.output_folder, f"model_ep_{self.epoch + 1}_{self.supervised_mode}.model"))
            self.save_checkpoint(join(self.output_folder, f"model_latest_{self.supervised_mode}.model"))
            self.print_to_log_file("done")

    def load_latest_checkpoint(self, supervised, train=True):
        assert supervised in ["full", "semi"]
        if isfile(join(self.output_folder, f"model_latest_{supervised}.model")):
            return self.load_checkpoint(join(self.output_folder, f"model_latest_{supervised}.model"), train=train)
        raise RuntimeError("No checkpoint found")

    def load_epoch_checkpoint(self, epoch, train=True):
        if isfile(join(self.output_folder, f"model_ep_{epoch}_semi.model")):
            return self.load_checkpoint(join(self.output_folder, f"model_ep_{epoch}_semi.model"), train=train)
        raise RuntimeError("No epoch checkpoint found")

    def load_final_checkpoint(self, train=False):
        filename = join(self.output_folder, "model_final_checkpoint.model")
        if not isfile(filename):
            raise RuntimeError("Final checkpoint not found. Expected: %s. Please finish the training first." % filename)
        return self.load_checkpoint(filename, train=train)

    def load_FS_checkpoint(self, train=True):
        filename = join(self.output_folder, "model_FS_checkpoint.model")
        if not isfile(filename):
            raise RuntimeError(
                "Fully supervised checkpoint not found. Expected: %s. Please finish the training first." % filename)
        return self.load_checkpoint(filename, train=train)

    def maybe_update_lr(self, epoch=None):
        """
        if epoch is not None we overwrite epoch. Else we use epoch = self.epoch + 1

        (maybe_update_lr is called in on_epoch_end which is called before epoch is incremented.
        herefore we need to do +1 here)

        :param epoch:
        :return:
        """
        if epoch is None:
            ep = self.epoch + 1
        else:
            ep = epoch
        self.optimizer.param_groups[0]['lr'] = poly_lr(ep, self.max_num_epochs, self.initial_lr, 0.9)
        self.print_to_log_file("lr:", np.round(self.optimizer.param_groups[0]['lr'], decimals=6))

    def on_epoch_end(self):
        """
        overwrite patient-based early stopping. Always run to 1000 epochs
        :return:
        """
        super().on_epoch_end()
        continue_training = self.epoch < self.max_num_epochs

        # it can rarely happen that the momentum of nnUNetTrainerV2 is too high for some dataset. If at epoch 100 the
        # estimated validation Dice is still 0 then we reduce the momentum from 0.99 to 0.95
        if self.epoch == 100:
            if self.all_val_eval_metrics[-1] == 0:
                self.optimizer.param_groups[0]["momentum"] = 0.95
                self.network.apply(InitWeights_He(1e-2))
                self.print_to_log_file("At epoch 100, the mean foreground Dice was 0. This can be caused by a too "
                                       "high momentum. High momentum (0.99) is good for datasets where it works, but "
                                       "sometimes causes issues such as this one. Momentum has now been reduced to "
                                       "0.95 and network weights have been reinitialized")
        return continue_training

    def run_training(self, supervised):
        """
        if we run with -c then we need to set the correct lr for the first epoch, otherwise it will run the first
        continued epoch with self.initial_lr

        we also need to make sure deep supervision in the network is enabled for training, thus the wrapper
        :return:
        """
        assert supervised in ["full", "semi"]
        self.maybe_update_lr(self.epoch)  # if we dont overwrite epoch then self.epoch+1 is used which is not what we
        # want at the start of the training
        ds = self.network.do_ds
        self.network.do_ds = True
        self.save_debug_information()
        if supervised == "full":
            self._run_training_internal_full()
        elif supervised == "semi":
            self._run_training_internal_semi()
        else:
            raise RuntimeError
        self.network.do_ds = ds
        return

    def _run_training_internal_full(self):
        print("beginning/continuing fully-supervised training")
        self.supervised_mode = "full"
        if not torch.cuda.is_available():
            self.print_to_log_file(
                "WARNING!!! You are attempting to run training on a CPU (torch.cuda.is_available() is False). This can be VERY slow!")

        _ = self.tr_gen.next()
        _ = self.val_gen.next()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._maybe_init_amp()

        maybe_mkdir_p(self.output_folder)
        self.plot_network_architecture()

        if cudnn.benchmark and cudnn.deterministic:
            warn("torch.backends.cudnn.deterministic is True indicating a deterministic training is desired. "
                 "But torch.backends.cudnn.benchmark is True as well and this will prevent deterministic training! "
                 "If you want deterministic then set benchmark=False")

        if not self.was_initialized:
            self.initialize(True)

        while self.epoch < self.max_num_epochs:
            self.print_to_log_file("\nepoch: ", self.epoch)
            epoch_start_time = time()
            train_losses_epoch = []

            # train one epoch
            self.network.train()

            if self.use_progress_bar:
                with trange(self.num_batches_per_epoch) as tbar:
                    for b in tbar:
                        tbar.set_description("Epoch {}/{}".format(self.epoch + 1, self.max_num_epochs))

                        l = self.run_iteration(self.tr_gen, True)

                        tbar.set_postfix(loss=l)
                        train_losses_epoch.append(l)
            else:
                for _ in range(self.num_batches_per_epoch):
                    l = self.run_iteration(self.tr_gen, True)
                    train_losses_epoch.append(l)

            self.all_tr_losses.append(np.mean(train_losses_epoch))
            self.print_to_log_file("train loss : %.4f" % self.all_tr_losses[-1])

            with torch.no_grad():
                # validation with train=False
                self.network.eval()
                val_losses = []
                for b in range(self.num_val_batches_per_epoch):
                    l = self.run_iteration(self.val_gen, False, True)
                    val_losses.append(l)
                self.all_val_losses.append(np.mean(val_losses))
                self.print_to_log_file("validation loss: %.4f" % self.all_val_losses[-1])

            self.update_train_loss_MA()  # needed for lr scheduler and stopping of training

            continue_training = self.on_epoch_end()

            epoch_end_time = time()

            if not continue_training:
                # allows for early stopping
                break

            self.epoch += 1
            self.print_to_log_file("This epoch took %f s\n" % (epoch_end_time - epoch_start_time))

        output_folder = self.predict_unlabeled()
        shutil.copytree(output_folder, join(self.output_folder, "first_predictions"))

        # set epoch back to 0 for the next round of training, then update lr accordingly
        # re-initialize self.optimizer to reset velocity
        self.epoch = 0
        self.optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,
                                         momentum=0.99, nesterov=True)
        self.maybe_update_lr(0)

        self.epoch = -1
        # this is because self.save_checkpoint increments self.epoch by 1 then saves it

        if self.save_final_checkpoint:
            self.save_checkpoint(join(self.output_folder, "model_FS_checkpoint.model"))
            self.save_checkpoint(join(self.output_folder, "model_ep_0_semi.model"))
        # now we can delete latest as it will be identical with final
        if isfile(join(self.output_folder, "model_latest_full.model")):
            os.remove(join(self.output_folder, "model_latest_full.model"))
        if isfile(join(self.output_folder, "model_latest_full.model.pkl")):
            os.remove(join(self.output_folder, "model_latest_full.model.pkl"))

        self.epoch = 0

    def _run_training_internal_semi(self, epoch_count=100):
        print("beginning/continuing semi-supervised training")
        self.supervised_mode = "semi"
        if not torch.cuda.is_available():
            self.print_to_log_file(
                "WARNING!!! You are attempting to run training on a CPU (torch.cuda.is_available() is False). This can be VERY slow!")

        count = 0
        self.recompute_generators()

        _ = self.tr_gen.next()
        _ = self.val_gen.next()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._maybe_init_amp()

        maybe_mkdir_p(self.output_folder)
        self.plot_network_architecture()

        if cudnn.benchmark and cudnn.deterministic:
            warn("torch.backends.cudnn.deterministic is True indicating a deterministic training is desired. "
                 "But torch.backends.cudnn.benchmark is True as well and this will prevent deterministic training! "
                 "If you want deterministic then set benchmark=False")

        if not self.was_initialized:
            self.initialize(True)

        while self.epoch < self.max_num_epochs and count < epoch_count:
            self.print_to_log_file("\nepoch: ", self.epoch)
            epoch_start_time = time()
            train_losses_epoch = []

            # train one epoch
            self.network.train()

            if self.use_progress_bar:
                with trange(self.num_batches_per_epoch) as tbar:
                    for b in tbar:
                        tbar.set_description("Epoch {}/{}".format(self.epoch + 1, self.max_num_epochs))

                        l = self.run_iteration(self.tr_gen, True)

                        tbar.set_postfix(loss=l)
                        train_losses_epoch.append(l)
            else:
                for _ in range(self.num_batches_per_epoch):
                    l = self.run_iteration(self.tr_gen, True)
                    train_losses_epoch.append(l)

            self.all_tr_losses.append(np.mean(train_losses_epoch))
            self.print_to_log_file("train loss : %.4f" % self.all_tr_losses[-1])

            with torch.no_grad():
                # validation with train=False
                self.network.eval()
                val_losses = []
                for b in range(self.num_val_batches_per_epoch):
                    l = self.run_iteration(self.val_gen, False, True)
                    val_losses.append(l)
                self.all_val_losses.append(np.mean(val_losses))
                self.print_to_log_file("validation loss: %.4f" % self.all_val_losses[-1])

            self.update_train_loss_MA()  # needed for lr scheduler and stopping of training

            continue_training = self.on_epoch_end()

            epoch_end_time = time()

            if not continue_training:
                # allows for early stopping
                break

            self.epoch += 1
            count += 1
            self.print_to_log_file("This epoch took %f s\n" % (epoch_end_time - epoch_start_time))

        assert self.epoch < 1001, "this should not happen"
        self.predict_unlabeled()

        if self.epoch == 1000:
            self.epoch -= 1  # if we don't do this we can get a problem with loading model_final_checkpoint.

            if self.save_final_checkpoint:
                self.save_checkpoint(join(self.output_folder, "model_final_checkpoint.model"))
            # now we can delete latest as it will be identical with final
            if isfile(join(self.output_folder, "model_latest_semi.model")):
                os.remove(join(self.output_folder, "model_latest_semi.model"))
            if isfile(join(self.output_folder, "model_latest_semi.model.pkl")):
                os.remove(join(self.output_folder, "model_latest_semi.model.pkl"))

    def recompute_generators(self):
        folder_with_pseudo_labeled_data = join(self.output_folder, "pseudo_labeled_data")
        if len(os.listdir(folder_with_pseudo_labeled_data)) == 0:
            self.print_to_log_file("no psuedo labels available, not recomputing generator.")
            return
        dataset_pseudo = load_dataset(folder_with_pseudo_labeled_data)
        self.print_to_log_file(f"found {len(dataset_pseudo)}, pseudo labels.")
        if self.unpack_data:
            self.print_to_log_file("unpacking pseudo labeled data")
            unpack_dataset(folder_with_pseudo_labeled_data)
            self.print_to_log_file("done")
        self.print_to_log_file("recomputing generators...")
        assert dataset_pseudo is not None
        if self.threeD:
            self.dl_tr = SS_DataLoader3D(self.dataset_tr, self.basic_generator_patch_size, self.patch_size,
                                         self.batch_size, False, 
                                         oversample_foreground_percent=self.oversample_foreground_percent,
                                         pad_mode="constant", pad_sides=self.pad_all_sides, memmap_mode='r',
                                         pseudo_data=dataset_pseudo)
        else:
            self.dl_tr = SS_DataLoader2D(self.dataset_tr, self.basic_generator_patch_size, self.patch_size,
                                         self.batch_size,
                                         oversample_foreground_percent=self.oversample_foreground_percent,
                                         pad_mode="constant", pad_sides=self.pad_all_sides, memmap_mode='r', 
                                         pseudo_data=dataset_pseudo)
        self.tr_gen = get_moreDA_augmentation_train(
            self.dl_tr, self.data_aug_params['patch_size_for_spatialtransform'], self.data_aug_params,
            deep_supervision_scales=self.deep_supervision_scales,
            pin_memory=self.pin_memory,
            use_nondetMultiThreadedAugmenter=False
        )
        self.print_to_log_file("recomputing generators done.")

    def unresample_and_save_softmax(self, segmentation_softmax: np.ndarray, out_fname: str,
                                             properties_dict: dict):
        # resample to shape after cropping
        current_shape = segmentation_softmax.shape
        shape_original_after_cropping = properties_dict.get('size_after_cropping')

        if np.any([i != j for i, j in zip(np.array(current_shape[1:]), np.array(shape_original_after_cropping))]):
            if get_do_separate_z(properties_dict.get('original_spacing')):
                do_separate_z = True
                lowres_axis = get_lowres_axis(properties_dict.get('original_spacing'))
            elif get_do_separate_z(properties_dict.get('spacing_after_resampling')):
                do_separate_z = True
                lowres_axis = get_lowres_axis(properties_dict.get('spacing_after_resampling'))
            else:
                do_separate_z = False
                lowres_axis = None

            if lowres_axis is not None and len(lowres_axis) != 1:
                # this happens for spacings like (0.24, 1.25, 1.25) for example. In that case we do not want to resample
                # separately in the out of plane axis
                do_separate_z = False

            seg_old_spacing = resample_data_or_seg(segmentation_softmax, shape_original_after_cropping, is_seg=False,
                                                   axis=lowres_axis, order=1, do_separate_z=do_separate_z,
                                                   order_z=0)
        else:
            seg_old_spacing = segmentation_softmax

        # convert to one-hot encoding
        seg_old_spacing = seg_old_spacing > 0.5
        np.savez_compressed(out_fname, one_hot=seg_old_spacing)

    def predict_unlabeled(self, use_sliding_window: bool = True, step_size: float = 0.5, use_gaussian: bool = True,
                          all_in_gpu: bool = False):
        """
        We need to wrap this because we need to enforce self.network.do_ds = False for prediction
        Runs predictions on all unlabeled data, creating a one-hot encoding for each and saving it as a
        {case_identifier}.npz file in output folder/predicted_unlabeled_data
        """
        ds = self.network.do_ds
        current_mode = self.network.training
        self.network.do_ds = False
        self.network.eval()

        assert self.was_initialized, "must initialize, ideally with checkpoint (or train first)"
        self.dataset_trul = self.datasets["imagesTrUL"]

        # predictions as they come from the network go here
        output_folder = join(self.output_folder, "predicted_unlabeled_data")
        pseudo_folder = join(self.output_folder, "pseudo_labeled_data")
        if isdir(output_folder):
            shutil.rmtree(output_folder)
        if isdir(pseudo_folder):
            shutil.rmtree(pseudo_folder)
        maybe_mkdir_p(output_folder)
        maybe_mkdir_p(pseudo_folder)

        # this is for debug purposes
        my_input_args = {'use_sliding_window': use_sliding_window,
                         'step_size': step_size,
                         'use_gaussian': use_gaussian,
                         'all_in_gpu': all_in_gpu,
                         }
        save_json(my_input_args, join(output_folder, "predict_args.json"))

        do_mirroring = False
        mirror_axes = (0,)
        self.print_to_log_file("running predictions for unlabeled data...")
        predict_start_time = time()
        for k in self.dataset_trul.keys():
            data = np.load(self.dataset_trul[k]['data_file'])['data'][:-1]
            if 'properties' in self.dataset_trul[k]:
                properties = self.dataset_trul[k]['properties']
            else:
                properties = load_pickle(self.dataset[k]['properties_file'])
            # this is in the cpu
            # this function returns a tuple (segmentation, softmax) with shapes (x, y, z) and (c, x, y, z)
            softmax_pred = self.predict_preprocessed_data_return_seg_and_softmax(data,
                                                                                 do_mirroring=do_mirroring,
                                                                                 mirror_axes=mirror_axes,
                                                                                 use_sliding_window=use_sliding_window,
                                                                                 step_size=step_size,
                                                                                 use_gaussian=use_gaussian,
                                                                                 all_in_gpu=all_in_gpu,
                                                                                 mixed_precision=self.fp16)[1]
            softmax_pred = softmax_pred.transpose([0] + [i + 1 for i in self.transpose_backward])
            # resample, clip to one-hot, save
            softmax_fname = join(output_folder, k + ".npz")
            self.unresample_and_save_softmax(softmax_pred, softmax_fname, properties)
        predict_end_time = time()
        self.print_to_log_file("running predictions done.")
        self.print_to_log_file(f"predictions saved in {output_folder}.")
        self.print_to_log_file(f"predictions took {predict_end_time - predict_start_time} s")
        self.network.train(current_mode)
        self.network.do_ds = ds
        return output_folder
