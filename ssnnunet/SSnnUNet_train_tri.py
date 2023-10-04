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


import argparse
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.run.default_configuration import get_default_configuration
from nnunet.training.cascade_stuff.predict_next_stage import predict_next_stage
from nnunet.training.network_training.nnUNetTrainer import nnUNetTrainer
from nnunet.training.network_training.nnUNetTrainerCascadeFullRes import nnUNetTrainerCascadeFullRes
from nnunet.training.network_training.nnUNetTrainerV2_CascadeFullRes import nnUNetTrainerV2CascadeFullRes
from nnunet.utilities.task_name_id_conversion import convert_id_to_task_name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("network")
    parser.add_argument("network_trainer")
    parser.add_argument("task", help="can be task name or task id")
    parser.add_argument("-val", "--validation_only", help="use this if you want to only run the validation",
                        action="store_true")
    parser.add_argument("-cf", "--continue_training_full",
                        help="use this if you want to continue a fully-supervised training",
                        action="store_true")
    # parser.add_argument("-cs", "--continue_training_semi",
    #                     help="use this if you want to continue a semi-supervised training",
    #                     action="store_true")
    # parser.add_argument("-ss", "--start_training_semi", help="use this if you want to start semi-supervised training. "
    #                                                          "(must have completed fully-supervised training)",
    #                     action="store_true")
    parser.add_argument("-p", help="data identifier. Only change this if you created a custom experiment planner",
                        default="SS_nnUNet_plannerv21", required=False)
    parser.add_argument("--npz", required=False, default=False, action="store_true", help="if set then nnUNet will "
                                                                                          "export npz files of "
                                                                                          "predicted segmentations "
                                                                                          "in the validation as well. "
                                                                                          "This is needed to run the "
                                                                                          "ensembling step so unless "
                                                                                          "you are developing nnUNet "
                                                                                          "you should enable this")
    parser.add_argument("--fp32", required=False, default=False, action="store_true",
                        help="disable mixed precision training and run old school fp32")
    parser.add_argument("--disable_saving", required=False, action='store_true',
                        help="If set nnU-Net will not save any parameter files (except a temporary checkpoint that "
                             "will be removed at the end of the training). Useful for development when you are "
                             "only interested in the results and want to save some disk space")
    parser.add_argument('--val_disable_overwrite', action='store_false', default=True,
                        help='Validation does not overwrite existing segmentations')
    parser.add_argument("--supervised_mode", type=str, help='Either full or semi.')
    parser.add_argument("--epoch", type=str, help='Used for semi-supervised portion of training. Which epoch to start from? Usually multiples of 100')
    parser.add_argument("--final_epoch", action='store_true', help='Is this the final epoch?')

    args = parser.parse_args()
    task = args.task
    network = args.network
    network_trainer = args.network_trainer
    plans_identifier = args.p
    fp32 = args.fp32
    run_mixed_precision = not fp32
    validation_only = args.validation_only
    supervised = args.supervised_mode
    assert supervised in ["full", "semi"], "--supervised_mode option must either be full or semi"
    continue_training_full = args.continue_training_full

    if not task.startswith("Task"):
        task_id = int(task)
        task = convert_id_to_task_name(task_id)
    assert task == "Task104_LiverSSTri", "wrong task"
    
    plans_file, output_folder_name, dataset_directory, batch_dice, stage, \
        trainer_class = get_default_configuration(network, task, network_trainer, plans_identifier)

    if trainer_class is None:
        raise RuntimeError("Could not find trainer class in nnunet.training.network_training")
    assert issubclass(trainer_class, nnUNetTrainer)

    learning_rate = 5e-4
    # output_folder_name = nnUNet_trained_models/nnUNet/3d_fullres/Task104_LiverSSTri/SS_nnUNetTrainer_tri__SS_nnUNet_plannerv21
    # dataset_directory = nnUNet_preprocessed/Task104_LiverSSTri
    # batch_dice = True
    # plans_file = nnUNet_preprocessed/Task104_LiverSSTri/SS_nnUNet_plannerv21_plans_3D.pkl
    trainer = trainer_class(plans_file, output_folder=output_folder_name, dataset_directory=dataset_directory,
                            batch_dice=batch_dice, stage=stage,
                            deterministic=False,
                            fp16=run_mixed_precision, learning_rate=learning_rate)
    if args.disable_saving:
        trainer.save_final_checkpoint = False  # whether or not to save the final checkpoint
        trainer.save_best_checkpoint = False  # whether or not to save the best checkpoint according to
        # self.best_val_eval_criterion_MA
        trainer.save_intermediate_checkpoints = True  # whether or not to save checkpoint_latest. We need that in case
        # the training crashes
        trainer.save_latest_only = True  # if false it will not store/overwrite _latest but separate files each

    trainer.initialize(not validation_only)
    if supervised == "full":
        if continue_training_full:
            trainer.load_latest_checkpoint(supervised='full')
        trainer.run_training(supervised='full')
        trainer.network.eval()
        trainer.validate(save_softmax=args.npz, run_postprocessing_on_folds=False, overwrite=args.val_disable_overwrite,
                          validation_folder_name="validation_raw_FS")
    elif supervised == "semi":
        epoch = args.epoch
        trainer.load_epoch_checkpoint(epoch)
        assert trainer.epoch == int(epoch), f"epoch mismatch: {trainer.epoch} vs {int(epoch)}"
        trainer.save_latest_only = False
        trainer.run_training(supervised='semi')
        if args.final_epoch:
            trainer.network.eval()
            trainer.validate(save_softmax=args.npz, run_postprocessing_on_folds=False, overwrite=args.val_disable_overwrite,
                          validation_folder_name="validation_raw_SS")


if __name__ == "__main__":
    main()
