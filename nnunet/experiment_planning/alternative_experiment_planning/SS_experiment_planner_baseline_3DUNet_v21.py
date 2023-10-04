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

import nnunet
import shutil
from copy import deepcopy
from collections import OrderedDict

import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.experiment_planning.experiment_planner_baseline_3DUNet_v21 import \
    ExperimentPlanner3D_v21
from nnunet.experiment_planning.common_utils import get_pool_and_conv_props
from nnunet.network_architecture.generic_UNet import Generic_UNet
from nnunet.configuration import default_num_threads
from nnunet.paths import *
from nnunet.training.model_restore import recursive_find_python_class


class SS_ExperimentPlanner3D_v21(ExperimentPlanner3D_v21):
    """
    Same as ExperimentPlanner3D_v21, but allows for adapative GPU VRAM size
    """

    def __init__(self, folder_with_cropped_data, preprocessed_output_folder, vram_size=36, tri_train=False,
                 ignore_unlabeled=False):
        # init from experiment_planner_baseline_3DUNet
        self.ignore_unlabeled = ignore_unlabeled
        self.folder_with_cropped_data = folder_with_cropped_data
        self.preprocessed_output_folder = preprocessed_output_folder

        self.folder_with_cropped_data_labeled = join(folder_with_cropped_data, "imagesTrL")
        self.list_of_cropped_npz_files_labeled = subfiles(self.folder_with_cropped_data_labeled, True, None, ".npz",
                                                          True)
        self.list_of_cropped_npz_files = self.list_of_cropped_npz_files_labeled

        if not self.ignore_unlabeled:
            self.folder_with_cropped_data_unlabeled = join(folder_with_cropped_data, "imagesTrUL")
            self.list_of_cropped_npz_files_unlabeled = subfiles(self.folder_with_cropped_data_unlabeled, True, None,
                                                                ".npz", True)
            self.list_of_cropped_npz_files += self.list_of_cropped_npz_files_unlabeled

        self.folder_with_cropped_data_validation = join(folder_with_cropped_data, "imagesVal")
        self.list_of_cropped_npz_files_validation = subfiles(self.folder_with_cropped_data_validation, True, None,
                                                             ".npz", True)

        self.list_of_cropped_npz_files += self.list_of_cropped_npz_files_validation

        self.preprocessor_name = "SS_GenericPreprocessor"

        assert isfile(join(self.folder_with_cropped_data, "dataset_properties.pkl")), \
            "folder_with_cropped_data must contain dataset_properties.pkl"
        self.dataset_properties = load_pickle(join(self.folder_with_cropped_data, "dataset_properties.pkl"))

        self.plans_per_stage = OrderedDict()
        self.plans = OrderedDict()

        self.transpose_forward = [0, 1, 2]
        self.transpose_backward = [0, 1, 2]

        self.unet_base_num_features = 32
        self.unet_max_num_filters = 320
        self.unet_max_numpool = 999
        self.unet_min_batch_size = 4  # this is to allow for 75% unlabeled, 25% labeled during the semi-supervised portion
        self.unet_featuremap_min_edge_length = 4

        self.target_spacing_percentile = 50
        self.anisotropy_threshold = 3
        self.how_much_of_a_patient_must_the_network_see_at_stage0 = 4  # 1/4 of a patient
        self.batch_size_covers_max_percent_of_dataset = 0.05  # all samples in the batch together cannot cover more
        # than 5% of the entire dataset

        self.conv_per_stage = 2

        self.data_identifier = "SS_nnUNet_plannerv21_data_3D"
        self.plans_fname = join(self.preprocessed_output_folder,
                                "SS_nnUNet_plannerv21_plans_3D.pkl")
        self.vram_size = vram_size
        self.tri_train = tri_train
        if not self.tri_train:
            # this is so that only fullres is computed
            self.how_much_of_a_patient_must_the_network_see_at_stage0 = 1000

    def save_properties_of_cropped(self, case_identifier, properties):
        base_filename = f"{case_identifier}.npz"
        filename_labeled = join(self.folder_with_cropped_data_labeled, base_filename)
        if not self.ignore_unlabeled:
            filename_unlabeled = join(self.folder_with_cropped_data_unlabeled, base_filename)
        filename_validation = join(self.folder_with_cropped_data_validation, base_filename)
        if filename_labeled in self.list_of_cropped_npz_files_labeled:
            filename_labeled = filename_labeled[:-3] + "pkl"
            with open(filename_labeled, 'wb') as f:
                pickle.dump(properties, f)
        elif (not self.ignore_unlabeled) and (filename_unlabeled in self.list_of_cropped_npz_files_unlabeled):
            filename_unlabeled = filename_unlabeled[:-3] + "pkl"
            with open(filename_unlabeled, 'wb') as f:
                pickle.dump(properties, f)
        elif filename_validation in self.list_of_cropped_npz_files_validation:
            filename_validation = filename_validation[:-3] + "pkl"
            with open(filename_validation, 'wb') as f:
                pickle.dump(properties, f)
        else:
            raise RuntimeError(f"Could not find npz file when saving properties: {base_filename}")

    def load_properties_of_cropped(self, case_identifier):
        base_filename = f"{case_identifier}.npz"
        filename_labeled = join(self.folder_with_cropped_data_labeled, base_filename)
        if not self.ignore_unlabeled:
            filename_unlabeled = join(self.folder_with_cropped_data_unlabeled, base_filename)
        filename_validation = join(self.folder_with_cropped_data_validation, base_filename)
        if filename_labeled in self.list_of_cropped_npz_files_labeled:
            filename_labeled = filename_labeled[:-3] + "pkl"
            with open(filename_labeled, 'rb') as f:
                return pickle.load(f)
        elif (not self.ignore_unlabeled) and (filename_unlabeled in self.list_of_cropped_npz_files_unlabeled):
            filename_unlabeled = filename_unlabeled[:-3] + "pkl"
            with open(filename_unlabeled, 'rb') as f:
                return pickle.load(f)
        elif filename_validation in self.list_of_cropped_npz_files_validation:
            filename_validation = filename_validation[:-3] + "pkl"
            with open(filename_validation, 'rb') as f:
                return pickle.load(f)
        else:
            print(self.list_of_cropped_npz_files_labeled[:5])
            raise RuntimeError(f"Could not find npz file when loading properties: {base_filename}")

    def get_properties_for_stage(self, current_spacing, original_spacing, original_shape, num_cases,
                                 num_modalities, num_classes):
        """
        We need to adapt ref
        """
        new_median_shape = np.round(original_spacing / current_spacing * original_shape).astype(int)
        dataset_num_voxels = np.prod(new_median_shape) * num_cases

        # the next line is what we had before as a default. The patch size had the same aspect ratio as the median shape of a patient. We swapped t
        # input_patch_size = new_median_shape

        # compute how many voxels are one mm
        input_patch_size = 1 / np.array(current_spacing)

        # normalize voxels per mm
        input_patch_size /= input_patch_size.mean()

        # create an isotropic patch of size 512x512x512mm
        input_patch_size *= 1 / min(input_patch_size) * 512  # to get a starting value
        input_patch_size = np.round(input_patch_size).astype(int)

        # clip it to the median shape of the dataset because patches larger then that make not much sense
        input_patch_size = [min(i, j) for i, j in zip(input_patch_size, new_median_shape)]

        network_num_pool_per_axis, pool_op_kernel_sizes, conv_kernel_sizes, new_shp, \
            shape_must_be_divisible_by = get_pool_and_conv_props(current_spacing, input_patch_size,
                                                                 self.unet_featuremap_min_edge_length,
                                                                 self.unet_max_numpool)
        #     use_this_for_batch_size_computation_3D = 520000000 # 505789440
        # typical ExperimentPlanner3D_v21 configurations use ~8.5GB, but on a A100 we have 40. Allow for more space
        # to be used
        # dynamically adapts to GPU memory

        ref = Generic_UNet.use_this_for_batch_size_computation_3D * (2 / self.unet_min_batch_size) * self.vram_size / 8
        here = Generic_UNet.compute_approx_vram_consumption(new_shp, network_num_pool_per_axis,
                                                            self.unet_base_num_features,
                                                            self.unet_max_num_filters, num_modalities,
                                                            num_classes,
                                                            pool_op_kernel_sizes, conv_per_stage=self.conv_per_stage)
        while here > ref:
            axis_to_be_reduced = np.argsort(new_shp / new_median_shape)[-1]

            tmp = deepcopy(new_shp)
            tmp[axis_to_be_reduced] -= shape_must_be_divisible_by[axis_to_be_reduced]
            _, _, _, _, shape_must_be_divisible_by_new = \
                get_pool_and_conv_props(current_spacing, tmp,
                                        self.unet_featuremap_min_edge_length,
                                        self.unet_max_numpool,
                                        )
            new_shp[axis_to_be_reduced] -= shape_must_be_divisible_by_new[axis_to_be_reduced]

            # we have to recompute numpool now:
            network_num_pool_per_axis, pool_op_kernel_sizes, conv_kernel_sizes, new_shp, \
                shape_must_be_divisible_by = get_pool_and_conv_props(current_spacing, new_shp,
                                                                     self.unet_featuremap_min_edge_length,
                                                                     self.unet_max_numpool,
                                                                     )

            here = Generic_UNet.compute_approx_vram_consumption(new_shp, network_num_pool_per_axis,
                                                                self.unet_base_num_features,
                                                                self.unet_max_num_filters, num_modalities,
                                                                num_classes, pool_op_kernel_sizes,
                                                                conv_per_stage=self.conv_per_stage)
            # print(new_shp)
        input_patch_size = new_shp

        # batch_size = Generic_UNet.DEFAULT_BATCH_SIZE_3D  # This is what works with 128**3
        # doing this because i want to have 75% unlabeled training data
        batch_size = 4
        batch_size = int(np.floor(max(ref / here, 1) * batch_size))

        # check if batch size is too large
        max_batch_size = np.round(self.batch_size_covers_max_percent_of_dataset * dataset_num_voxels /
                                  np.prod(input_patch_size, dtype=np.int64)).astype(int)
        max_batch_size = max(max_batch_size, self.unet_min_batch_size)
        batch_size = max(1, min(batch_size, max_batch_size))

        do_dummy_2D_data_aug = (max(input_patch_size) / input_patch_size[
            0]) > self.anisotropy_threshold

        plan = {
            'batch_size': batch_size,
            'num_pool_per_axis': network_num_pool_per_axis,
            'patch_size': input_patch_size,
            'median_patient_size_in_voxels': new_median_shape,
            'current_spacing': current_spacing,
            'original_spacing': original_spacing,
            'do_dummy_2D_data_aug': do_dummy_2D_data_aug,
            'pool_op_kernel_sizes': pool_op_kernel_sizes,
            'conv_kernel_sizes': conv_kernel_sizes,
        }
        return plan

    def run_preprocessing(self, num_threads):
        subfolders = ["imagesTrL", "imagesVal"]
        for subfolder in subfolders:
            if os.path.isdir(join(self.preprocessed_output_folder, subfolder, "gt_segmentations")):
                shutil.rmtree(join(self.preprocessed_output_folder, subfolder, "gt_segmentations"))
            assert subfolder != "imagesTrUL"
            shutil.copytree(join(self.folder_with_cropped_data, subfolder, "gt_segmentations"),
                            join(self.preprocessed_output_folder, subfolder, "gt_segmentations"))
        normalization_schemes = self.plans['normalization_schemes']
        use_nonzero_mask_for_normalization = self.plans['use_mask_for_norm']
        intensityproperties = self.plans['dataset_properties']['intensityproperties']
        preprocessor_class = recursive_find_python_class([join(nnunet.__path__[0], "preprocessing")],
                                                         self.preprocessor_name, current_module="nnunet.preprocessing")
        assert preprocessor_class is not None
        preprocessor = preprocessor_class(normalization_schemes, use_nonzero_mask_for_normalization,
                                          self.transpose_forward,
                                          intensityproperties)
        target_spacings = [i["current_spacing"] for i in self.plans_per_stage.values()]
        if self.plans['num_stages'] > 1 and not isinstance(num_threads, (list, tuple)):
            num_threads = (default_num_threads, num_threads)
        elif self.plans['num_stages'] == 1 and isinstance(num_threads, (list, tuple)):
            num_threads = num_threads[-1]
        if not self.ignore_unlabeled:
            subfolders = ["imagesTrL", "imagesTrUL", "imagesVal"]
        for subfolder in subfolders:
            preprocessor.run(target_spacings, join(self.folder_with_cropped_data, subfolder),
                             join(self.preprocessed_output_folder, subfolder),
                             self.plans['data_identifier'], num_threads)
