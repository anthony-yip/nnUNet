from nnunet.paths import network_training_output_dir, preprocessing_output_dir
from batchgenerators.utilities.file_and_folder_operations import *
import numpy as np
from nnunet.preprocessing.preprocessing import get_lowres_axis, get_do_separate_z, resample_data_or_seg


"""
nnUNet_trained_models
--nnUNet
----2d/Task106_LiverSSTri/SS_nnUNetTrainer_tri__SS_nnUNet_plannerv21
------predicted_unlabeled_data
++++++++liver_60.npz
------pseudo_labeled_data

----3d_fullres/Task106_LiverSSTri/SS_nnUNetTrainer_tri__SS_nnUNet_plannerv21
------predicted_unlabeled_data
++++++++liver_60.npz
------pseudo_labeled_data

----3d_lowres/Task106_LiverSSTri/SS_nnUNetTrainer_tri__SS_nnUNet_plannerv21
------predicted_unlabeled_data
++++++++liver_60.npz
------pseudo_labeled_data
"""


def get_case_identifiers(folder):
    case_identifiers = sorted([i[:-4] for i in os.listdir(folder) if i.endswith("npz")])
    return case_identifiers


def dice(a, b):
    # a and b are numpy arrays with shape (c, x, y, z), where c is the number of classes
    a = a.astype(int)
    b = b.astype(int)
    tp = a * b
    fp = a * (1-b)
    fn = (1-a) * b
    axes = (1, 2, 3)
    tp = np.sum(tp, axis=axes)
    fp = np.sum(fp, axis=axes)
    fn - np.sum(fn, axis=axes)
    dc = (2 * tp) / (2 * tp + fp + fn + 1e-8)
    return dc.mean()

def get_plans_identifier(num):
    if num == 0:
        return "SS_nnUNet_plannerv21_data_2D_stage0"
    elif num == 1:
        return "SS_nnUNet_plannerv21_data_3D_stage1"
    elif num == 2:
        return "SS_nnUNet_plannerv21_data_3D_stage0"


def resample_seg_and_append(data, seg, properties, transpose=[0,1,2]):
    assert len(data.shape) == 4, "data must be c x y z"
    assert len(seg.shape) == 4, "seg must be 1 x y z"

    new_shape = data[0].shape
    assert new_shape == properties.get("size_after_resampling"), "size in data and properties don't match."

    if get_do_separate_z(properties.get('original_spacing')):
        do_separate_z = True
        lowres_axis = get_lowres_axis(properties.get('original_spacing'))
    elif get_do_separate_z(properties.get('spacing_after_resampling')):
        do_separate_z = True
        lowres_axis = get_lowres_axis(properties.get('spacing_after_resampling'))
    else:
        do_separate_z = False
        lowres_axis = None
    if lowres_axis is not None and len(lowres_axis) != 1:
        # this happens for spacings like (0.24, 1.25, 1.25) for example. In that case we do not want to resample
        # separately in the out of plane axis
        do_separate_z = False

    seg_reshaped = resample_data_or_seg(seg, new_shape, True, lowres_axis, 1, do_separate_z, order_z=0)
    assert seg_reshaped[0].shape == new_shape, "failed resampling"
    data[-1:] = seg_reshaped
    return data


def main():
    task_name = "Task106_LiverSSTri"
    transpose = [0, 1, 2]
    trainer_planner = "SS_nnUNetTrainer_tri__SS_nnUNet_plannerv21"
    networks = ["2d", "3d_fullres", "3d_lowres"]

    # obtain list of case identifiers and verify each folder has the same case identifiers.
    case_identifiers = []
    for i, network in enumerate(networks):
        folder = join(network_training_output_dir, network, task_name, trainer_planner, "predicted_unlabeled_data")
        case_identifiers[i] = get_case_identifiers(folder)
    assert case_identifiers[0] == case_identifiers[1] == case_identifiers[2], "case identifiers do not match."
    case_identifiers = case_identifiers[0]
    dataset_directory = join(preprocessing_output_dir, task_name, "imagesTrUL")

    # begin main loop
    for identifier in case_identifiers:
        one_hot_list = []
        print(f"processing {identifier}...")
        for i, network in enumerate(networks):
            fname = join(network_training_output_dir, network, task_name, trainer_planner, identifier + ".npz")
            one_hot_list[i] = np.load(fname)["one_hot"]
        assert one_hot_list[0].shape == one_hot_list[1].shape == one_hot_list[2].shape, "one_hots are different sizes."
        dice_AB = dice(one_hot_list[0], one_hot_list[1])
        dice_BC = dice(one_hot_list[1], one_hot_list[2])
        dice_AC = dice(one_hot_list[0], one_hot_list[2])
        # this order is in the order of the excluded seg map i.e. A, B, C or 2d, 3d_fullres, 3d_lowres
        dices = [dice_BC, dice_AC, dice_AB]
        print(f"Dice scores:\n3d_fullres and 3d_lowres: {dice_BC}\n2d and 3d_lowres: {dice_AC}\n2d and 3d_fullres:{dice_AB}")
        assert dice_BC != dice_AC != dice_AB, "how did this happen"

        from_index = 3 - into_index - np.argmin(dices)
        seg = np.argmax(one_hot_list[from_index], 0, keepdims=True)
        folder_with_preprocessed_npz = join(preprocessing_output_dir, task_name, "imagesTrUL", get_plans_identifier(from_index))
        data = np.load(join(folder_with_preprocessed_npz, identifier + ".npz"))["data"]
        prop = load_pickle(join(folder_with_preprocessed_npz, identifier + ".pkl"))
        print(f"combining segmentation from {networks[from_index]} and data from {folder_with_preprocessed_npz}")
        
        data_pseudo = resample_seg_and_append(data, seg, prop, transpose)

        into_index = np.argmax(dices)
        into_network = networks[into_index]
        fname_base = join(network_training_output_dir, into_network, task_name, trainer_planner, "pseudo_labeled_data")
        data_fname = join(fname_base, identifier + ".npz")
        prop_fname = join(fname_base, identifier + ".pkl")
        np.savez_compressed(data_fname, data=data_pseudo)
        write_pickle(prop, prop_fname)
        print(f"finished writing to {data_fname}.")
        # also don't forget to change the SS_nnUNet_train_tri back to the normal.
