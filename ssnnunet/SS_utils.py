import json
import os
import pickle
import shutil
from collections import OrderedDict
from multiprocessing import Pool

import numpy as np
from batchgenerators.utilities.file_and_folder_operations import join, isdir, maybe_mkdir_p, subfiles, subdirs, isfile
from nnunet.configuration import default_num_threads
from nnunet.experiment_planning.DatasetAnalyzer import DatasetAnalyzer
from nnunet.experiment_planning.common_utils import split_4d_nifti
from nnunet.paths import nnUNet_raw_data, nnUNet_cropped_data, preprocessing_output_dir
from nnunet.preprocessing.cropping import ImageCropper

from natsort import natsorted


def ss_split_data(label_count_in, unlabel_count_in, ratio=0.2):
    # do some math
    total_count = label_count_in + unlabel_count_in
    test_count_out = int(0.1 * total_count)
    val_count_out = int(0.2 * total_count)
    test_plus_val = test_count_out + val_count_out
    assert test_plus_val < label_count_in, "not enough labeled data to form validation and test set."

    training_count = total_count - test_plus_val
    # ensures labeled data (in) can accommodate both test data, validation data and labeled training data (out)
    label_count_out = min(int(ratio * training_count), label_count_in - test_plus_val)
    unlabel_count_out = training_count - label_count_out
    assert test_count_out + val_count_out + unlabel_count_out + label_count_out == total_count, \
        "Error in SS_utils line 23, num mismatch"
    return test_count_out, val_count_out, unlabel_count_out, label_count_out


def convert_supervised_to_semi_supervised(folder_to_convert, ratio=0.2):
    # TO FUTURE DO: add functionality for multi-modality (label and image mismatch)
    # folder_to_convert is in the form of "Task003_Liver"
    while folder_to_convert.endswith("/"):
        folder_to_convert = folder_to_convert[:-1]

    input_folder_name = join(nnUNet_raw_data, folder_to_convert)
    assert isdir(join(input_folder_name, "imagesTr")) and isdir(join(input_folder_name, "labelsTr")) and \
           isdir(join(input_folder_name, "imagesTs")) and isfile(join(input_folder_name, "dataset.json")), \
        "The input folder must be a valid Task folder with at least the " \
        "imagesTr, imagesTs and labelsTr subfolders and the dataset.json file"

    # make empty folders
    output_folder_name = join(nnUNet_raw_data, "Task103_LiverSS")
    if isdir(output_folder_name):
        shutil.rmtree(output_folder_name)
    maybe_mkdir_p(output_folder_name)
    maybe_mkdir_p(join(output_folder_name, "imagesTrL"))
    maybe_mkdir_p(join(output_folder_name, "imagesTrUL"))
    maybe_mkdir_p(join(output_folder_name, "imagesTs"))
    maybe_mkdir_p(join(output_folder_name, "imagesVal"))
    maybe_mkdir_p(join(output_folder_name, "labelsTrL"))
    maybe_mkdir_p(join(output_folder_name, "labelsTs"))
    maybe_mkdir_p(join(output_folder_name, "labelsVal"))

    # collate full file names for all input nii files
    input_Tr_name = join(input_folder_name, "imagesTr")
    input_Ts_name = join(input_folder_name, "imagesTs")
    input_labels_name = join(input_folder_name, "labelsTr")
    nii_files_Tr = [join(input_Tr_name, i) for i in os.listdir(input_Tr_name) if i.endswith(".nii.gz")]
    nii_files_Ts = [join(input_Ts_name, i) for i in os.listdir(input_Ts_name) if i.endswith(".nii.gz")]
    labels_Tr = [join(input_labels_name, i) for i in os.listdir(input_labels_name) if i.endswith(".nii.gz")]
    nii_files_Tr = natsorted(nii_files_Tr)
    nii_files_Ts = natsorted(nii_files_Ts)
    labels_Tr = natsorted(labels_Tr)
    # print(nii_files_Tr[:10])
    # print(nii_files_Ts[:10])
    # print(labels_Tr[:10])

    assert len(labels_Tr) == len(nii_files_Tr), "mismatched labels"

    # do math and obtain the output split
    test_count, val_count, unlabel_count, label_count = ss_split_data(len(nii_files_Tr), len(nii_files_Ts), ratio)
    print(f"test count: {test_count}, validation count: {val_count}, unlabeled training count: {unlabel_count}, "
          f"labeled training count: {label_count}")
    test_plus_val = test_count + val_count

    # create list representations of output directories
    imagesTs = nii_files_Tr[:test_count]
    labelsTs = labels_Tr[:test_count]
    imagesVal = nii_files_Tr[test_count:test_plus_val]
    labelsVal = labels_Tr[test_count:test_plus_val]
    imagesTrL = nii_files_Tr[test_plus_val:test_plus_val + label_count]
    labelsTrL = labels_Tr[test_plus_val:test_plus_val + label_count]
    imagesTrUL = nii_files_Tr[test_plus_val + label_count:]
    remaining_count = unlabel_count - len(imagesTrUL)
    imagesTrUL += nii_files_Ts[:remaining_count]
    assert len(imagesTrUL) == unlabel_count, "incorrect number of unlabeled images formed"

    for file in imagesTs:
        shutil.copy2(file, join(output_folder_name, "imagesTs"))
    for file in labelsTs:
        shutil.copy2(file, join(output_folder_name, "labelsTs"))
    for file in imagesTrL:
        shutil.copy2(file, join(output_folder_name, "imagesTrL"))
    for file in labelsTrL:
        shutil.copy2(file, join(output_folder_name, "labelsTrL"))
    for file in imagesTrUL:
        shutil.copy2(file, join(output_folder_name, "imagesTrUL"))
    for file in imagesVal:
        shutil.copy2(file, join(output_folder_name, "imagesVal"))
    for file in labelsVal:
        shutil.copy2(file, join(output_folder_name, "labelsVal"))
    shutil.copy(join(input_folder_name, "dataset.json"), join(output_folder_name, "dataset.json"))


def create_lists_from_splitted_dataset(base_folder_splitted, folder_marker, label_marker=None):
    # folder_marker such as "imagesTrL"
    lists = []

    json_file = join(base_folder_splitted, "dataset.json")
    with open(json_file) as jsn:
        d = json.load(jsn)
    num_modalities = len(d['modality'].keys())
    patient_list = [i[:-12] for i in os.listdir(join(base_folder_splitted, folder_marker)) if i.endswith("0000.nii.gz")]
    print(f"detected {len(patient_list)} unique patient cases in {join(base_folder_splitted, folder_marker)}.")
    for id in patient_list:
        # id looks like "liver_60"
        # delete this later

        cur_pat = []
        for mod in range(num_modalities):
            # list of all the filenames
            cur_pat.append(join(base_folder_splitted, folder_marker, id +
                                "_%04.0d.nii.gz" % mod))
        if label_marker is not None:
            cur_pat.append(join(base_folder_splitted, label_marker, id + ".nii.gz"))
        else:
            cur_pat.append(None)
        lists.append(cur_pat)
    return lists


def crop(task_string, override=False, num_threads=default_num_threads, ignore_unlabeled=False):
    cropped_out_dir = join(nnUNet_cropped_data, task_string)
    maybe_mkdir_p(cropped_out_dir)

    if override and isdir(cropped_out_dir):
        shutil.rmtree(cropped_out_dir)
        maybe_mkdir_p(cropped_out_dir)

    raw_dir = join(nnUNet_raw_data, task_string)
    subfolders = [("imagesTrL", "labelsTrL"), ("imagesVal", "labelsVal"), ("imagesTrUL", None)]
    if ignore_unlabeled:
        subfolders = subfolders[:-1]

    for images_subfolder, labels_subfolder in subfolders:
        print("doing cropping for ", images_subfolder, " and ", labels_subfolder)
        lists = create_lists_from_splitted_dataset(raw_dir, images_subfolder, labels_subfolder)
        imgcrop = ImageCropper(num_threads, cropped_out_dir)
        imgcrop.ss_run_cropping(lists, cropped_out_dir, images_subfolder, overwrite_existing=override)
        shutil.copy(join(nnUNet_raw_data, task_string, "dataset.json"), cropped_out_dir)
