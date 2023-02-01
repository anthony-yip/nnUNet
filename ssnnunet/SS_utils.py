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
    test_count_out = int(0.2 * total_count)
    assert test_count_out < label_count_in, "Not enough labeled data to form test set"

    training_count = total_count - test_count_out
    # ensures labeled data (in) can accommodate both test data and labeled training data (out)
    label_count_out = min(int(ratio * training_count), label_count_in - test_count_out)
    unlabel_count_out = training_count - label_count_out
    assert test_count_out + unlabel_count_out + label_count_out == total_count, "Error in SS_utils line 23, num mismatch"
    return test_count_out, unlabel_count_out, label_count_out


def convert_supervised_to_semi_supervised(folder_to_convert, ratio=0.2):
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
    maybe_mkdir_p(join(output_folder_name, "labelsTrL"))
    maybe_mkdir_p(join(output_folder_name, "labelsTs"))

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
    print(nii_files_Tr[:10])
    print(nii_files_Ts[:10])
    print(labels_Tr[:10])

    assert len(labels_Tr) == len(nii_files_Tr), "mismatched labels"

    # do math and obtain the output split
    test_count, unlabel_count, label_count = ss_split_data(len(nii_files_Tr), len(nii_files_Ts), ratio)
    print(f"test count: {test_count}, unlabeled training count: {unlabel_count}, labeled training count: {label_count}")

    # create list representations of output directories
    imagesTs = nii_files_Tr[:test_count]
    labelsTs = labels_Tr[:test_count]
    imagesTrL = nii_files_Tr[test_count:test_count + label_count]
    labelsTrL = labels_Tr[test_count:test_count + label_count]
    imagesTrUL = nii_files_Tr[test_count + label_count:]
    remaining_count = unlabel_count - len(imagesTrUL)
    imagesTrUL += nii_files_Ts
    assert len(imagesTrUL) == unlabel_count

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

#
# def ss_split_4d(input_folder, ratio, num_processes=default_num_threads, overwrite_task_output_id=None):
#     assert isdir(join(input_folder, "imagesTr")) and isdir(join(input_folder, "labelsTr")) and \
#            isfile(join(input_folder, "dataset.json")), \
#         "The input folder must be a valid Task folder from the Medical Segmentation Decathlon with at least the " \
#         "imagesTr and labelsTr subfolders and the dataset.json file"
#
#     while input_folder.endswith("/"):
#         input_folder = input_folder[:-1]
#
#     full_task_name = input_folder.split("/")[-1]
#
#     assert full_task_name.startswith("Task"), "The input folder must point to a folder that starts with TaskXX_"
#
#     first_underscore = full_task_name.find("_")
#     assert first_underscore == 6, "Input folder start with TaskXX with XX being a 3-digit id: 00, 01, 02 etc"
#
#     input_task_id = int(full_task_name[4:6])  # change task like 05 to 105.
#     if overwrite_task_output_id is None:
#         overwrite_task_output_id = input_task_id
#
#     task_name = full_task_name[7:]
#
#     output_folder = join(nnUNet_raw_data, "Task%03.0d_" % overwrite_task_output_id + task_name)
#
#     if isdir(output_folder):
#         shutil.rmtree(output_folder)
#
#     files = []
#     output_dirs = []
#
#     maybe_mkdir_p(output_folder)
#     #my code begins here
#     curr_dir = join(input_folder, "imagesTr")
#     nii_files_Tr = [join(curr_dir, i) for i in os.listdir(curr_dir) if i.endswith(".nii.gz")]
#     curr_dir = join(input_folder, "imagesTs")
#     nii_files_Ts = [join(curr_dir, i) for i in os.listdir(curr_dir) if i.endswith(".nii.gz")]
#     curr_dir = join(input_folder, "labelsTr")
#     labels_Tr = [join(curr_dir, i) for i in os.listdir(curr_dir) if i.endswith(".nii.gz")]
#     nii_files_Tr.sort()
#     nii_files_Ts.sort()
#     labels_Tr.sort()
#     assert len(labels_Tr) == len(nii_files_Tr), "mismatched labels"
#     test_count, unlabel_count, label_count = ss_split_data(len(nii_files_Tr), len(nii_files_Ts), ratio)
#     print(f"test count: {test_count}, unlabeled training count: {unlabel_count}, labeled training count: {label_count}")
#     # gives the correct counts at this point (40 test, 129 unlabeled training, 32 labeled training)
#     maybe_mkdir_p(join(output_folder, "labelsTrL"))
#     maybe_mkdir_p(join(output_folder, "labelsTs"))
#     for i in range(label_count):
#         files.append(nii_files_Tr.pop())
#         output_dirs.append(join(output_folder, "imagesTrL"))
#         shutil.copy2(labels_Tr.pop(), join(output_folder, "labelsTrL"))
#     for i in range(test_count):
#         files.append(nii_files_Tr.pop())
#         output_dirs.append(join(output_folder, "imagesTs"))
#         shutil.copy2(labels_Tr.pop(), join(output_folder, "labelsTs"))
#     for i in range(unlabel_count):
#         if nii_files_Tr:
#             files.append(nii_files_Tr.pop())
#         else:
#             files.append(nii_files_Ts.pop())
#         output_dirs.append(join(output_folder, "imagesTrUL"))
#     p = Pool(num_processes)
#     p.starmap(split_4d_nifti, zip(files, output_dirs))
#     p.close()
#     p.join()
#     shutil.copy(join(input_folder, "dataset.json"), output_folder)
