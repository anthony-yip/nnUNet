#!/bin/bash
# CUDA_VISIBLE_DEVICES=2 python SS_nnUNet_train_tri_custom.py 2d SS_nnUNetTrainer_tri 104 --supervised=semi > /dev/null 2>>error.txt &
# wait
CUDA_VISIBLE_DEVICES=2 python SS_nnUNet_train_tri_custom.py 3d_fullres SS_nnUNetTrainer_tri 104 --supervised=semi > /dev/null 2>>error1.txt
CUDA_VISIBLE_DEVICES=2 python SS_nnUNet_train_tri_custom.py 3d_lowres SS_nnUNetTrainer_tri 104 --supervised=semi > /dev/null 2>>error2.txt
