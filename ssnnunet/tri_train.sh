#!/bin/bash
# CUDA_VISIBLE_DEVICES=0 python ssnnunet/SSnnUNet_train_tri.py 3d_fullres SS_nnUNetTrainer_tri 104 --supervised=full > /dev/null 2>&1 &
# CUDA_VISIBLE_DEVICES=1 python ssnnunet/SSnnUNet_train_tri.py 3d_lowres SS_nnUNetTrainer_tri 104 --supervised=full > /dev/null 2>&1 &
# CUDA_VISIBLE_DEVICES=2 python ssnnunet/SSnnUNet_train_tri.py 2d SS_nnUNetTrainer_tri 104 --supervised=full > /dev/null 2>&1 &
# wait
for i in {0..950..50}
do
	python SS_run_voting.py
	if [ $i -eq 950 ]
	then
	        CUDA_VISIBLE_DEVICES=0 python SSnnUNet_train_tri.py 3d_fullres SS_nnUNetTrainer_tri 104 --supervised=semi --epoch=$i --final_epoch > /dev/null 2>> error1.txt &
	        CUDA_VISIBLE_DEVICES=1 python SSnnUNet_train_tri.py 3d_lowres SS_nnUNetTrainer_tri 104 --supervised=semi --epoch=$i --final_epoch > /dev/null 2>> error2.txt &
	        CUDA_VISIBLE_DEVICES=2 python SSnnUNet_train_tri.py 2d SS_nnUNetTrainer_tri 104 --supervised=semi --epoch=$i --final_epoch > /dev/null 2>> error3.txt &
		wait
		break
	fi
	CUDA_VISIBLE_DEVICES=0 python SSnnUNet_train_tri.py 3d_fullres SS_nnUNetTrainer_tri 104 --supervised=semi --epoch=$i > /dev/null 2>> error1.txt &
	CUDA_VISIBLE_DEVICES=1 python SSnnUNet_train_tri.py 3d_lowres SS_nnUNetTrainer_tri 104 --supervised=semi --epoch=$i > /dev/null 2>> error2.txt &
	CUDA_VISIBLE_DEVICES=2 python SSnnUNet_train_tri.py 2d SS_nnUNetTrainer_tri 104 --supervised=semi --epoch=$i > /dev/null 2>> error3.txt &
	wait
done
