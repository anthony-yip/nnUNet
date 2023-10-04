#!/bin/bash
# CUDA_VISIBLE_DEVICES=0 python ssnnunet/SSnnUNet_train_tri.py 3d_fullres SS_nnUNetTrainer_tri 104 --supervised=full > /dev/null 2>&1 &
# CUDA_VISIBLE_DEVICES=1 python ssnnunet/SSnnUNet_train_tri.py 3d_lowres SS_nnUNetTrainer_tri 104 --supervised=full > /dev/null 2>&1 &
# CUDA_VISIBLE_DEVICES=2 python ssnnunet/SSnnUNet_train_tri.py 2d SS_nnUNetTrainer_tri 104 --supervised=full > /dev/null 2>&1 &
# wait
exit_codes=0
for i in {0..900..100}
do
	if [ $i -eq 900 ]
	then
	        CUDA_VISIBLE_DEVICES=2 python SSnnUNet_train_tri.py 3d_fullres SS_nnUNetTrainer_tri 104 --supervised=semi --epoch=$i --final_epoch > /dev/null 2>> error1.txt
	        CUDA_VISIBLE_DEVICES=2 python SSnnUNet_train_tri.py 3d_lowres SS_nnUNetTrainer_tri 104 --supervised=semi --epoch=$i --final_epoch > /dev/null 2>> error2.txt
	        CUDA_VISIBLE_DEVICES=2 python SSnnUNet_train_tri.py 2d SS_nnUNetTrainer_tri 104 --supervised=semi --epoch=$i --final_epoch > /dev/null 2>> error3.txt
		break
	fi
	CUDA_VISIBLE_DEVICES=2 python SSnnUNet_train_tri.py 3d_fullres SS_nnUNetTrainer_tri 104 --supervised=semi --epoch=$i > /dev/null 2>> error1.txt
	exit_codes=$(( exit_codes + $? ))
	CUDA_VISIBLE_DEVICES=2 python SSnnUNet_train_tri.py 3d_lowres SS_nnUNetTrainer_tri 104 --supervised=semi --epoch=$i > /dev/null 2>> error2.txt
	exit_codes=$(( exit_codes + $? ))
	CUDA_VISIBLE_DEVICES=2 python SSnnUNet_train_tri.py 2d SS_nnUNetTrainer_tri 104 --supervised=semi --epoch=$i > /dev/null 2>> error3.txt
	exit_codes=$(( exit_codes + $? ))
	if [[ $exit_codes -eq 0 ]]
	then
		python SS_run_voting.py &>> "voting_log$i.txt"
	else
		echo "FAILURE" > "FAILURE_${i}.txt"
		break
	fi
done
