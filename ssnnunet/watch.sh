#!/bin/bash
until [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ];
do
	sleep 10
done
bash tri_train.sh
