#!/bin/bash
### ResNet (Baseline, w/o NC control) Neural Collapse Evaluation
SAVE_DIR=./lp_results_resnet_evalNC
BS=256 # Batch size 128
EPOCH=30 # Training epochs
GPU=0,1,2,3
DATA_DIR=./data/
INPUT=224 # Input Image Resolution
NUM_CLS=100 # Num of classes in ID dataset

## ID Dataset
SET='imagenet100'
EXPT_NAME=evalNC_ResNet18_IN_${NUM_CLS}c_baseline_${SET}
FEAT_DIR=./features/${SET}
#CKPT_DIR=./pretrained_resnet18
CKPT_DIR=/home/yousuf/code_nc_icml/resnet/pretrained_resnet18
ARCH='ResNet18' #'ResNet18' #'ResNet34'
CKPT=best_ResNet18_IN_${NUM_CLS}c_baseline_${NUM_CLS}.pth

CUDA_VISIBLE_DEVICES=${GPU} python -u main_resnet.py \
--expt_name ${EXPT_NAME} \
--task lin_probe \
--no_wandb 0 \
--nc_control 0 \
--data_dir ${DATA_DIR} \
--save_dir ${SAVE_DIR} \
--set ${SET} \
--input_size ${INPUT} \
--seed 0 \
--arch ${ARCH} \
--ckpt_root ${CKPT_DIR} \
--ckpt_paths ${CKPT} \
--ckpt_info ${EXPT_NAME} \
--lr 0.001 \
--weight_decay 0 \
--epochs ${EPOCH} \
--batch_size ${BS} \
--extract_features 1 \
--features_per_file 20000 \
--features_root ${FEAT_DIR}
