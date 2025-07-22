#!/bin/bash
### ResNet Linear Probing
## Specify 'ResNet18' and 'ResNet34' in arch
# Specify input resolution as 224 in INPUT
# Specify ood dataset names in SET
BS=256 # batch size
EPOCH=30 # Num of epochs to train linear probes
GPU=0,1,2,3 # gpu
SAVE_DIR=./lp_results_resnet_oodg
DATA_DIR=./data/ # Path to OOD Datasets
INPUT=224 # Input Image Resolution
NUM_CLS=100 # Num of Classes in ID Dataset (ImageNet-100)

### Linear Probing (ResNet-18/34 baseline w/o any mechanisms to control Neural Collapse)
for SET in imagenet_r200 CIFAR100 Flower102 ninco CUB200 Aircrafts Pet Stl
do
    EXPT_NAME=LP_ResNet18_IN_${NUM_CLS}c_baseline_${SET}
    FEAT_DIR=./features/${SET} # For Storing Extracted Features
    CKPT_DIR=/home/yousuf/code_nc_icml/resnet/pretrained_resnet18
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
    --arch 'ResNet18' \
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
done
