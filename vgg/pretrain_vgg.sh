#!/usr/bin/env bash
EXPT_NAME=VGG17_IN_100c_baseline # No ETF Projector, No Entropy Regularizer
DATA_DIR=/data/datasets/ImageNet-100 # Path to pretrain dataset
SAVE_DIR=./pretrained_vgg17
LR=0.006 # Learning Rate
WD=0.05 # Weight Decay
BS=512 # Batch Size
NUM_CLASS=100 # Num of classes in pretrain dataset
INPUT_SIZE=224 # Input Image Resolution
NUM_EPOCHS=3 #100
GPU=0,1,2,3


CUDA_VISIBLE_DEVICES=${GPU} python -u pretrain_vgg.py \
--data ${DATA_DIR} \
--save_dir ${SAVE_DIR} \
--num_classes ${NUM_CLASS} \
--image_size ${INPUT_SIZE} \
--lr ${LR} \
--wd ${WD} \
--epochs ${NUM_EPOCHS} \
-b ${BS} \
-p 250 \
--alpha 0 \
--output_dim 0 \
--hidden_mlp 0 \
--augmentation \
--ckpt_file ${EXPT_NAME}_${NUM_CLASS}.pth \
--expt_name ${EXPT_NAME} > logs/${EXPT_NAME}.log
