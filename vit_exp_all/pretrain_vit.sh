#!/bin/bash
# Pretrain ViT-T/S baseline that does not control Nerual Collapse
## --model vit_small
## --model vit_tiny
## Set alpha=0 to omit entropy regularization
EXPT_NAME=ViT_small_IN100_baseline # or ViT_tiny_IN100_baseline
EPOCHS=100
NUM_CLASS=100 # Num of Classes in ImageNet-100 (ID dataset)
LR=0.0004 #(0.0008 for ViT-Tiny and 0.0004 for ViT-Small when batch size=256)
BS=256
RES=224 # Input Image Resolution
GPU=0,1,2,3 # gpu
SAVE_DIR=./results_vit_small/${EXPT_NAME}
CKPT=${EXPT_NAME}_ckpt.pth


CUDA_VISIBLE_DEVICES=${GPU} python -m torch.distributed.launch \
--nproc_per_node=4 \
--use_env main_pretrain.py \
--model vit_small \
--data-path /data/datasets/ImageNet-100 \
--data-set IMNET \
--output_dir ${SAVE_DIR} \
--ckpt_name ${CKPT} \
--num_class ${NUM_CLASS} \
--input-size ${RES} \
--batch-size ${BS} \
--lr ${LR} \
--smoothing 0.1 \
--epochs ${EPOCHS} \
--alpha 0 \
--mixup 0 \
--cutmix 0 \
--warmup-epochs 5 > logs/${EXPT_NAME}.log
