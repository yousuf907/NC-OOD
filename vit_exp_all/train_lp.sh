#!/bin/bash
# OOD generalization evaluation of ViT baseline (w/o NC control)
## --model vit_small
## --model vit_tiny
SAVE_DIR=./ood_gen_results
EPOCHS=30 # Num of epochs to train linear probes
GPU=0,1,2,3 # gpu
DATA_DIR=./data/ # Path to datasets
INPUT_SIZE=224 # Input Image Resolution
INST=ViT_small_IN100_baseline # ViT_tiny_IN100_baseline


for DATASET in 'IMNETR' 'CIFAR' 'Flower102' 'NINCO' 'CUB200' 'Aircraft' 'Pet' 'Stl'
do
    EXPT_NAME=OOD_Gen_${INST}_${DATASET}
    CKPT=./results_vit_small/${INST}/best_${INST}_ckpt.pth

    CUDA_VISIBLE_DEVICES=${GPU} python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --use_env main_lp.py \
    --expt_name ${EXPT_NAME} \
    --model vit_small \
    --data-set ${DATASET} \
    --data-path ${DATA_DIR} \
    --output_dir ${SAVE_DIR} \
    --input-size ${INPUT_SIZE} \
    --batch-size 512 \
    --lr 0.01 \
    --epochs ${EPOCHS} \
    --seed 0 \
    --warmup-epochs 0 \
    --smoothing 0.1 \
    --finetune ${CKPT} \
    --mixup 0.0 \
    --cutmix 0.0 > logs/${EXPT_NAME}.log
done
