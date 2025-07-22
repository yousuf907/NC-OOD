#!/bin/bash
# OOD detection evaluation of ViT that controls NC (Proposed)
## --model evit_small (ViT-Samll + ETF Projector)
## --model evit_tiny (ViT-Tiny + ETF Projector)
SAVE_DIR=./ood_det_results
EPOCHS=30
GPU=0,1,2,3 # gpu
DATA_DIR=./data/
INPUT_SIZE=224 # Input Image Resolution
DATASET='IMNET' # ID dataset > ImageNet-100
INST=ViT_small_IN100_etf_proj_ent_reg # ViT_tiny_IN100_etf_proj_ent_reg
EXPT_NAME=OOD_Det_${INST}_${DATASET}
CKPT=./results_vit_small_nc/${INST}/best_${INST}_ckpt.pth


CUDA_VISIBLE_DEVICES=${GPU} python -m torch.distributed.launch \
--nproc_per_node=4 \
--use_env main_ood_det.py \
--expt_name ${EXPT_NAME} \
--model evit_small \
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
