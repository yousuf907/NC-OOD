Code for Pretraining and Evaluating Vision Transformer
======================================================

Code to reproduce ViT results presented in the paper Controlling Neural Collapse Enhances Out-of-Distribution Detection and Transfer Learning.

The code requires the timm package. Keep timm folder inside the vit_exp_all folder.

To use the code, simply run the following scripts.

## Pretraining

- To pretrain ViT Baseline (without NC control): `pretrain_vit.sh`

- To pretrain ViT with NC control (Ours): `pretrain_vit_nc.sh`


## OOD Generalization

- To evaluate OOD generalization performance of ViT Baseline (without NC control): `train_lp.sh`

- To evaluate OOD generalization performance of ViT with NC control (Ours): `train_lp_nc.sh`


## OOD Detection

- To evaluate OOD detection performance of ViT Baseline (without NC control): `eval_ood_det.sh`

- To evaluate OOD detection performance of ViT with NC control (Ours): `eval_ood_det_nc.sh`


## Neural Collapse Evaluation

- To evaluate NC of ViT Baseline (without NC control): `eval_nc_baseline.sh`

- To evaluate NC of ViT with NC control (Ours): `eval_nc_ours.sh`


To get results for other configures, change the relevant arguments.