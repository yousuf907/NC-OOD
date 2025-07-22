# Code for Measuring OOD Generalization or Transfer Learning using Linear Probing

Code to reproduce results presented in the paper Controlling Neural Collapse Enhances Out-of-Distribution Detection and Transfer Learning

To use the code, simply run the following:

- To measure OOD generalization performance for VGG17 Baseline (without NC control): `train_lp_vgg.sh`

- To measure OOD generalization performance for VGG17 with NC control (Ours): `train_lp_vgg_nc.sh`

- To measure OOD generalization performance for ResNet18/34 Baseline (without NC control): `train_lp_resnet.sh`

- To measure OOD generalization performance for ResNet18/34 with NC control (Ours): `train_lp_resnet_nc.sh`

To get results for other configures, change the relevant arguments.