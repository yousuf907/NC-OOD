from config import Config
from linear_probe_det import Linear_Probe, _set_seed ## Energy-based OOD Detection
import wandb
import numpy as np
import os
import shutil
import time

if __name__ == "__main__":
    cfg = Config().parse(None)

    if cfg.set == "ninco":
        from feature_extractor_ninco_resnet import Feature_Extractor
    else:
        from feature_extractor_resnet import Feature_Extractor

    ## Feature Extraction:
    if cfg.task == 'lin_probe':
        proj_name = 'linear_probe'
    proj_name = proj_name if cfg.wandb_project_name=='' else cfg.wandb_project_name

    #set seed
    _set_seed(cfg.seed)
    start = time.time()

    feat_dir="./features"
    if not os.path.exists(feat_dir):
        os.makedirs(feat_dir)

    if not cfg.no_wandb:
        wandb.init(project=proj_name, entity=cfg.wandb_entity, name=cfg.expt_name, config=cfg, mode = cfg.wandb_offline)

    if cfg.extract_features:
        for ckpt_path, feature_path in zip(cfg.ckpt_full_paths, cfg.features_full_paths):
            Feature_Extractor(cfg, ckpt_path, feature_path).extract()

    #### ~~~~ OOD Datasets ~~~~~~ ####
    #datasets = ['ninco']
    datasets = ['imagenet_r200', 'CIFAR100', 'Flower102', 'ninco', 'CUB200', 'Aircrafts', 'Pet', 'Stl']
    cfg = Config().parse(None)

    for k in datasets:
        cfg.set = k
        cfg.features_root = './features/' + k
        print('Processing dataset ', cfg.set)
        #print(cfg.features_root)
        start0 = time.time()

        from pathlib import Path
        if cfg.set == "ninco":
            from feature_extractor_ninco_resnet import Feature_Extractor

        ## Create directories to save features
        cfg.features_full_paths2 = [
            Path(cfg.features_root, conf) for conf in cfg.ckpt_paths
        ]

        for features_fp in cfg.features_full_paths2:
            features_fp.parent.mkdir(parents=True, exist_ok=True)

        for ckpt_path, feature_path in zip(cfg.ckpt_full_paths, cfg.features_full_paths2):
            Feature_Extractor(cfg, ckpt_path, feature_path).extract()

        #### ~~~~~~~~~~~ ####
        for ckpt_path, feature_path, feature_path2, ckpt_info in zip(cfg.ckpt_full_paths, cfg.features_full_paths,
            cfg.features_full_paths2, cfg.ckpt_info):

            lp_acc, fpr_scores, auroc_scores = Linear_Probe(cfg, feature_path, feature_path2, ckpt_info).probe()


        exp_dir = cfg.save_dir
        filename0 = cfg.expt_name + '.npy'
        filename5 = cfg.expt_name + '_' + cfg.set  + '_fpr.npy'
        filename6 = cfg.expt_name + '_' + cfg.set + '_auroc.npy'

        if not os.path.exists(exp_dir):
            os.makedirs(exp_dir)
        ## ID linear probe acc
        np.save(os.path.join(exp_dir, filename0), lp_acc)
        ## OOD Detection
        np.save(os.path.join(exp_dir, filename5), fpr_scores)
        np.save(os.path.join(exp_dir, filename6), auroc_scores)

        print("Files are saved!")
        print("ID Linear Probe Accuracy:", lp_acc)
        print("\nID Error (%):", 100-lp_acc)
        print('\nProcessing dataset ', cfg.set)
        print("\nFPR (%):", 100*fpr_scores)
        print("\nAUROC (%):", 100*auroc_scores)
        print("\n")

        spent_time = int((time.time() - start0) / 60)
        print("\nRuntime (in minutes):", spent_time)

        ## remove saved OOD features
        if os.path.isdir(cfg.features_root):
            shutil.rmtree(cfg.features_root)  # remove dir and all contains
            print("Saved OOD features removed")


    ## remove saved ID features
    features_root_id = "./features/imagenet100"
    if os.path.isdir(features_root_id):
        shutil.rmtree(features_root_id)  # remove dir and all contains
        print("Saved ID features removed")

    spent_time = int((time.time() - start) / 60)
    print("\nTotal Runtime (in minutes):", spent_time)
