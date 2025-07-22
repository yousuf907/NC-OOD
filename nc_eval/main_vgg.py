from config import Config
from linear_probe_nc import Linear_Probe, _set_seed
import wandb
import numpy as np
import os
import shutil
import time

if __name__ == "__main__":
    cfg = Config().parse(None)

    if cfg.set == "ninco":
        from feature_extractor_ninco import Feature_Extractor
    else:
        from feature_extractor_vgg import Feature_Extractor

    ## Feature extraction:
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

    #### ~~~~~~ Pick any OOD dataset ~~~~~ ####
    cfg = Config().parse(None)
    cfg.set = "ninco"
    cfg.features_root = "./features/ninco"
    from pathlib import Path

    if cfg.set == "ninco":
        from feature_extractor_ninco import Feature_Extractor

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

        lp_acc, nc1, nc2, nc3, nc4, sw, sb, fpr_scores, auroc_scores = Linear_Probe(cfg, feature_path, feature_path2, ckpt_info).probe()
    
    
    ## Uncomment for saving results
    '''
    exp_dir = cfg.save_dir
    filename0 = cfg.expt_name + '.npy'
    filename1 = cfg.expt_name + '_nc1.npy'
    filename2 = cfg.expt_name + '_nc2.npy'
    filename3 = cfg.expt_name + '_nc3.npy'
    filename4 = cfg.expt_name + '_nc4.npy'
    filename5 = cfg.expt_name + '_' + cfg.set  + '_fpr.npy'
    filename6 = cfg.expt_name + '_' + cfg.set + '_auroc.npy'
    filename7 = cfg.expt_name + '_SW.npy'
    filename8 = cfg.expt_name + '_SB.npy'

    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    ## OOD generalization
    np.save(os.path.join(exp_dir, filename0), lp_acc)
    ## NC
    np.save(os.path.join(exp_dir, filename1), nc1)
    np.save(os.path.join(exp_dir, filename2), nc2)
    np.save(os.path.join(exp_dir, filename3), nc3)
    np.save(os.path.join(exp_dir, filename4), nc4)
    ## OOD Detection
    np.save(os.path.join(exp_dir, filename5), fpr_scores)
    np.save(os.path.join(exp_dir, filename6), auroc_scores)
    ## Sw and SB
    np.save(os.path.join(exp_dir, filename7), sw)
    np.save(os.path.join(exp_dir, filename8), sb)
    '''

    print("Files are saved!")
    print("Linear Probe Accuracy:", lp_acc)
    print("\nNC Evaluation Results:")
    print("\nNC1 :", np.round(nc1, 3))
    print("\nNC2 :", np.round(nc2, 3))
    print("\nNC3 :", np.round(nc3, 3))
    print("\nNC4 :", np.round(nc4, 3))
    print("\nID Error:", 100-lp_acc)
    print("\nFPR :", 100*fpr_scores)
    print("\nAUROC :", 100*auroc_scores)
    print("\n")

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
