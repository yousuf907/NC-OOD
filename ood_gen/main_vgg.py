from config import Config
from linear_probe import Linear_Probe, _set_seed
import wandb
import numpy as np
import os
import shutil
import time

if __name__ == "__main__":
    cfg = Config().parse(None)
    start = time.time()

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

    feat_dir="./features"
    if not os.path.exists(feat_dir):
        os.makedirs(feat_dir)

    if not cfg.no_wandb:
        wandb.init(project=proj_name, entity=cfg.wandb_entity, name=cfg.expt_name, config=cfg, mode = cfg.wandb_offline)

    if cfg.extract_features:
        for ckpt_path, feature_path in zip(cfg.ckpt_full_paths, cfg.features_full_paths):
            Feature_Extractor(cfg, ckpt_path, feature_path).extract()

    for ckpt_path, feature_path, ckpt_info in zip(cfg.ckpt_full_paths, cfg.features_full_paths, cfg.ckpt_info):
        lp_acc = Linear_Probe(cfg, feature_path, ckpt_info).probe()


    exp_dir = cfg.save_dir
    filename = cfg.expt_name + '.npy'

    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    np.save(os.path.join(exp_dir, filename), lp_acc)
    print("File is saved")
    print("OOD Gen Accuracy:", lp_acc)
    print("OOD Gen Error:", 100-lp_acc)

    ## remove saved features
    if os.path.isdir(cfg.features_root):
        shutil.rmtree(cfg.features_root)  # remove dir and all contains
        print("Saved features removed")

    spent_time = int((time.time() - start) / 60)
    print("\nTotal Runtime (in minutes):", spent_time)
