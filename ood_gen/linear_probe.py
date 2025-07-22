import torch
import wandb
import os
import torch.optim
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
from tqdm.contrib import tqdm
from feature_loader import load_features
from utils import knn_accuracy, AverageMeter, accuracy
import numpy as np
import random


def _set_seed(seed):
    print("Set seed", seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class LinearClassifier(nn.Module):
    """Linear layer to train on top of frozen features"""
    def __init__(self, dim, num_class=100):
        super(LinearClassifier, self).__init__()

        self.classifier = nn.Linear(dim, num_class)
        self.classifier.weight.data.normal_(mean=0.0, std=0.01)
        self.classifier.bias.data.zero_()

    def forward(self, x):
        return self.classifier(x)

class Linear_Probe:
    def __init__(self, cfg, feature_dir, ckpt_info):
        self.feature_dir = feature_dir
        self.ckpt_info = ckpt_info
        self.cfg = cfg
        self.train_loader, self.val_loader, self.test_loader = load_features(cfg, feature_dir)
        self.test_modes = []
        if self.val_loader is not None:
            self.test_modes.append('val')
        if self.test_loader is not None:
            self.test_modes.append('test')
        assert len(self.test_modes) > 0, "Must test on something"
        # How many layers do we wanna probe on
        self.len_layers = self.train_loader.dataset.len_layers()
        # knn usefule variables
        self.K = cfg.knn_K

    def _probe(self, train_loader, test_loader, layer_name, l_idx, feat_dim, progress_bar=False):
        classifier = LinearClassifier(dim=feat_dim, num_class=self.cfg.num_cls)
        classifier = torch.nn.DataParallel(classifier).cuda()
        classifier.train()
        optimizer = torch.optim.AdamW(classifier.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1).cuda()
        best_acc1 = 0
        best_acc5 = 0

        for e in range(self.cfg.epochs):
            for train_idx, (train_features, train_targets) in enumerate(train_loader):
                train_features, train_targets = train_features.cuda(), train_targets.cuda()
                classifier.train()
                optimizer.zero_grad()
                outputs = classifier(train_features)
                loss = criterion(outputs, train_targets)
                loss.backward()
                optimizer.step()

            ## test
            probas = torch.zeros((len(test_loader.dataset), self.cfg.num_cls), dtype=torch.float64)
            all_lbls = torch.zeros((len(test_loader.dataset)), dtype=torch.float32)
            start_ix = 0
            for test_idx, (test_features, test_targets) in enumerate(test_loader):
                test_features, test_targets = test_features.cuda(), test_targets.cuda()
                classifier.eval()
                outputs = classifier(test_features)
                end_ix = start_ix + len(test_targets)
                probas[start_ix:end_ix] = F.softmax(outputs.data, dim=1)
                all_lbls[start_ix:end_ix] = test_targets.squeeze()
                start_ix = end_ix
            ## calculate accuracy
            acc1, acc5 = accuracy(probas, all_lbls, topk=(1, 5))
            ## remember best acc@1 and save checkpoint
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)
            best_acc5 = max(acc5, best_acc5)

        ## wandb
        wandb.log({
        "accuracy": best_acc1,
        "Error": 100-best_acc1
        })
        return best_acc1


    def on_probe_start(self, layer_name):
        # initialize meters
        self.acc_meters = {k:AverageMeter(f"Acc@{k}", "6:4") for k in self.K}

    def on_probe_end(self, layer_idx, mode):
        self.accs = {k:meter.avg for k, meter in self.acc_meters.items()}
        for k, meter in self.acc_meters.items():
            meter.reset()

            # Logging
            if not self.cfg.no_wandb:
                model_config =  'OOD_vs_ID'
                info = 'Linear_Probe'
                wandb.log({
                    f'{info}': self.accs[k]
                })
        return self.accs[k]

    #@torch.no_grad()
    def probe(self):
        lp_acc = []
        for l_idx in range(0, self.len_layers):
            layer_name, feat_dim = self.train_loader.dataset.set_layer(l_idx)
            print("Layer:", l_idx, "Feature dim:", feat_dim)
            if feat_dim != self.cfg.num_cls:
                print(f'processing layer {layer_name}')
                #self.on_probe_start(layer_name)
                for test_mode in self.test_modes:
                    if test_mode == 'val':
                        self.val_loader.dataset.set_layer(l_idx)
                        top1_acc = self._probe(self.train_loader, self.val_loader, layer_name, l_idx, feat_dim)
                        lp_acc = np.append(lp_acc, top1_acc.cpu().numpy())

                    elif test_mode == 'test':
                        self.test_loader.dataset.set_layer(l_idx)
                        self._probe(self.train_loader, self.test_loader, layer_name)
                        self.on_probe_end(l_idx, 'test')
        return lp_acc
