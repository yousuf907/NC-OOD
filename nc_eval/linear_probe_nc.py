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
from nc_criteria import analysis, nc_values
from metrics import compute_ood_det

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
    def __init__(self, cfg, feature_dir, feature_dir2, ckpt_info):
        self.feature_dir = feature_dir
        self.ckpt_info = ckpt_info
        self.cfg = cfg
        self.train_loader, self.val_loader, self.test_loader = load_features(cfg, feature_dir)
        self.train_loader2, self.val_loader2, self.test_loader2 = load_features(cfg, feature_dir2)
        # Which split is the target for knn? (val, test or both)
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

    def _probe(self, train_loader, test_loader, test_loader2, layer_name, l_idx, feat_dim, progress_bar=False):
        classifier = LinearClassifier(dim=feat_dim, num_class=self.cfg.num_cls)
        classifier = torch.nn.DataParallel(classifier).cuda()
        classifier.train()
        optimizer = torch.optim.AdamW(classifier.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay) # wd=1e-2

        criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1).cuda()
        best_acc1 = 0
        best_acc5 = 0

        for e in range(self.cfg.epochs):
            for train_idx, (train_features, train_targets) in enumerate(train_loader):
                train_features, train_targets = train_features.cuda(), train_targets.cuda()

                ## L2 normalize >> NC
                if "projection_head2" in layer_name:
                    #print("Head")
                    train_features = F.normalize(train_features, eps=1e-8, p=2, dim=-1)

                classifier.train()
                optimizer.zero_grad()
                outputs = classifier(train_features)
                loss = criterion(outputs, train_targets)
                loss.backward()
                optimizer.step()

            ## test
            logits = torch.zeros((len(test_loader.dataset), self.cfg.num_cls), dtype=torch.float64)
            probas = torch.zeros((len(test_loader.dataset), self.cfg.num_cls), dtype=torch.float64)
            all_lbls = torch.zeros((len(test_loader.dataset)), dtype=torch.float32)
            start_ix = 0
            for test_idx, (test_features, test_targets) in enumerate(test_loader):
                test_features, test_targets = test_features.cuda(), test_targets.cuda()

                ## L2 normalize >> NC
                if "projection_head2" in layer_name:
                    test_features = F.normalize(test_features, eps=1e-8, p=2, dim=-1)

                classifier.eval()
                outputs = classifier(test_features)
                end_ix = start_ix + len(test_targets)
                ##
                logits[start_ix:end_ix] = outputs
                probas[start_ix:end_ix] = F.softmax(outputs.data, dim=1)
                all_lbls[start_ix:end_ix] = test_targets.squeeze()
                start_ix = end_ix
            ## calculate accuracy
            acc1, acc5 = accuracy(probas, all_lbls, topk=(1, 5))
            ## remember best acc@1 and save checkpoint
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)
            best_acc5 = max(acc5, best_acc5)

        ## energy-scores
        known = torch.logsumexp(logits.data.cpu(), dim=1).numpy()
        print("\nID training has completed and OOD evaluation begins..")

        ## OOD Eval
        logits = torch.zeros((len(test_loader2.dataset), self.cfg.num_cls), dtype=torch.float64)
        probas = torch.zeros((len(test_loader2.dataset), self.cfg.num_cls), dtype=torch.float64)
        all_lbls2 = torch.zeros((len(test_loader2.dataset)), dtype=torch.float32)
        start_ix = 0
        for test_idx, (test_features, test_targets) in enumerate(test_loader2):
            test_features, test_targets = test_features.cuda(), test_targets.cuda()

            ## L2 normalize >> NC
            if "projection_head2" in layer_name:
                test_features = F.normalize(test_features, eps=1e-8, p=2, dim=-1)

            classifier.eval()
            outputs = classifier(test_features)
            end_ix = start_ix + len(test_targets)
            ##
            logits[start_ix:end_ix] = outputs
            probas[start_ix:end_ix] = F.softmax(outputs.data, dim=1)
            all_lbls2[start_ix:end_ix] = test_targets.squeeze()
            start_ix = end_ix

        ## energy-scores
        novel = torch.logsumexp(logits.data.cpu(), dim=1).numpy()

        ## ///////////// Neural Collapse /////////////////
        W_ls, mu_G, mu_c_dict,  Sigma_W, Sigma_B = analysis(self.cfg.num_cls, train_loader)

        #print("Final Sigma_W trace:", np.trace(Sigma_W.cpu().numpy()))
        #print("Final Sigma_B trace:", np.trace(Sigma_B.cpu().numpy()))

        sw_trace = np.trace(Sigma_W.cpu().numpy())
        sb_trace = np.trace(Sigma_B.cpu().numpy())

        ## For projector: learned last layer classifier
        Wo = classifier.state_dict()['module.classifier.weight']
        bo = classifier.state_dict()['module.classifier.bias']

        ## NC eval
        nc1, nc2, nc3, nc4 = nc_values(Sigma_W.cpu().numpy(), Sigma_B.cpu().numpy(), mu_c_dict, mu_G, Wo, bo)
        ## ///////////// Neural Collapse /////////////////

        ## wandb
        #wandb.log({
        #"accuracy": best_acc1,
        #"NC1": nc1,
        #"NC2": nc2,
        #"NC3": nc3,
        #"NC4": nc4,
        #})

        ### OOD Detection
        fpr, auroc = compute_ood_det(known, novel, out_datasets="Dataset", method="Energy")

        ## wandb
        #wandb.log({
        #"FPR": fpr,
        #"AUROC": auroc
        #})

        return best_acc1, nc1, nc2, nc3, nc4, sw_trace, sb_trace, fpr, auroc


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
        neucol1 = []
        neucol2 = []
        neucol3 = []
        neucol4 = []
        s_w = []
        s_b = []
        fpr_scores = []
        auroc_scores = []

        for l_idx in range(0, self.len_layers):
            layer_name, feat_dim = self.train_loader.dataset.set_layer(l_idx)
            print("Layer:", l_idx, "Feature dim:", feat_dim)
            if feat_dim != self.cfg.num_cls:
                print(f'processing layer {layer_name}')
                #self.on_probe_start(layer_name)
                for test_mode in self.test_modes:
                    if test_mode == 'val':
                        self.val_loader.dataset.set_layer(l_idx)
                        self.val_loader2.dataset.set_layer(l_idx)
                        top1_acc, nc1, nc2, nc3, nc4, sigma_w, sigma_b, fpr, auroc = self._probe(self.train_loader, self.val_loader,
                            self.val_loader2, layer_name, l_idx, feat_dim)
                        ###
                        lp_acc = np.append(lp_acc, top1_acc.cpu().numpy())
                        neucol1 = np.append(neucol1, nc1)
                        neucol2 = np.append(neucol2, nc2)
                        neucol3 = np.append(neucol3, nc3)
                        neucol4 = np.append(neucol4, nc4)
                        s_w = np.append(s_w, sigma_w)
                        s_b = np.append(s_b, sigma_b)
                        fpr_scores = np.append(fpr_scores, fpr)
                        auroc_scores = np.append(auroc_scores, auroc)

                    elif test_mode == 'test':
                        self.test_loader.dataset.set_layer(l_idx)
                        self._probe(self.train_loader, self.test_loader, layer_name)
                        self.on_probe_end(l_idx, 'test')

        return lp_acc, neucol1, neucol2, neucol3, neucol4, s_w, s_b, fpr_scores, auroc_scores
