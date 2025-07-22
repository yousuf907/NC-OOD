import os
import torch
import numpy as np
import time
from torch.nn import functional as F
from tqdm import tqdm
import scipy.linalg as scilin

def analysis(num_classes, loader):
    device='cuda'
    C = num_classes
    epoch=1
    weight_decay = 0.05 #args.weight_decay

    N             = [0 for _ in range(C)]
    mean          = [0 for _ in range(C)]
    Sw            = 0

    loss          = 0
    #net_correct   = 0
    #NCC_match_net = 0
    mu_c_dict = dict()
    
    with torch.no_grad():
    
        for computation in ['Mean','Cov']:
            pbar = tqdm(total=len(loader), position=0, leave=True)
            for batch_idx, (data, target) in enumerate(loader, start=1):

                data, target = data.to(device), target.to(device)

                h = data

                for c in range(C):
                    # features belonging to class c
                    idxs = (target == c).nonzero(as_tuple=True)[0]
                    
                    if len(idxs) == 0: # If no class-c in this batch
                        continue

                    h_c = h[idxs,:] # B CHW

                    if computation == 'Mean':
                        # update class means
                        mean[c] += torch.sum(h_c, dim=0) # CHW
                        N[c] += h_c.shape[0]
                        
                    elif computation == 'Cov':
                        # update within-class cov

                        z = h_c - mean[c].unsqueeze(0) # B CHW
                        cov = torch.matmul(z.unsqueeze(-1), # B CHW 1
                                        z.unsqueeze(1))  # B 1 CHW
                        Sw += torch.sum(cov, dim=0)

                pbar.update(1)
                pbar.set_description(
                    'Analysis {}\t'
                    'Epoch: {} [{}/{} ({:.0f}%)]'.format(
                        computation,
                        epoch,
                        batch_idx,
                        len(loader),
                        100. * batch_idx/ len(loader)))
                
                #if debug and batch_idx > 20:
                #    break
            pbar.close()
            
            if computation == 'Mean':
                for c in range(C):
                    mean[c] /= N[c]
                    mu_c_dict[c] = mean[c]
                    #print(mu_c_dict[c].shape)
                    M = torch.stack(mean).T
                loss /= sum(N)
            elif computation == 'Cov':
                Sw /= sum(N)

        ## global mean
        muG = torch.mean(M, dim=1, keepdim=True) # CHW 1
        
        ## between-class covariance
        M_ = M - muG
        Sb = torch.matmul(M_, M_.T) / C

        wd = 0.5 * weight_decay # "\lambda" in manuscript, so this is halved
        St = Sw+Sb
        size_last_layer = Sb.shape[0]
        eye_P = torch.eye(size_last_layer).to(device)
        eye_C = torch.eye(C).to(device)

        St_inv = torch.inverse(St + (wd/(wd+1))*(muG @ muG.T) + wd*eye_P)

        w_LS = 1 / C * (M.T - 1 / (1 + wd) * muG.T) @ St_inv
        b_LS = (1/C * torch.ones(C).to(device) - w_LS @ muG.T.squeeze(0)) / (1+wd)
        w_LS_ = torch.cat([w_LS, b_LS.unsqueeze(-1)], dim=1)  # c x n

    return w_LS_, muG.squeeze(), mu_c_dict, Sw, Sb


def compute_ETF(W):
    K = W.shape[0]
    WWT = torch.mm(W, W.T)
    WWT /= torch.norm(WWT, p='fro')

    sub = (torch.eye(K) - 1 / K * torch.ones((K, K))).cuda() / pow(K - 1, 0.5)
    ETF_metric = torch.norm(WWT - sub, p='fro')
    return ETF_metric.detach().cpu().numpy().item()

def compute_W_H_relation(W, mu_c_dict, mu_G):
    K = len(mu_c_dict)
    H = torch.empty(mu_c_dict[0].shape[0], K)
    
    for i in range(K):
        H[:, i] = mu_c_dict[i] - mu_G

    WH = torch.mm(W, H.cuda())
    WH /= torch.norm(WH, p='fro')
    sub = 1 / pow(K - 1, 0.5) * (torch.eye(K) - 1 / K * torch.ones((K, K))).cuda()

    res = torch.norm(WH - sub, p='fro')
    return res.detach().cpu().numpy().item(), H


def compute_Wh_b_relation(W, mu_G, b):
    Wh = torch.mv(W, mu_G.cuda())
    res_b = torch.norm(Wh + b, p='fro')
    return res_b.detach().cpu().numpy().item()


def nc_values(Sigma_W, Sigma_B, mu_c_dict_train, mu_G_train, W, b):
    collapse_metric = np.trace(Sigma_W @ scilin.pinv(Sigma_B)) / len(mu_c_dict_train)
    ETF_metric = compute_ETF(W)
    WH_relation_metric, H = compute_W_H_relation(W, mu_c_dict_train, mu_G_train)
    Wh_b_relation_metric = compute_Wh_b_relation(W, mu_G_train, b)

    return collapse_metric, ETF_metric, WH_relation_metric, Wh_b_relation_metric # NC1, NC2, NC3, NC4