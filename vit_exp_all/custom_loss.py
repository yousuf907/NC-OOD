import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class EntropyRegLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.pdist = nn.PairwiseDistance(2, eps=1e-8)

    def pairwise_NNs_inner(self, x):
        """
        Pairwise nearest neighbors for L2-normalized vectors.
        """
        # parwise dot products (= inverse distance)
        dots = torch.mm(x, x.t())
        n = x.shape[0]
        dots.view(-1)[:: (n + 1)].fill_(-1)  # Trick to fill diagonal with -1
        # max inner prod -> min distance
        _, I = torch.max(dots, dim=1)
        return I

    def forward(self, dnn_output, eps=1e-8):
        """
        Args:
            dnn_output (BxD): backbone output of a DNN
        """

        dnn_output = F.normalize(dnn_output, eps=eps, p=2, dim=-1)
        I = self.pairwise_NNs_inner(dnn_output)
        distances = self.pdist(dnn_output, dnn_output[I])  # BxD, BxD -> B
        loss = -torch.log(distances + eps).mean()

        return loss
