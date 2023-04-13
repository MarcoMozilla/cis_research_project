import sys

from f_nn.nnUtil import show_model_info

sys.path.append('..')

import gc
import os.path
from bisect import bisect_right

import netron
import numpy as np
import pandas as pd
import torch
from torch import nn, Tensor
from torch.nn import functional
from torch.nn.modules.loss import _Loss
import matplotlib
from pprint import pprint as p
# matplotlib.use('Agg')

import matplotlib.pyplot as plt

get_tout = lambda tin, k, s, p: (tin + 2 * p + 1 - k) // s


class ResNetSimple(nn.Module):

    def __init__(self, features, clusters, hidden_size=32, layers=3):
        super(ResNetSimple, self).__init__()

        self.F = features

        self.H = hidden_size
        self.K = clusters
        self.L = layers
        self.ln_in_hid = nn.Linear(self.F, self.H)
        self.yn_in_hid_b = nn.BatchNorm1d(self.F)
        # self.yn_in_hid_b = nn.BatchNorm1d(self.H)

        # self.ln_in_out = nn.Linear(self.F, self.K)
        # self.yn_in_out_a = nn.BatchNorm1d(self.F)
        # self.yn_in_out_b = nn.BatchNorm1d(self.K)

        # self.af_in_hid = nn.PReLU()
        # self.af_in_out = nn.PReLU()

        self.lns = nn.ModuleList()

        self.yns_b = nn.ModuleList()
        self.afs = nn.ModuleList()

        for i in range(self.L):
            self.lns.append(nn.Linear(self.H, self.H))
            # self.yns_a.append(nn.BatchNorm1d(self.H))
            self.yns_b.append(nn.BatchNorm1d(self.H))
            self.afs.append(nn.PReLU())

        self.yn_out_b = nn.BatchNorm1d(self.H)
        self.ln_out = nn.Linear(self.H, self.K)

        # self.yn_out_a = nn.BatchNorm1d(self.H)
        # self.af_hid = nn.PReLU()

        self.af_out = nn.Softmax(dim=1)

    def forward(self, x_BxF: torch.Tensor):
        B, F = x_BxF.shape
        z_in_hid_BxH = (self.ln_in_hid(self.yn_in_hid_b(x_BxF)))
        # z_in_out_BxK = (self.yn_in_out_b(self.ln_in_out(self.yn_in_out_a(x_BxF))))

        for i in range(self.L):
            ln_hidden = self.lns[i]
            # yn_hidden_a = self.yns_a[i]
            yn_hidden_b = self.yns_b[i]
            af_hidden = self.afs[i]

            z_in_hid_BxH = (z_in_hid_BxH + af_hidden(ln_hidden(yn_hidden_b(z_in_hid_BxH))))

        # z_hid_out_BxK = (self.yn_out_b(self.ln_out(self.yn_out_a(z_in_hid_BxH))))
        z_hid_out_BxK = (self.ln_out(self.yn_out_b(z_in_hid_BxH)))

        zout_BxK = self.af_out(z_hid_out_BxK)
        return zout_BxK


if __name__ == '__main__':
    pass

    N_batch = 4000
    N_feature = 128
    N_cluster = 32

    resnet = ResNetSimple(features=N_feature, clusters=N_cluster)
    show_model_info(resnet)

    x = torch.randn(N_batch, N_feature)
    y = resnet(x)

    print(f'#x : {x.shape}')
    print(f'#y : {y.shape}')
