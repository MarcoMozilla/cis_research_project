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


class ResNet(nn.Module):

    def __init__(self, features, clusters):
        super(ResNet, self).__init__()

        self.F = features

        self.H = clusters
        self.active = nn.ReLU()

        self.ln1 = nn.Linear(self.F, self.H)

        self.ln2 = nn.Linear(self.H, self.H)
        self.ln3 = nn.Linear(self.H, self.H)

        self.ln4 = nn.Linear(self.H, self.H)
        self.ln5 = nn.Linear(self.H, self.H)

        self.yn1 = nn.BatchNorm1d(self.H)
        self.yn2 = nn.BatchNorm1d(self.H)
        self.yn3 = nn.BatchNorm1d(self.H)
        self.yn4 = nn.BatchNorm1d(self.H)
        self.yn5 = nn.BatchNorm1d(self.H)

        self.ln_out = nn.Linear(self.H, self.H)

        self.factive = nn.Softmax(dim = 1)

    def forward(self, x_BxF: torch.Tensor):
        B, F = x_BxF.shape

        z12_BxH = self.active(self.yn1(self.ln1(x_BxF)))
        z23_BxH = z12_BxH + self.active(self.yn2(self.ln2(z12_BxH)))
        z34_BxH = z23_BxH + self.active(self.yn3(self.ln3(z23_BxH)))

        z45_BxH = z34_BxH + self.active(self.yn4(self.ln4(z34_BxH)))
        z56_BxH = z45_BxH + self.active(self.yn5(self.ln5(z45_BxH)))

        return self.factive(self.ln_out(z56_BxH))


if __name__ == '__main__':
    pass

    N_batch = 4000
    N_feature = 128
    N_cluster = 32

    resnet = ResNet(features=N_feature, clusters=N_cluster)
    show_model_info(resnet)

    x = torch.randn(N_batch, N_feature)
    y = resnet(x)

    print(f'#x : {x.shape}')
    print(f'#y : {y.shape}')
