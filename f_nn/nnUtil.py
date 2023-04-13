import gc
import os
import sys

sys.path.append('..')

import pandas as pd
import torch
import numpy as np
from torch import nn


def try_gpu(i=0):
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def weight_init(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, nn.Conv1d):
        nn.init.normal_(m.weight.data)
        if m.bias is not None:
            nn.init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        nn.init.normal_(m.weight.data)
        if m.bias is not None:
            nn.init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.normal_(m.weight.data, mean=1, std=0.02)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, mean=1, std=0.02)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        nn.init.normal_(m.weight.data, mean=1, std=0.02)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.data)
        nn.init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                nn.init.orthogonal_(param.data)
            else:
                nn.init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                nn.init.orthogonal_(param.data)
            else:
                nn.init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                nn.init.orthogonal_(param.data)
            else:
                nn.init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                nn.init.orthogonal_(param.data)
            else:
                nn.init.normal_(param.data)


def show_model_info(model):
    table = []
    total_ele = 0
    total_mem = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        tsr = parameter.data

        nele = tsr.nelement()
        sele = tsr.element_size()

        msize = nele * sele

        table.append([name, nele, round(msize / 1024, 2)])
        total_ele += nele
        total_mem += msize
    df = pd.DataFrame(table, columns=["Modules", "Parameters", "Mem (KB)"])

    with pd.option_context('display.max_rows', None,
                           'display.max_columns', None,
                           'display.precision', 3,
                           ):
        print(df)

    print(f"Total Trainable : {total_ele} ele , {round(total_mem / 1024 ** 2, 4)} MB\n")
    return total_ele


class MemCache:

    @staticmethod
    def byte2MB(bt):
        return round(bt / (1024 ** 2), 3)

    def __init__(self):
        self.dctn = {}
        self.max_reserved = 0
        self.max_allocate = 0

    def mclean(self):
        r0 = torch.cuda.memory_reserved(0)
        a0 = torch.cuda.memory_allocated(0)
        f0 = r0 - a0

        for key in list(self.dctn.keys()):
            del self.dctn[key]
        gc.collect()
        torch.cuda.empty_cache()

        r1 = torch.cuda.memory_reserved(0)
        a1 = torch.cuda.memory_allocated(0)
        f1 = r1 - a1

        # print('Mem Free')
        # print(f'Reserved  \t {MemCache.byte2MB(r1 - r0)}MB')
        # print(f'Allocated \t {MemCache.byte2MB(a1 - a0)}MB')
        # print(f'Free      \t {MemCache.byte2MB(f1 - f0)}MB')

    def __setitem__(self, key, value):
        self.dctn[key] = value
        self.max_reserved = max(self.max_reserved, torch.cuda.memory_reserved(0))
        self.max_allocate = max(self.max_allocate, torch.cuda.memory_allocated(0))

    def __getitem__(self, item):
        return self.dctn[item]

    def __delitem__(self, *keys):
        r0 = torch.cuda.memory_reserved(0)
        a0 = torch.cuda.memory_allocated(0)
        f0 = r0 - a0

        for key in keys:
            del self.dctn[key]

        r1 = torch.cuda.memory_reserved(0)
        a1 = torch.cuda.memory_allocated(0)
        f1 = r1 - a1

        print('Cuda Free')
        print(f'Reserved  \t {MemCache.byte2MB(r1 - r0)}MB')
        print(f'Allocated \t {MemCache.byte2MB(a1 - a0)}MB')
        print(f'Free      \t {MemCache.byte2MB(f1 - f0)}MB')

    def show_cuda_info(self):
        t = torch.cuda.get_device_properties(0).total_memory
        r = torch.cuda.memory_reserved(0)
        a = torch.cuda.memory_allocated(0)
        f = r - a

        print('Cuda Info')
        print(f'Total     \t{MemCache.byte2MB(t)} MB')
        print(f'Reserved  \t{MemCache.byte2MB(r)} [{MemCache.byte2MB(self.max_reserved)}] MB')
        print(f'Allocated \t{MemCache.byte2MB(a)} [{MemCache.byte2MB(self.max_allocate)}] MB')
        print(f'Free      \t{MemCache.byte2MB(f)} MB')


MM_device = try_gpu()

MM_dtype = dtype = torch.float32


def array2tensor(aray: np.array,requires_grad=True):
    res = torch.tensor(aray, dtype=MM_dtype, device=MM_device, requires_grad=requires_grad)
    return res

def to_one_hot(y_N: np.ndarray, K_labels):
    return np.identity(K_labels)[y_N]


def fm_one_hot(y_NxK: np.ndarray):
    return np.argmax(y_NxK, axis=1)
