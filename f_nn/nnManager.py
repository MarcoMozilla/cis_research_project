import sys

from e_main.tools import Watch
from f_nn.nnUtil import MM_dtype, MM_device, array2tensor, weight_init

sys.path.append('..')

import gc
import json
import os.path
from bisect import bisect_right

import numpy as np
import pandas as pd
import torch
from torch import nn


def get_gen_balance_nidx(cidx2nidxs, N_samples_per_cluster=None):
    if N_samples_per_cluster is None:
        N_samples_per_cluster = max([len(nidxs) for nidxs in cidx2nidxs])

    N_clusters = len(cidx2nidxs)

    print(f"#cluster x #samples_per_cluster = {N_clusters}*{N_samples_per_cluster} = {N_clusters * N_samples_per_cluster}")

    while True:
        res_idxs = np.concatenate([np.random.choice(nidxs, N_samples_per_cluster) for nidxs in cidx2nidxs])
        yield res_idxs


class ModelManager:

    def __init__(self, model, fpath):
        self.model = model.to(dtype=MM_dtype, device=MM_device)
        self.fpath = fpath
        try:
            self.model.load_state_dict(torch.load(self.fpath))
            print(f'load NN model state')
            self.model.eval()
        except Exception:
            pass
            print(f'load NN model state fail ; init state')

            self.model.apply(weight_init)

    def train_with_minibatch(self, X: np.ndarray, y: np.ndarray, gen_nidxs, lr=0.01, epoch=10000, epoch_show_every=100, loss_fctn=nn.MSELoss()):
        optimizer = torch.optim.NAdam(self.model.parameters(), lr=lr)

        w = Watch()
        self.model.train(True)
        for ep in range(epoch):

            idxs = next(gen_nidxs)
            tensor_X = array2tensor(X[idxs])
            tensor_y = array2tensor(y[idxs])

            optimizer.zero_grad()
            tensor_y_pred = self.model(tensor_X)
            tensor_loss = loss_fctn(tensor_y_pred, tensor_y)
            ls = tensor_loss.detach().cpu().numpy().tolist()
            if np.isnan(ls):
                print('shut down for nan loss')
                break

            tensor_loss.backward()
            optimizer.step()

            if ep % epoch_show_every == 0:
                ls = tensor_loss.detach().cpu().numpy().tolist()

                cost = w.see()
                remain = (epoch - ep - 1) / epoch_show_every * cost
                print(f'{ep}/{epoch} [{len(idxs)}] loss {np.round(ls, 6)} cost {Watch.pdtd2HMS(cost)} remain {Watch.pdtd2HMS(remain)}')
                self.save()

        self.model.train(False)

    def train_with_batch(self, X: np.ndarray, y: np.ndarray, lr=0.1, epoch=10000, epoch_show_every=10, loss_fctn=nn.MSELoss()):
        optimizer = torch.optim.NAdam(self.model.parameters(), lr=lr)

        tensor_X = array2tensor(X)
        tensor_y = array2tensor(y)

        w = Watch()
        self.model.train(True)
        for ep in range(epoch):
            optimizer.zero_grad()
            tensor_y_pred = self.model(tensor_X)
            tensor_loss = loss_fctn(tensor_y_pred, tensor_y)
            ls = tensor_loss.detach().cpu().numpy().tolist()
            if np.isnan(ls):
                print('shut down for nan loss')
                break

            tensor_loss.backward()
            optimizer.step()

            if ep % epoch_show_every == 0:
                ls = tensor_loss.detach().cpu().numpy().tolist()

                cost = w.see()
                remain = (epoch - ep - 1) / epoch_show_every * cost
                print(f'{ep}/{epoch} [{len(X)}] loss {np.round(ls, 6)} cost {Watch.pdtd2HMS(cost)} remain {Watch.pdtd2HMS(remain)}')
                self.save()

        self.model.train(False)

    def save(self):
        torch.save(self.model.state_dict(), self.fpath)
        print(f'save to {self.fpath}')

    def predict(self, pred_X: np.ndarray) -> np.ndarray:

        self.model.train(False)
        pred_Y = self.model(array2tensor(pred_X)).detach().cpu().numpy()
        self.model.train(True)
        return pred_Y


if __name__ == '__main__':
    pass
