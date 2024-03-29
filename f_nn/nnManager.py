import os
import pathlib
import sys
import time
import traceback

from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm

from e_main import const
from e_main.preset import Preset
from e_main.tools import Watch
from f_nn.nnUtil import MM_dtype, MM_device, array2tensor, weight_init, fm_one_hot, to_one_hot, MemCache

sys.path.append('..')

import numpy as np
import pandas as pd
import torch
from torch import nn

col_label = 'label'
col_precision = 'precision'
col_recall = 'recall'
col_accuracy = 'accuracy'
col_f1 = 'F1'

fpath_tables = os.path.join(Preset.root, r'd_tables')
fpath_models_custom = os.path.join(Preset.root, r'a_models_custom')


def get_confuseMatrix_and_scoreTable(labels_real, labels_pred, labels=None, path_cfmx=None, path_score=None):
    mtx = confusion_matrix(labels_real, labels_pred, labels=labels)
    df_mtx = pd.DataFrame(mtx, index=labels, columns=labels)

    labels = sorted(list(set(labels_real) | set(labels_pred)), key=lambda lb: labels.index(lb))
    report_score = classification_report([labels.index(lb) for lb in labels_real], [labels.index(lb) for lb in labels_pred], target_names=labels, digits=4, output_dict=True, zero_division=0)
    df_score = pd.DataFrame(report_score).transpose()
    # print(df_score)

    dctn = {}
    dctn[col_accuracy] = df_score['f1-score']['accuracy']
    dctn[col_f1] = df_score['f1-score']['macro avg']
    dctn[col_precision] = df_score['precision']['macro avg']
    dctn[col_recall] = df_score['recall']['macro avg']

    count = 3
    while count >= 0:
        try:
            if path_cfmx:
                df_mtx.to_csv(path_cfmx, encoding=const.ecd_utf8sig)
            if path_score:
                df_score.to_csv(path_score, encoding=const.ecd_utf8sig)
            return dctn
        except Exception as e:
            print(traceback.format_exc())
            count -= 1
            time.sleep(1)

    return dctn


def get_gen_minibatch_ridxs(N_items, max_N_items_per_pass=10000):
    N_items_per_pass = min(N_items, max_N_items_per_pass)
    print(f'N_items_per_pass= {N_items_per_pass}')
    while True:
        res_idxs = np.random.choice(N_items, N_items_per_pass, replace=False)
        yield res_idxs


def get_label_weights(y, N_clusters):
    counts = np.array([np.sum(y == cid) for cid in range(N_clusters)]) + 1
    weights = 1 / counts
    weights = weights / np.sum(weights)
    # weights = weights / np.max(weights)
    print(f'weights = {weights}')
    return weights


class ModelManager:

    def __init__(self, model, fname_model, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, refresh=True):
        self.model = model.to(dtype=MM_dtype, device=MM_device)

        self.fname_model = fname_model
        self.fname_model_body = self.fname_model.replace(pathlib.Path(self.fname_model).suffix, '')

        self.fpath_pt = os.path.join(fpath_models_custom, fname_model)
        self.fpath_pt_view = os.path.join(fpath_models_custom, f'{self.fname_model_body}.pth')

        self.X_train = X_train
        self.y_train = y_train
        self.y1hot_train = to_one_hot(self.y_train, K_labels=self.model.K)
        self.X_test = X_test
        self.y_test = y_test

        self.mc = MemCache()
        self.mc.mclean()
        # self.mc.show_cuda_info()

        loaded = False
        if not refresh:
            try:
                self.model.load_state_dict(torch.load(self.fpath_pt))
                print(f'load NN model state')
                self.model.eval()
                loaded = True
            except Exception:
                pass

        if not loaded:
            print(f'load NN model state fail ; init state')
            self.model.apply(weight_init)

    def train_with_minibatch(self, X: np.ndarray = None, y1hot: np.ndarray = None, gen_ridxs=None, lr=1e-3, epoch=100000, epoch_show_every=100, loss_fctn=None):

        if X is None:
            X = self.X_train
        if y1hot is None:
            y1hot = self.y1hot_train
        if gen_ridxs is None:
            gen_ridxs = get_gen_minibatch_ridxs(len(self.y_train))
        if loss_fctn is None:
            weights = get_label_weights(self.y_train, self.model.K)
            weights_Tensor = array2tensor(weights, requires_grad=False)

            loss_fctn = nn.CrossEntropyLoss(weight=weights_Tensor)

        prev_train_f1 = 0
        forgive = 12
        f1_early_stop = 0.995
        optimizer = torch.optim.NAdam(self.model.parameters(), lr=lr)

        w = Watch()

        self.model.train(True)
        for ep in tqdm(range(epoch), total=epoch):

            idxs = next(gen_ridxs)
            self.mc['tensor_X'] = array2tensor(X[idxs])
            self.mc['tensor_y'] = array2tensor(y1hot[idxs])

            optimizer.zero_grad()
            tensor_y_pred = self.model(self.mc['tensor_X'])
            tensor_loss = loss_fctn(tensor_y_pred, self.mc['tensor_y'])
            ls = tensor_loss.detach().cpu().numpy().tolist()
            if np.isnan(ls):
                print('shut down for nan loss')
                break

            tensor_loss.backward()
            optimizer.step()

            if ep % epoch_show_every == 0:

                cost = w.see()
                remain = (epoch - ep - 1) / epoch_show_every * cost

                bk = False
                tdctn_train, tdctn_test = self.evaluate(self.X_train, self.y_train, self.X_test, self.y_test)

                if tdctn_train[col_f1] >= f1_early_stop:
                    bk = True
                if tdctn_train[col_f1] <= prev_train_f1:
                    forgive -= 1
                    lr = lr * 0.5
                    optimizer = torch.optim.NAdam(self.model.parameters(), lr=lr)
                    print(f'lr = {lr}')
                    if forgive == 0:
                        bk = True
                prev_train_f1 = tdctn_train[col_f1]

                sent_after = ','.join([f"{col}={round(tdctn_train[col], 4)}|{round(tdctn_test[col], 4)}" for col in [col_f1, col_accuracy, col_precision, col_recall]])
                sent_full = f'[score=train|test] {sent_after}'

                print(f'{ep}/{epoch} [{len(idxs)}] loss {np.round(ls, 6)} {sent_full} cost {Watch.pdtd2HMS(cost)} remain {Watch.pdtd2HMS(remain)}')
                self.save()

                if bk:
                    break

            self.mc.mclean()
        self.model.train(False)

    def predict(self, pred_X: np.ndarray) -> np.ndarray:

        training = self.model.training
        self.model.train(False)

        slot_size = 100000

        pred_Y_s = []
        for rid in range(0, pred_X.shape[0], slot_size):
            self.mc['pred_X_tensor'] = array2tensor(pred_X[rid:rid + slot_size, :])
            pred_Y = self.model(self.mc['pred_X_tensor']).detach().cpu().numpy()
            pred_Y_s.append(pred_Y)
            self.mc.mclean()

        self.model.train(training)
        return np.concatenate(pred_Y_s)

    def save(self):
        training = self.model.training
        self.model.train(False)
        torch.save(self.model.state_dict(), self.fpath_pt)
        traced_model = torch.jit.trace(self.model, array2tensor(np.ones((2, self.model.F))))
        torch.jit.save(traced_model, self.fpath_pt_view)
        self.model.train(training)
        # print(f'save to {self.fpath}')

    def evaluate(self, X_train, y_train, X_test, y_test, labels=None):
        if labels is None:
            labels = list(range(self.model.K))

        path_train_cfmx = os.path.join(fpath_tables, f'{self.fname_model_body}.train.cfmx.csv')
        path_train_score = os.path.join(fpath_tables, f'{self.fname_model_body}.train.score.csv')
        path_test_cfmx = os.path.join(fpath_tables, f'{self.fname_model_body}.test.cfmx.csv')
        path_test_score = os.path.join(fpath_tables, f'{self.fname_model_body}.test.score.csv')

        y_train_pred = fm_one_hot(self.predict(X_train))
        y_test_pred = fm_one_hot(self.predict(X_test))

        tdctn_train = get_confuseMatrix_and_scoreTable(y_train, y_train_pred, labels=labels, path_cfmx=path_train_cfmx, path_score=path_train_score)
        tdctn_test = get_confuseMatrix_and_scoreTable(y_test, y_test_pred, labels=labels, path_cfmx=path_test_cfmx, path_score=path_test_score)

        return tdctn_train, tdctn_test


if __name__ == '__main__':
    pass

    labels_real = [1, 0, 1]
    labels_pred = [1, 1, 1]

    labels = [0, 1, 2]
    mtx = confusion_matrix(labels_real, labels_pred, labels=labels)

    df_mtx = pd.DataFrame(mtx, index=labels, columns=labels)

    labels = sorted(list(set(labels_real) | set(labels_pred)))
    report_score = classification_report(labels_real, labels_pred, target_names=labels, digits=4, output_dict=True, zero_division=0)
    df_score = pd.DataFrame(report_score).transpose()

    dctn = {}

    dctn[col_accuracy] = df_score['accuracy']['f1-score']
    dctn[col_f1] = df_score['macro avg']['f1-score']
    dctn[col_precision] = df_score['macro avg']['precision']
    dctn[col_recall] = df_score['macro avg']['recall']
