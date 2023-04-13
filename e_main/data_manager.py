import os
import pathlib
import random

import numpy as np
from tqdm import tqdm

from e_main.preset import Preset

# draw train vector space in to figures and see distribution
import matplotlib.pyplot as plt
import matplotlib as mpl
from pprint import pprint as p
import scipy

from e_main.tools import read_jsonx

mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['axes.unicode_minus'] = False
plt.style.use('Solarize_Light2')

path_data = os.path.join(Preset.root, r'b_data')
path_buffer = os.path.join(Preset.root, r'c_buffer')

path_idxs = os.path.join(Preset.root, 'c_idxs')


class DataManager:

    def __init__(self, path_data_jsonl_train, path_data_jsonl_test):
        self.path_buffer_npy_train = path_data_jsonl_train.replace('.jsonl', '.X.npy')
        self.path_buffer_npy_test = path_data_jsonl_test.replace('.jsonl', '.X.npy')

        self.fpath_data_jsonl_train = os.path.join(path_data, path_data_jsonl_train)
        self.fpath_data_jsonl_test = os.path.join(path_data, path_data_jsonl_test)
        self.fpath_buffer_npy_train = os.path.join(path_buffer, self.path_buffer_npy_train)
        self.fpath_buffer_npy_test = os.path.join(path_buffer, self.path_buffer_npy_test)

        for fname in [self.fpath_data_jsonl_train, self.fpath_data_jsonl_test, self.fpath_buffer_npy_train, self.fpath_buffer_npy_test]:
            if os.path.exists(fname):
                print(f'EXIST {fname}')
            else:
                print(f'NOT FOUND {fname}')

        self.dctns_train = self.read_dctns_train()
        self.dctns_test = self.read_dctns_test()
        self.cid2label, self.label2cid = self.get_label_map()

        self.N_data_train = len(self.dctns_train)
        self.N_data_test = len(self.dctns_test)
        self.N_labels = len(self.cid2label)

        self.cid2rids_train = self.get_cid2rids_train()
        self.cid2rids_test = self.get_cid2rids_test()

        self.rids_data_train = [rid for rid in range(self.N_data_train)]

        print(f'{[(cid, rids.shape[0]) for cid, rids in enumerate(self.cid2rids_train)]}')
        print(f'{[(cid, rids.shape[0]) for cid, rids in enumerate(self.cid2rids_test)]}')

    def read_dctns_train(self):
        dctns = read_jsonx(self.fpath_data_jsonl_train)
        # print(f'{self.fpath_data_jsonl_train} : #{len(dctns)}')
        return dctns

    def read_dctns_test(self):
        dctns = read_jsonx(self.fpath_data_jsonl_test)
        # print(f'{self.fpath_data_jsonl_train} : #{len(dctns)}')
        return dctns

    def get_X_train(self):
        X = np.load(self.fpath_buffer_npy_train)
        # print(f'{self.fpath_buffer_npy_train} [X]: #{X.shape}')

        return X

    def get_X_test(self):
        X = np.load(self.fpath_buffer_npy_test)
        # print(f'{self.fpath_buffer_npy_test} [X]: #{X.shape}')
        return X

    def get_label_map(self):
        labels = {dctn['label'] for dctn in self.dctns_train} | {dctn['label'] for dctn in self.dctns_test}
        idx2label = sorted(list(labels))
        label2idx = {label: idx for idx, label in enumerate(idx2label)}

        return idx2label, label2idx

    def get_y_train(self):
        y = np.array([self.label2cid[dctn['label']] for dctn in self.dctns_train])
        return y

    def get_y_test(self):
        y = np.array([self.label2cid[dctn['label']] for dctn in self.dctns_test])
        return y

    def get_cid2rids_train(self):
        cid2rids = [[] for _ in range(self.N_labels)]
        for rid, cid in enumerate(self.get_y_train()):
            cid2rids[cid].append(rid)

        for cid in range(self.N_labels):
            cid2rids[cid] = np.array(cid2rids[cid])
        return cid2rids

    def get_cid2rids_test(self):
        cid2rids = [[] for _ in range(self.N_labels)]
        for rid, cid in enumerate(self.get_y_test()):
            cid2rids[cid].append(rid)

        for cid in range(self.N_labels):
            cid2rids[cid] = np.array(cid2rids[cid])
        return cid2rids

    def get_rids_random_sample(self, seed=24):
        random.seed(seed)
        rids_random_sample = random.sample(self.rids_data_train, len(self.rids_data_train))
        return np.array(rids_random_sample)

    def get_rids_sample(self, reorder_method_name):
        """

        :param reorder_method_name:
        reorder_method_AF
        reorder_method_BF
        :return:
        """
        fname = f"{reorder_method_name}.{self.path_buffer_npy_train.replace('.X.npy', '.idxs.npy')}"
        fpath = os.path.join(path_idxs, fname)
        idxs = np.load(fpath)
        return idxs

    def get_rids_with_label(self):

        cids_B = [self.label2cid[dctn['label']] for dctn in self.dctns_train]

        cid2rids = [[] for _ in self.label2cid]
        for rid, cid in enumerate(cids_B):
            cid2rids[cid].append(rid)

        for cid in range(self.N_labels):
            random.shuffle(cid2rids[cid])

        new_ridxs = []
        while len(new_ridxs) < len(self.dctns_train):
            for cid in range(self.N_labels):
                if len(cid2rids[cid]) > 0:
                    new_ridxs.append(cid2rids[cid].pop())

        return np.array(new_ridxs)


if __name__ == '__main__':
    pass

    dm = DataManager(path_data_jsonl_train=r'emotion.train.16000x6.jsonl', path_data_jsonl_test=r'emotion.test.2000x6.jsonl')

    data_BxF = dm.get_X_train()
    data_BxK = scipy.linalg.orth(data_BxF)
    d = sorted(np.sum(data_BxK ** 2, axis=1))
    # plt.scatter(d, np.zeros_like(d))
    # plt.show(block=True)
