import math
import time
import traceback
from queue import Queue

import numpy as np
from matplotlib import pyplot as plt, cm
from matplotlib.colors import LightSource
from scipy.linalg import polar
import os
import pathlib
import random

import numpy as np
from sklearn.cluster import BisectingKMeans
from tqdm import tqdm
from bisect import bisect_left

from e_main import const
from e_main.data_manager import DataManager
from e_main.preset import Preset

# draw train vector space in to figures and see distribution
import matplotlib.pyplot as plt
import matplotlib as mpl
from pprint import pprint as p
import scipy
from e_main.tools import Watch

mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['axes.unicode_minus'] = False
plt.style.use('Solarize_Light2')


def get_method_COS_REAL_MBTree_idxs_N_ck2idxs(data_BxF, idxs):
    N = len(idxs)
    pick = 1
    if len(idxs) == 1:
        mean_idx = idxs[0]
        return [mean_idx], N, {}
    else:
        B, F = data_BxF.shape
        po = 2 ** np.arange(F)

        mean_data_F = (np.max(data_BxF[idxs, :], axis=0) + np.min(data_BxF[idxs, :], axis=0)) / 2
        mean_local_idx = np.argmax(np.sum((data_BxF[idxs, :] * mean_data_F), axis=1))

        mean_idx = idxs[mean_local_idx]
        wash_idxs = idxs[:mean_local_idx] + idxs[mean_local_idx + 1:]

        ck2idxs = {}
        if len(wash_idxs) > 0:
            chain_sub_keys = np.array(data_BxF[wash_idxs, :] < data_BxF[mean_idx, :], dtype=bool) @ po
            for ck, idx in zip(chain_sub_keys, wash_idxs):
                if not ck in ck2idxs:
                    ck2idxs[ck] = []
                ck2idxs[ck].append(idx)

        return [mean_idx], N, ck2idxs


def get_method_EUC_REAL_MBTree_idxs_N_ck2idxs(data_BxF, idxs):
    N = len(idxs)

    if len(idxs) == 1:
        mean_idx = idxs[0]
        return [mean_idx], N, {}
    else:
        B, F = data_BxF.shape
        po = 2 ** np.arange(F)

        mean_data_F = (np.max(data_BxF[idxs, :], axis=0) + np.min(data_BxF[idxs, :], axis=0)) / 2
        mean_local_idx = np.argmin(np.sum((data_BxF[idxs, :] - mean_data_F) ** 2, axis=1))

        mean_idx = idxs[mean_local_idx]
        wash_idxs = idxs[:mean_local_idx] + idxs[mean_local_idx + 1:]

        ck2idxs = {}
        if len(wash_idxs) > 0:
            chain_sub_keys = np.array(data_BxF[wash_idxs, :] < data_BxF[mean_idx, :], dtype=bool) @ po
            for ck, idx in zip(chain_sub_keys, wash_idxs):
                if not ck in ck2idxs:
                    ck2idxs[ck] = []
                ck2idxs[ck].append(idx)

        return [mean_idx], N, ck2idxs


def get_method_COS_VIRT_MBTree_idxs_N_ck2idxs(data_BxF, idxs):
    N = len(idxs)

    if len(idxs) == 1:
        mean_idx = idxs[0]
        return [mean_idx], N, {}
    else:
        B, F = data_BxF.shape
        po = 2 ** np.arange(F)

        mean_data_F = (np.max(data_BxF[idxs, :], axis=0) + np.min(data_BxF[idxs, :], axis=0)) / 2
        mean_local_idx = np.argmax(np.sum((data_BxF[idxs, :] * mean_data_F), axis=1))

        mean_idx = idxs[mean_local_idx]
        wash_idxs = idxs[:mean_local_idx] + idxs[mean_local_idx + 1:]

        ck2idxs = {}
        if len(wash_idxs) > 0:
            chain_sub_keys = np.array(data_BxF[wash_idxs, :] < mean_data_F, dtype=bool) @ po
            for ck, idx in zip(chain_sub_keys, wash_idxs):
                if not ck in ck2idxs:
                    ck2idxs[ck] = []
                ck2idxs[ck].append(idx)

        return [mean_idx], N, ck2idxs


def get_method_EUC_VIRT_MBTree_idxs_N_ck2idxs(data_BxF, idxs):
    N = len(idxs)

    if len(idxs) == 1:
        mean_idx = idxs[0]
        return [mean_idx], N, {}
    else:
        B, K = data_BxF.shape
        po = 2 ** np.arange(K)

        mean_data_F = (np.max(data_BxF[idxs, :], axis=0) + np.min(data_BxF[idxs, :], axis=0)) / 2
        mean_local_idx = np.argmin(np.sum((data_BxF[idxs, :] - mean_data_F) ** 2, axis=1))

        mean_idx = idxs[mean_local_idx]
        wash_idxs = idxs[:mean_local_idx] + idxs[mean_local_idx + 1:]

        ck2idxs = {}
        if len(wash_idxs) > 0:
            chain_sub_keys = np.array(data_BxF[wash_idxs, :] < mean_data_F, dtype=bool) @ po
            for ck, idx in zip(chain_sub_keys, wash_idxs):
                if not ck in ck2idxs:
                    ck2idxs[ck] = []
                ck2idxs[ck].append(idx)

        return [mean_idx], N, ck2idxs


def get_method_COS_VIRT_MEAN_MBTree_idxs_N_ck2idxs(data_BxF, idxs):
    N = len(idxs)

    if len(idxs) == 1:
        mean_idx = idxs[0]
        return [mean_idx], N, {}
    else:
        B, F = data_BxF.shape
        po = 2 ** np.arange(F)

        mean_data_F = np.mean(data_BxF[idxs, :], axis=0)
        mean_local_idx = np.argmax(np.sum((data_BxF[idxs, :] * mean_data_F), axis=1))

        mean_idx = idxs[mean_local_idx]
        wash_idxs = idxs[:mean_local_idx] + idxs[mean_local_idx + 1:]

        ck2idxs = {}

        center_data_F = (np.max(data_BxF[idxs, :], axis=0) + np.min(data_BxF[idxs, :], axis=0)) / 2
        if len(wash_idxs) > 0:
            chain_sub_keys = np.array(data_BxF[wash_idxs, :] < center_data_F, dtype=bool) @ po
            for ck, idx in zip(chain_sub_keys, wash_idxs):
                if not ck in ck2idxs:
                    ck2idxs[ck] = []
                ck2idxs[ck].append(idx)

        return [mean_idx], N, ck2idxs


def get_method_EUC_VIRT_MEAN_MBTree_idxs_N_ck2idxs(data_BxF, idxs):
    N = len(idxs)

    if len(idxs) == 1:
        mean_idx = idxs[0]
        return [mean_idx], N, {}
    else:
        B, K = data_BxF.shape
        po = 2 ** np.arange(K)

        # use mean represent the whole part
        mean_data_F = np.mean(data_BxF[idxs, :], axis=0)
        mean_local_idx = np.argmin(np.sum((data_BxF[idxs, :] - mean_data_F) ** 2, axis=1))

        mean_idx = idxs[mean_local_idx]
        wash_idxs = idxs[:mean_local_idx] + idxs[mean_local_idx + 1:]

        # cut by center
        center_data_F = (np.max(data_BxF[idxs, :], axis=0) + np.min(data_BxF[idxs, :], axis=0)) / 2

        ck2idxs = {}
        if len(wash_idxs) > 0:
            chain_sub_keys = np.array(data_BxF[wash_idxs, :] < center_data_F, dtype=bool) @ po
            for ck, idx in zip(chain_sub_keys, wash_idxs):
                if not ck in ck2idxs:
                    ck2idxs[ck] = []
                ck2idxs[ck].append(idx)

        return [mean_idx], N, ck2idxs


def get_method_COS_VIRT_MEDIAN_MBTree_idxs_N_ck2idxs(data_BxF, idxs):
    N = len(idxs)

    if len(idxs) == 1:
        mean_idx = idxs[0]
        return [mean_idx], N, {}
    else:
        B, F = data_BxF.shape
        po = 2 ** np.arange(F)

        mean_data_F = np.median(data_BxF[idxs, :], axis=0)
        mean_local_idx = np.argmax(np.sum((data_BxF[idxs, :] * mean_data_F), axis=1))

        mean_idx = idxs[mean_local_idx]
        wash_idxs = idxs[:mean_local_idx] + idxs[mean_local_idx + 1:]

        ck2idxs = {}

        center_data_F = (np.max(data_BxF[idxs, :], axis=0) + np.min(data_BxF[idxs, :], axis=0)) / 2
        if len(wash_idxs) > 0:
            chain_sub_keys = np.array(data_BxF[wash_idxs, :] < center_data_F, dtype=bool) @ po
            for ck, idx in zip(chain_sub_keys, wash_idxs):
                if not ck in ck2idxs:
                    ck2idxs[ck] = []
                ck2idxs[ck].append(idx)

        return [mean_idx], N, ck2idxs


def get_method_EUC_VIRT_MEDIAN_MBTree_idxs_N_ck2idxs(data_BxF, idxs):
    N = len(idxs)

    if len(idxs) == 1:
        mean_idx = idxs[0]
        return [mean_idx], N, {}
    else:
        B, K = data_BxF.shape
        po = 2 ** np.arange(K)

        # use mean represent the whole part
        mean_data_F = np.median(data_BxF[idxs, :], axis=0)
        mean_local_idx = np.argmin(np.sum((data_BxF[idxs, :] - mean_data_F) ** 2, axis=1))

        mean_idx = idxs[mean_local_idx]
        wash_idxs = idxs[:mean_local_idx] + idxs[mean_local_idx + 1:]

        # cut by center
        center_data_F = (np.max(data_BxF[idxs, :], axis=0) + np.min(data_BxF[idxs, :], axis=0)) / 2

        ck2idxs = {}
        if len(wash_idxs) > 0:
            chain_sub_keys = np.array(data_BxF[wash_idxs, :] < center_data_F, dtype=bool) @ po
            for ck, idx in zip(chain_sub_keys, wash_idxs):
                if not ck in ck2idxs:
                    ck2idxs[ck] = []
                ck2idxs[ck].append(idx)

        return [mean_idx], N, ck2idxs


def get_MBTree(data_BxF, get_method_MBTree_idxs_N_ck2idxs, idxs=None, d=0):
    if idxs is None:
        idxs = list(range(len(data_BxF)))
    pick_idxs, N, ck2idxs = get_method_MBTree_idxs_N_ck2idxs(data_BxF, idxs)

    subs = []
    for ck, sub_idxs in ck2idxs.items():
        # print(len(sub_idxs))
        sub = get_MBTree(data_BxF, get_method_MBTree_idxs_N_ck2idxs, idxs=sub_idxs, d=d + 1)
        subs.append(sub)

    return {'idxs': pick_idxs, 'N': N, 'subs': subs, 'd': d}


def reorder_by_MBTree(data_BxF, get_method_MBTree_idxs_N_ck2idxs, idxs=None):
    if idxs is None:
        idxs = list(range(len(data_BxF)))

    B, F = data_BxF.shape

    mb_tree = get_MBTree(data_BxF, get_method_MBTree_idxs_N_ck2idxs, idxs=idxs)

    level2idxs = {}
    q = Queue()
    q.put(mb_tree)
    while not q.empty():
        t = q.get()
        level = t['d']
        idxs = t['idxs']
        subs = t['subs']
        if not level in level2idxs:
            level2idxs[level] = []
        level2idxs[level].extend(idxs)
        for sub in subs:
            q.put(sub)

    # p(level2idxs)

    new_idxs = []
    for level in sorted(list(level2idxs.keys())):
        idxs = level2idxs[level]
        random.shuffle(idxs)
        new_idxs.extend(idxs)

    if not len(new_idxs) == len(idxs):
        hold = set()
        new_idxs_buffer = []
        for nidx in new_idxs:
            if not nidx in hold:
                hold.add(nidx)
                new_idxs_buffer.append(nidx)
        new_idxs = new_idxs_buffer
    return new_idxs


def reorder_method_CR(data_BxF):
    res = reorder_by_MBTree(data_BxF, get_method_COS_REAL_MBTree_idxs_N_ck2idxs, idxs=None)
    return res


def reorder_method_ER(data_BxF):
    res = reorder_by_MBTree(data_BxF, get_method_EUC_REAL_MBTree_idxs_N_ck2idxs, idxs=None)
    return res


def reorder_method_CV(data_BxF):
    res = reorder_by_MBTree(data_BxF, get_method_COS_VIRT_MBTree_idxs_N_ck2idxs, idxs=None)
    return res


def reorder_method_EV(data_BxF):
    res = reorder_by_MBTree(data_BxF, get_method_EUC_VIRT_MBTree_idxs_N_ck2idxs, idxs=None)
    return res


def reorder_method_CVMean(data_BxF):
    res = reorder_by_MBTree(data_BxF, get_method_COS_VIRT_MEAN_MBTree_idxs_N_ck2idxs, idxs=None)
    return res


def reorder_method_EVMean(data_BxF):
    res = reorder_by_MBTree(data_BxF, get_method_EUC_VIRT_MEAN_MBTree_idxs_N_ck2idxs, idxs=None)
    return res


def reorder_method_CVMedian(data_BxF):
    res = reorder_by_MBTree(data_BxF, get_method_COS_VIRT_MEDIAN_MBTree_idxs_N_ck2idxs, idxs=None)
    return res


def reorder_method_EVMedian(data_BxF):
    res = reorder_by_MBTree(data_BxF, get_method_EUC_VIRT_MEDIAN_MBTree_idxs_N_ck2idxs, idxs=None)
    return res


def reorder_method_CR_CV(data_BxF):
    idxs_CR = reorder_method_CR(data_BxF)
    idxs_CV = reorder_method_CV(data_BxF)

    ridxs = np.arange(len(data_BxF)).tolist()

    ridx2rank = {ridx: 0 for ridx in ridxs}

    for idxs_M in [idxs_CV, idxs_CR]:
        for rank, ridx in enumerate(idxs_M):
            ridx2rank[ridx] += rank

    ridxs.sort(key=lambda ridx: ridx2rank[ridx])
    return ridxs


def reorder_method_CR_CV_RAND(data_BxF):
    idxs_CR = reorder_method_CR(data_BxF)
    idxs_CV = reorder_method_CV(data_BxF)
    idxs_RAND = reorder_method_RAND(data_BxF)
    ridxs = np.arange(len(data_BxF)).tolist()

    ridx2rank = {ridx: 0 for ridx in ridxs}

    for idxs_M in [idxs_CV, idxs_CR, idxs_RAND]:
        for rank, ridx in enumerate(idxs_M):
            ridx2rank[ridx] += rank

    ridxs.sort(key=lambda ridx: ridx2rank[ridx])
    return ridxs


def reorder_method_CR_RAND(data_BxF):
    idxs_CR = reorder_method_CR(data_BxF)
    idxs_RAND = reorder_method_RAND(data_BxF)
    ridxs = np.arange(len(data_BxF)).tolist()

    ridx2rank = {ridx: 0 for ridx in ridxs}

    for idxs_M in [idxs_CR, idxs_RAND]:
        for rank, ridx in enumerate(idxs_M):
            ridx2rank[ridx] += rank

    ridxs.sort(key=lambda ridx: ridx2rank[ridx])
    return ridxs


def reorder_method_CV_RAND(data_BxF):
    idxs_CV = reorder_method_CV(data_BxF)
    idxs_RAND = reorder_method_RAND(data_BxF)
    ridxs = np.arange(len(data_BxF)).tolist()

    ridx2rank = {ridx: 0 for ridx in ridxs}

    for idxs_M in [idxs_CV, idxs_RAND]:
        for rank, ridx in enumerate(idxs_M):
            ridx2rank[ridx] += rank

    ridxs.sort(key=lambda ridx: ridx2rank[ridx])
    return ridxs


def reorder_method_CV_EV(data_BxF):
    idxs_CV = reorder_method_CV(data_BxF)
    idxs_EV = reorder_method_EV(data_BxF)
    ridxs = np.arange(len(data_BxF)).tolist()

    ridx2rank = {ridx: 0 for ridx in ridxs}

    for idxs_M in [idxs_CV, idxs_EV]:
        for rank, ridx in enumerate(idxs_M):
            ridx2rank[ridx] += rank

    ridxs.sort(key=lambda ridx: ridx2rank[ridx])
    return ridxs


def reorder_method_CV_EV_RAND(data_BxF):
    idxs_CV = reorder_method_CV(data_BxF)
    idxs_EV = reorder_method_EV(data_BxF)
    idxs_RAND = reorder_method_RAND(data_BxF)
    ridxs = np.arange(len(data_BxF)).tolist()

    ridx2rank = {ridx: 0 for ridx in ridxs}

    for idxs_M in [idxs_CV, idxs_EV, idxs_RAND]:
        for rank, ridx in enumerate(idxs_M):
            ridx2rank[ridx] += rank

    ridxs.sort(key=lambda ridx: ridx2rank[ridx])
    return ridxs


def reorder_method_RAND(data_BxF):
    ridxs = np.arange(len(data_BxF))
    np.random.shuffle(ridxs)
    return ridxs


methods = [
    reorder_method_RAND
    , reorder_method_CVMean
    , reorder_method_EVMean
    , reorder_method_CVMedian
    , reorder_method_EVMedian
    , reorder_method_CV_EV_RAND
]

np.random.seed(const.seed)
random.seed(const.seed)

if __name__ == '__main__':
    pass

    # data_B = np.random.rand(5000) * 2 * np.pi

    data_B = np.clip(np.random.normal(0, np.pi / 8, 5000), -np.pi, np.pi)

    data_B1 = np.cos(data_B)
    data_B2 = np.sin(data_B)
    data_BxF = np.stack([data_B1, data_B2], axis=1)

    # bisect_means = BisectingKMeans().fit(data_BxF)

    # ax = plt.axes(projection="3d")
    # ax.scatter3D(data_B1, data_B2, bisect_means.labels_)
    #
    # plt.show(block=True)

    # for d in [2, 4, 6, 8]:
    #     data_B = np.clip(np.random.normal(0, np.pi / d, 100), -np.pi, np.pi)
    #
    #     data_B1 = np.cos(data_B)
    #     data_B2 = np.sin(data_B)
    #
    #     data_BxF = np.stack([data_B1, data_B2], axis=1)
    #
    #     # get_method_E_MBTree_idxs_N_ck2idxs(data_BxF)
    #
    #     data_BxK = scipy.linalg.orth(data_BxF)
    #
    #     # plt.scatter(data_BxK[:, 0], data_BxK[:, 1])
    #     # plt.scatter(data_BxF[:, 0], data_BxF[:, 1])
    #
    #     # for K, F in zip(data_BxK, data_BxF):
    #     #     plt.arrow(F[1], F[1], K[0] - F[0], K[1] - F[1], color='white')
    #
    #     v_B = np.sum(data_BxK, axis=1)
    #     idxs = np.argsort(v_B)
    #
    #     # plt.scatter(v_B, data_B)
    #
    #     # Creating plot
    #     ax.scatter3D(data_B1, data_B2, v_B)
    # # plt.scatter(data_B, np.arange(len(data_B)))
    # #
    # plt.show(block=True)

    # plot sample in 2D
    generate_sample_on_simulate = True
    if generate_sample_on_simulate:
        fig, axs = plt.subplots(len(methods), 1, figsize=(13, 50), sharex='all')
        for i, method in enumerate(tqdm(methods)):
            w = Watch()
            nidxs = method(data_BxF)

            cost = w.see()

            axs[i].scatter(data_B[nidxs], np.linspace(0, len(nidxs), len(nidxs)), s=0.5, label=f'{method.__name__}')
            axs[i].legend()

        axs[i].set_xlabel(f'theta value, cost = {cost.total_seconds()} s')
        axs[i].set_ylabel('rank')
        path_figure = os.path.join(Preset.root, r'd_figures')
        figfpath = os.path.join(path_figure, f'method_on_Circle_Gussian.png')
        plt.savefig(figfpath, bbox_inches='tight')
        plt.close(fig)

        # fig = plt.figure(figsize=plt.figaspect(0.5))
        # for mi, method in enumerate(methods):
        #     ax = fig.add_subplot(4, 4, mi + 1, projection="3d")
        # 
        #     # Creating plot
        #     nidxs = methods[mi](data_BxF)
        #     ax.scatter3D(data_B1[nidxs], data_B2[nidxs], np.linspace(0, len(nidxs), len(nidxs)), s=0.5, label=f'{method.__name__}')
        #     ax.legend()
        # plt.show(block=True)

    # calculate time cost
    generate_idxs_buffer = True
    if generate_idxs_buffer:
        ptrain_ptest = [
                           (r'emotion.train.16000x6.jsonl', r'emotion.test.2000x6.jsonl'),
                           (r'yelp.train.650000x5.jsonl', r'yelp.test.50000x5.jsonl'),
                           (r'agnews.train.120000x4.jsonl', r'agnews.test.7600x4.jsonl'),
                           (r'imdb.train.25000x2.jsonl', r'imdb.test.25000x2.jsonl'),
                           (r'dbpedia.train.560000x14.jsonl', r'dbpedia.test.70000x14.jsonl'),
                           (r'amazonpolar.train.3600000x2.jsonl', r'amazonpolar.test.400000x2.jsonl')
                       ][:]

        X_path_s = []
        for ptrain, ptest in tqdm(ptrain_ptest):
            dataset_name = ptrain.split('.')[0]

            dm = DataManager(path_data_jsonl_train=ptrain, path_data_jsonl_test=ptest)
            X_train = dm.get_X_train()
            X_test = dm.get_X_test()

            X_path_s.append((X_train, ptrain))
            X_path_s.append((X_test, ptest))

        X_path_s.sort(key=lambda X_path: len(X_path[0]))

        Ns = [len(X_path[0]) for X_path in X_path_s][:]
        method2costs = {}
        for method in tqdm(methods[:]):
            method2costs[method.__name__] = []
            for X, path in tqdm(X_path_s[:]):
                N = len(X)
                w = Watch()
                fpath = os.path.join(Preset.root, 'c_idxs', f"{method.__name__}.{path.replace('.jsonl', '.idxs.npy')}")
                if 'train' in path and (not os.path.exists(fpath)):

                    idxs = method(X)
                    time.sleep(5)
                    try:
                        idxs = np.array(idxs)
                        np.save(fpath, idxs)
                    except Exception as e:
                        print(traceback.format_exc())
                        pass
                cost = w.see()
                print(f"{N}, {cost}")
                method2costs[method.__name__].append(cost)

        fig3 = plt.figure(figsize=(24, 13))

        for method_name, costs in method2costs.items():
            plt.plot(Ns[:len(costs)], [c.total_seconds() for c in costs])
            plt.scatter(Ns[:len(costs)], [c.total_seconds() for c in costs], color=plt.gca().lines[-1].get_color(), label=f'{method_name}')

        plt.xlabel('#items')
        plt.ylabel('time cost of sample method (sec)')

        plt.legend()

        path_figure = os.path.join(Preset.root, r'd_figures')
        figfpath = os.path.join(path_figure, f'methods_time_cost_compare.png')
        plt.savefig(figfpath, bbox_inches='tight')
        plt.close()

    # plt.show(block=True)
