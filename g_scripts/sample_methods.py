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
from tqdm import tqdm

from e_main.data_manager import DataManager
from e_main.preset import Preset

# draw train vector space in to figures and see distribution
import matplotlib.pyplot as plt
import matplotlib as mpl
from pprint import pprint as p

from e_main.tools import Watch

mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['axes.unicode_minus'] = False
plt.style.use('Solarize_Light2')


def get_method_A_MBTree_idxs_N_ck2idxs(data_BxF, idxs):
    B, F = data_BxF.shape
    N = len(idxs)
    po = 2 ** np.arange(F)
    if len(idxs) == 1:
        mean_idx = idxs[0]
        return [mean_idx], N, {}
    else:
        mean_data_F = np.mean(data_BxF[idxs, :], axis=0)
        # mean_local_idx = np.argmin(np.sum((data_BxF[idxs, :] - mean_data_F) ** 2, axis=1))
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


def get_method_B_MBTree_idxs_N_ck2idxs(data_BxF, idxs):
    B, F = data_BxF.shape
    N = len(idxs)
    if len(idxs) <= 2:
        return idxs, N, {}
    else:
        mean_data_F = np.mean(data_BxF[idxs, :], axis=0)
        # mean_local_idx = np.argmin(np.sum((data_BxF[idxs, :] - mean_data_F) ** 2, axis=1))

        similarity_B = np.sum((data_BxF[idxs, :] * mean_data_F), axis=1)
        mean_local_idx_max = np.argmax(similarity_B)
        mean_local_idx_min = np.argmin(similarity_B)

        # mean_idxs = [idxs[mean_local_idx_max], idxs[mean_local_idx_min]]

        mean_idxs = [mean_local_idx_max, mean_local_idx_min]
        # print(mean_idxs)
        similar_to_mean_max_B = np.sum(data_BxF[idxs, :] * data_BxF[mean_local_idx_max, :], axis=1)
        similar_to_mean_min_B = np.sum(data_BxF[idxs, :] * data_BxF[mean_local_idx_min, :], axis=1)

        cross_B = (similar_to_mean_max_B < similar_to_mean_min_B).astype(int)
        cross_B[mean_idxs] = -1

        # print(cross_B)

        ck2idxs = {0: [], 1: []}
        for i, ck in enumerate(cross_B):
            if ck >= 0:
                if not ck in ck2idxs:
                    ck2idxs[ck] = []
                ck2idxs[ck].append(idxs[i])

        return [idxs[i] for i in mean_idxs], N, ck2idxs


def get_method_C_MBTree_idxs_N_ck2idxs(data_BxF, idxs):
    B, F = data_BxF.shape
    N = len(idxs)
    if len(idxs) <= 2:
        return idxs, N, {}
    else:
        mean_data_F = np.mean(data_BxF[idxs, :], axis=0)
        # mean_local_idx = np.argmin(np.sum((data_BxF[idxs, :] - mean_data_F) ** 2, axis=1))

        similarity_B = np.sum((data_BxF[idxs, :] * mean_data_F), axis=1)
        mean_local_idx_max = np.argmax(similarity_B)
        mean_local_idx_min = np.argmin(similarity_B)

        # mean_idxs = [idxs[mean_local_idx_max], idxs[mean_local_idx_min]]

        mean_idxs = [mean_local_idx_max, mean_local_idx_min]
        # print(mean_idxs)
        similar_to_mean_max_B = np.sum(data_BxF[idxs, :] * data_BxF[mean_local_idx_max, :], axis=1)
        # similar_to_mean_min_B = np.sum(data_BxF[idxs, :] * data_BxF[mean_local_idx_min, :], axis=1)

        cross_B = (similar_to_mean_max_B > 0).astype(int)
        cross_B[mean_idxs] = -1

        # print(cross_B)

        ck2idxs = {0: [], 1: []}
        for i, ck in enumerate(cross_B):
            if ck >= 0:
                if not ck in ck2idxs:
                    ck2idxs[ck] = []
                ck2idxs[ck].append(idxs[i])

        return [idxs[i] for i in mean_idxs], N, ck2idxs


def get_method_D_MBTree_idxs_N_ck2idxs(data_BxF, idxs):
    B, F = data_BxF.shape
    N = len(idxs)
    if len(idxs) <= 2:
        return idxs, N, {}
    else:
        mean_data_F = np.mean(data_BxF[idxs, :], axis=0)
        # mean_local_idx = np.argmin(np.sum((data_BxF[idxs, :] - mean_data_F) ** 2, axis=1))

        similarity_B = np.sum((data_BxF[idxs, :] * mean_data_F), axis=1)
        mean_local_idx_max = np.argmax(similarity_B)
        mean_local_idx_min = np.argmin(similarity_B)

        # mean_idxs = [idxs[mean_local_idx_max], idxs[mean_local_idx_min]]

        mean_idxs = [mean_local_idx_max, mean_local_idx_min]
        # print(mean_idxs)
        similarity_B_N = np.sum((data_BxF[idxs, :] * -mean_data_F), axis=1)

        cross_B = (similarity_B_N > similarity_B).astype(int)
        cross_B[mean_idxs] = -1

        # print(cross_B)

        ck2idxs = {0: [], 1: []}
        for i, ck in enumerate(cross_B):
            if ck >= 0:
                if not ck in ck2idxs:
                    ck2idxs[ck] = []
                ck2idxs[ck].append(idxs[i])

        return [idxs[i] for i in mean_idxs], N, ck2idxs


def get_method_E_MBTree_idxs_N_ck2idxs(data_BxF):
    B, F = data_BxF.shape

    max_pos_B = np.argmax(np.abs(data_BxF), axis=1)
    max_pos_sign_B = np.sign(data_BxF[np.arange(B), max_pos_B])

    cks = ((max_pos_sign_B + 1) // 2 + 2 * max_pos_B).astype(int)

    ck2idxs = {}
    pick_idxs = []
    for idx, ck in enumerate(cks):
        if not ck in ck2idxs:
            ck2idxs[ck] = []
        ck2idxs[ck].append(idx)

    ck2vct_F = {}
    for ck in ck2idxs:
        vct_F = np.zeros(F)
        sign = int((ck % 2 * 2) - 1)
        pos = ck // 2
        vct_F[pos] = sign
        ck2vct_F[ck] = vct_F

    p(ck2vct_F)

    for ck, idxs in ck2idxs.items():
        similarity_B = np.sum((data_BxF[idxs, :] * ck2vct_F[ck]), axis=1)
        local_pick_idx = np.argmax(similarity_B)
        pick_idx = idxs[local_pick_idx]
        pick_idxs.append(pick_idx)
        new_idxs = idxs[:local_pick_idx] + idxs[local_pick_idx + 1:]
        ck2idxs[ck] = new_idxs
        print(f"{ck}:{pick_idx},{new_idxs}")


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


def reorder_by_MBTree(data_BxF, get_method_MBTree_idxs_N_ck2idxs, idxs=None, sub_reorder=True):
    if idxs is None:
        idxs = list(range(len(data_BxF)))

    B, F = data_BxF.shape
    limit = F

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
    for level, idxs in level2idxs.items():
        if len(idxs) <= limit:
            random.shuffle(idxs)
            new_idxs.extend(idxs)
        else:
            if sub_reorder:
                idxs = reorder_by_MBTree(data_BxF, get_method_MBTree_idxs_N_ck2idxs, idxs, sub_reorder=sub_reorder)
                # random.shuffle(idxs)
                new_idxs.extend(idxs)
            else:
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


def reorder_method_AT(data_BxF):
    return reorder_by_MBTree(data_BxF, get_method_A_MBTree_idxs_N_ck2idxs, idxs=None, sub_reorder=True)


def reorder_method_AF(data_BxF):
    return reorder_by_MBTree(data_BxF, get_method_A_MBTree_idxs_N_ck2idxs, idxs=None, sub_reorder=False)


def reorder_method_BT(data_BxF):
    return reorder_by_MBTree(data_BxF, get_method_B_MBTree_idxs_N_ck2idxs, idxs=None, sub_reorder=False)


def reorder_method_BF(data_BxF):
    return reorder_by_MBTree(data_BxF, get_method_B_MBTree_idxs_N_ck2idxs, idxs=None, sub_reorder=False)


def reorder_method_CT(data_BxF):
    return reorder_by_MBTree(data_BxF, get_method_C_MBTree_idxs_N_ck2idxs, idxs=None, sub_reorder=False)


def reorder_method_CF(data_BxF):
    return reorder_by_MBTree(data_BxF, get_method_C_MBTree_idxs_N_ck2idxs, idxs=None, sub_reorder=False)


# def reorder_method_DT(data_BxF):
#     return reorder_by_MBTree(data_BxF, get_method_D_MBTree_idxs_N_ck2idxs, idxs=None, sub_reorder=False)
#
#
# def reorder_method_DF(data_BxF):
#     return reorder_by_MBTree(data_BxF, get_method_D_MBTree_idxs_N_ck2idxs, idxs=None, sub_reorder=False)


def reorder_method_rand(data_BxF):
    ridxs = list(range(len(data_BxF)))
    random.shuffle(ridxs)

    return ridxs


methods = [reorder_method_rand,
           # reorder_method_AT,
           reorder_method_AF,
           # reorder_method_BT,
           reorder_method_BF
           # reorder_method_CT,
           # reorder_method_CF
           # reorder_method_DT,
           # reorder_method_DF
           ]

if __name__ == '__main__':
    pass

    np.random.seed(24)
    #
    # data_B = np.random.rand(5000) * 2 * np.pi
    data_B = np.clip(np.random.normal(0, np.pi / 2, 100), -np.pi, np.pi)

    data_B1 = np.cos(data_B)
    data_B2 = np.sin(data_B)

    data_BxF = np.stack([data_B1, data_B2], axis=1)

    get_method_E_MBTree_idxs_N_ck2idxs(data_BxF)

    # fig, axs = plt.subplots(len(methods), 1, figsize=(13, 50), sharex='all')
    #
    # for i, method in enumerate(methods):
    #     w = Watch()
    #     nidxs = method(data_BxF)
    #
    #     cost = w.see()
    #
    #     axs[i].scatter(data_B[nidxs], np.linspace(0, len(nidxs), len(nidxs)), s=0.5, label=f'{method.__name__}')
    #     axs[i].legend()
    #
    #     axs[i].set_xlabel(f'theta value, cost = {cost.total_seconds()} s')
    #     axs[i].set_ylabel('rank')
    #
    # path_figure = os.path.join(Preset.root, r'd_figures')
    # figfpath = os.path.join(path_figure, f'method_on_Circle_Gussian.png')
    # plt.savefig(figfpath, bbox_inches='tight')
    # plt.close(fig)

    # calculate time cost

    # ptrain_ptest = [
    #                    (r'emotion.train.16000x6.jsonl', r'emotion.test.2000x6.jsonl'),
    #                    (r'yelp.train.650000x5.jsonl', r'yelp.test.50000x5.jsonl'),
    #                    (r'agnews.train.120000x4.jsonl', r'agnews.test.7600x4.jsonl'),
    #                    (r'imdb.train.25000x2.jsonl', r'imdb.test.25000x2.jsonl'),
    #                    (r'dbpedia.train.560000x14.jsonl', r'dbpedia.test.70000x14.jsonl'),
    #                    (r'amazonpolar.train.3600000x2.jsonl', r'amazonpolar.test.400000x2.jsonl')
    #                ][:]
    #
    # X_path_s = []
    # for ptrain, ptest in tqdm(ptrain_ptest):
    #     dataset_name = ptrain.split('.')[0]
    #
    #     dm = DataManager(path_data_jsonl_train=ptrain, path_data_jsonl_test=ptest)
    #     X_train = dm.get_X_train()
    #     X_test = dm.get_X_test()
    #
    #     X_path_s.append((X_train, ptrain))
    #     X_path_s.append((X_test, ptest))
    #
    # X_path_s.sort(key=lambda X_path: len(X_path[0]))
    #
    # Ns = [len(X_path[0]) for X_path in X_path_s][:]
    # method2costs = {}
    # for method in tqdm(methods[:]):
    #     method2costs[method.__name__] = []
    #     for X, path in tqdm(X_path_s[:]):
    #         N = len(X)
    #
    #         w = Watch()
    #         try:
    #             idxs = method(X)
    #
    #             if 'train' in path:
    #                 idxs = np.array(idxs)
    #
    #                 fpath = os.path.join(Preset.root, 'c_idxs', f"{method.__name__}.{path.replace('.jsonl', '.idxs.npy')}")
    #                 np.save(fpath, idxs)
    #
    #         except Exception as e:
    #             print(traceback.format_exc())
    #             pass
    #         cost = w.see()
    #
    #         print(f"{N}, {cost}")
    #
    #         method2costs[method.__name__].append(cost)
    #
    # fig3 = plt.figure(figsize=(24, 13))
    #
    # for method_name, costs in method2costs.items():
    #     plt.plot(Ns[:len(costs)], [c.total_seconds() for c in costs])
    #     plt.scatter(Ns[:len(costs)], [c.total_seconds() for c in costs], color=plt.gca().lines[-1].get_color(), label=f'{method_name}')
    #
    # plt.legend()
    #
    # path_figure = os.path.join(Preset.root, r'd_figures')
    # figfpath = os.path.join(path_figure, f'methods_time_cost_compare.png')
    # plt.savefig(figfpath, bbox_inches='tight')
    # plt.close()

    # plt.show(block=True)

    # from pick last to first,                                              black to white
    # cmap = plt.cm.get_cmap('bone')
    # plt.scatter(data_BxF[nidxs[::-1], 0], data_BxF[nidxs[::-1], 1], marker='o', s=5, c=cmap(np.linspace(0, 1, len(nidxs) + 1))[:-1])

    #
    # tree = get_MBTree(data_BxF)
    #
    # fig, ax = plt.subplots(figsize=(10, 10))
    #
    #
    # def draw(ax, tree, data_BxF):
    #
    #     cur_idx = tree['idx']
    #
    #     subs = tree['subs']
    #     cur_point = data_BxF[cur_idx, :]
    #
    #     for sub in subs:
    #         sub_idx = sub['idx']
    #         sub_point = data_BxF[sub_idx, :]
    #
    #         plt.arrow(cur_point[0], cur_point[1], sub_point[0] - cur_point[0], sub_point[1] - cur_point[1], color='white')
    #         # ax.plot([cur_point[0], sub_point[0]], [cur_point[1], sub_point[1]], marker='o', color='white')
    #
    #         print(f'{cur_idx}->{sub_idx}')
    #         draw(ax, sub, data_BxF)
    #     ax.annotate(f"{tree['d']}",
    #                 xy=(cur_point[0], cur_point[1]), xycoords='data',
    #                 xytext=(0, 0), textcoords='offset points',
    #                 horizontalalignment='right', verticalalignment='bottom')
    #
    #
    # draw(ax, tree, data_BxF)
    # plt.show(block=True)

    #
    # chain2idxs = {0: list(range(len(data_BxF)))}
    # build_tree(data_BxF, chain2idxs)
    # print(f"draw graph")
    #
    # chain2idxs = chain2idxs[0]
    # for c0, c1_chain2idxs in chain2idxs.items():
    #     if isinstance(c1_chain2idxs, list):
    #         plt.scatter(data_BxF[c1_chain2idxs, 0], data_BxF[c1_chain2idxs, 1], s=0.5, label=f"cluster={c0}")
    #     elif isinstance(c1_chain2idxs, dict):
    #         for c1, c2_chain2idxs in c1_chain2idxs.items():
    #             idxs = get_list(c2_chain2idxs)
    #             plt.scatter(data_BxF[idxs, 0], data_BxF[idxs, 1], s=0.5, label=f"cluster={c0}x{c1}")
    # plt.legend()
    # plt.show(block=True)
