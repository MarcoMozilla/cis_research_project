import numpy as np
from matplotlib import pyplot as plt
from scipy.linalg import polar
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

mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['axes.unicode_minus'] = False
plt.style.use('Solarize_Light2')




def get_MBTree_idx_N_ck2idxs(data_BxF, idxs):
    B, F = data_BxF.shape
    N = len(idxs)
    po = 2 ** np.arange(F)
    if len(idxs) == 1:
        mean_idx = idxs[0]
        return mean_idx, N, {}
    else:
        mean_data_F = np.mean(data_BxF[idxs, :], axis=0)
        mean_local_idx = np.argmin(np.sum((data_BxF[idxs, :] - mean_data_F) ** 2, axis=1))
        mean_idx = idxs[mean_local_idx]
        wash_idxs = idxs[:mean_local_idx] + idxs[mean_local_idx + 1:]

        ck2idxs = {}
        if len(wash_idxs) > 1:
            chain_sub_keys = np.array(data_BxF[wash_idxs, :] < data_BxF[mean_idx, :], dtype=bool) @ po
            for ck, idx in zip(chain_sub_keys, wash_idxs):
                if not ck in ck2idxs:
                    ck2idxs[ck] = []
                ck2idxs[ck].append(idx)

        return mean_idx, N, ck2idxs


def get_MBTree(data_BxF, idxs=None, d=0):
    if idxs is None:
        idxs = list(range(len(data_BxF)))
    mean_idx, N, ck2idxs = get_MBTree_idx_N_ck2idxs(data_BxF, idxs)

    subs = []
    for ck, sub_idxs in ck2idxs.items():
        sub = get_MBTree(data_BxF, sub_idxs, d=d + 1)
        subs.append(sub)

    return {'idx': mean_idx, 'N': N, 'subs': subs, 'd': d}


if __name__ == '__main__':
    pass

    data_BxF = np.random.rand(1000, 2) * 2 - 1

    tree = get_MBTree(data_BxF)

    fig, ax = plt.subplots(figsize=(20, 20))


    def draw(ax, tree, data_BxF):

        cur_idx = tree['idx']

        subs = tree['subs']
        cur_point = data_BxF[cur_idx, :]

        for sub in subs:
            sub_idx = sub['idx']
            sub_point = data_BxF[sub_idx, :]

            plt.arrow(cur_point[0], cur_point[1], sub_point[0] - cur_point[0], sub_point[1] - cur_point[1], color='white')
            # ax.plot([cur_point[0], sub_point[0]], [cur_point[1], sub_point[1]], marker='o', color='white')

            print(f'{cur_idx}->{sub_idx}')
            draw(ax, sub, data_BxF)
        ax.annotate(f"{tree['d']}",
                    xy=(cur_point[0], cur_point[1]), xycoords='data',
                    xytext=(0, 0), textcoords='offset points',
                    horizontalalignment='right', verticalalignment='bottom')


    draw(ax, tree, data_BxF)
    plt.show(block=True)

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
