import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.linalg import polar
import os
import pathlib
import random

import numpy as np
from tqdm import tqdm

from e_main import const
from e_main.preset import Preset

# draw train vector space in to figures and see distribution
import matplotlib.pyplot as plt
import matplotlib as mpl
from pprint import pprint as p

mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['axes.unicode_minus'] = False
plt.style.use('Solarize_Light2')

from f_nn.nnManager import col_f1, col_accuracy, col_precision, col_recall

path_tables = os.path.join(Preset.root, 'd_tables')
path_report = os.path.join(Preset.root, 'd_report')

fnames = [fname for fname in os.listdir(path_tables) if fname.endswith('.score.csv')]

dctns = []
dataset_names = set()
algo_names = set()
nns = set()

dataset_name2K = {}

for fname in fnames[:]:
    fpath = os.path.join(path_tables, fname)
    body, tp, _, _ = fname.split('.')
    body_parts = body.split('_')

    pcent = body_parts[-1]
    algo_name = '_'.join(body_parts[6:-1])
    L = body_parts[5]
    H = body_parts[4]
    K = body_parts[3]
    F = body_parts[2]
    NN = body_parts[1]
    dataset_name = body_parts[0]

    dataset_name2K[dataset_name] = float(K[1:])

    nnsent = f"{NN}_{F}_{K}_{H}_{L}"
    nns.add(nnsent)

    dataset_names.add(dataset_name)
    algo_names.add(algo_name)

    df_score = pd.read_csv(fpath, encoding=const.ecd_utf8sig, index_col='Unnamed: 0')
    dctn_score = {}
    dctn_score[col_accuracy] = df_score['f1-score']['accuracy']
    dctn_score[col_f1] = df_score['f1-score']['macro avg']
    dctn_score[col_precision] = df_score['precision']['macro avg']
    dctn_score[col_recall] = df_score['recall']['macro avg']

    for col in [col_f1, col_accuracy, col_precision, col_recall]:
        dctn = {'dataset_name': dataset_name,
                'tp': tp,
                'algo_name': algo_name,
                'pcent': int(pcent),
                'col': col,
                'nn': nnsent,
                'value': dctn_score[col]
                }
        dctns.append(dctn)

dctns.sort(key=lambda dctn: dctn['pcent'])
rdf = pd.DataFrame(dctns)
fpath = os.path.join(path_report, r'smry_score.csv')
rdf.to_csv(fpath, index=False, encoding=const.ecd_utf8sig)

cols = [col_f1, col_precision, col_recall, col_accuracy]

dataset_names = ['amazonpolar', 'dbpedia', 'yelp', 'agnews', 'imdb', 'emotion'][:]

# dataset_names = ['emotion', 'yelp']


markers = ['.', 'x', 'x', 'x', 'x', 'x', 'x']
algo_names = [
    'RAND'
    , 'CV'
    , 'EV'
    , 'CV_EV'
    , 'CV_RAND'
    , 'CR_CV_RAND'
    , 'CV_EV_RAND'
]

colors = plt.get_cmap('brg')(np.linspace(0, 1, len(algo_names) * 2 + 2))[1:-1:2]

for aidx, col in enumerate(cols):
    fig, axs = plt.subplots(len(dataset_names), 1, figsize=(13, 24), sharex='all')

    for dataset_id, dataset_name in enumerate(dataset_names):
        pcents_all = set()
        for nn in nns:
            for algo_id, (algo_name, marker) in enumerate(zip(algo_names, markers)):
                for tp in ['train', 'test']:
                    label = f'{dataset_name}-{nn}-{algo_name}-{tp}'

                    sub_rdf = rdf[(rdf['pcent'] < 16000) & (rdf['dataset_name'] == dataset_name) & (rdf['nn'] == nn) & (rdf['algo_name'] == algo_name) & (rdf['col'] == col) & (rdf['tp'] == tp)]
                    if len(sub_rdf) > 0:
                        values = sub_rdf['value'].tolist()[:]
                        pcents = sub_rdf['pcent'].tolist()[:]
                        pcents_all |= set(pcents)
                        # color = colors[dataset_id]

                        if tp == 'train':
                            axs[dataset_id].plot(pcents, values, alpha=1, marker=marker, color=colors[algo_id], label=f"{label}={round(values[-1], 4)}")

                        elif tp == 'test':
                            # color = plt.gca().lines[-1].get_color()
                            axs[dataset_id].plot(pcents, values, alpha=1, marker=marker, color=colors[algo_id], linestyle='dotted', label=f"{label}={round(values[-1], 4)}")

        trivial_score = 1 / dataset_name2K[dataset_name]
        axs[dataset_id].axhline(y=trivial_score, color='black', label=f'{dataset_name}-trivial={round(trivial_score, 4)}')

        axs[dataset_id].legend(bbox_to_anchor=(1.01, 1.01), loc=2)
        pcents_all = sorted(list(pcents_all))
        axs[dataset_id].set_xscale('log')
        axs[dataset_id].set_xticks(pcents_all)
        axs[dataset_id].set_xticklabels(pcents_all, minor=False, rotation=-67.5)
        axs[dataset_id].get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())

        axs[dataset_id].ticklabel_format(useOffset=False)
        axs[dataset_id].set_ylabel(f'ave {col} score')

        axs[dataset_id].set_xlabel('#items of train dataset (log scaled)')

    path_figure = os.path.join(Preset.root, r'd_figures')
    figfpath = os.path.join(path_figure, f'model_score_metrix.{col}.png')
    plt.savefig(figfpath, bbox_inches='tight')
    plt.close(fig)

for dataset_id, dataset_name in enumerate(dataset_names):
    fig, axs = plt.subplots(len(cols), 1, figsize=(13, 24), sharex='all')
    for aidx, col in enumerate(cols):

        pcents_all = set()
        for nn in nns:
            for algo_id, (algo_name, marker) in enumerate(zip(algo_names, markers)):
                for tp in ['train', 'test']:
                    label = f'{dataset_name}-{nn}-{algo_name}-{tp}'

                    sub_rdf = rdf[(rdf['dataset_name'] == dataset_name) & (rdf['nn'] == nn) & (rdf['algo_name'] == algo_name) & (rdf['col'] == col) & (rdf['tp'] == tp)]
                    if len(sub_rdf) > 0:
                        values = sub_rdf['value'].tolist()[:]
                        pcents = sub_rdf['pcent'].tolist()[:]
                        pcents_all |= set(pcents)
                        # color = colors[dataset_id]

                        if tp == 'train':
                            axs[aidx].plot(pcents, values, alpha=1, marker=marker, color=colors[algo_id], label=f"{label}={round(values[-1], 4)}")

                        elif tp == 'test':
                            # color = plt.gca().lines[-1].get_color()
                            axs[aidx].plot(pcents, values, alpha=1, marker=marker, color=colors[algo_id], linestyle='dotted', label=f"{label}={round(values[-1], 4)}")

        trivial_score = 1 / dataset_name2K[dataset_name]
        axs[aidx].axhline(y=trivial_score, color='black', label=f'{dataset_name}-trivial={round(trivial_score, 4)}')

        axs[aidx].legend(bbox_to_anchor=(1.01, 1.01), loc=2)

        pcents_all = sorted(list(pcents_all))
        axs[aidx].set_xscale('log')
        axs[aidx].set_xticks(pcents_all)
        axs[aidx].set_xticklabels(pcents_all, minor=False, rotation=-67.5)
        axs[aidx].get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())

        axs[aidx].set_ylabel(f'ave {col} score')
        axs[aidx].set_xlabel('#items of train dataset (log scaled)')

    path_figure = os.path.join(Preset.root, r'd_figures')
    figfpath = os.path.join(path_figure, f'model_score_dataset.{dataset_name}.png')
    plt.savefig(figfpath, bbox_inches='tight')
    plt.close(fig)

# plt.show(block=True)

if __name__ == '__main__':
    pass
