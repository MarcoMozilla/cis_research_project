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
for fname in fnames[:]:
    fpath = os.path.join(path_tables, fname)
    body, tp, _, _ = fname.split('.')
    dataset_name, NN, F, K, H, L, algo_name, pcent = body.split('_')

    nnsent = f"{NN}_{F}_{K}_{H}_{L}"
    nns.add(nnsent)

    dataset_names.add(dataset_name)
    algo_names.add(algo_name)

    df = pd.read_csv(fpath, encoding=const.ecd_utf8sig)

    for col in [col_f1, col_accuracy, col_precision, col_recall]:
        dctn = {'dataset_name': dataset_name,
                'tp': tp,
                'algo_name': algo_name,
                'pcent': int(pcent),
                'col': col,
                'nn': nnsent,
                'value': df.iloc[-1][col]
                }
        dctns.append(dctn)

dctns.sort(key=lambda dctn: dctn['pcent'])
rdf = pd.DataFrame(dctns)
fpath = os.path.join(path_report, r'smry_score.csv')
rdf.to_csv(fpath, index=False, encoding=const.ecd_utf8sig)

cols = [col_f1, col_accuracy, col_precision, col_recall][:1]

dataset_names = ['agnews', 'amazonpolar', 'dbpedia', 'emotion', 'imdb', 'yelp']
# dataset_names = ['emotion', 'yelp']

fig, axs = plt.subplots(len(dataset_names), 1, figsize=(13, 24), sharex='all')

markers = ['.', '^', 'v']
algo_names = ['randomsample', 'SampleAF', 'SampleBF']

for aidx, col in enumerate(cols[:1]):

    for dataset_id, dataset_name in enumerate(dataset_names):
        for nn in nns:
            # colors = plt.get_cmap('brg')(np.linspace(0, 1, len(dataset_names) + 2))[1:-1]
            for algo_name, marker in zip(algo_names, markers):
                for tp in ['train', 'test']:
                    label = f'{dataset_name}-{nn}-{algo_name}-{tp}'

                    sub_rdf = rdf[(rdf['dataset_name'] == dataset_name) & (rdf['nn'] == nn) & (rdf['algo_name'] == algo_name) & (rdf['col'] == col) & (rdf['tp'] == tp)]
                    if len(sub_rdf) > 0:
                        values = sub_rdf['value'].tolist()
                        pcents = sub_rdf['pcent'].tolist()

                        # color = colors[dataset_id]

                        if tp == 'train':
                            axs[dataset_id].plot(pcents, values, alpha=1, marker=marker, label=label)

                        elif tp == 'test':
                            # color = plt.gca().lines[-1].get_color()
                            axs[dataset_id].plot(pcents, values, alpha=1, marker=marker, linestyle='dotted', label=label)

        axs[dataset_id].legend()
        axs[dataset_id].set_xscale('log')

plt.show(block=True)

if __name__ == '__main__':
    pass
