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

mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['axes.unicode_minus'] = False
plt.style.use('Solarize_Light2')

path_buffer = os.path.join(Preset.root, r'c_buffer')
path_figure = os.path.join(Preset.root, r'd_figures')

fnames = [fname for fname in os.listdir(path_buffer) if pathlib.Path(fname).suffix == '.npy']

fname2X = {}
for fname in tqdm(fnames):
    name, tp, shape, varname, suffix = fname.split('.')

    if tp in {'train'}:

        if not fname in fname2X:
            fpath = os.path.join(path_buffer, fname)
            X = np.load(fpath)
            fname2X[fname] = X

N_features = 10

ratio_cross_feature = 1
# for fidx in tqdm(range(N_features), total=N_features):
#
#     fig = plt.figure(figsize=(24, 13))
#     fig2 = plt.figure(figsize=(24, 13))
#     for fname in fnames:
#         if fname in fname2X:
#             X = fname2X[fname]
#             dens, binEdges = np.histogram(X[:, fidx], bins=100)
#             bincenters = 0.5 * (binEdges[1:] + binEdges[:-1])
#             dens = dens / X.shape[0]
#             plt.plot(bincenters, dens, alpha=0.7, linewidth=1, label=fname)
#
#     plt.legend()
#     figfpath = os.path.join(path_figure, f'encode_vector_F{fidx}.png')
#     plt.savefig(figfpath, bbox_inches='tight')
#     plt.close(fig)
#
#     for fidx_a in tqdm(range(N_features), total=N_features):
#         if fidx_a > fidx and random.random() < ratio_cross_feature:
#             fig = plt.figure(figsize=(24, 24))
#             for fname in fnames:
#                 if fname in fname2X:
#                     X = fname2X[fname]
#                     plt.scatter(X[:, fidx], X[:, fidx_a], alpha=0.1, s=0.5 ** 2, label=fname)
#             plt.legend()
#             figfpath = os.path.join(path_figure, f'encode_vector_F{fidx}xF{fidx_a}.png')
#             plt.savefig(figfpath, bbox_inches='tight')
#             plt.close(fig)
#
# N_features = 768
# fig2 = plt.figure(figsize=(24, 13))
# for fname in fnames:
#     if fname in fname2X:
#         X = fname2X[fname]
#
#         a = (np.sqrt(np.sum(X[:, :N_features // 2] ** 2, axis=1)))
#         b = (np.sqrt(np.sum(X[:, N_features // 2:] ** 2, axis=1)))
#
#         plt.scatter(a, b, alpha=0.1, s=0.5 ** 2, label=fname)
#
# plt.legend()
# figfpath = os.path.join(path_figure, f'encode_vector_length_distribution.png')
# plt.savefig(figfpath, bbox_inches='tight')
# plt.close(fig2)

# plt.show()


from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, KernelPCA

fname_pick = 'emotion.train.16000x6.X.npy'

X = fname2X[fname_pick]

tsf_TSNE = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3)
tsf_PCA = PCA(n_components=2)
tsf_KernelPCA_linear = KernelPCA(n_components=2, kernel='linear')
tsf_KernelPCA_poly = KernelPCA(n_components=2, kernel='poly')
tsf_KernelPCA_rbf = KernelPCA(n_components=2, kernel='rbf')
tsf_KernelPCA_sigmoid = KernelPCA(n_components=2, kernel='sigmoid')
tsf_KernelPCA_cosine = KernelPCA(n_components=2, kernel='cosine')

name2model = {

    'tsf_TSNE': tsf_TSNE,
    'tsf_PCA': tsf_PCA,
    'tsf_KernelPCA_linear': tsf_KernelPCA_linear,
    'tsf_KernelPCA_poly': tsf_KernelPCA_poly,
    'tsf_KernelPCA_rbf': tsf_KernelPCA_rbf,
    'tsf_KernelPCA_sigmoid': tsf_KernelPCA_sigmoid,
    'tsf_KernelPCA_cosine': tsf_KernelPCA_cosine

}

dm = DataManager(path_data_jsonl_train=r'emotion.train.16000x6.jsonl', path_data_jsonl_test=r'emotion.test.2000x6.jsonl')

for model_name, model_reduce_dim in tqdm(name2model.items()):
    fig3 = plt.figure(figsize=(24, 24))
    x2d = model_reduce_dim.fit_transform(X)

    for cid, rids in enumerate(dm.cid2rids_train):
        plt.scatter(x2d[rids, 0], x2d[rids, 1], alpha=1, s=0.5 ** 2, label=f"{fname_pick}, cid={cid}")

    plt.legend()

    figfpath = os.path.join(path_figure, f'{fname_pick}.{model_name}.png')
    plt.savefig(figfpath, bbox_inches='tight')
    plt.close(fig3)
