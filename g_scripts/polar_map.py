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

path_figure = os.path.join(Preset.root, r'd_figures')


sample_data = np.array([np.random.rand(2) * 2 - 1 for i in range(100000)])
normal_data = sample_data / np.sqrt(np.sum(sample_data ** 2, axis=1))[:, None]

theta = np.arcsin(normal_data[:,1])


fig, axs = plt.subplots(2,1,figsize=(13,24))

axs[0].scatter(sample_data[:, 0], sample_data[:, 1], s=0.5, label='sample')
axs[0].scatter(normal_data[:, 0], normal_data[:, 1], s=0.5, label='normal')


axs[1].hist(theta,bins=1000,label='density')


plt.legend()

figfpath = os.path.join(path_figure, f'random_sample_map_to_polar_coord_theta_density.png')
plt.savefig(figfpath, bbox_inches='tight')
plt.close(fig)
