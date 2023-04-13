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

names = [(r'emotion.train.16000x6.jsonl', r'emotion.test.2000x6.jsonl'),
         (r'yelp.train.650000x5.jsonl', r'yelp.test.50000x5.jsonl'),
         (r'agnews.train.120000x4.jsonl', r'agnews.test.7600x4.jsonl'),
         (r'imdb.train.25000x2.jsonl', r'imdb.test.25000x2.jsonl'),
         (r'dbpedia.train.560000x14.jsonl', r'dbpedia.test.70000x14.jsonl'),
         (r'amazonpolar.train.3600000x2.jsonl', r'amazonpolar.test.400000x2.jsonl')]

name_test2N = {}
name_test2C = {}
for fname_train, fname_test in names:
    name_test, tp, N_x_C, suffix = fname_test.split('.')

    N, C = N_x_C.split('x')
    N = int(N)
    C = int(C)
    name_test2N[name_test] = N
    name_test2C[name_test] = C

path_smry = os.path.join(r'C:\Users\PC\Dropbox\_Research_Course\project\cis_research_project\d_report', 'smry_score.csv')

df = pd.read_csv(path_smry, encoding=const.ecd_utf8sig)

algos = [
    'RAND'
    , 'CVMean'
    , 'CVMedian'
    , 'EVMedian'
    , 'EVMean'
    , 'CV_EV_RAND'
]

df = df[(df['tp'] == 'test') & (df['col'] == 'F1') & (df['algo_name'].isin(algos))]
df.drop(columns=['nn'])
df['Ns'] = [name_test2N[nm] for nm in df['dataset_name'].values.tolist()]
df['Cs'] = [name_test2C[nm] for nm in df['dataset_name'].values.tolist()]

df['precent'] = df['pcent'] / df['Ns']

box = []
for dataset_name in set(df['dataset_name'].values.tolist()):
    for Nitems in sorted(list(set(df['pcent'].values.tolist())))[:-1]:
        df_sub = df[(df['dataset_name'] == dataset_name) & (df['pcent'] == Nitems)]

        try:
            v_rand = df_sub[df_sub['algo_name'] == 'RAND']['value'].values.tolist()[0]
            vs_norand = np.array(df_sub[df_sub['algo_name'] != 'RAND']['value'].values.tolist())

            vs_norand_min = np.min(vs_norand)
            vs_norand_ave = np.mean(vs_norand)
            vs_norand_max = np.max(vs_norand)

            vs_norand_min_shift = vs_norand_min - v_rand
            vs_norand_ave_shift = vs_norand_ave - v_rand
            vs_norand_max_shift = vs_norand_max - v_rand

            dctn = {
                'dataset_name': dataset_name,
                '#items': Nitems,
                'tp': df_sub['tp'].values.tolist()[0],
                'percent': df_sub['precent'].values.tolist()[0],
                'F1 (trivial)': np.round(1 / name_test2C[dataset_name], 4),
                'F1 (rand)': np.round(v_rand, 4),
                'F1 (VSD min)': np.round(vs_norand_min, 4),
                'F1 (VSD ave)': np.round(vs_norand_ave, 4),
                'F1 (VSD max)': np.round(vs_norand_max, 4),
                'F1 (VSD min - rand)': np.round(vs_norand_min_shift, 4),
                'F1 (VSD ave - rand)': np.round(vs_norand_ave_shift, 4),
                'F1 (VSD max - rand)': np.round(vs_norand_max_shift, 4),
            }

            box.append(dctn)
        except Exception as err:
            pass

path_smry_make = os.path.join(r'C:\Users\PC\Dropbox\_Research_Course\project\cis_research_project\d_report', 'smry_score.make.csv')

ndf = pd.DataFrame(box, )
ndf.sort_values(by=['#items', 'dataset_name'], inplace=True)
ndf.to_csv(path_smry_make, encoding=const.ecd_utf8sig, index=False)

if __name__ == '__main__':
    pass
