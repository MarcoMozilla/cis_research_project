import numpy as np
import spacy
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

from e_main.tools import read_jsonx

mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['axes.unicode_minus'] = False
plt.style.use('Solarize_Light2')

path_dir = os.path.join(Preset.root, r'b_data')
# fnames = [fname for fname in os.listdir(path_dir) if pathlib.Path(fname).suffix == '.jsonl']

fnames = [

    'agnews.test.7600x4.jsonl',
    'agnews.train.120000x4.jsonl',
    'amazonpolar.test.400000x2.jsonl',
    'amazonpolar.train.3600000x2.jsonl',
    'dbpedia.test.70000x14.jsonl',
    'dbpedia.train.560000x14.jsonl',
    'emotion.test.2000x6.jsonl',
    'emotion.train.16000x6.jsonl',
    'imdb.test.25000x2.jsonl',
    'imdb.train.25000x2.jsonl',
    'yelp.test.50000x5.jsonl',
    'yelp.train.650000x5.jsonl'
]

# nlp = spacy.load("en_core_web_trf")
# text = "How are you today? \n\nI hope you have a great day"
# tokens = nlp(text)
# for sent in tokens.sents:
#     print(sent.text)


fname2dctns = {}
for fname in fnames[:]:
    fpath = os.path.join(path_dir, fname)

    dctns = read_jsonx(fpath)
    fname2dctns[fname] = dctns

    for dctn in tqdm(dctns):
        text = dctn['text']

        text = text.replace('<br />', '\n')
        paras = text.split('\n')

        text = '\n'.join([para for para in paras if para])
        len_char = len(text)

        dctn['len_char'] = len_char

fig, axs = plt.subplots(len(fname2dctns) // 2, 1, figsize=(13, 24))

for idx in range(0, len(fnames) // 2):
    len_char_s_test = [dctn['len_char'] for dctn in fname2dctns[fnames[idx * 2]]]
    len_char_s_train = [dctn['len_char'] for dctn in fname2dctns[fnames[idx * 2 + 1]]]

    axs[idx].hist(len_char_s_test, bins=50, density=True, alpha=0.5, label=f"{fnames[idx * 2]}")
    axs[idx].hist(len_char_s_train, bins=50, density=True, alpha=0.5, label=f"{fnames[idx * 2 + 1]}")

    min_len_char = min(min(len_char_s_test), min(len_char_s_train))
    max_len_char = max(max(len_char_s_test), max(len_char_s_train))

    print(f"{fnames[idx * 2]}:")
    print(f"min_len_char: {min_len_char} ")
    print(f"max_len_char: {max_len_char} ")

    axs[idx].legend()

plt.show(block=True)
