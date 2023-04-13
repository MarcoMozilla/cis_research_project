import os
import pathlib

import numpy as np
from tqdm import tqdm

from e_main.tools import read_jsonx
from preset import Preset
from sentence_transformers import SentenceTransformer

path_model_all_mpnet_base_v2 = os.path.join(Preset.root, r'a_models', r'all-mpnet-base-v2')

if not os.path.exists(path_model_all_mpnet_base_v2):
    model_sent2vct = SentenceTransformer('all-mpnet-base-v2')
    model_sent2vct.save(path_model_all_mpnet_base_v2)

model_sent2vct = SentenceTransformer(path_model_all_mpnet_base_v2)


# TODO using models, preprocess text data to buffer
# the buffer will contains text to vector pair
# by analyzing vectors we could find some clusters


def make_encode_vector_buffer():
    path_dir = os.path.join(Preset.root, r'b_data')
    fnames = [fname for fname in os.listdir(path_dir) if pathlib.Path(fname).suffix == '.jsonl']

    for fname in tqdm(fnames):

        fname_out = fname.replace('.jsonl', '.X.npy')
        fpath_out = os.path.join(Preset.root, r'c_buffer', fname_out)

        if not os.path.exists(fpath_out):

            fpath = os.path.join(path_dir, fname)

            dctns = read_jsonx(fpath)

            Xs = []
            for dctn in tqdm(dctns):
                text = dctn['text']

                vct = model_sent2vct.encode(text)
                Xs.append(vct)
            X = np.stack(Xs, axis=0)

            np.save(fpath_out, X)
            print(f'save #X = {X.shape}')


if __name__ == '__main__':
    pass
    make_encode_vector_buffer()
