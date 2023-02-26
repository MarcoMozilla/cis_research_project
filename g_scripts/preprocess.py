import os

from datasets import load_dataset

from e_main.preset import Preset
from e_main.tools import save_jsonx


# add preprocess script from b_data to c_buffer


def load_imdb_dataset(dataset=None):
    path_train = os.path.join(Preset.root, r'b_data', 'imdb.train.jsonl')
    path_test = os.path.join(Preset.root, r'b_data', 'imdb.test.jsonl')

    if dataset is None:
        dataset = load_dataset('imdb')

    for tag, fname in zip(['train', 'test'], [path_train, path_test]):
        dctns = []
        for idx in range(len(dataset[tag])):
            dataset_tag = dataset[tag]

            dctn_cook = {'text': dataset_tag[idx]['text'],
                         'label': str(dataset_tag[idx]['label'])}
            dctns.append(dctn_cook)
        save_jsonx(dctns, fname)
        print(f'{fname} : #{len(dctns)}')


# TODO other dataset loading function


if __name__ == '__main__':
    pass
    load_imdb_dataset()
