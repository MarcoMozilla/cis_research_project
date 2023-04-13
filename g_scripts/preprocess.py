import os

from datasets import load_dataset

from e_main.preset import Preset
from e_main.tools import save_jsonx


# add preprocess script from b_data to c_buffer

path_b_data = os.path.join(Preset.root, r'b_data')

def load_dataset_imdb(dataset):
    body_name = 'imdb'

    for tp in ['train', 'test']:
        dctns = []
        for idx in range(len(dataset[tp])):
            dataset_tag = dataset[tp]

            dctn_cook = {'text': dataset_tag[idx]['text'],
                         'label': str(dataset_tag[idx]['label'])}
            dctns.append(dctn_cook)

        B = len(dctns)
        C = len({dctn['label'] for dctn in dctns})
        fname = os.path.join(path_b_data, f'{body_name}.{tp}.{B}x{C}.jsonl')

        save_jsonx(dctns, fname)
        print(f'{fname} : #{len(dctns)}')


def load_dataset_agnews(dataset):
    body_name = 'agnews'

    for tp in ['train', 'test']:
        dctns = []
        for idx in range(len(dataset[tp])):
            dataset_tag = dataset[tp]

            dctn_cook = {'text': dataset_tag[idx]['text'],
                         'label': str(dataset_tag[idx]['label'])}
            dctns.append(dctn_cook)

        B = len(dctns)
        C = len({dctn['label'] for dctn in dctns})
        fname = os.path.join(path_b_data, f'{body_name}.{tp}.{B}x{C}.jsonl')

        save_jsonx(dctns, fname)
        print(f'{fname} : #{len(dctns)}')


# TODO other dataset loading function
def load_dataset_dbpedia(dataset):
    body_name = 'dbpedia'

    for tp in ['train', 'test']:
        dctns = []
        for idx in range(len(dataset[tp])):
            dataset_tag = dataset[tp]

            dctn_cook = {'text': f"""# {dataset_tag[idx]['title']}\n{dataset_tag[idx]['content']}""",
                         'label': str(dataset_tag[idx]['label'])}
            dctns.append(dctn_cook)

        B = len(dctns)
        C = len({dctn['label'] for dctn in dctns})
        fname = os.path.join(path_b_data, f'{body_name}.{tp}.{B}x{C}.jsonl')

        save_jsonx(dctns, fname)
        print(f'{fname} : #{len(dctns)}')


def load_dataset_emotion(dataset):
    body_name = 'emotion'

    for tp in ['train', 'test']:
        dctns = []
        for idx in range(len(dataset[tp])):
            dataset_tag = dataset[tp]

            dctn_cook = {'text': dataset_tag[idx]['text'],
                         'label': str(dataset_tag[idx]['label'])}
            dctns.append(dctn_cook)

        B = len(dctns)
        C = len({dctn['label'] for dctn in dctns})
        fname = os.path.join(path_b_data, f'{body_name}.{tp}.{B}x{C}.jsonl')

        save_jsonx(dctns, fname)
        print(f'{fname} : #{len(dctns)}')

def load_dataset_yelp(dataset):
    body_name = 'yelp'

    for tp in ['train', 'test']:
        dctns = []
        for idx in range(len(dataset[tp])):
            dataset_tag = dataset[tp]

            dctn_cook = {'text': dataset_tag[idx]['text'],
                         'label': str(dataset_tag[idx]['label'])}
            dctns.append(dctn_cook)

        B = len(dctns)
        C = len({dctn['label'] for dctn in dctns})
        fname = os.path.join(path_b_data, f'{body_name}.{tp}.{B}x{C}.jsonl')

        save_jsonx(dctns, fname)
        print(f'{fname} : #{len(dctns)}')

def load_dataset_amazonpolar(dataset):
    body_name = 'amazonpolar'

    for tp in ['train', 'test']:
        dctns = []
        for idx in range(len(dataset[tp])):
            dataset_tag = dataset[tp]

            dctn_cook = {'text': f"""# {dataset_tag[idx]['title']}\n{dataset_tag[idx]['content']}""",
                         'label': str(dataset_tag[idx]['label'])}
            dctns.append(dctn_cook)

        B = len(dctns)
        C = len({dctn['label'] for dctn in dctns})
        fname = os.path.join(path_b_data, f'{body_name}.{tp}.{B}x{C}.jsonl')

        save_jsonx(dctns, fname)
        print(f'{fname} : #{len(dctns)}')

if __name__ == '__main__':
    pass

    # dataset_imdb = load_dataset('imdb')
    # load_dataset_imdb(dataset_imdb)
    #
    # dataset_agnews = load_dataset('ag_news')
    # load_dataset_agnews(dataset_agnews)
    #
    # dataset_dbpedia = load_dataset('dbpedia_14')
    # load_dataset_dbpedia(dataset_dbpedia)

    # dataset_emotion = load_dataset('emotion')
    # load_dataset_emotion(dataset_emotion)

    # dataset_yelp = load_dataset('yelp_review_full')
    # load_dataset_yelp(dataset_yelp)

    # dataset_amazon = load_dataset('amazon_polarity')
    # load_dataset_amazonpolar(dataset_amazon)

