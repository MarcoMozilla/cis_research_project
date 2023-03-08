import os
import numpy as np
from torch import nn

from e_main.data_manager import DataManager
from e_main.preset import Preset
from f_nn.nnManager import ModelManager, get_gen_minibatch_ridxs, get_label_weights, get_confuseMatrix_and_scoreTable
from f_nn.nnModel import ResNetSimple
from f_nn.nnUtil import array2tensor, fm_one_hot, to_one_hot

N_features = 768

ptrain_ptest = [
    (r'emotion.train.16000x6.jsonl', r'emotion.test.2000x6.jsonl'),
    (r'yelp.train.650000x5.jsonl', r'yelp.test.50000x5.jsonl'),
    (r'agnews.train.120000x4.jsonl', r'agnews.test.7600x4.jsonl'),
    (r'imdb.train.25000x2.jsonl', r'imdb.test.25000x2.jsonl'),
    (r'dbpedia.train.560000x14.jsonl', r'dbpedia.test.70000x14.jsonl'),
    (r'amazonpolar.train.3600000x2.jsonl', r'amazonpolar.test.400000x2.jsonl')
]

for ptrain, ptest in ptrain_ptest:
    dataset_name = ptrain.split('.')[0]

    dm = DataManager(path_data_jsonl_train=ptrain, path_data_jsonl_test=ptest)
    X_train = dm.get_X_train()
    y_train = dm.get_y_train()
    X_test = dm.get_X_test()
    y_test = dm.get_y_test()
    rids_RandomSample = dm.get_rids_random_sample()

    N_labels = dm.N_labels

    N_hidden = int(N_labels * 2)
    N_layers = int(N_labels * 2)
    nn_model = ResNetSimple(features=N_features, clusters=N_labels, hidden_size=N_hidden, layers=N_layers)

    print(f'dataset_name={dataset_name}')
    print(f'N_labels={N_labels}')

    pcents = [
        100, 200, 300, 500,
        1000, 2000, 3000, 5000,
        10000, 20000, 30000, 50000,
        100000, 200000, 300000, 500000,
        1000000, 2000000, 3000000, 5000000
    ]
    
    for pcent in pcents:

        if pcent <= len(rids_RandomSample):
            print(f'pcent={pcent}')

            rids_RandomSample_sub = rids_RandomSample[:pcent]

            fname_model = f'{dataset_name}_ResNetSimple_F{N_features}_K{N_labels}_H{N_hidden}_L{N_layers}_randomsample_{pcent}.pt'

            if not os.path.exists(fname_model):
                mm_pcent_sub = ModelManager(model=nn_model,
                                            fname_model=fname_model,
                                            X_train=X_train[rids_RandomSample_sub, :],
                                            y_train=y_train[rids_RandomSample_sub],
                                            X_test=X_test,
                                            y_test=y_test
                                            )

                mm_pcent_sub.train_with_minibatch(lr=1e-2, epoch_show_every=50)

if __name__ == '__main__':
    pass
