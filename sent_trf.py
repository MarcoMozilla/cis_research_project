import os
from preset import Preset
from sentence_transformers import SentenceTransformer

path_model_all_mpnet_base_v2 = os.path.join(Preset.root, r'f_models', r'all-mpnet-base-v2')

if not os.path.exists(path_model_all_mpnet_base_v2):
    model_sent2vct = SentenceTransformer('all-mpnet-base-v2')
    model_sent2vct.save(path_model_all_mpnet_base_v2)

model_sent2vct = SentenceTransformer(path_model_all_mpnet_base_v2)

# TODO using models, preprocess text data to buffer
# the buffer will contains text to vector pair
# by analyzing vectors we could find some clusters
