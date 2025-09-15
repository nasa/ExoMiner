"""Test Deep SHAP Explainer for ExoMiner."""

# 3rd party
import shap
from pathlib import Path
# import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import custom_object_scope
# import numpy as np
import  yaml

# local
from models.models_keras import Time2Vec, SplitLayer
from src.utils.utils_dataio import InputFnv2 as InputFn

#%% set filepaths

model_fp = Path('/Users/msaragoc/Projects/exoplanet_transit_classification/exoplanet_dl/exominer_pipeline/data/model.keras')
config_fp = Path('/Users/msaragoc/Projects/exoplanet_transit_classification/exoplanet_dl/exominer_pipeline/config_predict_model.yaml')

#%% load model

print('Loading model...')

custom_objects = {"Time2Vec": Time2Vec, 'SplitLayer': SplitLayer}
with custom_object_scope(custom_objects):
    model = load_model(filepath=model_fp, compile=False)

#%% create background dataset for SHAP

# load config filepath
with open(config_fp, 'r') as config_yaml:
    config = yaml.unsafe_load(config_yaml)

dataset = 'test'
n_bckgrnd_batches = 5
n_bckgrnd_samples = n_bckgrnd_batches * config['inference']['batch_size']

datasets_fps = []

predict_input_fn = InputFn(
        file_paths=config['datasets_fps'][dataset],
        batch_size=config['inference']['batch_size'],
        mode='PREDICT',
        label_map=config['label_map'],
        features_set=config['features_set'],
        multiclass=config['config']['multi_class'],
        feature_map=config['feature_map'],
        label_field_name=config['label_field_name'],
    )

bckgrnd_feats_dict = {}
for batch_i in range(n_bckgrnd_batches):
    
    batch = next(iter(predict_input_fn()))
    batch_feats = batch[0] if isinstance(batch, tuple) else batch
    
    for k, v in batch_feats.items():
        if k not in bckgrnd_feats_dict:
            bckgrnd_feats_dict[k] = [v]
        else:
            bckgrnd_feats_dict[k].append(v)
            
# first_batch = next(iter(predict_input_fn()))
# feats_dict  = first_batch[0] if isinstance(first_batch, tuple) else first_batch
# batch_n     = next(iter(feats_dict.values())).shape[0]

# bg_idx = tf.constant(
#     np.random.choice(config['inference']['batch_size'], 
#                      min(MIN_REF, config['inference']['batch_size']), 
#                      replace=False), dtype=tf.int32
# )
# background = {k: tf.gather(v, bg_idx, axis=0) for k, v in feats_dict.items()}

# input_order       = [t.name.split(":")[0] for t in model.inputs]
# background_inputs = [background[k].numpy() for k in input_order]

background_inputs = [feat_examples.numpy() for _, feat_examples in bckgrnd_feats_dict.items()]

#%% create explainer

explainer = shap.GradientExplainer(model, background_inputs)


