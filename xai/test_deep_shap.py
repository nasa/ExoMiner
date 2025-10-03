"""Test Deep SHAP Explainer for ExoMiner."""

#%% load libraries

# 3rd party
import shap
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import custom_object_scope, plot_model
import numpy as np
import yaml
from functools import partial
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.python.framework import ops
from keras import backend as K
from tqdm import tqdm

# local
from models.models_keras import Time2Vec, SplitLayer, ExoMinerPlusPlus
from src.utils.utils_dataio import InputFnv2 as InputFn
# from src.train.utils_train import filter_examples_tfrecord_label
from src.utils.utils_dataio import get_data_from_tfrecords_for_predictions_table
from query_dv_report_and_summary import get_dv_dataproducts_list, correct_sector_field

#%% set filepaths

experiment_dir = Path('/Users/msaragoc/Projects/exoplanet_transit_classification/experiments/xai/test_deep-shap_clf-head-stellar-dvtce')
model_fp = Path('/Users/msaragoc/Projects/exoplanet_transit_classification/exoplanet_dl/exominer_pipeline/data/exominer-plusplus_cv-iter0-model0_tess-spoc-2min-s1s67_tess-kepler.keras')
config_fp = Path('/Users/msaragoc/Projects/exoplanet_transit_classification/experiments/tess_spoc_2min/tess_paper/cv_tess-spoc-2min_s1-s67_kepler_trainset_tcenumtransits_tcenumtransitsobs_1-12-2025_1036/cv_iter_0/models/model0/config_cv.yaml')
dataset_dir = Path('/Users/msaragoc/Projects/exoplanet_transit_classification/data/tfrecords/tess/cv_tfrecords_tess_spoc_2min_s1-s67_9-24-2024_1159_tcedikcorat_crowdsap_tcedikcocorr_11-23-2024_0047_eval_normalized_cv_iter_0/norm_data')

experiment_dir.mkdir(parents=True, exist_ok=True)

# load config filepath
with open(config_fp, 'r') as config_yaml:
    config = yaml.unsafe_load(config_yaml)

#%% set dataset filepaths

# update datasets filepaths
config['datasets_fps'] = {dataset: [dataset_dir / dataset_fp.name for dataset_fp in dataset_fps] for dataset, dataset_fps in config['datasets_fps'].items()}

dataset_dir_fps = [str(fp) for fp in dataset_dir.glob('shard-*')]

#%% get predictions for model

pred_tbl_dir = Path('/Users/msaragoc/Projects/exoplanet_transit_classification/experiments/tess_spoc_2min/tess_paper/cv_tess-spoc-2min_s1-s67_kepler_trainset_tcenumtransits_tcenumtransitsobs_1-12-2025_1036/cv_iter_0/ensemble_model')
pred_tbls_lst = []
for fp in pred_tbl_dir.glob('ranked_predictions*.csv'):
    pred_tbl = pd.read_csv(fp)
    pred_tbl['dataset'] = fp.stem.split('_')[-1][:-3]
    pred_tbls_lst.append(pred_tbl)
    
pred_tbl = pd.concat(pred_tbls_lst, axis=0, ignore_index=True)

#%% load model

print('Loading model...')

# exominer_config_fp = '/Users/msaragoc/Projects/exoplanet_transit_classification/exoplanet_dl/models/exominer_plusplus.yaml'
# with open(exominer_config_fp, 'r') as config_yaml:
#     exominer_config = yaml.unsafe_load(config_yaml)
# config.update(exominer_config)

custom_objects = {"Time2Vec": Time2Vec, 'SplitLayer': SplitLayer}
with custom_object_scope(custom_objects):
    full_model = load_model(filepath=model_fp, compile=False)
    # model_weights = model.get_weights()

# # set weights using new model
# config['config']['force_softmax'] = False
# config['config']['transformer_branches'] = None
# new_model_obj = ExoMinerPlusPlus(config, config['features_set'])
# new_model = new_model_obj.kerasModel
# new_model.set_weights(model_weights)

print('Model loaded.')

#%% create classification head model and conv model

input_clfh_layers_names = [
    # 'flatten_convfc_diff_img',
    # 'flatten_wscalar_global_flux',
    # 'flatten_wscalar_local_centroid',
    # 'flatten_momentum_dump',
    # 'flatten_wscalar_flux_trend',
    # 'flatten_wscalar_flux_periodogram',
    # 'flatten_wscalar_local_odd_even',
    # 'flatten_wscalar_local_flux',
    # 'flatten_wscalar_local_weak_secondary',
    # 'flatten_wscalar_local_unfolded_flux',
    # 'flatten_wscalar_diff_img_scalars',
    'fc_prelu_global_flux',
    'fc_prelu_local_centroid',
    'fc_prelu_momentum_dump',
    'fc_prelu_flux_trend',
    'fc_prelu_flux_periodogram',
    'fc_prelu_local_odd_even',
    'fc_prelu_local_flux',
    'fc_prelu_local_weak_secondary',
    'fc_prelu_local_unfolded_flux',
    'fc_prelu_diff_img',
    # 'fc_prelu_stellar_scalar',
    # 'fc_prelu_dv_tce_fit_scalar',
    
    'stellar_scalar_input',
    # 'tce_sden_norm',
    # 'tce_steff_norm',
    # 'tce_smet_norm',
    # 'tce_slogg_norm',
    # 'tce_smass_norm',
    # 'tce_sradius_norm',
    
    'dv_tce_fit_scalar_input',
    # 'boot_fap_norm',
    # 'tce_cap_stat_norm',
    # 'tce_hap_stat_norm',
    # 'tce_period_norm',
    # 'tce_max_mult_ev_norm',
    # 'tce_max_sngle_ev_norm',
    # 'tce_robstat_norm',
    # 'tce_model_chisq_norm',
    # 'tce_prad_norm',
]
# input_clfh_model = [model_layer.output for model_layer in full_model.layers if model_layer.name in input_clfh_layers_names]
input_clfh_model = []
for layer_name in input_clfh_layers_names:
    layer = full_model.get_layer(name=layer_name)
    if layer is None:
        raise ValueError(f'Layer {layer_name} not found in model.')
    input_clfh_model.append(layer.output)

clfh_model = tf.keras.Model(inputs=input_clfh_model, outputs=full_model.output)
conv_model = tf.keras.Model(inputs=full_model.inputs, outputs=input_clfh_model)

with open(experiment_dir / 'conv-model-relu-output_summary.txt', 'w') as f:
    conv_model.summary(print_fn=lambda x: f.write(x + '\n'))
with open(experiment_dir / 'clf-head-model-relu-output_summary.txt', 'w') as f:
    clfh_model.summary(print_fn=lambda x: f.write(x + '\n'))
    
plot_model(conv_model,
            to_file=experiment_dir / 'conv-model-relu-output.png',
            show_shapes=True,
            show_layer_names=True,
            rankdir='TB',
            expand_nested=False,
            dpi=96)
plot_model(clfh_model,
            to_file=experiment_dir / 'clf-head-model-relu-output.png',
            show_shapes=True,
            show_layer_names=True,
            rankdir='TB',
            expand_nested=False,
            dpi=96)

# save the models
clfh_model.save(experiment_dir / 'clf-head-model-relu-output.keras')

full_model.save(experiment_dir / 'full-model.keras')

#%% create dataset of inputs to classification head

config['eval'] = {'batch_size': 512}
create_dataset_input_fn = InputFn(
        file_paths=dataset_dir_fps,
        batch_size=config['eval']['batch_size'],
        mode='PREDICT',
        label_map=config['label_map'],
        features_set=config['features_set'],
        multiclass=config['config']['multi_class'],
        feature_map=config['feature_map'],
        label_field_name=config['label_field_name'],
    )

conv_outputs = conv_model.predict(create_dataset_input_fn())

#%% build CSV files with conv outputs

# # inputs for classification head model using sets of three features per branch
# feature_cols = [[f'{l.name}_{feat_i}' for feat_i in range(l.shape[1])] for l in conv_model.outputs]
# feature_cols = [item for sublist in feature_cols for item in sublist]

# inputs for classification head model using the sets of three features per branch except for stellar params and DV TCE fit
stellar_params_feature_order = ['tce_sdens_norm', 'tce_steff_norm', 'tce_smet_norm', 'tce_slogg_norm', 'tce_smass_norm', 'tce_sradius_norm']
dv_tce_fit_feature_order = [
    'boot_fap_norm',
    'tce_cap_stat_norm',
    'tce_hap_stat_norm',
    'tce_period_norm',
    'tce_max_mult_ev_norm',
    'tce_max_sngle_ev_norm',
    'tce_robstat_norm',
    'tce_model_chisq_norm',
    'tce_prad_norm',
]

feature_cols = [[f'{l.name}_{feat_i}' for feat_i in range(l.shape[1])] for l in conv_model.outputs if l.name not in ['stellar_scalar_input/concat:0', 'dv_tce_fit_scalar_input/concat:0']]
feature_cols = [item for sublist in feature_cols for item in sublist]
feature_cols = feature_cols + stellar_params_feature_order + dv_tce_fit_feature_order

stacked_features = np.concatenate(conv_outputs, axis=-1)

dataset_df = pd.DataFrame(stacked_features, columns=feature_cols)

# add metadata columns
data_fields = ['uid', 'label']
metadata = get_data_from_tfrecords_for_predictions_table(['all'],
                                                     data_fields,
                                                    {'all': dataset_dir_fps}
                                                    )
metadata_df = pd.DataFrame(metadata['all'])
full_df = pd.concat([metadata_df, dataset_df], axis=1, ignore_index=False)
# add metadata
full_df.attrs['dataset'] = str(dataset_dir)
full_df.attrs['model'] =  f'convolutional model up to ReLU output plus stellar parameters and DV TCE fit inputs of {str(model_fp)}'
full_df.attrs['created'] = str(pd.Timestamp.now().floor('min'))
with open(experiment_dir / 'conv_features_dataset.csv', "w") as f:
    for key, value in full_df.attrs.items():
        f.write(f"# {key}: {value}\n")
    full_df.to_csv(f, index=False)

#%% plot feature distributions

plot_dir = experiment_dir / 'conv_features_hist'
plot_dir.mkdir(parents=True, exist_ok=True)

full_df = pd.read_csv(experiment_dir / 'conv_features_dataset.csv', comment='#')

bins = np.logspace(-9, 4, 500)
hist_density = False

# feature_cols = [col for col in full_df.columns if col not in ['uid', 'label']]
feature_unique = list(set([col.split('/')[0] for col in feature_cols]))
# for col_i in range(0, len(feature_cols), 3):
for feature in feature_unique:
    
    feats_in_grp = [col for col in feature_cols if feature in col]
    if '/' in feats_in_grp[0]:
        feature_grp_name = feats_in_grp[0].split('/')[0]
    else:
        feature_grp_name = feats_in_grp[0]

    if feature_grp_name == 'boot_fap_norm':
        aaa
    f, ax = plt.subplots(figsize=(8, 6))
    # for col in feature_cols[col_i:col_i+3]:
    for col in feats_in_grp:
        # ax = sns.histplot(full_df, x=col, bins=50, stat='density', common_norm=False, element='step', fill=False)
        ax.hist(full_df[col], bins=bins, density=False, histtype='step', label=col)
    ax.set_title(f'Feature: {feature_grp_name}')
    ax.set_xlabel('Feature Value')
    ax.set_ylabel('TCE Density' if hist_density else 'TCE Count')
    ax.set_xscale('log')
    ax.legend()
    f.tight_layout()
    f.savefig(plot_dir / f'hist_feature_{feature_grp_name}.png')
    plt.close(f)
    
#%% create background dataset for SHAP

# datasets_fps = {
#     'train': dataset_dir_fps[:-1],
#     'test': dataset_dir_fps[-1:],
# }
# config['datasets_fps'] = datasets_fps
# config['label_map'] = {
#     'KP': 1,
#     'CP': 1,
#     'EB': 0,
#     'FP': 0,
#     'NTP': 0,
#     # 'UNK': 0,
#     'NEB': 0,
#     'NPC': 0,
#     'BD': 0,
# }

# predict_input_fn = InputFn(
#         file_paths=config['datasets_fps'][dataset],
#         batch_size=config['inference']['batch_size'],
#         mode='PREDICT',
#         label_map=config['label_map'],
#         features_set=config['features_set'],
#         multiclass=config['config']['multi_class'],
#         feature_map=config['feature_map'],
#         label_field_name=config['label_field_name'],
#         filter_fn=partial(filter_examples_tfrecord_label, label='KP'),
#     )

#%% get background inputs

def filter_planet_examples_tfrecord_label(parsed_features, label_id):
    """ Filters out examples whose label_id is `1`.

    Args:
        parsed_features: tf tensor, parsed features for example
        label_id: tf tensor, label id for example

    Returns: tf boolean tensor

    """

    return tf.squeeze(label_id== 1)

def filter_by_uid(parsed_features, allowed_uids):
    """
    Filters out examples whose 'uid' is not in the allowed_uids list.

    Args:
        parsed_features: dict of tf tensors, parsed features for example
        allowed_uids: tf tensor or python list of allowed UID strings

    Returns:
        tf boolean tensor indicating whether the example should be kept
    """
    
    uid = parsed_features['uid']
    # ensure allowed_uids is a tf constant for comparison
    allowed_uids_tensor = tf.constant(allowed_uids, dtype=tf.string)
    return tf.reduce_any(tf.equal(uid, allowed_uids_tensor))


def make_uid_filter_fn(allowed_uids):
    allowed_uids_tensor = tf.constant(allowed_uids, dtype=tf.string)

    def filter_fn(parsed_features, label_id=None):
        uid = tf.squeeze(parsed_features['uid'])
        return tf.reduce_any(tf.equal(uid, allowed_uids_tensor))

    return filter_fn

def make_uid_filter_out_fn(excluded_uids):
    excluded_uids_tensor = tf.constant(excluded_uids, dtype=tf.string)

    def filter_fn(parsed_features, label_id=None):
        uid = tf.squeeze(parsed_features['uid'])
        return tf.logical_not(tf.reduce_any(tf.equal(uid, excluded_uids_tensor)))

    return filter_fn

# select 500 planet examples for background dataset using dataframe
full_df = pd.read_csv(experiment_dir / 'conv_features_dataset.csv', comment='#')
planets_df = full_df[full_df['label'].isin(['CP', 'KP'])]
planets_df = planets_df.sample(n=500, random_state=42, replace=False)

config['features_set']['quality'] = {'dim': [5, 1], 'dtype': 'float'}

# filter_fn = make_uid_filter_fn(planets_uid_lst)
#
# dataset = 'train'
# n_bckgrnd_batches = 5
# n_bckgrnd_samples = 500
# batch_size = n_bckgrnd_samples // n_bckgrnd_batches
# features_set = config['features_set']
# features_set.update({'uid': {'dim': [1,], 'dtype': 'string'}, 'label': {'dim': [1,], 'dtype': 'string'}})

# background_input_fn = InputFn(
#         file_paths=dataset_dir_fps,  # config['datasets_fps'][dataset],
#         batch_size=batch_size,
#         mode='EVAL',
#         label_map=config['label_map'],
#         features_set=features_set,
#         multiclass=config['config']['multi_class'],
#         feature_map=config['feature_map'],
#         label_field_name=config['label_field_name'],
#         filter_fn=filter_fn,  # partial(filter_by_uid, allowed_uids=planets_uid_lst),  # filter_planet_examples_tfrecord_label, # partial(filter_examples_tfrecord_label, label='KP'),
#     )
# bckgrnd_feats_dict, bckgrnd_data_to_tbl = {}, {field: [] for field in ['uid', 'label']}
bckgrnd_data_to_tbl = {field: [] for field in ['uid', 'label']}
# # for batch_i in range(n_bckgrnd_batches):
# for batch in background_input_fn():

    
#     # batch = next(iter(background_input_fn()))
#     batch_feats = batch[0] if isinstance(batch, tuple) else batch
    
#     for k, v in batch_feats.items():
#         if k in ['uid', 'label']:
#             bckgrnd_data_to_tbl[k].extend([el.decode('utf') for el in np.array(v.numpy().tolist()).flatten()])
#             continue
#         if k not in bckgrnd_feats_dict:
#             bckgrnd_feats_dict[k] = [v]
#         else:
#             bckgrnd_feats_dict[k].append(v)

# for k in bckgrnd_feats_dict:
#     bckgrnd_feats_dict[k] = tf.concat(bckgrnd_feats_dict[k], axis=0)

# input_order = [t.name.split(":")[0] for t in full_model.inputs]
# background_inputs = [bckgrnd_feats_dict[k].numpy() for k in input_order]

bckground_df = pd.DataFrame(bckgrnd_data_to_tbl)

# # get inputs for all other examples
# filter_fn = make_uid_filter_out_fn(planets_uid_lst)
#
# foreground_input_fn = InputFn(
#         file_paths=dataset_dir_fps,  # config['datasets_fps'][dataset],
#         batch_size=batch_size,
#         mode='EVAL',
#         label_map=config['label_map'],
#         features_set=features_set,
#         multiclass=config['config']['multi_class'],
#         feature_map=config['feature_map'],
#         label_field_name=config['label_field_name'],
#         filter_fn=filter_fn,  # partial(filter_by_uid, allowed_uids=planets_uid_lst),  # filter_planet_examples_tfrecord_label, # partial(filter_examples_tfrecord_label, label='KP'),
#     )
# foregrnd_feats_dict, foregrnd_data_to_tbl = {}, {field: [] for field in ['uid', 'label']}
foregrnd_data_to_tbl = {field: [] for field in ['uid', 'label']}

# for batch in foreground_input_fn():
    
#     batch_feats = batch[0] if isinstance(batch, tuple) else batch
    
#     for k, v in batch_feats.items():
#         if k in ['uid', 'label']:
#             foregrnd_data_to_tbl[k].extend([el.decode('utf') for el in np.array(v.numpy().tolist()).flatten()])
#             continue
#         if k not in foregrnd_feats_dict:
#             foregrnd_feats_dict[k] = [v]
#         else:
#             foregrnd_feats_dict[k].append(v)

# for k in foregrnd_feats_dict:
#     foregrnd_feats_dict[k] = tf.concat(foregrnd_feats_dict[k], axis=0)

# input_order = [t.name.split(":")[0] for t in full_model.inputs]
# foreground_inputs = [foregrnd_feats_dict[k].numpy() for k in input_order]

foreground_df = pd.DataFrame(foregrnd_data_to_tbl)

feature_cols = [col for col in full_df.columns if col not in ['uid', 'label']]
# # feature groups of 3 corresponding to each branch
# feature_grps = [feature_cols[col_i:col_i+3] for col_i in range(0, len(feature_cols), 3)]
# feature groups when including stellar params and DV TCE fit inputs
feature_grps = [feature_cols[col_i:col_i+3] for col_i in range(0, len(feature_cols) - len(stellar_params_feature_order) - len(dv_tce_fit_feature_order), 3)]
feature_grps += [stellar_params_feature_order, dv_tce_fit_feature_order]
background_inputs = [planets_df[feature_grp].to_numpy() for feature_grp in feature_grps]

foreground_df = full_df.loc[~full_df['uid'].isin(planets_df['uid'])]
foreground_inputs = [foreground_df[feature_grp].to_numpy() for feature_grp in feature_grps]

#%% Check mean score for background dataset

planets_uid_lst = planets_df['uid'].to_list()
filter_fn = make_uid_filter_fn(planets_uid_lst)
batch_size = 500
features_set = config['features_set']
features_set.update({'uid': {'dim': [1,], 'dtype': 'string'}})
# features_set['quality'] = {'dim': [5, 1], 'dtype': 'float'}

# custom_objects = {"Time2Vec": Time2Vec, 'SplitLayer': SplitLayer}
# with custom_object_scope(custom_objects):
#     full_model = load_model(filepath='/Users/msaragoc/Projects/exoplanet_transit_classification/exoplanet_dl/exominer_pipeline/data/exominer-plusplus_cv-iter0-model0_tess-spoc-2min-s1s67_tess-kepler.keras', compile=False)
    
predict_input_fn = InputFn(
        file_paths=dataset_dir_fps,  # config['datasets_fps'][dataset],
        batch_size=batch_size,
        mode='PREDICT',
        label_map=config['label_map'],
        features_set=features_set,
        multiclass=config['config']['multi_class'],
        feature_map=config['feature_map'],
        label_field_name=config['label_field_name'],
        filter_fn=filter_fn, 
    )

full_model.trainable = False
bckgrnd_preds_full_model = full_model.predict(predict_input_fn())
print(f'Background dataset mean score for full model: {bckgrnd_preds_full_model.mean():.4f}')

clfh_model.trainable = False
bckgrnd_preds_clfh_model = clfh_model(background_inputs, training=False)
print(f'Background dataset mean score for classification head model: {bckgrnd_preds_clfh_model.numpy().mean():.4f}')

model_background_mean_pred = pred_tbl.loc[pred_tbl['uid'].isin(planets_df['uid']), 'score'].mean()
print(f'Background dataset mean score for ensemble 0 model based on prediction tables: {model_background_mean_pred:.4f}')

#%% create explainer

# # select 500 planet examples for background dataset using dataframe
# full_df = pd.read_csv(experiment_dir / 'conv_features_dataset.csv', comment='#')
# planets_df = full_df[full_df['label'].isin(['CP', 'KP'])]
# planets_df = planets_df.sample(n=500, random_state=42, replace=False)

# feature_cols = [col for col in full_df.columns if col not in ['uid', 'label', 'base_value']]
# feature_grps = [feature_cols[col_i:col_i+3] for col_i in range(0, len(feature_cols), 3)]

# background_inputs = [planets_df[feature_grp].to_numpy() for feature_grp in feature_grps]

# foreground_df = full_df.loc[~full_df['uid'].isin(planets_df['uid'])]
# foreground_inputs = [foreground_df[feature_grp].to_numpy() for feature_grp in feature_grps]

model_fp = experiment_dir / 'clf-head-model-relu-output.keras'

model = load_model(model_fp, compile=False)
model.save(model_fp.parent / f'{model_fp.stem}', save_format="tf")

# Disable eager execution
tf.compat.v1.disable_eager_execution()

# Register custom gradient for Neg
@ops.RegisterGradient("shap_Neg")
def _shap_neg_grad(op, grad):
    return [-grad]

# Create a new graph and override gradients
with tf.Graph().as_default() as g:
    with g.gradient_override_map({"Neg": "shap_Neg"}):
        sess = tf.compat.v1.Session(graph=g)
        
        tf.compat.v1.keras.backend.set_session(sess)         
        tf.compat.v1.keras.backend.set_learning_phase(0)

        with sess.as_default():
            
            # Load model inside the session context
            print(f'Loading model {model_fp} inside session...')
            # model = load_model(filepath=model_fp, compile=False)
            # model = tf.keras.models.load_model(model_fp, compile=True)
            model = tf.keras.models.load_model(model_fp.parent / f'{model_fp.stem}', compile=False)
            
            # Initialize variables
            sess.run(tf.compat.v1.global_variables_initializer())
            
            # K.set_learning_phase(0)
            model.trainable = False
                        
            model_output_in_session = model(background_inputs).eval(session=sess)

            print("Mean model output in session:", model_output_in_session.mean())

            # Create DeepExplainer with the actual session
            print('Creating Explainer object...')
            deep_explainer = shap.DeepExplainer(model, background_inputs, session=sess)

            print("SHAP expected value:", deep_explainer.expected_value)

            # Compute SHAP values
            print('Computing SHAP values...')
            shap_values = deep_explainer(foreground_inputs)

#%% prepare shap values and features data for explanation objects from shap vlaues explanation object

n_features_in_grps = [shap_feat_grp.shape[1] for shap_feat_grp in shap_values]

shap_values_arr = [np.array(shap_feat_grp.values).squeeze() for shap_feat_grp in shap_values]
shap_values_arr = np.concatenate(shap_values_arr, axis=1)

feature_values_arr = [np.array(shap_feat_grp.data).squeeze() for shap_feat_grp in shap_values]
feature_values_arr = np.concatenate(feature_values_arr, axis=1)

# shap_values_arr = np.array(shap_values.values).squeeze()
# n_features_grps, n_examples, n_features_in_grp = shap_values_arr.shape

# shap_values_arr = shap_values_arr.transpose(1, 0, 2)
# shap_values_arr = shap_values_arr.reshape(n_examples, n_features_grps * n_features_in_grp)

# feature_values_arr = np.array(shap_values.data)
# feature_values_arr = feature_values_arr.transpose(1, 0, 2)
# feature_values_arr = feature_values_arr.reshape(n_examples, n_features_grps * n_features_in_grp)

#%% build SHAP values table

shap_df_fp = experiment_dir / f'shap-values_{model_fp.stem}_foreground-examples.csv'

shap_df = pd.DataFrame(shap_values_arr, columns=feature_cols)
shap_df = pd.concat([foreground_df[['uid', 'label']].reset_index(drop=True), shap_df.reset_index(drop=True)], axis=1)

shap_df['base_value'] = float(deep_explainer.expected_value)

# add metadata
shap_df.attrs['dataset'] = str(dataset_dir)
shap_df.attrs['model'] = f'convolutional model up to ReLU output plus stellar parameters and DV TCE fit inputs of {str(model_fp)}'  # f'full ExoMiner++ model in {str(model_fp)}'
shap_df.attrs['created'] = str(pd.Timestamp.now().floor('min'))
shap_df.attrs['shap_framework'] = 'deep explainer'
shap_df.attrs['base_value'] = float(deep_explainer.expected_value[0])
shap_df.attrs['reference_examples'] = '500 CP/KP (planet) TCEs'
shap_df.attrs['examples_set'] = 'examples not used for reference'
with open(shap_df_fp, "w") as f:
    for key, value in shap_df.attrs.items():
        f.write(f"# {key}: {value}\n")
    shap_df.to_csv(f, index=False)

#%% load SHAP values and features from CSV tables

shap_df_fp = experiment_dir / f'shap-values_{model_fp.stem}_foreground-examples.csv'
shap_df = pd.read_csv(shap_df_fp, comment='#')

feature_cols = [col for col in shap_df.columns if col not in ['uid', 'label', 'base_value']]

shap_values_arr = shap_df[feature_cols].to_numpy()

features_df_fp = experiment_dir / 'conv_features_dataset.csv'
features_df = pd.read_csv(features_df_fp, comment='#')
features_df = features_df[features_df['uid'].isin(shap_df['uid'])]
feature_values_arr = features_df[feature_cols].to_numpy()

base_value = shap_df['base_value'][0]

#%% aggregate SHAP values per branch


# def aggregate_values_by_branch(arr, n_features_in_grp, method='sum'):
#     """
#     Aggregates each set of 3 consecutive columns in a (N, 36) array.

#     Parameters:
#     - arr: np.ndarray of shape (n_examples, n_features)
#     - n_features_in_grp: int, number of features in each group
#     - method: str, one of ['sum', 'mean', 'max', 'l2']

#     Returns:
#     - aggregated_arr: np.ndarray of shape (N, 12)
#     """
    
#     if arr.shape[1] % n_features_in_grp != 0:
#         raise ValueError(f"Number of columns must be divisible by number of features in group ({n_features_in_grp}).")

#     n_grps = arr.shape[1] // n_features_in_grp
#     n_examples = arr.shape[0]
    
#     reshaped = arr.reshape(n_examples, n_grps, n_features_in_grp)  

#     if method == 'sum':
#         aggregated = reshaped.sum(axis=2)
#     elif method == 'median':
#         aggregated = reshaped.median(axis=2)
#     elif method == 'mean':
#         aggregated = reshaped.mean(axis=2)
#     elif method == 'max_abs':
#         # aggregated = reshaped.max(axis=2)
        
#        # get indices of max absolute values along the last axis
#         idx = np.abs(reshaped).argmax(axis=2)
        
#         # use advanced indexing to get the signed values at those indices
#         aggregated = np.take_along_axis(reshaped, idx[..., np.newaxis], axis=2).squeeze(axis=2)

#     elif method == 'l2':
#         aggregated = np.sqrt((reshaped ** 2).sum(axis=2))
#     else:
#         raise ValueError(f"Unknown method: {method}")

#     return aggregated


def aggregate_values_by_branch(arr, group_sizes, method='sum'):
    """
    Aggregates variable-sized groups of columns in a (N, M) array.

    Parameters:
    - arr: np.ndarray of shape (n_examples, n_features)
    - group_sizes: list of ints, sizes of each group (must sum to arr.shape[1])
    - method: str, one of ['sum', 'mean', 'max_abs', 'l2']

    Returns:
    - aggregated_arr: np.ndarray of shape (N, len(group_sizes))
    """
    if sum(group_sizes) != arr.shape[1]:
        raise ValueError("Sum of group sizes must match number of columns in arr.")

    aggregated_list = []
    start = 0

    for size in group_sizes:
        group = arr[:, start:start + size]

        if method == 'sum':
            aggregated = group.sum(axis=1)
        elif method == 'mean':
            aggregated = group.mean(axis=1)
        elif method == 'max_abs':
            idx = np.abs(group).argmax(axis=1)
            aggregated = group[np.arange(group.shape[0]), idx]
        elif method == 'l2':
            aggregated = np.sqrt((group ** 2).sum(axis=1))
        else:
            raise ValueError(f"Unknown method: {method}")

        aggregated_list.append(aggregated)
        start += size

    return np.stack(aggregated_list, axis=1)

agg_method = 'sum'

n_feats_agg = []
for n_feats_grp in n_features_in_grps:
    if n_feats_grp == 3:
        n_feats_agg.append(3)
    else:
        n_feats_agg += [1] * n_feats_grp

agg_shap_values_arr = aggregate_values_by_branch(shap_values_arr, group_sizes=n_feats_agg, method=agg_method)
agg_feature_values_arr = aggregate_values_by_branch(feature_values_arr, group_sizes=n_feats_agg, method=agg_method)

agg_feature_cols = [feature_cols[i].split('/')[0] for i in range(0, len(feature_cols) - len(stellar_params_feature_order) - len(dv_tce_fit_feature_order), 3)]
agg_feature_cols += stellar_params_feature_order + dv_tce_fit_feature_order

#%% plot SHAP values per feature

plot_dir = experiment_dir / 'shap_values'

shap_df_fp = experiment_dir / f'shap-values_{model_fp.stem}_foreground-examples.csv'

plot_dir.mkdir(parents=True, exist_ok=True)

shap_df = pd.read_csv(shap_df_fp, comment='#')

bins = np.linspace(-0.1, 0.1, 100)  # 'auto'  # np.logspace(-9, 4, 500)
hist_density = False

# for groups of 3 extacted features (corresponding to each branch)
# for col_i in range(0, len(feature_cols), 3):
for feat_grp in feature_grps:
# for col_i in range(0, len(feature_cols), 3):
    if '/' in feat_grp[0]:
        feat_grp_name = feat_grp[0].split("/")[0]
    elif 'tce_sdens_norm' in feat_grp:
        feat_grp_name = 'stellar_params'
    elif 'boot_fap_norm' in feat_grp:
        feat_grp_name = 'dv_tce_fit_params'
    else:
        raise ValueError(f'Unknown feature group: {feat_grp}')
        
    f, ax = plt.subplots(figsize=(8, 6))
    # for col in feature_cols[col_i:col_i+3]:
    for col_i in range(len(feat_grp)):
        col = feat_grp[col_i]
        if '/' in col:
            feature_name = col.split('/')[0]
        else:
            feature_name = col
        # ax = sns.histplot(full_df, x=col, bins=50, stat='density', common_norm=False, element='step', fill=False)
        ax.hist(shap_df[col], bins=bins, density=False, histtype='step', label=col)
    ax.set_title(f'Feature: {feat_grp_name}')
    ax.set_xlabel('SHAP Value')
    ax.set_ylabel('TCE Density' if hist_density else 'TCE Count')
    # ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    f.tight_layout()
    f.savefig(plot_dir / f'hist_shap_{feat_grp_name}.png')
    plt.close(f)

plot_dir = experiment_dir / f'shap_values_agg-{agg_method}'
plot_dir.mkdir(parents=True, exist_ok=True)

# per branch by summing SHAP values for features in same branch
for agg_feature_name_i, agg_feature_name in enumerate(agg_feature_cols):

    f, ax = plt.subplots(figsize=(8, 6))
    ax.hist(agg_shap_values_arr[:, agg_feature_name_i], bins=bins, density=False, histtype='step', label=col)
    ax.set_title(f'Branch: {agg_feature_name} (SHAP values aggregated by {agg_method})')
    ax.set_xlabel('SHAP Value')
    ax.set_ylabel('TCE Density' if hist_density else 'TCE Count')
    # ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    f.tight_layout()
    f.savefig(plot_dir / f'hist_shap_{agg_feature_name}_agg-method-{agg_method}.png')
    plt.close(f)

#%% create explanation objects for individual features or aggregated by branch

# shap_df_fp = experiment_dir / 'shap-values_clf-head-model-relu-output_foreground-examples.csv'
# shap_df = pd.read_csv(shap_df_fp, comment='#')

# feature_cols = [col for col in shap_df.columns if col not in ['uid', 'label', 'base_value']]    
# shap_values_arr = shap_df[feature_cols].to_numpy()
    
# # aggregate SHAP values per branch
# agg_feature_cols = [feature_cols[i] for i in range(0, len(feature_cols), 3)]
# agg_shap_values_arr = shap_values_arr.reshape(shap_values_arr.shape[0],12, 3).sum(axis=2)
# foreground_arr = foreground_df[feature_cols].to_numpy()
# agg_foreground_arr = foreground_arr.reshape(foreground_arr.shape[0], 12, 3).sum(axis=2)

explainer_output = shap.Explanation(
    values=shap_values_arr,  # shap_values_reshaped,
    data=feature_values_arr,
    feature_names=feature_cols,
    base_values=base_value  # scalar or array of base values
)
agg_explainer_output = shap.Explanation(
    values=agg_shap_values_arr,  # shap_values_arr,
    data=agg_feature_values_arr,  # agg_foreground_arr,  # foreground_arr,  
    feature_names=agg_feature_cols,  # feature_cols
    base_values=base_value,
)

#%% global plots: bar, beeswarm, violin

plot_dir = experiment_dir / f'shap_global_plots_{model_fp.stem}_agg-{agg_method}'
plot_dir.mkdir(parents=True, exist_ok=True)

max_display = len(agg_feature_cols)

f, ax = plt.subplots(figsize=(10, 12))
shap.plots.bar(agg_explainer_output, max_display=max_display, ax=ax, show=False)
# plt.xscale('log')
ax.set_xscale('linear')
# ax.set_xlim([-0.5, 0.5])
f.tight_layout()
f.savefig(plot_dir / 'shap_global_bar_plot.png')
plt.close()

f, ax = plt.subplots(figsize=(10, 12))
shap.plots.beeswarm(agg_explainer_output, max_display=max_display, ax=ax, show=False, plot_size=None, log_scale=True)
ax.set_xlim([-0.5, 0.5])
f.tight_layout()
f.savefig(plot_dir / 'shap_global_beeswarm_plot.png')

f, ax = plt.subplots(figsize=(10, 12))
shap.plots.violin(agg_explainer_output, max_display=max_display, show=False, plot_size=None)
f.tight_layout()
f.savefig(plot_dir / 'shap_global_violin_plot.png')

#%% scatter plots for each feature

plot_dir = experiment_dir / f'shap_scatter_plots_{model_fp.stem}'

plot_dir.mkdir(parents=True, exist_ok=True)

idx_start_feat = 0
for agg_feature_col, n_feats in zip(agg_feature_cols, n_feats_agg):
        
    fig, ax = plt.subplots(n_feats, 1, figsize=(8, 12))
    
    if n_feats == 1:
        ax = [ax]
    
    for feature_grp_i in range(n_feats):
    
        explainer_output_feature = explainer_output[:, idx_start_feat + feature_grp_i]
    
        shap.plots.scatter(explainer_output_feature, 
                        xmin=explainer_output_feature.percentile(1), xmax=explainer_output_feature.percentile(99), 
                        ymin=explainer_output_feature.percentile(1), ymax=explainer_output_feature.percentile(99),
                        alpha=0.2, dot_size=5, 
                        ax=ax[feature_grp_i], show=False)
        ax[feature_grp_i].set_ylabel('SHAP Value')

    fig.suptitle(f'SHAP Scatter Plot: {agg_feature_col}')
    
    fig.tight_layout()
    plt.show()

    fig.savefig(plot_dir / f"shap_scatter_{agg_feature_col}.png")
    plt.close(fig)
    
    idx_start_feat += n_feats

#%% global plots per group such as label

chosen_grp = 'label'
grp_values = shap_df['label'].unique()
max_display = len(agg_feature_cols)

plot_dir = experiment_dir / f'shap_global_plots_{model_fp.stem}_agg-{agg_method}_{chosen_grp}'
plot_dir.mkdir(parents=True, exist_ok=True)

# loop over each label
for grp_val in grp_values:
    
    print(f'Generating SHAP plots for {chosen_grp}: {grp_val}')
    
    # get indices for this label
    idxs_examples_in_grp = shap_df[shap_df[chosen_grp] == grp_val].index.to_numpy()
    
    # subset the Explanation object
    agg_explainer_output_grp = shap.Explanation(
        values=agg_shap_values_arr[idxs_examples_in_grp],  # shap_values_arr,
        data=agg_feature_values_arr[idxs_examples_in_grp],  # agg_foreground_arr,  # foreground_arr,  
        feature_names=agg_feature_cols,  # feature_cols
        base_values=base_value,
    )
    
    f, ax = plt.subplots(figsize=(10, 12))
    shap.plots.bar(agg_explainer_output_grp, max_display=max_display, ax=ax, show=False)
    # plt.xscale('log')
    ax.set_xscale('linear')
    # ax.set_xlim([-0.5, 0.5])
    f.tight_layout()
    f.savefig(plot_dir / f'shap_global_bar_plot_{chosen_grp}-{grp_val}.png')
    plt.close()

    f, ax = plt.subplots(figsize=(10, 12))
    shap.plots.beeswarm(agg_explainer_output_grp, max_display=max_display, ax=ax, show=False, plot_size=None, log_scale=True)
    # ax.set_xlim([-0.5, 0.5])
    f.tight_layout()
    f.savefig(plot_dir / f'shap_global_beeswarm_plot_{chosen_grp}-{grp_val}.png')
    plt.close()
    
    f, ax = plt.subplots(figsize=(10, 12))
    shap.plots.violin(agg_explainer_output_grp, max_display=max_display, show=False, plot_size=None)
    f.tight_layout()
    f.savefig(plot_dir / f'shap_global_violin_plot_{chosen_grp}-{grp_val}.png')
    plt.close()


#%% local waterfall plot

agg = True
download_dv_reports = False
sample_name = '_sample-top100_nebs_by_dikco_msky'
# sample_name = '_sample-few-examples'

tce_tbl = pd.read_csv('/Users/msaragoc/Projects/exoplanet_transit_classification/data/ephemeris_tables/tess/tess_spoc_2min/DV_SPOC_mat_files/preprocessing_tce_tables/09-25-2023_1608/tess_2min_tces_dv_s1-s68_all_msectors_11-29-2023_2157_newlabels_nebs_npcs_bds_ebsntps_to_unks_sg1master_allephemmatches_exofoptois_dv_mast_urls.csv')
nebs = tce_tbl.loc[tce_tbl['label'] == 'NEB'].sort_values('tce_dikco_msky', ascending=False)
tce_uid_lst = nebs['uid'].to_list()[:100]

# tce_uid_lst = [
#     '26801525-1-S57', # secondary | EB
#     '309787037-1-S35',  # centroid | FP
#     '167526485-1-S6',  # flux trend ? | EB
#     '273574141-1-S14',  # odd-even | EB
#     '83053699-1-S57', # centroid motion | NEB   
# ]


if agg:
    plot_dir = experiment_dir / f'shap_waterfall_plots_{model_fp.stem}_agg-{agg_method}{sample_name}'
else:
    plot_dir = experiment_dir / f'shap_waterfall_plots_{model_fp.stem}{sample_name}'

plot_dir.mkdir(parents=True, exist_ok=True)

if download_dv_reports:
    tce_uid_lst_corr = [correct_sector_field(tce_uid) for tce_uid in tce_uid_lst if tce_uid in shap_df['uid'].values]
    download_dvr_dir = plot_dir / 'downloaded_dv_reports'
    download_dvr_dir.mkdir(parents=True, exist_ok=True)
    spoc_ffi = False
    data_products_lst = ['DV mini-report']  # ['DV TCE summary report', 'Full DV report', 'DV mini-report']
    print('Downloading DV reports for TCE UIDs..')
    get_dv_dataproducts_list(tce_uid_lst_corr, download_dir=download_dvr_dir, data_products_lst=data_products_lst, reports='dv_mini_report', download_products=True, verbose=True, get_most_recent_products=True, spoc_ffi=spoc_ffi)
    print('Finished downloading DV reports for TCE UIDs.')

for tce_uid in tqdm(tce_uid_lst):
    
    if tce_uid not in shap_df['uid'].values:
        raise ValueError(f'TCE UID {tce_uid} not found in foreground dataset.')
    
    idx_example_tce = shap_df[shap_df['uid'] == tce_uid].index.to_numpy()[0]

    if agg:
        explainer_output_tce = shap.Explanation(
            values=agg_shap_values_arr[idx_example_tce],  
            data=agg_feature_values_arr[idx_example_tce],  
            feature_names=agg_feature_cols, 
            base_values=base_value,
        )
    else:
        explainer_output_tce = shap.Explanation(
            values=shap_values_arr[idx_example_tce],  
            data=feature_values_arr[idx_example_tce],  
            feature_names=feature_cols, 
            base_values=base_value,
        )

    shap.plots.waterfall(explainer_output_tce, max_display=36, show=False)
    plt.savefig(plot_dir / f'shap_waterfall_{tce_uid}.png', bbox_inches='tight')
    plt.close()

#%%

# shap.plots.force(
#     deep_explainer.expected_value[0],
#     agg_shap_values_arr,
#     agg_feature_values_arr,
#     agg_feature_cols
#     )
# %%
