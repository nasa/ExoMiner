"""Test Deep SHAP Explainer for ExoMiner."""

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

# local
from models.models_keras import Time2Vec, SplitLayer, ExoMinerPlusPlus
from src.utils.utils_dataio import InputFnv2 as InputFn
# from src.train.utils_train import filter_examples_tfrecord_label
from src.utils.utils_dataio import get_data_from_tfrecords_for_predictions_table

#%% set filepaths

experiment_dir = Path('/Users/msaragoc/Projects/exoplanet_transit_classification/experiments/xai/test_deep_shap')
model_fp = Path('/Users/msaragoc/Projects/exoplanet_transit_classification/exoplanet_dl/exominer_pipeline/data/exominer-plusplus_cv-iter0-model0_tess-spoc-2min-s1s67_tess-kepler.keras')
config_fp = Path('/Users/msaragoc/Projects/exoplanet_transit_classification/exoplanet_dl/exominer_pipeline/config_predict_model.yaml')
dataset_dir = Path('/Users/msaragoc/Projects/exoplanet_transit_classification/data/tfrecords/tess/cv_tfrecords_tess_spoc_2min_s1-s67_9-24-2024_1159_tcedikcorat_crowdsap_tcedikcocorr_11-23-2024_0047_eval_normalized_cv_iter_0/norm_data')

# load config filepath
with open(config_fp, 'r') as config_yaml:
    config = yaml.unsafe_load(config_yaml)

#%% set dataset filepaths

dataset_dir_fps = [str(fp) for fp in dataset_dir.glob('shard-*')]

#%% load model

print('Loading model...')

# exominer_config_fp = '/Users/msaragoc/Projects/exoplanet_transit_classification/exoplanet_dl/models/exominer_plusplus.yaml'
# with open(exominer_config_fp, 'r') as config_yaml:
#     exominer_config = yaml.unsafe_load(config_yaml)
# config.update(exominer_config)

custom_objects = {"Time2Vec": Time2Vec, 'SplitLayer': SplitLayer}
with custom_object_scope(custom_objects):
    model = load_model(filepath=model_fp, compile=False)
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
    'fc_prelu_stellar_scalar',
    'fc_prelu_dv_tce_fit_scalar',
]
input_clfh_model = [model_layer.output for model_layer in model.layers if model_layer.name in input_clfh_layers_names]

clfh_model = tf.keras.Model(inputs=input_clfh_model, outputs=model.output)
conv_model = tf.keras.Model(inputs=model.inputs, outputs=input_clfh_model)

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

model.save(experiment_dir / 'full-model.keras')

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

feature_cols = [[f'{l.name}_{feat_i}' for feat_i in range(l.shape[1])] for l in conv_model.outputs]
feature_cols = [item for sublist in feature_cols for item in sublist]

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
full_df.attrs['model'] = f'convolutional model up to ReLU output of {str(model_fp)}'
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

feature_cols = [col for col in full_df.columns if col not in ['uid', 'label']]
for col_i in range(0, len(feature_cols), 3):
    
    # f = plt.figure(figsize=(8, 6))
    f, ax = plt.subplots(figsize=(8, 6))
    for col in feature_cols[col_i:col_i+3]:
        feature_name = col.split('/')[0]
        # ax = sns.histplot(full_df, x=col, bins=50, stat='density', common_norm=False, element='step', fill=False)
        ax.hist(full_df[col], bins=bins, density=False, histtype='step', label=col)
    ax.set_title(f'Feature: {feature_name}')
    ax.set_xlabel('Feature Value')
    ax.set_ylabel('TCE Density' if hist_density else 'TCE Count')
    ax.set_xscale('log')
    ax.legend()
    f.tight_layout()
    f.savefig(plot_dir / f'hist_feature_{feature_name}.png')
    plt.close(f)
    
#%% create background dataset for SHAP

datasets_fps = {
    'train': dataset_dir_fps[:-1],
    'test': dataset_dir_fps[-1:],
}
config['datasets_fps'] = datasets_fps
config['label_map'] = {
    'KP': 1,
    'CP': 1,
    'EB': 0,
    'FP': 0,
    'NTP': 0,
    # 'UNK': 0,
    'NEB': 0,
    'NPC': 0,
    'BD': 0,
}

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

full_df = pd.read_csv(experiment_dir / 'conv_features_dataset.csv', comment='#')
planets_df = full_df[full_df['label'].isin(['CP', 'KP'])]
planets_df = planets_df.sample(n=500, random_state=42, replace=False)
planets_uid_lst = planets_df['uid'].to_list()

config['features_set']['quality'] = {'dim': [5, 1], 'dtype': 'float'}

filter_fn = make_uid_filter_fn(planets_uid_lst)

dataset = 'train'
n_bckgrnd_batches = 5
n_bckgrnd_samples = 500
batch_size = n_bckgrnd_samples // n_bckgrnd_batches
features_set = config['features_set']
features_set.update({'uid': {'dim': [1,], 'dtype': 'string'}, 'label': {'dim': [1,], 'dtype': 'string'}})

background_input_fn = InputFn(
        file_paths=dataset_dir_fps,  # config['datasets_fps'][dataset],
        batch_size=batch_size,
        mode='EVAL',
        label_map=config['label_map'],
        features_set=features_set,
        multiclass=config['config']['multi_class'],
        feature_map=config['feature_map'],
        label_field_name=config['label_field_name'],
        filter_fn=filter_fn,  # partial(filter_by_uid, allowed_uids=planets_uid_lst),  # filter_planet_examples_tfrecord_label, # partial(filter_examples_tfrecord_label, label='KP'),
    )
bckgrnd_feats_dict = {}
for batch_i in range(n_bckgrnd_batches):
    
    batch = next(iter(background_input_fn()))
    batch_feats = batch[0] if isinstance(batch, tuple) else batch
    
    for k, v in batch_feats.items():
        if k in ['uid', 'label']:
            continue
        if k not in bckgrnd_feats_dict:
            bckgrnd_feats_dict[k] = [v]
        else:
            bckgrnd_feats_dict[k].append(v)


for k in bckgrnd_feats_dict:
    bckgrnd_feats_dict[k] = tf.concat(bckgrnd_feats_dict[k], axis=0)

input_order       = [t.name.split(":")[0] for t in model.inputs]
background_inputs = [bckgrnd_feats_dict[k].numpy() for k in input_order]

# get inputs for all other examples
filter_fn = make_uid_filter_out_fn(planets_uid_lst)

foreground_input_fn = InputFn(
        file_paths=dataset_dir_fps,  # config['datasets_fps'][dataset],
        batch_size=batch_size,
        mode='EVAL',
        label_map=config['label_map'],
        features_set=features_set,
        multiclass=config['config']['multi_class'],
        feature_map=config['feature_map'],
        label_field_name=config['label_field_name'],
        filter_fn=filter_fn,  # partial(filter_by_uid, allowed_uids=planets_uid_lst),  # filter_planet_examples_tfrecord_label, # partial(filter_examples_tfrecord_label, label='KP'),
    )
foregrnd_feats_dict, foregrnd_data_to_tbl = {}, {field: [] for field in ['uid', 'label']}
# for batch_i in range(n_bckgrnd_batches):
    # batch = next(iter(background_input_fn()))
for batch in foreground_input_fn():
    
    batch_feats = batch[0] if isinstance(batch, tuple) else batch
    
    for k, v in batch_feats.items():
        if k in ['uid', 'label']:
            foregrnd_data_to_tbl[k].extend(v.numpy().tolist())
            continue
        if k not in foregrnd_feats_dict:
            foregrnd_feats_dict[k] = [v]
        else:
            foregrnd_feats_dict[k].append(v)

for k in foregrnd_feats_dict:
    foregrnd_feats_dict[k] = tf.concat(foregrnd_feats_dict[k], axis=0)

input_order       = [t.name.split(":")[0] for t in model.inputs]
foreground_inputs = [foregrnd_feats_dict[k].numpy() for k in input_order]

foreground_df = pd.DataFrame(foregrnd_data_to_tbl)

# # select 500 planet examples for background dataset using dataframe
# full_df = pd.read_csv(experiment_dir / 'conv_features_dataset.csv', comment='#')
# planets_df = full_df[full_df['label'].isin(['CP', 'KP'])]
# planets_df = planets_df.sample(n=500, random_state=42, replace=False)

# feature_cols = [col for col in full_df.columns if col not in ['uid', 'label']]
# feature_grps = [feature_cols[col_i:col_i+3] for col_i in range(0, len(feature_cols), 3)]
# background_inputs = [planets_df[feature_grp].to_numpy() for feature_grp in feature_grps]

# foreground_df = full_df.loc[~full_df['uid'].isin(planets_df['uid'])]
# foreground_inputs = [foreground_df[feature_grp].to_numpy() for feature_grp in feature_grps]

#%% create explainer

# explainer = shap.GradientExplainer(model, background_inputs)
# deep_explainer = shap.DeepExplainer(model, background_inputs)
# grad_explainer = shap.GradientExplainer(model, background_inputs)

# def model_predict(inputs):
#     return model(inputs).numpy()
# kernel_explainer = shap.KernelExplainer(model_predict, background_inputs)

# # Create a new graph and override gradients
# with tf.Graph().as_default() as g:
#     with g.gradient_override_map({"Neg": "shap_Neg"}):
#         # Build or load your model here
#         # Run SHAP explainer inside this context

#         clfh_model = load_model(filepath=experiment_dir / 'clf-head-model-relu-output.keras', compile=False)

#         deep_explainer = shap.DeepExplainer(clfh_model, background_inputs, session='tensorflow')
#         # grad_explainer = shap.GradientExplainer(clfh_model, background_inputs, session='tensorflow')
        
#         shap_values = deep_explainer.shap_values(examples_left_inputs)

model_fp = experiment_dir / 'full-model.keras'

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

        with sess.as_default():
            
            # Load model inside the session context
            model = load_model(filepath=model_fp, compile=False)
            
            # Initialize variables
            sess.run(tf.compat.v1.global_variables_initializer())
            
            # Create DeepExplainer with the actual session
            print('Creating Explainer object...')
            deep_explainer = shap.DeepExplainer(model, background_inputs, session=sess)

            # Compute SHAP values
            print('Computing SHAP values...')
            # shap_values = deep_explainer.shap_values(examples_left_inputs)
            shap_values = deep_explainer(foreground_inputs)

#%% prepare SHAP values table

shap_df_fp = experiment_dir / 'shap_values_examples-not-reference_full-model.csv'

shap_values_arr = np.array(shap_values.values)
n_features_grps, n_examples, n_features_in_grp, extra_dim = shap_values_arr.shape
shap_values_arr = shap_values_arr.transpose(1, 0, 2, 3)
shap_values_arr = shap_values_arr.reshape(n_examples, n_features_grps * n_features_in_grp)

shap_df = pd.DataFrame(shap_values_arr, columns=feature_cols)
shap_df = pd.concat([foreground_df[['uid', 'label']], shap_df], axis=1)

# add metadata
shap_df.attrs['dataset'] = str(dataset_dir)
shap_df.attrs['model'] = f'full ExoMiner++ model in {str(model_fp)}'
shap_df.attrs['created'] = str(pd.Timestamp.now().floor('min'))
shap_df.attrs['shap_framework'] = 'deep explainer'
shap_df.attrs['reference_examples'] = '500 CP/KP (planet) TCEs'
shap_df.attrs['examples_set'] = 'examples not used for reference'
with open(shap_df_fp, "w") as f:
    for key, value in shap_df.attrs.items():
        f.write(f"# {key}: {value}\n")
    shap_df.to_csv(f, index=False)
    
#%% plot SHAP values

plot_dir = experiment_dir / 'shap_values'
shap_df_fp = experiment_dir / 'shap_values_examples-not-reference_full-model.csv'

plot_dir.mkdir(parents=True, exist_ok=True)

shap_df = pd.read_csv(shap_df_fp, comment='#')

bins = np.linspace(-0.1, 0.1, 100)  # 'auto'  # np.logspace(-9, 4, 500)
hist_density = False

# for groups of 3 extacted features (corresponding to each branch)
feature_cols = [col for col in shap_df.columns if col not in ['uid', 'label']]
for col_i in range(0, len(feature_cols), 3):
    
    f, ax = plt.subplots(figsize=(8, 6))
    for col in feature_cols[col_i:col_i+3]:
        feature_name = col.split('/')[0]
        # ax = sns.histplot(full_df, x=col, bins=50, stat='density', common_norm=False, element='step', fill=False)
        ax.hist(shap_df[col], bins=bins, density=False, histtype='step', label=col)
    ax.set_title(f'Feature: {feature_name}')
    ax.set_xlabel('SHAP Value')
    ax.set_ylabel('TCE Density' if hist_density else 'TCE Count')
    # ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    f.tight_layout()
    f.savefig(plot_dir / f'hist_feature_{feature_name}.png')
    plt.close(f)

# per branch by summing SHAP values for features in same branch
map_branch_to_features = {
    'Full-Orbit': ['fc_prelu_global_flux'],
    'Centroid Motion': [],
    'Momentum Dump': ['fc_prelu_momentum_dump'],
    'Periodogram': ['fc_prelu_flux_periodogram'],
    'Flux Trend': ['fc_prelu_flux_trend'],
    'Diff Image': ['fc_prelu_diff_img'],
}
for branch_name, branch_features in map_branch_to_features.items():
    
    shap_branch = shap_df[branch_features].sum(axis=1)

    f, ax = plt.subplots(figsize=(8, 6))
    ax.hist(shap_branch[col], bins=bins, density=False, histtype='step', label=col)
    ax.set_title(f'Branch: {branch_name}')
    ax.set_xlabel('SHAP Value')
    ax.set_ylabel('TCE Density' if hist_density else 'TCE Count')
    # ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    f.tight_layout()
    f.savefig(plot_dir / f'hist_shap_{branch_name}.png')
    plt.close(f)

#%% create SHAP built-in plots

shap_df_fp = experiment_dir / 'shap_values_examples-not-reference_full-model.csv'
shap_df = pd.read_csv(shap_df_fp, comment='#')

shap_values_arr = shap_df[feature_cols].to_numpy()
feature_cols = [col for col in shap_df.columns if col not in ['uid', 'label']]    
    
# aggregate SHAP values per branch
agg_feature_cols = [feature_cols[i] for i in range(0, len(feature_cols), 3)]
agg_shap_values_arr = shap_values_arr.reshape(shap_values_arr.shape[0],12, 3).sum(axis=2)
foreground_arr = foreground_df[feature_cols].to_numpy()
# agg_examples_left_arr = examples_left_arr.reshape(examples_left_arr.shape[0], 12, 3).sum(axis=2)

explainer_output = shap.Explanation(
    values=agg_shap_values_arr,  # shap_values_arr,
    data=foreground_inputs,  # foreground_arr,  
    feature_names=agg_feature_cols,  # feature_cols
)

plot_dir = experiment_dir / 'shap_plots_agg'
plot_dir.mkdir(parents=True, exist_ok=True)

#%% global bar plot and beeswarm

f, ax = plt.subplots(figsize=(10, 12))
shap.plots.bar(explainer_output, max_display=36, ax=ax, show=False)
# plt.xscale('log')
ax.set_xscale('linear')
# ax.set_xlim([-0.5, 0.5])
f.tight_layout()
f.savefig(plot_dir / 'shap_global_bar_plot.png')
plt.close()

f, ax = plt.subplots(figsize=(10, 12))
shap.plots.beeswarm(explainer_output, max_display=36, ax=ax, show=False, plot_size=None, log_scale=True)
ax.set_xlim([-0.5, 0.5])
f.tight_layout()
f.savefig(plot_dir / 'shap_global_beeswarm_plot.png')

#%% analyze SHAP values per label

# loop over each label
for label in foreground_df['label'].unique():
    
    # get indices for this label
    idxs_label = foreground_df['label'] == label
    
    foreground_df_label = foreground_df.loc[foreground_df['label'] == label]
    foreground_df_label_arr = foreground_df_label[feature_cols].to_numpy()
    agg_foreground_df_label_arr = foreground_df_label_arr.reshape(foreground_df_label_arr.shape[0],12, 3).sum(axis=2)
    
    shap_df_label = shap_df.loc[shap_df['label'] == label]
    shap_arr_label = shap_df_label[feature_cols].to_numpy()
    agg_shap_arr_label = shap_arr_label.reshape(shap_arr_label.shape[0],12, 3).sum(axis=2)

    # subset the Explanation object
    explainer_output = shap.Explanation(
        values=agg_shap_arr_label,
        data=foreground_df_label_arr,
        feature_names=agg_feature_cols,
    )
    
    # # subset the Explanation object
    # subset_expl = shap.Explanation(
    #     values=explainer_output.values[idx],
    #     data=explainer_output.data[idx],
    #     feature_names=explainer_output.feature_names
    # )

    # # plot
    # print(f"SHAP bar plot for label: {label}")
    # shap.plots.bar(subset_expl, max_display=36)
    
    f, ax = plt.subplots(figsize=(10, 12))
    shap.plots.bar(explainer_output, max_display=36, ax=ax, show=False)
    # plt.xscale('log')
    ax.set_xscale('linear')
    # ax.set_xlim([-0.5, 0.5])
    f.tight_layout()
    f.savefig(plot_dir / f'shap_global_bar_plot_{label}.png')
    plt.close()

    f, ax = plt.subplots(figsize=(10, 12))
    shap.plots.beeswarm(explainer_output, max_display=36, ax=ax, show=False, plot_size=None, log_scale=True)
    ax.set_xlim([-0.5, 0.5])
    f.tight_layout()
    f.savefig(plot_dir / f'shap_global_beeswarm_plot_{label}.png')
    plt.close()


#%% local bar plot

foreground_df.reset_index(drop=True, inplace=True)
example_idx = foreground_df.loc[foreground_df['uid'] == '26801525-1-S57'].index.values[0]

# expl = shap.Explanation(
#     values=shap_values[0],  # or shap_values if single output
#     data=examples_left_inputs,
#     feature_names=[f"feature_{i}" for i in range(examples_left_inputs.shape[1])]
# )

shap.plots.bar(explainer_output[example_idx], max_display=36)

#%%

# Extract values and feature names
shap_vals = explainer_output.values  # shape: (n_samples, n_features)
feature_names = explainer_output.feature_names

# Group features in triplets
grouped_names = [feature_names[i].rsplit('_', 1)[0] for i in range(0, len(feature_names), 3)]

# Sum absolute SHAP values across each group
grouped_shap_vals = []
for i in range(0, len(feature_names), 3):
    group_abs_sum = np.abs(shap_vals[:, i:i+3]).sum(axis=1)  # sum across the 3 features
    grouped_shap_vals.append(group_abs_sum)

# Stack into (n_samples, n_groups)
grouped_shap_vals = np.vstack(grouped_shap_vals).T

# Create new Explanation object
grouped_expl = shap.Explanation(
    values=grouped_shap_vals,
    data=None,  # optional
    feature_names=grouped_names
)

# Plot
shap.plots.bar(grouped_expl, max_display=36)


#%% explain a few examples

# def filter_fp_examples_tfrecord_label(parsed_features, label_id):
#     """ Filters out examples whose label_id is `0`.

#     Args:
#         parsed_features: tf tensor, parsed features for example
#         label_id: tf tensor, label id for example

#     Returns: tf boolean tensor

#     """

#     return tf.squeeze(label_id== 0)

# dataset = 'test'
# explain_input_fn = InputFn(
#         file_paths=config['datasets_fps'][dataset],
#         batch_size=config['eval']['batch_size'],
#         mode='EVAL',
#         label_map=config['label_map'],
#         features_set=config['features_set'],
#         multiclass=config['config']['multi_class'],
#         feature_map=config['feature_map'],
#         label_field_name=config['label_field_name'],
#         filter_fn=filter_fp_examples_tfrecord_label,
#     )

# test_example = [feat_data[0] for feat_data in background_inputs]
# # explanation = deep_explainer.explain_row()
# for batch in explain_input_fn():
    
#     batch_feats = batch[0] if isinstance(batch, tuple) else batch
#     test_example = [batch_feats[k][0:1].numpy() for k in input_order]
#     # shap_values = deep_explainer.shap_values(test_example)
#     # shap_values = grad_explainer.shap_values(test_example)
#     shap_values = kernel_explainer.shap_values(test_example, nsamples=100)
#     break

# shap_values = deep_explainer.shap_values(examples_left_inputs)
# shap_values = grad_explainer.shap_values(examples_left_inputs)

