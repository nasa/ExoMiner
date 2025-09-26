"""Test Deep SHAP Explainer for ExoMiner."""

# 3rd party
import shap
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import custom_object_scope, plot_model
import numpy as np
import  yaml
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
aaa
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

# def filter_planet_examples_tfrecord_label(parsed_features, label_id):
#     """ Filters out examples whose label_id is `1`.

#     Args:
#         parsed_features: tf tensor, parsed features for example
#         label_id: tf tensor, label id for example

#     Returns: tf boolean tensor

#     """

#     return tf.squeeze(label_id== 1)

# config['eval'] = {'batch_size': 16}
# dataset = 'train'
# n_bckgrnd_batches = 5
# n_bckgrnd_samples = n_bckgrnd_batches * config['inference']['batch_size']

# background_input_fn = InputFn(
#         file_paths=config['datasets_fps'][dataset],
#         batch_size=config['eval']['batch_size'],
#         mode='EVAL',
#         label_map=config['label_map'],
#         features_set=config['features_set'],
#         multiclass=config['config']['multi_class'],
#         feature_map=config['feature_map'],
#         label_field_name=config['label_field_name'],
#         filter_fn=filter_planet_examples_tfrecord_label, # partial(filter_examples_tfrecord_label, label='KP'),
#     )
# bckgrnd_feats_dict = {}
# for batch_i in range(n_bckgrnd_batches):
    
#     batch = next(iter(background_input_fn()))
#     batch_feats = batch[0] if isinstance(batch, tuple) else batch
    
#     for k, v in batch_feats.items():
#         if k not in bckgrnd_feats_dict:
#             bckgrnd_feats_dict[k] = [v]
#         else:
#             bckgrnd_feats_dict[k].append(v)


# for k in bckgrnd_feats_dict:
#     bckgrnd_feats_dict[k] = tf.concat(bckgrnd_feats_dict[k], axis=0)

# # first_batch = next(iter(predict_input_fn()))
# # feats_dict  = first_batch[0] if isinstance(first_batch, tuple) else first_batch
# # batch_n     = next(iter(feats_dict.values())).shape[0]

# # bg_idx = tf.constant(
# #     np.random.choice(config['inference']['batch_size'], 
# #                      min(MIN_REF, config['inference']['batch_size']), 
# #                      replace=False), dtype=tf.int32
# # )
# # background = {k: tf.gather(v, bg_idx, axis=0) for k, v in feats_dict.items()}

# # input_order       = [t.name.split(":")[0] for t in model.inputs]
# # background_inputs = [background[k].numpy() for k in input_order]

# # background_inputs = [feat_examples.numpy() for _, feat_examples in bckgrnd_feats_dict.items()]
# input_order       = [t.name.split(":")[0] for t in model.inputs]
# background_inputs = [bckgrnd_feats_dict[k].numpy() for k in input_order]

full_df = pd.read_csv(experiment_dir / 'conv_features_dataset.csv', comment='#')

planets_df = full_df[full_df['label'].isin(['CP', 'KP'])]
planets_df = planets_df.sample(n=500, random_state=42, replace=False)

feature_cols = [col for col in full_df.columns if col not in ['uid', 'label']]
feature_grps = [feature_cols[col_i:col_i+3] for col_i in range(0, len(feature_cols), 3)]
background_inputs = [planets_df[feature_grp].to_numpy() for feature_grp in feature_grps]

examples_left_df = full_df.loc[~full_df['uid'].isin(planets_df['uid'])]
examples_left_inputs = [examples_left_df[feature_grp].to_numpy() for feature_grp in feature_grps]

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
            clfh_model = load_model(filepath=experiment_dir / 'clf-head-model-relu-output.keras', compile=False)
            
            # Initialize variables
            sess.run(tf.compat.v1.global_variables_initializer())
            
            # Create DeepExplainer with the actual session
            deep_explainer = shap.DeepExplainer(clfh_model, background_inputs, session=sess)

            # Compute SHAP values
            # shap_values = deep_explainer.shap_values(examples_left_inputs)
            shap_values = deep_explainer(examples_left_inputs)

#%% prepare SHAP values table

shap_values_arr = np.array(shap_values.values)
n_features_grps, n_examples, n_features_in_grp, extra_dim = shap_values_arr.shape
shap_values_arr = shap_values_arr.transpose(1, 0, 2, 3)
shap_values_arr = shap_values_arr.reshape(n_examples, n_features_grps * n_features_in_grp)

shap_df = pd.DataFrame(shap_values_arr, columns=feature_cols)
shap_df = pd.concat([examples_left_df[['uid', 'label']], shap_df], axis=1)

# add metadata
shap_df.attrs['dataset'] = str(dataset_dir)
shap_df.attrs['model'] = f'classification head model from ReLU output of {str(model_fp)}'
shap_df.attrs['created'] = str(pd.Timestamp.now().floor('min'))
shap_df.attrs['shap_framework'] = 'deep-explainer'
shap_df.attrs['reference_examples'] = '500 CP/KP (planet) TCEs'
shap_df.attrs['examples_set'] = 'examples not used for reference'
with open(experiment_dir / 'shap_values_examples-not-reference.csv', "w") as f:
    for key, value in shap_df.attrs.items():
        f.write(f"# {key}: {value}\n")
    shap_df.to_csv(f, index=False)
    
#%% plot SHAP values

plot_dir = experiment_dir / 'shap_values'
plot_dir.mkdir(parents=True, exist_ok=True)

shap_df = pd.read_csv(experiment_dir / 'shap_values_examples-not-reference.csv', comment='#')

bins = np.linspace(-0.1, 0.1, 100)  # 'auto'  # np.logspace(-9, 4, 500)
hist_density = False

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

#%% create SHAP built-in plots

explainer_output = shap.Explanation(
    values=shap_values_arr,
    data=examples_left_df[feature_cols].values,
    feature_names=feature_cols
)

#%% global bar plot and beeswarm

shap.plots.bar(explainer_output, max_display=36)

shap.plots.beeswarm(explainer_output, max_display=36)

#%%


# Loop over each label
for label in examples_left_df['label'].unique():
    # Get indices for this label
    idx = examples_left_df['label'] == label

    # Subset the Explanation object
    subset_expl = shap.Explanation(
        values=explainer_output.values[idx],
        data=explainer_output.data[idx],
        feature_names=explainer_output.feature_names
    )

    # Plot
    print(f"SHAP bar plot for label: {label}")
    shap.plots.bar(subset_expl, max_display=36)


#%% local bar plot

examples_left_df.reset_index(drop=True, inplace=True)
example_idx = examples_left_df.loc[examples_left_df['uid'] == '26801525-1-S57'].index.values[0]

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

