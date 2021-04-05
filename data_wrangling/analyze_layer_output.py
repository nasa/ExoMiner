"""
Analyze outputs of layers in the model.
"""

# 3rd party
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import load_model
import pandas as pd
import logging
from pathlib import Path
import matplotlib.pyplot as plt
from astropy.stats import mad_std
from datetime import datetime

# local
import paths
from src.utils_dataio import get_data_from_tfrecord
from src.utils_dataio import InputFnv2 as InputFn
from src.models_keras import create_ensemble
import src.config_keras
from src_hpo import utils_hpo

#%%


def create_output_layer_model(ensemble_model, layer_name):
    """ Creates a list of truncated models at the selected layer based on the models that compose the ensemble.

    :param ensemble_model: TF Keras model, ensemble
    :param layer_name: str, name of the layer to obtain output
    :return:
        output_layer_smodels: list, truncated models in the ensemble that output the values for the selected layer
    """

    single_models = [layer for layer in ensemble_model.layers if 'model' in layer.name]

    output_layer_smodels = []
    for single_model in single_models:
        layer_smodel = [layer for layer in single_model.layers if layer.name == layer_name][0]
        output_layer_smodels.append(keras.Model(inputs=single_model.input, outputs=layer_smodel.output))

    return output_layer_smodels


# name of the study
study = 'test_outputlayer_full_kepler_model'

# number of batches to iterate through
batch_steps = 200

# select layers
layers_names = [
    # 'fc_prelu_global_centr_view_std_noclip',
    # 'fc_prelu_global_flux_view_fluxnorm',
    # 'fc_prelu_local_centr_view_std_noclip',
    # 'fc_prelu_local_flux_oddeven_views',
    # 'fc_prelu_local_flux_view_fluxnorm',
    # 'fc_prelu_local_weak_secondary_view_max_flux-wks_norm',
    # 'flatten_global_centr_view_std_noclip',
    # 'flatten_global_flux_view_fluxnorm',
    'flatten_wscalar_local_centr_view_std_noclip',
    # 'flatten_local_flux_oddeven_views',
    'flatten_wscalar_local_flux_view_fluxnorm',
    'flatten_wscalar_local_weak_secondary_view_max_flux-wks_norm',
    # 'tce_steff_norm',
    # 'tce_slogg_norm',
    # 'tce_smet_norm',
    # 'tce_sradius_norm',
    # 'tce_smass_norm',
    # 'tce_sdens_norm',
    # 'tce_cap_stat_norm',
    # 'tce_hap_stat_norm',
    # 'tce_rb_tcount0_norm',
    # 'boot_fap_norm',
    # 'tce_period_norm',
    # 'tce_prad_norm',
]

# results directory
save_path = Path(paths.pathresultsensemble) / study / f'{datetime.now().strftime("%m-%d-%Y_%H-%M")}'
save_path.mkdir(exist_ok=True)

# set up logger
logger = logging.getLogger(name='predict_run')
logger_handler = logging.FileHandler(filename=save_path / f'predict_run.log', mode='w')
logger_formatter = logging.Formatter('%(asctime)s - %(message)s')
logger.setLevel(logging.INFO)
logger_handler.setFormatter(logger_formatter)
logger.addHandler(logger_handler)
logger.info(f'Starting run {study}...')

# TFRecord files directory
data_dir = os.path.join(paths.path_tfrecs,
                         'Kepler',
                         'Q1-Q17_DR25',
                         'tfrecordskeplerdr25-dv_g301-l31_6tr_spline_nongapped_flux-loe-lwks-centroid-centroidfdl-6stellar-bfap-ghost-rollband-stdts_secsymphase_correctprimarygapping_confirmedkoiperiod_data/tfrecordskeplerdr25-dv_g301-l31_6tr_spline_nongapped_flux-loe-lwks-centroid-centroidfdl-6stellar-bfap-ghost-rollband-stdts_secsymphase_correctprimarygapping_confirmedkoiperiod_starshuffle_experiment-labels-norm_nopps_secparams_prad_period'
                         )
# tfrec_dir = os.path.join(paths.path_tfrecs,
#                          'TESS',
#                          'tfrecordstess_spoctois_g301-l31_spline_nongapped_flux-oe-wks-centroid-noDV_nosecparams_data/tfrecordstess_spoctois_g301-l31_spline_nongapped_flux-oe-wks-centroid-noDV_nosecparams-normkeplerdv')

logger.info(f'Using data from {data_dir}')

# datasets used; choose from 'train', 'val', 'test', 'predict' - needs to follow naming of TFRecord files
datasets = ['train', 'val', 'test']
# datasets = ['predict']

logger.info(f'Datasets to be evaluated/tested: {datasets}')

# fields to be extracted from the TFRecords and that show up in the ranking created for each dataset
# set to None if not adding other fields
fields = {'target_id': 'int_scalar',
          'tce_plnt_num': 'int_scalar',
          'label': 'string',
          # 'oi': 'string',
          'tce_period': 'float_scalar',
          'tce_duration': 'float_scalar',
          'tce_time0bk': 'float_scalar',
          'original_label': 'string',
          'transit_depth': 'float_scalar',
          # 'Signal-to-noise': 'float_scalar'
          }
# fields = ['target_id', 'label', 'tce_plnt_num', 'tce_period', 'tce_duration', 'tce_time0bk', 'original_label']
# fields = ['target_id', 'tce_plnt_num', 'label', 'tce_period', 'tce_duration', 'tce_time0bk', 'original_label',
#           'mag', 'ra', 'dec', 'tce_max_mult_ev', 'tce_insol', 'tce_eqt', 'tce_sma', 'tce_prad', 'tce_model_snr',
#           'tce_ingress', 'tce_impact', 'tce_incl', 'tce_dor', 'tce_ror']

# print('Fields to be extracted from the TFRecords: {}'.format(fields))

multi_class = False  # multiclass classification
ce_weights_args = {'tfrec_dir': data_dir,
                   'datasets': ['train'],
                   'label_fieldname': 'label',
                   'verbose': False
                   }
use_kepler_ce = False  # use weighted CE loss based on the class proportions in the training set
satellite = 'kepler'  # if 'kepler' in tfrec_dir else 'tess

generate_csv_pred = True

# get models for the ensemble
models_study = '/data5/tess_project/git_repo/trained_models/experiments_paper(9-14-2020_to_1-19-2021)/keplerdr25-dv_g301-l31_6tr_spline_nongapped_norobovetterkois_starshuffle_configK_secsymphase_wksnormmaxflux-wks_corrprimgap_ptempstat_albedostat_wstdepth_fwmstat_nopps_ckoiper_secparams_prad_per'  # study
models_dir = Path(paths.pathtrainedmodels) / models_study / 'models'
models_filepaths = [model_dir / f'{model_dir.stem}.h5' for model_dir in models_dir.iterdir() if 'model' in
                    model_dir.stem]
logger.info(f'Models\' file paths: {models_filepaths}')

# intializing configuration dictionary; can set configuration manually
config = {}

# name of the HPO study from which to get a configuration; config needs to be set to None
hpo_study = 'experiments_paper(9-14-2020_to_1-19-2021)/ConfigK-bohb_keplerdr25-dv_g301-l31_spline_nongapped_starshuffle_norobovetterkois_glflux-glcentr_std_noclip-loe-lwks-6stellar-bfap-ghost-rollingband-convscalars_loesubtract'

# set the configuration from a HPO study
if hpo_study is not None:
    hpo_study_fp = Path(paths.path_hpoconfigs) / hpo_study
    res = utils_hpo.logged_results_to_HBS_result(hpo_study_fp, f'_{hpo_study_fp.name}')
    # get ID to config mapping
    id2config = res.get_id2config_mapping()
    # best config - incumbent
    incumbent = res.get_incumbent_id()
    config_id_hpo = incumbent
    config = id2config[config_id_hpo]['config']

    # select a specific config based on its ID
    # example - check config.json
    # config = id2config[(8, 0, 3)]['config']

    logger.info(f'Using configuration from HPO study {hpo_study}')
    logger.info(f'HPO Config {config_id_hpo}: {config}')

config.update({
    # 'num_loc_conv_blocks': 2,
    # 'num_glob_conv_blocks': 5,
    # 'init_fc_neurons': 512,
    # 'num_fc_layers': 4,
    # 'pool_size_loc': 7,
    # 'pool_size_glob': 5,
    # 'pool_stride': 2,
    # 'conv_ls_per_block': 2,
    # 'init_conv_filters': 4,
    # 'kernel_size': 5,
    # 'kernel_stride': 1,
    'non_lin_fn': 'prelu',  # 'relu',
    # 'optimizer': 'Adam',
    # 'lr': 1e-5,
    # 'batch_size': 64,
    # 'dropout_rate': 0,
})

# add dataset parameters
config = src.config_keras.add_dataset_params(satellite, multi_class, use_kepler_ce, ce_weights_args, config)

# add missing parameters in hpo with default values
config = src.config_keras.add_default_missing_params(config=config)
# print('Configuration used: ', config)

logger.info(f'Final configuration used: {config}')

features_set = {
    # flux related features
    'global_flux_view_fluxnorm': {'dim': (301, 1), 'dtype': tf.float32},
    'local_flux_view_fluxnorm': {'dim': (31, 1), 'dtype': tf.float32},
    'transit_depth_norm': {'dim': (1,), 'dtype': tf.float32},
    # odd-even flux views
    'local_flux_odd_view_fluxnorm': {'dim': (31, 1), 'dtype': tf.float32},
    'local_flux_even_view_fluxnorm': {'dim': (31, 1), 'dtype': tf.float32},
    # secondary flux views
    # 'local_weak_secondary_view_fluxnorm': {'dim': (31, 1), 'dtype': tf.float32},
    # 'local_weak_secondary_view_selfnorm': {'dim': (31, 1), 'dtype': tf.float32},
    'local_weak_secondary_view_max_flux-wks_norm': {'dim': (31, 1), 'dtype': tf.float32},
    # secondary flux related features
    'tce_maxmes_norm': {'dim': (1,), 'dtype': tf.float32},
    'wst_depth_norm': {'dim': (1,), 'dtype': tf.float32},
    # 'tce_albedo_norm': {'dim': (1,), 'dtype': tf.float32},
    'tce_albedo_stat_norm': {'dim': (1,), 'dtype': tf.float32},
    # 'tce_ptemp_norm': {'dim': (1,), 'dtype': tf.float32},
    'tce_ptemp_stat_norm': {'dim': (1,), 'dtype': tf.float32},
    # centroid views
    'global_centr_view_std_noclip': {'dim': (301, 1), 'dtype': tf.float32},
    'local_centr_view_std_noclip': {'dim': (31, 1), 'dtype': tf.float32},
    # 'global_centr_fdl_view_norm': {'dim': (2001, 1), 'dtype': tf.float32},
    # 'local_centr_fdl_view_norm': {'dim': (201, 1), 'dtype': tf.float32},
    # centroid related features
    'tce_fwm_stat_norm': {'dim': (1,), 'dtype': tf.float32},
    'tce_dikco_msky_norm': {'dim': (1,), 'dtype': tf.float32},
    'tce_dikco_msky_err_norm': {'dim': (1,), 'dtype': tf.float32},
    'tce_dicco_msky_norm': {'dim': (1,), 'dtype': tf.float32},
    'tce_dicco_msky_err_norm': {'dim': (1,), 'dtype': tf.float32},
    # other diagnostic parameters
    'boot_fap_norm': {'dim': (1,), 'dtype': tf.float32},
    'tce_cap_stat_norm': {'dim': (1,), 'dtype': tf.float32},
    'tce_hap_stat_norm': {'dim': (1,), 'dtype': tf.float32},
    'tce_rb_tcount0_norm': {'dim': (1,), 'dtype': tf.float32},
    # stellar parameters
    'tce_sdens_norm': {'dim': (1,), 'dtype': tf.float32},
    'tce_steff_norm': {'dim': (1,), 'dtype': tf.float32},
    'tce_smet_norm': {'dim': (1,), 'dtype': tf.float32},
    'tce_slogg_norm': {'dim': (1,), 'dtype': tf.float32},
    'tce_smass_norm': {'dim': (1,), 'dtype': tf.float32},
    'tce_sradius_norm': {'dim': (1,), 'dtype': tf.float32},
    # tce parameters
    'tce_prad_norm': {'dim': (1,), 'dtype': tf.float32},
    'tce_period_norm': {'dim': (1,), 'dtype': tf.float32},
}
logger.info(f'Feature set: {features_set}')

verbose = False if 'home6' in paths.path_hpoconfigs else True

# instantiate variable to get data from the TFRecords
data = {dataset: {field: [] for field in fields} for dataset in datasets}

tfrec_files = [file for file in os.listdir(data_dir) if file.split('-')[0] in datasets]
for tfrec_file in tfrec_files:

    # get dataset of the TFRecord
    dataset = tfrec_file.split('-')[0]

    fields_aux = fields

    data_aux = get_data_from_tfrecord(os.path.join(data_dir, tfrec_file), fields_aux, config['label_map'])

    for field in data_aux:
        data[dataset][field].extend(data_aux[field])

# convert from list to numpy array
# TODO: should make this a numpy array from the beginning
for dataset in datasets:
    data[dataset]['label'] = np.array(data[dataset]['label'])
    data[dataset]['original_label'] = np.array(data[dataset]['original_label'])

# create ensemble
model_list = []
for model_i, model_filepath in enumerate(models_filepaths):

    model = load_model(filepath=model_filepath, compile=False)
    model._name = f'model{model_i}'

    model_list.append(model)

ensemble_model = create_ensemble(features=features_set, models=model_list)

ensemble_model.summary()

# predict on given datasets - needed for computing the output distribution and produce a ranking
for dataset in datasets:
    predict_input_fn = InputFn(file_pattern=data_dir + '/' + dataset + '*',
                               batch_size=config['batch_size'],
                               mode=tf.estimator.ModeKeys.PREDICT,
                               label_map=config['label_map'],
                               features_set=features_set)

    for layer_name in layers_names:

        save_path_layer = save_path / f'{layer_name}'
        save_path_layer.mkdir(exist_ok=True)

        output_layer_smodels = create_output_layer_model(ensemble_model, layer_name)
        for smodel_i, output_layer_smodel in enumerate(output_layer_smodels):

            print(f'Generating output for layer {layer_name} on dataset {dataset} for model {smodel_i + 1} out of '
                  f'{len(output_layer_smodels)}...')

            layer_output = output_layer_smodel.predict(predict_input_fn(),
                                                     batch_size=None,
                                                     verbose=verbose,
                                                     steps=batch_steps,
                                                     callbacks=None,
                                                     max_queue_size=10,
                                                     workers=1,
                                                     use_multiprocessing=False,
                                                       )

            model_csv_fp = save_path_layer / f'{dataset}_model{smodel_i + 1}.csv'
            data_df = pd.DataFrame(layer_output)
            data_df = pd.concat([data_df,
                                 pd.DataFrame(data[dataset])[:min(len(data_df), batch_steps * config['batch_size'])]],
                                axis=1)
            data_df.to_csv(model_csv_fp, index=False)

#%% Used when we did not have the name of the layers for the convolutional branches (outdated)

res_dir = Path('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Kepler_planet_finder/results_ensemble/test_outputlayer_full_kepler_model/')
dataset = 'train'
layer_name = 'convbranch_concat'
smodels_dfs = [pd.read_csv(fp) for fp in res_dir.iterdir() if 'train' in fp.stem and layer_name in fp.stem]

mean_df = pd.concat(smodels_dfs).groupby(level=0).mean()
std_df = pd.concat(smodels_dfs).groupby(level=0).std()

bins = np.linspace(-1, 1, 100)
# for branch_i in range(6):
#     f, ax = plt.subplots()
#     hist_data = mean_df[[str(el) for el in range(branch_i * 4, branch_i * 4 + 4)]].values.flatten()
#     stats_dict = {
#         'mean': np.mean(hist_data),
#         'median': np.median(hist_data),
#         'mad_std': mad_std(hist_data),
#         'min': np.min(hist_data),
#         'max': np.max(hist_data)
#     }
#     ax.hist(hist_data, bins=bins)
#     ax.set_xlim([bins[0], bins[-1]])
#     ax.set_ylabel('Counts')
#     ax.set_xlabel('Activation')
#     title_str = ''
#     for stat_name, stat_val in stats_dict.items():
#         title_str += f'{stat_name}={stat_val:.4} '
#     ax.set_title(title_str, fontsize=10)
#     f.suptitle(f'Branch {branch_i + 1}')
#     f.savefig(res_dir / 'histograms' / f'{layer_name}_{dataset}_branch{branch_i + 1}.png')

f, ax = plt.subplots(2, 3, figsize=(18, 9))
row, col = 0, 0
for conv_branch_i in range(6):
    hist_data = mean_df[[str(el) for el in range(4 * conv_branch_i, 4 * conv_branch_i + 4)]].values.flatten()
    stats_dict = {
        'mean': np.mean(hist_data),
        'median': np.median(hist_data),
        'mad_std': mad_std(hist_data),
        'min': np.min(hist_data),
        'max': np.max(hist_data)
    }
    ax[row, col].hist(hist_data, bins=bins, density=True)
    ax[row, col].set_xlim([bins[0], bins[-1]])
    ax[row, col].set_ylabel('Counts')
    ax[row, col].set_xlabel('Activation')
    title_str = f'Conv branch {conv_branch_i + 1}\n'
    for stat_name, stat_val in stats_dict.items():
        title_str += f'{stat_name}={stat_val:.4} '
    ax[row, col].set_title(title_str, fontsize=10)
    col += 1
    if col == 3:
        col = 0
        row += 1
f.subplots_adjust(top=0.943,
bottom=0.066,
left=0.042,
right=0.984,
hspace=0.765,
wspace=0.177)
f.savefig(res_dir / 'histograms' / f'{layer_name}_{dataset}_convbranch.png')

#%% Used when we did not have the name of the layers for the scalar parameters (outdated)

res_dir = Path('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Kepler_planet_finder/results_ensemble/test_outputlayer_full_kepler_model/')
dataset = 'train'
layer_name = 'stellar_dv_scalar_input_'
smodels_dfs = [pd.read_csv(fp) for fp in res_dir.iterdir() if 'train' in fp.stem and layer_name in fp.stem]

mean_df = pd.concat(smodels_dfs).groupby(level=0).mean()
std_df = pd.concat(smodels_dfs).groupby(level=0).std()

bins = np.linspace(-1, 1, 100)
# for scalar_param_i in range(12):
#     f, ax = plt.subplots()
#     hist_data = mean_df[str(scalar_param_i)].values.flatten()
#     stats_dict = {
#         'mean': np.mean(hist_data),
#         'median': np.median(hist_data),
#         'mad_std': mad_std(hist_data),
#         'min': np.min(hist_data),
#         'max': np.max(hist_data)
#     }
#     ax.hist(hist_data, bins=bins)
#     ax.set_xlim([bins[0], bins[-1]])
#     ax.set_ylabel('Counts')
#     ax.set_xlabel('Activation')
#     title_str = ''
#     for stat_name, stat_val in stats_dict.items():
#         title_str += f'{stat_name}={stat_val:.4} '
#     ax.set_title(title_str, fontsize=10)
#     f.suptitle(f'Scalar parameter {scalar_param_i + 1}')
#     f.savefig(res_dir / 'histograms' / f'{layer_name}_{dataset}_branch{scalar_param_i + 1}.png')

f, ax = plt.subplots(4, 3, figsize=(18, 9))
row, col = 0, 0
for scalar_param_i in range(12):
    hist_data = mean_df[str(scalar_param_i)].values.flatten()
    stats_dict = {
        'mean': np.mean(hist_data),
        'median': np.median(hist_data),
        'mad_std': mad_std(hist_data),
        'min': np.min(hist_data),
        'max': np.max(hist_data)
    }
    ax[row, col].hist(hist_data, bins=bins)
    ax[row, col].set_xlim([bins[0], bins[-1]])
    ax[row, col].set_ylabel('Counts')
    ax[row, col].set_xlabel('Activation')
    title_str = f'Scalar parameter {scalar_param_i + 1}\n'
    for stat_name, stat_val in stats_dict.items():
        title_str += f'{stat_name}={stat_val:.4} '
    ax[row, col].set_title(title_str, fontsize=10)
    col += 1
    if col == 3:
        col = 0
        row += 1
f.subplots_adjust(top=0.943,
bottom=0.066,
left=0.042,
right=0.984,
hspace=0.765,
wspace=0.177)
f.savefig(res_dir / 'histograms' / f'{layer_name}_{dataset}_scalarparams.png')

#%% Analyze output from convolutional branches FC and scalar parameters

res_dir_fp = '/home/msaragoc/Projects/Kepler-TESS_exoplanet/Kepler_planet_finder/results_ensemble/test_outputlayer_full_kepler_model/03-25-2021_13-37/'
res_dir = Path(res_dir_fp)
(res_dir / 'histograms').mkdir(exist_ok=True)

layers_names = [
    'fc_prelu_global_centr_view_std_noclip',
    'fc_prelu_global_flux_view_fluxnorm',
    'fc_prelu_local_centr_view_std_noclip',
    'fc_prelu_local_flux_oddeven_views',
    'fc_prelu_local_flux_view_fluxnorm',
    'fc_prelu_local_weak_secondary_view_max_flux-wks_norm',
    # 'tce_steff_norm',
    # 'tce_slogg_norm',
    # 'tce_smet_norm',
    # 'tce_sradius_norm',
    # 'tce_smass_norm',
    # 'tce_sdens_norm',
    # 'tce_cap_stat_norm',
    # 'tce_hap_stat_norm',
    # 'tce_rb_tcount0_norm',
    # 'boot_fap_norm',
    # 'tce_period_norm',
    # 'tce_prad_norm',
    # 'flatten_global_centr_view_std_noclip',
    # 'flatten_global_flux_view_fluxnorm',
    # 'flatten_wscalar_local_centr_view_std_noclip',
    # 'flatten_local_flux_oddeven_views',
    # 'flatten_wscalar_local_flux_view_fluxnorm',
    # 'flatten_wscalarlocal_weak_secondary_view_max_flux-wks_norm',
]

dataset = 'train'

# bins = np.linspace(-1, 1, 100)
bins = np.logspace(-5, 0, 100)

f, ax = plt.subplots(2, 3, figsize=(18, 9))
row, col = 0, 0
for layer_i, layer_name in enumerate(layers_names):
    layer_dir = res_dir / layer_name
    smodels_dfs = [pd.read_csv(fp) for fp in layer_dir.iterdir() if 'train' in fp.stem]

    mean_df = pd.concat(smodels_dfs).groupby(level=0).mean()
    std_df = pd.concat(smodels_dfs).groupby(level=0).std()
    hist_data = mean_df[[str(el) for el in range(4)]].values.flatten()
    # hist_data = mean_df[['0']].values.flatten()
    stats_dict = {
        'mean': np.mean(hist_data),
        'median': np.median(hist_data),
        'mad_std': mad_std(hist_data),
        'min': np.min(hist_data),
        'max': np.max(hist_data)
    }
    ax[row, col].hist(hist_data, bins=bins, density=False)
    ax[row, col].set_xlim([bins[0], bins[-1]])
    ax[row, col].set_ylabel('Counts')
    ax[row, col].set_xlabel('Activation')
    ax[row, col].set_yscale('log')
    title_str = f'{layer_name}\n'
    for stat_name, stat_val in stats_dict.items():
        title_str += f'{stat_name}={stat_val:.4} '
    ax[row, col].set_title(title_str, fontsize=10)
    col += 1
    if col == 3:
        col = 0
        row += 1
f.subplots_adjust(top=0.943,
bottom=0.066,
left=0.042,
right=0.984,
hspace=0.765,
wspace=0.177)
# f.subplots_adjust(
# top=0.942,
# bottom=0.066,
# left=0.044,
# right=0.983,
# hspace=0.771,
# wspace=0.191
# )
f.savefig(res_dir / 'histograms' / f'{dataset}_convbranches.png')

dispositions = {
    'PC': {'zorder': 3},
    'AFP': {'zorder': 2},
    'NTP': {'zorder': 1}
}
f, ax = plt.subplots(2, 3, figsize=(18, 9))
row, col = 0, 0
for conv_branch_i, layer_name in enumerate(layers_names):
    layer_dir = res_dir / layer_name
    smodels_dfs = [pd.read_csv(fp) for fp in layer_dir.iterdir() if 'train' in fp.stem]

    mean_df = pd.concat(smodels_dfs).groupby(level=0).mean()
    mean_df = pd.concat([mean_df, smodels_dfs[0]['original_label']], axis=1)
    # std_df = pd.concat(smodels_dfs).groupby(level=0).std()
    for disp in dispositions:
        mean_df_disp = mean_df.loc[mean_df['original_label'] == disp]
        hist_data = mean_df_disp[[str(el) for el in range(4)]].values.flatten()
        # hist_data = mean_df_disp[['0']].values.flatten()
        # stats_dict = {
        #     'mean': np.mean(hist_data),
        #     'median': np.median(hist_data),
        #     'mad_std': mad_std(hist_data),
        #     'min': np.min(hist_data),
        #     'max': np.max(hist_data)
        # }
        ax[row, col].hist(hist_data, bins=bins, density=False, label=disp, zorder=dispositions[disp]['zorder'], stacked=False, histtype='step', cumulative=False)
    ax[row, col].set_xlim([bins[0], bins[-1]])
    ax[row, col].set_ylabel('Counts')
    ax[row, col].set_xlabel('Activation')
    ax[row, col].set_yscale('log')
    ax[row, col].set_xscale('log')
    title_str = f'{layer_name}\n'
    # for stat_name, stat_val in stats_dict.items():
    #     title_str += f'{stat_name}={stat_val:.4} '
    ax[row, col].set_title(title_str, fontsize=10)
    ax[row, col].legend(loc='upper left')
    col += 1
    if col == 3:
        col = 0
        row += 1
f.subplots_adjust(top=0.943,
bottom=0.066,
left=0.042,
right=0.984,
hspace=0.765,
wspace=0.177)
# f.subplots_adjust(top=0.943,
# bottom=0.066,
# left=0.042,
# right=0.984,
# hspace=0.765,
# wspace=0.177)
f.savefig(res_dir / 'histograms' / f'{dataset}_convbranches_dispositions.png')

#%% Analyze output from convolutional branches before FC (local wks, local centroid, local flux branches)

res_dir_fp = '/home/msaragoc/Projects/Kepler-TESS_exoplanet/Kepler_planet_finder/results_ensemble/test_outputlayer_full_kepler_model/03-25-2021_17-11/'
res_dir = Path(res_dir_fp)
(res_dir / 'histograms').mkdir(exist_ok=True)

layers_names = {
    'flatten_wscalar_local_centr_view_std_noclip': {'indx_s': 672, 'indx_e': 676,
                                                    'plots': (2, 3),
                                                    'feature_name': ['local_centr_view_std_noclip',
                                                                     'tce_dikco_msky_norm',
                                                                     'tce_dikco_msky_err_norm',
                                                                     'tce_dicco_msky_norm',
                                                                     'tce_dicco_msky_err_norm',
                                                                     'tce_fwm_stat_norm']},
    'flatten_wscalar_local_flux_view_fluxnorm': {'indx_s': 672, 'indx_e': 672,
                                                 'plots': (1, 2),
                                                 'feature_name': ['local_flux_view_fluxnorm',
                                                                  'transit_depth_norm']},
    'flatten_wscalar_local_weak_secondary_view_max_flux-wks_norm': {'indx_s': 672, 'indx_e': 675,
                                                                    'plots': (2, 3),
                                                                    'feature_name': ['local_weak_secondary_view_max_flux-wks_norm',
                                                                                     'tce_maxmes_norm',
                                                                                     'tce_albedo_stat_norm',
                                                                                     'tce_ptemp_stat_norm',
                                                                                     'wst_depth_norm']},
}

dataset = 'train'

bins = np.linspace(-20, 20, 100)
bins_log = np.logspace(-5, 2, 100)

for layer_name in layers_names:

    num_plots = layers_names[layer_name]['indx_e'] - layers_names[layer_name]['indx_s'] + 1

    layer_dir = res_dir / layer_name
    smodels_dfs = [pd.read_csv(fp) for fp in layer_dir.iterdir() if 'train' in fp.stem]

    mean_df = pd.concat(smodels_dfs).groupby(level=0).mean()
    std_df = pd.concat(smodels_dfs).groupby(level=0).std()

    f, ax = plt.subplots(layers_names[layer_name]['plots'][0], layers_names[layer_name]['plots'][1], figsize=(18, 9), squeeze=False)
    hist_data = mean_df[[str(el) for el in range(layers_names[layer_name]['indx_s'])]].values.flatten()
    stats_dict = {
        'mean': np.mean(hist_data),
        'median': np.median(hist_data),
        'mad_std': mad_std(hist_data),
        'min': np.min(hist_data),
        'max': np.max(hist_data)
    }
    ax[0, 0].hist(hist_data, bins=bins, density=False)
    ax[0, 0].set_xlim([bins[0], bins[-1]])
    ax[0, 0].set_ylabel('Counts')
    ax[0, 0].set_xlabel('Activation')
    ax[0, 0].set_yscale('log')
    title_str = f'{layers_names[layer_name]["feature_name"][0]}\n'
    for stat_name, stat_val in stats_dict.items():
        title_str += f'{stat_name}={stat_val:.4} '
    ax[0, 0].set_title(title_str, fontsize=10)

    row, col = 0, 1
    for plot_i in range(0, num_plots):
        hist_data = mean_df[f'{layers_names[layer_name]["indx_s"] + plot_i}'].values.flatten()
        stats_dict = {
            'mean': np.mean(hist_data),
            'median': np.median(hist_data),
            'mad_std': mad_std(hist_data),
            'min': np.min(hist_data),
            'max': np.max(hist_data)
        }
        ax[row, col].hist(hist_data, bins=bins, density=False)
        ax[row, col].set_xlim([bins[0], bins[-1]])
        ax[row, col].set_ylabel('Counts')
        ax[row, col].set_xlabel('Activation')
        ax[row, col].set_yscale('log')
        title_str = f'{layers_names[layer_name]["feature_name"][plot_i + 1]}\n'
        for stat_name, stat_val in stats_dict.items():
            title_str += f'{stat_name}={stat_val:.4} '
        ax[row, col].set_title(title_str, fontsize=10)

        col += 1
        if col == layers_names[layer_name]['plots'][1]:
            col = 0
            row += 1

    f.subplots_adjust(
        top=0.942,
        bottom=0.066,
        left=0.039,
        right=0.986,
        hspace=0.278,
        wspace=0.157
    )
    f.savefig(res_dir / 'histograms' / f'{dataset}_{layer_name}.png')

dispositions = {
    'PC': {'zorder': 3},
    'AFP': {'zorder': 2},
    'NTP': {'zorder': 1}
}

for layer_name in layers_names:

    num_plots = layers_names[layer_name]['indx_e'] - layers_names[layer_name]['indx_s'] + 1

    layer_dir = res_dir / layer_name
    smodels_dfs = [pd.read_csv(fp) for fp in layer_dir.iterdir() if 'train' in fp.stem]

    mean_df = pd.concat(smodels_dfs).groupby(level=0).mean()
    mean_df = pd.concat([mean_df, smodels_dfs[0]['original_label']], axis=1)
    # std_df = pd.concat(smodels_dfs).groupby(level=0).std()

    f, ax = plt.subplots(layers_names[layer_name]['plots'][0], layers_names[layer_name]['plots'][1], figsize=(18, 9), squeeze=False)
    hist_data = mean_df[[str(el) for el in range(layers_names[layer_name]['indx_s'])]].values.flatten()
    for disp in dispositions:
        mean_df_disp = mean_df.loc[mean_df['original_label'] == disp]
        hist_data = mean_df_disp[[str(el) for el in range(layers_names[layer_name]['indx_s'])]].values.flatten()
        ax[0, 0].hist(hist_data, bins=bins_log, density=False, label=disp, zorder=dispositions[disp]['zorder'],
                          stacked=False, histtype='step', cumulative=False)
    ax[0, 0].set_xlim([bins_log[0], bins_log[-1]])
    ax[0, 0].set_ylabel('Counts')
    ax[0, 0].set_xlabel('Activation')
    ax[0, 0].set_yscale('log')
    ax[0, 0].set_xscale('log')
    ax[0, 0].legend(loc='upper left')
    title_str = f'{layers_names[layer_name]["feature_name"][0]}\n'
    ax[0, 0].set_title(title_str, fontsize=10)

    row, col = 0, 1
    for plot_i in range(0, num_plots):
        for disp in dispositions:
            mean_df_disp = mean_df.loc[mean_df['original_label'] == disp]
            hist_data = mean_df_disp[f'{layers_names[layer_name]["indx_s"] + plot_i}'].values.flatten()
            ax[row, col].hist(hist_data, bins=bins, density=False, label=disp, zorder=dispositions[disp]['zorder'], stacked=False, histtype='step', cumulative=False)
        ax[row, col].set_xlim([bins[0], bins[-1]])
        ax[row, col].set_ylabel('Counts')
        ax[row, col].set_xlabel('Activation')
        ax[row, col].set_yscale('log')
        # ax[row, col].set_xscale('log')
        ax[row, col].legend(loc='upper left')
        title_str = f'{layers_names[layer_name]["feature_name"][plot_i + 1]}\n'
        ax[row, col].set_title(title_str, fontsize=10)

        col += 1
        if col == layers_names[layer_name]['plots'][1]:
            col = 0
            row += 1

    f.subplots_adjust(
        top=0.942,
        bottom=0.066,
        left=0.039,
        right=0.986,
        hspace=0.278,
        wspace=0.157
    )
    f.savefig(res_dir / 'histograms' / f'{dataset}_{layer_name}_dispositions.png')

