""" Performing probability calibration. """

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from tensorflow.keras import Model
from tensorflow.keras.models import load_model

import paths
from models.old import config_keras
from models.models_keras import create_inputs
from src.utils_dataio import InputFnv2 as InputFn
from src.utils_dataio import get_data_from_tfrecord
from src_hpo import utils_hpo


# %%

def compute_ece(acc, conf, num_per_bin, num_total):
    """ Compute expected calibration error.

    :param acc: NumPy array, accuracy values per bin
    :param conf: NumPy array, confidence values per bin
    :param num_per_bin: NumPy array, number of examples per bin
    :param num_total: int, total number of examples
    :return:
        expected calibration error
    """

    return np.sum(np.abs(acc - conf) * num_per_bin) / num_total


def _platt_label_smoothing(el, num_pos_examples, num_neg_examples):

    if el == 0:
        return 1 / (num_neg_examples + 2)
    else:
        return (num_pos_examples + 1) / (num_pos_examples + 2)


#%% Plot reliability curve for each dataset for a given experiment

dataset = 'val'
experimentRootDir = Path('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Kepler_planet_finder/results_ensemble/')

experiment = 'keplerdr25-dv_g301-l31_6tr_spline_nongapped_norobovetterkois_starshuffle_configK_secsymphase_wksnormmaxflux-wks_corrprimgap_ptempstat_albedostat_wstdepth_fwmstat_nopps_ckoiper_secparams_prad_per'
rankingTbl = pd.read_csv(experimentRootDir / experiment / f'ensemble_ranked_predictions_{dataset}set.csv')

num_bins = 9
fractionPositives, meanPredictedVal = calibration_curve(rankingTbl['label'],
                                                        rankingTbl['score'],
                                                        n_bins=num_bins,
                                                        normalize=False,
                                                        strategy='uniform')
bin_hist_values, bin_edges = np.histogram(rankingTbl['score'], bins=num_bins, range=(0, 1))
total_num_examples = len(rankingTbl)
ece = compute_ece(fractionPositives, meanPredictedVal, bin_hist_values, total_num_examples)

f, ax = plt.subplots(2, 1)
ax[0].plot([0, 1], [0, 1], 'k:')
ax[0].plot(meanPredictedVal, fractionPositives)
ax[0].scatter(meanPredictedVal, fractionPositives, c='r')
ax[0].set_xticks(np.linspace(0, 1, 11, endpoint=True))
ax[0].set_yticks(np.linspace(0, 1, 11, endpoint=True))
ax[0].set_xlim([0, 1])
ax[0].set_ylim([0, 1])
ax[0].grid(True)
ax[0].set_ylabel('Fraction of Positives')
ax[0].set_title(f'ECE={ece:.4f}')
# ax[1].hist(rankingTbl.loc[rankingTbl['label'] == 1]['score'], bins=10, range=(0, 1), histtype='step', lw=2)
ax[1].hist(rankingTbl['score'], bins=10, range=(0, 1), histtype='step', lw=2)
ax[1].set_xticks(np.linspace(0, 1, 11, endpoint=True))
ax[1].set_xlim([0, 1])
ax[1].set_yscale('log')
ax[1].set_xlabel('Mean Predicted Value')
ax[1].set_ylabel('Counts')
f.savefig(experimentRootDir / experiment / f'reliability_curve_{dataset}.svg')

#%% Plot reliability curve for all datasets for a given experiment

datasets = ['train', 'val', 'test']
datasetsLabels = {'train': 'Train set', 'val': 'Validation set', 'test': 'Test set'}
datasetsColors = {'train': 'b', 'val': 'r', 'test': 'orange'}

experimentRootDir = Path('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Kepler_planet_finder/results_ensemble/')
experiment = 'keplerdr25-dv_g301-l31_6tr_spline_nongapped_norobovetterkois_starshuffle_configK_secsymphase_wksnormmaxflux-wks_corrprimgap_ptempstat_albedostat_wstdepth_fwmstat_nopps_ckoiper_secparams_prad_per'

reliabilityCurve = {dataset: {'mean predicted value': 0.0, 'fraction of positives': 0.0, 'scores': []}
                    for dataset in datasets}
num_bins = 5
for dataset in datasets:
    rankingTbl = pd.read_csv(experimentRootDir / experiment / f'ensemble_ranked_predictions_{dataset}set.csv')

    reliabilityCurve[dataset]['fraction of positives'], \
    reliabilityCurve[dataset]['mean predicted value'] = calibration_curve(rankingTbl['label'],
                                                                          rankingTbl['score'],
                                                                          n_bins=num_bins,
                                                                          normalize=False,
                                                                          strategy='uniform')
    reliabilityCurve[dataset]['scores'] = rankingTbl['score']

    bin_hist_values, _ = np.histogram(rankingTbl['score'],
                                      bins=len(reliabilityCurve[dataset]['mean predicted value']),
                                      range=(0, 1))
    total_num_examples = len(rankingTbl)
    reliabilityCurve[dataset]['ece'] = compute_ece(reliabilityCurve[dataset]['fraction of positives'],
                                                   reliabilityCurve[dataset]['mean predicted value'],
                                                   bin_hist_values,
                                                   total_num_examples)

f, ax = plt.subplots(2, 1, figsize=(10, 10))
ax[0].plot([0, 1], [0, 1], 'k:')
for dataset in datasets:
    ax[0].plot(reliabilityCurve[dataset]['mean predicted value'],
               reliabilityCurve[dataset]['fraction of positives'],
               datasetsColors[dataset], label=f'{datasetsLabels[dataset]} ECE {reliabilityCurve[dataset]["ece"]:.4f}')
    ax[0].scatter(reliabilityCurve[dataset]['mean predicted value'],
                  reliabilityCurve[dataset]['fraction of positives'],
                  c=datasetsColors[dataset])
    ax[1].hist(reliabilityCurve[dataset]['scores'], bins=num_bins, range=(0, 1), histtype='step', lw=2,
               label=datasetsLabels[dataset], color=datasetsColors[dataset])

ax[0].set_xticks(np.linspace(0, 1, 11, endpoint=True))
ax[0].set_yticks(np.linspace(0, 1, 11, endpoint=True))
ax[0].set_xlim([0, 1])
ax[0].set_ylim([0, 1])
ax[0].grid(True)
ax[0].set_ylabel('Fraction of Positives')
ax[0].legend()

ax[1].set_xticks(np.linspace(0, 1, 11, endpoint=True))
ax[1].set_xlim([0, 1])
ax[1].set_yscale('log')
ax[1].set_xlabel('Mean Predicted Value')
ax[1].set_ylabel('Counts')
ax[1].legend(loc='upper center')

f.savefig(experimentRootDir / experiment / 'calibration_analysis' / 'reliability_curve_alldatasets.svg')

#%%

experimentRootDir = Path('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Kepler_planet_finder/results_ensemble/')

experiment = 'keplerdr25-dv_g301-l31_6tr_spline_nongapped_norobovetterkois_starshuffle_configK_secsymphase_wksnormmaxflux-wks_corrprimgap_ptempstat_albedostat_wstdepth_fwmstat_nopps_ckoiper_secparams_prad_per'
rankingTbl = pd.read_csv(experimentRootDir / experiment / f'ensemble_ranked_predictions_valset.csv')

LR_args = {
    'penalty': 'l2',
    'C': 0.001,
    'class_weight': 'balanced',
    'random_state': 123,
    'solver': 'lbfgs',
    'max_iter': 100,
    'verbose': 1,
}

IR_args = {
    'y_min': 0,
    'y_max': 1,
    'increasing': 'auto',
    'out_of_bounds': 'nan'
}

display = False

# add_description = f'classweight{LR_args["class_weight"]}_regularizer{LR_args["penalty"]}_C{LR_args["C"]}'
add_description = f'ymin{IR_args["y_min"]}_ymax{IR_args["y_max"]}_increasing{IR_args["increasing"]}_' \
                  f'outofbounds{IR_args["out_of_bounds"]}'

# num_pos_examples = len(rankingTbl.loc[rankingTbl['label'] == 1])
# num_neg_examples = len(rankingTbl.loc[rankingTbl['label'] == 0])
# rankingTbl['label_smooth'] = rankingTbl['label'].apply(_platt_label_smoothing, args=(num_pos_examples,
#                                                                                   num_neg_examples))
# clf = LR(**LR_args)
# clf.fit(np.expand_dims(rankingTbl['score'].values, axis=-1), rankingTbl['label'].values)
clf = IsotonicRegression(**IR_args)
clf.fit(rankingTbl['score'].values, rankingTbl['label'].values)

num_bins = 10

for dataset in ['train', 'val', 'test']:

    rankingTbl = pd.read_csv(experimentRootDir / experiment / f'ensemble_ranked_predictions_{dataset}set.csv')

    # rankingTbl['calibrated_score'] = clf.predict_proba(np.expand_dims(rankingTbl['score'], axis=-1))[:, 1]
    rankingTbl['calibrated_score'] = clf.predict(rankingTbl['score'])

    f, ax = plt.subplots(2, 2, figsize=(8, 6))

    for i, score_type in enumerate(['score', 'calibrated_score']):

        fractionPositives, meanPredictedVal = calibration_curve(rankingTbl['label'],
                                                                rankingTbl[f'{score_type}'],
                                                                n_bins=num_bins,
                                                                normalize=False,
                                                                strategy='uniform')
        num_bins = len(fractionPositives)
        bin_hist_values, bin_edges = np.histogram(rankingTbl[f'{score_type}'], bins=num_bins, range=(0, 1))
        total_num_examples = len(rankingTbl)
        ece = compute_ece(fractionPositives, meanPredictedVal, bin_hist_values, total_num_examples)

        ax[0, i].plot([0, 1], [0, 1], 'k:')
        ax[0, i].plot(meanPredictedVal, fractionPositives)
        ax[0, i].scatter(meanPredictedVal, fractionPositives, c='r')
        ax[0, i].set_xticks(np.linspace(0, 1, 11, endpoint=True))
        ax[0, i].set_yticks(np.linspace(0, 1, 11, endpoint=True))
        ax[0, i].set_xlim([0, 1])
        ax[0, i].set_ylim([0, 1])
        ax[0, i].grid(True)
        if i == 0:
            ax[0, i].set_ylabel('Fraction of Positives')
        ax[0, i].set_title(f'{score_type} {dataset} set\nECE={ece:.4f}')
        # ax[1].hist(rankingTbl.loc[rankingTbl['label'] == 1]['score'], bins=10, range=(0, 1), histtype='step', lw=2)
        ax[1, i].hist(rankingTbl[f'{score_type}'], bins=10, range=(0, 1), histtype='step', lw=2)
        ax[1, i].set_xticks(np.linspace(0, 1, 11, endpoint=True))
        ax[1, i].set_xlim([0, 1])
        ax[1, i].set_yscale('log')
        ax[1, i].set_xlabel('Mean Predicted Value')
        if i == 0:
            ax[1, i].set_ylabel('Counts')
    f.suptitle(f'{add_description}')
    f.savefig(experimentRootDir / experiment / 'calibration_analysis' / 'ensemble_score_to_label_calibration' /
              f'reliability_curve_{dataset}_calibrated_plattscaling_{add_description}.svg')
    if not display:
        plt.close()

#%%

study = 'keplerdr25-dv_g301-l31_6tr_spline_nongapped_norobovetterkois_starshuffle_configK_secsymphase_wksnormmaxflux-wks_corrprimgap_ptempstat_albedostat_wstdepth_fwmstat_nopps_ckoiper_secparams_prad_per'
datasets = ['train', 'val', 'test']
res_dir = experimentRootDir / study / 'calibration_analysis' / 'singlemodels_logits_to_label_calibration'
res_dir.mkdir(exist_ok=True)

# get ensemble models filepaths
models_dir = Path(paths.pathtrainedmodels) / study / 'models'
models_filepaths = [model_dir / f'{model_dir.stem}.h5' for model_dir in models_dir.iterdir() if 'model' in
                    model_dir.stem]

# get TFRecord directory
data_dir = Path(paths.path_tfrecs) / 'Kepler' / 'Q1-Q17_DR25' / \
           'tfrecordskeplerdr25-dv_g301-l31_6tr_spline_nongapped_flux-loe-lwks-centroid-centroidfdl-6stellar-bfap-ghost-rollband-stdts_secsymphase_correctprimarygapping_confirmedkoiperiod_data/tfrecordskeplerdr25-dv_g301-l31_6tr_spline_nongapped_flux-loe-lwks-centroid-centroidfdl-6stellar-bfap-ghost-rollband-stdts_secsymphase_correctprimarygapping_confirmedkoiperiod_starshuffle_experiment-labels-norm_nopps_secparams_prad_period'

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
    # 'global_centr_fdl_view_norm': {'dim': (301, 1), 'dtype': tf.float32},
    # 'local_centr_fdl_view_norm': {'dim': (31, 1), 'dtype': tf.float32},
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

config = {}

# name of the HPO study from which to get a configuration; config needs to be set to None
hpo_study = 'ConfigK-bohb_keplerdr25-dv_g301-l31_spline_nongapped_starshuffle_norobovetterkois_glflux-glcentr_std_noclip-loe-lwks-6stellar-bfap-ghost-rollingband-convscalars_loesubtract'
# set the configuration from a HPO study
if hpo_study is not None:
    res = utils_hpo.logged_results_to_HBS_result(Path(paths.path_hpoconfigs) / hpo_study, f'_{hpo_study}')
    # get ID to config mapping
    id2config = res.get_id2config_mapping()
    # best config - incumbent
    incumbent = res.get_incumbent_id()
    config_id_hpo = incumbent
    config = id2config[config_id_hpo]['config']

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
ce_weights_args = {'tfrec_dir': data_dir,
                   'datasets': ['train'],
                   'label_fieldname': 'label',
                   'verbose': False
                   }
config = config_keras.add_dataset_params(satellite='kepler', multi_class=False, use_kepler_ce=False,
                                         ce_weights_args=ce_weights_args, config=config)

# add missing parameters in hpo with default values
config = config_keras.add_default_missing_params(config=config)

# create ensemble
model_list = []
for model_i, model_filepath in enumerate(models_filepaths):
    model = load_model(filepath=model_filepath, compile=False)
    model._name = f'model{model_i}'

    model_list.append(model)

num_models = len(model_list)
inputs = create_inputs(features=features_set, scalar_params_idxs=None)

single_models_logits = [Model(inputs=model.inputs, outputs=model.get_layer('logits').output) for model in model_list]

single_models_outputs = [model(inputs) for model in model_list]
outputs = tf.keras.layers.Average()(single_models_outputs)
# ensemble_model = Model(inputs=inputs, outputs=[outputs] + single_models_outputs)
ensemble_model = Model(inputs=inputs, outputs=outputs)
ensemble_model.summary()

fields = ['target_id', 'tce_plnt_num', 'tce_period', 'tce_duration', 'tce_time0bk', 'original_label', 'label']
data = {dataset: {field: [] for field in fields} for dataset in datasets}
tfrec_files = [file for file in data_dir.iterdir() if file.stem.split('-')[0] in datasets]
for tfrec_file in tfrec_files:

    # get dataset of the TFRecord
    dataset = tfrec_file.stem.split('-')[0]

    data_aux = get_data_from_tfrecord(tfrec_file, fields, config['label_map'])
    for field in data_aux:
        data[dataset][field].extend(data_aux[field])

# convert from list to numpy array
for dataset in datasets:
    data[dataset]['label'] = np.array(data[dataset]['label'])
    # data[dataset]['original_label'] = np.array(data[dataset]['original_label'])

# predict on given datasets - needed for computing the output distribution and produce a ranking
scores = {dataset: {field: [] for field in ['ensemble_score', 'ensemble_pred_class'] +
                    [f'logit_model{model_i}' for model_i in range(num_models)]}
          for dataset in datasets}
for dataset in scores:
    predict_input_fn = InputFn(file_pattern=str(data_dir) + '/' + dataset + '*',
                               batch_size=config['batch_size'],
                               mode=tf.estimator.ModeKeys.PREDICT,
                               label_map=config['label_map'],
                               features_set=features_set,
                               scalar_params_idxs=None)

    scores[dataset]['ensemble_score'] = ensemble_model.predict(predict_input_fn(),
                                                               batch_size=None,
                                                               verbose=1,
                                                               steps=None,
                                                               callbacks=None,
                                                               max_queue_size=10,
                                                               workers=1,
                                                               use_multiprocessing=False)
    for model_i in range(num_models):
        scores[dataset][f'logit_model{model_i}'] = single_models_logits[model_i].predict(predict_input_fn(),
                                                                                          batch_size=None,
                                                                                          verbose=1,
                                                                                          steps=None,
                                                                                          callbacks=None,
                                                                                          max_queue_size=10,
                                                                                          workers=1,
                                                                                          use_multiprocessing=False)

for dataset in datasets:
    # threshold for classification
    scores[dataset]['ensemble_pred_class'] = np.zeros(len(scores[dataset]['ensemble_score']))
    scores[dataset]['ensemble_pred_class'][scores[dataset]['ensemble_score'].ravel() >= config['clf_thr']] = 1

print('Generating csv file(s) with ranking(s)...')
# add predictions to the data dict
for dataset in datasets:
    for field in scores[dataset]:
        data[dataset][field] = scores[dataset][field].ravel()
        # data[dataset][field] = scores[dataset][field]

# write results to a txt file
for dataset in datasets:
    print(f'Saving ranked predictions in dataset {dataset} to {res_dir / f"ranked_predictions_{dataset}"}...')
    data_df = pd.DataFrame(data[dataset])

    data_df['mean_logit'] = data_df[[f'logit_model{model_i}' for model_i in range(num_models)]].mean(axis=1)
    data_df['std_logit'] = data_df[[f'logit_model{model_i}' for model_i in range(num_models)]].std(axis=1)

    # sort in descending order of output
    data_df.sort_values(by='ensemble_score', ascending=False, inplace=True)
    data_df.to_csv(res_dir / 'ensemble_ranked_predictions_{}set.csv'.format(dataset), index=False)

#%%

rankingTbl_train = pd.read_csv(res_dir / 'ensemble_ranked_predictions_valset.csv')
# interval_bound = [0.05, 0.95]
# rankingTbl_train = rankingTbl_train.loc[(rankingTbl_train['ensemble_score'] < interval_bound[0]) |
#                                         (rankingTbl_train['ensemble_score'] > interval_bound[1])]
# rankingTbl = pd.concat([pd.read_csv(res_dir / 'ensemble_ranked_predictions_valset.csv'),
#                         pd.read_csv(res_dir / 'ensemble_ranked_predictions_testset.csv')], axis=0)

LR_args = {
    'penalty': 'l2',  # either 'none', 'l2', 'l1', 'elasticnet'
    'C': 0.01,
    'class_weight': None,  # either 'None' or 'balanced'
    'random_state': 123,
    'solver': 'lbfgs',
    'max_iter': 100,
    'verbose': 1,
    'n_jobs': 10,
}

display = True

add_description = f'classweight{LR_args["class_weight"]}_regularizer{LR_args["penalty"]}_C{LR_args["C"]}'

num_pos_examples = len(rankingTbl_train.loc[rankingTbl_train['label'] == 1])
num_neg_examples = len(rankingTbl_train.loc[rankingTbl_train['label'] == 0])
rankingTbl_train['label_smooth'] = rankingTbl_train['label'].apply(_platt_label_smoothing, args=(num_pos_examples,
                                                                                                 num_neg_examples))
# clf = LR(**LR_args)
clf = IsotonicRegression(**IR_args)
clf.fit(rankingTbl_train[[f'logit_model{model_i}' for model_i in range(num_models)]].values,
        rankingTbl_train['label'].values)
# scores = cross_val_score(
#     clf,
#     rankingTbl[[f'logit_model{model_i}' for model_i in range(num_models)]].values,
#     rankingTbl['label'].values,
#     cv=LeaveOneOut()
# )
# clf_cal = CalibratedClassifierCV(
#     clf,
#     'sigmoid',
#     LeaveOneOut()
# )
# clf_cal.fit(rankingTbl[[f'logit_model{model_i}' for model_i in range(num_models)]].values,
#             rankingTbl['label'].values)

num_bins = 20

for dataset in ['train', 'val', 'test']:

    rankingTbl = pd.read_csv(res_dir / f'ensemble_ranked_predictions_{dataset}set.csv')

    rankingTbl['ensemble_calibrated_score'] = clf.predict_proba(
        rankingTbl[[f'logit_model{model_i}' for model_i in range(num_models)]])[:, 1]
    # rankingTbl = rankingTbl.loc[(rankingTbl['ensemble_score'] < interval_bound[0]) |
    #                             (rankingTbl['ensemble_score'] > interval_bound[1])]

    f, ax = plt.subplots(2, 2, figsize=(8, 6))

    for i, score_type in enumerate(['ensemble_score', 'ensemble_calibrated_score']):

        fractionPositives, meanPredictedVal = calibration_curve(rankingTbl['label'],
                                                                rankingTbl[f'{score_type}'],
                                                                n_bins=num_bins,
                                                                normalize=False,
                                                                strategy='uniform')
        num_bins = len(fractionPositives)
        bin_hist_values, bin_edges = np.histogram(rankingTbl[f'{score_type}'], bins=num_bins, range=(0, 1))
        total_num_examples = len(rankingTbl)
        ece = compute_ece(fractionPositives, meanPredictedVal, bin_hist_values, total_num_examples)

        ax[0, i].plot([0, 1], [0, 1], 'k:')
        ax[0, i].plot(meanPredictedVal, fractionPositives)
        ax[0, i].scatter(meanPredictedVal, fractionPositives, c='r')
        ax[0, i].set_xticks(np.linspace(0, 1, 11, endpoint=True))
        ax[0, i].set_yticks(np.linspace(0, 1, 11, endpoint=True))
        ax[0, i].set_xlim([0, 1])
        ax[0, i].set_ylim([0, 1])
        ax[0, i].grid(True)
        if i == 0:
            ax[0, i].set_ylabel('Fraction of Positives')
        ax[0, i].set_title(f'{score_type} {dataset} set\nECE={ece:.4f}')
        # ax[1].hist(rankingTbl.loc[rankingTbl['label'] == 1]['score'], bins=10, range=(0, 1), histtype='step', lw=2)
        ax[1, i].hist(rankingTbl[f'{score_type}'], bins=10, range=(0, 1), histtype='step', lw=2)
        ax[1, i].set_xticks(np.linspace(0, 1, 11, endpoint=True))
        ax[1, i].set_xlim([0, 1])
        ax[1, i].set_yscale('log')
        ax[1, i].set_xlabel('Mean Predicted Value')
        if i == 0:
            ax[1, i].set_ylabel('Counts')
    f.suptitle(f'{add_description}')
    f.savefig(res_dir / f'reliability_curve_{dataset}_calibrated_plattscaling_{add_description}.svg')
    if not display:
        plt.close('all')
