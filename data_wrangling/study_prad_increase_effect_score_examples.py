"""
Studying effect of changing TCE planet radius on the score.
"""

# 3rd party
import tensorflow as tf
from pathlib import Path
import numpy as np
import yaml
import pandas as pd

# local
from src_preprocessing.tf_util import example_util
from utils.utils_dataio import is_yamlble

#%% Increase planet radius by a given factor

src_tfrec_dir = Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/data/tfrecords/kepler/tfrecordskeplerq1q17dr25-dv_g301-l31_5tr_spline_nongapped_all_features_paper_rbat0norm_8-20-2021_predict_tcesnotused')

# test values assuming 1 Re planet from 0.5 Re to 30* Re (largest exoplanet found with 1.7 Rj planet radius ~ 19 Re)
prad_factor_arr = [0.5, 1.5, 2, 5, 10, 15, 20, 25, 30]
dest_tfrec_dir_root = src_tfrec_dir.parent / f'{src_tfrec_dir.name}_prad-f_7-7-2022'
dest_tfrec_dir_root.mkdir(exist_ok=True)

for prad_f in prad_factor_arr:
    print(f'Creating data set for planet radius factor {prad_f}...')
    dest_tfrec_dir = dest_tfrec_dir_root / f'{src_tfrec_dir.name}_prad-f-{prad_f}'
    dest_tfrec_dir.mkdir(exist_ok=True)

    src_tfrec_fps = [fp for fp in list(src_tfrec_dir.iterdir()) if not fp.suffix == '.csv' and fp.is_file()]

    for src_tfrec_fp in src_tfrec_fps:

        with tf.io.TFRecordWriter(str(dest_tfrec_dir / src_tfrec_fp.name)) as writer:

            # iterate through the source shard
            tfrecord_dataset = tf.data.TFRecordDataset(str(src_tfrec_fp))

            for string_i, string_record in enumerate(tfrecord_dataset.as_numpy_iterator()):

                example = tf.train.Example()
                example.ParseFromString(string_record)

                # target_id = example.features.feature['target_id'].int64_list.value[0]
                # tce_plnt_num = example.features.feature['tce_plnt_num'].int64_list.value[0]

                # if f'{target_id}.{tce_plnt_num}' == '11030475.1':
                tce_prad = example.features.feature['tce_prad'].float_list.value[0]
                # tce_prad_new = tce_prad * np.sqrt(2)
                tce_prad_new = tce_prad * prad_f
                #     print(f'Found KIC 11030475.1. Changing planet radius for this TCE from {tce_prad} to {tce_prad_new} '
                #           f'Re.')
                example_util.set_float_feature(example, 'tce_prad', [tce_prad_new], allow_overwrite=True)

                writer.write(example.SerializeToString())

# %% Check if value was changed

tfrec_dir = Path(
    '/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/data/tfrecords/kepler/tfrecordskeplerq1q17dr25-dv_g301-l31_5tr_spline_nongapped_all_features_paper_rbat0norm_8-20-2021_predict_tcesnotused_koi-2248.01_plnt_rad_sqrt2_increase_7-7-2022')

tfrec_fps = [fp for fp in list(tfrec_dir.iterdir()) if not fp.suffix == '.csv' and fp.is_file()]

for tfrec_fp in tfrec_fps:

    # iterate through the source shard
    tfrecord_dataset = tf.data.TFRecordDataset(str(tfrec_fp))

    for string_i, string_record in enumerate(tfrecord_dataset.as_numpy_iterator()):

        example = tf.train.Example()
        example.ParseFromString(string_record)

        target_id = example.features.feature['target_id'].int64_list.value[0]
        tce_plnt_num = example.features.feature['tce_plnt_num'].int64_list.value[0]

        if f'{target_id}.{tce_plnt_num}' == '11030475.1':
            tce_prad = example.features.feature['tce_prad'].float_list.value[0]
            tce_prad_new = tce_prad * np.sqrt(2)
            print(f'Found KIC 11030475.1. Changing planet radius for this TCE from {tce_prad} to {tce_prad_new} '
                  f'Re.')
            aaa

#%% Create data set runs

exp_dir = Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/experiments/cv_predict_keplerq1q17dr25_prad_range_7-7-2022')
# exp_dir.mkdir(exist_ok=True)
configs_dir = exp_dir / 'configs'
configs_dir.mkdir(exist_ok=True)
datasets_fps = [fp for fp in list(dest_tfrec_dir_root.iterdir()) if fp.is_dir()]
default_config_fp = Path(exp_dir / 'config_cv_predict.yaml')
config_runs_fp = exp_dir / 'configs.txt'
for datasets_fp in datasets_fps:

    print(f'Create config file for run {datasets_fp.name}...')

    dataset_name = datasets_fp.name.split('_')[-1]

    with(open(default_config_fp, 'r')) as file:  # read default YAML configuration file
        config = yaml.safe_load(file)

    # update path to TFRecord directory
    config['paths']['tfrec_dir'] = str(datasets_fp)

    # set experiment directory
    config['paths']['experiment_dir'] = str(exp_dir / dataset_name)

    # save the YAML config pred file
    yaml_dict = {key: val for key, val in config.items() if is_yamlble(val)}  # check if parameters are YAMLble
    config_fn = f'config_{dataset_name}.yaml'
    with open(configs_dir / config_fn, 'w') as config_file:
        yaml.dump(yaml_dict, config_file)

    # add configuration to list
    with open(config_runs_fp, 'a') as list_configs_file:
        list_configs_file.write(f'{config_fn}\n')

# %% Combine predictions from all CV iterations in the not-used dataset

exp_dir_root = Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/experiments/cv_predict_keplerq1q17dr25_prad_range_7-7-2022/')
exp_runs = [fp for fp in exp_dir_root.iterdir() if 'prad' in fp.name]

# for each run, aggregate all scores across CV folds and compute mean and std scores
for exp_run in exp_runs:

    cv_iters_dirs = [fp for fp in exp_run.iterdir() if fp.is_dir() and fp.name.startswith('cv_iter')]

    cv_iters_tbls = []
    for cv_iter_dir in cv_iters_dirs:
        ranking_tbl = pd.read_csv(cv_iter_dir / 'ensemble_ranked_predictions_predictset.csv')
        ranking_tbl['fold'] = cv_iter_dir.name.split('_')[-1]
        cv_iters_tbls.append(ranking_tbl)

    ranking_tbl_cv = pd.concat(cv_iters_tbls, axis=0)
    ranking_tbl_cv.to_csv(exp_run / 'ensemble_ranked_predictions_allfolds.csv', index=False)

    # get mean and std score across all folds for the prediction on the not-used dataset
    tbl = None
    for cv_iter_dir in cv_iters_dirs:
        ranking_tbl = pd.read_csv(cv_iter_dir / 'ensemble_ranked_predictions_predictset.csv')
        ranking_tbl['fold'] = cv_iter_dir.name.split('_')[-1]
        ranking_tbl[f'score_fold_{cv_iter_dir.name.split("_")[-1]}'] = ranking_tbl['score']
        if tbl is None:
            tbl = ranking_tbl
        else:
            tbl = pd.merge(tbl,
                           ranking_tbl[['target_id', 'tce_plnt_num', f'score_fold_{cv_iter_dir.name.split("_")[-1]}']],
                           on=['target_id', 'tce_plnt_num'])

    tbl['mean_score'] = tbl[[f'score_fold_{i}' for i in range(9)]].mean(axis=1)
    tbl['std_score'] = tbl[[f'score_fold_{i}' for i in range(9)]].std(axis=1)
    tbl[['target_id', 'tce_plnt_num', 'label', 'tce_period', 'tce_duration',
         'tce_time0bk', 'original_label', 'transit_depth',
         'score_fold_4', 'score_fold_9',
         'score_fold_3', 'score_fold_7', 'score_fold_0', 'score_fold_2',
         'score_fold_8', 'score_fold_5', 'score_fold_1', 'score_fold_6', 'mean_score', 'std_score']].to_csv(
        exp_run / 'ensemble_ranked_predictions_allfolds_avg.csv', index=False)

# aggregate the average scores from all runs
prad_factor_arr = [0.5, 1, 1.5, 2, 5, 10, 15, 20, 25, 30]
exp_runs = [exp_dir_root / f'prad-f-{prad_f}' for prad_f in prad_factor_arr]
for exp_run_i, exp_run in enumerate(exp_runs):
    prad_f = exp_run.name.split('-')[-1]
    tbl = pd.read_csv(exp_run / 'ensemble_ranked_predictions_allfolds_avg.csv')
    tbl = tbl.rename(columns={'mean_score': f'{prad_f}'})

    if exp_run_i == 0:
        tbl = tbl[['target_id', 'tce_plnt_num', 'label', 'tce_period', 'tce_duration', 'tce_time0bk', 'original_label',
                   'transit_depth', f'{prad_f}']]
        runs_tbl = tbl
    else:
        tbl = tbl[['target_id', 'tce_plnt_num', f'{prad_f}']]
        runs_tbl = runs_tbl.merge(tbl, on=['target_id', 'tce_plnt_num'], validate='one_to_one')

# add planet radius
tce_tbl = pd.read_csv('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/data/ephemeris_tables/kepler/q1-q17_dr25/11-17-2021_1243/q1_q17_dr25_tce_2020.09.28_10.36.22_stellar_koi_cfp_norobovetterlabels_renamedcols_nomissingval_symsecphase_cpkoiperiod_rba_cnt0n_valpc_modelchisqr_ruwe_magcat_uid.csv')
runs_tbl = runs_tbl.merge(tce_tbl[['target_id', 'tce_plnt_num', 'tce_prad', 'koi_prad']],
                          on=['target_id', 'tce_plnt_num'], validate='one_to_one')

runs_tbl.to_csv(exp_dir_root / 'ensemble_ranked_predicitons_allruns_avg.csv', index=False)

