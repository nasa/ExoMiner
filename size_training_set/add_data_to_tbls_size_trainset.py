""" Add information to training and validation set tables used in the training set size experiments. """

# 3rd party
from pathlib import Path
import pandas as pd

# %% add dispositions

disp_tbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17_DR25/'
                       '19-08-21_07:21/q1_q17_dr25_tce_2020.09.28_10.36.22_stellar_koi_cfp_norobovetterlabels_renamedcols_nomissingval_symsecphase_cpkoiperiod_rba_cnt0n.csv')

dataset_tbls_dir = Path('/data5/tess_project/Data/tfrecords/Kepler/Q1-Q17_DR25/'
                        'tfrecordskeplerq1q17dr25-dv_g301-l31_5tr_spline_nongapped_all_features_paper_rbat0norm_8-20-2021_data/tfrecordskeplerq1q17dr25-dv_g301-l31_5tr_spline_nongapped_all_features_paper_rbat0norm_8-20-2021_starshuffle_experiment-labels/')

trainset_tbl = pd.read_csv(dataset_tbls_dir / 'train_set_shards.csv')
trainset_tbl = trainset_tbl.merge(
    disp_tbl[['target_id', 'tce_plnt_num', 'label', 'koi_disposition', 'fpwg_disp_status', 'kepoi_name']],
    on=['target_id', 'tce_plnt_num'])
trainset_tbl.to_csv(dataset_tbls_dir / 'train_set_shards_disp.csv', index=False)

valset_tbl = pd.read_csv(dataset_tbls_dir / 'val_set_shards.csv')
valset_tbl = valset_tbl.merge(
    disp_tbl[['target_id', 'tce_plnt_num', 'label', 'koi_disposition', 'fpwg_disp_status', 'kepoi_name']],
    on=['target_id', 'tce_plnt_num'])
valset_tbl.to_csv(dataset_tbls_dir / 'val_set_shards_disp.csv', index=False)

# %% add column that shows which TCEs are secondary matched TCEs

disp_tbl = pd.read_csv(
    '/home/msaragoc/Projects/Kepler-TESS_exoplanet/Analysis/wks_tce_match/19-08-21_06:09/matched_tces_phasematchthr_0.05_periodmatchthr_0.001_19-08-21_06:09_final.csv')
disp_tbl['is_secondary_tce'] = True
disp_tbl['tce_plnt_num'] = disp_tbl['secondary_tce']
disp_tbl['UID'] = disp_tbl[['target_id', 'tce_plnt_num']].apply(
    lambda x: '{}-{}'.format(x['target_id'], x['tce_plnt_num']), axis=1)

dataset_tbls_dir = Path('/data5/tess_project/Data/tfrecords/Kepler/Q1-Q17_DR25/'
                        'tfrecordskeplerq1q17dr25-dv_g301-l31_5tr_spline_nongapped_all_features_paper_rbat0norm_8-20-2021_data/tfrecordskeplerq1q17dr25-dv_g301-l31_5tr_spline_nongapped_all_features_paper_rbat0norm_8-20-2021_starshuffle_experiment-labels/')

trainset_tbl = pd.read_csv(dataset_tbls_dir / 'train_set_shards_disp.csv')
trainset_tbl['is_secondary_tce'] = False
trainset_tbl['UID'] = trainset_tbl[['target_id', 'tce_plnt_num']].apply(
    lambda x: '{}-{}'.format(x['target_id'], x['tce_plnt_num']), axis=1)
trainset_tbl.loc[trainset_tbl['UID'].isin(disp_tbl['UID']), 'is_secondary_tce'] = True
trainset_tbl = trainset_tbl.drop(columns='UID')
trainset_tbl.to_csv(dataset_tbls_dir / 'train_set_shards_disp_sectce.csv', index=False)

valset_tbl = pd.read_csv(dataset_tbls_dir / 'val_set_shards_disp.csv')
valset_tbl['is_secondary_tce'] = False
valset_tbl['UID'] = valset_tbl[['target_id', 'tce_plnt_num']].apply(
    lambda x: '{}-{}'.format(x['target_id'], x['tce_plnt_num']), axis=1)
valset_tbl.loc[valset_tbl['UID'].isin(disp_tbl['UID']), 'is_secondary_tce'] = True
valset_tbl = valset_tbl.drop(columns='UID')
valset_tbl.to_csv(dataset_tbls_dir / 'val_set_shards_disp_sectce.csv', index=False)
