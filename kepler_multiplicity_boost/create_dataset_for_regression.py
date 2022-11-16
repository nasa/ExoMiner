"""
Create dataset for running ML models for Kepler multiplicity boots experiments.
"""

# 3rd party
import sys
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime

#%% Experiment initial setup

res_root_dir = Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/experiments/kepler_multiplicity_boost')
res_dir = res_root_dir / f'dataset_for_ml_{datetime.now().strftime("%m-%d-%Y_%H%M")}'
res_dir.mkdir(exist_ok=True)

# set up logger
logger = logging.getLogger(name='kepler_multiplicity_boost_run')
logger_handler = logging.FileHandler(filename=res_dir / f'kepler_multiplicity_boost_run.log', mode='w')
logger_formatter = logging.Formatter('%(asctime)s - %(message)s')
logger.setLevel(logging.INFO)
logger_handler.setFormatter(logger_formatter)
logger.addHandler(logger_handler)
handler = logging.StreamHandler(sys.stdout)
logger.addHandler(handler)
logger.info(f'Starting run...')

#%% Create catalog dataset to be used

# using TCE table
# tce_tbl_fp = Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/data/ephemeris_tables/kepler/q1-q17_dr25/11-17-2021_1243/q1_q17_dr25_tce_2020.09.28_10.36.22_stellar_koi_cfp_norobovetterlabels_renamedcols_nomissingval_symsecphase_cpkoiperiod_rba_cnt0n_valpc_modelchisqr_ruwe_magcat_uid_rv_posprob.csv')
# tce_tbl = pd.read_csv(tce_tbl_fp)
# logger.info(f'Using TCE table from: {tce_tbl_fp}')
#
# # # set planets with RUWE > 1.2 and not validated by RV to UNKs
# # tce_tbl.loc[(tce_tbl['ruwe'] > 1.2) & (tce_tbl['RV_status'] == 0) & (tce_tbl['label'] == 'PC'), 'label'] = 'UNK'
# # logger.info('Set planets with RUWE > 1.2 and not validated by RV to UNK TCEs.')
#
# # remove rogue TCEs
# tce_tbl = tce_tbl.loc[tce_tbl['tce_rogue_flag'] == 0]
# logger.info('Removed rogue TCEs.')

# using ranking table instead of TCE table
tce_tbl_fp = Path('/Users/msaragoc/Downloads/ranking_comparison_with_paper_12-18-2020_merged_ra_dec_prad_CV_v20 (1).csv')
tce_tbl = pd.read_csv(tce_tbl_fp)
logger.info(f'Using TCE table from: {tce_tbl_fp}')
# remove examples not used in the data set for the ExoMiner 2021 paper
tce_tbl = tce_tbl.loc[~((tce_tbl['original_label'] == 'NTP') & (tce_tbl['dataset'] == 'not_used'))]
tce_tbl = tce_tbl.loc[~((tce_tbl['original_label'] == 'PC') & (tce_tbl['dataset'] == 'not_used'))]
tce_tbl.drop(columns='label', inplace=True)
tce_tbl = tce_tbl.rename(columns={'original_label': 'label'})
logger.info(f'Labels of examples: \n{tce_tbl["label"].value_counts()}')

#%% count different quantities needed for estimates that are plugged into the statistical framework

tbls = []

# number of TCEs per target in the TCE table
cnt_tces_target = \
    tce_tbl['target_id'].value_counts().to_frame(name='num_tces_target').reset_index().rename(columns={'index': 'target_id'})
tbls.append(cnt_tces_target)
# number of KOIs per target in the TCE table
cnt_kois_target = \
    tce_tbl.loc[~tce_tbl['kepoi_name'].isna(), 'target_id'].value_counts().to_frame(name='num_kois_target').reset_index().rename(columns={'index': 'target_id'})
tbls.append(cnt_kois_target)
# number of FPs per target (AFPs in the TCE table)
cnt_fps_target = \
    tce_tbl.loc[((tce_tbl['label'] == 'AFP') | ((tce_tbl['label'] == 'NTP') & (tce_tbl['fpwg_disp_status'] == 'CERTIFIED FA'))), 'target_id'].value_counts().to_frame(name='num_fps_target').reset_index().rename(columns={'index': 'target_id'})
tbls.append(cnt_fps_target)
# number of Confirmed Planets (PCs in the TCE table)
cnt_planets_target = \
    tce_tbl.loc[tce_tbl['label'] == 'PC', 'target_id'].value_counts().to_frame(name='num_planets_target').reset_index().rename(columns={'index': 'target_id'})
tbls.append(cnt_planets_target)

for tbl in tbls:
    tce_tbl = tce_tbl.merge(tbl, on=['target_id'], how='left', validate='many_to_one')

count_lbls = [
    'num_kois_target',
    'num_fps_target',
    'num_planets_target',
]
for count_lbl in count_lbls:
    tce_tbl.loc[tce_tbl[count_lbl].isna(), count_lbl] = 0

logger.info(f'Number of TCEs per KIC:\n{tce_tbl["num_tces_target"].value_counts().sort_index()}')
logger.info(f'Number of KOIs per KIC:\n{tce_tbl["num_kois_target"].value_counts().sort_index()}')
logger.info(f'Number of Planets per KIC:\n{tce_tbl["num_planets_target"].value_counts().sort_index()}')
logger.info(f'Number of FPs per KIC:\n{tce_tbl["num_fps_target"].value_counts().sort_index()}')

tce_tbl.to_csv(res_dir / f'{tce_tbl_fp.stem}_counts.csv', index=False)

#%% Check counts

for tce_i, tce in tce_tbl.iterrows():

    if tce['num_tces_target'] < tce['num_kois_target']:
        print(f'[{tce["uid"]}] Number of TCEs ({tce["num_tces_target"]}) is '
              f'smaller than number of KOIs ({tce["num_kois_target"]}).')

    if tce['num_kois_target'] < tce['num_fps_target'] + tce['num_planets_target']:
        print(f'[{tce["uid"]}] Number of KOIs ({tce["num_kois_target"]}) is '
              f'smaller than number of FPs+Planets ({tce["num_fps_target"] + tce["num_planets_target"]}).')
