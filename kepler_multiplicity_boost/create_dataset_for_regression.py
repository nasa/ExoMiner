"""
Create dataset for running ML models for Kepler multiplicity boots experiments.
"""

# 3rd party
import sys
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime

# local

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

tce_tbl_fp = Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/data/ephemeris_tables/kepler/q1-q17_dr25/11-17-2021_1243/q1_q17_dr25_tce_2020.09.28_10.36.22_stellar_koi_cfp_norobovetterlabels_renamedcols_nomissingval_symsecphase_cpkoiperiod_rba_cnt0n_valpc_modelchisqr_ruwe_magcat_uid.csv')
tce_tbl = pd.read_csv(tce_tbl_fp)
logger.info(f'Using TCE table from: {tce_tbl_fp}')

#%% count different quantities needed for estimates that are plugged into the statistical framework

tbls = []

# number of TCEs per target in the TCE table
cnt_tces_target = \
    tce_tbl['target_id'].value_counts().to_frame(name='num_tces_target').reset_index().rename(columns={'index': 'target_id'})
tbls.append(cnt_tces_target)
# number of FPs per target (AFPs in the TCE table)
cnt_fps_target = \
    tce_tbl.loc[(tce_tbl['label'] == 'AFP') | tce_tbl['fpwg_disp_status'] == 'CERTIFIED FA', 'target_id'].value_counts().to_frame(name='num_fps_target').reset_index().rename(columns={'index': 'target_id'})
tbls.append(cnt_fps_target)
# number of Confirmed Planets (PCs in the TCE table)
cnt_planets_target = \
    tce_tbl.loc[tce_tbl['label'] == 'PC', 'target_id'].value_counts().to_frame(name='num_planets_target').reset_index().rename(columns={'index': 'target_id'})
tbls.append(cnt_planets_target)

for tbl in tbls:
    tce_tbl = tce_tbl.merge(tbl, on=['target_id'], how='left', validate='many_to_one')

count_lbls = [
    'num_fps_target',
    'num_planets_target',
]
for count_lbl in count_lbls:
    tce_tbl.loc[tce_tbl[count_lbl].isna(), count_lbl] = 0

tce_tbl.to_csv(res_dir / f'{tce_tbl_fp.stem}_counts.csv', index=False)
