"""
Filter lc FITS files for KICs for which TCEs were not detected in Q1-Q17 DR25 observed, but in the inverted and
scrambled runs.
"""

# 3rd party
from pathlib import Path
import pandas as pd
import logging
import shutil

# src_root_dir = Path('/data5/tess_project/Data/Kepler-Q1-Q17-DR25/lc/dr_25_all_final')
src_root_dir = Path('/Users/msaragoc/Projects/exoplanet_transit_classification/data/fits_files/kepler/q1_q17_dr25/lc')
dest_dir = src_root_dir.parent / 'lc_obs_inv_scr'
dest_dir.mkdir(exist_ok=True)

# set up logger
logger = logging.getLogger(name=f'filter_kics')
logger_handler = logging.FileHandler(filename=src_root_dir.parent / f'filter_kics_scr_inv_8-14-2023_1523.log', mode='w')
logger_formatter = logging.Formatter('%(asctime)s - %(message)s')
logger.setLevel(logging.INFO)
logger_handler.setFormatter(logger_formatter)
logger.addHandler(logger_handler)
logger.info(f'Starting run...')

# get only targets with TCEs in the observed, inverted and scrambled runs of Q1-Q17 DR25
# obs_tce_tbl = pd.read_csv('')
sim_tce_tbl = pd.read_csv('/Users/msaragoc/Projects/exoplanet_transit_classification/data/ephemeris_tables/kepler/q1-q17_dr25/simulated_data/dvOutputMatrix_allruns_updtstellar_final_7-24-2023_1548.csv')
# filter plti targets from the simulated TCE table
sim_tce_tbl = sim_tce_tbl.loc[sim_tce_tbl['dataset'].isin(['INV', 'SCR1', 'SCR2'])]
# filter targets that are in the observed TCE table
obs_tce_tbl = pd.read_csv('/Users/msaragoc/Projects/exoplanet_transit_classification/data/ephemeris_tables/kepler/q1-q17_dr25/11-17-2021_1243/q1_q17_dr25_tce_3-6-2023_1734.csv')
sim_tce_tbl = sim_tce_tbl.loc[~sim_tce_tbl['target_id'].isin(obs_tce_tbl['target_id'])]

# kics_valid = pd.concat([obs_tce_tbl['target_id'],sim_tce_tbl['target_id']], axis=0)['target_id'].unique()
kics_valid = sim_tce_tbl['target_id'].unique()
# kics_valid = obs_tce_tbl['target_id'].unique()

logger.info(f'Found {len(kics_valid)} KIC targets in DV TCE table.')

# copy FITS files of KICs found to destination folder
for kic_i, kic in enumerate(kics_valid):

    kic_str = f'{kic}'.zfill(9)

    src_path = src_root_dir / kic_str[:4] / kic_str
    dest_path = dest_dir / kic_str[:4] / kic_str

    if not src_path.exists():
        logger.info(f'No files found for KIC {kic} in {src_path}')
        continue

    shutil.copytree(src_path, dest_path)  # , dirs_exist_ok=True)

    if kic_i % 1000 == 0:
        logger.info(f'Iterated through {kic_i + 1} FITS files out of {len(kics_valid)}.')

logger.info(f'Finished copying FITS files.')
