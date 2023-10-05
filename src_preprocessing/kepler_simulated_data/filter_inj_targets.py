"""
Filter lc FITS files for KICs for which no TCEs were detected in a given injection group run.
"""

# 3rd party
from pathlib import Path
import pandas as pd
import logging
import shutil
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--inj_grp', type=str, help='Injected group folder name `inj1`, `inj2`, or `inj3`', default=None)
args = parser.parse_args()

# src_root_dir = Path('/Users/msaragoc/Projects/exoplanet_transit_classification/data/fits_files/kepler/q1_q17_dr25_plti/')
src_root_dir = Path('/data5/tess_project/Data/Kepler-Q1-Q17-DR25/injected/pixel_level_injection/')
inj_grp = args.inj_grp  # 'inj1'  # choose one injection group
dest_dir = Path('/Users/msaragoc/Projects/exoplanet_transit_classification/data/fits_files/kepler/q1_q17_dr25_plti/inj1_filtered/')
dest_dir = Path(f'/data5/tess_project/Data/Kepler-Q1-Q17-DR25/injected/pixel_level_injection/{inj_grp}_filtered')
dest_dir.mkdir(exist_ok=True)

# set up logger
logger = logging.getLogger(name=f'filter_kics_{inj_grp}')
logger_handler = logging.FileHandler(filename=src_root_dir / f'filter_kics_for_{inj_grp}.log', mode='w')
logger_formatter = logging.Formatter('%(asctime)s - %(message)s')
logger.setLevel(logging.INFO)
logger_handler.setFormatter(logger_formatter)
logger.addHandler(logger_handler)
logger.info(f'Starting run for injection group {inj_grp}...')

inj_root_dir = src_root_dir / inj_grp

# get only targets for the specific injection group
# simulated_tbl_fp = Path('/Users/msaragoc/Projects/exoplanet_transit_classification/data/ephemeris_tables/kepler/q1-q17_dr25/simulated_data/dvOutputMatrix_allruns_updtstellar_final_7-24-2023_1548.csv')
simulated_tbl_fp = Path('/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17_DR25/simulated_data/dvOutputMatrix_allruns_updtstellar_final_7-24-2023_1548.csv')
logger.info(f'Using DV TCE table from {simulated_tbl_fp}.')
simulated_tbl = pd.read_csv(simulated_tbl_fp)
kics_inj = simulated_tbl.loc[simulated_tbl['dataset'] == f'{inj_grp.upper()}', 'target_id'].unique()

logger.info(f'Found {len(kics_inj)} KIC targets in DV TCE table.')

all_kics_fps = [fp for fp in inj_root_dir.iterdir() if fp.suffix == '.fits']
kics_found = np.unique([int(fp.stem.split('_')[0].split('-')[0][4:]) for fp in all_kics_fps])
logger.info(f'Found {len(all_kics_fps)} FITS files for a total of {len(kics_found)} KICs.')

kics_in_inj_fps = [fp for fp in all_kics_fps if int(fp.stem.split('_')[0].split('-')[0][4:]) in kics_inj]
kics_in_inj_found = np.unique([int(fp.stem.split('_')[0].split('-')[0][4:]) for fp in kics_in_inj_fps])
logger.info(f'Found {len(kics_in_inj_fps)} FITS files that match a total of {len(kics_in_inj_found)} KICs '
            f'in the DV TCE table.')

# copy FITS files of KICs found to destination folder
for fp_i, fp in enumerate(kics_in_inj_fps):
    shutil.copy(fp, dest_dir / fp.name)
    if fp_i % 1000 == 0:
        logger.info(f'Copied {fp_i + 1} FITS files out of {len(kics_in_inj_fps)}.')

logger.info(f'Finished copying FITS files.')
