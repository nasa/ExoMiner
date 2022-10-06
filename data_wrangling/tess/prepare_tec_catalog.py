"""
Prepare TEC flux triage catalog to be used for:
1) labeling TCEs as NTPs to build an NTP set for TESS;
2) matching secondaries with primaries.
"""

# 3rd party
import pandas as pd
from pathlib import Path
import logging

# %% Processing TEC flux triage tables

tec_dir = Path('/data5/tess_project/Data/Ephemeris_tables/TESS/TEC_SPOC')

tec_tbls = []
tec_tbls_dirs = [fp for fp in tec_dir.iterdir() if fp.is_dir()]
for run_dir in tec_tbls_dirs:
    print(f'Iterating over {run_dir}')
    triage0_fps = [fp for fp in run_dir.iterdir() if 'fluxtriage' in fp.stem]
    if len(triage0_fps) == 0:
        print(f'No triage file for {run_dir}')
        continue
    assert len(triage0_fps) == 1
    triage0_fp = triage0_fps[0]
    print(f'Reading table {triage0_fp}')

    tec_tbl = pd.read_csv(triage0_fp, names=['target_id', 'tce_plnt_num', 'pass', 'comment'], sep=r'\s+')

    sector_run = [el for el in triage0_fp.stem.split('_') if 'sector' in el][0][6:]
    print(f'Sector run: {sector_run}')
    tec_tbl['sector_run'] = sector_run
    tec_tbls.append(tec_tbl)

tec_tbl_full = pd.concat(tec_tbls)
tec_tbl_full.rename(columns={'pass': 'tec_fluxtriage_pass', 'comment': 'tec_fluxtriage_comment'}, inplace=True)
tec_tbl_full.to_csv(
    '/data5/tess_project/Data/Ephemeris_tables/TESS/DV_SPOC_mat_files/11-29-2021/tess_tces_s1-s40_11-23-2021_1409_stellarparams_updated.csv',
    index=False)

# tec_tbl_full['id'] = tec_tbl_full.apply(lambda x: f'{x["target_id"]}_{x["tce_plnt_num"]}_{x["sector_run"]}', axis=1)

#%% Adding TEC results to TCE table

res_dir = Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/data/ephemeris_tables/tess/DV_SPOC_mat_files/10-05-2022_1338')

# set up logger
logger = logging.getLogger(name='add_tec_to_tce_tbl')
logger_handler = logging.FileHandler(filename=res_dir / f'add_tec_to_tce_tbl.log', mode='w')
logger_formatter = logging.Formatter('%(asctime)s - %(message)s')
logger.setLevel(logging.INFO)
logger_handler.setFormatter(logger_formatter)
logger.addHandler(logger_handler)
logger.info(f'Starting run...')

tec_tbl_fp = Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/data/ephemeris_tables/tess/TEC_SPOC/tec_tbl_fluxtriage_s1-s41_10-29-2021.csv')
tec_tbl = pd.read_csv(tec_tbl_fp)
logger.info(f'Loaded TEC table from {tec_tbl_fp}')
logger.info(f'Number of TCEs from TEC: {len(tec_tbl)}')

tce_tbl_fp = res_dir / 'tess_tces_dv_s1-s55_10-05-2022_1338_ticstellar_ruwe.csv'
tce_tbl = pd.read_csv(tce_tbl_fp)
logger.info(f'Loaded TCE table from {str(tce_tbl_fp)}')

tce_tbl = tce_tbl.merge(tec_tbl, on=['target_id', 'tce_plnt_num', 'sector_run'], how='left', validate='one_to_one')

logger.info(f'Number of TCEs with TEC results: {tce_tbl["tec_fluxtriage_pass"].isna().sum()} out of {len(tce_tbl)} TCEs.')
logger.info(f'Counts for flux triage:\n{tce_tbl["tec_fluxtriage_pass"].value_counts()}')
logger.info(f'Counts for flux triage comment:\n{tce_tbl["tec_fluxtriage_comment"].value_counts()}')

tce_tbl.to_csv(res_dir / f'{tce_tbl_fp.stem}_tec.csv', index=False)
logger.info('Saved TCE table with TEC results')
