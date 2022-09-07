"""
Matching TCEs with TOIs based on ephemeris matching results from DV.
"""

# 3rd party
import pandas as pd
from pathlib import Path
import logging
import matplotlib.pyplot as plt
import numpy as np

#%%

tce_tbl_fp = Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/data/ephemeris_tables/tess/DV_SPOC_mat_files/11-29-2021/tess_tces_s1-s40_11-23-2021_1409_stellarparams_updated_eb_tso_tec_label_modelchisqr_astronet_ruwe_magcat_uid_corrtsoebs_corraltdetfail.csv')
tce_tbl = pd.read_csv(tce_tbl_fp)
dv_mat_root_dir = Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/data/ephemeris_tables/tess/DV_SPOC_mat_files/')
sngl_sector_dir = dv_mat_root_dir / 'single-sector' / 'csv_tables'
multi_secyot_dir = dv_mat_root_dir / 'multi-sector' / 'csv_tables'
sector_dir = sorted(list(sngl_sector_dir.iterdir())) + sorted(list(multi_secyot_dir.iterdir()))

matching_thr_dv = 0.75

save_dir = Path('/Users/msaragoc/Downloads/toi-tce_matching_dv')
save_dir.mkdir(exist_ok=True)

# set up logger
logger = logging.getLogger()
logger_handler = logging.FileHandler(filename=save_dir / f'match_dv_toi_tce_tbl.log', mode='w')
logger_formatter = logging.Formatter('%(asctime)s - %(message)s')
logger.setLevel(logging.INFO)
logger_handler.setFormatter(logger_formatter)
logger.addHandler(logger_handler)
logger.info(f'Starting run...')

logger.info(f'Using TCE table: {tce_tbl_fp}')

tce_tbl = tce_tbl.set_index('uid')

tce_tbl[['toi_dv', 'toi_dv_corr']] = np.nan, np.nan
sector_cnts_dict = {'sector_run': [], 'num_targets': [], 'num_tces': [], 'num_tois': []}
for sector_run_tbl_fp in sector_dir:

    sector_run = sector_run_tbl_fp.stem[14:]
    if 'single-sector' in str(sector_run_tbl_fp):
        sector_run = int(sector_run)
    else:
        s_sector, e_sector = int(sector_run[:2]), int(sector_run[2:])
        sector_run = f'{s_sector}-{e_sector}'

    logger.info(f'Iterating through DV TCE table for sector run {sector_run}')

    sector_cnts_dict['sector_run'].append(sector_run)

    sector_run_tbl = pd.read_csv(sector_run_tbl_fp)
    sector_cnts_dict['num_tces'].append(len(sector_run_tbl))
    sector_cnts_dict['num_targets'].append(len(sector_run_tbl['catId'].unique()))

    sector_run_tbl = sector_run_tbl.astype({'catId': 'int32', 'planetIndexNumber': 'int32'})

    # filter for TCEs matched to TOIs
    sector_run_tbl = sector_run_tbl.loc[sector_run_tbl['planetToiCorrelation'] > matching_thr_dv]
    num_tces_matched_tois = len(sector_run_tbl)  # sector_run_tbl["targetToiId"] != -1.0).sum()

    sector_cnts_dict['num_tois'].append(num_tces_matched_tois)

    logger.info(f'Sector {sector_run} {len(sector_run_tbl)} TCEs. {num_tces_matched_tois} TCEs matched to TOIs.')
    if num_tces_matched_tois == 0:
        continue
    sector_run_tbl['sector_run'] = sector_run
    sector_run_tbl['uid'] = \
        sector_run_tbl[['catId', 'planetIndexNumber', 'sector_run']].apply(
            lambda x: f'{x["catId"]}-{x["planetIndexNumber"]}-S{x["sector_run"]}', axis=1)

    sector_run_tbl = sector_run_tbl[['uid', 'planetToiId', 'planetToiCorrelation']]
    sector_run_tbl.rename(columns={'planetToiId': 'toi_dv', 'planetToiCorrelation': 'toi_dv_corr'}, inplace=True)
    sector_run_tbl = sector_run_tbl.set_index('uid')
    # tce_tbl[['toi_dv', 'toi_dv_corr']].update(sector_run_tbl[['toi_dv', 'toi_dv_corr']])
    tce_tbl.update(sector_run_tbl)

tce_tbl.reset_index(inplace=True)

logger.info(f'Number of TCEs in the TCE table matched with TOIs using DV matching: {(~tce_tbl["toi_dv"].isna()).sum()}')

tce_tbl.to_csv(save_dir / tce_tbl_fp.name, index=False)
sector_cnts_tbl = pd.DataFrame(sector_cnts_dict)
sector_cnts_tbl.to_csv(save_dir / 'sector_run_counts.csv', index=False)

#%% plot dv matching score against my matching distance

tce_tbl_tois = tce_tbl.loc[tce_tbl['match_dist'] < 1]

f, ax = plt.subplots()
ax.scatter(1 - tce_tbl_tois['match_dist'], tce_tbl_tois['toi_dv_corr'], s=8)
ax.set_xlabel('Template Ephemeris Matching Score')
ax.set_ylabel('DV Pearson Correlation Score')
ax.set_xlim([0, 1])
ax.set_ylim([0.75, 1])
f.savefig(save_dir / 'scatter_template_ephem_match-vs-dv_pearson_corr.png')

#%%

# cases I matched but DV didn't
# ((tce_tbl['match_dist'] < 0.25) & (tce_tbl['toi_dv_corr'] < 0.75)).sum()

# cases I didn't match but DV did
((tce_tbl['match_dist'] >= 0.25) & (tce_tbl['toi_dv_corr'] >= 0.75)).sum()
