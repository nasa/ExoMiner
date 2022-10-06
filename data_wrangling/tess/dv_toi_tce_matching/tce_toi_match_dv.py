"""
Matching TCEs with TOIs based on ephemeris matching results from DV.
"""

# 3rd party
import pandas as pd
from pathlib import Path
import logging
import matplotlib.pyplot as plt
import numpy as np

#%% Matching TCEs with TOIs using the TOI matching produced by Joe at the end of each DV sector run

tce_tbl_fp = Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/data/ephemeris_tables/tess/DV_SPOC_mat_files/11-29-2021/tess_tces_s1-s40_11-23-2021_1409_stellarparams_updated_eb_tso_tec_label_modelchisqr_astronet_ruwe_magcat_uid_corrtsoebs_corraltdetfail.csv')
tce_tbl = pd.read_csv(tce_tbl_fp)
dv_mat_root_dir = Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/data/ephemeris_tables/tess/DV_SPOC_mat_files/')
sngl_sector_dir = dv_mat_root_dir / 'single-sector' / 'csv_tables'
multi_secyot_dir = dv_mat_root_dir / 'multi-sector' / 'csv_tables'
sector_dir = sorted(list(sngl_sector_dir.iterdir())) + sorted(list(multi_secyot_dir.iterdir()))

matching_thr_dv = 0.75

save_dir = Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/data/ephemeris_matching/toi-tce_matching_dv/')
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

tce_tbl.to_csv(save_dir / f'{tce_tbl_fp.stem}_toidv.csv', index=False)
sector_cnts_tbl = pd.DataFrame(sector_cnts_dict)
sector_cnts_tbl.to_csv(save_dir / 'sector_run_counts.csv', index=False)

#%% Fixing bug for sector runs S8 and S9; TOIs IDs are missing the first digit

logger = logging.getLogger()
logger_handler = logging.FileHandler(filename=save_dir / f'fix_bug_sectors_8_9.log', mode='w')
logger_formatter = logging.Formatter('%(asctime)s - %(message)s')
logger.setLevel(logging.INFO)
logger_handler.setFormatter(logger_formatter)
logger.addHandler(logger_handler)
logger.info(f'Starting run...')

toi_tbl = pd.read_csv(save_dir / 'exofop_toi_cat_9-7-2022.csv')
tce_tbl = pd.read_csv(save_dir / 'tess_tces_s1-s40_11-23-2021_1409_stellarparams_updated_eb_tso_tec_label_modelchisqr_astronet_ruwe_magcat_uid_corrtsoebs_corraltdetfail_toidv.csv')

tce_tbl_dir = Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/data/ephemeris_tables/tess/DV_SPOC_mat_files/10-05-2022_1338')
toi_tbl = pd.read_csv('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/data/ephemeris_tables/tess/EXOFOP_TOI_lists/TOI/10-3-2022/exofop_tess_tois-4.csv')
for tce_tbl_fp in [fp for fp in tce_tbl_dir.iterdir() if fp.name.startswith('tess_tces')]:
    print(f'TCE tbl: {tce_tbl_fp}')
    tce_tbl = pd.read_csv(tce_tbl_fp)
    # tce_tbl = tce_tbl.rename(columns={'planetToiId': 'toi_dv', 'planetToiCorrelation': 'toi_dv_corr'})
    for tce_i, tce in tce_tbl.iterrows():
        if tce['sector_run'] in ['8', '9', '1-9'] and tce['toi_dv'] != -1:  # ~np.isnan(tce['toi_dv']):
            # print(tce)
            # logger.info(f' #### Iterating over TCE TIC {tce["uid"]}... ####')
            print(f' #### Iterating over TCE TIC {tce["uid"]}... ####')
            tce_toi_plnt_num = int(str(tce['toi_dv']).split('.')[1])
            tois_in_tic = toi_tbl.loc[toi_tbl['TIC ID'] == tce['target_id']]
            tois_in_tic['toi_plnt_num'] = tois_in_tic['TOI'].astype(str).str.extract('\.(.*)').astype(int)
            tois_in_tic_same_plnt_num = tois_in_tic['toi_plnt_num'] == tce_toi_plnt_num
            assert tois_in_tic_same_plnt_num.sum() == 1
            # logger.info(f'Number of TOIs in TIC with same TOI planet number: {len(tois_in_tic_same_plnt_num)}')
            print(f'Number of TOIs in TIC with same TOI planet number: {len(tois_in_tic_same_plnt_num)}')
            toi_for_tce = tois_in_tic.loc[tois_in_tic_same_plnt_num]
            # logger.info(f'Current TCE TOI ID: {tce["toi_dv"]} | TOI ID from TOI catalog: {toi_for_tce["TOI"].values[0]} {toi_for_tce["TFOPWG Disposition"].values[0]}')
            print(f'Current TCE TOI ID: {tce["toi_dv"]} | TOI ID from TOI catalog: {toi_for_tce["TOI"].values[0]} {toi_for_tce["TFOPWG Disposition"].values[0]}')
            tce_tbl.loc[tce_i, 'toi_dv'] = toi_for_tce['TOI'].values[0]
    # aaa
    tce_tbl.to_csv(tce_tbl_fp, index=False)

tce_tbl.to_csv(save_dir / 'tess_tces_s1-s40_11-23-2021_1409_stellarparams_updated_eb_tso_tec_label_modelchisqr_astronet_ruwe_magcat_uid_corrtsoebs_corraltdetfail_toidv.csv', index=False)

#%% Update labels for TCEs associated with TOIs

tce_tbl = pd.read_csv(save_dir / 'tess_tces_s1-s40_11-23-2021_1409_stellarparams_updated_eb_tso_tec_label_modelchisqr_astronet_ruwe_magcat_uid_corrtsoebs_corraltdetfail_toidv.csv')
toi_tbl = pd.read_csv(save_dir / 'exofop_toi_cat_9-7-2022.csv')

toi_cols = [
    'TFOPWG Disposition',
    'TESS Disposition',
    'Period (days)',
    'Duration (hours)',
    'Depth (ppm)',
    # 'Epoch(TBJD)',
    'Epoch (BJD)',
    'Planet Radius (R_Earth)',
    'Comments'
]
tce_tbl.drop(columns=toi_cols, inplace=True, errors='ignore')
tce_tbl.drop(columns=['toi_sectors'])
toi_tbl.rename(columns={'TOI': 'toi_dv'}, inplace=True)
tce_tbl = tce_tbl.merge(toi_tbl[['toi_dv'] + toi_cols], on=['toi_dv'], how='left', validate='many_to_one')

tce_tbl['label_source'] = 'N/A'
tce_tbl['label'] = tce_tbl['TFOPWG Disposition']
tce_tbl.loc[~tce_tbl['label'].isna(), 'label_source'] = 'TFOPWG Disposition'
tce_tbl.loc[tce_tbl['label'].isna(), 'label'] = 'UNK'

tce_tbl.to_csv(save_dir / 'tess_tces_s1-s40_11-23-2021_1409_stellarparams_updated_eb_tso_tec_label_modelchisqr_astronet_ruwe_magcat_uid_corrtsoebs_corraltdetfail_toidv.csv', index=False)

#%% Update some TCE ephemerides and parameters with TOI parameters

tce_tbl = pd.read_csv(save_dir / 'tess_tces_s1-s40_11-23-2021_1409_stellarparams_updated_eb_tso_tec_label_modelchisqr_astronet_ruwe_magcat_uid_corrtsoebs_corraltdetfail_toidv.csv')

# plot TCE vs TOI parameters
aux_tce_tbl = tce_tbl.loc[~tce_tbl['toi_dv'].isna()]
for param_pair in [('tce_period', 'Period (days)'), ('tce_duration', 'Duration (hours)'), ('tce_depth', 'Depth (ppm)'), ('tce_prad', 'Planet Radius (R_Earth)')]:

    f, ax = plt.subplots()
    ax.scatter(aux_tce_tbl[param_pair[0]], aux_tce_tbl[param_pair[1]], s=8)
    ax.set_xlabel(param_pair[0])
    ax.set_ylabel(param_pair[1])
    if param_pair[0] == 'tce_depth':  #  or param_pair[0] == 'tce_period':
        ax.set_xscale('log')
        ax.set_yscale('log')
    else:
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
    f.savefig(save_dir / f'scatter_{param_pair[0]}_{param_pair[1]}.png')
    if param_pair[0] == 'tce_period':
        ax.set_xlim([0, 50])
        ax.set_ylim([0, 50])
        f.savefig(save_dir / f'scatter_{param_pair[0]}_{param_pair[1]}_shortperiodrange.png')
    plt.close()

# update TCE with TOI parameters
cols_of_interest = ['uid',
                    'toi_dv',
                    'target_id',
                    'tce_period',
                    'Period (days)',
                    'tce_duration',
                    'Duration (hours)',
                    'tce_depth',
                    'Depth (ppm)',
                    'tce_prad',
                    'Planet Radius (R_Earth)'
                    ]
data_aux_tbl = {col: [] for col in cols_of_interest}
data_aux_tbl.update({col: [] for col in ['period_diff_rel', 'duration_diff_rel', 'depth_diff_rel', 'prad_diff_rel']})
for tce_i, tce in tce_tbl.iterrows():

    if ~np.isnan(tce['toi_dv']):

        for col in cols_of_interest:
            data_aux_tbl[col].append(tce_tbl.loc[tce_i, col])
        data_aux_tbl['period_diff_rel'].append(np.abs(tce_tbl.loc[tce_i, 'tce_period'] - tce['Period (days)']) / tce['Period (days)'])
        tce_tbl.loc[tce_i, 'tce_period'] = tce['Period (days)']
        data_aux_tbl['duration_diff_rel'].append(np.abs(tce_tbl.loc[tce_i, 'tce_duration'] - tce['Duration (hours)']) / tce['Duration (hours)'])
        tce_tbl.loc[tce_i, 'tce_duration'] = tce['Duration (hours)']
        data_aux_tbl['depth_diff_rel'].append(np.abs(tce_tbl.loc[tce_i, 'tce_depth'] - tce['Depth (ppm)']) / tce['Depth (ppm)'])
        tce_tbl.loc[tce_i, ['tce_depth', 'transit_depth']] = tce['Depth (ppm)']
        data_aux_tbl['prad_diff_rel'].append(np.abs(tce_tbl.loc[tce_i, 'tce_prad'] - tce['Planet Radius (R_Earth)']) / tce['Planet Radius (R_Earth)'])
        tce_tbl.loc[tce_i, 'tce_prad'] = tce['Planet Radius (R_Earth)']

data_aux_tbl = pd.DataFrame(data_aux_tbl)
data_aux_tbl.to_csv(save_dir / 'updated_tce_params.csv', index=False)
tce_tbl.to_csv(save_dir / 'tess_tces_s1-s40_11-23-2021_1409_stellarparams_updated_eb_tso_tec_label_modelchisqr_astronet_ruwe_magcat_uid_corrtsoebs_corraltdetfail_toidv.csv', index=False)

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

#%% Get number of TCEs per TOI

cnt_tces_toi = \
    tce_tbl['toi_dv'].value_counts().to_frame(name='num_tces_toi').reset_index().rename(columns={'index': 'toi_dv'})

toi_tbl = cnt_tces_toi.merge(tce_tbl[['label', 'toi_dv', 'toi_dv_corr', 'target_id']], on=['toi_dv'], how='left',
                             validate='one_to_many')
toi_tbl.to_csv(save_dir / 'tces_per_toi.csv', index=False)

for label in toi_tbl['label'].unique():
    f, ax = plt.subplots()
    ax.hist(toi_tbl.loc[toi_tbl['label'] == label, 'num_tces_toi'], bins=np.arange(1, 20), edgecolor='k', align='left')
    ax.set_xlabel('Number of TCEs per TOI')
    ax.set_ylabel('TOI Counts')
    ax.set_title(label)
    ax.set_xticks(np.arange(1, 19))
    ax.set_xlim([0.5, 18.5])
    f.savefig(save_dir / f'hist_num_tces_per_toi_{label}.png')
