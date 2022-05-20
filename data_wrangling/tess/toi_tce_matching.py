"""
Script used to match TOIs to TCEs from different single- and multi-sector runs using DV TCE tables available in the
MAST.
"""

# 3rd party
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import multiprocessing

# local
from data_wrangling.tess.toi_tce_ephemeris import match_set_tois_tces, get_bin_ts_toi_tce

#%% Matching between TOIs and TCEs using the cosine distance between period templates

res_dir = Path(f'/home/msaragoc/Projects/Kepler-TESS_exoplanet/Analysis/toi_tce_matching/'
               f'{datetime.now().strftime("%m-%d-%Y_%H%M")}')
res_dir.mkdir(exist_ok=True)

# load TOI catalog
toi_dir = Path('/data5/tess_project/Data/Ephemeris_tables/TESS/EXOFOP_TOI_lists/TOI/7-30-2021')
toi_tbl = pd.read_csv(toi_dir / f'exofop_toilists_nomissingpephem.csv')

# load TCE tables
tce_root_dir = Path('/data5/tess_project/Data/Ephemeris_tables/TESS/DV_ephemeris')
multisector_tce_dir = tce_root_dir / 'multi-sector runs'
singlesector_tce_dir = tce_root_dir / 'single-sector runs'

multisector_tce_tbls = {(int(file.stem.split('-')[1][1:]), int(file.stem.split('-')[2][1:5])): pd.read_csv(file,
                                                                                                           header=6)
                        for file in multisector_tce_dir.iterdir() if 'tcestats' in file.name and file.suffix == '.csv'}
# filter runs
multisector_tce_tbls = {sector_run: tbl for sector_run, tbl in multisector_tce_tbls.items() if sector_run == (1, 36)}
singlesector_tce_tbls = {int(file.stem.split('-')[1][1:]): pd.read_csv(file, header=6)
                         for file in singlesector_tce_dir.iterdir()
                         if 'tcestats' in file.name and file.suffix == '.csv'}
# filter runs
singlesector_tce_tbls = {sector_run: tbl for sector_run, tbl in singlesector_tce_tbls.items() if sector_run > 35}
# singlesector_tce_tbls[21].drop_duplicates(subset='tceid', inplace=True, ignore_index=True)

map_tce_tbl_names = {
    '<=30': {'period': 'orbitalPeriodDays', 'epoch': 'transitEpochBtjd', 'duration': 'transitDurationHours'},
    '>30': {'period': 'tce_period', 'epoch': 'tce_time0bt', 'duration': 'tce_duration'},
}

# threshold used to accept matching
match_thr = np.inf  # np.inf  # 0.25
sampling_interval = 0.00001  # sampling interval used to create binary time series; approximately 1 min (TESS is 2min)

# set maximum number of TCEs as total number of runs
max_num_tces = len(singlesector_tce_tbls) + len(multisector_tce_tbls)

# renaming columns to names expected by the matching function
toi_tbl.rename(columns={'Epoch (TBJD)': 'epoch', 'Duration (hours)': 'duration', 'Period (days)': 'period',
                        'TIC ID': 'target_id'}, inplace=True)
tce_tbls_col_mapper = {'ticid': 'target_id', 'tce_plnt_num': 'tce_plnt_num', 'tce_period': 'period',
                       'tce_time0bt': 'epoch', 'tce_duration': 'duration'}
singlesector_tce_tbls = {sector_run: tbl.rename(columns=tce_tbls_col_mapper)
                         for sector_run, tbl in singlesector_tce_tbls.items()}
multisector_tce_tbls = {sector_run: tbl.rename(columns=tce_tbls_col_mapper)
                        for sector_run, tbl in multisector_tce_tbls.items()}

match_tbl_cols = ['TOI ID', 'TIC', 'Matched TCEs'] + [f'matching_dist_{i}' for i in range(max_num_tces)]
n_processes = 15
tbl_jobs = np.array_split(toi_tbl, n_processes)
pool = multiprocessing.Pool(processes=n_processes)
jobs = [(tbl_job.reset_index(inplace=False), tbl_job_i) +
        (match_tbl_cols, singlesector_tce_tbls, multisector_tce_tbls, match_thr, sampling_interval, max_num_tces,
         res_dir) for tbl_job_i, tbl_job in enumerate(tbl_jobs)]
async_results = [pool.apply_async(match_set_tois_tces, job) for job in jobs]
pool.close()

matching_tbl = pd.concat([match_tbl_job.get() for match_tbl_job in async_results], axis=0)

matching_tbl.to_csv(res_dir / f'tois_matchedtces_ephmerismatching_thr{match_thr}_samplint{sampling_interval}.csv',
                    index=False)


#%% Plot histograms for matching distance

bins = np.linspace(0, 1, 21, endpoint=True)
f, ax = plt.subplots()
ax.hist(matching_tbl['matching_dist_0'], bins, edgecolor='k')
ax.set_yscale('log')
ax.set_xlabel('Matching distance')
ax.set_ylabel('Counts')
ax.set_xlim([0, 1])
f.savefig(res_dir / 'hist_matching_dist_0.png')

f, ax = plt.subplots()
ax.hist(matching_tbl['matching_dist_0'], bins, edgecolor='k', cumulative=True, density=True)
# ax.set_yscale('log')
ax.set_xlabel('Matching distance')
ax.set_ylabel('Normalized Cumulative Counts')
ax.set_xlim([0, 1])
f.savefig(res_dir / 'cumhist_norm_matching_dist_0.png')


#%% Plot binary time series between TOI and TCE

toi_id = 185.01
toi = toi_tbl.loc[toi_tbl['TOI'] == toi_id].squeeze()
tce_id = '100100827'.zfill(11) + '-01'
tce = singlesector_tce_tbls[2].loc[singlesector_tce_tbls[2]['tceid'] == tce_id].squeeze()
phase = max(toi['period'], tce['period'])

tce['epoch'] = toi['epoch']
# tce['period'] = toi['period'] + toi['period'] / 2
toi_bin_ts, tce_bin_ts, match_dist = get_bin_ts_toi_tce(toi, tce, sampling_interval)
nsamples = len(toi_bin_ts)
phase_range = np.around(np.linspace(-phase / 2, phase / 2, nsamples, endpoint=True), 3)

f, ax = plt.subplots()
ax.plot(toi_bin_ts)
ax.plot(tce_bin_ts, 'r--')
xticks_val = np.linspace(0, nsamples - 1, 10, endpoint=True, dtype='int')
ax.set_xticks(np.arange(nsamples)[xticks_val])
ax.set_xticklabels(phase_range[xticks_val], rotation=0)
ax.set_xlabel('Phase (day)')
ax.set_ylabel('Transit ephemerides template')
ax.set_title(f'Matching distance={match_dist:.4f}')
f.suptitle(f'TOI {toi_id} | TCE {tce_id}')
f.savefig(f'/home/msaragoc/Downloads/{toi_id}_{tce_id}_ephem_template_matching.svg')