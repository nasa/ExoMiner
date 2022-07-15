""" Script used to match TOIs to TCEs from single- and multi-sector runs DV mat files. """

# 3rd party
import multiprocessing
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import logging

# local
from data_wrangling.tess.toi_tce_matching_dvmat_utils import match_set_tois_tces

# %% Matching between TOIs and TCEs using the cosine distance between period templates


if __name__ == '__main__':

    # set results directory
    res_dir = Path(f'/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/data/ephemeris_matching/toi_tce_matching/'
                   f'{datetime.now().strftime("%m-%d-%Y_%H%M")}')
    res_dir.mkdir(exist_ok=True)
    (res_dir / 'plots').mkdir(exist_ok=True)

    # set up logger
    logger = logging.getLogger(name='toi_tce_matching')
    logger_handler = logging.FileHandler(filename=res_dir / f'toi_tce_matching.log', mode='w')
    logger_formatter = logging.Formatter('%(asctime)s - %(message)s')
    logger.setLevel(logging.INFO)
    logger_handler.setFormatter(logger_formatter)
    logger.addHandler(logger_handler)
    logger.info(f'Starting run...')

    # get TOI table
    toi_tbl_fp = Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/data/ephemeris_tables/tess/EXOFOP_TOI_lists/TOI/7-11-2022/exofop_toilists_nomissingpephem.csv')
    # columns to be used from the TOI table
    toi_cols = ['TOI', 'TIC ID', 'Sectors', 'Period (days)', 'Duration (hours)', 'Epoch (TBJD)', 'Depth (ppm)']
    toi_tbl = pd.read_csv(toi_tbl_fp, usecols=toi_cols)[:15]
    logger.info(f'Using TOI table: {toi_tbl_fp}')

    # get DV TCE tables for single- and multi-sector runs
    tce_root_dir = Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/data/ephemeris_tables/tess/DV_SPOC_mat_files')
    logger.info(f'Using TCE tables from {tce_root_dir}')
    # tce_root_dir = Path('/data5/tess_project/Data/Ephemeris_tables/TESS/DV_ephemeris')
    multisector_tce_dir = tce_root_dir / 'multi-sector' / 'csv_tables'
    singlesector_tce_dir = tce_root_dir / 'single-sector' / 'csv_tables'
    # columns to be used from the TCE tables
    tce_cols = ['catId', 'planetIndexNumber', 'allTransitsFit_orbitalPeriodDays_value',
                'allTransitsFit_transitDurationHours_value', 'allTransitsFit_transitEpochBtjd_value']
    multisector_tce_tbls = {(int(file.stem[14:16]), int(file.stem[16:18])): pd.read_csv(file, usecols=tce_cols)
                            for file in multisector_tce_dir.iterdir()}
    logger.info(f'Multi-sector runs available:  {list(multisector_tce_tbls.keys())}')
    singlesector_tce_tbls = {int(file.stem[14:16]): pd.read_csv(file, usecols=tce_cols)
                             for file in singlesector_tce_dir.iterdir()}
    logger.info(f'Single-sector runs available:  {list(singlesector_tce_tbls.keys())}')

    # set matching threshold and sampling interval
    match_thr = np.inf  # np.inf  # 0.25
    logger.info(f'Using matching threshold = {match_thr}')
    sampling_interval = 0.0001  # approximately 10 seconds
    logger.info(f'Using sampling interval {sampling_interval}')

    # maximum number of TCEs that can be associated with a TOI
    max_num_tces = len(singlesector_tce_tbls) + len(multisector_tce_tbls)

    # columns in matching table
    match_tbl_cols = ['TOI ID', 'TIC', 'Matched TCEs'] + [f'matching_dist_{i}' for i in range(max_num_tces)]
    n_processes = 2  # number of processes to parallelize the matching across TOIs
    tbl_jobs = np.array_split(toi_tbl, n_processes)
    logger.info(f'Starting ephemeris matching by splitting the TOI table across {n_processes} processes...')
    pool = multiprocessing.Pool(processes=n_processes)
    jobs = [(tbl_job.reset_index(inplace=False), tbl_job_i) +
            (match_tbl_cols, singlesector_tce_tbls, multisector_tce_tbls, match_thr, sampling_interval, max_num_tces,
             res_dir)
            for tbl_job_i, tbl_job in enumerate(tbl_jobs)]
    async_results = [pool.apply_async(match_set_tois_tces, job) for job in jobs]
    pool.close()
    matching_tbl = pd.concat([match_tbl_job.get() for match_tbl_job in async_results], axis=0)
    logger.info(f'Finished matching.')
    matching_tbl.to_csv(res_dir / f'tois_matchedtces_ephmerismatching_thr{match_thr}_samplint{sampling_interval}.csv',
                        index=False)

    # %% Plot histograms for matching distance

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
