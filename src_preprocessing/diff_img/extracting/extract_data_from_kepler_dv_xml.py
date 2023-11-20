"""
Extracting difference image data from the Kepler Q1-Q17 DR25 DV XML file.
"""

# 3rd party
import pandas as pd
import numpy as np
from pathlib import Path
import multiprocessing

# local
from src_preprocessing.diff_img.extracting.utils_diff_img import get_data_from_kepler_dv_xml_multiproc

if __name__ == '__main__':

    # DV XML file path
    dv_xml_fp = Path('/Users/msaragoc/Projects/exoplanet_transit_classification/data/fits_files/kepler/q1_q17_dr25/dv/kplr20160128150956_dv.xml')
    # TCE table file path
    tce_tbl_fp = Path('/Users/msaragoc/Projects/exoplanet_transit_classification/data/ephemeris_tables/kepler/q1-q17_dr25/11-17-2021_1243/q1_q17_dr25_tce_3-6-2023_1734.csv')
    # run directory
    run_dir = Path('/Users/msaragoc/Projects/exoplanet_transit_classification/data/fits_files/kepler/q1_q17_dr25/dv/preprocessing/11-14-2023_1027')

    # create run directory
    run_dir.mkdir(exist_ok=True, parents=True)

    data_dir = run_dir / 'data'
    data_dir.mkdir(exist_ok=True)

    # load TCE table to get examples for which the image data are going to be extracted
    tce_tbl = pd.read_csv(tce_tbl_fp, usecols=['uid', 'label'])
    # tce_tbl = tce_tbl.loc[tce_tbl['uid'].isin(['5130369-1'])]
    tce_tbl.set_index('uid', inplace=True)
    tces = tce_tbl

    # create plotting directory
    plot_dir = run_dir / 'plots'
    plot_dir.mkdir(exist_ok=True)
    plot_prob = 0.01

    # create log directory
    log_dir = run_dir / 'logs'
    log_dir.mkdir(exist_ok=True)

    # # sequential
    # data = get_data_from_kepler_dv_xml_multiproc(dv_xml_fp, tces, data_dir, plot_dir, plot_prob, log_dir, 0)

    # parallelize
    n_processes = 10
    pool = multiprocessing.Pool(processes=n_processes)
    n_jobs = 10
    tces_jobs = np.array_split(tces, n_jobs)
    jobs = [(dv_xml_fp, tces_job, data_dir, plot_dir, plot_prob, log_dir, job_i)
            for job_i, tces_job in enumerate(tces_jobs)]
    async_results = [pool.apply_async(get_data_from_kepler_dv_xml_multiproc, job) for job in jobs]
    pool.close()

    # aggregating difference image data into a single numpy file
    data = {}
    for data_fp in sorted([fp for fp in data_dir.iterdir()
                           if 'keplerq1q17_dr25_diffimg_' in fp.name and fp.suffix == '.npy']):
        data.update(np.load(data_fp, allow_pickle=True).item())
    np.save(run_dir / 'keplerq1q17_dr25_diffimg.npy', data)
