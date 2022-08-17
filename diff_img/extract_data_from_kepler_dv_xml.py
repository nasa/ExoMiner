"""
Extracting difference image data from the Kepler Q1-Q17 DR25 DV XML file.
"""

# 3rd party
import pandas as pd
import numpy as np
from pathlib import Path
import multiprocessing

# local
from diff_img.utils_diff_img import get_data_from_kepler_dv_xml_multiproc

if __name__ == '__main__':

    # DV XML file path
    dv_xml_fp = Path('/data5/tess_project/Data/Kepler-Q1-Q17-DR25/dv/dv_xml/kplr20160128150956_dv.xml')
    # TCE table file path
    tce_tbl_fp = Path(
        '/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17_DR25/11-17-2021_1243/q1_q17_dr25_tce_2020.09.28_10.36.22_stellar_koi_cfp_norobovetterlabels_renamedcols_nomissingval_symsecphase_cpkoiperiod_rba_cnt0n_valpc_modelchisqr_ruwe_magcat_uid.csv')
    # run directory
    run_dir = Path('/data5/tess_project/Data/Kepler-Q1-Q17-DR25/dv/dv_xml/preprocessing/8-17-2022_1205')

    # create run directory
    run_dir.mkdir(exist_ok=True, parents=True)

    # load TCE table to get examples for which the image data are going to be extracted
    tce_tbl = pd.read_csv(tce_tbl_fp, usecols=['uid', 'label'])
    tce_tbl.set_index('uid', inplace=True)
    tces = tce_tbl

    # creating plotting directory
    plot_dir = run_dir / 'plots'
    plot_dir.mkdir(exist_ok=True)
    plot_prob = 0.01

    n_processes = 5
    pool = multiprocessing.Pool(processes=n_processes)
    n_jobs = 10
    tces_jobs = np.array_split(tces, n_jobs)
    jobs = [(dv_xml_fp, tces_job, run_dir, plot_prob, job_i) for job_i, tces_job in enumerate(tces_jobs)]
    async_results = [pool.apply_async(get_data_from_kepler_dv_xml_multiproc, job) for job in jobs]
    pool.close()

    # data = {}
    # for async_res in async_results:
    #     data.update(async_res.get())

    # data = get_data_from_dv_xml(dv_xml_fp, tces, plot_prob, plot_dir)

    # np.save(run_dir / 'keplerq1q17_dr25_diffimg.npy', data)

    # aggregating difference image data into a single numpy file
    data = {}
    for data_fp in sorted([fp for fp in run_dir.iterdir() if 'keplerq1q17_dr25_diffimg_' in fp.name and
                                                             fp.suffix == '.npy']):
        data.update(np.load(data_fp, allow_pickle=True).item())
    np.save(run_dir / 'keplerq1q17_dr25_diffimg.npy', data)
