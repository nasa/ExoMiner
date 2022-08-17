"""
Extracting difference image data from the TESS DV XML files.
"""

# 3rd party
import numpy as np
from pathlib import Path
import multiprocessing
import pandas as pd

# local
from diff_img.utils_diff_img import get_data_from_tess_dv_xml_multiproc

if __name__ == '__main__':
    # DV XML file path
    dv_xml_root_fp = Path('/data5/tess_project/Data/TESS_dv_fits/dv_xml/sector_runs')
    single_sector_runs = [fp for fp in (dv_xml_root_fp / 'single-sector').iterdir() if fp.is_dir()]
    multi_sector_runs = [fp for fp in (dv_xml_root_fp / 'multi-sector').iterdir() if fp.is_dir()]
    dv_xml_runs = list(single_sector_runs) + list(multi_sector_runs)

    # # TCE table file path
    tce_tbl_fp = Path(
        '/data5/tess_project/Data/Ephemeris_tables/TESS/DV_SPOC_mat_files/11-29-2021/tess_tces_s1-s40_11-23-2021_1409_stellarparams_updated_eb_tso_tec_label_modelchisqr_astronet_ruwe_magcat_uid_corrtsoebs_corraltdetfail.csv')
    tce_tbl = pd.read_csv(tce_tbl_fp, usecols=['uid', 'label'])
    tce_tbl.set_index('uid', inplace=True)

    # run directory
    run_dir = Path('/data5/tess_project/Data/TESS_dv_fits/dv_xml/preprocessing/8-17-2022_1611')

    # create run directory
    run_dir.mkdir(exist_ok=True, parents=True)

    # creating plotting directory
    plot_dir = run_dir / 'plots'
    plot_dir.mkdir(exist_ok=True)
    plot_prob = 0.01

    n_processes = 10
    pool = multiprocessing.Pool(processes=n_processes)
    # n_jobs = len(dv_xml_runs)
    # dv_xml_runs_jobs = np.array_split(dv_xml_runs, n_jobs)
    jobs = [(dv_xml_run, run_dir, plot_prob, tce_tbl, job_i) for job_i, dv_xml_run in enumerate(dv_xml_runs)]
    async_results = [pool.apply_async(get_data_from_tess_dv_xml_multiproc, job) for job in jobs]
    pool.close()

    # data = {}
    # for async_res in async_results:
    #     data.update(async_res.get())

    # data = get_data_from_dv_xml(dv_xml_run, run_dir, plot_prob, tce_tbl)

    # np.save(run_dir / 'tess_diffimg.npy', data)
