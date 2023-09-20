"""
Extracting difference image data from the TESS DV XML files.
"""

# 3rd party
from pathlib import Path
import multiprocessing
import logging

# local
from diff_img.extracting.utils_diff_img import get_data_from_tess_dv_xml_multiproc

if __name__ == '__main__':
    # DV XML file path
    dv_xml_root_fp = Path('/data5/tess_project/Data/TESS_dv_fits/dv_xml/sector_runs')
    single_sector_runs = [fp for fp in (dv_xml_root_fp / 'single-sector').iterdir() if fp.is_dir()]
    multi_sector_runs = [fp for fp in (dv_xml_root_fp / 'multi-sector').iterdir() if fp.is_dir()]
    dv_xml_runs = list(single_sector_runs) + list(multi_sector_runs)

    # # # TCE table file path
    # tce_tbl_fp = Path(
    #     '/data5/tess_project/Data/Ephemeris_tables/TESS/DV_SPOC_mat_files/11-29-2021/tess_tces_s1-s40_11-23-2021_1409_stellarparams_updated_eb_tso_tec_label_modelchisqr_astronet_ruwe_magcat_uid_corrtsoebs_corraltdetfail.csv')
    # tce_tbl = pd.read_csv(tce_tbl_fp, usecols=['uid', 'label'])
    # tce_tbl.set_index('uid', inplace=True)

    # run directory
    run_dir = Path('/data5/tess_project/Data/TESS_dv_fits/dv_xml/preprocessing/8-17-2022_1611')
    # create run directory
    run_dir.mkdir(exist_ok=True, parents=True)

    data_dir = run_dir / 'data'
    data_dir.mkdir(exist_ok=True)

    # create plotting directory
    plot_dir = run_dir / 'plots'
    plot_dir.mkdir(exist_ok=True)
    plot_prob = 0.01

    # create log directory
    log_dir = run_dir / 'logs'
    log_dir.mkdir(exist_ok=True)

    # set up logger
    logger = logging.getLogger(name=f'extract_img_data_tess_dv_xml_main')
    logger_handler = logging.FileHandler(filename=log_dir / f'extract_img_data_from_tess_dv_xml_main.log',
                                         mode='w')
    logger_formatter = logging.Formatter('%(asctime)s - %(message)s')
    logger.setLevel(logging.INFO)
    logger_handler.setFormatter(logger_formatter)
    logger.addHandler(logger_handler)
    logger.info(f'Starting preprocessing run...')
    logger.info(f'Number of runs: {len(dv_xml_runs)}')
    logger.info(f'Runs set for preprocessing:')
    for dv_xml_run in dv_xml_runs:
        logger.info(f'Run {str(dv_xml_run)}')

    n_processes = 14
    logger.info(f'Using {n_processes} processes...')
    pool = multiprocessing.Pool(processes=n_processes)
    # n_jobs = len(dv_xml_runs)
    # dv_xml_runs_jobs = np.array_split(dv_xml_runs, n_jobs)
    jobs = [(dv_xml_run, data_dir, plot_dir, plot_prob, log_dir, job_i)
            for job_i, dv_xml_run in enumerate(dv_xml_runs)]
    logger.info(f'Setting {len(jobs)} jobs.')
    logger.info('Started running jobs.')
    async_results = [pool.apply_async(get_data_from_tess_dv_xml_multiproc, job) for job in jobs]
    pool.close()

    logger.info('Finished preprocessing.')
