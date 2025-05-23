"""
Extracting difference image data from the TESS DV XML files.
"""

# 3rd party
from pathlib import Path
import multiprocessing
import logging

# local
from src_preprocessing.diff_img.extracting.utils_diff_img import get_data_from_tess_dv_xml_multiproc

if __name__ == '__main__':

    # DV XML file path
    dv_xml_root_fp = Path('')
    single_sector_runs = [fp for fp in (dv_xml_root_fp / 'single-sector').iterdir() if fp.is_dir()]
    multi_sector_runs = [fp for fp in (dv_xml_root_fp / 'multi-sector').iterdir() if fp.is_dir()]
    dv_xml_runs = list(single_sector_runs) + list(multi_sector_runs)

    # run directory
    run_dir = Path('')
    plot_prob = 0.0001  # plot probability
    n_processes = 88  # number of processes used to parallelize extraction

    # create run directory
    run_dir.mkdir(exist_ok=True, parents=True)
    # setting up data directory
    data_dir = run_dir / 'data'
    data_dir.mkdir(exist_ok=True)
    # create plotting directory
    plot_dir = run_dir / 'plots'
    plot_dir.mkdir(exist_ok=True)
    # create log directory
    log_dir = run_dir / 'logs'
    log_dir.mkdir(exist_ok=True)

    # set up logger
    logger = logging.getLogger(name=f'extract_img_data_tess_dv_xml')
    logger_handler = logging.FileHandler(filename=log_dir / f'extract_img_data_from_tess_dv_xml_main.log', mode='a')
    logger_formatter = logging.Formatter('%(asctime)s - %(message)s')
    logger.setLevel(logging.INFO)
    logger_handler.setFormatter(logger_formatter)
    logger.addHandler(logger_handler)
    logger.info(f'Starting preprocessing run...')

    # check if NumPy file for a given sector run already exists in the run directory
    dv_xml_runs_res_found = [dv_xml_run for dv_xml_run in dv_xml_runs
                             if (run_dir / 'data' / f'tess_diffimg_{dv_xml_run.name}.npy').exists()]
    logger.info(f'Found NumPy files for the following sector runs in {run_dir}:\n {dv_xml_runs_res_found}\n Skipping '
                f'those sector runs...')
    dv_xml_runs = [dv_xml_run for dv_xml_run in dv_xml_runs if dv_xml_run not in dv_xml_runs_res_found]

    logger.info(f'Number of runs: {len(dv_xml_runs)}')
    logger.info(f'Runs set for preprocessing:')
    for dv_xml_run in dv_xml_runs:
        logger.info(f'Run {str(dv_xml_run)}')

    jobs = [(dv_xml_run, data_dir, plot_dir, plot_prob, log_dir, job_i)
            for job_i, dv_xml_run in enumerate(dv_xml_runs)]
    n_jobs = len(jobs)
    logger.info(f'Setting {len(jobs)} jobs.')
    logger.info('Started running jobs.')

    n_processes = min(n_processes, n_jobs)
    logger.info(f'Using {n_processes} processes...')
    pool = multiprocessing.Pool(processes=n_processes)
    async_results = [pool.apply_async(get_data_from_tess_dv_xml_multiproc, job) for job in jobs]
    pool.close()
    pool.join()

    # for job in jobs:
    #     print('Starting job')
    #     get_data_from_tess_dv_xml_multiproc(*job)

    logger.info('Finished extracting difference image data from DV xml files.')
