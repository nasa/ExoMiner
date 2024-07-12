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
    dv_xml_root_fp = Path('/data5/tess_project/Data/tess_spoc_ffi_data/dv/xml_files/')
    single_sector_runs = [fp for fp in (dv_xml_root_fp / 'single-sector').iterdir() if fp.is_dir()]  #  and fp.stem in ['s0062', 's0057', 's0056', 's0055']]  # [dv_xml_root_fp / 'single-sector' / 's0051']
    multi_sector_runs = []  # [fp for fp in (dv_xml_root_fp / 'multi-sector').iterdir() if fp.is_dir()]
    dv_xml_runs = list(single_sector_runs) + list(multi_sector_runs)

    # run directory
    run_dir = Path('/data5/tess_project/Data/tess_spoc_ffi_data/dv/diff_img/extracted_data/s36-s68_singlesectorsonly_3-20-2024_0943')
    plot_prob = 0.01  # plot probability
    n_processes = 14  # number of processes used to parallelize extraction

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
    logger_handler = logging.FileHandler(filename=log_dir / f'extract_img_data_from_tess_dv_xml_main.log', mode='w')
    logger_formatter = logging.Formatter('%(asctime)s - %(message)s')
    logger.setLevel(logging.INFO)
    logger_handler.setFormatter(logger_formatter)
    logger.addHandler(logger_handler)
    logger.info(f'Starting preprocessing run...')
    logger.info(f'Number of runs: {len(dv_xml_runs)}')
    logger.info(f'Runs set for preprocessing:')
    for dv_xml_run in dv_xml_runs:
        logger.info(f'Run {str(dv_xml_run)}')

    logger.info(f'Using {n_processes} processes...')
    pool = multiprocessing.Pool(processes=n_processes)
    jobs = [(dv_xml_run, data_dir, plot_dir, plot_prob, log_dir, job_i)
            for job_i, dv_xml_run in enumerate(dv_xml_runs)]
    logger.info(f'Setting {len(jobs)} jobs.')
    logger.info('Started running jobs.')
    async_results = [pool.apply_async(get_data_from_tess_dv_xml_multiproc, job) for job in jobs]
    pool.close()
    pool.join()

    # for job in jobs:
    #     print('Starting job')
    #     get_data_from_tess_dv_xml_multiproc(*job)

    logger.info('Finished preprocessing.')
