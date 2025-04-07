"""
Get sectors observed for set of TICs in each sector run.
"""

# 3rd party
import pandas as pd
from pathlib import Path
import xml.etree.cElementTree as et
import logging
import multiprocessing
import re
import os


MAX_NUM_SECTORS = 150


def get_sectors_observed_from_tess_dv_xml(dv_xml_run, logger, job_i=-1):
    """ Get sectors observed for the set of TICs in a sector run.

    Args:
        dv_xml_run: Path, DV xml directory for a given sector run.
        logger: logger for dv xml run

    Returns:
        data_df, pandas DataFrame with sectors observed for each TIC in the sector run
    """

    data_dict = {field: [] for field in ['target_id', 'sectors_observed']}

    # get dv xml filepaths for this sector run
    dv_xml_run_fps = list(dv_xml_run.rglob("*.xml"))

    n_targets = len(dv_xml_run_fps)

    # get sector run id
    s_sector, e_sector = re.findall('-s[0-9]+', dv_xml_run_fps[0].stem)
    s_sector, e_sector = int(s_sector[2:]), int(e_sector[2:])
    if s_sector != e_sector:  # multi-sector run
        sector_run_id = f'{s_sector}-{e_sector}'
    else:  # single-sector run
        sector_run_id = f'{s_sector}'

    # iterate on targets dv xml files for the sector run
    for target_i, dv_xml_fp in enumerate(dv_xml_run_fps):

        if target_i % 1000 == 0:
            logger.info(f'[{job_i}] [Sector run {sector_run_id}] Iterating over TIC {target_i + 1}/{n_targets} in '
                        f'{dv_xml_fp.name}.')

        try:
            tree = et.parse(dv_xml_fp)
        except Exception as e:
            logger.info(f'Exception found when reading {dv_xml_fp}: {e}.')
            continue
        root = tree.getroot()

        # check if there are results for more than one processing run for this TIC and sector run
        tic_id = root.attrib['ticId']
        data_dict['target_id'].append(tic_id)
        sectors_obs = root.attrib['sectorsObserved']
        data_dict['sectors_observed'].append(sectors_obs.ljust(MAX_NUM_SECTORS, '0'))

        # tic_drs = [fp for fp in dv_xml_run.glob(f'*{tic_id.zfill(16)}*')]
        # if len(tic_drs) > 1:
        #     curr_dr = int(dv_xml_fp.stem.split('-')[-1][:-4])
        #     latest_dr = sorted([int(fp.stem.split('-')[-1][:-4])
        #                         for fp in dv_xml_run.glob(f'*{tic_id.zfill(16)}*')])[-1]
        #     if curr_dr != latest_dr:
        #         logger.info(f'[{proc_id}] [Sector run {sector_run_id}] '
        #                     f'Skipping {dv_xml_fp.name} for TIC {tic_id} since there is '
        #                     f'more recent processed results (current release {curr_dr}, latest release {latest_dr})'
        #                     f'... ({target_i}/{n_targets} targets)')
        #         continue
        #
        # # get tce planet results
        # planet_res_lst = [el for el in root if 'planetResults' in el.tag]
        # logger.info(f'Found {len(planet_res_lst)} TCEs in TIC {tic_id} for sector {sector_run_id}.')
        #
        # n_sectors_expected = sectors_obs.count('1')
        # logger.info(f'Target TIC {tic_id} observed in {n_sectors_expected} sectors for sector run {sector_run_id}.')
        #
        # # get tces detected in this target for this sector run
        # for planet_res in planet_res_lst:
        #
        #     uid = f'{root.attrib["ticId"]}-{planet_res.attrib["planetNumber"]}-S{sector_run_id}'
        #
        #     data_dict['uid'].append(uid)
        #
        # data_df = pd.DataFrame.from_dict(data_dict)
        # data_df['sectors_observed'] = sectors_obs.zfill(MAX_NUM_SECTORS)

    data_df = pd.DataFrame.from_dict(data_dict)
    data_df['sector_run'] = sector_run_id

    return data_df


def get_sectors_observed_from_tess_dv_xml_multiproc(dv_xml_run, save_dir, log_dir, job_i):
    """ Wrapper for `get_sectors_observed_from_tess_dv_xml()`. Extract sectors observed from the DV XML files for a
    TESS sector run.

    :param dv_xml_run: Path, path to sector run with DV XML files.
    :param save_dir: Path, save directory
    :param log_dir: Path, log directory
    :param job_i: int, job id

    :return:
    """

    # set up logger
    logger = logging.getLogger(name=f'extract_sectors_observed_tess_dv_xml_{dv_xml_run.name}')
    logger_handler = logging.FileHandler(
        filename=log_dir / f'extract_sectors_observed_tics_from_tess_dv_xml-{dv_xml_run.name}.log',
        mode='w')
    logger_formatter = logging.Formatter('%(asctime)s - %(message)s')
    logger.setLevel(logging.INFO)
    logger_handler.setFormatter(logger_formatter)
    logger.addHandler(logger_handler)
    logger.info(f'[{job_i}] Starting run {dv_xml_run.name}...')

    data_df = get_sectors_observed_from_tess_dv_xml(dv_xml_run, logger, job_i)
    data_df.to_csv(save_dir / f'{dv_xml_run.name}_tics_sectors_observed.csv', index=False)

    logger.info(f'[{job_i}] Finished run {dv_xml_run.name}.')


if __name__ == '__main__':

    # DV XML file path
    dv_xml_root_fp = Path('/data5/tess_project/Data/tess_spoc_2min_data/dv/xml_files/sector_runs/')
    single_sector_runs = [fp for fp in (dv_xml_root_fp / 'single-sector').iterdir() if fp.is_dir()]
    multi_sector_runs = [fp for fp in (dv_xml_root_fp / 'multi-sector').iterdir() if fp.is_dir()]
    dv_xml_runs = list(single_sector_runs) + list(multi_sector_runs)

    # run directory
    run_dir = Path('/home/msaragoc/Projects/exoplnt_dl/experiments/extract_sectors_observed_from_tess_dv_xml')
    n_processes = 14  # number of processes used to parallelize extraction

    # create run directory
    run_dir.mkdir(exist_ok=True, parents=True)
    # setting up data directory
    data_dir = run_dir / 'data'
    data_dir.mkdir(exist_ok=True)
    # create log directory
    log_dir = run_dir / 'logs'
    log_dir.mkdir(exist_ok=True)

    # set up logger
    logger = logging.getLogger(name=f'extract_sectors_observed_tics_tess_dv_xml')
    logger_handler = logging.FileHandler(filename=log_dir / f'extract_sectors_observed_tics_from_tess_dv_xml_main.log',
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

    logger.info(f'Using {n_processes} processes...')
    pool = multiprocessing.Pool(processes=n_processes)
    jobs = [(dv_xml_run, data_dir, log_dir, job_i) for job_i, dv_xml_run in enumerate(dv_xml_runs)]
    logger.info(f'Setting {len(jobs)} jobs.')
    logger.info('Started running jobs.')
    async_results = [pool.apply_async(get_sectors_observed_from_tess_dv_xml_multiproc, job) for job in jobs]
    pool.close()
    pool.join()

    # for job in jobs:
    #     print(f'Starting job for sectror run {job[0]}')
    #     get_sectors_observed_from_tess_dv_xml_multiproc(*job)

    logger.info('Finished preprocessing.')
