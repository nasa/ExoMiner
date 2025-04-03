"""
Map neighbors celestial coordinates to pixel coordinates in the CCD of the corresponding target star.

The neighbors table must contain columns 'target_id' (in case the target pixel files of the target stars are being used),
'ID' (of the neighboring object), 'ra' and 'dec' coordinates of the neighboring object, with RA and Dec in degrees
"""

# 3rd party
import numpy as np
from pathlib import Path
import pandas as pd
# import subprocess
import multiprocessing
import logging
# from tess_stars2px import tess_stars2px_function_entry
import lightkurve as lk
import shutil
import time
import datetime
import re


def log_or_print(message, logger=None):
    """ Either log or print `message` to stdout.

    Args:
        message: str, message
        logger: logger

    Returns:
    """

    if logger:
        logger.info(message)
    else:
        print(message)


def map_objects_celestial_coordinates_to_ccd_px_coordinates(objs_tbl, sector, temp_fits_dir, logger=None):

    data_dict = {field: [] for field in ['target_id', 'ID', 'col_px', 'row_px']}

    # using WCS transformation stored in the target pixel files for each object
    target_grps = objs_tbl.groupby('target_id')

    for target_i, (target_id, objs) in enumerate(target_grps):

        data_dict['ID'] += objs['ID'].tolist()
        data_dict['target_id'] += [target_id] * len(objs)

        if target_i % 100 == 0:
            log_or_print(f'Iterated through {target_i}/{len(target_grps)} targets.', logger)

        search_result = lk.search_targetpixelfile(target=f'tic{target_id}', mission='TESS',
                                                  author=('TESS-SPOC', 'SPOC'),
                                                  exptime=120, sector=sector)

        if len(search_result) == 0:
            log_or_print(f'No target pixel files found for target {target_id} in sector {sector}. Setting pixel '
                         f'coordinates to NaN.', logger)

            data_dict['col_px'] += [np.nan] * len(objs)
            data_dict['row_px'] += [np.nan] * len(objs)
            continue

        if len(search_result) > 1:
            log_or_print(f'Found {len(search_result)} target pixel files for target {target_id} in sector '
                         f'{sector}. Considering only first result:\n {search_result.table}', logger)

        # Download the first search result
        download_successful = False
        while not download_successful:
            try:
                tpf = search_result.download(download_dir=str(temp_fits_dir), quality_bitmask='default')
                download_successful = True
            except Exception as e:
                log_or_print(f'Found issue when downloading target pixel file for target {target_id} in sector '
                             f'{sector}:\n {search_result}.\nError {e}\n Retrying after 60 seconds...', logger)

                # shutil.rmtree(Path(tpf.path).parent, ignore_errors=True)

                time.sleep(60)

        # use WCS transformation to map objects to CCD pixel frame
        try:
            outColPix_obj, outRowPix_obj = tpf.wcs.all_world2pix(objs['ra'], objs['dec'], 0)

            data_dict['col_px'] += list(outColPix_obj)
            data_dict['row_px'] += list(outRowPix_obj)

        except Exception as e:
            log_or_print(f'Found issue when using WCS transformation in {tpf.path} to map coordinates of '
                         f'neighbors for '
                         f'target {target_id} in sector:\nError {e}\n Setting values to NaN.', logger)

            data_dict['col_px'] += [np.nan] * len(objs)
            data_dict['row_px'] += [np.nan] * len(objs)

            continue

        # delete target pixel file if everything went successful up to this point
        try:
            shutil.rmtree(Path(tpf.path).parent, ignore_errors=False)
        except Exception as e:
            log_or_print(f'Tried to delete target pixel file for tpf.path but failed with error: {e}', logger)

    # # using tess-point
    # # initialize scinfo
    # _, _, _, _, _, _, _, _, scinfo = tess_stars2px_function_entry(stars_tbl['ID'][0], stars_tbl['ra'][0],
    #                                                               stars_tbl['dec'][0], trySector=sector)
    #
    # stars_ids, _, _, _, _, _, outColPix, outRowPix, scinfo = (
    #     tess_stars2px_function_entry(stars_tbl['ID'], stars_tbl['ra'], stars_tbl['dec'], scInfo=scinfo,
    #                                  trySector=sector))

    log_or_print(f'Iterated through all {len(target_grps)} targets.', logger)

    stars_px_coords_tbl = pd.DataFrame(data_dict)

    return stars_px_coords_tbl


def map_objects_celestial_coordinates_to_ccd_px_coordinates_main(neighbors_tbl_fp, temp_fits_dir, use_logs=True):
    """ Maps celestial coordinates of neighboring objects (RA and Dec) in degrees to the pixel coordinates
    (columns 'col_px' and 'row_px') for each star in sector `sector_id`.

    Args:
        neighbors_tbl_fp: Path, filepath to neighbors table
        sector: int, sector for which to get the pointing
        temp_fits_dir: Path, temporary directory to store target pixel FITS files
        use_logs: bool, if True writes information about run to logs. Otherwise, it prints them to the console

    """

    datetime_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    match = re.search(r'S(\d+)', neighbors_tbl_fp.name)
    if match:
        sector = match.group(1)
    else:
        raise ValueError(f"Could not find sector ID based on neighbors table filename: {neighbors_tbl_fp.name}\n "
                         f"Expected pattern S[0-9]+")

    neighbors_dir = neighbors_tbl_fp.parent

    # create directory to save sector search results
    map_px_res_dir = neighbors_dir / f'maping_results'
    map_px_res_dir.mkdir(parents=True, exist_ok=True)

    if use_logs:
        log_dir = neighbors_dir / 'logs_map_to_px'
        log_dir.mkdir(exist_ok=True)

        # set up logger
        logger = logging.getLogger(name=f'map_pxcoords_neighbors_targets_S{sector}')
        logger_handler = logging.FileHandler(filename=log_dir / f'map_pxcoords_neighbors_targets_{datetime_str}.log',
                                             mode='a')
        logger_formatter = logging.Formatter('%(asctime)s - %(message)s')
        logger.setLevel(logging.INFO)
        logger_handler.setFormatter(logger_formatter)
        logger.addHandler(logger_handler)
        logger.info(f'Started run for targets in S{sector}.')
        logger.info(f'Number of targets: {len(sector)}')
    else:
        logger = None

    log_or_print(f'Reading neighbors table {neighbors_tbl_fp}...', logger)
    neighbors_df = pd.read_csv(neighbors_tbl_fp)

    # map celestial coordinates to CCD from tess-point for these neighbors
    log_or_print('Mapping celestial to pixel coordinates for neighbors...', logger)

    stars_px_coords_tbl = map_objects_celestial_coordinates_to_ccd_px_coordinates(
        neighbors_df,
        sector,
        temp_fits_dir,
        logger
    )

    log_or_print(f'Created table with pixel coordinates for {len(stars_px_coords_tbl)} neighbors of the '
                 f'{len(neighbors_df["target_id"].unique())} targets.', logger)

    log_or_print(f'Adding pixel coordinates to neighbors...', logger)

    neighbors_df['ID'] = neighbors_df['ID'].astype(int)
    neighbors_px_coords = stars_px_coords_tbl.merge(neighbors_df, how='left', on=['ID', 'target_id'],
                                                    validate='one_to_one')
    neighbors_px_coords.to_csv(map_px_res_dir / f'neighbors_S{sector}_withpxcoords_{datetime_str}.csv', index=False)

    log_or_print(f'Finished run for targets in S{sector}.', logger)


if __name__ == "__main__":

    temp_tpf_dir = Path('/nobackupp19/msaragoc/work_dir/Kepler-TESS_exoplanet/experiments/search_neighboring_stars/')
    neighbors_dir = Path('/nobackupp19/msaragoc/work_dir/Kepler-TESS_exoplanet/experiments/search_neighboring_stars/test_tess_spoc_2min_s1-s88_search_radius_arcsec_168.0_tpf_wcs_4-2-2025_2155')
    # multiprocessing parameters
    # n_jobs = 1
    n_procs = 6
    use_logs = True

    neighbors_tbls_fps = []
    print(f'Found {len(neighbors_tbls_fps)} neighbors table(s).')

    jobs = [(neighbors_tbl_fp, temp_tpf_dir) for neighbors_tbl_fp in neighbors_tbls_fps]
    n_jobs = len(jobs)
    print(f'Split work into {n_jobs} job(s).')

    # # parallelize jobs
    # n_procs = min(n_jobs, n_procs)
    # pool = multiprocessing.Pool(processes=n_procs)
    # async_results = [pool.apply_async(get_ccd_coords_neighbors_targets, job) for job in jobs]
    # pool.close()
    # pool.join()

    for job in jobs:
        map_objects_celestial_coordinates_to_ccd_px_coordinates_main(*job)

    print(f'Finished.')
