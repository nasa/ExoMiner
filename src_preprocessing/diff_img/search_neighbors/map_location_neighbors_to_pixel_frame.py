"""
Map neighbors celestial coordinates to pixel coordinates in the CCD of the corresponding target star.
"""

# 3rd party
import numpy as np
from pathlib import Path
import pandas as pd
# import subprocess
import multiprocessing
import logging
# from tess_stars2px import tess_stars2px_function_entry
import datetime
import re
from astropy.io import fits
from astropy import wcs


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


def map_neighbors_loc_to_target_ccd_px_coordinates_sector(objs_tbl, sector_lc_dir, logger=None):
    """ Maps celestial coordinates of objects in `objs_tbl` ('ra' and 'dec' in degrees) that are neighbors of target
    with TIC ID 'target_id' to pixel coordinates in CCD for sector in `sector_lc_dir`.

    Uses WCS transformation stored in lightcurve FITS files of the targets of interest ('target_id' in `objs_tbl`) for
    the sector of interest defined in `sector_lc_dir`.

    `objs_tbl` table must contain columns 'target_id' (in case the target pixel files of the target stars are being
    used), 'ID' (of the neighboring object), 'ra' and 'dec' coordinates of the neighboring object, with RA and Dec in
    degrees.

    Args:
        objs_tbl: pandas DataFrame, table with objects with celestial coordinates to be mapped to CCD pixel coordinates
        sector_lc_dir: Path, path to directory with lightcurve FITS files for a sector
        logger: logger

    Returns: stars_px_coords_tbl, pandas DataFrame with mapped CCD pixel coordinates
    """

    data_dict = {field: [] for field in ['target_id', 'ID', 'col_px', 'row_px']}

    # using WCS transformation stored in the target pixel files for each object
    target_grps = objs_tbl.groupby('target_id')

    for target_i, (target_id, objs) in enumerate(target_grps):

        data_dict['ID'] += objs['ID'].tolist()
        data_dict['target_id'] += [target_id] * len(objs)

        if target_i % 100 == 0:
            log_or_print(f'Iterated through {target_i}/{len(target_grps)} targets.', logger)

        lc_fp = list(sector_lc_dir.rglob(f'*{target_id}*lc.fits'))

        if len(lc_fp) != 1:
            log_or_print(f'Found {len(lc_fp)} lightcurve FITS files for target {target_id} in '
                         f'{sector_lc_dir.name}. Expected one file.\n Setting pixel coordinates to NaN.', logger)

            data_dict['col_px'] += [np.nan] * len(objs)
            data_dict['row_px'] += [np.nan] * len(objs)
            continue

        lc_fp = lc_fp[0]
        lc_aperture_header = fits.getheader(lc_fp, extname='APERTURE')
        wcs_lc = wcs.WCS(lc_aperture_header)

        # use WCS transformation to map objects to CCD pixel frame
        try:
            outColPix_obj, outRowPix_obj = wcs_lc.all_world2pix(objs['ra'], objs['dec'], 0)

            data_dict['col_px'] += list(outColPix_obj)
            data_dict['row_px'] += list(outRowPix_obj)

        except Exception as e:
            log_or_print(f'Found issue when using WCS transformation in {lc_fp} to map coordinates of '
                         f'neighbors for target {target_id} in sector:\nError {e}\n Setting values to NaN.', logger)

            data_dict['col_px'] += [np.nan] * len(objs)
            data_dict['row_px'] += [np.nan] * len(objs)

            continue

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

def map_neighbors_loc_to_target_ccd_px_coordinates_sector_splits(objs_tbl, sector_lc_dir, res_dir, n_splits_objs,
                                                                 logger=None, targets_filtered=False):
    """ Maps celestial coordinates of objects in `objs_tbl` ('ra' and 'dec' in degrees) that are neighbors of target
    with TIC ID 'target_id' to pixel coordinates in CCD for sector in `sector_lc_dir`. Splits `objs_tbl` into
    `n_splits_objs` tables before mapping celestial coordinates. The mapping results of each split are saved into a
    table with 'col_px' and 'row_px' coordinates.

    Args:
        objs_tbl: pandas DataFrame, table with objects with celestial coordinates to be mapped to CCD pixel coordinates
        sector_lc_dir: Path, path to directory with lightcurve FITS files for a sector
        res_dir: Path, path to directory to save results
        n_splits_objs: int, split search results into multiple tables by splitting objects into `n_splits_objs`
        logger: logger
        targets_filtered: bool, if `True`, it assumes that these are targets that were not processed before

    Returns:
        search_res: pandas DataFrame, table with neighbors. Returns `None` is `save_subtbl_fp` is not `None`
    """

    if targets_filtered:
        res_splits_found = [int(fp.stem.split('_')[-1]) for fp in res_dir.glob('neighbors_pxcoords_*.csv')]
        new_split_i = max(res_splits_found) + 1
        log_or_print(f'Adding results for new targets to previous results. New tables start at {new_split_i}',
                     logger)
    else:
        new_split_i = 0

    objs_splits = np.array_split(objs_tbl, n_splits_objs)

    for obj_split_i, objs_split in enumerate(objs_splits):

        if len(objs_split) == 0:
            continue

        res_obj_split_i = obj_split_i + new_split_i

        if (res_dir / f'neighbors_pxcoords_{res_obj_split_i}.csv').exists():
            log_or_print(f'Found table for target split {res_obj_split_i} (out of {n_splits_objs} splits). '
                         f'Skipping {len(objs_split)} targets in this split...', logger)
            continue

        log_or_print(f'Iterating through {len(objs_split)} objects in split '
                     f'{res_obj_split_i} (out of {len(objs_tbl)} in total) split across {n_splits_objs} splits',
                     logger)
        
        objs_split_pxcoords = map_neighbors_loc_to_target_ccd_px_coordinates_sector(objs_split, sector_lc_dir, logger)

        log_or_print(f'Adding pixel coordinates to neighbors...', logger)
        objs_split['ID'] = objs_split['ID'].astype(int)
        objs_split = objs_split.merge(objs_split_pxcoords, how='left', on=['ID', 'target_id'], validate='one_to_one')

        log_or_print(f'Saving subset table of results...', logger)
        objs_split.to_csv(res_dir / f'neighbors_pxcoords_{res_obj_split_i}.csv', index=False)

    log_or_print(f'Iterated through all {len(objs_tbl)} across {n_splits_objs} splits.', logger)


def filter_neighbors_tbl(neighbors_tbl, filter_target_tbl):
    """ Filter targets in `neighbors_tbl` based on targets in `filter_target_tbl` for the corresponding sector.

    Args:
        neighbors_tbl: pandas DataFrame, table with objects with celestial coordinates to be mapped to CCD pixel
            coordinates
        filter_target_tbl: pandas DataFrame, table used to filter targets in `neighbors_tbl` for the corresponding
            sector
    Returns:
        neighbor_tbl: removed neighbors whose targets are not in `filter_target_tbl`
    """

    neighbors_tbl = neighbors_tbl.loc[neighbors_tbl['target_id'].isin(filter_target_tbl['target_id'])]

    return neighbors_tbl


def map_neighbors_loc_to_target_ccd_px_coordinates_sector_main(neighbors_tbl_fp, lc_sector_dir, n_splits_objs,
                                                               use_logs=True, filter_target_tbl=None):
    """ Maps celestial coordinates of objects in `neighbors_tbl_fp` ('ra' and 'dec' in degrees) that are neighbors of
    target with TIC ID 'target_id' to pixel coordinates in CCD for sector in `lc_sector_dir`.

    Uses WCS transformation stored in lightcurve FITS files of the targets of interest ('target_id' in
    `neighbors_tbl_fp`) for the sector of interest defined in `lc_sector_dir`.

    `neighbors_tbl_fp` table must contain columns 'target_id' (in case the target pixel files of the target stars are
    being used), 'ID' (of the neighboring object), 'ra' and 'dec' coordinates of the neighboring object, with RA and Dec
    in degrees.

    Args:
        neighbors_tbl_fp: Path, filepath to neighbors table
        lc_sector_dir: Path, path to directory that contains lightcurve FITS files for the corresponding sector
        n_splits_objs: int, split search results into multiple tables by splitting objects into `n_splits_objs`
        use_logs: bool, if True writes information about run to logs. Otherwise, it prints them to the console
        filter_target_tbl: pandas DataFrame, table used to filter targets in `neighbors_tbl`
    """

    match = re.search(r'S(\d+)', neighbors_tbl_fp.name)
    if match:
        sector = match.group(1)
    else:
        raise ValueError(f"Could not find sector ID based on neighbors table filename: {neighbors_tbl_fp.name}\n "
                         f"Expected pattern S[0-9]+")

    neighbors_dir = neighbors_tbl_fp.parent

    # create directory to save sector search results
    map_px_res_dir = neighbors_dir / f'mapping_results'
    map_px_res_dir.mkdir(parents=True, exist_ok=True)

    if use_logs:

        # set up logger
        logger = logging.getLogger(name=f'map_pxcoords_neighbors_targets_S{sector}')
        logger_handler = logging.FileHandler(filename=map_px_res_dir / f'map_pxcoords_neighbors_targets.log',
                                             mode='a')
        logger_formatter = logging.Formatter('%(asctime)s - %(message)s')
        logger.setLevel(logging.INFO)
        logger_handler.setFormatter(logger_formatter)
        logger.addHandler(logger_handler)
        logger.info(f'Started run for targets in S{sector}.')
    else:
        logger = None

    log_or_print(f'Reading neighbors table {neighbors_tbl_fp}...', logger)
    neighbors_df = pd.read_csv(neighbors_tbl_fp)

    if filter_target_tbl:

        filter_target_tbl_sector = filter_target_tbl.loc[filter_target_tbl['sector'] == sector]

        log_or_print(f'Getting results for a subset of {len(filter_target_tbl_sector)} targets', logger)

        neighbors_df = filter_neighbors_tbl(neighbors_df, filter_target_tbl_sector)

        logger.info(f'Number of targets after filtering: {len(neighbors_df["target_id"].unique())}')

    # map celestial coordinates to CCD from tess-point for these neighbors
    log_or_print('Mapping celestial to pixel coordinates for neighbors...', logger)

    map_neighbors_loc_to_target_ccd_px_coordinates_sector_splits(
        neighbors_df,
        lc_sector_dir,
        map_px_res_dir,
        n_splits_objs,
        logger
    )

    # get filepaths, read csv files and concatenate them
    pattern = re.compile(r'neighbors_pxcoords_\d+\.csv')
    neighbors_px_coords_df = pd.concat([pd.read_csv(fp) for fp in map_px_res_dir.glob('*.csv')
                                        if pattern.match(fp.name)], axis=0, ignore_index=True)
    log_or_print(f'Found {len(neighbors_px_coords_df)} neighbors. Saving...', logger)
    neighbors_px_coords_df.to_csv(map_px_res_dir / f'neighbors_pxcoords_S{sector}.csv', index=False)
    log_or_print(f'Created table with pixel coordinates for {len(neighbors_px_coords_df)} neighbors of the '
                 f'{len(neighbors_df["target_id"].unique())} targets.', logger)

    log_or_print(f'Finished run for targets in S{sector}.', logger)


if __name__ == "__main__":

    neighbors_dir = Path('/nobackupp19/msaragoc/work_dir/Kepler-TESS_exoplanet/experiments/search_neighboring_stars/tess_spoc_2min_s1-s88_search_radius_arcsec_168.0_tpf_wcs_4-3-2025_1233')
    n_procs = 36  # multiprocessing parameters
    use_logs = True
    n_splits_objs = 20
    lc_root_dir = Path('/nobackup/msaragoc/work_dir/Kepler-TESS_exoplanet/data/FITS_files/TESS/spoc_2min/lc')
    filter_target_tbl = pd.read_csv('/u/msaragoc/work_dir/Kepler-TESS_exoplanet/experiments/search_neighboring_stars/tess_spoc_2min_s1-s88_search_radius_arcsec_168.0_tpf_wcs_4-3-2025_1233/missing_targets_lcs_4-14-2025_1016.csv')

    neighbors_subdirs = list(neighbors_dir.glob('S*'))
    # neighbors_subdirs = list(neighbors_dir.glob('S65'))
    neighbors_tbls_fps = [list(neighbors_subdir.glob('neighbors_S*.csv'))[0] for neighbors_subdir in neighbors_subdirs]
    print(f'Found {len(neighbors_tbls_fps)} neighbors table(s).')

    jobs = [(neighbors_tbl_fp,
            #  lc_root_dir / f's{neighbors_tbl_fp.stem.split("_")[1][1:].zfill(4)}',  # for FFI data
             lc_root_dir / f'sector_{neighbors_tbl_fp.stem.split("_")[1][1:]}',  # for 2-min data
             n_splits_objs,
             use_logs,
             filter_target_tbl,
             )
            for neighbors_tbl_fp in neighbors_tbls_fps]
    n_jobs = len(jobs)
    print(f'Split work into {n_jobs} job(s).')

    # parallelize jobs
    n_procs = min(n_jobs, n_procs)
    pool = multiprocessing.Pool(processes=n_procs)
    async_results = [pool.apply_async(map_neighbors_loc_to_target_ccd_px_coordinates_sector_main, job) for job in jobs]
    pool.close()
    pool.join()

    # for job in jobs:
    #     map_neighbors_loc_to_target_ccd_px_coordinates_sector_main(*job)

    print(f'Finished.')
