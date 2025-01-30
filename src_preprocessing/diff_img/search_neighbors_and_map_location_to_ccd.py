"""
Get neighbors in a cone search for a set of targets, and map their celestial coordinates to CCD pixel frame for the
sectors these neighbors were observed.
"""

# 3rd party
from astroquery.mast import Catalogs
import astropy.units as u
import numpy as np
from pathlib import Path
import pandas as pd
# import subprocess
import multiprocessing
import logging
from tess_stars2px import tess_stars2px_function_entry


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


def get_neighbors_in_search_radius(targets, search_radius_arcsec, logger=None):
    """ Get neighbors in search radius `search_radius_arcsec` around each target in `targets`.

    Args:
        targets: list, list of targets to perform search
        search_radius_arcsec: astropy quantity, search radius in arcsec for cone search
        logger: logger

    Returns:
        search_res: pandas DataFrame, table with neighbors
    """

    # perform the cone search
    search_res_lst = []
    for target_i, target in enumerate(targets):
        if target_i % 100 == 0:
            log_or_print(f'Iterated through {target_i}/{len(targets)} targets.', logger)

        search_res_target = Catalogs.query_object(catalog="TIC", objectname=f'TIC{target}',
                                               radius=search_radius_arcsec.to(u.degree))
        search_res_target = search_res_target.to_pandas()
        search_res_target['ID'] = search_res_target['ID'].astype(int)
        search_res_target['target_id'] = target
        # set magnitudes relative to the target star
        target_idx = search_res_target['ID'] == target
        # search_res_target['Tmag_rel'] = search_res_target['Tmag'] / search_res_target.loc[target_idx, 'Tmag'].values[0]
        search_res_target = search_res_target.loc[~target_idx]  # exclude own target
        # log_or_print(f'Found {len(search_res_target)} neighbors in a search radius {search_radius_arcsec} '
        #              f'arcsec for target {target}.')
        search_res_lst.append(search_res_target)

    log_or_print(f'Iterated through all {len(targets)}.', logger)
    search_res = pd.concat(search_res_lst, axis=0, ignore_index=True)
    log_or_print(f'Concatenated search results into a single table for all targets. Found {len(search_res)} '
                 f'neighbors.', logger)

    return search_res


def filter_neighbors_by_magnitude(neighbors_tbl, mag_thr):
    """ Exclude neighbors whose magnitude exceeds `mag_thr`.

    Args:
        neighbors_tbl: pandas DataFrame, table with neighbors
        mag_thr: float, magnitude threshold

    Returns: pandas DataFrame, table with targets filtered by magnitude threshold

    """

    return neighbors_tbl.loc[neighbors_tbl['Tmag'] < mag_thr]


def create_targets_to_neighbors_data(neighbors_tbl, save_fp):
    """ Create dictionary that maps targets to their neighbors. The dictionary maps target ID to a subdictionary with
    their neighbors. Each neighbor ID maps to a dictionary containing 'dstArcSec', the distance between the two starts
    in arcseconds, and 'mag`, the magnitude of the neighboring star.

    Args:
        neighbors_tbl: pandas DataFrame, table with neighbors. Must contain column 'target_id' for which the search was
        performed.
        save_fp: Path, save filepath

    Returns:

    """

    targets_dict = {target_id: {neighbor_id: {'dstArcSec': neighbor_dst_arc_sec, 'mag': neighbor_mag}
                                for neighbor_id, neighbor_dst_arc_sec, neighbor_mag in
                                zip(neighbors_target['ID'].tolist(), neighbors_target['dstArcSec'].tolist(),
                                    neighbors_target['Tmag'].tolist())}
                    for target_id, neighbors_target in neighbors_tbl.groupby('target_id')}

    np.save(save_fp, targets_dict)

    # targets_dict_df = {'target_id': [], 'neighbors': [], 'dstArcSec': [], 'Tmag_rel': []}
    # for target_id, neighbors_target in neighbors_tbl.groupby('target_id'):
    #     targets_dict_df['target_id'].append(target_id)
    #     targets_dict_df['neighbors'].append(neighbors_target['ID'].tolist())
    #     targets_dict_df['dstArcSec'].append(neighbors_target['dstArcSec'].tolist())
    #     targets_dict_df['Tmag_rel'].append(neighbors_target['Tmag_rel'].tolist())
    #
    # targets_df = pd.DataFrame(targets_dict_df)
    #
    # targets_df.to_csv(save_fp.parent / f'{save_fp.stem}.csv', index=False)


def map_star_celestial_coordinates_to_ccd(stars_tbl):
    """ Maps celestial coordinates of stars (RA and Dec) in degrees to the pixel coordinates for each star in
    one/multiple observed sectors (columns 'col_px' and 'row_px').

    Args:
        stars_tbl: pandas DataFrame, table with stars. Must contain columns 'ID', 'ra', and 'dec', with RA and
        Dec in degrees
        # save_fp: Path, save filepath
        # logger: logger

    Returns:
        stars_px_coords_tbl: pandas DataFrame, CCD pixel coordinates for each star in one/multiple observed sectors

    """

    # input_fp = save_fp.parent / f'{save_fp.stem}_input_to_tess_point.txt'

    stars_ids, outEclipLong, outEclipLat, outSec, outCam, outCcd, outColPix, outRowPix, scinfo = (
        tess_stars2px_function_entry(stars_tbl['ID'], stars_tbl['ra'], stars_tbl['dec']))

    stars_px_coords_tbl = pd.DataFrame(
        {'ID': stars_ids,
         'col_px': outColPix,
         'row_px': outRowPix,
         }
    )

    return stars_px_coords_tbl

    # targets_tbl[['ID', 'ra', 'dec']].to_csv(input_fp, header=None, index=None, sep=' ')
    # output_fp = save_fp.parent / f'{save_fp.stem}_output_to_tess_point.txt'  # define output filepath

    # subprocess_res = subprocess.run(['python', '-m', 'tess_stars2px', '-f', str(input_fp), '-o', str(output_fp)],
    #                                 capture_output=True, text=True)
    # if subprocess_res.stdout:
    #     log_or_print(subprocess_res.stdout, logger)
    # if subprocess_res.stderr:
    #     log_or_print(subprocess_res.stderr, logger)

    # # convert txt file to dataframe
    # tess_point_columns = ['ID', 'ra_tess_point', 'dec_tess_point', 'eclip_long', 'eclip_lat', 'sector', 'camera',
    #                       'detector', 'col_px', 'row_px', 'sc_info']

    # tess_point_output = pd.read_csv(output_fp, header=None, skiprows=16, sep='|', names=tess_point_columns)

    # # delete tess-point files
    # log_or_print('Deleting tess-point files.', logger)
    # input_fp.unlink()
    # output_fp.unlink()

    # return tess_point_output


def create_neighbors_table(neighbors_tbl, save_fp, logger=None):
    """ Create neighbors table that contains the pixel coordinates for each neighbor in different sectors
    (columns 'col_px' and 'row_px').

    Args:
        neighbors_tbl: pandas DataFrame, table with neighbors. Must contain columns 'ID', 'ra', and 'dec', with RA and
        Dec in degrees
        save_fp: Path, save filepath
        logger: logger

    Returns:

    """

    neighbors = neighbors_tbl.drop_duplicates('ID')  # drop duplicates
    neighbors = neighbors.drop(columns=['dstArcSec', 'target_id'])
    log_or_print(f'Found {len(neighbors)} unique neighbors.', logger)

    # map celestial coordinates to CCD from tess-point for these neighbors
    # create txt file as input to tess-point
    log_or_print('Getting pixel coordinates for neighbors using tess-point...', logger)
    stars_px_coords_tbl = map_star_celestial_coordinates_to_ccd(neighbors)

    # add pixel coordinates to neighbors table
    log_or_print(f'Adding pixel coordinates to neighbors...', logger)
    neighbors['ID'] = neighbors['ID'].astype(int)
    neighbors_px_coords = stars_px_coords_tbl.merge(neighbors, how='left', on='ID', validate='many_to_one')
    neighbors_px_coords.to_csv(save_fp, index=False)


def get_ccd_coords_neighbors_targets(targets, search_radius_arcsec, mag_thr, res_dir, res_suffix='', use_logs=True):
    """ Pipeline to search for neighbors within `search_radius_arcsec` arcseconds of the `targets` and map their
    celestial coordinates to the pixel CCD frame.

    Args:
        targets: list, list of target IDs
        search_radius_arcsec: astropy quantity, search radius in arcseconds
        mag_thr: float, magnitude threshold
        res_dir: Path, directory to save output files
        res_suffix: str, suffix to add to the output filenames
        use_logs: bool, if True writes information about run to logs. Otherwise, it prints them to the console

    Returns:

    """

    if use_logs:
        log_dir = res_dir / 'logs'
        log_dir.mkdir(exist_ok=True)

        # set up logger
        logger = logging.getLogger(name=f'targets_S{res_suffix}')
        logger_handler = logging.FileHandler(filename=log_dir / f'targets_S{res_suffix}.log', mode='w')
        logger_formatter = logging.Formatter('%(asctime)s - %(message)s')
        logger.setLevel(logging.INFO)
        logger_handler.setFormatter(logger_formatter)
        logger.addHandler(logger_handler)
        logger.info(f'Started run for targets in S{res_suffix}.')
        logger.info(f'Number of targets: {len(targets)}')
    else:
        logger = None

    log_or_print(f'Searching for neighbors in a radius of {search_radius_arcsec} arcsec '
                 f'for {len(targets)} targets...', logger)
    search_res_targets = get_neighbors_in_search_radius(targets, search_radius_arcsec, logger)

    log_or_print(f'Filtering neighbors based on magnitude: magnitude threshold = {mag_thr}...', logger)
    search_res_tics_mag = filter_neighbors_by_magnitude(search_res_targets, mag_thr)

    # save_fp_targets = res_dir / f'targets{tbl_suffix}_search_radius_{search_radius_arcsec.value}_mag_thr_{mag_thr}.npy'
    log_or_print('Creating target data based on search results...', logger)
    save_fp_targets = res_dir / f'targets_S{res_suffix}.npy'
    create_targets_to_neighbors_data(search_res_tics_mag, save_fp_targets)

    # save_fp_neighbors = res_dir / f'neighbors{tbl_suffix}_search_radius_{search_radius_arcsec.value}_mag_thr_{mag_thr}.csv'
    log_or_print('Creating table with neighbors...', logger)
    save_fp_neighbors = res_dir / f'neighbors_S{res_suffix}.csv'
    create_neighbors_table(search_res_tics_mag, save_fp_neighbors, logger)

    log_or_print(f'Finished run for targets in S{res_suffix}.', logger)


if __name__ == "__main__":

    # load tce table
    tce_tbl_cols = [
        'target_id',
        'sectors_observed',
    ]
    tce_tbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/TESS/DV_SPOC_mat_files/preprocessing_tce_tables/09-25-2023_1608/tess_2min_tces_dv_s1-s68_all_msectors_11-29-2023_2157_newlabels_nebs_npcs_bds_ebsntps_to_unks_sg1master_allephemmatches_exofoptois.csv',
                          usecols=tce_tbl_cols)
    # search parameters
    search_radius_arcsec = 168 * u.arcsec  # np.sqrt(2) * 121 / 2 * u.arcsec  # 168 DV (Joe)
    mag_thr = np.inf
    # results directory
    res_dir = Path(f'/home/msaragoc/Projects/exoplnt_dl/experiments/search_neighboring_stars/tess_spoc_2min_s1-s68_search_radius_arcsec_{search_radius_arcsec.value}_mag_thr_{mag_thr}_1-29-2025_1609')
    # multiprocessing parameters
    # n_jobs = 1
    n_procs = 14

    use_logs = True

    res_dir.mkdir(parents=True, exist_ok=True)

    # get sectors observed for each target in the table
    targets_dict = {'target_id': [], 'sector': []}
    targets = tce_tbl.groupby('target_id')
    for target_id, target_data in targets:
        obs_sectors_target = np.nonzero(np.prod(
            target_data['sectors_observed'].apply(lambda x: np.array([int(el) for el in x])).values, axis=0))[0]
        for obs_sector in obs_sectors_target:
            targets_dict['target_id'].append(target_id)
            targets_dict['sector'].append(obs_sector)

    targets_df = pd.DataFrame(targets_dict)
    targets_df.to_csv(res_dir / 'target_sector_pairs.csv', index=False)
    print(f'Found {len(targets_df)} target-sector pairs.')

    # targets_df = targets_df.loc[targets_df['sector'] == 2][:3]

    # # get unique targets in tce table
    # tics = tce_tbl['target_id'].unique()
    # print(f'There are {len(tics)} targets in the TCE table.')

    # get_ccd_coords_neighbors_targets(tics, search_radius_arcsec, mag_thr, res_dir, '0')

    # split based on targets
    # tics_jobs = np.array_split(tics, n_jobs)

    # split based on sector runs
    tics_jobs = {sector: targets_srun['target_id']
                 for sector, targets_srun in targets_df.groupby('sector')}
    print(f'Found {len(tics_jobs)} sectors.')

    n_jobs = len(tics_jobs)
    print(f'Split work into {n_jobs} jobs.')

    # parallelize jobs
    pool = multiprocessing.Pool(processes=n_procs)
    # jobs = [(tics_job, search_radius_arcsec, mag_thr, res_dir, str(tics_job_i))
    #         for tics_job_i, tics_job in enumerate(tics_jobs)]
    jobs = [(tics_job_sector_run, search_radius_arcsec, mag_thr, res_dir, str(sector_run), use_logs)
            for sector_run, tics_job_sector_run in tics_jobs.items()]
    async_results = [pool.apply_async(get_ccd_coords_neighbors_targets, job) for job in jobs]
    pool.close()
    pool.join()

    # for job in jobs:
    #     get_ccd_coords_neighbors_targets(*job)

    print(f'Finished.')
