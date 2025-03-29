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
# from tess_stars2px import tess_stars2px_function_entry
import lightkurve as lk
import shutil
# import time


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


def get_neighbors_in_search_radius(targets, search_radius_arcsec, mag_thr=np.inf, logger=None):
    """ Get neighbors in search radius `search_radius_arcsec` around each target in `targets`.

    Args:
        targets: pandas DataFrame, table of targets ('target_id') to perform search
        search_radius_arcsec: astropy quantity, search radius in arcsec for cone search
        mag_thr: float, magnitude threshold
        logger: logger

    Returns:
        search_res: pandas DataFrame, table with neighbors
    """

    # perform the cone search
    search_res_lst = []
    for target_i, target_data in targets.iterrows():
        if target_i % 100 == 0:
            log_or_print(f'Iterated through {target_i}/{len(targets)} targets.', logger)

        target_id = int(target_data['target_id'])

        search_res_target = Catalogs.query_object(catalog="TIC", objectname=f'TIC{target_id}',
                                                  radius=search_radius_arcsec.to(u.degree))
        search_res_target = search_res_target.to_pandas()
        search_res_target['ID'] = search_res_target['ID'].astype(int)
        search_res_target['target_id'] = target_id
        target_idx = search_res_target['ID'] == target_id
        search_res_target = search_res_target.loc[~target_idx]  # exclude own target

        # log_or_print(f'Filtering neighbors based on magnitude: magnitude threshold = {mag_thr}...', logger)
        search_res_target = filter_neighbors_by_magnitude(search_res_target, target_data['mag'] + mag_thr)

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
    """ Create dictionary that maps targets to their neighbors. The dictionary maps target ID to a sub-dictionary with
    their neighbors. Each neighbor ID maps to a dictionary containing 'dstArcSec', the distance between the two starts
    in arcseconds, and 'mag`, the magnitude of the neighboring star.

    Args:
        neighbors_tbl: pandas DataFrame, table with neighbors. Must contain column 'target_id' for which the search was
            performed, and for each neighbor of the target, its 'ID', the distance of the neighbor to the target in
            arcseconds, 'dstArcSec', its magnitude 'Tmag', and RA and Dec coordinates 'ra' and 'dec'
        save_fp: Path, save filepath

    Returns:

    """

    targets_dict = {target_id: {neighbor_id: {'dstArcSec': neighbor_dst_arc_sec, 'mag': neighbor_mag, 'ra': neighbor_ra,
                                              'dec': neighbor_dec}
                                for neighbor_id, neighbor_dst_arc_sec, neighbor_mag, neighbor_ra, neighbor_dec in
                                zip(neighbors_target['ID'].tolist(), neighbors_target['dstArcSec'].tolist(),
                                    neighbors_target['Tmag'].tolist(), neighbors_target['ra'].tolist(),
                                    neighbors_target['dec'].tolist())}
                    for target_id, neighbors_target in neighbors_tbl.groupby('target_id')}

    np.save(save_fp, targets_dict)


def map_star_celestial_coordinates_to_ccd(stars_tbl, temp_fits_dir, sector=None, logger=None):
    """ Maps celestial coordinates of stars (RA and Dec) in degrees to the pixel coordinates for each star in
    one/multiple observed sectors (columns 'col_px' and 'row_px').

    Args:
        stars_tbl: pandas DataFrame, table with stars. Must contain columns 'target_id' (in case the target pixel files
            of the target stars are being used), 'ID' (of the neighboring object), 'ra' and 'dec' coordinates of the
            neighboring object, with RA and Dec in degrees
        temp_fits_dir: Path, temporary directory to store target pixel FITS files
        sector: int, sector for which to get the pointing. If `None`, it considers all available sectors
        logger: logger

    Returns:
        stars_px_coords_tbl: pandas DataFrame, CCD pixel coordinates for each star in one/multiple observed sectors

    """

    # using WCS transformation stored in the target pixel files for each object
    outColPix, outRowPix, stars_ids = [], [], stars_tbl['ID']
    target_grps = stars_tbl.groupby('target_id')
    for target_i, (target_id, objs) in enumerate(target_grps):

        if target_i % 100 == 0:
            log_or_print(f'Iterated through {target_i}/{len(target_grps)} targets.', logger)

        search_result = lk.search_targetpixelfile(target=f'tic{target_id}', mission='TESS',
                                                  author=('TESS-SPOC', 'SPOC'),
                                                  exptime=120, sector=sector)

        if len(search_result) == 0:
            log_or_print(f'No target pixel files found for target {target_id} in sector {sector}. Setting pixel '
                         f'coordinates to NaN.', logger)
            outColPix += [np.nan] * len(objs)
            outRowPix += [np.nan] * len(objs)
            continue

        if len(search_result) > 1:
            log_or_print(f'Found {len(search_result)} target pixel files for target {target_id} in sector '
                         f'{sector}. Considering only first result:\n {search_result.table}', logger)

        # Download the first search result
        tpf = search_result.download(download_dir=str(temp_fits_dir), quality_bitmask='default')

        # use WCS transformation to map objects to CCD pixel frame
        outColPix_obj, outRowPix_obj = tpf.wcs.all_world2pix(objs['ra'], objs['dec'], 0)

        outColPix += list(outColPix_obj)
        outRowPix += list(outRowPix_obj)

        # delete target pixel file
        shutil.rmtree(Path(tpf.path).parent, ignore_errors=True)

    # # using tess-point
    # # initialize scinfo
    # _, _, _, _, _, _, _, _, scinfo = tess_stars2px_function_entry(stars_tbl['ID'][0], stars_tbl['ra'][0],
    #                                                               stars_tbl['dec'][0], trySector=sector)
    #
    # stars_ids, _, _, _, _, _, outColPix, outRowPix, scinfo = (
    #     tess_stars2px_function_entry(stars_tbl['ID'], stars_tbl['ra'], stars_tbl['dec'], scInfo=scinfo,
    #                                  trySector=sector))

    log_or_print(f'Iterated through all {len(target_grps)} targets.', logger)

    stars_px_coords_tbl = pd.DataFrame(
        {'ID': stars_ids,
         'col_px': outColPix,
         'row_px': outRowPix,
         }
    )

    log_or_print(f'Created table with pixel coordinates for the neighbors of the {len(target_grps)} targets.',
                 logger)

    return stars_px_coords_tbl


def create_neighbors_table(neighbors_tbl, save_fp, temp_fits_dir, sector=None, logger=None):
    """ Create neighbors table that contains the pixel coordinates for each neighbor in different sectors
    (columns 'col_px' and 'row_px').

    Args:
        neighbors_tbl: pandas DataFrame, table with neighbors. Must contain columns 'ID', 'ra', and 'dec', with RA and
        Dec in degrees
        save_fp: Path, save filepath
        temp_fits_dir: Path, temporary directory to store target pixel FITS files
        sector: int, sector for which to get the pointing. If `None`, it considers all available sectors
        logger: logger

    Returns:

    """

    neighbors = neighbors_tbl.drop_duplicates('ID')  # drop duplicates
    neighbors = neighbors.drop(columns=['dstArcSec'])  # , 'target_id'])
    log_or_print(f'Found {len(neighbors)} unique neighbors.', logger)

    # map celestial coordinates to CCD from tess-point for these neighbors
    log_or_print('Getting pixel coordinates for neighbors...', logger)
    stars_px_coords_tbl = map_star_celestial_coordinates_to_ccd(neighbors, temp_fits_dir, sector, logger)

    # add pixel coordinates to neighbors table
    log_or_print(f'Adding pixel coordinates to neighbors...', logger)
    neighbors['ID'] = neighbors['ID'].astype(int)
    neighbors_px_coords = stars_px_coords_tbl.merge(neighbors, how='left', on='ID', validate='many_to_one')
    neighbors_px_coords.to_csv(save_fp, index=False)


def get_ccd_coords_neighbors_targets(targets, search_radius_arcsec, mag_thr, res_dir, temp_fits_dir, res_suffix='',
                                     use_logs=True):
    """ Pipeline to search for neighbors within `search_radius_arcsec` arcseconds of the `targets` and map their
    celestial coordinates to the pixel CCD frame.

    Args:
        targets: pandas DataFrame, table of targets
        search_radius_arcsec: astropy quantity, search radius in arcseconds
        mag_thr: float, magnitude threshold
        res_dir: Path, directory to save output files
        temp_fits_dir: Path, temporary directory to store target pixel FITS files
        res_suffix: str, suffix to add to the output filenames
        use_logs: bool, if True writes information about run to logs. Otherwise, it prints them to the console

    Returns:

    """

    if use_logs:
        log_dir = res_dir / 'logs'
        log_dir.mkdir(exist_ok=True)

        # set up logger
        logger = logging.getLogger(name=f'targets_S{res_suffix}')
        logger_handler = logging.FileHandler(filename=log_dir / f'targets_S{res_suffix}.log', mode='a')
        logger_formatter = logging.Formatter('%(asctime)s - %(message)s')
        logger.setLevel(logging.INFO)
        logger_handler.setFormatter(logger_formatter)
        logger.addHandler(logger_handler)
        logger.info(f'Started run for targets in S{res_suffix}.')
        logger.info(f'Number of targets: {len(targets)}')
    else:
        logger = None

    if (res_dir / f'targets_S{res_suffix}.npy').exists():
        log_or_print(f'Found NumPy file '
                     f'for targets in sector {res_suffix}: {res_dir / f"targets_S{res_suffix}.npy"}. Loading data...',
                     logger)
        targets_dict = np.load(res_dir / f'targets_S{res_suffix}.npy', allow_pickle=True).item()
        search_res_targets = pd.DataFrame.from_dict({(i, j): targets_dict[i][j]
                                                     for i in targets_dict.keys()
                                                     for j in targets_dict[i].keys()},
                                                    orient='index')
        search_res_targets.index.names = ['target_id', 'ID']
        search_res_targets = search_res_targets.reset_index()

    else:
        log_or_print(f'Searching for neighbors in a radius of {search_radius_arcsec} arcsec '
                     f'for {len(targets)} targets...', logger)

        search_res_targets = get_neighbors_in_search_radius(targets, search_radius_arcsec, mag_thr, logger)

        # log_or_print(f'Filtering neighbors based on magnitude: magnitude threshold = {mag_thr}...', logger)
        # search_res_tics_mag = filter_neighbors_by_magnitude(search_res_targets, mag_thr)

        log_or_print('Creating target data based on search results...', logger)
        save_fp_targets = res_dir / f'targets_S{res_suffix}.npy'
        create_targets_to_neighbors_data(search_res_targets, save_fp_targets)

    log_or_print('Creating table with neighbors...', logger)
    save_fp_neighbors = res_dir / f'neighbors_S{res_suffix}.csv'
    create_neighbors_table(search_res_targets, save_fp_neighbors, temp_fits_dir, int(res_suffix), logger)

    log_or_print(f'Finished run for targets in S{res_suffix}.', logger)


if __name__ == "__main__":

    # load tce table
    tce_tbl_cols = [
        'target_id',
        'sectors_observed',
        'mag',
        # 'uid'
    ]

    temp_tpf_dir = Path('/nobackupp19/msaragoc/work_dir/Kepler-TESS_exoplanet/experiments/search_neighboring_stars/')

    tce_tbl = pd.read_csv('/nobackupp19/msaragoc/work_dir/Kepler-TESS_exoplanet/data/Ephemeris_tables/TESS/tess_2min_tces_dv_s1-s68_all_msectors_11-29-2023_2157_newlabels_nebs_npcs_bds_ebsntps_to_unks_sg1master_allephemmatches_exofoptois.csv',
                          usecols=tce_tbl_cols)
    # tce_tbl = tce_tbl.loc[tce_tbl['uid'] == '182588086-1-S34']

    # search parameters
    search_radius_arcsec = 168 * u.arcsec  # np.sqrt(2) * 121 / 2 * u.arcsec  # 168 DV (Joe)
    mag_thr = np.inf
    # results directory
    # res_dir = Path(f'/Users/msaragoc/Projects/exoplanet_transit_classification/experiments/search_neighboring_stars/tess_spoc_2min_s1-s68_search_radius_arcsec_{search_radius_arcsec.value}_tess-point_2-12-2025_2114')
    res_dir = Path('/nobackupp19/msaragoc/work_dir/Kepler-TESS_exoplanet/experiments/search_neighboring_stars/tess_spoc_2min_s1-s68_search_radius_arcsec_168.0_tpf_wcs_2-28-2025_1217')
    # multiprocessing parameters
    # n_jobs = 1
    n_procs = 60

    use_logs = True

    res_dir.mkdir(parents=True, exist_ok=True)

    # get sectors observed for each target in the table
    if (res_dir / 'target_sector_pairs.csv').exists():
        print('Reading table for target-sector pairs already created.')
        targets_df = pd.read_csv(res_dir / 'target_sector_pairs.csv')
    else:
        targets_dict = {'target_id': [], 'sector': [], 'mag': []}
        targets = tce_tbl.groupby('target_id')
        # get observed sectors for each target based on its TCEs
        for target_id, target_data in targets:
            target_data = target_data.reset_index(drop=True)
            obs_sectors_target = np.nonzero(np.sum(
                target_data['sectors_observed'].apply(lambda x: np.array([int(el) for el in x])).values, axis=0))[0]
            for obs_sector in obs_sectors_target:
                targets_dict['target_id'].append(target_id)
                targets_dict['sector'].append(obs_sector)
                targets_dict['mag'].append(target_data.loc[0, 'mag'])

        targets_df = pd.DataFrame(targets_dict)
        targets_df.to_csv(res_dir / 'target_sector_pairs.csv', index=False)

    print(f'Found {len(targets_df)} target-sector pairs.')

    # targets_df = targets_df.loc[targets_df['sector'] == 34]
    # targets_df = targets_df.loc[targets_df['uid'] == '182588086-1-S34']

    # find sectors without neighbors results (when run did not finish or there was an error)
    sectors_finished = [int(el.stem.split('_')[-1][1:]) for el in res_dir.glob('neighbors_S*.csv')]

    # split based on sector runs
    tics_jobs = {sector: targets_srun.reset_index(drop=True)
                 for sector, targets_srun in targets_df.groupby('sector') if sector not in sectors_finished}
    print(f'Found {len(tics_jobs)} sectors.')

    n_jobs = len(tics_jobs)
    print(f'Split work into {n_jobs} jobs.')

    # parallelize jobs
    n_procs = min(n_jobs, n_procs)
    pool = multiprocessing.Pool(processes=n_procs)
    jobs = [(tics_job_sector_run, search_radius_arcsec, mag_thr, res_dir, temp_tpf_dir, str(sector_run), use_logs)
            for sector_run, tics_job_sector_run in tics_jobs.items()]
    async_results = [pool.apply_async(get_ccd_coords_neighbors_targets, job) for job in jobs]
    pool.close()
    pool.join()

    # for job in jobs:
    #     get_ccd_coords_neighbors_targets(*job)

    print(f'Finished.')
