"""
Get neighbors in a cone search for a set of targets, and map their celestial coordinates to CCD pixel frame for the
sectors these neighbors were observed. Targets table must contain column 'target_id' with the TIC IDs of the targets for
which to search for neighboring objects.
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
import re

QUERY_OBJECT_COLUMNS = [
    'ID',
    'Tmag',
    'dstArcSec',
    'ra',
    'dec',
]

RENAME_QUERY_COLUMNS = {
    # 'ID': 'neighbor_ticid',
    # 'mag':
}

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


def get_neighbors_in_search_radius(targets, search_radius_arcsec, mag_thr=np.inf):
    """ Get neighbors in search radius `search_radius_arcsec` around each target in `targets`.

    Args:
        targets: pandas DataFrame, table of targets ('target_id') to perform search
        search_radius_arcsec: astropy quantity, search radius in arcsec for cone search
        mag_thr: float, magnitude threshold

    Returns:
        search_res: pandas DataFrame, table with neighbors. Returns `None` is `save_subtbl_fp` is not `None`
    """

    # perform the cone search
    search_res_lst, subtbl_i = [], 0
    for target_i, target_data in targets.iterrows():

        # if target_i % 100 == 0:
        #     log_or_print(f'Iterated through {target_i}/{len(targets)} targets.', logger)

        target_id = int(target_data['target_id'])

        search_res_target = Catalogs.query_object(catalog="TIC", objectname=f'TIC{target_id}',
                                                  radius=search_radius_arcsec.to(u.degree),
                                                  )
        search_res_target = search_res_target.to_pandas()
        
        search_res_target = search_res_target[QUERY_OBJECT_COLUMNS]

        search_res_target['ID'] = search_res_target['ID'].astype(int)

        search_res_target['target_id'] = target_id
        target_idx = search_res_target['ID'] == target_id
        search_res_target = search_res_target.loc[~target_idx]  # exclude own target

        # rename columns
        search_res_target = search_res_target.rename(RENAME_QUERY_COLUMNS)

        # log_or_print(f'Filtering neighbors based on magnitude: magnitude threshold = {mag_thr}...', logger)
        search_res_target = filter_neighbors_by_magnitude(search_res_target, mag_thr)

        search_res_lst.append(search_res_target)

    search_res = pd.concat(search_res_lst, axis=0, ignore_index=True)

    return search_res


def get_neighbors_in_search_radius_splits(targets, search_radius_arcsec, save_subtbl_dir_fp, n_splits_targets,
                                          mag_thr=np.inf, logger=None):
    """ Get neighbors in search radius `search_radius_arcsec` around each target in `targets`.

    Args:
        targets: pandas DataFrame, table of targets ('target_id') to perform search
        search_radius_arcsec: astropy quantity, search radius in arcsec for cone search
        save_subtbl_dir_fp: Path, if provided, it will save a table with the search results for the last a target split
            in `save_subtbl_dir_fp` directory name 'neighbors_{target_split_i}.csv'. For each table, it adds a suffix
            integer starting at zero representing the target split the table referred to.
        n_splits_targets: int, split search results into multiple tables by splitting targets into `n_splits_targets`
        mag_thr: float, magnitude threshold
        logger: logger

    Returns:
        search_res: pandas DataFrame, table with neighbors. Returns `None` is `save_subtbl_fp` is not `None`
    """

    targets_splits = np.array_split(targets, n_splits_targets)

    for target_split_i, target_split in enumerate(targets_splits):

        log_or_print(f'Iterated through {len(target_split)}/{len(target_split)} targets in split '
                     f'{target_split_i} (out of {len(targets)} in total) split across {n_splits_targets} splits',
                     logger)

        if len(target_split) == 0:
            continue

        if (save_subtbl_dir_fp / f'neighbors_{target_split_i}.csv').exists():
            log_or_print(f'Found table for target split {target_split_i} (out of {n_splits_targets} splits). '
                         f'Skipping {len(target_split)} targets in this split...', logger)
            continue

        neighbors_split = get_neighbors_in_search_radius(target_split, search_radius_arcsec, mag_thr=mag_thr)

        log_or_print(f'Saving subset table of results...', logger)
        neighbors_split.to_csv(save_subtbl_dir_fp / f'neighbors_{target_split_i}.csv', index=False)

    log_or_print(f'Iterated through all {len(targets)} across {n_splits_targets} splits.', logger)


def filter_neighbors_by_magnitude(neighbors_tbl, mag_thr):
    """ Exclude neighbors whose magnitude exceeds `mag_thr`.

    Args:
        neighbors_tbl: pandas DataFrame, table with neighbors
        mag_thr: float, magnitude threshold

    Returns: pandas DataFrame, table with targets filtered by magnitude threshold

    """

    return neighbors_tbl.loc[neighbors_tbl['Tmag'] < mag_thr]


def get_neighbors_in_search_radius_main(targets, search_radius_arcsec, sector_id, res_dir, n_splits_targets,
                                        mag_thr=np.inf, use_logs=True):
    """ Pipeline to search for neighbors within `search_radius_arcsec` arcseconds of the `targets` in sector
    `sector_id`.

    Args:
        targets: pandas DataFrame, table of targets
        search_radius_arcsec: astropy quantity, search radius in arcseconds
        sector_id: str, sector ID like '1', '2', ...
        res_dir: Path, directory to save output files
        n_splits_targets: int, split search results into multiple tables by splitting targets into `n_splits_targets`
        mag_thr: float, magnitude threshold
        use_logs: bool, if True writes information about run to logs. Otherwise, it prints them to the console

    Returns:

    """

    # create directory to save sector search results
    sector_res_dir = res_dir / f'S{sector_id}'
    sector_res_dir.mkdir(parents=True, exist_ok=True)

    if use_logs:
        log_dir = res_dir / 'logs'
        log_dir.mkdir(exist_ok=True)

        # set up logger
        logger = logging.getLogger(name=f'targets_S{sector_id}')
        logger_handler = logging.FileHandler(filename=log_dir / f'targets_S{sector_id}.log', mode='a')
        logger_formatter = logging.Formatter('%(asctime)s - %(message)s')
        logger.setLevel(logging.INFO)
        logger_handler.setFormatter(logger_formatter)
        logger.addHandler(logger_handler)
        logger.info(f'Started run for targets in S{sector_id}.')
        logger.info(f'Number of targets: {len(targets)}')
    else:
        logger = None

    if (sector_res_dir / f'neighbors_S{sector_id}.csv').exists():

        log_or_print(f'Found neighbors table '
                     f'for targets in sector {sector_id}: {res_dir / f"neighbors_S{sector_id}.csv"}. Loading data...',
                     logger)

        search_res_targets_neighbors_df = pd.read_csv(sector_res_dir / f'neighbors_S{sector_id}.csv')

        log_or_print(f'Found {len(search_res_targets_neighbors_df)} neighbors.', logger)

    else:

        log_or_print(f'Searching for neighbors in a radius of {search_radius_arcsec} arcsec '
                     f'for {len(targets)} targets...', logger)

        get_neighbors_in_search_radius_splits(
            targets,
            search_radius_arcsec,
            sector_res_dir,
            n_splits_targets,
            mag_thr=mag_thr,
            logger=logger
        )

        log_or_print(f'Finished search for neighbors in a radius of {search_radius_arcsec} arcsec '
                     f'for {len(targets)} targets...', logger)

        log_or_print(f'Aggregating results tables into a neighbors table for the full sector in a search '
                     f'radius of {search_radius_arcsec} arcsec for {len(targets)} targets', logger)

        # get filepaths, read csv files and concatenate them
        pattern = re.compile(r'neighbors_\d+\.csv')
        search_res_targets_neighbors_df = pd.concat([pd.read_csv(fp) for fp in sector_res_dir.glob('*.csv')
                                                    if pattern.match(fp.name)], axis=0, ignore_index=True)
        log_or_print(f'Found {len(search_res_targets_neighbors_df)} neighbors. Saving...', logger)
        search_res_targets_neighbors_df.to_csv(sector_res_dir / f'neighbors_S{sector_id}.csv', index=False)
        log_or_print('Save neighbors table.', logger)

    log_or_print(f'Finished run for targets in S{sector_id}.', logger)


if __name__ == "__main__":

    # search parameters
    search_radius_arcsec = 168 * u.arcsec  # np.sqrt(2) * 121 / 2 * u.arcsec  # 168 DV (Joe)
    mag_thr = np.inf
    n_splits_targets = 20
    # results directory
    # res_dir = Path(f'/Users/msaragoc/Projects/exoplanet_transit_classification/experiments/search_neighboring_stars/tess_spoc_2min_s1-s68_search_radius_arcsec_{search_radius_arcsec.value}_tess-point_2-12-2025_2114')
    res_dir = Path('/nobackupp19/msaragoc/work_dir/Kepler-TESS_exoplanet/experiments/search_neighboring_stars/tess_spoc_2min_s1-s88_search_radius_arcsec_168.0_tpf_wcs_4-3-2025_1233')
    # multiprocessing parameters
    # n_jobs = 1
    n_procs = 6
    targets_tbl_fp = Path('/nobackupp19/msaragoc/work_dir/Kepler-TESS_exoplanet/experiments/search_neighboring_stars/target_sector_pairs_tess_2min_tces_dv_s1-s88_4-3-2025_1231.csv')
    use_logs = True

    res_dir.mkdir(parents=True, exist_ok=True)

    targets_tbl = pd.read_csv(targets_tbl_fp)

    print(f'Found {len(targets_tbl)} target-sector pairs.')

    # targets_df = targets_df.loc[targets_df['sector'] == 34]
    # targets_df = targets_df.loc[targets_df['uid'] == '182588086-1-S34']

    # find sectors without neighbors results (when run did not finish or there was an error)
    sectors_finished = [int(el.stem.split('_')[-1][1:]) for el in res_dir.rglob('targets_neighbors_pxcoords_S*.csv')]

    # split based on sector runs
    tics_jobs = {sector: targets_srun.reset_index(drop=True)
                 for sector, targets_srun in targets_tbl.groupby('sector') if sector not in sectors_finished}
    print(f'Found {len(tics_jobs)} sector(s).')

    jobs = [(tics_job_sector_run, search_radius_arcsec, str(sector_run), res_dir, n_splits_targets, mag_thr, use_logs)
            for sector_run, tics_job_sector_run in tics_jobs.items()]
    n_jobs = len(jobs)
    print(f'Split work into {n_jobs} job(s).')

    # parallelize jobs
    n_procs = min(n_jobs, n_procs)
    pool = multiprocessing.Pool(processes=n_procs)
    async_results = [pool.apply_async(get_neighbors_in_search_radius_main, job) for job in jobs]
    pool.close()
    pool.join()

    # for job in jobs:
    #     get_neighbors_in_search_radius_main(*job)

    print(f'Finished.')
