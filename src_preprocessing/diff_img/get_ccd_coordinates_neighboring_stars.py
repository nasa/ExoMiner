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
import subprocess
import multiprocessing


def get_neighbors_in_search_radius(targets, search_radius_arcsec):
    """ Get neighbors in search radius `search_radius_arcsec` around each target in `targets`.

    Args:
        targets: list, list of targets to perform search
        search_radius_arcsec: astropy quantity, search radius in arcsec for cone search

    Returns:
        search_res: pandas DataFrame, table with neighbors
    """

    # perform the cone search
    search_res_lst = []
    for target_i, target in enumerate(targets):
        if target_i % 100 == 0:
            print(f'Iterated through {target_i}/{len(targets)}')

        search_res_target = Catalogs.query_object(catalog="TIC", objectname=f'TIC{target}',
                                               radius=search_radius_arcsec.to(u.degree))
        search_res_target = search_res_target.to_pandas()
        search_res_target['ID'] = search_res_target['ID'].astype(int)
        search_res_target['target_id'] = target
        # set magnitudes relative to the target star
        target_idx = search_res_target['ID'] == target
        search_res_target['Tmag_rel'] = search_res_target['Tmag'] / search_res_target.loc[target_idx, 'Tmag'].values[0]
        search_res_target = search_res_target.loc[~target_idx]  # exclude own target
        search_res_lst.append(search_res_target)

    search_res = pd.concat(search_res_lst, axis=0, ignore_index=True)

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
    in arcseconds, and 'Tmag_rel`, the relative magnitude between the two targets (mag_neighbor / mag_target).

    Args:
        neighbors_tbl: pandas DataFrame, table with neighbors. Must contain column 'target_id' for which the search was
        performed.
        save_fp: Path, save filepath

    Returns:

    """

    targets_dict = {target_id: {neighbor_id: {'dstArcSec': neighbor_dst_arc_sec, 'Tmag_rel': neighbor_dst_arc_sec}
                                for neighbor_id, neighbor_dst_arc_sec, neighbor_mag_rel in
                                zip(neighbors_target['ID'].tolist(), neighbors_target['dstArcSec'].tolist(),
                                    neighbors_target['Tmag_rel'].tolist())}
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


def create_neighbors_table(neighbors_tbl, save_fp):
    """ Create neighbors table that contains the pixel coordinates for each neighbor in different sectors
    (columns 'col_px' and 'row_px').

    Args:
        neighbors_tbl: pandas DataFrame, table with neighbors. Must contain columns 'ID', 'ra', and 'dec', with RA and
        Dec in degrees
        save_fp: Path, save filepath

    Returns:

    """

    neighbors = neighbors_tbl.drop_duplicates('ID')  # drop duplicates
    neighbors = neighbors.drop(columns=['dstArcSec', 'target_id', 'Tmag_rel'])
    print(f'Found {len(neighbors)} unique neighbors.')

    # create txt file as input to tess-point
    input_fp = save_fp.parent / f'{save_fp.stem}_input_to_tess_point.txt'
    neighbors[['ID', 'ra', 'dec']].to_csv(input_fp, header=None, index=None, sep=' ')

    # get output from tess-point for these neighbors
    output_fp = save_fp.parent / f'{save_fp.stem}_output_to_tess_point.txt'
    _ = subprocess.run(['python', '-m', 'tess_stars2px', '-f', str(input_fp), '-o', str(output_fp)],
                       capture_output=True, text=True)

    # convert txt file to dataframe
    tess_point_columns = ['ID', 'ra_tess_point', 'dec_tess_point', 'eclip_long', 'eclip_lat', 'sector', 'camera',
                          'detector', 'col_px', 'row_px', 'col11']
    tess_point_output = pd.read_csv(output_fp, header=None, skiprows=16, sep='|', names=tess_point_columns)

    # add pixel coordinates to neighbors table
    neighbors['ID'] = neighbors['ID'].astype(int)
    tess_point_output = tess_point_output.merge(neighbors, how='left', on='ID', validate='many_to_one')
    tess_point_output.to_csv(save_fp, index=False)

    # delete tess-point files
    input_fp.unlink()
    output_fp.unlink()


def get_ccd_coords_neighbors_targets(targets, search_radius_arcsec, mag_thr, res_dir, tbl_suffix=''):
    """ Pipeline to search for neighbors within `search_radius_arcsec` arcseconds of the `targets` and map their
    celestial coordinates to the pixel CCD frame.

    Args:
        targets: list, list of target IDs
        search_radius_arcsec: astropy quantity, search radius in arcseconds
        mag_thr: float, magnitude threshold
        res_dir: Path, directory to save output files
        tbl_suffix: str, suffix to add to the output tables names

    Returns:

    """

    search_res_tics = get_neighbors_in_search_radius(targets, search_radius_arcsec)

    search_res_tics_mag = filter_neighbors_by_magnitude(search_res_tics, mag_thr)

    save_fp_targets = res_dir / f'targets{tbl_suffix}_search_radius_{search_radius_arcsec.value}_mag_thr_{mag_thr}.npy'
    create_targets_to_neighbors_data(search_res_tics_mag, save_fp_targets)

    save_fp_neighbors = res_dir / f'neighbors{tbl_suffix}_search_radius_{search_radius_arcsec.value}_mag_thr_{mag_thr}.csv'
    create_neighbors_table(search_res_tics_mag, save_fp_neighbors)


if __name__ == "__main__":

    # load tce table
    tce_tbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/TESS/DV_SPOC_mat_files/preprocessing_tce_tables/09-25-2023_1608/tess_2min_tces_dv_s1-s68_all_msectors_11-29-2023_2157_newlabels_nebs_npcs_bds_ebsntps_to_unks_sg1master_allephemmatches_exofoptois.csv')
    # search parameters
    search_radius_arcsec = 168 * u.arcsec  # np.sqrt(2) * 121 / 2 * u.arcsec  # 168 DV (Joe)
    mag_thr = np.inf
    # results directory
    res_dir = Path(f'~/Projects/exoplnt_dl/experiments/search_neighboring_stars/tess_spoc_2min_s1-s68_search_radius_arcsec_{search_radius_arcsec.value}_ mag_thr_{mag_thr}_1-23-2025_1246')
    # multiprocessing parameters
    n_jobs = 256
    n_procs = 16

    res_dir.mkdir(parents=True, exist_ok=True)

    # get unique targets in tce table
    tics = tce_tbl['target_id'].unique()
    print(f'There are {len(tics)} targets in the TCE table.')

    tics_jobs = np.array_split(tics, n_jobs)
    print(f'Split work into {n_jobs} jobs.')

    # parallelize jobs
    pool = multiprocessing.Pool(processes=n_procs)
    jobs = [(tics_job, search_radius_arcsec, mag_thr, res_dir, str(tics_job_i))
            for tics_job_i, tics_job in enumerate(tics_jobs)]
    async_results = [pool.apply_async(get_ccd_coords_neighbors_targets, job) for job in jobs]
    pool.close()
    pool.join()
