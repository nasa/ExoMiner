"""
Filter TIC lc sh files for TESS 2-min/FFI data based on a target table that contains columns 'target_id' and 'sector' with
the TIC ID (int) and observed sector (int).
"""

# 3rd party
from pathlib import Path
import re
import pandas as pd
import numpy as np
import multiprocessing
from astropy.io import fits


def _convert_sectors_observed_format(x):
    """ Convert 'sectors_observed' column from a string with sector IDs separated by '_' to a binary 
    string of 0/1 characters of length 150.
    
        Args:
            x: pandas Series, row of a DataFrame with 'sectors_observed' column
        
        Returns: x, pandas Series with 'sectors_observed' column converted to binary string format
    """

    MAX_NUM_SECTORS = 150

    if '_' in x['sectors_observed'] or len(x['sectors_observed']) != MAX_NUM_SECTORS:
        sectors_lst = [int(sector_str) for sector_str in x['sectors_observed'].split('_')]
        sectors_observed = ''.join(['0' if sector_i not in sectors_lst else '1' for sector_i in
                                    range(MAX_NUM_SECTORS)])
        x['sectors_observed'] = sectors_observed

    return x
    
def create_target_sector_table_from_tce_table(tce_table):
    """ Uses TCE table with columns 'target_id' and 'sectors_observed' to create a table where each row describes a
    sector for which the target was observed. `sectors_observed` column should be a binary string of 0/1 characters of length 150 
    where the position indicates whether the target was observed in the corresponding sector, or a string with the IDs of the sectors separated by '_'.

        Args:
            tce_table: pandas DataFrame, TCE table

        Returns: targets_df, pandas DataFrame with columns 'target_id' and 'sector'
    """

    # get sectors observed for each target in the table
    targets_dict = {'target_id': [], 'sector': []}
    
    tce_table = tce_table.apply(_convert_sectors_observed_format, axis=1)

    targets = tce_table.groupby('target_id')

    # get observed sectors for each target based on its TCEs
    for target_id, target_data in targets:
        obs_sectors_target = np.nonzero(np.sum(
            target_data['sectors_observed'].apply(lambda x: np.array([int(el) for el in x])).values, axis=0))[0]
        for obs_sector in obs_sectors_target:
            targets_dict['target_id'].append(target_id)
            targets_dict['sector'].append(obs_sector)

    targets_df = pd.DataFrame(targets_dict)
    targets_df = targets_df.astype({'target_id': 'int64', 'sector': 'int64'})

    return targets_df


def create_targets_lc_sh_files_sector_using_target_table(targets_in_sector, obs_sector, src_lc_sh_dir_fp,
                                                         dest_lc_sh_dir_fp, data_collection_mode_flag, lc_dir=None):
    """ Uses a table with columns 'target_id' and 'sector', `targets_sector` to determine which target lc curl
    statements should be copied from `src_lc_sh_fp` to a destination targets lc sh file `dest_lc_sh_fp` for the given
    sector.

        Args:
            targets_in_sector: pandas DataFrame, targets observed in 'sector'
            obs_sector: int, sector
            src_lc_sh_dir_fp: Path, directory with sector sh files with all targets lc curl statements
            dest_lc_sh_dir_fp: Path, directory used to save sh files iwth lc curl statements for targets in
                `targets_sector` table
            data_collection_mode_flag: bool, either '2min' or 'ffi' for TESS SPOC 2-min/FFI lc FITS files, respectively
            lc_dir: if `lc_dir` is a Path object, then it will check whether the lc FITS files were already downloaded
                for the targets in `targets_in_sector` and if they are not corrupted, their curl statements are excluded
                from being added to new target sector lc sh files. It should be the path to the root directory in which
                the lightcurve FITS files were downloaded to.

        Returns: targets_added_sector, pandas DataFrame with columns 'target_id' and 'sector' describing targets whose
            curl statement was added to the destination targets lc sh file
    """

    print(f'Iterating over sector {obs_sector} with {len(targets_in_sector)} targets...')

    if lc_dir:
        print(f'Checking for existence of target lc FITS files in {lc_dir} for targets in table...\nExclude targets'
              f' whose lc FITS file was already downloaded and is not corrupted.')
        exclude_targets = check_existence_lc_files_for_targets_in_table(targets_in_sector,
                                                                          lc_dir,
                                                                          data_collection_mode_flag)

        print(f'Found target light curve FITS files for {len(exclude_targets)} targets in table. Excluded them.')
        targets_in_sector = targets_in_sector.merge(exclude_targets, on=['target_id', 'sector'], how='left', indicator=True)
        targets_in_sector = targets_in_sector.loc[targets_in_sector['_merge'] == 'left_only'].drop(columns='_merge')

        print(f'Finished checking for existence of target lc FITS files in {lc_dir} for targets in table.')

    # set filepath to source lc targets sh file for the sector run
    if data_collection_mode_flag == '2min':
        src_targets_lc_sh_sector_fp = (src_lc_sh_dir_fp /
                                       f'tesscurl_sector_{obs_sector}_lc.sh')
    elif data_collection_mode_flag == 'ffi':
        src_targets_lc_sh_sector_fp = (src_lc_sh_dir_fp /
                                       f'hlsp_tess-spoc_tess_phot_s{str(obs_sector).zfill(4)}_tess_v1_dl-lc.sh')
    else:
        raise ValueError(f'Data collection mode {data_collection_mode_flag} not recognized. Currently supporting '
                         f'`2min` or `ffi` TESS SPOC lc FITS files.')
    if not src_targets_lc_sh_sector_fp.exists():
        print(f'{src_targets_lc_sh_sector_fp} not found. Skipping this sector.')

        targets_added_sector = {field: [] for field in ['target_id', 'sector']}
        targets_added_sector = pd.DataFrame(targets_added_sector)

        return targets_added_sector

    # set destination filepath for sh file with lc targets
    dest_targets_lc_sh_sector_fp = dest_lc_sh_dir_fp / src_targets_lc_sh_sector_fp.name

    print(f'Getting curl statements for {len(targets_in_sector)} targets in sector {obs_sector}...')

    targets_added_sector = {field: [] for field in ['target_id', 'sector']}

    with open(dest_targets_lc_sh_sector_fp, 'w') as dest_file:

        with open(src_targets_lc_sh_sector_fp) as src_file:

            for curl_target_lc_line in src_file:

                # get target id in curl statement
                target_in_src_file_pattern_search_res = re.search('\d{16}', curl_target_lc_line)
                if target_in_src_file_pattern_search_res:
                    target_in_src_file = int(target_in_src_file_pattern_search_res[0])

                    # add curl statement to destination sh file if target is found in the sector table
                    if (targets_in_sector['target_id'] == target_in_src_file).any():

                        dest_file.write(curl_target_lc_line)

                        targets_added_sector['target_id'].append(target_in_src_file)
                        targets_added_sector['sector'].append(obs_sector)

    targets_added_sector = pd.DataFrame(targets_added_sector)
    if lc_dir:  # consider targets with lc files already downloaded and not corrupt as added
        targets_added_sector = pd.concat([targets_added_sector, exclude_targets], axis=0, ignore_index=True)

    print(f'Finished getting curl statements for {len(targets_in_sector)} targets in sector {obs_sector}.')

    return targets_added_sector


def check_existence_lc_files_for_targets_in_table(targets_tbl, lc_dir, data_collection_mode_flag):
    """ Check whether there already exist target lc files in `lc_dir` that are not corrupted. If `data_collection_mode_flag` is `2min`,
    then it assumes that the lc files are in `lc_dir/sector_{sector}/` (e.g., "sector_1") and if it is `ffi`, then it assumes that the lc files are in
    directories `lc_dir/s{sector with 4 digits zero-padded to the left}/target/{first 4 digits of ticid}/{next 4 digits of ticid}/{next 4 digits of ticid}/{last 4 digits of ticid}/`.

        Args:
            targets_tbl: pandas DataFrame, targets observed in 'sector'
            lc_dir: int, sector
            data_collection_mode_flag: bool, either '2min' or 'ffi' for TESS SPOC 2-min/FFI lc FITS files, respectively

        Returns: targets_tbl, pandas DataFrame without targets excluded since lc FITS files were found
    """

    exclude_targets = {field: [] for field in ['target_id', 'sector']}
    n_lc_files_corrupted = 0
    for _, target_data in targets_tbl.iterrows():

        ticid_str = str(target_data['target_id']).zfill(16)

        if data_collection_mode_flag == '2min':
            sector_dir = lc_dir / f'sector_{target_data["sector"]}'
            target_lc_fp_search = list(sector_dir.glob(f'*{ticid_str}*lc.fits'))

        elif data_collection_mode_flag == 'ffi':
            sector_dir = lc_dir / f's{str(target_data["sector"]).zfill(4)}'
            target_dir = sector_dir / 'target' / ticid_str[:4] / ticid_str[4:8] / ticid_str[8:12] / ticid_str[12:]
            target_lc_fp_search = list(target_dir.glob(f'*{target_data["target_id"]}*lc.fits'))

        else:
            raise ValueError(f'Data collection mode {data_collection_mode_flag} not recognized. Currently supporting '
                             f'`2min` or `ffi` TESS SPOC lc FITS files.')

        if len(target_lc_fp_search) == 1:

            # check FITS file integrity
            try:
                with fits.open(target_lc_fp_search[0]) as _:
                    # add target to exclusion list if the file was not corrupted
                    exclude_targets['target_id'].append(target_data['target_id'])
                    exclude_targets['sector'].append(target_data['sector'])
            except Exception as e:  # remove file if it is corrupted
                print(f'Error when reading target lc FITS file for target {target_data["target_id"]} '
                      f'in sector {target_data["sector"]} in {target_lc_fp_search[0]}\n{e}')
                # target_lc_fp_search[0].unlink()
                n_lc_files_corrupted += 1

        elif len(target_lc_fp_search) > 1:
            raise ValueError(f'Found more than two light curve files for target {target_data["target_id"]} in '
                             f'sector {target_data["sector"]}')
        elif len(target_lc_fp_search) == 0:
            continue

    sectors_in_tbl = targets_tbl['sector'].unique()
    print(f'{n_lc_files_corrupted} target lc FITS files were corrupted and deleted in sector(s): '
          f'{sectors_in_tbl}.')

    exclude_targets = pd.DataFrame(exclude_targets)

    return exclude_targets


if __name__ == '__main__':

    # Set up paths

    # directory with lc sh files
    src_lc_sh_dir = Path('/data3/exoplnt_dl/lc_fits/tesscurl_sectors_lcs/all_targets')
    # destination directory to save new lc sh files after removing curl statements for targets without DV results
    dest_lc_sh_dir = Path(f'/data3/exoplnt_dl/lc_fits/tesscurl_sectors_lcs/download_missing_targets_sectors_lcs_s1-s92_9-16-2025_1257')
    # ffi or 2-min idiosyncrasies
    data_collection_mode = '2min'  # `2min` or `ffi`
    # root directory in which lightcurve files are downloaded to; checks lc files already downloaded and not corrupted
    # - those are excluded; set to None for no verification
    lc_dir_fp = Path('/data3/exoplnt_dl/lc_fits/2-min')
    # parallelize using multiprocessing
    n_processes = 14  # set to None for sequential
    # target table with 'sector' and 'target_id' columns
    tce_tbl = pd.read_csv('/data3/exoplnt_dl/ephemeris_tables/tess/tess_spoc_2min/tess-spoc-2min-tces-dv_s1-s92_9-16-2025/tess-spoc-2min-tces-dv_s1-s92_9-16-2025_uid.csv',
                          usecols=['target_id', 'sectors_observed'])
    target_tbl = create_target_sector_table_from_tce_table(tce_tbl)

    dest_lc_sh_dir.mkdir(exist_ok=True)

    target_tbl.to_csv(dest_lc_sh_dir / 'targets_sectors_for_download_lcs.csv', index=False)

    # Get curl statements for targets in each sector run
    n_sectors = len(target_tbl['sector'].unique())
    targets_sectors = target_tbl.groupby('sector')
    targets_sectors_jobs = [(targets_sector, sector, src_lc_sh_dir, dest_lc_sh_dir, data_collection_mode,
                             lc_dir_fp)
                            for sector, targets_sector in targets_sectors]
    print(f'Iterating over {n_sectors} sectors...')

    # parallelized
    if n_processes:
        print(f'Using {n_processes} processes...')
        pool = multiprocessing.Pool(processes=n_processes)
        async_results = [pool.apply_async(create_targets_lc_sh_files_sector_using_target_table, job)
                         for job in targets_sectors_jobs]
        pool.close()
        pool.join()
        targets_added_lst = [async_result.get() for async_result in async_results]
    else:
        # sequential
        targets_added_lst = []
        # iterate over sectors in targets table
        for sector_i, targets_sector_job in enumerate(targets_sectors_jobs):
            print(f'Iterating over sector {targets_sector_job[1]} ({sector_i + 1}/{n_sectors}) '
                  f'with {len(targets_sector_job[0])} targets...')
            targets_added_sector_tbl = create_targets_lc_sh_files_sector_using_target_table(
                targets_sector_job[0],
                targets_sector_job[1],
                src_lc_sh_dir,
                dest_lc_sh_dir,
                data_collection_mode,
            )
            targets_added_lst.append(targets_added_sector_tbl)

    # aggregate outputs from each sector
    # information on targets missed
    targets_added = pd.concat(targets_added_lst, axis=0, ignore_index=True)
    targets_missed = target_tbl.merge(targets_added, on=['target_id', 'sector'], how='outer', indicator=True)
    targets_missed = targets_missed.loc[targets_missed['_merge'] == 'left_only']
    targets_missed = targets_missed.drop(columns=['_merge'])
    print(f'Number of missed targets: {len(targets_missed)}')
    targets_missed.to_csv(dest_lc_sh_dir / 'missed_targets_sectors.csv', index=False)
