"""
Get lightcurve FITS files for targets with TCE results. The TCE table should contain information about which sectors
a target was observed for

For each TIC lc FITS file, check whether there are TCEs in that TIC in the TCE table. If there isn't, then the lc FITS
file is skipped and the data for the TIC in the corresponding sector is not downloaded.
"""

# 3rd party
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import lightkurve as lk
import multiprocessing


def download_lc_file_target_in_sector(target_id, sector, save_dir):
    """ Download lightcurve FITS file for TIC `target_id` in sector `sector` to directory `save_dir`.

    Args:
        target_id: int, TIC ID
        sector: int, sector
        save_dir: str, save directory
    """

    search_lc_res = lk.search_lightcurve(target=f"tic{target_id}", mission='TESS',
                                         author=('TESS-SPOC', 'SPOC'),
                                         exptime=120, cadence='long', sector=sector)

    if len(search_lc_res) == 0:
        print(f'Found no lightcurve results for TIC {target_id} in sector {sector}. Skipping')
    else:
        _ = search_lc_res[0].download(download_dir=save_dir)
        lc_fp = list((Path(save_dir) / 'mastDownload' / 'TESS').rglob("*lc.fits"))[0]
        # move file to save_dir because lightkurve decided to create subdirectories...
        shutil.move(lc_fp, Path(save_dir) / lc_fp.name)
        shutil.rmtree(Path(save_dir) / 'mastDownload')


def download_lc_file_for_set_targets(targets_df, lc_root_dir):
    """ Download lightcurve FITS file for TICs in `targets_df` (with columns 'target_id' and 'sector'). Lightcurve FITS
    files are saved to `lc_root_dir/sector_{sector}`.

    Args:
        targets_df: pandas DataFrame, targets table
        lc_root_dir: Path, root directory for saving lightcurve FITS files
    """

    targets_not_downloaded = {field: [] for field in ['target_id', 'sector', 'error']}
    for target_i, target_data in targets_df.iterrows():

        if target_i % 100 == 0:
            print(f'Iterated through {target_i + 1} target-sector pairs out of {len(targets_df)} total...')

        sector_dir = lc_root_dir / f'sector_{target_data["sector"]}'
        lc_fps = list(sector_dir.glob(f'*{target_data["target_id"]}*lc.fits'))

        if len(lc_fps) == 0:  # did not find target lc fits file for the sector
            print(f'Downloading lightcurve data for TIC {target_data["target_id"]} in sector {target_data["sector"]}')
            try:
                download_lc_file_target_in_sector(target_data['target_id'], target_data['sector'], str(sector_dir))
            except Exception as e:
                print(f'Found error {e} when downloading lightcurve data for target {target_data["target_id"]} in '
                      f'sector {target_data["sector"]}')

                targets_not_downloaded['target_id'].append(target_data['target_id'])
                targets_not_downloaded['sector'].append(target_data['sector'])
                targets_not_downloaded['error'].append(e)

    targets_not_downloaded = pd.DataFrame(targets_not_downloaded)
    print(f'Number of targets whose light curve was not downloaded: {len(targets_not_downloaded)}\nSector statistics: '
          f'{targets_not_downloaded["sector"].value_counts()}')

    return targets_not_downloaded


if __name__ == "__main__":

    tce_tbl = pd.read_csv('/nobackupp19/msaragoc/work_dir/Kepler-TESS_exoplanet/data/Ephemeris_tables/TESS/tess_2min_tces_dv_s1-s88_3-27-2025_1316.csv')
    lc_root_dir = Path('/nobackup/msaragoc/work_dir/Kepler-TESS_exoplanet/data/FITS_files/TESS/spoc_2min/lc/')

    # filter for subset of TCEs from new runs
    tce_tbl = tce_tbl.loc[tce_tbl['sector_run'].isin(['1-69', '14-78', '2-72'])]

    def _convert_sectors_observed_format(x):

        MAX_NUM_SECTORS = 150

        if '_' in x['sectors_observed'] or len(x['sectors_observed']) != MAX_NUM_SECTORS:
            sectors_lst = [int(sector_str) for sector_str in x['sectors_observed'].split('_')]
            sectors_observed = ''.join(['0' if sector_i not in sectors_lst else '1' for sector_i in
                                        range(MAX_NUM_SECTORS)])
            x['sectors_observed'] = sectors_observed

        return x

    tce_tbl = tce_tbl.apply(_convert_sectors_observed_format, axis=1)

    # get sectors observed for each target in the table
    targets_dict = {'target_id': [], 'sector': []}  # , 'mag': []}
    targets = tce_tbl.groupby('target_id')
    # get observed sectors for each target based on its TCEs
    for target_id, target_data in targets:
        target_data = target_data.reset_index(drop=True)
        obs_sectors_target = np.nonzero(np.sum(
            target_data['sectors_observed'].apply(lambda x: np.array([int(el) for el in x])).values, axis=0))[0]
        for obs_sector in obs_sectors_target:
            targets_dict['target_id'].append(target_id)
            targets_dict['sector'].append(obs_sector)
            # targets_dict['mag'].append(target_data.loc[0, 'mag'])

    targets_df = pd.DataFrame(targets_dict)

    jobs = [(targets_df, lc_root_dir)]
    n_jobs = len(jobs)
    print(f'Split work into {n_jobs} job(s).')

    # parallelize jobs
    n_procs = 10
    n_procs = min(n_jobs, n_procs)
    pool = multiprocessing.Pool(processes=n_procs)
    async_results = [pool.apply_async(download_lc_file_for_set_targets, job) for job in jobs]
    pool.close()
    pool.join()

    # for job in jobs:
    #     _ = download_lc_file_for_set_targets(*job)

    print(f'Finished.')
