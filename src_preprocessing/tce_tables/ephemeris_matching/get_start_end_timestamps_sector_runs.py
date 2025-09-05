"""
Get start and end timestamps for each TIC observed in each sector run based on information contained in the header of
the lc FITS file.
"""

# 3rd party
from pathlib import Path
import pandas as pd
from astropy.io import fits


def get_start_end_timestamps_tics_sector_run(sector_dir):
    """ Get start and end timestamps for the TICs in a given sector run.

    Args:
        sector_dir: Path, directory with the LC FITS files for the TICs observed in the given sector run

    Returns: pandas DataFrame, contains start and end timestamps in BTJD for each TIC observed in the given sector run
    """

    # get lc filepaths for all targets in the sector run
    target_fits_fps = list(sector_dir.rglob('*lc.fits'))
    print(f'Found {len(target_fits_fps)} target lc files for sector run {sector_dir.name}.')

    target_timestamps = {'target': [], 'start': [], 'end': [], 'sector': []}
    for target_lc_fp_i, target_lc_fp in enumerate(target_fits_fps):
        if target_lc_fp_i % 500 == 0:
            print(f'Iterating over {target_lc_fp_i + 1} ouf of '
                  f'{len(target_fits_fps)} target lc files in sector run {sector_dir.name}...')
        # if target_lc_fp_i == 10:
        #     break
        try:
            target_timestamps['target'].append(fits.getheader(target_lc_fp, ignore_missing_end=True)['TICID'])
            target_timestamps['start'].append(fits.getheader(target_lc_fp, ignore_missing_end=True)['TSTART'])
            target_timestamps['end'].append(fits.getheader(target_lc_fp, ignore_missing_end=True)['TSTOP'])
            target_timestamps['sector'].append(str(fits.getheader(target_lc_fp, ignore_missing_end=True)['Sector']))
        except Exception as e:
            print(f'{target_lc_fp}|Error getting start and end timestamps from header of lc FITS file: {e}')

    target_timestamps = pd.DataFrame(target_timestamps)

    print(f'Iterated over all target lc files in sector run {sector_dir.name}.')

    return target_timestamps


def get_start_end_timestamps_tics_sector_runs(sector_dirs, save_dir):
    """ Wrapper used to get start and end timestamps for the TICs in a given set of sector runs.

    Args:
        sector_dirs: list of Paths, directories for the sector runs
        save_dir: Path, directory used to save the tables created for each sector run with the start and end timestamps
        for the TICs observed

    Returns:

    """

    for sector_dir in sector_dirs:
        print(f'Iterating over sector {sector_dir.name}...', flush=True)
        target_sector_run_timestamps = get_start_end_timestamps_tics_sector_run(sector_dir)
        target_sector_run_timestamps.to_csv(save_dir / f'{sector_dir.name}_times_btjd_start_end.csv', index=False)


if __name__ == '__main__':

    # directory used to save start/end timestamps target tables for each sector run
    res_dir = Path('/home/msaragoc/Projects/exoplnt_dl/experiments/ephemeris_matching/tess_spoc_ffi_start_end_timestamps_tics_lc_s36-s69_7-9-2024_1044')
    # lightcurve root directory for the target data of interest from where to get the timestamps
    lc_root_dir = Path('/data5/tess_project/Data/tess_spoc_ffi_data/lc/fits_files/')

    # 2min data
    # sector_dirs = [fp for fp in lc_root_dir.iterdir() if fp.name.startswith('sector_')]
    # ffi data
    sector_dirs_fps = [fp for fp in lc_root_dir.iterdir() if fp.name.startswith('s0061')]
    # sector_dirs_fps = [sector_dirs_fps[0]]

    print(f'Extracting start/end timestamps for targets in {len(sector_dirs_fps)} sector runs.')

    res_dir.mkdir(exist_ok=True)

    get_start_end_timestamps_tics_sector_runs(list(sector_dirs_fps), res_dir)

    # aggregate tables into a single table
    target_sector_run_timestamps_all = \
        pd.concat([pd.read_csv(fp)
                   for fp in res_dir.iterdir()], axis=0).to_csv(res_dir.parent / f'{res_dir.name}.csv', index=False)

    print('Finished.')
