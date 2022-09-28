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

    target_timestamps = {'target': [], 'start': [], 'end': []}
    for target_lc_fp in sector_dir.iterdir():
        target_timestamps['target'].append(fits.getheader(target_lc_fp)['TICID'])
        target_timestamps['start'].append(fits.getheader(target_lc_fp)['TSTART'])
        target_timestamps['end'].append(fits.getheader(target_lc_fp)['TSTOP'])

    target_timestamps = pd.DataFrame(target_timestamps)
    target_timestamps['sector'] = int(sector_dir.name.split('_')[-1])

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

    root_dir = Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/experiments/ephemeris_matching_dv/')
    # lc_root_dir = Path('/data5/tess_project/Data/TESS_lc_fits')
    lc_root_dir = Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/data/fits_files/tess/lc/')
    sector_dirs = [fp for fp in lc_root_dir.iterdir() if fp.name.startswith('sector_')]
    save_dir = root_dir / 'start_end_timestamps_tics_lc'
    save_dir.mkdir(exist_ok=True)

    get_start_end_timestamps_tics_sector_runs(list(sector_dirs), save_dir)

    target_sector_run_timestamps_all = \
        pd.concat([pd.read_csv(fp)
                   for fp in save_dir.iterdir()], axis=0).to_csv(root_dir /
                                                                 'all_sectors_times_btjd_start_end.csv', index=False)
