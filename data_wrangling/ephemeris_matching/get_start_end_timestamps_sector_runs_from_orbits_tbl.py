"""
Get start and end timestamps for each TIC observed in each sector run based on the orbit times table for TESS.
"""

# 3rd party
import pandas as pd
from astropy.time import Time
from pathlib import Path

#%% Get start and end timestamps of sector runs using orbit times table


def convert_utc_time_to_btjd(date_utc):
    """ Convert datetime from UTC to Barycentric TESS Julian Date (BTJD)

    Args:
        date_utc: str, datetime in UTC scale.

    Returns: float, datetime in BTJD

    """

    # date_utc = datetime.strptime(date_utc, '%m/%d/%Y %I:%M:%S %p')

    t_utc = Time(date_utc, scale='utc')
    t_bjd = t_utc.tdb.jd
    TESS_BJD0 = 2457000
    t_btjd = t_bjd - TESS_BJD0

    return t_btjd


root_dir = Path()
sector_times_tbl = pd.read_csv(root_dir / 'orbit_times_20220906_1434.csv')

sector_timestamps = {'sector': [], 'start': [], 'end': []}
for sector_run in sector_times_tbl['Sector'].unique():
    sector_timestamps['sector'].append(sector_run)
    sector_times = sector_times_tbl.loc[sector_times_tbl['Sector'] == sector_run]
    sector_timestamps['start'].append(sector_times['Start of Orbit'].values[0])
    sector_timestamps['end'].append(sector_times['End of Orbit'].values[-1])

sector_timestamps = pd.DataFrame(sector_timestamps)
sector_timestamps['start'] = sector_timestamps['start'].apply(convert_utc_time_to_btjd)
sector_timestamps['end'] = sector_timestamps['end'].apply(convert_utc_time_to_btjd)

sector_timestamps.to_csv(root_dir / f'sector_times_btjd.csv', index=False)
