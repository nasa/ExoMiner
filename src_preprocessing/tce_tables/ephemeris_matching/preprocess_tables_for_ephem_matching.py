"""
Preprocesses tables to make them ready for ephemeris matching.
"""

# 3rd party
import pandas as pd
from pathlib import Path
import numpy as np
import re

#%% ExoFOP TOI catalog

tbl_fp = Path('/home6/msaragoc/work_dir/Kepler-TESS_exoplanet/data/Ephemeris_tables/TESS/exofop_tois/tois_3-2-2026.csv')
tbl = pd.read_csv(tbl_fp)

# rename columns
tbl.rename(
    columns={'Epoch (BJD)': 'epoch', 'Period (days)': 'period', 'Duration (hours)': 'duration', 'TOI': 'uid',
             'TIC ID': 'target_id'},
    inplace=True, errors='raise')

# set uid as string
tbl['uid'] = tbl['uid'].astype('str')

# set epoch as TBJD
tbl['epoch'] = tbl['epoch'] - 2457000

# exclude TOIs with zero period (single-transit TOIs)
tbl = tbl.loc[tbl['period'] > 0]

tbl.to_csv(tbl_fp.parent / f'{tbl_fp.stem}_processed_ephem_matching.csv', index=False)

#%% SG1 TOI catalog

tbl_fp = Path('/home6/msaragoc/work_dir/Kepler-TESS_exoplanet/data/Ephemeris_tables/TESS/sg1/sg1_tois_2-25-2026.csv', dtype={'Tc_BTJD': np.float64})
tbl = pd.read_csv(tbl_fp)

# rename columns
tbl.rename(
    columns={'Tc_BTJD': 'epoch', 'P': 'period', 'Duration (hrs)': 'duration', 'TOI': 'uid', 'TIC': 'target_id'},
    inplace=True, errors='raise')

# set uid as string
tbl['uid'] = tbl['uid'].astype('str')

# parse epochs of TOIs that are a list
def parse_epoch_cell(s):
    if pd.isna(s):
        return np.nan
    # Extract floats (handles decimals, optional sign, scientific notation if you tweak the regex)
    nums = re.findall(r'[-+]?\d*\.\d+|\d+', str(s).replace('\n', ' ').replace('\r', ' '))
    return [float(x) for x in nums]

tbl['epoch_list'] = tbl['epoch'].apply(parse_epoch_cell)

# choose one epoch
tbl['epoch'] = tbl['epoch_list'].apply(lambda x: x[0])

tbl['epoch'] = tbl['epoch'].astype('float')

# exclude TOIs with zero period (single-transit TOIs)
tbl = tbl.loc[tbl['period'] > 0]

tbl.to_csv(tbl_fp.parent / f'{tbl_fp.stem}_processed_ephem_matching.csv', index=False)

# %%
