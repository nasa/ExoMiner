"""
Preprocesses tables to make them ready for ephemeris matching.
"""

# 3rd party
import pandas as pd
from pathlib import Path
import numpy as np

#%% ExoFOP TOI catalog

tbl_fp = Path('/')
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

tbl_fp = Path('', dtype={'Tc_BTJD': np.float64})
tbl = pd.read_csv(tbl_fp)

# rename columns
tbl.rename(
    columns={'Tc_BTJD': 'epoch', 'P': 'period', 'Duration (hrs)': 'duration', 'TOI': 'uid', 'TIC': 'target_id'},
    inplace=True, errors='raise')

# set uid as string
tbl['uid'] = tbl['uid'].astype('str')

tbl['epoch'] = tbl['epoch'].astype('float')

# exclude TOIs with zero period (single-transit TOIs)
tbl = tbl.loc[tbl['period'] > 0]

tbl.to_csv(tbl_fp.parent / f'{tbl_fp.stem}_processed_ephem_matching.csv', index=False)
