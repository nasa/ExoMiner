""" Preprocess features in the Kepler TCE table. """

# 3rd party
import pandas as pd
from pathlib import Path
import numpy as np

#%% Read TCE table

tce_tbl_fp = Path('')
tce_tbl = pd.read_csv(tce_tbl_fp)

#%%

tce_tbl['mission'] = 1

# create categorical magnitude
kepler_mag_thr = 12
tce_tbl['mag_cat'] = 0.0
tce_tbl.loc[tce_tbl['mag'] > kepler_mag_thr, 'mag_cat'] = 1.0
tce_tbl.loc[tce_tbl['mag'].isna(), 'mag_cat'] = np.nan  # set to nan if magnitude is nan

# set shifted magnitude
tce_tbl['mag_shift'] = tce_tbl['mag'] - kepler_mag_thr

# create normalized count for rolling band level 0
# columns_rba = ['tce_rb_tcount1', 'tce_rb_tcount2', 'tce_rb_tcount3', 'tce_rb_tcount4']
# tce_tbl['tce_rb_tcount0n'] = tce_tbl['tce_rb_tcount0'] / tce_tbl[['tce_rb_tcount0'] + columns_rba].sum(axis=1,
#                                                                                                        skipna=True)
tce_tbl['tce_rb_tcount0n'] = tce_tbl['tce_rb_tcount0']
tce_tbl.loc[tce_tbl['tce_rb_tcount0n'] == -1, 'tce_rb_tcount0n'] = np.nan

# create transit source offset ratios
tce_tbl['tce_dikco_msky_rat'] = (
    tce_tbl.apply(lambda x: np.nan if x['tce_dikco_msky_err'] == -1 else x['tce_dikco_msky'] /
                                                                         x['tce_dikco_msky'],
                  axis=1))
tce_tbl['tce_dicco_msky_rat'] = (
    tce_tbl.apply(lambda x: np.nan if x['tce_dicco_msky_err'] == -1 else x['tce_dicco_msky'] /
                                                                         x['tce_dicco_msky'],
                  axis=1))

tce_tbl.to_csv(tce_tbl_fp.parent / f'{tce_tbl_fp.stem}_preprocessed.csv', index=False)
