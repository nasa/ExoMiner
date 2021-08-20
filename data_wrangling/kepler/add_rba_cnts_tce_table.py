"""
Add RBA counts for other levels above level 0 and normalize level 0 counts by the total number of transits reported
across the different levels.
"""

from pathlib import Path

import numpy as np
import pandas as pd

rba_cnts_tce_tbl = pd.read_csv(
    '/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17_DR25/q1_q17_dr25_tce_2020.09.28_10.36.22_stellar_koi_cfp_norobovetterlabels_renamedcols_nomissingval_symsecphase_confirmedkoiperiod_sec_rba_cnt0n.csv')
tce_tbl_fp = Path('/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17_DR25/'
                  '19-08-21_07:21/q1_q17_dr25_tce_2020.09.28_10.36.22_stellar_koi_cfp_norobovetterlabels_renamedcols_nomissingval_symsecphase_cpkoiperiod.csv')
tce_tbl = pd.read_csv(tce_tbl_fp)

columns_rba = ['tce_rb_tcount1', 'tce_rb_tcount2', 'tce_rb_tcount3', 'tce_rb_tcount4']
for col in columns_rba:
    tce_tbl[col] = np.nan

for tce_i, tce in tce_tbl.iterrows():
    tce_tbl.loc[tce_i, columns_rba] = rba_cnts_tce_tbl.loc[(rba_cnts_tce_tbl['target_idtc'] == tce['target_id']) &
                                                           (rba_cnts_tce_tbl['tce_plnt_num'] == tce['tce_plnt_num']),
                                                           columns_rba].values[0]

# normalize level zero counts by total
tce_tbl['tce_rb_tcount0n'] = tce_tbl['tce_rb_tcount0'] / tce_tbl[['tce_rb_tcount0'] + columns_rba].sum(axis=1,
                                                                                                       skipna=True)

tce_tbl.to_csv(tce_tbl_fp.parent / f'{tce_tbl_fp.stem}_rba_cnt0n.csv', index=False)
