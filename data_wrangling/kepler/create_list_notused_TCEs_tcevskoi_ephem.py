"""
Get Kepler TCEs from the set not used for training and validation that have different TCE and KOI orbital period.
"""

from pathlib import Path
import pandas as pd
import numpy as np

res_dir = Path('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Analysis/tce_vs_koi_per_notusedtces')
tce_tbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17_DR25/q1_q17_dr25_tce_2020.09.28_10.36.22_stellar_koi_cfp_norobovetterlabels_renamedcols_nomissingval_symsecphase_confirmedkoiperiod_sec_rba_cnt0n_koiperiodonlydiff.csv')

# remove rogue TCEs
tce_tbl = tce_tbl.loc[tce_tbl['tce_rogue_flag'] == 0]
# remove Confirmed KOIs
tce_tbl = tce_tbl.loc[tce_tbl['koi_disposition'] != 'CONFIRMED']
# remove CFP and CFA
tce_tbl = tce_tbl.loc[~tce_tbl['fpwg_disp_status'].isin(['CERTIFIED FP', 'CERTIFIED FA'])]
# remove non-KOI
tce_tbl = tce_tbl.loc[~tce_tbl['kepoi_name'].isna()]

print(f'Number of not used TCEs: {len(tce_tbl)}')

tce_tbl['tce_vs_koi_per'] = 0

for tce_i, tce in tce_tbl.iterrows():
        tce_tbl.loc[tce_i, 'tce_vs_koi_per'] = np.abs(tce['tce_period'] - tce['koi_period']) / max(tce['tce_period'], tce['koi_period'])

tce_tbl.to_csv(res_dir / 'q1_q17_dr25_tce_2020.09.28_10.36.22_stellar_koi_cfp_norobovetterlabels_renamedcols_nomissingval_symsecphase_confirmedkoiperiod_sec_rba_cnt0n_koiperiodonlydiff_notusedtces.csv', index=False)
