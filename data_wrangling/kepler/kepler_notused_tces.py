"""
Create TCE and KIC dataset with Kepler Q1-Q17 DR25 TCEs not used to train/validate ExoMiner.
"""

import pandas as pd
from pathlib import Path

# %%

res_dir = Path('/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17_DR25/')
tce_tbl_fp = res_dir / 'q1_q17_dr25_tce_2020.09.28_10.36.22_stellar_koi_cfp_norobovetterlabels_renamedcols_nomissingval_symsecphase_confirmedkoiperiod_sec_rba_cnt0n_koiperiodonlydiff_nanstellar.csv'
tce_tbl = pd.read_csv(tce_tbl_fp)

# remove rogue TCEs
tce_tbl = tce_tbl.loc[tce_tbl['tce_rogue_flag'] == 0]
# remove Confirmed KOIs
tce_tbl = tce_tbl.loc[tce_tbl['koi_disposition'] != 'CONFIRMED']
# remove CFP and CFA
tce_tbl = tce_tbl.loc[~tce_tbl['fpwg_disp_status'].isin(['CERTIFIED FP', 'CERTIFIED FA'])]
# remove non-KOI
tce_tbl = tce_tbl.loc[~tce_tbl['kepoi_name'].isna()]

print(f'Number of not used TCEs: {len(tce_tbl)}')

tce_tbl.to_csv(res_dir / f'{tce_tbl_fp.name}_notusedtces.csv', index=False)

# get KICs
kic_cols = ['target_id', 'tce_steff', 'tce_smet', 'tce_slogg', 'tce_smass', 'tce_sradius', 'tce_sdens']
kic_tbl = tce_tbl.drop_duplicates(subset='target_id', ignore_index=True)[kic_cols]
print(f'Number of KICs for the not used TCEs: {len(kic_tbl)}')
kic_tbl.to_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/KIC_tables/'
               'q1_q17_dr25_stellar_plus_supp_gaiadr2_notusedtces.csv', index=False)
