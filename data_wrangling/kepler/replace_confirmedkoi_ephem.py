"""
Script used to replace orbital period and epoch for Confirmed KOIs that have significantly different period from the
TCEs.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd


def update_albedo_and_ptemp(x):
    return pd.Series(data={'tce_albedo': 1, 'tce_ptemp': x['tce_eqt']})


# %% Only replace TCE period by KOI period for Confirmed KOIs whose period significantly changed
# (e.g., TCE period is a multiple of the KOI period)

tbl_dir = Path('/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17_DR25/')

tce_tbl = pd.read_csv(
    tbl_dir / 'q1_q17_dr25_tce_2020.09.28_10.36.22_stellar_koi_cfp_norobovetterlabels_renamedcols_nomissingval_symsecphase_confirmedkoiperiod_sec_rba_cnt0n.csv')

tce_tbl_tcephem = pd.read_csv(
    tbl_dir / '/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17_DR25/q1_q17_dr25_tce_2020.09.28_10.36.22_stellar_koi_cfp_norobovetterlabels_renamedcols_nomissingval_symsecphase.csv')

thr = 0.3

cnt_tces = {'reverted': 0, 'koi_replaced': 0}
for tce_i, tce in tce_tbl.iterrows():
    if tce['koi_disposition'] == 'CONFIRMED':  # look only at Confirmed KOIs
        tce_found = tce_tbl_tcephem.loc[(tce_tbl_tcephem['target_id'] == tce['target_id']) &
                                        (tce_tbl_tcephem['tce_plnt_num'] == tce['tce_plnt_num'])]

        if np.abs(tce['tce_period'] - tce_found['tce_period'].values[0]) / \
                max(tce['tce_period'], tce_found['tce_period'].values[0]) > thr:
            tce_tbl.loc[tce_i, ['tce_period', 'tce_time0bk']] = tce_found[['koi_period', 'koi_time0bk']].values[0]
            cnt_tces['koi_replaced'] += 1
        else:
            tce_tbl.loc[tce_i, ['tce_period']] = tce_found[['tce_period']].values[0]
            cnt_tces['reverted'] += 1

print(f'TCE count: {cnt_tces}')
print(f'Total number of TCEs affected: {np.sum(list(cnt_tces.values()))} '
      f'({len(tce_tbl.loc[tce_tbl["koi_disposition"] == "CONFIRMED"])} Confirmed KOIs)')

tce_tbl.to_csv(
    tbl_dir / 'q1_q17_dr25_tce_2020.09.28_10.36.22_stellar_koi_cfp_norobovetterlabels_renamedcols_nomissingval_symsecphase_confirmedkoiperiod_sec_rba_cnt0n_koiperiodonlydiff.csv',
    index=False)


# %% For Confirmed KOIs whose period was updated, set secondary MES and transit depth, and secondary geometric albedo
# and planet effective temperature comparison statistics to zero


tce_tbl = pd.read_csv(
    '/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17_DR25/q1_q17_dr25_tce_2020.09.28_10.36.22_stellar_koi_cfp_norobovetterlabels_renamedcols_nomissingval_symsecphase_confirmedkoiperiod_sec_rba_cnt0n_koiperiodonlydiff_recomputedparams_7-26-2021.csv')

koi_tce_period_same = tce_tbl["tce_period"] == tce_tbl["koi_period"]

print(f'Number of Confirmed KOIs with updated period: {len(tce_tbl.loc[koi_tce_period_same])}')

tce_tbl.loc[koi_tce_period_same, ['wst_depth', 'tce_maxmes', 'tce_albedo_stat', 'tce_ptemp_stat']] = 0

# set albedo to 1 and planet effective temp. to planet equilibrium temp

tce_tbl.loc[koi_tce_period_same, ['tce_albedo', 'tce_ptemp']] = \
    tce_tbl.loc[koi_tce_period_same, ['tce_eqt']].apply(update_albedo_and_ptemp, axis=1)

tce_tbl.to_csv(
    '/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17_DR25/q1_q17_dr25_tce_2020.09.28_10.36.22_stellar_koi_cfp_norobovetterlabels_renamedcols_nomissingval_symsecphase_confirmedkoiperiod_sec_rba_cnt0n_koiperiodonlydiffsec_recomputedparams_7-26-2021.csv',
    index=False)

# %%

tbl_fp = Path('/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17_DR25/'
              '19-08-21_07:21/q1_q17_dr25_tce_2020.09.28_10.36.22_stellar_koi_cfp_norobovetterlabels_renamedcols_nomissingval_symsecphase.csv')

logging.basicConfig(filename=tbl_fp.parent / f'confirmed_kois_ephem.log',
                    level=logging.INFO,
                    format='%(message)s',
                    filemode='w')

tce_tbl = pd.read_csv(tbl_fp)
logging.info(f'Using TCE table: {tbl_fp}')

thr = 0.3
logging.info(f'Using threshold: {thr}')

confkois_ephem_replaced = []
for tce_i, tce in tce_tbl.iterrows():
    if tce['koi_disposition'] == 'CONFIRMED':  # look only at Confirmed KOIs

        if np.abs(tce['tce_period'] - tce['koi_period']) / \
                max(tce['tce_period'], tce['koi_period']) > thr:  # period difference is more than thr * 100%
            tce_tbl.loc[tce_i, ['tce_period', 'tce_time0bk']] = tce[['koi_period', 'koi_time0bk']].values

            tce_tbl.loc[tce_i, ['wst_depth', 'tce_maxmes', 'tce_albedo_stat', 'tce_ptemp_stat']] = 0
            tce_tbl.loc[tce_i, ['tce_albedo', 'tce_ptemp']] = np.array([1, tce_tbl.loc[tce_i, 'tce_eqt']])

            confkois_ephem_replaced.append((tce['target_id'], tce['tce_plnt_num']))

logging.info(f'TCE count: {len(confkois_ephem_replaced)} '
             f'({len(tce_tbl.loc[tce_tbl["koi_disposition"] == "CONFIRMED"])} Confirmed KOIs)')
logging.info(f'TCEs affected: {confkois_ephem_replaced}')

tce_tbl_fp_out = tbl_fp.parent / f'{tbl_fp.stem}_cpkoiperiod.csv'
logging.info(f'Saving TCE table: {tce_tbl_fp_out}')
tce_tbl.to_csv(tce_tbl_fp_out, index=False)
