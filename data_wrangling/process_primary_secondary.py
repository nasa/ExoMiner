""" Set matched secondary TCEs features according to their matched primary TCE. """

import logging
# 3rd party
import os
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.simplefilter(('error'))

# local
from data_wrangling.compute_derived_parameters import compute_plnt_eff_temp_stat, compute_sec_geo_albedo_stat, \
    estimate_plnt_eff_temp, estimate_sec_geo_albedo

timestamp = datetime.now().strftime('%d-%m-%y_%H:%M')

workRootDir = Path('/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17_DR25/')
workDir = workRootDir / f'{timestamp}'
workDir.mkdir(exist_ok=True)

logging.basicConfig(filename=os.path.join(workDir, f'sec_updt_params_match_{timestamp}.log'),
                    level=logging.INFO,
                    format='%(message)s',
                    filemode='w')

tceTblFp = Path('/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17_DR25/'
                'q1_q17_dr25_tce_2020.09.28_10.36.22_stellar_koi_cfp_norobovetterlabels_renamedcols_nomissingval.csv')
tceTbl = pd.read_csv(tceTblFp)
logging.info(f'Using TCE table: {tceTblFp}')

tceTblOutFp = workDir / f'{tceTblFp.stem}_symsecphase.csv'

secondaryMatchedTblFp = Path('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Analysis/wks_tce_match/'
                             '19-08-21_06:09/matched_tces_phasematchthr_0.05_periodmatchthr_0.001_19-08-21_06:09_final.csv')
secondaryMatchedTbl = pd.read_csv(secondaryMatchedTblFp)
logging.info(f'Using matched primary-secondary TCE table: {secondaryMatchedTblFp}')

updt_cols = ['tce_albedo', 'tce_albedo_err', 'tce_albedo_stat', 'tce_ptemp', 'tce_ptemp_err', 'tce_ptemp_stat']
for col in updt_cols:
    tceTbl[f'{col}_old'] = np.nan
for tce_i, tce in secondaryMatchedTbl.iterrows():
    primaryTce = (tceTbl['target_id'] == tce['target_id']) & (tceTbl['tce_plnt_num'] == tce['primary_tce'])
    secondaryTce = (tceTbl['target_id'] == tce['target_id']) & (tceTbl['tce_plnt_num'] == tce['secondary_tce'])

    tceTbl.loc[secondaryTce, [f'{col}_old' for col in updt_cols]] = tceTbl.loc[secondaryTce, updt_cols].values

    # change phase of the matched secondary to the symmetric of the phase of the matched primary
    tceTbl.loc[secondaryTce, ['tce_maxmesd']] = - tceTbl.loc[primaryTce, ['tce_maxmesd']].values[0]
    # change secondary MES and transit depth of matched secondary to the values of the matched primary
    tceTbl.loc[secondaryTce, ['tce_maxmes', 'wst_depth']] = \
        tceTbl.loc[primaryTce, ['tce_max_mult_ev', 'transit_depth']].values

    # recompute secondary geometric albedo and planet effective temperature
    sg_albedo, sg_albedo_unc = estimate_sec_geo_albedo(
        tceTbl.loc[primaryTce, 'transit_depth'].values[0],
        tceTbl.loc[secondaryTce, 'tce_prad'].values[0],
        tceTbl.loc[secondaryTce, 'tce_sma'].values[0],
        sec_tr_depth_unc=tceTbl.loc[primaryTce, 'tce_depth_err'].values[0],
        plnt_radius_unc=tceTbl.loc[secondaryTce, 'tce_prad_err'].values[0],
        sma_unc=tceTbl.loc[secondaryTce, 'tce_sma_err'].values[0],
    )
    tceTbl.loc[secondaryTce, ['tce_albedo', 'tce_albedo_err']] = [sg_albedo, sg_albedo_unc]

    plnt_eff_temp, plnt_eff_temp_unc = estimate_plnt_eff_temp(
        tceTbl.loc[secondaryTce, 'tce_steff'].values[0],
        tceTbl.loc[primaryTce, 'transit_depth'].values[0],
        tceTbl.loc[secondaryTce, 'tce_ror'].values[0],
        st_eff_temp_unc=tceTbl.loc[secondaryTce, 'tce_steff_err'].values[0],
        sec_tr_depth_unc=tceTbl.loc[primaryTce, 'tce_depth_err'].values[0],
        ror_unc=tceTbl.loc[secondaryTce, 'tce_ror_err'].values[0],
    )
    tceTbl.loc[secondaryTce, ['tce_ptemp', 'tce_ptemp_err']] = [plnt_eff_temp, plnt_eff_temp_unc]

    tceTbl.loc[secondaryTce, ['tce_albedo_stat']] = compute_sec_geo_albedo_stat(
        sg_albedo,
        sg_albedo_unc
    )

    tceTbl.loc[secondaryTce, ['tce_ptemp_stat']] = compute_plnt_eff_temp_stat(
        plnt_eff_temp,
        tceTbl.loc[secondaryTce, ['tce_eqt']].values[0],
        plnt_eff_temp_unc,
        tceTbl.loc[secondaryTce, ['tce_eqt_err']].values[0]
    )

tceTbl.to_csv(tceTblOutFp, index=False)

logging.info(f'Created TCE table: {tceTblOutFp}')

logging.shutdown()

# %%

for col in updt_cols:
    f, ax = plt.subplots(2, 1)
    ax[0].scatter(tceTbl[col], tceTbl[f'{col}_old'], c='b', s=8)
    ax[1].set_xlabel(f'{col}_old')
    ax[0].set_ylabel(f'{col}_new')
    ax[1].set_ylabel(f'Relative error {col}')
    ax[1].scatter(tceTbl[f'{col}_old'], np.abs(tceTbl[f'{col}_old'] - tceTbl[f'{col}']) /
                  tceTbl[f'{col}_old'], c='b', s=8)
    # ax.set_yscale('log')
    # ax.set_xscale('log')

    if col == 'tce_ptemp':
        ax[0].set_xlim([0, 1e4])
        ax[0].set_ylim([0, 1e4])
        ax[1].set_xlim([0, 1e4])
        ax[1].set_ylim([0, 1])
        # ax[0].set_yscale('log')
        # ax[0].set_xscale('log')
    elif col == 'tce_ptemp_err':
        ax[0].set_xlim([-1, 1e4])
        ax[0].set_ylim([-1, 1e4])
        ax[1].set_xlim([-1, 1e4])
        ax[1].set_ylim([0, 1])
    elif col == 'tce_albedo':
        ax[0].set_xlim([0, 1e4])
        ax[0].set_ylim([0, 1e4])
        ax[1].set_xlim([0, 1e4])
        ax[1].set_ylim([0, 1])
    elif col == 'tce_albedo_err':
        ax[0].set_xlim([-1, 1e4])
        ax[0].set_ylim([-1, 1e4])
        ax[1].set_xlim([-1, 1e4])
        ax[1].set_ylim([0, 1])
    elif col == 'tce_albedo_stat':
        ax[0].set_xlim([-1e2, 1e2])
        ax[0].set_ylim([-1e2, 1e2])
        ax[1].set_xlim([-1e2, 1e2])
        ax[1].set_ylim([0, 1])
    elif col == 'tce_ptemp_stat':
        ax[0].set_xlim([-20, 1e2])
        ax[0].set_ylim([-20, 1e2])
        ax[1].set_xlim([-20, 1e2])
        ax[1].set_ylim([0, 1])

    ax[0].grid(True)
    ax[1].grid(True)
    f.savefig(workDir / f'scatter_{col}.png')
    # plt.close()
