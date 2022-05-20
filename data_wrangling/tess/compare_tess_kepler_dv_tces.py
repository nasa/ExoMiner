"""
Compare TCE DV parameters between TESS and Kepler.
"""

# 3rd party
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# %%

res_dir = Path('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Analysis/kepler_tess_dv_tces')
res_dir.mkdir(exist_ok=True)

kepler_tce_tbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17_DR25/'
                             'q1_q17_dr25_tce_2020.09.28_10.36.22_stellar_koi_cfp_norobovetterlabels_renamedcols_nomissingval_symsecphase_confirmedkoiperiod_sec_rba_cnt0n_koiperiodonlydiff_nanstellar.csv')
tess_tce_tbl = pd.read_csv(
    '/data5/tess_project/Data/Ephemeris_tables/TESS/DV_SPOC_mat_files/5-10-2021/tess_tces_s1-s35_renamedcols.csv')

features = [
    # 'tce_albedo',
    # 'tce_albedo_err',
    'tce_albedo_stat',
    # 'tce_ptemp',
    # 'tce_ptemp_err',
    'tce_ptemp_stat',
    'wst_depth',
    'tce_maxmes',
    # 'tce_eqt',
    'transit_depth',
    # 'tce_depth_err',
    'tce_duration',
    # 'tce_duration_err',
    'tce_prad',
    'tce_period',
    # 'tce_period_err',
    'tce_dikco_msky',
    'tce_dikco_msky_err',
    'tce_dicco_msky',
    'tce_dicco_msky_err',
    # 'tce_fwm_stat',
    # 'tce_fwm_srao',
    # 'tce_fwm_srao_err',
    # 'tce_fwm_sdeco',
    # 'tce_fwm_sdeco_err',
    # 'tce_fwm_prao',
    # 'tce_fwm_prao_err',
    # 'tce_fwm_pdeco',
    # 'tce_fwm_pdeco_err',
    # 'tce_max_mult_ev',
    # 'tce_rb_tcount0_trnorm',
    'tce_impact',
    # 'mag',
    # 'tce_rb_tcount0n',
    'tce_smass',
    'tce_steff',
    'tce_slogg',
    'tce_smet',
    'tce_sradius',
    'tce_sdens',
    'boot_fap',
    'tce_cap_stat',
    'tce_hap_stat',
    # 'tce_cap_hap_diff_stat',
]

bins = {
    # 'tce_eqt': np.linspace(-1e4, 1e4, 50, endpoint=True),
    # 'tce_albedo': np.linspace(0, 100, 50, endpoint=True),
    # 'tce_albedo_err': np.linspace(-10, 100, 50, endpoint=True),
    'tce_albedo_stat': np.linspace(-4, 4, 50, endpoint=True),
    # 'tce_ptemp': np.linspace(0, 1e4, 50, endpoint=True),
    # 'tce_ptemp_err': np.linspace(-10, 1e3, 50, endpoint=True),
    'tce_ptemp_stat': np.linspace(-10, 30, 100, endpoint=True),
    'tce_maxmes': np.linspace(0, 100, 50, endpoint=True),
    'wst_depth': np.linspace(-2e4, 2e4, 100, endpoint=True),
    'transit_depth': np.linspace(0, 2e4, 50, endpoint=True),
    # 'tce_depth_err': np.linspace(-1, 100, 50, endpoint=True),
    'tce_duration': np.linspace(0, 100, 50, endpoint=True),
    # 'tce_duration_err': np.linspace(-1, 20, 50, endpoint=True),
    'tce_prad': np.linspace(0, 20, 100, endpoint=True),
    'tce_period': np.linspace(0, 800, 100, endpoint=True),
    # 'tce_period_err': np.linspace(0, 0.1, 100, endpoint=True),
    'tce_dikco_msky': np.linspace(0, 20, 50, endpoint=True),
    'tce_dikco_msky_err': np.linspace(-1, 10, 50, endpoint=True),
    'tce_dicco_msky': np.linspace(0, 20, 50, endpoint=True),
    'tce_dicco_msky_err': np.linspace(-1, 10, 50, endpoint=True),
    # 'tce_fwm_stat': np.linspace(0, 1000, 50, endpoint=True),
    # 'tce_fwm_srao': np.linspace(-100, 100, 50, endpoint=True),
    # 'tce_fwm_srao_err': np.linspace(-100, 100, 50, endpoint=True),
    # 'tce_fwm_sdeco': np.linspace(-100, 100, 50, endpoint=True),
    # 'tce_fwm_sdeco_err': np.linspace(-10, 100, 50, endpoint=True),
    # 'tce_fwm_prao': np.linspace(-1e-1, 1e-1, 50, endpoint=True),
    # 'tce_fwm_prao_err': np.linspace(0, 1e-1, 50, endpoint=True),
    # 'tce_fwm_pdeco': np.linspace(-1e-1, 1e-1, 50, endpoint=True),
    # 'tce_fwm_pdeco_err': np.linspace(0, 1e-1, 50, endpoint=True),
    # 'tce_max_mult_ev': np.linspace(7.1, 1000, 50, endpoint=True),
    'tce_impact': np.linspace(0, 1, 100, endpoint=True),
    # 'tce_rb_tcount0_trnorm': np.linspace(0, 1, 100, endpoint=True),
    # 'mag': np.linspace(0, 20, 100, endpoint=True),
    # 'tce_rb_tcount0n': np.linspace(0, 1, 100, endpoint=True),
    'tce_smass': np.linspace(0, 4, 100, endpoint=True),
    'tce_steff': np.linspace(2000, 20000, 100, endpoint=True),
    'tce_slogg': np.linspace(-6, 6, 100, endpoint=True),
    'tce_smet': np.linspace(-2, 2, 100, endpoint=True),
    'tce_sradius': np.linspace(0, 10, 100, endpoint=True),
    'tce_sdens': np.linspace(0, 100, 100, endpoint=True),
    'boot_fap': np.logspace(-34, 0, 100, endpoint=True),
    'tce_cap_stat': np.linspace(-2000, 2000, 100, endpoint=True),
    'tce_hap_stat': np.linspace(-2000, 2000, 100, endpoint=True),
    # 'tce_cap_hap_diff_stat': np.linspace(-2000, 2000, 100, endpoint=True),
}

log_yscale = [
    'tce_fwm_stat',
    'tce_fwm_sdeco',
    'tce_fwm_sdeco_err',
    'tce_fwm_prao',
    'tce_fwm_prao_err',
    'tce_fwm_pdeco',
    'tce_fwm_pdeco_err',
    'tce_period_err',
    'tce_max_mult_ev',
    'tce_dikco_msky',
    'tce_dikco_msky_err',
    'tce_dicco_msky',
    'tce_dicco_msky_err',
    'transit_depth',
    'tce_duration',
    'tce_period',
    'wst_depth',
    'tce_rb_tcount0n',
    'boot_fap',
    'tce_ptemp_stat',
    'tce_maxmes',
    'tce_sdens',
    'tce_albedo_stat',
    'tce_hap_stat',
    'tce_cap_stat',
    'tce_cap_hap_diff_stat',
]

log_xscale = [
    'boot_fap'
]

labels = {'kepler': {'PC': {'zorder': 3, 'alpha': 0.7},
                     'AFP': {'zorder': 2, 'alpha': 0.7},
                     'NTP': {'zorder': 1, 'alpha': 1.0}
                     },
          'tess': {'KP': {'zorder': 6, 'alpha': 1.0},
                   'CP': {'zorder': 5, 'alpha': 0.6},
                   'FP': {'zorder': 4, 'alpha': 0.4},
                   # 'PC': {'zorder': 1},
                   # 'APC': {'zorder': 2},
                   'FA': {'zorder': 6, 'alpha': 0.4}}
          }
for feature in features:

    f, ax = plt.subplots(2, 2, figsize=(12, 8))
    f.suptitle('{}'.format(feature))
    ax[0, 0].hist(kepler_tce_tbl[feature], bins[feature], edgecolor='k')
    ax[0, 0].set_ylabel('Counts')
    ax[0, 0].set_title('Kepler')
    ax[0, 0].set_xlim([bins[feature][0], bins[feature][-1]])

    ax[0, 1].hist(tess_tce_tbl[feature], bins[feature], edgecolor='k')
    ax[0, 1].set_title('TESS')
    ax[0, 1].set_xlim([bins[feature][0], bins[feature][-1]])

    for disp in labels['kepler']:
        tce_tbl_disp_feat = kepler_tce_tbl.loc[kepler_tce_tbl['label'] == disp, feature]
        ax[1, 0].hist(tce_tbl_disp_feat, bins[feature], edgecolor='k', label=disp,
                      zorder=labels['kepler'][disp]['zorder'], alpha=labels['kepler'][disp]['alpha'])
    ax[1, 0].set_ylabel('Counts')
    ax[1, 0].set_xlabel('Feature Value')
    ax[1, 0].set_xlim([bins[feature][0], bins[feature][-1]])
    ax[1, 0].legend()

    for disp in labels['tess']:
        tce_tbl_disp_feat = tess_tce_tbl.loc[tess_tce_tbl['label'] == disp, feature]
        ax[1, 1].hist(tce_tbl_disp_feat, bins[feature], edgecolor='k', label=disp,
                      zorder=labels['tess'][disp]['zorder'], alpha=labels['tess'][disp]['alpha'])
    ax[1, 1].set_xlabel('Feature Value')
    ax[1, 1].set_xlim([bins[feature][0], bins[feature][-1]])
    ax[1, 1].legend()

    if feature in log_yscale:
        ax[0, 0].set_yscale('log')
        ax[0, 1].set_yscale('log')
        ax[1, 0].set_yscale('log')
        ax[1, 1].set_yscale('log')
    if feature in log_xscale:
        ax[0, 0].set_xscale('log')
        ax[0, 1].set_xscale('log')
        ax[1, 0].set_xscale('log')
        ax[1, 1].set_xscale('log')

    f.subplots_adjust(top=0.917,
                      bottom=0.073,
                      left=0.058,
                      right=0.973,
                      hspace=0.122,
                      wspace=0.132)
    f.savefig(res_dir / 'hist_{}.svg'.format(feature))
    plt.close()
