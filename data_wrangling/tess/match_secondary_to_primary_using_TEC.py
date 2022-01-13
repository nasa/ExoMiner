"""
Match secondary to primary TCEs in TESS using flux triage TEC information about secondary of previous TCEs. Setting
phase of matched secondary to negative of primary phase. Settung secondary transit depth and MES to primary and
recompute planet effective temperature and geometric albedo and respective comparison statistics.
"""

# 3rd party
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

# local
from data_wrangling.compute_derived_parameters import compute_plnt_eff_temp_stat, compute_sec_geo_albedo_stat, \
    estimate_plnt_eff_temp, estimate_sec_geo_albedo

# TCE table
tce_tbl_fp = Path(
    '/data5/tess_project/Data/Ephemeris_tables/TESS/DV_SPOC_mat_files/11-29-2021/tess_tces_s1-s40_11-23-2021_1409_stellarparams_updated_tecfluxtriage_eb_label.csv')
tce_tbl = pd.read_csv(tce_tbl_fp)

# set NaN TEC flux triage comments to 'N/A' in order to filter TCE table for those TCEs that are secondary of
# previous TCEs
tce_tbl.loc[tce_tbl['tec_fluxtriage_comment'].isna(), 'tec_fluxtriage_comment'] = 'N/A'
tce_tbl_sec = tce_tbl.loc[tce_tbl['tec_fluxtriage_comment'].str.contains('SecondaryOfPN')]

# copy original values of secondary parameters
sec_cols = [
    'tce_maxmes',
    'tce_maxmesd',
    'wst_depth',
    'tce_albedo',
    'tce_albedo_err',
    'tce_ptemp',
    'tce_ptemp_err',
    'tce_albedo_stat',
    'tce_ptemp_stat'
]
tce_tbl = pd.concat([tce_tbl,
                     tce_tbl[sec_cols].copy(deep=True).rename(columns={col_name: f'{col_name}_old'
                                                                       for col_name in sec_cols}, inplace=False)],
                    axis=1)

tce_tbl['label_of_primary'] = 'N/A'  # column that holds the label of the respective primary

for tce_i, tce in tce_tbl_sec.iterrows():
    primary_tce_plnt_num = int(tce['tec_fluxtriage_comment'].split('_')[1])

    # find primary TCE
    primary_tce = tce_tbl.loc[(tce_tbl['target_id'] == tce['target_id']) &
                              (tce_tbl['tce_plnt_num'] == primary_tce_plnt_num) &
                              (tce_tbl['sector_run'] == tce['sector_run'])]

    assert len(primary_tce) == 1

    # set new values for secondary MES, transit depth and phase based on the matched primary TCE
    tce_tbl.loc[tce_i, 'tce_maxmes'] = primary_tce['tce_max_mult_ev'].values[0]
    tce_tbl.loc[tce_i, 'wst_depth'] = primary_tce['tce_depth'].values[0]
    tce_tbl.loc[tce_i, 'tce_maxmesd'] = -1 * primary_tce['tce_maxmesd'].values[0]
    tce_tbl.loc[tce_i, 'label_of_primary'] = primary_tce['label'].values[0]

    # recompute secondary geometric albedo and planet effective temperature
    sg_albedo, sg_albedo_unc = estimate_sec_geo_albedo(
        primary_tce['tce_depth'].values[0],
        tce['tce_prad'],
        tce['tce_sma'],
        primary_tce['tce_depth_err'].values[0],
        tce['tce_prad_err'],
        tce['tce_sma_err'],
    )
    tce_tbl.loc[tce_i, ['tce_albedo', 'tce_albedo_err']] = [sg_albedo, sg_albedo_unc]

    plnt_eff_temp, plnt_eff_temp_unc = estimate_plnt_eff_temp(
        tce['tce_steff'],
        primary_tce['tce_depth'].values[0],
        tce['tce_ror'],
        tce['tce_steff_err'],
        primary_tce['tce_depth_err'].values[0],
        tce['tce_ror_err'],
    )
    tce_tbl.loc[tce_i, ['tce_ptemp', 'tce_ptemp_err']] = [plnt_eff_temp, plnt_eff_temp_unc]

    tce_tbl.loc[tce_i, ['tce_albedo_stat']] = compute_sec_geo_albedo_stat(
        sg_albedo,
        sg_albedo_unc
    )

    tce_tbl.loc[tce_i, ['tce_ptemp_stat']] = compute_plnt_eff_temp_stat(
        plnt_eff_temp,
        tce['tce_eqt'],
        plnt_eff_temp_unc,
        tce['tce_eqt_err']
    )

tce_tbl.to_csv(tce_tbl_fp.parent / f'{tce_tbl_fp.stem}_tecsec.csv', index=False)

# %% plot changed secondary features

res_dir = Path('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Analysis/tess_tec_secondaries_analysis_12-10-2021')
res_dir.mkdir(exist_ok=True)

tce_tbl_fp = Path(
    '/data5/tess_project/Data/Ephemeris_tables/TESS/DV_SPOC_mat_files/11-29-2021/tess_tces_s1-s40_11-23-2021_1409_stellarparams_updated_tecfluxtriage_eb_label_tecsec.csv')
tce_tbl = pd.read_csv(tce_tbl_fp)

tce_tbl.loc[tce_tbl['tec_fluxtriage_comment'].isna(), 'tec_fluxtriage_comment'] = 'N/A'
tce_tbl_sec = tce_tbl.loc[tce_tbl['tec_fluxtriage_comment'].str.contains('SecondaryOfPN')]

sec_cols = [
    'tce_maxmes',
    # 'tce_maxmesd',
    'wst_depth',
    # 'tce_albedo',
    # 'tce_albedo_err',
    # 'tce_ptemp',
    # 'tce_ptemp_err',
    'tce_albedo_stat',
    'tce_ptemp_stat'
]

for feature in sec_cols:

    f, ax = plt.subplots()
    ax.scatter(tce_tbl_sec[feature], tce_tbl_sec[f'{feature}_old'], s=8)
    ax.set_ylabel(f'{feature}_old')
    ax.set_xlabel(feature)
    if feature in ['wst_depth']:
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_xlim([1, 1e6])
        ax.set_ylim([1, 1e6])
    elif feature in ['tce_maxmes']:
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_xlim([7.1, 1e3])
        ax.set_ylim([7.1, 1e3])
    elif feature in ['tce_ptemp_stat']:
        ax.set_xlim([0, 1e2])
        ax.set_ylim([0, 1e2])
    elif feature in ['tce_albedo_stat']:
        ax.set_xlim([0, 2e1])
        ax.set_ylim([-2.5e1, 1e1])
    f.savefig(res_dir / f'scatter_{feature}_sec.png')
    plt.close()
    # aa
