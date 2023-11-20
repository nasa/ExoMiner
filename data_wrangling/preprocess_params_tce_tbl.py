""" Preprocess features in the TCE table. """

# 3rd party
import pandas as pd
from pathlib import Path
import numpy as np

#%% For TESS

exp_dir = Path('/Users/msaragoc/Projects/exoplanet_transit_classification/data/ephemeris_tables/tess/DV_SPOC_mat_files/preprocessing_tce_tables/09-25-2023_1608/')
tce_tbl_fp = exp_dir / 'tess_2min_tces_dv_s1-s68_09-25-2023_1608_ruwe_ticstellar.csv'
tce_tbl = pd.read_csv(tce_tbl_fp)

# create categorical magnitude
tess_mag_thr = 7
tce_tbl['mag_cat'] = 0.0
tce_tbl.loc[tce_tbl['mag'] > tess_mag_thr, 'mag_cat'] = 1.0
tce_tbl.loc[tce_tbl['mag'].isna(), 'mag_cat'] = np.nan  # set to nan if magnitude is nan
# set shifted magnitude
tce_tbl['mag_shift'] = tce_tbl['mag'] - tess_mag_thr

# create adjusted
tess_px_scale = 21  # arcsec
kepler_px_scale = 3.98  # arcsec
tess_to_kepler_px_scale_factor = tess_px_scale / kepler_px_scale
kepler_lower_bound_dicco_msky_err = 0.0667
tess_lower_bound_dicco_msky_err = 2.5
delta_lower_bound_dicco_msky_err = tess_lower_bound_dicco_msky_err / tess_to_kepler_px_scale_factor - kepler_lower_bound_dicco_msky_err
for diff_img_centr_feat in ['tce_dikco_msky', 'tce_dikco_msky_err', 'tce_dicco_msky', 'tce_dicco_msky_err']:
    if diff_img_centr_feat in ['tce_dicco_msky_err', 'tce_dikco_msky_err']:
        tce_tbl[f'{diff_img_centr_feat}_adjscl'] = tce_tbl[diff_img_centr_feat] / tess_to_kepler_px_scale_factor - \
                                                   delta_lower_bound_dicco_msky_err
    else:
        tce_tbl[f'{diff_img_centr_feat}_adjscl'] = tce_tbl[diff_img_centr_feat] / tess_to_kepler_px_scale_factor

# set missing values to placeholder value
tce_tbl.loc[tce_tbl['tce_dikco_msky_err'] == -1, ['tce_dikco_msky_adjscl', 'tce_dikco_msky_err_adjscl']] = [0, -1]
tce_tbl.loc[tce_tbl['tce_dicco_msky_err'] == -1, ['tce_dicco_msky_adjscl', 'tce_dicco_msky_err_adjscl']] = [0, -1]

tce_tbl.to_csv(tce_tbl_fp.parent / f'{tce_tbl_fp.stem}_features_adjusted.csv', index=False)

#%% For Kepler

# exp_dir = Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/data/ephemeris_tables/')
# tce_tbl_fp = exp_dir / '.csv'
tce_tbl_fp = Path('/Users/msaragoc/Projects/exoplanet_transit_classification/data/ephemeris_tables/kepler/q1-q17_dr25/simulated_data/dvOutputMatrix_allruns_renamed_updtstellar.csv')
tce_tbl = pd.read_csv(tce_tbl_fp)

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

tce_tbl.to_csv(tce_tbl_fp.parent / f'{tce_tbl_fp.stem}_preprocessed.csv', index=False)
