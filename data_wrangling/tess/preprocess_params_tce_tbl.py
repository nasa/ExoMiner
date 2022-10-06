""" """

# 3rd party
import pandas as pd
from pathlib import Path

#%% For TESS

exp_dir = Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/data/ephemeris_tables/tess/DV_SPOC_mat_files/10-05-2022_1338/')
tce_tbl_fp = exp_dir / 'tess_tces_dv_s1-s55_10-05-2022_1338_ticstellar_ruwe_tec_tsoebs_ourmatch.csv'
tce_tbl = pd.read_csv(tce_tbl_fp)

# create categorical magnitude
tess_mag_thr = 7
tce_tbl['mag_cat'] = 0
tce_tbl.loc[tce_tbl['mag'] > tess_mag_thr, 'mag_cat'] = 1

# create adjusted
tess_px_scale = 21  # arcsec
kepler_px_scale = 3.98  # arcsec
tess_to_kepler_px_scale_factor = tess_px_scale / kepler_px_scale
kepler_lower_bound_dicco_msky_err = 0.667
tess_lower_bound_dicco_msky_err = 2.5
delta_lower_bound_dicco_msky_err = tess_lower_bound_dicco_msky_err - kepler_lower_bound_dicco_msky_err
for diff_img_centr_feat in ['tce_dikco_msky', 'tce_dikco_msky_err', 'tce_dicco_msky', 'tce_dicco_msky_err']:
    if diff_img_centr_feat in ['tce_dicco_msky_err', 'tce_dikco_msky_err']:
        tce_tbl[f'{diff_img_centr_feat}_adjscl'] = (tce_tbl[diff_img_centr_feat] - delta_lower_bound_dicco_msky_err) / \
                                                   tess_to_kepler_px_scale_factor
    else:
        tce_tbl[f'{diff_img_centr_feat}_adjscl'] = tce_tbl[diff_img_centr_feat] / tess_to_kepler_px_scale_factor

tce_tbl.to_csv(tce_tbl_fp.parent / f'{tce_tbl_fp.stem}_preproc.csv', index=False)

#%% For Kepler

exp_dir = Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/data/ephemeris_tables/')
tce_tbl_fp = exp_dir / '.csv'
tce_tbl = pd.read_csv(tce_tbl_fp)

# create categorical magnitude
kepler_mag_thr = 12
tce_tbl['mag_cat'] = 0
tce_tbl.loc[tce_tbl['mag'] > tess_mag_thr, 'mag_cat'] = 1

tce_tbl.to_csv(tce_tbl_fp.parent / f'{tce_tbl_fp.stem}_preproc.csv', index=False)
