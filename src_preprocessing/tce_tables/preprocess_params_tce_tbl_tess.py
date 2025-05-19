"""
Preprocess features in the TESS TCE table.
- Set categorial magnitude based on saturation threshold: 0 for saturated targets; 1 otherwise.
- Shift (translate) magnitude range so saturation threshold is set to zero.
- Adjust transit source centroid offset estimates from out-of-transit ('tce_dicco_msky') and target's TIC coordinates
('tce_dikco_msky') based on pixel scale.
ratio between TESS and Kepler.
- Create transit source offset ratio by dividing value by uncertainty for both oot and target offsets.
"""

# 3rd party
import pandas as pd
from pathlib import Path
import numpy as np

TESS_PX_SCALE = 21  # arcsec
KEPLER_PX_SCALE = 3.98  # arcsec
TESS_TO_KEPLER_PX_SCALE_RATIO = TESS_PX_SCALE / KEPLER_PX_SCALE

TESS_MISSION_VALUE = 0

TESS_MAG_THR = 7  # TMag used for saturated stars

def preprocess_parameters_tess_tce_table(tce_tbl):
    """ Preprocess parameters in TCE table `tce_tbl`. `tce_tbl` must contain columns: TESS magnitude 'mag', and
    DV SPOC centroid offset and uncertainty estimates ('_err') from the out-of-transit 'tce_dicco_msky' and target TIC
    coordinates 'tce_dikco_msky'.

    :param tce_tbl: pandas DataFrame, TCE table

    :return: tce_tbl, pandas DataFrame with updated parameters.
    """

    tce_tbl['mission'] = TESS_MISSION_VALUE

    # create categorical magnitude
    tce_tbl['mag_cat'] = 0.0
    tce_tbl.loc[tce_tbl['mag'] > TESS_MAG_THR, 'mag_cat'] = 1.0
    tce_tbl.loc[tce_tbl['mag'].isna(), 'mag_cat'] = np.nan  # set to nan if magnitude is nan
    # set shifted magnitude
    tce_tbl['mag_shift'] = tce_tbl['mag'] - TESS_MAG_THR

    # create adjusted
    for diff_img_centr_feat in ['tce_dikco_msky', 'tce_dikco_msky_err', 'tce_dicco_msky', 'tce_dicco_msky_err']:

        tce_tbl[f'{diff_img_centr_feat}_original'] = tce_tbl[diff_img_centr_feat]

        tce_tbl[f'{diff_img_centr_feat}'] = tce_tbl[diff_img_centr_feat] / TESS_TO_KEPLER_PX_SCALE_RATIO

    # set missing values to placeholder value
    tce_tbl.loc[tce_tbl['tce_dikco_msky_err_original'] == -1, ['tce_dikco_msky', 'tce_dikco_msky_err']] = [0, -1]
    tce_tbl.loc[tce_tbl['tce_dicco_msky_err_original'] == -1, ['tce_dicco_msky', 'tce_dicco_msky_err']] = [0, -1]

    # create transit source offset ratios
    tce_tbl['tce_dikco_msky_rat'] = (
        tce_tbl.apply(lambda x: np.nan if x['tce_dikco_msky_err_original'] == -1 else x['tce_dikco_msky_original'] /
                                                                                      x['tce_dikco_msky_err_original'],
                      axis=1))
    tce_tbl['tce_dicco_msky_rat'] = (
        tce_tbl.apply(lambda x: np.nan if x['tce_dicco_msky_err_original'] == -1 else x['tce_dicco_msky_original'] /
                                                                                      x['tce_dicco_msky_err_original'],
                      axis=1))

    return tce_tbl


if __name__ == '__main__':

    tce_tbl_fp = Path('')

    # load TCE table
    tce_tbl = pd.read_csv(tce_tbl_fp)

    tce_tbl = preprocess_parameters_tess_tce_table(tce_tbl)

    tce_tbl.to_csv(tce_tbl_fp.parent / f'{tce_tbl_fp.stem}_features_adjusted.csv', index=False)
