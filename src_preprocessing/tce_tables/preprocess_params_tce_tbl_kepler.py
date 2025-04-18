""" Preprocess features in the Kepler TCE table. """

# 3rd party
import pandas as pd
from pathlib import Path
import numpy as np

def preprocess_parameters_kepler_tce_table(tce_tbl):
    """ Preprocess parameters in TCE table `tce_tbl`. `tce_tbl` must contain columns: Kepler magnitude 'mag', and
    DV SPOC centroid offset and uncertainty estimates ('_err') from the out-of-transit 'tce_dicco_msky' and target KIC
    coordinates 'tce_dikco_msky'.

    :param tce_tbl: pandas DataFrame, TCE table

    :return: tce_tbl, pandas DataFrame with updated parameters.
    """

    tce_tbl['mission'] = 1

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

    # create transit source offset ratios
    tce_tbl['tce_dikco_msky_rat'] = (
        tce_tbl.apply(lambda x: np.nan if x['tce_dikco_msky_err'] == -1 else x['tce_dikco_msky'] /
                                                                             x['tce_dikco_msky'],
                      axis=1))
    tce_tbl['tce_dicco_msky_rat'] = (
        tce_tbl.apply(lambda x: np.nan if x['tce_dicco_msky_err'] == -1 else x['tce_dicco_msky'] /
                                                                             x['tce_dicco_msky'],
                      axis=1))

    return tce_tbl


if __name__ == '__main__':

    tce_tbl_fp = Path('')

    # load TCE table
    tce_tbl = pd.read_csv(tce_tbl_fp)

    tce_tbl = preprocess_parameters_kepler_tce_table(tce_tbl)

    tce_tbl.to_csv(tce_tbl_fp.parent / f'{tce_tbl_fp.stem}_features_adjusted.csv', index=False)
