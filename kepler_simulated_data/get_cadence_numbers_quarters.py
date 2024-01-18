"""
Compute interquarter gaps in cadence units. Required when preprocessing data for scrambling runs in Kepler Simulated
tests.
"""

# 3rd party
import numpy as np
from astropy.io import fits
from pathlib import Path
import pandas as pd

NUM_QUARTERS = 18  # 0-17
# find target with data in all quarters
lc_dir = Path('/Users/msaragoc/Projects/exoplanet_transit_classification/data/fits_files/kepler/q1_q17_dr25/lc/0011/001162345')

data_dict = {field: np.nan * np.ones(18, dtype='float')
             for field in ['quarter', 'first_cadence_no', 'last_cadence_no']}
for fp_i, lc_fp in enumerate(lc_dir.iterdir()):

    with fits.open(lc_fp) as hdu_list:

        data_dict['quarter'][fp_i] = hdu_list["PRIMARY"].header["QUARTER"]

        data_dict['first_cadence_no'][fp_i], data_dict['last_cadence_no'][fp_i] = hdu_list["LIGHTCURVE"].data["CADENCENO"][[0, -1]]

data_df = pd.DataFrame(data_dict)
data_df['total_no_cadences'] = data_df['last_cadence_no'] - data_df['first_cadence_no'] + 1
data_df = data_df.astype('int')
data_df = data_df.sort_values('quarter', ascending=True, inplace=False).reset_index(drop=True)

# compute interquarter cadence gap
data_df['interquarter_cadence_gap'] = 0
data_df['interquarter_cadence_gap'][:-1] = (data_df['first_cadence_no'][1:].values -
                                            data_df['last_cadence_no'][:-1].values - 1)

data_df['last_cadence_no_with_interquarter_gap'] = data_df['last_cadence_no'] + data_df['interquarter_cadence_gap']
data_df['total_no_cadences_with_interquarter_gap'] = data_df['total_no_cadences'] + data_df['interquarter_cadence_gap']

data_df['first_cadence_reset'] = data_df['first_cadence_no'] - 1105
data_df['last_cadence_reset'] = data_df['last_cadence_no_with_interquarter_gap'] - 1105

data_df.to_csv('/Users/msaragoc/Projects/exoplanet_transit_classification/data/ephemeris_tables/kepler/q1-q17_dr25/simulated_data/quarters_cadence_numbers.csv', index=False)
