""" Script created to remove Possible Planet KOIs from the training, validation and test Kepler TCE tables. """

# 3rd party
from pathlib import Path
import pandas as pd

# directory with the training, validation and test set tables
datasets_tbls_dir = Path('/data5/tess_project/Data/tfrecords/Kepler/Q1-Q17_DR25/train-val-test-sets/split_6-1-2020/')

# get TCE table with all 34k TCEs for Q1-Q17 DR25
tce_tbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17_DR25/q1_q17_dr25_tce_2020.09.28_10.36.22_stellar_koi_cfp_norobovetterlabels_renamedcols_nomissingval_symsecphase_confirmedkoiperiod_sec_rba_cnt0n_koiperiodonlydiff_nanstellar.csv')
print(f'Number of TCEs before removing Possible Planet KOIs from the dataset: {len(tce_tbl)}')

# get only Possible Planet KOIs
tce_tbl = tce_tbl.loc[(tce_tbl['fpwg_disp_status'] == 'POSSIBLE PLANET')
                        & (tce_tbl['koi_disposition'] != 'CONFIRMED')]
print(f'Number of TCEs after keeping only Possible Planet KOIs from the dataset: {len(tce_tbl)}')

count_tces = {}

# remove Possible Planet KOIs from training, validation and test sets
for dataset_tbl_fp in (datasets_tbls_dir / 'with_pps').iterdir():

    dataset_tbl = pd.read_csv(dataset_tbl_fp)

    pps_filt = dataset_tbl['kepoi_name'].isin(tce_tbl['kepoi_name'])

    dataset_tbl = dataset_tbl.loc[~pps_filt]

    count_tces[dataset_tbl_fp.stem] = {'pps_removed': pps_filt.sum(), 'num_tces': len(dataset_tbl)}

    dataset_tbl.to_csv(datasets_tbls_dir / f'{dataset_tbl_fp.name}', index=False)

print(count_tces)
