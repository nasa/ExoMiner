"""
Extract RUWE values for TICs from Gaia data releases.

This works by querying the TIC catalog for a set of TIC IDs. From the TIC catalog, we get the Gaia DR2 source ids that
can be used to query Gaia DR2 catalog and extract RUWE values for that set of TIC IDs.
"""

# 3rd party
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
# from datetime import datetime
import logging
from astroquery.mast import Catalogs

# local
from data_wrangling.ruwe.query_gaia import query_gaia

#%% Set run parameters

res_dir = Path('/Users/msaragoc/Projects/exoplanet_transit_classification/data/ephemeris_tables/tess/DV_SPOC_mat_files/preprocessing_tce_tables/09-25-2023_1608/gaiadr2_tics_ruwe')
res_dir.mkdir(exist_ok=True)

# set list of TIC IDs to query
# file path to TCE table
tce_tbl_fp = Path('/Users/msaragoc/Projects/exoplanet_transit_classification/data/ephemeris_tables/tess/DV_SPOC_mat_files/preprocessing_tce_tables/09-25-2023_1608/tess_2min_tces_dv_s1-s68_09-25-2023_1608.csv')
tce_tbl = pd.read_csv(tce_tbl_fp)
tic_ids = tce_tbl['target_id'].unique()

query_gaia_dr = 'gaiadr2'

#%% Run

logger = logging.getLogger(name='ruwe')  # set up logger
logger_handler = logging.FileHandler(filename=res_dir / f'ruwe_run.log', mode='w')
logger_formatter = logging.Formatter('%(asctime)s - %(message)s')
logger.setLevel(logging.INFO)
logger_handler.setFormatter(logger_formatter)
logger.addHandler(logger_handler)
logger.info(f'Starting run...')

# get Gaia DR2 source ids from tic ids in the tce table
logger.info('Getting Gaia DR2 source ids from TIC')
catalog_data = Catalogs.query_criteria(catalog='TIC', ID=tic_ids.tolist()).to_pandas()
catalog_data.to_csv(res_dir / 'tics.csv', index=False)

source_ids_tic_tbl = catalog_data[['ID', 'GAIA']].rename(columns={'GAIA': 'source_id', 'ID': 'target_id'})

# remove TICs with missing source id
logger.info('Removing TICs with missing Gaia source ID')
source_ids_tic_tbl = source_ids_tic_tbl.loc[~source_ids_tic_tbl['source_id'].isna()]

source_ids_tic_tbl = source_ids_tic_tbl.astype(dtype={'source_id': np.int64, 'target_id': np.int64})

# count number of occurrences for each source id
source_ids_cnts = \
    source_ids_tic_tbl['source_id'].value_counts().to_frame('n_occur_source_ids').reset_index().rename(
        columns={'index': 'source_id'})
source_ids_tic_tbl = source_ids_tic_tbl.merge(source_ids_cnts,
                                              on=['source_id'],
                                              how='left',
                                              validate='many_to_one')

# query Gaia DR2
query_gaia(source_ids_tic_tbl[['source_id']], query_gaia_dr, res_dir, logger=logger)
output_fp = res_dir / f'{query_gaia_dr}.csv'  # save results from query to csv file
logger.info(f'Query finished and results saved to {str(output_fp)}.')

ruwe_tbl = pd.read_csv(output_fp)

# in case there are duplicate source ids in the TIC-source id table, many_to_one
if (source_ids_tic_tbl["source_id"].value_counts() > 1).sum() > 0:
    ruwe_tbl = source_ids_tic_tbl[['target_id', 'source_id', 'n_occur_source_ids']].merge(
        ruwe_tbl.drop_duplicates('source_id'),
        on=['source_id'],
        how='left',
        validate='many_to_one')
else:
    ruwe_tbl = source_ids_tic_tbl[['target_id', 'source_id', 'n_occur_source_ids']].merge(
        ruwe_tbl.drop_duplicates('source_id'),
        on=['source_id'],
        how='left',
        validate='one_to_one')

ruwe_tbl.to_csv(res_dir / f'{output_fp.stem}_with_ticid.csv', index=False)
logger.info(f'Number of TICs without RUWE value: {ruwe_tbl["ruwe"].isna().sum()} ouf of {len(ruwe_tbl)}')

# plot histogram of RUWE values
bins = np.linspace(0, 2, 100)  # np.logspace(0, 2, 100)

f, ax = plt.subplots()
ax.hist(ruwe_tbl['ruwe'], bins=bins, edgecolor='k')
ax.set_xlabel('RUWE')
ax.set_ylabel('Target counts')
# ax.legend()
ax.set_title(f'RUWE source: {query_gaia_dr}')
ax.set_yscale('log')
ax.set_xlim([0, 2])
# ax.set_xscale('log')
# plt.show()
f.savefig(res_dir / 'hist_ruwe_tics.png')
