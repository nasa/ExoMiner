"""
Script used to test pipeline or parts of it.
"""

# 3rd party
# import lightkurve as lk
import numpy as np
from pathlib import Path
from astroquery.mast import Observations
from astropy.table import vstack
import pandas as pd
import re

#%%

ffi = pd.read_csv('/Users/msaragoc/Projects/exoplanet_transit_classification/data/ephemeris_tables/tess/tess_spoc_ffi/preprocessed_tce_tables/tess_spoc_ffi_s36-s72_multisector_s56-s69_fromdvxml_11-22-2024_0942/tess_spoc_ffi_s36-s72_multisector_s56-s69_sfromdvxml_11-22-2024_0942_renamed_cols_added_uid_ruwe_ticstellar_features_adjusted_label.csv')
twomin = pd.read_csv('/Users/msaragoc/Projects/exoplanet_transit_classification/data/ephemeris_tables/tess/tess_spoc_2min/tess_2min_tces_dv_s1-s88_3-27-2025_1316_label.csv')

#%%

# 1400803888 56-69 ffi  # 13829713 14-60 2min
tic_id = 13829713  # 1400803888  # 13829713
data_collection_mode = '2min'  # '2min'  # 'ffi'
# exp_time_collection_mode = 120
s_sector, e_sector = 14, 60
sector_arr = np.arange(s_sector, e_sector + 1)  # np.arange(56, 69)  # np.arange(14, 61)
lc_dir = Path('/Users/msaragoc/Downloads/test_exominer_pipeline')
lc_dir.mkdir(parents=True, exist_ok=True)

lc_sectors_patterns = [f'-s{str(sector).zfill(4)}' for sector in sector_arr]
sector_run_patern = f'-s{str(s_sector).zfill(4)}-s{str(e_sector).zfill(4)}'

# search_lc_res = lk.search_lightcurve(
#     target=f"tic{tic_id}",
#     mission='TESS',
#     author=('TESS-SPOC', 'SPOC'),
#     # exptime=exp_time_collection_mode,  # to select only 2-min data
#     sector=sector_arr,
#     mission='HLSP'
# )
# print(search_lc_res)
#
# # download FITS files that matched the query
# lc_res = search_lc_res.download_all(
#     download_dir=str(lc_dir),
#     flux_column='pdcsap_flux'
# )
# aa
obs_table = Observations.query_criteria(target_name=tic_id,
                                        obs_collection='TESS' if data_collection_mode == '2min' else 'HLSP',
                                        # obs_id='hlsp_tess-spoc*',
                                        )
# obs_table.write(str(lc_dir / f'obs_{tic_id}_{data_collection_mode}.csv'), format='csv')

# obs_lc_table = Observations.query_criteria(target_name=tic_id,
#                                         obs_collection='TESS',
#                                         provenance_name='SPOC',
#                                         dataURL='*lc.fits',
                                        # )
# # filter observation light curve table to get lc FITS files only for sectors requested
# obs_lc_table_sectors = obs_lc_table[[any(re.search(lc_sector_pattern, data_url) for lc_sector_pattern in lc_sectors_patterns) for data_url in obs_lc_table['dataURL']]]

# lc_products = Observations.get_product_list(obs_lc_table)
# lc_products = Observations.get_product_list(obs_lc_table_sectors)

# obs_dv_xml_table = Observations.query_criteria(target_name=tic_id,
#                                         obs_collection='TESS',
#                                         provenance='SPOC',
#                                         dataURL='*dvr.xml',
#                                         )
# filter observation light curve table to get lc FITS files only for sectors requested
# obs_lc_table_sectors = obs_lc_table[[any(re.search(lc_sector_pattern, data_url) for lc_sector_pattern in lc_sectors_patterns) for data_url in obs_lc_table['dataURL']]]

# lc_products = Observations.get_product_list(obs_lc_table_sectors)

# get table with all available products for queried observations
products = Observations.get_product_list(obs_table)
# products.write(str(lc_dir / f'products_{tic_id}_{data_collection_mode}.csv'), format='csv')

# filter for light curve FITS files
lc_products = products[[fn.endswith('lc.fits') for fn in products["productFilename"]]]
# filter lc FITS files for sectors of interest
lc_products = lc_products[[any(re.search(lc_sector_pattern, data_url) for lc_sector_pattern in lc_sectors_patterns) for data_url in lc_products['productFilename']]]
# lc_products.write(str(lc_dir / f'lc_products_{tic_id}{sector_run_patern}{data_collection_mode}.csv'), format='csv')

# filter for DV XML files
dv_xml_products = products[[fn.endswith('dvr.xml') for fn in products["productFilename"]]]
# filter DV XML files for sector run of interest
dv_xml_products = dv_xml_products[[bool(re.search(sector_run_patern, data_url)) for data_url in dv_xml_products['productFilename']]]

# combine tables for products to be downloaded
requested_products = vstack([lc_products, dv_xml_products])
requested_products.write(str(lc_dir / f'requested_products_{tic_id}{sector_run_patern}_{data_collection_mode}.csv'), format='csv', overwrite=True)

# Download requested products
requested_products_manifest = Observations.download_products(requested_products, download_dir=str(lc_dir), mrp_only=False)
requested_products_manifest.write(str(lc_dir / f'manifest_requested_products_{tic_id}{sector_run_patern}_{data_collection_mode}.csv'), format='csv', overwrite=True)

# # Download DV XML files
# dv_manifest = Observations.download_products(dv_xml_products, download_dir=str(lc_dir), mrp_only=False)
