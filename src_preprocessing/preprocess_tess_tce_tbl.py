"""
Preprocess TESS SPOC TCE results:
- Set unique ID 'uid' for each TCE based on {tic_id}-{tce_plnt_num}-{sector_run}.
- Rename DV fields.
- Update stellar parameters using TIC-8 catalog.
- Get RUWE values for TICs using Gaia.
- Preprocess parameters in TCE table (e.g., create categorical magnitude based on saturation threshold; scale transit
source offset estimates based on pixel scale ration between TESS and Kepler).

"""

# 3rd party
from pathlib import Path
import pandas as pd
import numpy as np

# local
from src_preprocessing.tess_spoc_ffi.xml_tbls_rename_cols_add_uid import rename_dv_xml_fields
from src_preprocessing.stellar_parameters.update_tess_stellar_parameters_tce_tbl import updated_stellar_parameters_with_tic8
from src_preprocessing.ruwe.ruwe_in_tics import query_gaiadr_for_ruwe
from src_preprocessing.preprocess_params_tce_tbl_tess import preprocess_parameters_tess_tce_table

# set TCE table filepath
tce_tbl_fp = Path('/nobackupp19/msaragoc/work_dir/Kepler-TESS_exoplanet/data/Ephemeris_tables/TESS/tess_spoc_2min/tess_spoc_2min_s14-s86_toi-2095_fromdvxml_4-10-2025_1014/tess_spoc_2min_s14-s86_toi-2095_fromdvxml_4-10-2025_1014.csv')

# set results directory
res_dir = tce_tbl_fp.parent

# load TCE table
tce_tbl = pd.read_csv(tce_tbl_fp)

# rename DV names
tce_tbl_renamed_cols_uid = rename_dv_xml_fields(tce_tbl)

# set uid
tce_tbl_renamed_cols_uid['uid'] = tce_tbl_renamed_cols_uid.apply(lambda x: '{}-{}-S{}'.format(x['target_id'],
                                                                                              x['tce_plnt_num'],
                                                                                              x['sector_run']),
                                                axis=1)
# move uid to become leftmost column
tce_cols = ['uid'] + [col for col in tce_tbl_renamed_cols_uid.columns if col != 'uid']
tce_tbl_renamed_cols_uid = tce_tbl_renamed_cols_uid[tce_cols]

# updated stellar parameters using TIC-8
res_dir_stellar = res_dir / 'stellar_tic8'
res_dir_stellar.mkdir(exist_ok=True)
tce_tbl_tic8stellar = updated_stellar_parameters_with_tic8(tce_tbl_renamed_cols_uid, res_dir_stellar)

# get RUWE values from Gaia DR2
res_dir_ruwe = res_dir / 'ruwe'
res_dir_ruwe.mkdir(exist_ok=True)
tce_tbl_gaiadr2_ruwe = query_gaiadr_for_ruwe(tce_tbl_tic8stellar, res_dir_ruwe)

# preprocess parameters in TCE table
tce_tbl_preprocessed_params = preprocess_parameters_tess_tce_table(tce_tbl_gaiadr2_ruwe)

# add dispositions
tce_tbl_preprocessed_params.loc[:, 'label'] = 'UNK'
tce_tbl_preprocessed_params.loc[:, 'label_source'] = np.nan

tce_tbl_preprocessed_params.to_csv(res_dir / f'{tce_tbl_fp.stem}_stellartic8_ruwegaiadr2_preproc.csv', index=False)
