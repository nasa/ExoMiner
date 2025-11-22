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
from src_preprocessing.tce_tables.xml_tbls_rename_cols_add_uid import rename_dv_xml_fields
from src_preprocessing.tce_tables.stellar_parameters.update_tess_stellar_parameters_tce_tbl import (
    updated_stellar_parameters)
from src_preprocessing.tce_tables.ruwe.ruwe_in_tics import query_gaiadr_for_ruwe
from src_preprocessing.tce_tables.preprocess_params_tce_tbl_tess import preprocess_parameters_tess_tce_table


def preprocess_tce_table(tce_tbl_fp, res_dir, stellar_parameters_source=None, ruwe_source=None):
    """ Preprocess TCE table.

    Args:
        tce_tbl_fp: Path, filepath to TCE table
        res_dir: Path, results directory
        stellar_parameters_source: str/Path, the stellar parameters source to use for the queried TICs.
        ruwe_source: str/Path, the RUWE source to use for the queried TICs.

    Returns: tce_tbl_preprocessed_params, pandas.DataFrame with preprocessed TCE table

    """

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

    # updated stellar parameters from TIC v8 or some other source defined by `stellar_parameters_source`
    res_dir_stellar = res_dir / 'stellar_parameters'
    res_dir_stellar.mkdir(exist_ok=True)
    tce_tbl_tic8stellar = updated_stellar_parameters(tce_tbl_renamed_cols_uid, res_dir_stellar,
                                                     stellar_parameters_source)

    # get RUWE values from Gaia DR2 or some other source defined by `ruwe_source`
    res_dir_ruwe = res_dir / 'ruwe'
    res_dir_ruwe.mkdir(exist_ok=True)
    tce_tbl_ruwe = query_gaiadr_for_ruwe(tce_tbl_tic8stellar, res_dir_ruwe, ruwe_source)

    # preprocess parameters in TCE table
    tce_tbl_preprocessed_params = preprocess_parameters_tess_tce_table(tce_tbl_ruwe)

    # add dispositions
    tce_tbl_preprocessed_params.loc[:, 'label'] = 'UNK'
    tce_tbl_preprocessed_params.loc[:, 'label_source'] = np.nan

    return tce_tbl_preprocessed_params


def update_num_tois_in_tic(
    tce_tbl: pd.DataFrame,
    toi_tbl: pd.DataFrame,
    toi_id_col: str = 'toi_id',
    target_col: str = 'target_id',
    separator: str = '_'
) -> pd.DataFrame:
    """
    Merge TOI information into a TCE table by counting and listing TOIs per target.

    Parameters
    ----------
    tce_tbl : pd.DataFrame
        The TCE table containing target IDs.
    toi_tbl : pd.DataFrame
        The TOI table containing target IDs and TOI IDs.
    toi_id_col : str, optional
        Column name for TOI IDs in toi_tbl. Default is 'toi_id'.
    target_col : str, optional
        Column name for target IDs in both tables. Default is 'target_id'.
    separator : str, optional
        Separator used when joining TOI IDs into a string. Default is '_'.

    Returns
    -------
    pd.DataFrame
        Updated TCE table with two new columns:
        - 'n_tois_in_tic': Number of TOIs for each target.
        - 'tois_in_tic': Sorted, unique TOI IDs joined by the separator.
    """

    # drop columns if they already exist
    tce_tbl = tce_tbl.drop(columns=['n_tois_in_tic', 'tois_in_tic'], errors='ignore')

    targets_tois = toi_tbl.groupby(target_col, sort=False).agg(
        n_tois_in_tic=(toi_id_col, 'count'),
        tois_in_tic=(toi_id_col, lambda x: separator.join(sorted(map(str, set(x)))))
    )
    
    # merge the result back into the original table
    tce_tbl = tce_tbl.merge(targets_tois, on=target_col, how='left', validate='many_to_one')

    # fill NaNs if any (in case some target_ids had no TOIs)
    tce_tbl['n_tois_in_tic'] = tce_tbl['n_tois_in_tic'].fillna(0).astype(int)
    tce_tbl['tois_in_tic'] = tce_tbl['tois_in_tic'].fillna('')
    
    return tce_tbl


if __name__ == "__main__":

    # # set TCE table filepath
    # tce_tbl_fp = Path('/nobackupp19/msaragoc/work_dir/Kepler-TESS_exoplanet/data/Ephemeris_tables/TESS/tess_spoc_2min/tess_spoc_2min_s14-s86_toi-2095_fromdvxml_4-10-2025_1014/tess_spoc_2min_s14-s86_toi-2095_fromdvxml_4-10-2025_1014.csv')
    # # set results directory
    # res_dir = tce_tbl_fp.parent
    # tce_tbl_preprocessed_params = preprocess_tce_table(tce_tbl_fp, res_dir)
    # tce_tbl_preprocessed_params.to_csv(res_dir / f'{tce_tbl_fp.stem}_stellartic8_ruwegaiadr2_preproc.csv', index=False)
    
    # update number of TOIs in TIC
    print('Updating TCE table with number of TOIs in TICs...')
    toi_tbl = pd.read_csv('/Users/msaragoc/Projects/exoplanet_transit_classification/data/ephemeris_tables/tess/exofop_tois/exofop_tois_9-11-2025_processed_ephem_matching.csv')
    tce_tbl_fp = Path('/Users/msaragoc/Projects/exoplanet_transit_classification/data/ephemeris_tables/tess/tess_spoc_2min/tess-spoc-2min-tces-dv_s1-s94_s1s92_9-19-2025_1518_exofop-sg1-tois_9-22-2025_fixedtointps.csv')
    tce_tbl = pd.read_csv(tce_tbl_fp)
    print(tce_tbl['n_tois_in_tic'].value_counts())
    tce_tbl = update_num_tois_in_tic(tce_tbl, toi_tbl, toi_id_col='uid', target_col='target_id')
    print(tce_tbl['n_tois_in_tic'].value_counts())
    tce_tbl.to_csv(tce_tbl_fp, index=False)
    print('Done updating TCE table with number of TOIs in TICs.')
