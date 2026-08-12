"""
Module to download TESS SPOC data products (light curve FITS files and DV XML files) for a set of TIC IDs and sector runs.

Input structure:
- Input table with columns "tic_id" and "sector_run" (with sector run ID in the pattern {start_sector}-{end_sector}) for which to download data

Struture of downloaded data:
- For 2-min data, all target files for a given sector/sector run are downloaded in the same subdirectory
    - light curves will be downloaded in subdirectories for each sector run in the pattern "sector_{sector_id}". For example, for sector run 1-3, light curves will be downloaded in the subdirectories "sector_1", "sector_2", "sector_3".
    - DV XML files will be downloaded in subdirectories for each sector run in the pattern "single-sector/sector_{sector_id}" for single-sector data or "multi-sector/multisector_s{start_sector_id 4-digit}-s{end_sector_id 4-digit}" for multisector data. 
    For example, for sector run 1-3, DV XML files will be downloaded in the subdirectory "multi-sector/multisector_s0001-s0003".

- For FFI data, the data for a given sector/sector run follows an additional subdirectories hierarchy for each in the pattern "s{sector_id 4-digit}/target/{tic_id first-4-digits}/{tic_id second-4-digits}/{tic_id third-4-digits}/{tic_id fourth-4-digits}". 
    - light curves will be downloaded in subdirectories for each sector run in the pattern "s{sector_id 4-digit}". For example, for sector run 1-3, light curves will be downloaded in the subdirectories "s0001", "s0002", "s0003".
    - DV XML files will be downloaded in subdirectories for each sector run in the pattern "s{sector_id 4-digit}" for single-sector data or "s{start_sector_id 4-digit}-s{end_sector_id 4-digit}" for multisector data. 
"""

# imports
import re
import numpy as np
import pandas as pd
from pathlib import Path
from astropy.table import Table
from astropy.table import vstack
from astroquery.mast import Observations
import logging
from tqdm import tqdm


def get_products_table_for_tic(tic_id: int, data_collection_mode: str, logger: logging.Logger|None=None) -> Table | None:
    """ Get table with available products for a given TIC ID and data collection mode (either 2-min or FFI).
    
    Args:
        tic_id: int, TIC ID for which to query available products
        data_collection_mode: str, either "2min" or "ffi" indicating the data collection mode from which TESS SPOC data products were generated
    Returns: products, Table with available products for the TIC ID and data collection mode of interest
    """
    
    # get table with observations for TIC and corresponding data collection mode (either 2-min or FFI)
    obs_table = Observations.query_criteria(target_name=tic_id,
                                            obs_collection='TESS' if data_collection_mode == '2min' else 'HLSP',
                                            )
    if len(obs_table) == 0:
        if logger is None:
            print(f'No observations found for TIC {tic_id}. Skipping...')
        else:
            logger.error(f'No observations found for TIC {tic_id}. Skipping...')    
        return None

    # get table with all available products for queried observations
    products = Observations.get_product_list(obs_table)

    if len(products) == 0:
        if logger is None:
            print(f'No products found for TIC {tic_id}. Skipping...')
        else:
            logger.error(f'No products found for TIC {tic_id}. Skipping...')
        return None

    return products


def create_download_dir_lcs(data_collection_mode: str, sector: int, tic_id: int, data_dir: Path) -> Path:
    """ Create directory to download light curve FITS files for a given TIC ID, sector, and data collection mode.
    
    Args:
        data_collection_mode: str, either "2min" or "ffi" indicating the data collection mode from which TESS SPOC data products were generated
        sector: int, sector for which to create download directory
        tic_id: int, TIC ID for which to create download directory
        data_dir: Path, root directory to save downloaded data
    
    Returns: download_dir, Path to directory to download light curve FITS files for the given TIC ID, sector, and data collection mode
    """
    if data_collection_mode == '2min':
        download_dir = data_dir / f'sector_{sector}'
    else:
        tic_id_str = str(tic_id).zfill(16)
        download_dir = data_dir / f's{str(sector).zfill(4)}' / 'target' /tic_id_str[0:4] / tic_id_str[4:8] / tic_id_str[8:12] / tic_id_str[12:16]

    return download_dir

def download_tess_lightcurves(tic_id: int, products: Table, sector_arr: np.ndarray, data_collection_mode: str, data_dir: Path, logger: logging.Logger|None=None) -> Table|None:
    """ Download light curve FITS files for a TIC ID and sectors provided in `tic_data` for the specified `data_collection_mode` mode.

    Args:
        tic_id: integer, TIC ID for which to download data
        products: Table with available products for the TIC ID and data collection mode of interest, as obtained from `get_products_table_for_tic` function.
        sector_arr: np.ndarray, array with sectors for which to download data
        data_collection_mode: str, either "2min" or "ffi" indicating the data collection mode from which TESS SPOC data
            products were generated
        data_dir: Path, root directory to save downloaded data
        logger: logger object
    
    Returns: requested_products_manifest, Table with the manifest of downloaded products
    """

    if data_collection_mode not in ['2min', 'ffi']:
        raise ValueError(f'Invalid data collection mode: {data_collection_mode}. Must be either "2min" or "ffi".')
    
    # create patterns for sectors and sector run ID to extract only products relevant to those
    lc_sectors_patterns = [f'-s{str(sector).zfill(4)}' for sector in sector_arr]

    # filter for light curve FITS files (exclude also 20-sec light curves)
    lc_products = products[[fn.endswith('lc.fits') and 'fast-lc' not in fn for fn in products["productFilename"]]]

    if len(lc_products) == 0:
        if logger is None:
            print(f'No TESS SPOC light curve files found for TIC {tic_id} for sector(s) {sector_arr} in {data_collection_mode} data. Skipping...')
        else:
            logger.info(f'No TESS SPOC light curve files found for TIC {tic_id} for sector(s) {sector_arr} in {data_collection_mode} data. Skipping...')
        return None

    requested_products_manifest_sector_lst = []
    for sector_i, sector in enumerate(sector_arr):
    
        download_dir = create_download_dir_lcs(data_collection_mode, sector, tic_id, data_dir)
        download_dir.mkdir(parents=True, exist_ok=True)

        # filter lc FITS files for sectors of interest
        lc_products_sector = lc_products[[bool(re.search(lc_sectors_patterns[sector_i], data_url)) for data_url in lc_products['productFilename']]]
        # it is expected that for most TICs in many sectors in the sector run there will be no light curve files
        if len(lc_products_sector) == 0:  
            # if logger is None:
            #     print(f'No TESS SPOC light curve files found for TIC {tic_id} in sector {sector} for {data_collection_mode} data. Skipping...')
            # else:
            #     logger.info(f'No TESS SPOC light curve files found for TIC {tic_id} in sector {sector} for {data_collection_mode} data. Skipping...')
            continue

        requested_products_manifest = Observations.download_products(lc_products_sector, download_dir=str(download_dir), mrp_only=False)
        if requested_products_manifest is not None:
            requested_products_manifest_sector_lst.append(requested_products_manifest)
    
    if len(requested_products_manifest_sector_lst) == 0:
        return None
    else:
        requested_products_manifest = vstack(requested_products_manifest_sector_lst)

    if not all([status in ['COMPLETE', 'SKIPPED'] for status in requested_products_manifest['Status']]):
        if logger is None:
            print(f'Could not download all requested products for TIC {tic_id} in sector(s) {sector_arr} '
                  f'({data_collection_mode} data. Skipping...)')
        else:
            logger.error(f'Could not download all requested products for TIC {tic_id} in sector(s) {sector_arr} '
                         f'({data_collection_mode} data. Skipping...)')
        return None

    return requested_products_manifest


def create_download_dir_dv_xmls(data_collection_mode: str, sector_run: list, tic_id: int, data_dir: Path) -> Path:
    """ Create directory to download DV XML files for a given TIC ID, sector run, and data collection mode.
    
    Args:
        data_collection_mode: str, either "2min" or "ffi" indicating the data collection mode from which TESS SPOC data products were generated
        sector_run: list, with start and end sectors for the run in the pattern {start_sector}-{end_sector}
        tic_id: int, TIC ID for which to create download directory
        data_dir: Path, root directory to save downloaded data
    
    Returns: download_dir, Path to directory to download DV XML files for the given TIC ID, sector run, and data collection mode
    """

    if data_collection_mode == '2min':
        if sector_run[0] == sector_run[-1]:
            download_dir = data_dir / f'single-sector/sector_{sector_run[0]}'
        else:
            download_dir = data_dir / f'multi-sector/multisector_s{str(sector_run[0]).zfill(4)}-s{str(sector_run[-1]).zfill(4)}'
    
    else:
        sector_str = f's{str(sector_run[0]).zfill(4)}' if sector_run[0] == sector_run[-1] else f's{str(sector_run[0]).zfill(4)}-s{str(sector_run[-1]).zfill(4)}'
        tic_id_str = str(tic_id).zfill(16)
        download_dir = data_dir / f'{sector_str}' / 'target' / tic_id_str[0:4] / tic_id_str[4:8] / tic_id_str[8:12] / tic_id_str[12:16]

    return download_dir


def download_tess_dv_xmls(tic_id: int, products: Table, sector_run: list, data_collection_mode: str, data_dir: Path, logger: logging.Logger|None=None) -> Table|None:
    """ Download DV XML files for a TIC ID and sector run provided in `tic_data` for the specified `data_collection_mode` mode.
    
    Args:
        tic_id: integer, TIC ID for which to download data
        products: pandas DataFrame with available products for the TIC ID and data collection mode of interest, as obtained from `get_products_table_for_tic` function.
        sector_run: list, with start and end sectors for the run in the pattern {start_sector}-{end_sector}
        data_collection_mode: str, either "2min" or "ffi" indicating the data collection mode from which TESS SPOC data products were generated
        data_dir: Path, directory to save downloaded data
        logger: logger object
    
    Returns: requested_products_manifest, Table with the manifest of downloaded products
    """
    
    if data_collection_mode not in ['2min', 'ffi']:
        raise ValueError(f'Invalid data collection mode: {data_collection_mode}. Must be either "2min" or "ffi".')
    
    download_dir = create_download_dir_dv_xmls(data_collection_mode, sector_run, tic_id, data_dir)
    download_dir.mkdir(parents=True, exist_ok=True)

    sector_run_pattern = f'-s{str(sector_run[0]).zfill(4)}-s{str(sector_run[-1]).zfill(4)}'

    # filter for DV XML files
    dv_xml_products = products[[fn.endswith('dvr.xml') for fn in products["productFilename"]]]
    
    # filter DV XML files for sector run of interest
    dv_xml_products = dv_xml_products[[bool(re.search(sector_run_pattern, data_url)) for data_url in dv_xml_products['productFilename']]]
    
    if len(dv_xml_products) == 0:
        if logger is None:
            print(f'TESS SPOC DV XML file found was not found for TIC {tic_id} in sector run '
                        f'{sector_run} for {data_collection_mode} '
                        f'data. Skipping...')
        return None

    # check for cases in which more than on DV XML file is available due to multiple SPOC runs
    if len(dv_xml_products) > 1:
        dv_xml_versions = [int(fn.split('-')[4].split('_')[0]) for fn in dv_xml_products['productFilename']]
        max_version_number = max(dv_xml_versions)
        if logger is None:
            print(f'Found more than one DV XML file for TIC {tic_id} in sector run '
                        f'{sector_run}. Versions found: {dv_xml_versions}. Considering only the most recent '
                        f'one: {max_version_number}')
        else:
            logger.info(f'Found more than one DV XML file for TIC {tic_id} in sector run '
                        f'{sector_run}. Versions found: {dv_xml_versions}. Considering only the most recent '
                        f'one: {max_version_number}')
        dv_xml_products['version_number'] = dv_xml_versions
        dv_xml_products = dv_xml_products[dv_xml_products['version_number'] == max_version_number]

    requested_products_manifest = Observations.download_products(dv_xml_products, download_dir=str(download_dir), mrp_only=False)

    if requested_products_manifest is not None:
        if not all([status in ['COMPLETE', 'SKIPPED'] for status in requested_products_manifest['Status']]):
            if logger is None:
                print(f'Could not download all requested products for TIC {tic_id} in sector run '
                        f'{sector_run} ({data_collection_mode} data. Skipping...)')
            else:
                logger.error(f'Could not download all requested products for TIC {tic_id} in sector run '
                            f'{sector_run} ({data_collection_mode} data. Skipping...)')
            return None
    
    return requested_products_manifest


def check_format_input_table(df: pd.DataFrame, required_columns: list, column_types: dict):
    """ Check that a pandas DataFrame has the required columns and that the columns have the correct types.

    Args:
        df: pandas DataFrame to check
        required_columns: list of required column names
        column_types: dict with column names as keys and expected types as values
    Raises:
        ValueError if any of the required columns are missing or if any of the columns have incorrect types
    """
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f'Missing required column: {col}')
    
    for col, expected_type in column_types.items():
        if col in df.columns:
            if expected_type == str:
                # Pandas stores strings as 'object' dtype by default
                if not (pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col])):
                    raise ValueError(f'Column {col} has incorrect type. Expected string, got {df[col].dtype}')
            elif not pd.api.types.is_dtype_equal(df[col].dtype, expected_type):
                raise ValueError(f'Column {col} has incorrect type. Expected {expected_type}, got {df[col].dtype}')
    
    # check that sector_run column has the correct format
    if 'sector_run' in df.columns:
        if not all(df['sector_run'].apply(lambda x: bool(re.match(r'^\d+-\d+$', str(x))))):
            raise ValueError('Column sector_run has incorrect format. Expected format: {start_sector}-{end_sector} with start_sector and end_sector as integers.')
        
def download_tess_data_products(tics_df: pd.DataFrame, data_collection_mode: str, 
                                get_lightcurves: bool, get_dv_xmls: bool, lc_root_data_dir: Path, xml_root_data_dir: Path, 
                                logger: logging.Logger|None=None) -> tuple[Table, Table]|tuple[None, None]:
    """ Download light curve FITS files afor the set of TIC IDs and sector runs provided in `tics_df` for
    the specified `data_collection_mode` mode.

    Args:
        tics_df: pandas DataFrame containing TIC IDs "tic_id" and sector runs "sector_run" for which the data are to be
            downloaded. "sector_run" column should show the start and end sector for the run in the pattern
            {start_sector}-{end_sector}. "tic_id" should show the TIC ID as an integer.
        data_collection_mode: str, either "2min" or "ffi" indicating the data collection mode from which TESS SPOC data
            products were generated
        get_lightcurves: bool, whether to download light curve FITS files
        get_dv_xmls: bool, whether to download DV XML files
        lc_root_data_dir: Path, root directory to save downloaded light curve FITS files; light curve files will be saved in subdirectories
        xml_root_data_dir: Path, root directory to save downloaded DV XML files; DV XML files will be saved in subdirectories
        logger: logger object

    Returns:
        requested_products, Table with the products that were requested to be downloaded (including their metadata)
        requested_products_manifest, Table with the manifest of downloaded products
    """

    check_format_input_table(tics_df, required_columns=['tic_id', 'sector_run'], column_types={'tic_id': int, 'sector_run': str})

    if get_dv_xmls and xml_root_data_dir is None:
        raise ValueError('xml_root_data_dir must be provided if get_dv_xmls is True')
    if get_lightcurves and lc_root_data_dir is None:
        raise ValueError('lc_root_data_dir must be provided if get_lightcurves is True')
    
    requested_products_lst, requested_products_manifest_lst = [], []
    for _, tic_data in tqdm(tics_df.iterrows(), total=len(tics_df), desc='Downloading TESS SPOC data products for TICs', unit='TIC'):

        if logger is None:
            print(f'Downloading light curve and DV XML data for TIC {tic_data["tic_id"]} in sector run '
                  f'{tic_data["sector_run"]} ({data_collection_mode} data)...')
        else:
            logger.info(f'Downloading light curve and DV XML data for TIC {tic_data["tic_id"]} in sector run '
                        f'{tic_data["sector_run"]} ({data_collection_mode} data)...')

        # create sector array from sector run ID
        s_sector, e_sector = [int(sector_id) for sector_id in tic_data['sector_run'].split('-')]
        sector_arr = np.arange(s_sector, e_sector + 1)

        products = get_products_table_for_tic(tic_data['tic_id'], data_collection_mode, logger)
        if products is None:
            if logger is None:
                print(f'No products found for TIC {tic_data["tic_id"]} for {data_collection_mode} data. Skipping...')
            else:
                logger.error(f'No products found for TIC {tic_data["tic_id"]} for {data_collection_mode} data. Skipping...')

            continue

        requested_products_tic = []
        if get_lightcurves:
            lc_products_manifest = download_tess_lightcurves(tic_data['tic_id'], products, sector_arr, data_collection_mode, lc_root_data_dir, logger)
            if lc_products_manifest is not None:
                requested_products_tic.append(lc_products_manifest)
        
        if get_dv_xmls:
            dv_xml_products_manifest = download_tess_dv_xmls(tic_data['tic_id'], products, [s_sector, e_sector], data_collection_mode, xml_root_data_dir, logger)
            if dv_xml_products_manifest is not None:
                requested_products_tic.append(dv_xml_products_manifest)

        if len(requested_products_tic) == 0:
            if logger is None:
                print(f'No requested products found for TIC {tic_data["tic_id"]} in sector run '
                            f'{tic_data["sector_run"]} ({data_collection_mode} data). Skipping...')
            else:
                logger.error(f'No requested products found for TIC {tic_data["tic_id"]} in sector run '
                            f'{tic_data["sector_run"]} ({data_collection_mode} data). Skipping...')
            continue

        requested_products = vstack(requested_products_tic)
        requested_products_lst.append(requested_products)

        # if not all(requested_products_manifest['Status']):
        #     if logger is None:
        #         print(f'Could not download all requested products for TIC {tic_data["tic_id"]} in sector run '
        #                     f'{tic_data["sector_run"]} ({data_collection_mode} data. Skipping...)')
        #     else:
        #         logger.error(f'Could not download all requested products for TIC {tic_data["tic_id"]} in sector run '
        #                         f'{tic_data["sector_run"]} ({data_collection_mode} data. Skipping...)')
        #     return None

        if logger is None:
            print(f'Finished downloading light curve and DV XML data for TIC {tic_data["tic_id"]} in sector run '
                        f'{tic_data["sector_run"]} ({data_collection_mode} data)...')
        else:
            logger.info(f'Finished downloading light curve and DV XML data for TIC {tic_data["tic_id"]} in sector run '
                        f'{tic_data["sector_run"]} ({data_collection_mode} data)...')

    if len(requested_products_lst) > 0:
        requested_products = vstack(requested_products_lst)
        # requested_products.write(str(data_dir / f'requested_products_{data_collection_mode}.csv'), format='csv', overwrite=True)
    else:
        requested_products = None
    
    if len(requested_products_manifest_lst) > 0:
        requested_products_manifest = vstack(requested_products_manifest_lst)
    else:
        requested_products_manifest = None

    # requested_products_manifest.write(
    #     str(data_dir / f'manifest_requested_products_{data_collection_mode}.csv'),
        # format='csv', overwrite=True)

    return requested_products, requested_products_manifest


if __name__ == '__main__':
    
    # example usage
    # tics_df = pd.DataFrame({'tic_id': [123456789, 987654321], 'sector_run': ['1-3', '2-4']})
    tics_df = pd.read_csv('/u/msaragoc/work_dir/Kepler-TESS_exoplanet/codebase_aux_loss_source_offset/data_wrangling/prototyping/test_targets_to_download_lcs/test_targets_to_download.csv')
    data_collection_mode = '2min'

    get_lcs = True
    # lc_root_data_dir = Path('/u/msaragoc/work_dir/Kepler-TESS_exoplanet/data/FITS_files/TESS/spoc_2min/lc/sectors')
    lc_root_data_dir = Path('/u/msaragoc/work_dir/Kepler-TESS_exoplanet/codebase_aux_loss_source_offset/data_wrangling/prototyping/test_targets_to_download_lcs/lcs')
    lc_root_data_dir.mkdir(parents=True, exist_ok=True)

    get_xmls = True
    xml_root_data_dir = Path('/u/msaragoc/work_dir/Kepler-TESS_exoplanet/codebase_aux_loss_source_offset/data_wrangling/prototyping/test_targets_to_download_lcs/xmls')
    xml_root_data_dir.mkdir(parents=True, exist_ok=True)

    logger = None
    if logger is not None:
        logger.setLevel(logging.INFO)
        logger.addHandler(logging.StreamHandler())

    requested_products, requested_products_manifest = download_tess_data_products(tics_df, data_collection_mode, get_lcs, get_xmls, lc_root_data_dir, xml_root_data_dir, logger)

