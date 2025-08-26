"""
Utility functions for running the ExoMiner pipeline.
"""

# 3rd party
import yaml
from pathlib import Path
import multiprocessing as mp
import numpy as np
from astroquery.mast import Observations
from astropy.table import vstack
import pandas as pd
import re
import sys
import logging

# local
from src_preprocessing.tce_tables.preprocess_tess_tce_tbl import preprocess_tce_table
from src_preprocessing.tce_tables.extract_tce_data_from_dv_xml import process_sector_run_of_dv_xmls
from src.predict.predict_model import predict_model


# redirect stdout
class StreamToLogger:
    def __init__(self, logger, level=logging.INFO):
        self.logger = logger
        self.level = level
        self.buffer = ''

    def write(self, message):
        if message.strip() != '':
            self.logger.log(self.level, message.strip())

    def flush(self):
        pass


def validate_tic_ids_csv_structure(tics_df, logger):
    """
    Validates the structure of the TIC IDs CSV file.
    
    Args:
        tics_df: pandas.DataFrame with TIC IDs data
        logger: logging.Logger object
        
    Returns:
        bool: True if structure is valid, False otherwise
        
    Raises:
        SystemExit: If critical structural issues are found
    """

    required_columns = ['tic_id', 'sector_run']
    
    # Check if required columns exist
    missing_columns = [col for col in required_columns if col not in tics_df.columns]
    if missing_columns:
        logger.error(f'TIC IDs CSV file is missing required columns: {missing_columns}. '
                    f'Required columns are: {required_columns}')
        raise SystemExit(f'Invalid TIC IDs CSV structure. Missing columns: {missing_columns}')
    
    # Check for empty dataframe
    if len(tics_df) == 0:
        logger.error('TIC IDs CSV file is empty.')
        raise SystemExit('TIC IDs CSV file contains no data.')
    
    validation_errors = []
    
    # Validate TIC IDs - should be numeric and positive
    invalid_tic_ids = []
    for idx, tic_id in enumerate(tics_df['tic_id']):
        if not (isinstance(tic_id, int) and tic_id > 0):
            invalid_tic_ids.append(f"Row {idx + 1}: '{tic_id}'")
    
    if invalid_tic_ids:
        validation_errors.append(f"Invalid TIC IDs found (must be numeric): {', '.join(invalid_tic_ids[:5])}")
        if len(invalid_tic_ids) > 5:
            validation_errors.append(f"... and {len(invalid_tic_ids) - 5} more invalid TIC IDs")
    
    # Validate sector_run format - should be like "6-6", "1-39", etc.
    sector_run_pattern = re.compile(r'^[1-9]\d{0,2}-[1-9]\d{0,2}$')

    invalid_sector_runs = []
    for idx, sector_run in enumerate(tics_df['sector_run']):
        if not sector_run_pattern.match(str(sector_run)):
            invalid_sector_runs.append(f"Row {idx + 1}: '{sector_run}'")
    
    if invalid_sector_runs:
        validation_errors.append(f"Invalid sector_run format found (must be 'X-Y' format without leading zeros): "
                                 f"{', '.join(invalid_sector_runs[:5])}")
        if len(invalid_sector_runs) > 5:
            validation_errors.append(f"... and {len(invalid_sector_runs) - 5} more invalid sector_run entries")
    
    # Log validation results
    if validation_errors:
        logger.warning(f'TIC IDs CSV structure validation found {len(validation_errors)} issue(s):')
        for error in validation_errors:
            logger.warning(f'  - {error}')
        logger.warning('Please check your TIC IDs CSV file format. Expected format:')
        logger.warning('  - Column "tic_id": numeric TIC identifier (e.g., 167526485)')
        logger.warning('  - Column "sector_run": sector range in format "X-Y" (e.g., "6-6", "1-39")')
        return False
    else:
        logger.info(f'TIC IDs CSV structure validation passed. Found {len(tics_df)} valid entries.')
        return True
    
    
def check_command_line_arguments(config_fp, tic_ids_fp, tic_ids, data_collection_mode, num_processes, num_jobs, logger):
    """ Check command-line arguments.

    Args:
        config_fp: str, filepath to the configuration file for the run.
        tic_ids_fp: str, filepath to the TIC IDs file for the run.
        data_collection_mode: str, either '2min' or 'ffi'.
        tic_ids: str, list of TIC IDs to process. Only used if `tic_ids_fp` is None.
        num_processes: int, number of processes to use.
        num_jobs: int, number of jobs to split the TIC IDs through.
        logger: logging.Logger object.

    Returns:

    """

    # check if configuration file exists
    config_fp = Path(config_fp)
    if not config_fp.exists():
        logger.error(f'Configuration file for the run does not exist: {str(config_fp)}')
        raise FileNotFoundError(f'Configuration file for the run does not exist: {str(config_fp)}')

    # # check if data collection mode is valid
    # if data_collection_mode not in ['2min', 'ffi']:
    #     logger.info(f'Data collection mode "{data_collection_mode}" is not supported. Choose from "2min" '
    #                 f'or "ffi".')
    #     raise SystemExit("Invalid data collection mode. Choose from '2min' or 'ffi'.")

    # check if at least a list of TIC IDs or a CSV file with TIC IDs was provided
    if tic_ids_fp is None and tic_ids is None:  # overwrite filepath in configuration file
        logger.error("Must specify either --tic_ids_fp or --tic_ids.")
        raise SystemExit("Must specify either --tic_ids_fp or --tic_ids.")

    # check if number of processes is valid
    if not isinstance(num_processes, int):
        logger.error(f'Number of processes is not an integer: {num_processes}')
        raise SystemExit(f"Number of processes is not an integer: {num_processes}")

    # check if number of jobs is valid
    if not isinstance(num_jobs, int):
        logger.error(f'Number of jobs is not an integer: {num_jobs}')
        raise SystemExit(f"Number of processes is not an integer: {num_jobs}")


def check_config(run_config, logger):
    """ Check validity of parameters in the configuration file.

    Args:
        run_config: dict, dictionary containing run parameters.
        logger: logging.Logger object.

    Returns:

    """

    # check if required fields exist in configuration file for the run
    required_fields_in_config = [
        'output_dir',
        'model_fp',
        'data_collection_mode',
        'tic_ids_fp',
        'num_processes',
        'num_jobs',
        'download_spoc_data_products',
        'external_data_repository',
    ]
    for field in required_fields_in_config:
        if field not in run_config:
            logger.error(f'Configuration file does not contain required field "{field}".')
            raise FileNotFoundError(f'Configuration file does not contain required field "{field}".')

    # check if model exists
    if not Path(run_config['model_fp']).exists():
        logger.error(f'Model file was not found: {run_config["model_fp"]}')
        raise FileNotFoundError(f'ExoMiner model was not found in: {run_config["model_fp"]}')

    # check if TIC ID CSV file exists
    if not Path(run_config['tic_ids_fp']).exists():
        logger.error(f'TIC IDs file was not found: {run_config["tic_ids_fp"]}')
        raise FileNotFoundError(f'TIC IDs file was not found: {run_config["tic_ids_fp"]}')

    # check if data collection mode is valid
    if run_config['data_collection_mode'] not in  ['2min', 'ffi']:
        logger.info(f'Data collection mode "{run_config["data_collection_mode"]}" is not supported. Choose from "2min" '
                    f'or "ffi".')
        raise SystemExit("Invalid data collection mode. Choose from '2min' or 'ffi'.")

    if not (isinstance(run_config['num_processes'], int) and run_config['num_processes'] > 0):
        logger.error(f'Number of processes is not a positive integer: {run_config["num_processes"]}')
        raise SystemExit(f"Number of processes is not a positive integer: {run_config['num_processes']}")

    if not (isinstance(run_config['num_jobs'], int) and run_config['num_jobs'] > 0):
        logger.error(f'Number of jobs is not a positive integer: {run_config["num_jobs"]}')
        raise SystemExit(f"Number of jobs is not a positive integer: {run_config['num_jobs']}")

    # check if data collection mode is valid
    if run_config['download_spoc_data_products'] not in  ['true', 'false']:
        logger.info(f'Downloading SPOC DV mini-report variable "{run_config["download_spoc_data_products"]}" is not supported. '
                    f'Choose from "true" or "false".')
        raise SystemExit("Invalid SPOC DV mini-report flag. Choose from 'true' or 'false'.")

    if run_config['external_data_repository'] is not None:
        if not Path(run_config['external_data_repository']).exists():
            logger.info(f'External data repository does not exist: {run_config["external_data_repository"]}.')
            raise SystemExit(f'Invalid external data repository path: {run_config["external_data_repository"]}')


def process_inputs(output_dir, config_fp, tic_ids_fp, data_collection_mode, logger, tic_ids=None, num_processes=1,
                   num_jobs=1, download_spoc_data_products='false', external_data_repository=None,
                   stellar_parameters_source='ticv8', ruwe_source='gaiadr2'):
    """ Process input arguments to prepare them for the run.

    Args:
        output_dir: str, directory to save the output files.
        config_fp: str, filepath to the configuration file for the run.
        tic_ids_fp: str, filepath to the TIC IDs file for the run.
        data_collection_mode: str, either '2min' or 'ffi'.
        logger: logging.Logger object.
        tic_ids: str, list of TIC IDs to process. Only used if `tic_ids_fp` is None.
        num_processes: int, number of processes to use.
        num_jobs: int, number of jobs to split the TIC IDs through.
        download_spoc_data_products: str, whether to download a CSV file with URLs to the SPOC DV reports
        external_data_repository: str, whether to use external data repository.
        stellar_parameters_source: str, the stellar parameters source to use for the queried TICs. Set to either
            'ticv8', 'tess-spoc', or filepath to external catalog of stellar parameters for the queried TICs.
        ruwe_source: str, the RUWE source to use for the queried TICs. Set to either 'gaiadr2', 'unavailable', or
            filepath to external catalog of RUWE values for the queried TICs.

    Returns:
        run_config: dict with parameters for running the ExoMiner pipeline.
        tics_df: pandas dataframe with TIC IDs and corresponding sector run IDs (optional).

    """

    if not Path(config_fp).exists():
        logger.error(f'Configuration file for the run does not exist: {str(config_fp)}')
        raise FileNotFoundError(f'Configuration file for the run does not exist: {str(config_fp)}')

    with open(config_fp, 'r') as f:  # read configuration file
        run_config = yaml.safe_load(f)

    # overwrite configuration parameters with command-line counterparts
    if output_dir is not None:
        run_config['output_dir'] = output_dir

    if tic_ids_fp is not None:  # overwrite filepath in configuration file
        run_config['tic_ids_fp'] = tic_ids_fp
        tics_df = pd.read_csv(tic_ids_fp, skipinitialspace=True)

    elif tic_ids is not None:
        tics_dict = {field: [] for field in ['tic_id', 'sector_run']}
        tics = tic_ids.split(',')
        for tic in tics:
            if '_' in tic:  # sector run also provided
                tic_id, sector_run = tic.split('_')
                tics_dict['tic_id'].append(tic_id)
                tics_dict['sector_run'].append(sector_run)
            else:
                tics_dict['tic_id'].append(tic)
                tics_dict['sector_run'].append('')

        tics_df = pd.DataFrame.from_dict(tics_dict)

        tic_ids_fp = Path(output_dir / 'tics_tbl.csv')
        run_config['tic_ids_fp'] = str(tic_ids_fp)

    else:
        logger.error('Must specify either --tic_ids_fp or --tic_ids.')
        raise ValueError("Must specify either --tic_ids_fp or --tic_ids.")

    tics_df.to_csv(tic_ids_fp, index=False)

    # overwrite number of processes using the command-line argument
    if num_processes != -1:
        run_config['num_processes'] = num_processes
    # check maximum number of available cores
    num_cores = mp.cpu_count()
    logger.info(f'Found {num_cores} CPUs in this system. Number of cores requested: {run_config["num_processes"]}. '
                f'Adjusting if needed...')
    run_config['num_processes'] = min(num_cores, run_config['num_processes'])

    # overwrite number of jobs using the command-line argument
    if num_jobs != -1:
        run_config['num_jobs'] = num_jobs

    # overwrite data collection mode using the command-line argument
    if data_collection_mode is not None:
        run_config['data_collection_mode'] = data_collection_mode

    # overwrite download DV mini-report flag using the command-line argument
    run_config['download_spoc_data_products'] = download_spoc_data_products

    # overwrite data repository path using the command-line argument
    if external_data_repository is not None:
        run_config['external_data_repository'] = external_data_repository

    if stellar_parameters_source not in ['ticv8', 'tess-spoc']:
        if not Path(stellar_parameters_source).exists():
            logger.error(f'TIC stellar parameters catalog does not exist: {str(stellar_parameters_source)}. Either '
                         f'set --stellar_parameters_source to "ticv8", "tess-spoc", or provide a path to an external '
                         f'catalog of TIC stellar parameters.')
            raise FileNotFoundError(f'TIC stellar parameters catalog does not exist: {str(stellar_parameters_source)}. '
                                    f'Either '
                         f'set --stellar_parameters_source to "ticv8", "tess-spoc", or provide a path to an external '
                         f'catalog of TIC stellar parameters.')

        stellar_parameters_source = Path(stellar_parameters_source)

        # TODO: check structure of catalog

    run_config['stellar_parameters_source'] = stellar_parameters_source

    if ruwe_source not in ['gaiadr2', 'unavailable']:
        if not Path(ruwe_source).exists():
            logger.error(f'TIC RUWE catalog does not exist: {str(ruwe_source)}. Either set --ruwe_source '
                        f'to "gaiadr2" or "unavailable" or provide a path to an external catalog with '
                        f'RUWE values.')
            raise FileNotFoundError(f'TIC RUWE catalog does not exist: {str(ruwe_source)}. Either set --ruwe_source '
                                    f'to "gaiadr2" or "unavailable" or provide a path to an external catalog with '
                                    f'RUWE values.')

        ruwe_source = Path(ruwe_source)

        # TODO: check structure of catalog

    run_config['ruwe_source'] = ruwe_source

    # update parameters in auxiliary configuration files
    with open(run_config['lc_preprocessing_config_fp'], 'r') as f:
        lc_preprocessing_config = yaml.unsafe_load(f)
    if run_config['data_collection_mode'] == '2min':
        lc_preprocessing_config['ffi_data'] = False
    elif run_config['data_collection_mode'] == 'ffi':
        lc_preprocessing_config['ffi_data'] = True
    with open(run_config['lc_preprocessing_config_fp'], 'w') as f:
        yaml.dump(lc_preprocessing_config, f, sort_keys=False)

    return run_config, tics_df


def download_tess_spoc_data_products(tics_df, data_collection_mode, data_dir, logger):
    """ Download light curve FITS files and DV XML data for the set of TIC IDs and sector runs provided in `tics_df` for
    the specified `data_collection_mode` mode.

    Args:
        tics_df: pandas DataFrame containing TIC IDs "tic_id" and sector runs "sector_run" for which the data are to be
            downloaded. "sector_run" column should show the start and end sector for the run in the pattern
            {start_sector}-{end_sector}. "tic_id" should show the TIC ID as an integer.
        data_collection_mode: str, either "2min" or "ffi" indicating the data collection mode from which TESS SPOC data
            products were generated
        data_dir: Path, directory to save downloaded data
        logger: logger object

    Returns:

    """

    sys.stdout = StreamToLogger(logger)

    requested_products_lst, requested_products_manifest_lst = [], []
    for _, tic_data in tics_df.iterrows():

        logger.info(f'Downloading light curve and DV XML data for TIC {tic_data["tic_id"]} in sector run '
                    f'{tic_data["sector_run"]} ({data_collection_mode} data)...')

        # create sector array from sector run ID
        s_sector, e_sector = [int(sector_id) for sector_id in tic_data['sector_run'].split('-')]
        sector_arr = np.arange(s_sector, e_sector + 1)

        # create patterns for sectors and sector run ID to extract only products relevant to those
        lc_sectors_patterns = [f'-s{str(sector).zfill(4)}' for sector in sector_arr]
        sector_run_patern = f'-s{str(s_sector).zfill(4)}-s{str(e_sector).zfill(4)}'

        # get table with observations for TIC and corresponding data collection mode (either 2-min or FFI)
        obs_table = Observations.query_criteria(target_name=tic_data['tic_id'],
                                                obs_collection='TESS' if data_collection_mode == '2min' else 'HLSP',
                                                )
        if len(obs_table) == 0:
            logger.error(f'No observations found for TIC {tic_data["tic_id"]}. Skipping...')
            continue

        # get table with all available products for queried observations
        products = Observations.get_product_list(obs_table)

        if len(products) == 0:
            logger.error(f'No products found for TIC {tic_data["tic_id"]}. Skipping...')
            continue

        # filter for light curve FITS files (exclude also 20-sec light curves)
        lc_products = products[[fn.endswith('lc.fits') and 'fast-lc' not in fn for fn in products["productFilename"]]]
        # filter lc FITS files for sectors of interest
        lc_products = lc_products[
            [any(re.search(lc_sector_pattern, data_url) for lc_sector_pattern in lc_sectors_patterns) for data_url in
             lc_products['productFilename']]]
        if len(lc_products) == 0:
            logger.error(f'No TESS SPOC light curve files found for TIC {tic_data["tic_id"]} in {data_collection_mode} '
                         f'data. Skipping...')
            continue

        # filter for DV XML files
        dv_xml_products = products[[fn.endswith('dvr.xml') for fn in products["productFilename"]]]
        # filter DV XML files for sector run of interest
        dv_xml_products = dv_xml_products[
            [bool(re.search(sector_run_patern, data_url)) for data_url in dv_xml_products['productFilename']]]
        if len(dv_xml_products) == 0:
            logger.error(f'TESS SPOC DV XML file found was not found for TIC {tic_data["tic_id"]} in sector run '
                         f'{tic_data["sector_run"]} for {data_collection_mode} '
                         f'data. Skipping...')
            continue

        # check for cases in which more than on DV XML file is available due to multiple SPOC runs
        if len(dv_xml_products) > 1:
            dv_xml_versions = [int(fn.split('-')[4].split('_')[0]) for fn in dv_xml_products['productFilename']]
            max_version_number = max(dv_xml_versions)
            logger.info(f'Found more than one DV XML file for TIC {tic_data["tic_id"]} in sector run '
                        f'{tic_data["sector_run"]}. Versions found: {dv_xml_versions}. Considering only the most recent '
                        f'one: {max_version_number}')
            dv_xml_products['version_number'] = dv_xml_versions
            dv_xml_products = dv_xml_products[dv_xml_products['version_number'] == max_version_number]

        # combine tables for products to be downloaded
        requested_products = vstack([lc_products, dv_xml_products])
        requested_products_lst.append(requested_products)

        # download requested products
        requested_products_manifest = Observations.download_products(requested_products, download_dir=str(data_dir),
                                                                     mrp_only=False)
        requested_products_manifest_lst.append(requested_products_manifest)

        if not all(requested_products_manifest['Status']):
            logger.error(f'Could not download all requested products for TIC {tic_data["tic_id"]} in sector run '
                         f'{tic_data["sector_run"]} ({data_collection_mode} data. Skipping...)')
            continue

        logger.info(f'Finished downloading light curve and DV XML data for TIC {tic_data["tic_id"]} in sector run '
                    f'{tic_data["sector_run"]} ({data_collection_mode} data)...')

    if len(requested_products_lst) == 0:
        logger.error(f'No requested products found for queried TICs. Stopping job...')
        raise ValueError('No requested products found for queried TICs. Stopping job...')

    requested_products = vstack(requested_products_lst)
    requested_products.write(str(data_dir / f'requested_products_{data_collection_mode}.csv'),
                             format='csv', overwrite=True)
    requested_products_manifest = vstack(requested_products_manifest_lst)
    requested_products_manifest.write(
        str(data_dir / f'manifest_requested_products_{data_collection_mode}.csv'),
        format='csv', overwrite=True)

    # restore stdout
    sys.stdout = sys.__stdout__


def create_tce_table(res_dir: Path, job_id: int, dv_xml_products_dir: Path, logger: logging.Logger,
                     stellar_parameters_source, ruwe_source, filter_tics=None) \
        -> pd.DataFrame:
    """ Create TCE table using data from DV XML files.

    Args:
        res_dir: Path, results directory
        job_id: int, table ID
        dv_xml_products_dir: Path, directory containing DV XML files
        logger: logging.Logger
        filter_tics: list of TIC IDs with sector run ID used to filter DV XML files; if None, no filtering is done
        stellar_parameters_source: str, the stellar parameters source to use for the queried TICs. Set to either
            'ticv8', 'tess-spoc', or filepath to external catalog of stellar parameters for the queried TICs.
        ruwe_source: str, the RUWE source to use for the queried TICs. Set to either 'gaiadr2', 'unavailable', or
            filepath to external catalog of RUWE values for the queried TICs.

    Returns: tce_tbl, pandas DataFrame containing TCEs to be processed and that were extracted from the DV XML files

    """

    dv_xml_tbl_fp = res_dir / f'tess-spoc-dv_tces_{job_id}.csv'

    logs_dir = dv_xml_tbl_fp.parent / 'logs'
    logs_dir.mkdir(exist_ok=True)

    try:
        process_sector_run_of_dv_xmls(dv_xml_products_dir, dv_xml_tbl_fp, filter_tics)
    except Exception as e:
        raise ValueError(f'Error while extracting TCE(s) information from DV XMLs. Ensure that the DV XML files were '
                         f'correctly downloaded for the queried TIC IDs and that the TIC ID is correct. Error:\n {e}')


    sys.stdout = StreamToLogger(logger)
    tce_tbl = preprocess_tce_table(dv_xml_tbl_fp, res_dir, stellar_parameters_source, ruwe_source)
    sys.stdout = sys.__stdout__

    return tce_tbl


def inference_pipeline(run_config, output_dir, tfrec_dir, logger):
    """ Run inference pipeline.

    Args:
        run_config: dict, run configuration
        output_dir: Path, results directory
        tfrec_dir: Path, directory containing the TFRecord dataset
        logger: logging.Logger object

    Returns:

    """

    sys.stdout = StreamToLogger(logger)

    with open(run_config['predict_config_fp'], 'r') as file:
        predict_config = yaml.unsafe_load(file)

    tfrec_shards_fps = list(tfrec_dir.glob('shard-*'))

    predict_config['datasets_fps'] = {
        'predict' : tfrec_shards_fps
    }

    predict_model(predict_config, run_config['model_fp'], output_dir, logger)

    # restore stdout
    sys.stdout = sys.__stdout__


def create_tic_id_pattern(row, data_collection_mode):

    tic_id = str(row['tic_id']).zfill(16)
    start_sector, end_sector = row['sector_run'].split("-")
    sector_id = f"s{start_sector.zfill(4)}-s{end_sector.zfill(4)}"

    if data_collection_mode == 'ffi':
        tic_id_pattern = f'{tic_id}-{sector_id}'
    elif data_collection_mode == '2min':
        tic_id_pattern = f'{sector_id}-{tic_id}'
    else:
        raise ValueError(f'Data collection mode must be either "ffi" or "2min": {data_collection_mode}')

    return tic_id_pattern