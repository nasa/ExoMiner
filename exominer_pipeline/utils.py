"""
Utility functions for running the ExoMiner pipeline.
"""

# 3rd party
import yaml
import concurrent.futures
from pathlib import Path
import multiprocessing as mp
import numpy as np
from astroquery.mast import Observations
from astropy.table import vstack
import pandas as pd
import re
import sys
import logging
import subprocess
from PIL import Image
import time
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model
import tensorflow as tf
from tqdm import tqdm
import os
import shutil
import tempfile
import psutil

# local
from src_preprocessing.tce_tables.preprocess_tess_tce_tbl import preprocess_tce_table
from src_preprocessing.tce_tables.extract_tce_data_from_dv_xml import process_sector_run_of_dv_xmls
from models.models_keras import Time2Vec, SplitLayer
from src.postprocessing.compute_dispositions_multiclass import map_softmax_predictions_to_class
from src.utils.utils_dataio import InputFnv2 as InputFn, set_tf_data_type_for_features, get_data_from_tfrecords_for_predictions_table
import models.custom_layers
from src.utils.utils import log_info

Observations.enable_cloud_dataset()

# redirect stdout
class StreamToLogger:
    def __init__(self, logger, level=logging.INFO):
        """Initialize the stream to logger redirector.
        
        Args:
            logger: logging.Logger object.
            level: int, logging level (default: logging.INFO).
        """
        self.logger = logger
        self.level = level
        self.buffer = ''

    def write(self, message):
        """Write a message to the logger if it is not empty.
        
        Args:
            message: str, the message string to log.
        """
        if message.strip() != '':
            self.logger.log(self.level, message.strip())

    def flush(self):
        """Flush the stream. Required for file-like objects, but does nothing here."""
        pass


def validate_tic_ids_csv_structure(tics_df, logger):
    """Validates the structure of the TIC IDs CSV file.
    
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


def check_cli_args(config_fp, tic_ids_fp, data_collection_mode, tic_ids, num_processes, num_jobs, logger):
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
        'get_mast_urls_dv_reports',
        'dv_xml_data_repository',
        'lc_data_repository',
        'task',
        'exominer_models',
        'max_model_workers',
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

    # check if task is valid
    if run_config['task'] not in  ['phot-vetting', 'planet-validation']:
        logger.info(f'Task "{run_config["task"]}" is not supported. Choose from "phot-vetting" '
                    f'or "planet-validation".')
        raise SystemExit("Invalid task. Choose from 'phot-vetting' or 'planet-validation'.")

    if not (isinstance(run_config['num_processes'], int) and run_config['num_processes'] > 0):
        logger.error(f'Number of processes is not a positive integer: {run_config["num_processes"]}')
        raise SystemExit(f"Number of processes is not a positive integer: {run_config['num_processes']}")

    if not (isinstance(run_config['num_jobs'], int) and run_config['num_jobs'] > 0):
        logger.error(f'Number of jobs is not a positive integer: {run_config["num_jobs"]}')
        raise SystemExit(f"Number of jobs is not a positive integer: {run_config['num_jobs']}")

    # check if data collection mode is valid
    if run_config['get_mast_urls_dv_reports'] not in  ['true', 'false']:
        logger.info(f'Get MAST URLs for SPOC DV reports variable "{run_config["get_mast_urls_dv_reports"]}" is not supported. '
                    f'Choose from "true" or "false".')
        raise SystemExit("Invalid get MAST URLs for SPOC DV reports flag. Choose from 'true' or 'false'.")

    if run_config['dv_xml_data_repository'] is not None:
        if not Path(run_config['dv_xml_data_repository']).exists():
            logger.info(f'DV XML data repository does not exist: {run_config["dv_xml_data_repository"]}.')
            raise SystemExit(f'Invalid DV XML data repository path: {run_config["dv_xml_data_repository"]}')
        
    if run_config['lc_data_repository'] is not None:
        if not Path(run_config['lc_data_repository']).exists():
            logger.info(f'Light curve data repository does not exist: {run_config["lc_data_repository"]}.')
            raise SystemExit(f'Invalid light curve data repository path: {run_config["lc_data_repository"]}')
    
    if not (isinstance(run_config['max_model_workers'], int) and run_config['max_model_workers'] > 0):
        logger.info(f'Maximum number of workers set for parallel inference is invalid: {run_config["max_model_workers"]}.')
        raise SystemExit(f'Maximum number of workers set for parallel inference is invalid: {run_config["max_model_workers"]}')


def check_ruwe_source(ruwe_source, tics_df, logger):
    """Check the validity of the RUWE source provided.

    :param Path ruwe_source: filepath to the RUWE catalog
    :param pandas DataFrame tics_df: input TIC IDs dataframe
    :param Python Logger logger: logger object
    :raises SystemExit: if the ruwe source is invalid because 1) some TIC IDs are missing from the catalog,  
        2) some required columns are missing from the catalog, or 3) wrong data types in some columns
    """
    
    logger.info(f'Checking RUWE source: {str(ruwe_source)}')
    
    # read stellar parameters catalog
    ruwe_df = pd.read_csv(ruwe_source, skipinitialspace=True)
    
    # check if required columns exist
    required_columns = [
        'ruwe',
        'target_id',
        ]
    for col in required_columns:
        if col not in ruwe_df.columns:
            logger.error(f'RUWE catalog is missing required column: {col}.')
            raise SystemExit(f'RUWE catalog is missing required column: {col}.')
    
    # check data types of columns
    if ruwe_df['target_id'].dtype != 'int64':
        logger.error(f'RUWE catalog column "target_id" must be of type int.')
        raise SystemExit(f'RUWE catalog column "target_id" must be of type int.')

    # check if all TIC IDs in tics_df are in ruwe_df
    n_missing_tics = (~tics_df['tic_id'].isin(ruwe_df['target_id'])).sum()
    if n_missing_tics > 0:
        logger.error(f'RUWE catalog is missing {n_missing_tics}/{len(tics_df)} TIC IDs provided for the run.')
        raise SystemExit(f'RUWE catalog is missing {n_missing_tics}/{len(tics_df)} TIC IDs provided for the run.')
    
    
def check_stellar_parameters_source(stellar_parameters_source, tics_df, logger):
    """Check the validity of the stellar parameters source provided.

    :param Path stellar_parameters_source: filepath to the stellar parameters catalog
    :param pandas DataFrame tics_df: input TIC IDs dataframe
    :param Python Logger logger: logger object
    :raises SystemExit: if the stellar parameters source is invalid because 1) some TIC IDs are missing from the catalog,  
        2) some required columns are missing from the catalog, or 3) wrong data types in some columns
    """
    
    logger.info(f'Checking stellar parameters source: {str(stellar_parameters_source)}')
    
    # read stellar parameters catalog
    stellar_params_df = pd.read_csv(stellar_parameters_source, skipinitialspace=True)
    
    # check if required columns exist
    required_columns = [
        'tic_steff',
        'tic_steff_err',
        'tic_smass',
        'tic_smass_err',
        'tic_smet',
        'tic_smet_err',
        'tic_sradius',
        'tic_sradius_err',
        'tic_sdens',
        'tic_sdens_err',
        'tic_slogg',
        'tic_slogg_err',
        'tic_ra',
        'tic_dec',
        'kic_id',
        'gaia_id',
        'tic_tmag',
        'tic_tmag_err',
        'target_id'
        ]
    for col in required_columns:
        if col not in stellar_params_df.columns:
            logger.error(f'Stellar parameters catalog is missing required column: {col}.')
            raise SystemExit(f'Stellar parameters catalog is missing required column: {col}.')
    
    # check data types of columns
    if stellar_params_df['target_id'].dtype != 'int64':
        logger.error(f'Stellar parameters catalog column "target_id" must be of type int.')
        raise SystemExit(f'Stellar parameters catalog column "target_id" must be of type int.')
    
    # check if all TIC IDs in tics_df are in stellar_params_df
    n_missing_tics = (~tics_df['tic_id'].isin(stellar_params_df['target_id'])).sum()
    if n_missing_tics > 0:
        logger.error(f'Stellar parameters catalog is missing {n_missing_tics}/{len(tics_df)} TIC IDs provided for the run.')
        raise SystemExit(f'Stellar parameters catalog is missing {n_missing_tics}/{len(tics_df)} TIC IDs provided for the run.')
    
    
def process_inputs(output_dir, config_fp, tic_ids_fp, data_collection_mode, logger, tic_ids=None, num_processes=1,
                   num_jobs=1, get_mast_urls_dv_reports='false', dv_xml_data_repository=None, lc_data_repository=None,
                   stellar_parameters_source='ticv8', ruwe_source='gaiadr2', task='phot-vetting', exominer_model='single',
                   max_model_workers=1):
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
        get_mast_urls_dv_reports: str, whether to download a CSV file with URLs to the SPOC DV reports
        dv_xml_data_repository: str, the data repository to use for DV XML files for queried TICs.
        lc_data_repository: str, the data repository to use for light curve FITS files for queried TICs.        
        stellar_parameters_source: str, the stellar parameters source to use for the queried TICs. Set to either
            'ticv8', 'tess-spoc', or filepath to external catalog of stellar parameters for the queried TICs.
        ruwe_source: str, the RUWE source to use for the queried TICs. Set to either 'gaiadr2', 'unavailable', or
            filepath to external catalog of RUWE values for the queried TICs.
        task: str, either 'phot-vetting' or 'planet-validation'
        exominer_model: str, which ExoMiner model to use for inference. Choose among "single", "cv_ensemble", and "full_cv_ensemble", 
            or provide the filepath to a TensorFlow Keras model that is compatible with the pipeline
        max_model_workers: int, max number of processes used to run inference in parallel

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
    run_config['data_collection_mode'] = data_collection_mode

    # overwrite download DV mini-report flag using the command-line argument
    run_config['get_mast_urls_dv_reports'] = get_mast_urls_dv_reports

    # overwrite data repository path using the command-line argument
    run_config['dv_xml_data_repository'] = dv_xml_data_repository
    run_config['lc_data_repository'] = lc_data_repository

    run_config['max_model_workers'] = max_model_workers

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

        check_stellar_parameters_source(stellar_parameters_source, tics_df, logger)

    run_config['stellar_parameters_source'] = stellar_parameters_source

    if ruwe_source not in ['gaiadr2', 'gaiaedr3', 'gaiadr3', 'unavailable']:
        if not Path(ruwe_source).exists():
            logger.error(f'TIC RUWE catalog does not exist: {str(ruwe_source)}. Either set --ruwe_source '
                        f'to "gaiadr2", "gaiaedr3", "gaiadr3", or "unavailable" or provide a path to an external catalog with '
                        f'RUWE values.')
            raise FileNotFoundError(f'TIC RUWE catalog does not exist: {str(ruwe_source)}. Either set --ruwe_source '
                                    f'to "gaiadr2", "gaiaedr3", "gaiadr3", or "unavailable" or provide a path to an external catalog with '
                                    f'RUWE values.')

        ruwe_source = Path(ruwe_source)

        check_ruwe_source(ruwe_source, tics_df, logger)

    run_config['ruwe_source'] = ruwe_source
    run_config['task'] = task
    run_config['exominer_model_name'] = exominer_model

    if task not in run_config['exominer_models']:
        logger.error(f'Task "{task}" is not supported. Choose from '
                         f'{list(run_config["exominer_models"].keys())}.')
        raise ValueError(f'Task "{task}" is not supported. Choose from '
                            f'{list(run_config["exominer_models"].keys())}.')

    # set model filepath to the selected ExoMiner model
    if Path(exominer_model).is_file():  #  not in run_config['exominer_models'][task]:
        
        logger.info(f'Provided external model. Checking if it exists.')
        
        # check if model is a valid filepath
        if not Path(exominer_model).exists():
            logger.error(f'ExoMiner model "{exominer_model}" is not supported. Choose from '
                         f'{list(run_config["exominer_models"][task].keys())} or provide the filepath to a TensorFlow Keras '
                         f'model that is compatible with the pipeline.')
            raise ValueError(f'ExoMiner model "{exominer_model}" is not supported. Choose from '
                             f'{list(run_config["exominer_models"][task].keys())} or provide the filepath to a TensorFlow Keras '
                             f'model that is compatible with the pipeline.')
        
        # # check if model is valid
        # check_custom_model(exominer_model)
        
        run_config['model_fp'] = exominer_model  # assume it's a filepath
    else:
        run_config['model_fp'] = run_config['exominer_models'][task]
    
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


def retry_mast_call(func, logger, max_retries=3, delay=5, *args, **kwargs):
    """
    Executes a function and retries it upon failure.
    """
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(f"MAST API call failed: {e}. Retrying in {delay} seconds... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
                delay *= 2  # Exponential backoff
            else:
                logger.error(f"MAST API call failed after {max_retries} attempts: {e}")
                raise

def download_tess_spoc_lightcurves(tics_df, data_collection_mode, data_dir, logger, max_retries=3, delay=5):
    """ Download light curve FITS files for the set of TIC IDs and sector runs provided in `tics_df` for
    the specified `data_collection_mode` mode.
    """

    sys.stdout = StreamToLogger(logger)

    requested_products_lst, requested_products_manifest_lst = [], []
    for _, tic_data in tics_df.iterrows():

        logger.info(f'Downloading light curve data for TIC {tic_data["tic_id"]} in sector run '
                    f'{tic_data["sector_run"]} ({data_collection_mode} data)...')

        # create sector array from sector run ID
        s_sector, e_sector = [int(sector_id) for sector_id in tic_data['sector_run'].split('-')]
        sector_arr = np.arange(s_sector, e_sector + 1)

        # create patterns for sectors to extract only products relevant to those
        lc_sectors_patterns = [f'-s{str(sector).zfill(4)}' for sector in sector_arr]

        try:
            # get table with observations for TIC and corresponding data collection mode (WITH RETRIES)
            obs_table = retry_mast_call(
                func=Observations.query_criteria, 
                logger=logger, 
                max_retries=max_retries,
                delay=5,
                target_name=tic_data['tic_id'],
                obs_collection='TESS' if data_collection_mode == '2min' else 'HLSP'
            )
            
            if len(obs_table) == 0:
                logger.error(f'No observations found for TIC {tic_data["tic_id"]}. Skipping...')
                continue

            # get table with all available products for queried observations (WITH RETRIES)
            products = retry_mast_call(
                func=Observations.get_product_list, 
                logger=logger, 
                max_retries=max_retries,
                delay=5,
                observations=obs_table
            )

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

            requested_products_lst.append(lc_products)

            # download requested products (WITH RETRIES)
            requested_products_manifest = retry_mast_call(
                func=Observations.download_products,
                logger=logger, 
                max_retries=max_retries,
                delay=5,
                products=lc_products, 
                download_dir=str(data_dir),
                mrp_only=False
            )
            
            requested_products_manifest_lst.append(requested_products_manifest)

            if not all(requested_products_manifest['Status']):
                logger.error(f'Could not download all requested light curves for TIC {tic_data["tic_id"]} in sector run '
                             f'{tic_data["sector_run"]} ({data_collection_mode} data. Skipping...)')
                continue

            logger.info(f'Finished downloading light curve data for TIC {tic_data["tic_id"]} in sector run '
                        f'{tic_data["sector_run"]} ({data_collection_mode} data)...')

        except Exception as e:
            logger.error(f'Failed processing TIC {tic_data["tic_id"]} due to network or MAST error: {e}. Skipping...')
            continue

    if len(requested_products_lst) == 0:
        logger.error(f'No requested light curves found for queried TICs. Stopping job...')
        sys.stdout = sys.__stdout__
        raise ValueError('No requested light curves found for queried TICs. Stopping job...')

    requested_products = vstack(requested_products_lst)
    requested_products.write(str(data_dir / f'requested_lightcurves_{data_collection_mode}.csv'),
                             format='csv', overwrite=True)
    requested_products_manifest = vstack(requested_products_manifest_lst)
    requested_products_manifest.write(
        str(data_dir / f'manifest_requested_lightcurves_{data_collection_mode}.csv'),
        format='csv', overwrite=True)

    # restore stdout
    sys.stdout = sys.__stdout__


def download_tess_spoc_dv_xmls(tics_df, data_collection_mode, data_dir, logger, max_retries=3, delay=5):
    """ Download DV XML data for the set of TIC IDs and sector runs provided in `tics_df` for
    the specified `data_collection_mode` mode.
    """

    sys.stdout = StreamToLogger(logger)

    requested_products_lst, requested_products_manifest_lst = [], []
    for _, tic_data in tics_df.iterrows():

        logger.info(f'Downloading DV XML data for TIC {tic_data["tic_id"]} in sector run '
                    f'{tic_data["sector_run"]} ({data_collection_mode} data)...')

        # create sector array from sector run ID
        s_sector, e_sector = [int(sector_id) for sector_id in tic_data['sector_run'].split('-')]
        sector_run_patern = f'-s{str(s_sector).zfill(4)}-s{str(e_sector).zfill(4)}'

        try:
            # get table with observations for TIC and corresponding data collection mode (WITH RETRIES)
            obs_table = retry_mast_call(
                Observations.query_criteria,
                logger=logger,
                max_retries=max_retries,
                delay=5,
                target_name=tic_data['tic_id'],
                obs_collection='TESS' if data_collection_mode == '2min' else 'HLSP'
            )
            
            if len(obs_table) == 0:
                logger.error(f'No observations found for TIC {tic_data["tic_id"]}. Skipping...')
                continue

            # get table with all available products for queried observations (WITH RETRIES)
            products = retry_mast_call(
                func=Observations.get_product_list,
                logger=logger,
                max_retries=max_retries,
                delay=5,
                observations=obs_table
            )

            if len(products) == 0:
                logger.error(f'No products found for TIC {tic_data["tic_id"]}. Skipping...')
                continue

            # filter for DV XML files
            dv_xml_products = products[[fn.endswith('dvr.xml') for fn in products["productFilename"]]]
            # filter DV XML files for sector run of interest
            dv_xml_products = dv_xml_products[
                [bool(re.search(sector_run_patern, data_url)) for data_url in dv_xml_products['productFilename']]]
                
            if len(dv_xml_products) == 0:
                logger.error(f'TESS SPOC DV XML file was not found for TIC {tic_data["tic_id"]} in sector run '
                             f'{tic_data["sector_run"]} for {data_collection_mode} '
                             f'data. Skipping...')
                continue

            # check for cases in which more than one DV XML file is available due to multiple SPOC runs
            if len(dv_xml_products) > 1:
                dv_xml_versions = [int(fn.split('-')[4].split('_')[0]) for fn in dv_xml_products['productFilename']]
                max_version_number = max(dv_xml_versions)
                logger.info(f'Found more than one DV XML file for TIC {tic_data["tic_id"]} in sector run '
                            f'{tic_data["sector_run"]}. Versions found: {dv_xml_versions}. Considering only the most recent '
                            f'one: {max_version_number}')
                dv_xml_products['version_number'] = dv_xml_versions
                dv_xml_products = dv_xml_products[dv_xml_products['version_number'] == max_version_number]

            requested_products_lst.append(dv_xml_products)

            # download requested products (WITH RETRIES)
            requested_products_manifest = retry_mast_call(
                func=Observations.download_products,
                logger=logger,
                max_retries=max_retries,
                delay=5,
                products=dv_xml_products,
                download_dir=str(data_dir),
                mrp_only=False
            )
            
            requested_products_manifest_lst.append(requested_products_manifest)

            if not all(requested_products_manifest['Status']):
                logger.error(f'Could not download all requested DV XML products for TIC {tic_data["tic_id"]} in sector run '
                             f'{tic_data["sector_run"]} ({data_collection_mode} data. Skipping...)')
                continue

            logger.info(f'Finished downloading DV XML data for TIC {tic_data["tic_id"]} in sector run '
                        f'{tic_data["sector_run"]} ({data_collection_mode} data)...')

        except Exception as e:
            logger.error(f'Failed processing TIC {tic_data["tic_id"]} due to network or MAST error: {e}. Skipping...')
            continue

    if len(requested_products_lst) == 0:
        logger.error(f'No requested DV XML products found for queried TICs. Stopping job...')
        sys.stdout = sys.__stdout__
        raise ValueError('No requested DV XML products found for queried TICs. Stopping job...')

    requested_products = vstack(requested_products_lst)
    requested_products.write(str(data_dir / f'requested_dv_xmls_{data_collection_mode}.csv'),
                             format='csv', overwrite=True)
    requested_products_manifest = vstack(requested_products_manifest_lst)
    requested_products_manifest.write(
        str(data_dir / f'manifest_requested_dv_xmls_{data_collection_mode}.csv'),
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
        stellar_parameters_source: str, the stellar parameters source to use for the queried TICs. Set to either
            'ticv8', 'tess-spoc', or filepath to external catalog of stellar parameters for the queried TICs.
        ruwe_source: str, the RUWE source to use for the queried TICs. Set to either 'gaiadr2', 'unavailable', or
            filepath to external catalog of RUWE values for the queried TICs.
        filter_tics: list of TIC IDs with sector run ID used to filter DV XML files; if None, no filtering is done

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


def check_custom_model(model_fp):
    """ Check if custom model loads.

    Args:
        model_fp: str, model filepath

    Returns:
    """
    
    # try:
    #     custom_objects = {"Time2Vec": Time2Vec, 'SplitLayer': SplitLayer}
    #     with custom_object_scope(custom_objects):
    #         model = load_model(filepath=model_fp, compile=False)
    # except Exception as e:
    #     raise ValueError(f'Error while loading custom model from {model_fp}. Ensure that the model is a valid '
    #                      f'TensorFlow Keras model and that it is compatible with the pipeline. Error:\n {e}')

    try:
        subprocess.run(
            [
                "python3", "-c",
                f"from tensorflow.keras.models import load_model; "
                f"from models.models_keras import Time2Vec, SplitLayer; "
                f"from tensorflow.keras.utils import custom_object_scope; "
                f"with custom_object_scope({{'Time2Vec': Time2Vec, 'SplitLayer': SplitLayer}}): "
                f"load_model('{model_fp}', compile=False)"
            ],
            check=True,
            timeout=10  # seconds
        )
        return True
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"Model loading timed out for {model_fp}")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Model loading failed: {e}")

def inference_pipeline(run_config, output_dir, tfrec_shards_fps, logger):
    """ Run inference pipeline.

    Args:
        run_config: dict, run configuration
        output_dir: Path, results directory
        tfrec_shards_fps: List[Path], list of TFRecord files
        logger: logging.Logger object

    Returns:

    """

    sys.stdout = StreamToLogger(logger)

    with open(run_config['predict_config_fp'], 'r') as file:
        predict_config = yaml.unsafe_load(file)

    if run_config.get('task') == 'phot-vetting':
        predict_config['config']['multi_class'] = True
        predict_config['label_map'] = run_config['label_map']['phot-vetting']
    elif run_config.get('task') == 'planet-validation':
        predict_config['config']['multi_class'] = False
        predict_config['label_map'] = run_config['label_map']['planet-validation']

    # tfrec_shards_fps = list(tfrec_dir.glob('shard-*'))

    predict_config['datasets_fps'] = {
        'predict' : tfrec_shards_fps
    }

    model_fp = Path(run_config['model_fp'])
    if model_fp.is_dir():
        model_fp = list(model_fp.glob('*.keras'))
    elif model_fp.is_file():
        model_fp = [model_fp]
    else:
        raise ValueError(f'Model filepath is not valid.')

    if isinstance(run_config['exominer_model_name'], str):
        if run_config['exominer_model_name'] == 'single':
            model_fp = model_fp[:1]
        elif run_config['exominer_model_name'] == 'cv_ensemble':
            model_fp = model_fp[:10]

    predict_model(predict_config, model_fp, output_dir, run_config['max_model_workers'], logger)

    # restore stdout
    sys.stdout = sys.__stdout__


def get_optimal_worker_count(model_ram_footprint_gb=0.5, max_limit=16, logger=None):
    """
    Dynamically calculates how many models can be safely loaded and run in parallel.
    
    :param model_ram_footprint_gb: Estimated RAM needed per model (default 500MB)
    :param max_limit: A hard cap to prevent thread-thrashing on massive servers
    """
    # 1. CPU Constraint: Leave 1 or 2 cores free for the OS / background tasks
    total_cpus = mp.cpu_count()
    # We use at least 1, but generally (total_cpus - 1)
    cpu_workers = max(1, total_cpus - 1)
    
    # 2. RAM Constraint: How much memory is actually free right now?
    # We use 'available' to avoid eating into memory already used by the OS
    available_ram_gb = psutil.virtual_memory().available / (1024 ** 3)
    
    # Leave a 2GB safety buffer for the OS and Python overhead
    safe_ram_gb = max(0, available_ram_gb - 2.0)
    ram_workers = max(1, int(safe_ram_gb / model_ram_footprint_gb))
    
    # 3. Final Calculation: Take the bottleneck (CPU or RAM), and apply the hard cap
    optimal_workers = min(cpu_workers, ram_workers, max_limit)
    
    if logger:
        logger.info(f"System specs: {total_cpus} CPUs, {available_ram_gb:.1f} GB Available RAM.")
        logger.info(f"Calculated optimal parallel workers: {optimal_workers} "
                    f"(CPU-bound: {cpu_workers}, RAM-bound: {ram_workers})")
        
    return optimal_workers


def _predict_single_model(model_i, model_fp, config, res_dir, fast_temp_dir, threads_per_model):
    """
    Worker function to process a single model in a thread.
    """

    os.environ['TF_NUM_INTRAOP_THREADS'] = str(threads_per_model)
    os.environ['TF_NUM_INTEROP_THREADS'] = str(threads_per_model)
    # os.environ['OMP_NUM_THREADS'] = str(threads_per_model)
    
    try:
        tf.config.threading.set_intra_op_parallelism_threads(threads_per_model)
        tf.config.threading.set_inter_op_parallelism_threads(threads_per_model)
    except RuntimeError:
        pass

    temp_model_path = os.path.join(fast_temp_dir, f"temp_model_{model_i}.keras")
    model_scores = {dataset: None for dataset in config['datasets']}
    
    try:
        # Copy to the fast/local directory
        shutil.copy(model_fp, temp_model_path)
        
        # Load from the local copy
        # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0' 
        # os.environ['TF_CPP_VMODULE'] = 'serving=2,saved_model=2,loader=2' 
        model = load_model(filepath=temp_model_path, compile=False)
        
        if config.get('write_model_summary') and model_i == 0:
            with open(res_dir / 'model_summary.txt', 'w') as f:
                model.summary(print_fn=lambda x: f.write(x + '\n'))

        if config.get('plot_model') and model_i == 0:
            plot_model(model,
                       to_file=res_dir / 'model.png',
                       show_shapes=True,
                       show_layer_names=True,
                       rankdir='TB',
                       expand_nested=False,
                       dpi=96)

        for dataset in config['datasets']:
            # log_info(f'Predicting on dataset {dataset} for model {model_i}...', logger)

            predict_input_fn = InputFn(
                file_paths=config['datasets_fps'][dataset],
                batch_size=config['inference']['batch_size'],
                mode='PREDICT',
                label_map=config['label_map'],
                features_set=config['features_set'],
                multiclass=config['config']['multi_class'],
                feature_map=config['feature_map'],
                label_field_name=config['label_field_name'],
            )

            # 2. Get the actual tf.data.Dataset object
            dataset_obj = predict_input_fn()
            
            # 3. Apply threading limits to the dataset
            options = tf.data.Options()
            # Force it to use a tiny number of threads (e.g., 1 or 2) per worker
            options.threading.private_threadpool_size = config['private_threadpool_size']
            options.threading.max_intra_op_parallelism = config['max_intra_op_parallelism']
            dataset_obj = dataset_obj.with_options(options)

            scores_output = model.predict(
                dataset_obj,
                verbose=0,
            )
            
            model_scores[dataset] = scores_output
            
    finally:
        # ALWAYS clean up to avoid memory/disk leaks
        if os.path.exists(temp_model_path):
            os.remove(temp_model_path)
        
        tf.keras.backend.clear_session()
            
    return model_i, model_scores


def predict_model(config, model_paths, res_dir, max_model_workers=1, logger=None):
    """ Run inference with a set of models and average their scores. """
    
    os.environ['OMP_WAIT_POLICY'] = 'PASSIVE'
    os.environ.pop('TF_NUM_INTRAOP_THREADS', None)
    os.environ.pop('TF_NUM_INTEROP_THREADS', None)
    os.environ.pop('OMP_NUM_THREADS', None)
    os.environ.pop('OPENBLAS_NUM_THREADS', None)

    config['features_set'] = set_tf_data_type_for_features(config['features_set'])
    
    if os.path.exists('/dev/shm'):
        fast_temp_dir = '/dev/shm'
    else:
        fast_temp_dir = tempfile.gettempdir()
        
    scores = {dataset: [] for dataset in config['datasets']}
    
    # ---------------------------------------------------------
    # PARALLELIZATION LOGIC
    # ---------------------------------------------------------
    # Set to 4 workers by default, or read from config
    # max_workers = config.get('inference', {}).get('max_workers', 10)
    max_workers = min(get_optimal_worker_count(max_limit=max_model_workers, logger=logger), len(model_paths))
    total_cores = mp.cpu_count()
    threads_per_model = max(1, total_cores // max_workers)
    log_info(f"Starting parallel prediction loop with {max_workers} workers, {threads_per_model} threads per model...", logger)

    # try:
    #     tf.config.threading.set_intra_op_parallelism_threads(threads_per_model)
    #     tf.config.threading.set_inter_op_parallelism_threads(threads_per_model)
    #     log_info(f"Set TensorFlow to use {threads_per_model} threads.", logger)
    # except RuntimeError as e:
    #     log_info(f"Could not change TF threads dynamically: {e}", logger)
    #     pass 

    # with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers, mp_context=mp.get_context('spawn')) as executor:

        # Submit all models to the thread pool
        futures = {
            executor.submit(_predict_single_model, i, fp, config, res_dir, fast_temp_dir, threads_per_model): i 
            for i, fp in enumerate(model_paths)
        }
        
        # Collect results as they finish (order doesn't matter since we append/accumulate)
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc='Iterating model (parallel)', unit='model'):
            model_i = futures[future]
            try:
                _, model_scores = future.result()
                
                # Unpack and structure the returned scores into the main dictionary
                for dataset, scores_model in model_scores.items():
                    if isinstance(scores_model, dict):  
                        if len(scores[dataset]) == 0: # Initialize dict structure on first hit
                            scores[dataset] = {k: [] for k in scores_model.keys()}
                        for score_k in scores_model:
                            scores[dataset][score_k].append(scores_model[score_k])
                    else:
                        scores[dataset].append(scores_model)
                        
            except Exception as e:
                log_info(f"Failed to process model {model_i}: {e}", logger)
    
    # ---------------------------------------------------------
    # AGGREGATION & SAVING LOGIC
    # ---------------------------------------------------------
    # We will create a new dictionary to hold the standard deviations
    scores_std = {dataset: {} for dataset in config['datasets']}
    
    # average scores across models and calculate std (uncertainty)
    for dataset in scores:
        if len(scores[dataset]) == 0:
            raise RuntimeError(f"All model predictions failed for dataset {dataset}. Check logs for abrupt terminations. ' \
            'Probably due to out-of-memory. Consider decreaseing the number of inference workers.")
        if isinstance(scores[dataset], dict):
            for score_k in scores[dataset]:
                raw_stacked_scores = scores[dataset][score_k]
                # Calculate std FIRST before overwriting the raw scores with the mean
                scores_std[dataset][score_k] = np.std(raw_stacked_scores, axis=0)
                scores[dataset][score_k] = np.mean(raw_stacked_scores, axis=0)
        else:
            raw_stacked_scores = scores[dataset]
            scores_std[dataset] = np.std(raw_stacked_scores, axis=0, ddof=1)
            scores[dataset] = np.mean(raw_stacked_scores, axis=0) 

    # get data from TFRecords files to be displayed in the table with predictions
    data = get_data_from_tfrecords_for_predictions_table(config['datasets'],
                                                         config['data_fields'],
                                                         config['datasets_fps'])
    
    # write results to a csv file
    for dataset in config['datasets']:
        
        log_info(f'Writing predictions for dataset {dataset}...', logger)
        
        if isinstance(scores[dataset], dict):
            scores_main = scores[dataset]['main']
            std_main = scores_std[dataset]['main']
            
            scores_aux = {k: v for k, v in scores[dataset].items() if k != 'main'}
            std_aux = {k: v for k, v in scores_std[dataset].items() if k != 'main'}
        else:  
            scores_main = scores[dataset]
            std_main = scores_std[dataset]
            scores_aux = {}
            std_aux = {}

        if not config['config']['multi_class']:
            data[dataset]['score'] = scores_main.ravel()
            data[dataset]['score_std'] = std_main.ravel() # Add binary std
        else:
            for class_label, label_id in config['label_map'].items():
                data[dataset][f'score_{class_label}'] = scores_main[:, label_id]
                data[dataset][f'score_std_{class_label}'] = std_main[:, label_id] # Add multi-class std

        # add auxiliary scores and their standard deviations
        for aux_score_name, aux_score_vals in scores_aux.items():
            data[dataset][f'score_{aux_score_name}'] = aux_score_vals
            data[dataset][f'score_std_{aux_score_name}'] = std_aux[aux_score_name]
            
        predictions_df = pd.DataFrame(data[dataset])

        # map labels to a label id that was used to train the model     
        if 'label' in predictions_df.columns:   
            predictions_df['label_id'] = predictions_df['label'].apply(lambda x: config['label_map'].get(x, -1)) 

        # sort in descending order of output (adjust based on multi-class vs binary)
        if not config['config']['multi_class']:
            predictions_df.sort_values(by='score', ascending=False, inplace=True)
        # Note: If it's multi_class, you usually don't sort, or you pick a specific class to sort by (like "exoplanet")
        
        predictions_df_fp = res_dir / f'predictions_{dataset}set.csv'
        
        # add metadata
        predictions_df.attrs['experiment'] = res_dir.name
        predictions_df.attrs['dataset'] = dataset
        if 'label_map' in config:
            predictions_df.attrs['label map'] =  config['label_map']
        predictions_df.attrs['created'] = str(pd.Timestamp.now().floor('min'))
        
        with open(predictions_df_fp, "w") as f:
            for key, value in predictions_df.attrs.items():
                f.write(f"# {key}: {value}\n")
            predictions_df.to_csv(f, index=False)

def create_tic_id_pattern(row, data_collection_mode):
    """ Create a formatted string pattern combining the TIC ID and sector ID.
    
    Args:
        row: pandas.Series or dict, contains 'tic_id' and 'sector_run' keys.
             'sector_run' should be in the format 'start-end' (e.g., '1-39').
        data_collection_mode: str, data collection mode, either 'ffi' or '2min'.
        
    Returns:
        str: The formatted TIC ID and sector pattern.
             For 'ffi', format is '{tic_id}-{sector_id}'.
             For '2min', format is '{sector_id}-{tic_id}'.
             
    Raises:
        ValueError: If `data_collection_mode` is not 'ffi' or '2min'.
    """
        
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

def assign_class(predictions_tbl, label_map, clf_thr):
    """ Assign class/disposition to each TCE based on ExoMiner score(s). Works for both binary and multiclass
     classification. If binary, must have column 'score'; if multiclass must have columns 'score_{class_0_name}', 
     'score_{class_1_name}', .... It must match classes in `label_map`.

     Args:
        predictions_tbl: pandas.DataFrame, containes ExoMiner scores table
        label_map: dict, maps labels/dispositions to label IDs used by ExoMiner
        clf_thr: float, classification threshold; when in multiclassification setting, class with max score is only 
            assigned to the TCE if the score is >= than `clf_thr`
    
    """

    label_map_reversed = {label_id: label for label, label_id in label_map.items()}

    if len(label_map) > 2:
        multiclass = True
    else:
        multiclass = False
    
    scores_cols = [col for col in predictions_tbl.columns if col.startswith('score')]
    if len(scores_cols) == 0:
        raise ValueError('No score columns found in predictions')

    if (multiclass and len(scores_cols) == 1):
        raise ValueError(f'Mismatch between number of score columns ({len(scores_cols)}) and multiclass setting.')
    if not multiclass and len(scores_cols) > 1:
        raise ValueError(f'Mismatch between number of score columns ({len(scores_cols)}) and binary class setting.')
    
    if not multiclass and scores_cols[0] != 'score':
        raise ValueError(f'Expected column `score` but found: {scores_cols}')
    
    if multiclass:
        predictions_tbl['prediction'] = predictions_tbl.apply(lambda row: map_softmax_predictions_to_class(row, scores_cols, label_map, clf_thr), axis=1)
        predictions_tbl['prediction'] = predictions_tbl['prediction'].map(label_map_reversed)
    else:
        neg_class, pos_class = label_map_reversed[0], label_map_reversed[1]
        predictions_tbl['prediction'] = neg_class
        predictions_tbl.loc[predictions_tbl['score'] >= clf_thr, 'prediction'] = pos_class

    return predictions_tbl

def compile_preprocessing_figures_to_pdf(tce, plot_dir, save_fp, delete_plots=False):
    """
    Compiles preprocessing PNG figures for a target and its TCEs into a single PDF.
    
    Args:
        target_uid (str): Target unique ID (e.g. TIC-Sector)
        tce_tbl (pandas.DataFrame): Table of TCEs for this target
        plot_dir (pathlib.Path): Directory containing the PNGs
        save_fp (pathlib.Path): Output filepath for the PDF
    """

    images_list = []
    
    imgs_fnames = [
        f'tess-spoc-tce_tic{tce["uid"]}_input-flux-and-centroid-views-to-exominer-model.png',
        f'tess-spoc-tce_tic{tce["uid"]}_input-flux-weak-secondary-views.png',
        f'tess-spoc-tce_tic{tce["uid"]}_input-flux-odd-even-views.png',
        f'tess-spoc-tce_tic{tce["uid"]}_input-periodogram-views.png',
    ]

    # find diff image figures for TCE
    diff_img_fnames = [fp.name for fp in plot_dir.glob(f'{tce["uid"]}*.png')]

    # keep track of images to delete after being aggregated to the PDF summary file
    del_img_fps = [plot_dir / fn for fn in imgs_fnames + diff_img_fnames]

    # add only not target centered
    filt_diff_img_fnames = [fn for fn in diff_img_fnames if not fn.endswith('tc.png')]
    # sort by sector; assuming pattern <uid>_diff_img_<sector_id>.png
    filt_diff_img_fnames.sort(key=lambda fn: int(fn.split('_')[3].split('.')[0]))

    imgs_fnames += filt_diff_img_fnames

    for img_fn in imgs_fnames:
        plot_fp = plot_dir / img_fn
        if plot_fp.exists():
            images_list.append(plot_fp)
                
    if not images_list:
        return
        
    try:
        imgs = [Image.open(img).convert('RGB') for img in images_list]
        imgs[0].save(str(save_fp), save_all=True, append_images=imgs[1:])
        for img in imgs:
            img.close()
    except Exception as e:
        print(f"Failed to compile PDF for {tce['uid']}: {e}")
    
    if delete_plots:
        for image_fp in del_img_fps:
            image_fp.unlink(image_fp)
