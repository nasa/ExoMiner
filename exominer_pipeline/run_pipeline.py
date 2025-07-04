"""
Main script used to run the ExoMiner pipeline.
"""

# 3rd party
import argparse
import logging
from pathlib import Path
import numpy as np
import multiprocessing as mp
import pandas as pd

# local
from exominer_pipeline.utils import (process_inputs, check_config, download_tess_spoc_data_products, create_tce_table,
                                     inference_pipeline)
from src_preprocessing.lc_preprocessing.generate_input_records import preprocess_lc_data
from src_preprocessing.diff_img.extracting.utils_diff_img import get_data_from_tess_dv_xml_multiproc
from src_preprocessing.diff_img.preprocessing.preprocess_diff_img import preprocess_diff_img_tces_main
from src_preprocessing.diff_img.preprocessing.add_data_to_tfrecords import write_diff_img_data_to_tfrec_files_main
from src_preprocessing.normalize_tfrecord_dataset.normalize_data_tfrecords import normalize_examples_main
from src_preprocessing.utils_manipulate_tfrecords import create_table_for_tfrecord_dataset


def run_exominer_pipeline(run_config, tics_df, job_id):
    """ Run ExoMiner pipeline for a set of TIC IDs.

    Args:
        run_config: dict, run configuration
        tics_df: pandas DataFrame containing TIC IDs "tic_id" and sector runs "sector_run" for which the data are to be
            downloaded. "sector_run" column should show the start and end sector for the run in the pattern
            {start_sector}-{end_sector}. "tic_id" should show the TIC ID as an integer.
        job_id: int, job ID

    Returns: dict with keys "success" and "error"

    """

    run_config['job_dir'] = run_config['output_dir'] / f'job_{job_id}'
    run_config['job_dir'].mkdir(exist_ok=True)

    logger = logging.getLogger(name=f'run_{job_id}.log')
    logger_handler = logging.FileHandler(filename=run_config['job_dir'] / f'run_{job_id}.log',
                                         mode='w')
    logger_formatter = logging.Formatter('%(asctime)s - %(message)s')
    logger.setLevel(logging.INFO)
    logger_handler.setFormatter(logger_formatter)
    logger.addHandler(logger_handler)
    logger.propagate = False
    logger.info(f'[{job_id}] Starting run for {len(tics_df)} TIC IDs in job {job_id}...')

    try:
        # download required data products - light curve FITS and DV XML files
        logger.info(f'[{job_id}] Downloading data products for the requested TIC IDs...')
        download_tess_spoc_data_products(tics_df, run_config['data_collection_mode'], run_config['job_dir'], logger)

        folder_name = 'HLSP' if run_config['data_collection_mode'] == 'ffi' else 'TESS'
        run_config['data_products_dir'] = run_config['job_dir'] / 'mastDownload' / folder_name

        # create TCE table from DV XML data
        logger.info(f'[{job_id}] Creating TESS SPOC TCE table from DV XML files downloaded for the requested TIC '
                    f'IDs...')
        tce_tbl_dir = run_config['job_dir'] / 'tce_table'
        tce_tbl_dir.mkdir(exist_ok=True)
        tce_tbl = create_tce_table(tce_tbl_dir, job_id, run_config['data_products_dir'], logger)
        tce_tbl_fp = tce_tbl_dir / f'tess-spoc-dv_tces_{job_id}_processed.csv'
        tce_tbl.to_csv(tce_tbl_fp, index=False)

        # preprocess light curve data to create TFRecord dataset
        logger.info(f'[{job_id}] Preprocessing light curve data for the requested TIC IDs...')
        tfrec_dir = run_config['job_dir'] / 'tfrecord_data'
        tfrec_dir.mkdir(exist_ok=True)
        preprocess_lc_data(run_config['lc_preprocessing_config_fp'], tfrec_dir,
                           run_config['data_products_dir'], tce_tbl_fp, -1, 1)
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        # create TFRecord dataset table
        tfrec_fps = [tfrec_fp for tfrec_fp in tfrec_dir.glob('shard-*') if tfrec_fp.suffix != '.csv']
        tfrec_tbl = create_table_for_tfrecord_dataset(tfrec_fps, run_config['data_fields_tfrec_tbl'],
                                                      delete_corrupted_tfrec_files=False, verbose=True, logger=logger)
        tfrec_tbl.to_csv(tfrec_dir / 'shards_tbl.csv', index=False)

        # extract difference image data
        logger.info(f'[{job_id}] Extracting difference image data from the DV XML files for the requested TIC IDs...')
        diff_img_dir = run_config['job_dir'] / 'diff_img_extracted'
        diff_img_dir.mkdir(exist_ok=True)
        get_data_from_tess_dv_xml_multiproc(run_config['data_products_dir'], diff_img_dir, neighbors_dir=None,
                                            plot_dir=diff_img_dir, plot_prob=0, log_dir=diff_img_dir,
                                            job_i=job_id)
        # preprocess difference image data
        logger.info(f'[{job_id}] Preprocessing difference image data for the requested TIC IDs...')
        preprocessed_diff_img_dir = run_config['job_dir'] / 'diff_img_preprocessed'
        preprocessed_diff_img_dir.mkdir(exist_ok=True)
        preprocess_diff_img_tces_main(run_config['diff_img_preprocessing_config_fp'], preprocessed_diff_img_dir,
                                      diff_img_dir)

        # create new TFRecord dataset with added preprocessed difference image data to the already preprocessed light
        # curve data TFRecord dataset
        logger.info(f'[{job_id}] Adding preprocessed difference image data to the TFRecord dataset for the requested '
                    f'TIC IDs...')
        write_diff_img_data_to_tfrec_files_main(run_config['diff_img_add_tfrecord_dataset_config_fp'], tfrec_dir,
                                                preprocessed_diff_img_dir)
        tfrec_dir_w_diff_img = tfrec_dir.parent / f'{tfrec_dir.name}_diffimg'

        # normalize features in TFRecord dataset
        logger.info(f'[{job_id}] Normalizing features for the TCEs in the TFRecord dataset using set of normalization '
                    f'statistics provided...')
        normalized_tfrec_dir = tfrec_dir_w_diff_img.parent / f'{tfrec_dir_w_diff_img.name}_normalized'
        normalize_examples_main(run_config['normalize_tfrec_data_config_fp'], tfrec_dir_w_diff_img,
                                normalized_tfrec_dir)

        # load trained model and run inference
        logger.info(f'[{job_id}] Running inference on TCEs in the TFRecord dataset using trained ExoMiner model...')
        prediction_dir = run_config['job_dir'] / 'predictions'
        prediction_dir.mkdir(exist_ok=True)
        inference_pipeline(run_config, prediction_dir, normalized_tfrec_dir, logger)

        logger.info(f'[{job_id}] Finished run for job {job_id} for {len(tics_df)} TIC IDs.')

        return {'job_id': job_id, 'success': True, 'error': None}

    except Exception as e:
        logger.error(f'[{job_id}] Error: {e}', exc_info=True)
        return {'job_id': job_id, 'success': False, 'error': e}


def run_exominer_pipeline_jobs_parallel(jobs, num_processes, logger):
    """ Run ExoMiner pipeline for a number of jobs.

    Args:
        jobs: list, jobs to run
        num_processes: int, number of processes to use
        logger: logger object

    Returns:

    """

    pool = mp.Pool(processes=num_processes)
    async_results = [pool.apply_async(run_exominer_pipeline, args=job) for job in jobs]
    pool.close()
    pool.join()

    for async_result in async_results:
        result_job = async_result.get()
        if not result_job['success']:
            logger.info(f'Error in job {result_job["job_id"]}: {result_job["error"]}')
        else:
            logger.info(f'Job {result_job["job_id"]} is complete.')

def run_exominer_pipeline_main(config_fp, output_dir, tic_ids_fp, data_collection_mode, tic_ids=None, num_processes=-1):
    """ Run ExoMiner pipeline.

    Args:
        config_fp: str, filepath to the configuration file for the run.
        output_dir: str, directory to save the output of the run.
        tic_ids_fp: str, filepath to the TIC IDs file for the run.
        data_collection_mode: str, either '2min' or 'ffi'.
        tic_ids: str, list of TIC IDs to process. Only used if `tic_ids_fp` is None.
        num_processes: int, number of processes to use.

    Returns:

    """

    output_dir = Path(output_dir)  # create results directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # create logger
    logger = logging.getLogger(name=f'run_main.log')
    logger_handler = logging.FileHandler(filename=output_dir / 'run_main.log',
                                         mode='w')
    logger_formatter = logging.Formatter('%(asctime)s - %(message)s')
    logger.setLevel(logging.INFO)
    logger_handler.setFormatter(logger_formatter)
    logger.addHandler(logger_handler)

    logger.info(f'Started run for ExoMiner.')

    logger.info(f'Preparing inputs and adjusting configuration file...')
    run_config, tics_df = process_inputs(output_dir, config_fp, tic_ids_fp, data_collection_mode, logger,
                                         tic_ids=tic_ids, num_processes=num_processes)
    logger.info('Done.')

    # TODO: check structure of TIC IDs CSV file

    logger.info(f'Checking validity of configuration file...')
    check_config(run_config, logger)
    logger.info('Done checking configuration file.')

    tics_tbl_fp = output_dir / "tics_tbl.csv"
    logger.info(f'Found {len(tics_df)} TIC IDs. Saving TIC IDs to {str(tic_ids_fp)}...')
    tics_df.to_csv(tics_tbl_fp, index=False)

    n_jobs = run_config['num_jobs'] if run_config['num_jobs'] != -1 else 1
    logger.info(f'Set number of jobs to {n_jobs}.')
    # split TIC IDs across jobs
    tics_df_jobs = np.array_split(tics_df, n_jobs)
    jobs = [(run_config, tics_job, job_id) for job_id, tics_job in enumerate(tics_df_jobs)]
    logger.info(f'Split TIC IDs into {len(jobs)} jobs to be processed in parallel using {run_config["num_processes"]} '
                f'process(es).')
    logger.info(f'Started running ExoMiner pipeline on {num_processes} processes for {len(jobs)} jobs...')
    run_exominer_pipeline_jobs_parallel(jobs, run_config['num_processes'], logger)

    # aggregate predictions across jobs
    predictions_tbl_fp = output_dir / f'predictions_{output_dir.name}.csv'
    logger.info(f'Aggregating predictions into a single table in {predictions_tbl_fp}...')
    predictions_tbls_fps = list(Path(output_dir).rglob('ranked_predictions_predictset.csv'))
    logger.info(f'Found {len(predictions_tbls_fps)} job predictions files in {output_dir}.')
    predictions_tbl = pd.concat([pd.read_csv(fp) for fp in predictions_tbls_fps], axis=0, ignore_index=True)
    predictions_tbl.to_csv(predictions_tbl_fp, index=False)
    logger.info(f'Saved predictions to {predictions_tbl_fp}.')

    logger.info(f'Finished running ExoMiner pipeline.')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--config_fp', type=str, help='Filepath to YAML configuration file.',
                        default=None)
    parser.add_argument('--output_dir', type=str, help='Output directory the results are saved into.',
                        default=None)
    parser.add_argument('--tic_ids_fp', type=str, help='Filepath to CSV file containing the TIC IDs and '
                                                       'the corresponding sector run. Must include header with columns '
                                                       '"tic_id" and "sector_run". Sector run should be provided as the '
                                                       'start and end sectors of the run '
                                                       '({start_sector}-{end_sector}). If no sector run is provided, '
                                                       'it will consider all available sector runs. If this argument is '
                                                       'set, it takes preference over argument '
                                                       '--tic_ids ', default=None)
    parser.add_argument('--tic_ids', type=str, help='Comma-separated list of TIC IDs with option for '
                                                    'including the corresponding sector run. The accepted format is '
                                                    '{tic_id_a}_{sector_run},{tic_id_b}-{sector_run}, ..., '
                                                    'or {tic_id_a},{tic_id_b},... if no sector run is provided.' 
                                                    'Alternative to --tic_ids_fp. This argument is ignored if '
                                                    '--tic_ids_fp is set.',
                        default=None)
    parser.add_argument('--data_collection_mode', type=str, help='Either "2min" of "FFI" to process '
                                                                 'TESS SPOC 2-min or FFI TCE data. By default, it '
                                                                 'is set to 2min.',
                        default="2min")
    parser.add_argument('--num_processes', type=int, help='Number of processes to use for parallelization. '
                                                          'Set to "-1" by default which means use the parameter defined '
                                                          'in the YAML configuration file. If valid, it overwrites '
                                                          'value set in the YAML configuration file.', default=-1)

    parsed_args = parser.parse_args()

    run_exominer_pipeline_main(parsed_args.config_fp, parsed_args.output_dir, parsed_args.tic_ids_fp,
                               parsed_args.data_collection_mode, parsed_args.tic_ids, parsed_args.num_processes)
