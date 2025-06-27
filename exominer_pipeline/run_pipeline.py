"""
Main script used to run the ExoMiner pipeline.
"""

# 3rd party
import argparse
import logging
from pathlib import Path
import numpy as np
import multiprocessing as mp

# local
from exominer_pipeline.utils import (process_inputs, check_config, download_tess_spoc_data_products, create_tce_table,
                                     inference_pipeline)
from src_preprocessing.lc_preprocessing.generate_input_records import preprocess_lc_data
from src_preprocessing.diff_img.extracting.utils_diff_img import get_data_from_tess_dv_xml_multiproc
from src_preprocessing.diff_img.preprocessing.preprocess_diff_img import preprocess_diff_img_tces_main
from src_preprocessing.diff_img.preprocessing.add_data_to_tfrecords import write_diff_img_data_to_tfrec_files_main
from src_preprocessing.normalize_tfrecord_dataset.normalize_data_tfrecords import normalize_examples_main
from src_preprocessing.utils_manipulate_tfrecords import create_table_for_tfrecord_dataset

logger = logging.getLogger(__name__)

def run_exominer_pipeline(run_config, tics_df, job_id):
    """ Run ExoMiner pipeline for a set of TIC IDs.

    Args:
        run_config: dict, run configuration
        tics_df: pandas DataFrame containing TIC IDs "tic_id" and sector runs "sector_run" for which the data are to be
            downloaded. "sector_run" column should show the start and end sector for the run in the pattern
            {start_sector}-{end_sector}. "tic_id" should show the TIC ID as an integer.
        job_id: int, job ID

    Returns:

    """

    run_config['job_dir'] = run_config['output_dir'] / f'job_{job_id}'
    run_config['job_dir'].mkdir(exist_ok=True)

    run_config['log_dir'] = run_config['job_dir'] / 'logs'
    run_config['log_dir'].mkdir(exist_ok=True)

    # download required data products - light curve FITS and DV XML files
    download_tess_spoc_data_products(tics_df, run_config['data_collection_mode'], run_config['job_dir'])

    folder_name = 'HLSP' if run_config['data_collection_mode'] == 'ffi' else 'TESS'
    run_config['data_products_dir'] = run_config['job_dir'] / 'mastDownload' / folder_name

    # create TCE table from DV XML data
    tce_tbl = create_tce_table(run_config['job_dir'], job_id, run_config['data_products_dir'])
    tce_tbl_fp = run_config['job_dir'] / f'tess-spoc-dv_tces_{job_id}_processed.csv'
    tce_tbl.to_csv(tce_tbl_fp, index=False)

    # preprocess light curve data to create TFRecord dataset
    tfrec_dir = run_config['job_dir'] / 'tfrecord_data'
    tfrec_dir.mkdir(exist_ok=True)
    preprocess_lc_data(run_config['lc_preprocessing_config_fp'], tfrec_dir,
                       run_config['data_products_dir'], tce_tbl_fp, -1, 1)
    tfrec_fps = [tfrec_fp for tfrec_fp in tfrec_dir.glob('shard-*') if tfrec_fp.suffix != '.csv']
    # create TFRecord dataset table
    tfrec_tbl = create_table_for_tfrecord_dataset(tfrec_fps, run_config['data_fields_tfrec_tbl'],
                                                  delete_corrupted_tfrec_files=False, verbose=True)
    tfrec_tbl.to_csv(tfrec_dir / 'shards_tbl.csv', index=False)

    # extract and preprocess difference image data
    get_data_from_tess_dv_xml_multiproc(run_config['data_products_dir'], run_config['job_dir'], neighbors_dir=None,
                                        plot_dir=run_config['job_dir'], plot_prob=0, log_dir=run_config['job_dir'],
                                        job_i=job_id)
    src_diff_img_dir = run_config['job_dir'] / 'data'
    preprocessed_diff_img_dir = run_config['job_dir'] / 'diff_img_preprocessed'
    preprocess_diff_img_tces_main(run_config['diff_img_preprocessing_config_fp'], preprocessed_diff_img_dir, src_diff_img_dir)

    # create new TFRecord dataset with added preprocessed difference image data to the already preprocessed light curve
    # data TFRecord dataset
    write_diff_img_data_to_tfrec_files_main(run_config['diff_img_add_tfrecord_dataset_config_fp'], tfrec_dir, preprocessed_diff_img_dir)
    tfrec_dir_w_diff_img = tfrec_dir.parent / f'{tfrec_dir.name}_diffimg'

    # normalize features in TFRecord dataset
    normalized_tfrec_dir = tfrec_dir_w_diff_img.parent / f'{tfrec_dir_w_diff_img.name}_normalized'
    normalize_examples_main(run_config['normalize_tfrec_data_config_fp'], tfrec_dir_w_diff_img, normalized_tfrec_dir)

    # load trained model and run inference
    prediction_dir = run_config['job_dir'] / 'predictions'
    prediction_dir.mkdir(exist_ok=True)
    inference_pipeline(run_config, prediction_dir, normalized_tfrec_dir)


def run_exominer_pipeline_jobs_parallel(jobs, num_processes):
    """ Run ExoMiner pipeline for a number of jobs.

    Args:
        jobs: list, jobs to run
        num_processes: int, number of processes to use

    Returns:

    """

    logger.info(f'Started running ExoMiner pipeline on {num_processes} processes...')
    pool = mp.Pool(processes=num_processes)
    async_results = [pool.apply_async(run_exominer_pipeline, job) for job in jobs]
    pool.close()
    pool.join()

    for async_result in async_results:
        async_result.get()

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

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # create logger
    logging.basicConfig(filename=output_dir / 'run.log',
                        level=logging.INFO,
                        format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        filemode='a',
                        )

    logger.info(f'Started run for ExoMiner.')

    # logger.info(f'Checking command-line inputs...')
    # check_command_line_arguments(config_fp, tic_ids_fp, tic_ids, data_collection_mode, num_processes)
    # logger.info('Done checking command-line inputs.')

    logger.info(f'Preparing inputs and adjusting configuration file...')
    run_config, tics_df = process_inputs(output_dir, config_fp, tic_ids_fp, data_collection_mode, tic_ids, num_processes)
    logger.info('Done.')

    # TODO: check structure of TIC IDs CSV file

    logger.info(f'Checking validity of configuration file...')
    check_config(run_config)
    logger.info('Done checking configuration file.')

    tics_tbl_fp = output_dir / "tics_tbl.csv"
    logger.info(f'Found {len(tics_df)} TIC IDs. Saving TIC IDs to {str(tic_ids_fp)}...')
    tics_df.to_csv(tics_tbl_fp, index=False)

    logger.info(f'Split TIC IDs into multiple jobs to be processed in parallel using {run_config["num_processes"]} '
                f'processes.')
    tics_df_jobs = np.array_split(tics_df, len(tics_df))
    jobs = [(run_config, tics_job, job_id) for job_id, tics_job in enumerate(tics_df_jobs)]
    logger.info(f'Created {len(jobs)} jobs.')

    run_exominer_pipeline_jobs_parallel(jobs, run_config['num_processes'])


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
