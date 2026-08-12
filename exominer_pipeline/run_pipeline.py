"""
Main script used to run the ExoMiner pipeline.
"""

# 3rd party
import tensorflow as tf
import argparse
import logging
from pathlib import Path
import shutil
import numpy as np
import multiprocessing as mp
import pandas as pd
import yaml
import traceback
import subprocess

# local
from exominer_pipeline.utils import (process_inputs, check_config, validate_tic_ids_csv_structure, 
                                     download_tess_spoc_dv_xmls, download_tess_spoc_lightcurves, create_tce_table,
                                     inference_pipeline, create_tic_id_pattern, assign_class, 
                                     compile_preprocessing_figures_to_pdf)
from src_preprocessing.lc_preprocessing.generate_input_records import preprocess_lc_data
from src_preprocessing.diff_img.extracting.utils_diff_img import get_data_from_tess_dv_xml_main
from src_preprocessing.diff_img.preprocessing.preprocess_diff_img import preprocess_diff_img_tces_main
from src_preprocessing.diff_img.preprocessing.add_data_to_tfrecords import write_diff_img_data_to_tfrec_files_main
from src_preprocessing.normalize_tfrecord_dataset.normalize_data_tfrecords import normalize_examples_main
from src_preprocessing.utils_manipulate_tfrecords import create_table_for_tfrecord_dataset
from query_dv_reports import get_dv_dataproducts_list, correct_sector_field, inverse_correct_sector_field

CONFIG_FP = 'exominer_pipeline/pipeline_run_config.yaml'

def force_cpu_inference():
    """
    Universally disables all GPUs (Metal, NVIDIA, AMD) so TensorFlow 
    runs purely on the CPU for both local Mac testing and Linux Podman deployments.
    """
    try:
        # Hide all GPUs from TensorFlow
        tf.config.set_visible_devices([], 'GPU')
        print("Hardware config: GPUs disabled. Running purely on CPU.")
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(f"Hardware config error: {e}")


def get_git_commit_hash():
    """ Get Git commit hash."""

    try:
        git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL).decode('ascii').strip()
    except Exception:
        git_hash = 'unavailable'
    
    return git_hash


def run_exominer_pipeline(run_config, tics_df, job_id):
    """ Run ExoMiner pipeline for a set of TIC IDs.

    Args:
        run_config: dict, run configuration
        tics_df: pandas DataFrame containing TIC IDs "tic_id" and sector runs "sector_run" for which the data are to be
            downloaded. "sector_run" column should show the start and end sector for the run in the pattern
            {start_sector}-{end_sector}. "tic_id" should show the TIC ID as an integer.
        job_id: int, job ID

    Returns: dict with:
                - job_id
                - success
                - error
                - traceback
                - tce_tracker_fp
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
    
    if len(tics_df) == 0:
        logger.info(f'[{job_id}] Finished run for job {job_id} for {len(tics_df)} TIC IDs.')
        return {'job_id': job_id, 'success': True, 'error': None}

    try:

        # sys.stdout = StreamToLogger(logger)

        if run_config['lc_data_repository'] is None:  # download light curve FITS files
            logger.info(f'[{job_id}] Downloading light curve FITS files for the requested TIC IDs...')
            download_tess_spoc_lightcurves(tics_df, run_config['data_collection_mode'], run_config['job_dir'], 
                                           logger, run_config['max_retries'], run_config['delay'])

            folder_name = 'HLSP' if run_config['data_collection_mode'] == 'ffi' else 'TESS'
            run_config['lc_data_products_dir'] = run_config['job_dir'] / 'mastDownload' / folder_name
        else:
            logger.info(f'[{job_id}] Using local light curve data repository...')
            run_config['lc_data_products_dir'] = Path(run_config['lc_data_repository'])
        
        if run_config['dv_xml_data_repository'] is None:  # download DV XML files
            logger.info(f'[{job_id}] Downloading DV XML files for the requested TIC IDs...')
            download_tess_spoc_dv_xmls(tics_df, run_config['data_collection_mode'], run_config['job_dir'], logger, 
                                       run_config['max_retries'], run_config['delay'])

            folder_name = 'HLSP' if run_config['data_collection_mode'] == 'ffi' else 'TESS'
            run_config['dv_xml_data_products_dir'] = run_config['job_dir'] / 'mastDownload' / folder_name
        else:
            logger.info(f'[{job_id}] Using local DV XML data repository...')
            run_config['dv_xml_data_products_dir'] = Path(run_config['dv_xml_data_repository'])

        # create TCE table from DV XML data
        logger.info(f'[{job_id}] Creating TESS SPOC TCE table from DV XML files downloaded for the requested TIC '
                    f'IDs...')
        tce_tbl_dir = run_config['job_dir'] / 'tce_table'
        tce_tbl_dir.mkdir(exist_ok=True)
        if run_config['dv_xml_data_repository'] is None:
            tics_pattern = None
        else:
            tics_pattern = tics_df.apply(lambda x: create_tic_id_pattern(x, data_collection_mode=run_config['data_collection_mode']), axis=1).to_list()
            
        tce_tbl = create_tce_table(tce_tbl_dir,
                                    job_id, 
                                    run_config['dv_xml_data_products_dir'], 
                                    logger,
                                    run_config['stellar_parameters_source'], 
                                    run_config['ruwe_source'],
                                    filter_tics=tics_pattern,
                                    )

        if len(tce_tbl) == 0:
            raise ValueError(f'[{job_id}] No TCEs found for the requested {len(tics_df)} TIC IDs. Finishing job.')
        tce_tbl_fp = tce_tbl_dir / f'tess-spoc-dv_tces_{job_id}_processed.csv'
        tce_tbl.to_csv(tce_tbl_fp, index=False)

        # set TCE tracker
        tce_tracker = tce_tbl[['target_id', 'sector_run', 'uid']].copy()
        tce_tracker['job_id'] = job_id
        tce_tracker['job_start_time'] = pd.Timestamp.now().isoformat()
        tce_tracker['in_tce_table'] = True
        tce_tracker['in_tfrecord'] = False
        tce_tracker['has_diff_img'] = False
        tce_tracker['is_normalized'] = False
        tce_tracker['has_prediction'] = False
        tce_tracker['has_mast_url'] = False

        # preprocess light curve data to create TFRecord dataset
        logger.info(f'[{job_id}] Preprocessing light curve data for the requested TIC IDs...')
        tfrec_dir = run_config['job_dir'] / 'tfrecord_data'
        tfrec_dir.mkdir(exist_ok=True)
        preprocess_lc_data(run_config['lc_preprocessing_config_fp'], tfrec_dir,
                           run_config['lc_data_products_dir'], tce_tbl_fp, -1, 1)
        logger.info(f'[{job_id}] Finished preprocessing light curve data for the requested TIC IDs.')
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        # create TFRecord dataset table
        logger.info(f'[{job_id}] Creating auxiliary shards table for the TFRecord dataset...')
        tfrec_fps = [tfrec_fp for tfrec_fp in tfrec_dir.glob('shard-*') if tfrec_fp.suffix != '.csv']
        tfrec_tbl = create_table_for_tfrecord_dataset(tfrec_fps, run_config['data_fields_tfrec_tbl'],
                                                      delete_corrupted_tfrec_files=False, verbose=True, logger=logger)
        if len(tfrec_tbl) == 0:
            raise ValueError(f'[{job_id}] No TCEs were preprocessed into valid light curve TFRecord files for the requested {len(tics_df)} TIC IDs. Finishing job.')
        tfrec_tbl.to_csv(tfrec_dir / 'shards_tbl.csv', index=False)
        logger.info(f'[{job_id}] Created auxiliary shards table for the TFRecord dataset.')

        # Update tracker: Check which UIDs made it into the TFRecord table
        if len(tfrec_tbl) > 0 and 'uid' in tfrec_tbl.columns:
            valid_tfrec_uids = tfrec_tbl['uid'].unique()
            tce_tracker.loc[tce_tracker['uid'].isin(valid_tfrec_uids), 'in_tfrecord'] = True

        # extract difference image data
        logger.info(f'[{job_id}] Extracting difference image data from the DV XML files for the requested TIC IDs...')
        diff_img_dir = run_config['job_dir'] / 'diff_img_extracted'
        diff_img_dir.mkdir(exist_ok=True)
        get_data_from_tess_dv_xml_main(
            run_config['dv_xml_data_products_dir'], 
            diff_img_dir, 
            neighbors_dir=None,
            lc_dir=None,
            plot_dir=diff_img_dir, 
            plot_prob=0, 
            log_dir=diff_img_dir,
            job_i=job_id,
            targets_sectors_tbl=tics_df,
            check_existence_multiple_versions=True,
            data_collection_mode=run_config['data_collection_mode'],
            )
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
        tfrec_fps = [tfrec_fp for tfrec_fp in tfrec_dir_w_diff_img.glob('shard-*') if tfrec_fp.suffix != '.csv']
        tfrec_tbl = create_table_for_tfrecord_dataset(tfrec_fps, run_config['data_fields_tfrec_tbl'],
                                                      delete_corrupted_tfrec_files=False, verbose=True, logger=logger)
        if len(tfrec_tbl) > 0 and 'uid' in tfrec_tbl.columns:
            valid_tfrec_uids = tfrec_tbl['uid'].unique()
            tce_tracker.loc[tce_tracker['uid'].isin(valid_tfrec_uids), 'has_diff_img'] = True

        # aggregate preprocessing figures into PDF files
        if run_config['plot_inputs_to_model']:
            logger.info(f'[{job_id}] Created TCE summary files for the preprocessed data...')
            for _, tce in tce_tbl.iterrows():
                compile_preprocessing_figures_to_pdf(tce, run_config['plot_dir'], 
                run_config['plot_dir']/ f'tess-spoc-tce_tic{tce["uid"]}_summary_exominer-pipeline.pdf',delete_plots=True)

        # normalize features in TFRecord dataset
        logger.info(f'[{job_id}] Normalizing features for the TCEs in the TFRecord dataset using set of normalization '
                    f'statistics provided...')
        normalized_tfrec_dir = tfrec_dir_w_diff_img.parent / f'{tfrec_dir_w_diff_img.name}_normalized'
        normalize_examples_main(run_config['normalize_tfrec_data_config_fp'], tfrec_dir_w_diff_img,
                                normalized_tfrec_dir)
        tfrec_fps = [tfrec_fp for tfrec_fp in normalized_tfrec_dir.glob('shard-*') if tfrec_fp.suffix != '.csv']
        tfrec_tbl = create_table_for_tfrecord_dataset(tfrec_fps, run_config['data_fields_tfrec_tbl'],
                                                      delete_corrupted_tfrec_files=False, verbose=True, logger=logger)
        if len(tfrec_tbl) > 0 and 'uid' in tfrec_tbl.columns:
            valid_tfrec_uids = tfrec_tbl['uid'].unique()
            tce_tracker.loc[tce_tracker['uid'].isin(valid_tfrec_uids), 'is_normalized'] = True

        # # load trained model and run inference
        # logger.info(f'[{job_id}] Running inference on TCEs in the TFRecord dataset using trained ExoMiner model...')
        # prediction_dir = run_config['job_dir'] / 'predictions'
        # prediction_dir.mkdir(exist_ok=True)
        # inference_pipeline(run_config, prediction_dir, normalized_tfrec_dir, logger)
        # pred_tbl_fps = list(prediction_dir.glob('*.csv'))
        # if len(pred_tbl_fps) > 0:
        #     pred_tbl = pd.concat([pd.read_csv(fp, comment='#') for fp in pred_tbl_fps], axis=0)
        #     if len(pred_tbl) > 0 and 'uid' in pred_tbl.columns:
        #         tce_tracker.loc[tce_tracker['uid'].isin(pred_tbl['uid']), 'has_prediction'] = True

        # download CSV table with URLs to DV reports for each TCE for the queried TICs in the MAST
        if run_config['get_mast_urls_dv_reports'] == 'true':
            tce_uids_lst = tce_tbl['uid'].apply(correct_sector_field).to_list()
            get_dv_dataproducts_list(tce_uids_lst,
                                     ['DV TCE summary report', 'Full DV report', 'DV mini-report'],
                                     run_config['job_dir'] , False, 'all',
                                     spoc_ffi=run_config['data_collection_mode'] == 'ffi',
                                     verbose=False,
                                     create_mast_url_csv=True,
                                     )
            
            mast_url_dvr_tbl_fp = run_config['job_dir'] / 'mast_urls_tables' / f'spoc-dv_mast-urls_job0.csv'
            mast_url_dvr_tbl = pd.read_csv(mast_url_dvr_tbl_fp, comment='#')
            if len(mast_url_dvr_tbl) > 0 and 'uid' in mast_url_dvr_tbl.columns:
                tces_uids_for_dvr = mast_url_dvr_tbl['uid'].apply(inverse_correct_sector_field)
                tce_tracker.loc[tce_tracker['uid'].isin(tces_uids_for_dvr), 'has_mast_url'] = True

        logger.info(f'[{job_id}] Finished run for job {job_id} for {len(tics_df)} TIC IDs.')

        # # restore stdout
        # sys.stdout = sys.__stdout__

        # save job tracker to disk
        tce_tracker['job_end_time'] = pd.Timestamp.now().isoformat()
        tracker_fp = run_config['job_dir'] / f'tce_tracker_job_{job_id}.csv'
        tce_tracker.to_csv(tracker_fp, index=False)

        return {'job_id': job_id, 'success': True, 'error': None, 'traceback': None, 'tce_tracker_fp': str(tracker_fp)}

    except Exception as e:
        
        tb_str = traceback.format_exc()
        
        logger.error(f'[{job_id}] Error: {e}', exc_info=True)

        # # restore stdout
        # sys.stdout = sys.__stdout__

        return {'job_id': job_id, 'success': False, 'error': str(e), 'traceback': tb_str, 'tce_tracker_fp': None}


def run_exominer_pipeline_jobs_parallel(jobs, num_processes, max_tasks_per_child, logger):
    """ Run ExoMiner pipeline for a number of jobs.

    Args:
        jobs: list, jobs to run
        num_processes: int, number of processes to use
        max_tasks_per_child: int, maximum number of tasks per child process (after that worker is 
            killed and new one is created that needs to import all libraries and binaries)
        logger: logger object

    Returns:

    """

    ctx = mp.get_context('spawn')
    pool = ctx.Pool(processes=num_processes, maxtasksperchild=max_tasks_per_child)
    async_results = [pool.apply_async(run_exominer_pipeline, args=job) for job in jobs]
    pool.close()
    pool.join()

    for async_result in async_results:
        result_job = async_result.get()
        if not result_job['success']:
            logger.info(f'Error in job {result_job["job_id"]}: {result_job["error"]}\nTraceback {result_job["traceback"]}')
        else:
            logger.info(f'Job {result_job["job_id"]} is complete.')

def aggregate_tce_trackers(output_dir, logger):
    """ Aggregate TCE tracking tables across all jobs. 
    
    Args:
        output_dir: Path to output directory
        logger: logger
    
    Returns: pd.DataFrame containing aggregated TCE tracking information.
    """
    
    master_tracker = pd.DataFrame({
        'target_id': pd.Series(dtype='int64'),   # or 'O' for string/object
        'sector_run': pd.Series(dtype='O'),      # object/string
        'uid': pd.Series(dtype='O'),         # or 'O' for string/object
        'job_id': pd.Series(dtype='int64'),
        'job_start_time': pd.Series(dtype='O'),
        'in_tce_table': pd.Series(dtype='bool'),
        'in_tfrecord': pd.Series(dtype='bool'),
        'has_diff_img': pd.Series(dtype='bool'),
        'is_normalized': pd.Series(dtype='bool'),
        'has_prediction': pd.Series(dtype='bool'),
        'has_mast_url': pd.Series(dtype='bool')
    })

    tracker_fps = list(output_dir.glob('job_*/tce_tracker_job_*.csv'))
    logger.info(f'Found {len(tracker_fps)} TCE tracking files to aggregate.')
    
    if len(tracker_fps) > 0:
        master_tracker = pd.concat([pd.read_csv(fp) for fp in tracker_fps], axis=0, ignore_index=True)
        master_tracker_fp = output_dir / 'master_tce_tracking_summary.csv'
        master_tracker.to_csv(master_tracker_fp, index=False)
        logger.info(f'Saved master TCE tracking summary to {master_tracker_fp}.')
        
    else:
        logger.warning("No TCE tracking files found to aggregate.")
        
    return master_tracker

def summarize_tce_stages(csv_filepath):
    """
    Reads the master TCE tracking summary CSV and returns a DataFrame 
    showing the total number of TCEs that passed each stage, grouped by 
    target_id and sector_run, while retaining the job start and stop times.
    """
    # Load the CSV
    df = pd.read_csv(csv_filepath)
    
    # Define the boolean stage columns to sum
    stage_cols = [
        'in_tce_table', 
        'in_tfrecord', 
        'has_diff_img', 
        'is_normalized', 
        'has_prediction', 
        'has_mast_url'
    ]
    
    # Build the aggregation dictionary dynamically
    # Sum the booleans to get counts
    agg_funcs = {col: 'sum' for col in stage_cols}
    
    # Grab the first value of the timestamps (since they are identical per group)
    if 'job_start_time' in df.columns:
        agg_funcs['job_start_time'] = 'first'
    if 'job_stop_time' in df.columns:
        agg_funcs['job_end_time'] = 'first'
        
    # Group by target_id and sector_run, then aggregate
    summary = df.groupby(['target_id', 'sector_run']).agg(agg_funcs).reset_index()
    
    # convert the summed columns back to standard integers (pandas sum might cast to float or generic types)
    for col in stage_cols:
        if col in summary.columns:
            summary[col] = summary[col].astype(int)
    
    # # optional: Display the results in a readable format
    # print(summary.to_string(index=False))
    
    return summary


def add_metadata_to_predictions_table(predictions_tbl, save_fp, run_config, output_dir, data_collection_mode, 
                                      stellar_parameters_source, ruwe_source, task, exominer_model):

    predictions_tbl['tce_duration'] *= 24  # convert back from days to hours 

    predictions_tbl = assign_class(predictions_tbl, run_config['label_map'][task], run_config['clf_thr'][task])
    with open(run_config['predict_metadata_fp'], 'r') as f:
        predict_metadata = yaml.safe_load(f)
    
    predictions_tbl.attrs['Run'] = str(output_dir.name)
    predictions_tbl.attrs['Created'] = pd.Timestamp.now().isoformat()
    with open(run_config['pipeline_metadata_fp'], 'r') as f:
        pipeline_metadata = yaml.safe_load(f)
    for column_name, column_info in pipeline_metadata.items():
        predictions_tbl.attrs[column_name] = column_info
    predictions_tbl.attrs['Task'] = task
    predictions_tbl.attrs['ExoMiner Model'] = exominer_model
    predictions_tbl.attrs['Label Map'] = run_config['label_map'][task]
    predictions_tbl.attrs['Data Collection Mode'] = data_collection_mode
    predictions_tbl.attrs['Stellar Parameters Source'] = stellar_parameters_source
    predictions_tbl.attrs['RUWE Source'] = ruwe_source
    predictions_tbl.attrs['Classification Threshold'] = run_config['clf_thr'][task]
    predictions_tbl.attrs['Disposition Definitions'] = '========================'
    for cls in run_config['label_map'][task]:
        predictions_tbl.attrs[cls] = run_config['classes_descriptions'][cls]
    predictions_tbl.attrs['Column Definitions'] = '========================'
    for column_name, column_info in predict_metadata.items():
        if column_name in predictions_tbl.columns or column_name in ['score/score_<disp>', 'score_std/score_std_<disp>']:
            predictions_tbl.attrs[f'Column: {column_name}'] = column_info
    with open(save_fp, 'w') as f:
        for attr_key, attr_value in predictions_tbl.attrs.items():
            f.write(f'# {attr_key}: {attr_value}\n')
        predictions_tbl.to_csv(f, index=False)
    

def aggregate_predictions_across_jobs(output_dir, run_config, task, exominer_model, data_collection_mode, stellar_parameters_source, ruwe_source, logger):
    """ Aggregate predictions tables across jobs.

    Args:
        output_dir: pathlib Path, output directory
        run_config: dict, pipeline run configuration parameters
        exominer_model: str, ExoMiner model type
        data_collection_mode: str, data collection mode; either '2min' of 'ffi'
        stellar_parameter_source: str, stellar parameters source
        ruwe_source: str, RUWE source
        logger: logger object
    """

    predictions_tbl_fp = output_dir / f'predictions_{output_dir.name}.csv'
    logger.info(f'Aggregating predictions into a single table in {predictions_tbl_fp.name}...')
    predictions_tbls_fps = list(output_dir.glob('job_*/predictions/*.csv'))
    logger.info(f'Found {len(predictions_tbls_fps)} job predictions files.')
    if len(predictions_tbls_fps) > 0:
        predictions_tbl = pd.concat([pd.read_csv(fp, comment='#') for fp in predictions_tbls_fps], axis=0, ignore_index=True)

        add_metadata_to_predictions_table(predictions_tbl, predictions_tbl_fp, run_config, output_dir, 
                                          data_collection_mode, stellar_parameters_source, ruwe_source, 
                                          task, exominer_model)
        
        logger.info(f'Saved predictions to {predictions_tbl_fp}.')
        
    else:
        logger.info(f'No predictions CSV file generated. See logs for more information.')


def aggregate_spoc_dv_reports(output_dir, run_config, data_collection_mode, logger):
    """ Aggregate table with MAST URLs for the SPOC DV reports for the TCEs/targets.

    Args:
        output_dir: pathlib Path, output directory
        run_config: dict, pipeline run configuration parameters
        data_collection_mode: str, data collection mode; either '2min' of 'ffi'
        logger: logger object
    """
        
    dv_reports_tbl_fp = output_dir / f'dv_reports_all_jobs.csv'
    logger.info(f'Aggregating CSVs with TESS SPOC DV reports URLs  across all jobs into a single table in '
                f'{dv_reports_tbl_fp}...')
    dv_reports_tbls_fps = list(Path(output_dir).glob('job_*/mast_urls_tables/spoc-dv_mast-urls_job*.csv'))
    if len(dv_reports_tbls_fps) > 0:
        dv_reports_tbl = pd.concat([pd.read_csv(fp) for fp in dv_reports_tbls_fps], axis=0, ignore_index=True)
        with open(run_config['dv_reports_metadata_fp'], 'r') as f:
            dv_reports_metadata = yaml.safe_load(f)
        dv_reports_tbl.attrs['Run'] = str(output_dir.name)
        dv_reports_tbl.attrs['Created'] = pd.Timestamp.now().isoformat()
        dv_reports_tbl.attrs['Data Collection Mode'] = data_collection_mode
        dv_reports_tbl.attrs['Column Definitions'] = '========================'
        for column_name, column_info in dv_reports_metadata.items():
            if column_name in dv_reports_tbl.columns:
                dv_reports_tbl.attrs[f'Column: {column_name}'] = column_info
        with open(dv_reports_tbl_fp, 'w') as f:
            for attr_key, attr_value in dv_reports_tbl.attrs.items():
                f.write(f'# {attr_key}: {attr_value}\n')
            dv_reports_tbl.to_csv(f, index=False)
    else:
        logger.info(f'No DV reports CSV file generated. See logs for more information.')


def run_exominer_pipeline_main(output_dir, tic_ids_fp, data_collection_mode, tic_ids=None, num_processes=1,
                               num_jobs=1, get_mast_urls_dv_reports='false', dv_xml_data_repository=None, lc_data_repository=None,
                               stellar_parameters_source='ticv8', ruwe_source='gaiadr2', task='phot-vetting', exominer_model='single',
                               plot_inputs_to_model=False, max_model_workers=1):
    """ Run ExoMiner pipeline.

    Args:
        output_dir: str, directory to save the output of the run.
        tic_ids_fp: str, filepath to the TIC IDs file for the run.
        data_collection_mode: str, either '2min' or 'ffi'.
        tic_ids: str, list of TIC IDs to process. Only used if `tic_ids_fp` is None.
        num_processes: int, number of processes to use.
        num_jobs: int, number of jobs to run in parallel.
        get_mast_urls_dv_reports: str, whether to create CSV file with URLs to SPOC data products for TCEs of
            queried TICs.
        dv_xml_data_repository: str, the data repository to use for DV XML files for queried TICs.
        lc_data_repository: str, the data repository to use for light curve FITS files for queried TICs.
        stellar_parameters_source: str, the stellar parameters source to use for the queried TICs. Set to either
            'ticv8', 'tess-spoc', or filepath to external catalog of stellar parameters for the queried TICs.
        ruwe_source: str, the RUWE source to use for the queried TICs. Set to either 'gaiadr2', 'gaiadr3', 'gaiaedr3', 'unavailable', or
            filepath to external catalog of RUWE values for the queried TICs.
        task: str, choose between classification task "phot-vetting" and "planet-validation"
        exominer_model: str, which ExoMiner model to use for inference. Choose among "single", "cv_ensemble", and "full_cv_ensemble", 
            or provide the filepath to a TensorFlow Keras model that is compatible with the pipeline
        plot_inputs_to_model: bool, if True saves plots of the preprocessed inputs to the ExoMiner model for each TCE in the run
        max_model_workers: int, max number of processes to use for running inference in parallel

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
    print('\n######################\nDocumentation can be found at: '
          'https://github.com/nasa/Exominer/tree/main/docs/index.md\n######################')
    logger.info('\n######################\nDocumentation can be found at: '
                'https://github.com/nasa/Exominer/tree/main/docs/index.md\n######################')
    print(f'Started run for ExoMiner...')
    print(f'Logging information to run_main.log.')

    logger.info(f'Preparing inputs and adjusting configuration file...')
    run_config, tics_df = process_inputs(output_dir, CONFIG_FP, tic_ids_fp, data_collection_mode, logger,
                                         tic_ids=tic_ids, 
                                         num_processes=num_processes, 
                                         num_jobs=num_jobs,
                                         get_mast_urls_dv_reports=get_mast_urls_dv_reports,
                                         dv_xml_data_repository=dv_xml_data_repository,
                                         lc_data_repository=lc_data_repository,
                                         stellar_parameters_source=stellar_parameters_source,
                                         ruwe_source=ruwe_source,
                                         task=task,
                                         exominer_model=exominer_model,
                                         max_model_workers=max_model_workers,
                                         )
    logger.info('Done.')

    start_time = pd.Timestamp.now()
    run_config['Start Time'] = start_time.isoformat()
    
    # Validate structure of TIC IDs CSV file
    logger.info('Validating TIC IDs CSV file structure...')
    validate_tic_ids_flag = validate_tic_ids_csv_structure(tics_df, logger)
    if not validate_tic_ids_flag:
        raise SystemExit('TIC IDs CSV file structure validation failed.')
    else:
        logger.info('TIC IDs CSV file structure validation completed.')

    output_tics_tbl_fp = output_dir / 'tics_tbl.csv'
    tics_df.to_csv(output_tics_tbl_fp, index=False)
    logger.info(f'Found {len(tics_df)} TIC IDs. Saving TIC IDs to {str(output_tics_tbl_fp.name)}...')

    logger.info(f'Checking validity of configuration file...')
    check_config(run_config, logger)
    logger.info('Done checking configuration file.')

    run_config['plot_inputs_to_exominer_model'] = plot_inputs_to_model

    # copy config files
    config_dir = output_dir / 'config_files'
    config_dir.mkdir(exist_ok=True)
    config_files_k_lst = [
        'lc_preprocessing_config_fp',
        'diff_img_preprocessing_config_fp',
        # 'diff_img_add_tfrecord_dataset_config_fp',
        'normalize_tfrec_data_config_fp',
        'predict_config_fp',
    ]
    for config_file_k in config_files_k_lst:
        new_config_fp = config_dir / run_config[config_file_k].split('/')[-1]
        shutil.copy(run_config[config_file_k], new_config_fp)
        run_config[config_file_k] = new_config_fp
        
    # update light curve preprocessing config
    with open(run_config['lc_preprocessing_config_fp'], 'r') as f:
        lc_config = yaml.safe_load(f)

    lc_config['data_collection_mode'] = data_collection_mode
    lc_config['plot_inputs_to_exominer_model'] = plot_inputs_to_model

    # update difference image preprocessing config
    with open(run_config['diff_img_preprocessing_config_fp'], 'r') as f:
        diffimg_config = yaml.safe_load(f)

    # only create directory and add 'plot_dir' if True
    run_config['plot_inputs_to_model'] = plot_inputs_to_model
    if plot_inputs_to_model:
        preproc_plot_dir = output_dir / 'plot_inputs_to_model'
        preproc_plot_dir.mkdir(exist_ok=True, parents=True)
        run_config['plot_dir'] = preproc_plot_dir
        lc_config['plot_dir'] = str(preproc_plot_dir)
        diffimg_config['plot_prob'] = 1
        diffimg_config['plot_dir'] = preproc_plot_dir
    
    with open(run_config['normalize_tfrec_data_config_fp'], 'r') as f:
        norm_config = yaml.safe_load(f)
    norm_config['normStats'] = norm_config['normStats'][task]

    with open(run_config['lc_preprocessing_config_fp'], 'w') as f:
        yaml.dump(lc_config, f, sort_keys=False)
    
    with open(run_config['diff_img_preprocessing_config_fp'], 'w') as f:
        yaml.dump(diffimg_config, f, sort_keys=False)
    
    with open(run_config['normalize_tfrec_data_config_fp'], 'w') as f:
        yaml.dump(norm_config, f, sort_keys=False)

    with open(output_dir / 'pipeline_run_config.yaml', 'w') as f:
        yaml.dump(run_config, f, sort_keys=False)

    print(f'Splitting TIC IDs across {run_config["num_jobs"]} job(s) for parallel processing using '
          f'{run_config["num_processes"]} process(es)...')
    print(f'The results for each job are saved into their own directories under {output_dir.name} '
          f'following pattern job_{{job_id}}.')

    # split TIC IDs across jobs
    indices = np.array_split(np.arange(len(tics_df)), run_config['num_jobs'])
    tics_df_jobs = [tics_df.iloc[idx] for idx in indices]
    jobs = [(run_config, tics_job, job_id) for job_id, tics_job in enumerate(tics_df_jobs)]
    logger.info(f'Split TIC IDs into {run_config["num_jobs"]} jobs to be processed in parallel using '
                f'{run_config["num_processes"]} '
                f'process(es).')
    logger.info(f'Started running ExoMiner pipeline on {run_config["num_processes"]} process(es) for '
                f'{run_config["num_jobs"]} job(s)...')
    run_exominer_pipeline_jobs_parallel(jobs, run_config['num_processes'], run_config['max_tasks_per_child'], logger)

    print(f'Finished running preprocessing pipeline on {run_config["num_processes"]} process(es) for '
          f'{run_config["num_jobs"]} job(s).')

    print(f'Aggregating TCE tracking summaries across all jobs...')
    master_tracker = aggregate_tce_trackers(output_dir, logger)

    # load trained model and run inference
    logger.info('Running inference on TCEs in the TFRecord dataset using trained ExoMiner model...')
    print(f'Running inference on TCEs in the TFRecord dataset using trained ExoMiner model...')
    # find all files starting with "shard-tess" inside any "tfrecord_data_diffimg_normalized" folder
    tfrec_files = list(output_dir.rglob('**/tfrecord_data_diffimg_normalized/shard-tess*'))    
    inference_pipeline(run_config, output_dir, tfrec_files, logger)
    pred_tbl_fp = output_dir / 'predictions_predictset.csv'
    if pred_tbl_fp.exists():
        pred_tbl = pd.read_csv(pred_tbl_fp, comment='#')
        add_metadata_to_predictions_table(pred_tbl, pred_tbl_fp, run_config, output_dir, 
                                          data_collection_mode, stellar_parameters_source, ruwe_source, 
                                          task, exominer_model)
        logger.info(f'Saved predictions to {pred_tbl_fp}.')
        
        if len(pred_tbl) > 0 and 'uid' in pred_tbl.columns:
            master_tracker.loc[master_tracker['uid'].isin(pred_tbl['uid']), 'has_prediction'] = True
            master_tracker.to_csv(output_dir / 'master_tce_tracking_summary.csv', index=False)
    else:
        logger.info(f'No predictions CSV file generated. See logs for more information.')

    # print a quick summary to the console/log
    logger.info("Pipeline Attrition Summary:")
    logger.info(f"Total TCEs initially found: {master_tracker['in_tce_table'].sum()}")
    logger.info(f"TCEs successfully written to TFRecords: {master_tracker['in_tfrecord'].sum()}")
    logger.info(f"TCEs with Diff Images added: {master_tracker['has_diff_img'].sum()}")
    logger.info(f"TCEs normalized: {master_tracker['is_normalized'].sum()}")
    logger.info(f"TCEs with scores: {master_tracker['has_prediction'].sum()}")

    # summarize tracking per target id and sector run
    print(f'Create TIC tracking summaries...')
    tic_tracker_df = summarize_tce_stages(output_dir / 'master_tce_tracking_summary.csv')
    tic_tracker_df.to_csv(output_dir / 'master_tic_tracking_summary.csv', index=False)

    if get_mast_urls_dv_reports == 'true':
        print(f'Aggregating CSVs with TESS SPOC DV reports URLs  across all jobs into a single table...')
        aggregate_spoc_dv_reports(output_dir, run_config, data_collection_mode, logger)
    
    end_time = pd.Timestamp.now()
    run_config['End Time'] = end_time.isoformat()
    spent_time = end_time - start_time
    run_config['Spent Time'] = str(spent_time)

    with open(output_dir / 'pipeline_run_config.yaml', 'w') as f:
        yaml.dump(run_config, f, sort_keys=False)

    logger.info(f'Finished running ExoMiner pipeline in {spent_time} ({spent_time.total_seconds()} seconds).')
    print(f'Finished running ExoMiner pipeline in {spent_time} ({spent_time.total_seconds()} seconds).')


if __name__ == "__main__":

    # Execute before any other TensorFlow logic
    force_cpu_inference()

    parser = argparse.ArgumentParser(description='ExoMiner Pipeline: A tool for transit-signal vetting for the TESS '
                                                 'Mission. For more information see NASA\'s GitHub repository at '
                                                 'https://github.com/nasa/Exominer/tree/main/docs/index.md')

    parser.add_argument('--output_dir', type=str, help='Output directory the results are saved into.',
                        default=None)
    parser.add_argument('--tic_ids_fp', type=str, help='Filepath to CSV file containing the TIC IDs and '
                                                       'the corresponding sector run. Must include header with columns '
                                                       '"tic_id" and "sector_run". Sector run should be provided as '
                                                       'the '
                                                       'start and end sectors of the run '
                                                       '({start_sector}-{end_sector}). If this argument is '
                                                       'set, it takes preference over argument '
                                                       '--tic_ids ', default=None)
    parser.add_argument('--tic_ids', type=str, help='Comma-separated list of TIC IDs with option for '
                                                    'including the corresponding sector run. The accepted format is '
                                                    '{tic_id_a}_{sector_run},{tic_id_b}-{sector_run}, ..., .'
                                                    'Alternative to --tic_ids_fp. This argument is ignored if '
                                                    '--tic_ids_fp is set.',
                        default=None)
    parser.add_argument('--data_collection_mode', type=str, help='Either "2min" of "FFI" to process '
                                                                 'TESS SPOC 2-min or FFI TCE data. By default, it '
                                                                 'is set to "2min".',
                        default="2min")
    parser.add_argument('--num_processes', type=int, help='Number of processes to use for '
                                                          'parallelization. '
                                                          'Set to "1" by default.', default=-1)
    parser.add_argument('--num_jobs', type=int, help='Number of jobs to split the TIC IDs through. '
                                                     'Set to "1" by default.', default=1)

    parser.add_argument('--get_mast_urls_dv_reports', type=str, help='Set to "true" to download a CSV '
                                                                        'file containing the URLs to the TESS SPOC DV '
                                                                        'reports for the TCEs of the queried TICs in the '
                                                                        'MAST.'
                                                                        'Set to "false" by default.',
                        default='false', choices=['true', 'false'])

    # parser.add_argument('--external_data_repository', type=str, help='Provide the path to a directory '
    #                                                                  'containing the light curve FITS files and DV XML '
    #                                                                  'files for the TIC IDs and sector runs that you '
    #                                                                  'want to query. Otherwise, set to "null" so the '
    #                                                                  'pipeline downloads the required files from the '
    #                                                                  'MAST. By default, it is set to "null".',
    #                     default=None)

    parser.add_argument('--dv_xml_data_repository', type=str, help='Provide the path to a directory '
                                                                    'containing the DV XML '
                                                                    'files for the TIC IDs and sector runs that you '
                                                                    'want to query. Otherwise, set to "null" so the '
                                                                    'pipeline downloads the required files from the '
                                                                    'MAST. By default, it is set to `None`.',
                        default=None)
    parser.add_argument('--lc_data_repository', type=str, help='Provide the path to a directory '
                                                                'containing the light curve FITS '
                                                                'files for the TIC IDs and sector runs that you '
                                                                'want to query. Otherwise, set to "null" so the '
                                                                'pipeline downloads the required files from the '
                                                                'MAST. By default, it is set to `None`.',
                        default=None)

    parser.add_argument('--stellar_parameters_source', type=str, help='Provide the path to a CSV file '
                                                                     'containing the stellar parameters for the queried '
                                                                      'TICs. Set to "ticv8" to query TIC-8 from the '
                                                                      'MAST. Set to "tess-spoc" to use the values '
                                                                      'provided in DV (no stellar mass provided, which '
                                                                      'is set to 1). '
                                                                      'By default, this argument is set to "ticv8".',
                        default='ticv8')

    parser.add_argument('--ruwe_source', type=str, help='Provide the path to a CSV file '
                                                                     'containing the Gaia RUWE value for the queried '
                                                                      'TICs. Set to "gaiadr2" to query Gaia DR2. '
                                                                      'MAST. Set to "unavailable" to set RUWE to missing'
                                                                      ' value.'
                                                                      'By default, this argument is set to "gaiadr2".',
                        default='gaiadr2')
    
    parser.add_argument('--task', type=str, help='Specify which classification task to run. You can choose between "phot-vetting" and ' \
                        '"planet-validation"', default='phot-vetting')
    
    parser.add_argument('--exominer_model', type=str, help='Specify which ExoMiner model to use for inference. Currently, '
                        'you can choose among "single", "cv_ensemble", and "full_cv_ensemble",' 
                        'or provide the filepath to a TensorFlow Keras model that is compatible with the pipeline.',
                        default='single')

    parser.add_argument('--plot_inputs_to_model', action='store_true', help='If set, saves figures of plots of the preprocessed inputs given to the ' \
    'ExoMiner model for each TCE in the run.')


    parser.add_argument('--max_model_workers', type=int, help='Specify how many workers to use for running inference in parallel.',
                        default=1)
    

    parsed_args = parser.parse_args()

    run_exominer_pipeline_main(parsed_args.output_dir, 
                               parsed_args.tic_ids_fp,
                               parsed_args.data_collection_mode, 
                               parsed_args.tic_ids, 
                               parsed_args.num_processes,
                               parsed_args.num_jobs, 
                               parsed_args.get_mast_urls_dv_reports,
                               parsed_args.dv_xml_data_repository, 
                               parsed_args.lc_data_repository,
                               parsed_args.stellar_parameters_source,
                               parsed_args.ruwe_source,
                               parsed_args.task,
                               parsed_args.exominer_model,
                               parsed_args.plot_inputs_to_model,
                               parsed_args.max_model_workers,
                               )
