""" Run the ExoMiner Pipeline for different TICs tables."""

# 3rd party
# import subprocess
import datetime
from pathlib import Path
import logging
import datetime
import shutil
import pandas as pd

# local
from exominer_pipeline.run_pipeline import run_exominer_pipeline_main

#%% setup

runs_root_dir = Path('/data3/exoplnt_dl/experiments/exominer_pipeline/runs')

# create logger
logger = logging.getLogger(name=f'run_main.log')
logger_handler = logging.FileHandler(filename=runs_root_dir / f'exominer_pipeline_runs_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.log', mode='w')
logger_formatter = logging.Formatter('%(asctime)s - %(message)s')
logger.setLevel(logging.INFO)
logger_handler.setFormatter(logger_formatter)
logger.addHandler(logger_handler)

# base directory for inputs
inputs_dir = "/data3/exoplnt_dl/experiments/exominer_pipeline/inputs"

tics_tbls_fps = list(Path(inputs_dir).glob("tics*.csv"))

process_target_sectors_fp = Path('/data3/exoplnt_dl/experiments/exominer_pipeline/runs/target_sector_run_pairs_processed_8-25-2025_1559.csv')
if process_target_sectors_fp is not None:
    process_target_sectors = pd.read_csv(process_target_sectors_fp)
    process_target_sectors.rename(columns={'target_id': 'tic_id'}, inplace=True)
    logger.info(f"Read {process_target_sectors_fp} to exclude already processed target/sector run pairs.")

logger.info(f"Found {len(tics_tbls_fps)} TICs tables in {inputs_dir}.")

run_pipeline_sh_script_fp = Path('/data3/exoplnt_dl/codebase/exominer_pipeline/run_pipeline.sh')
run_pipeline_python_script_fp = Path('/data3/exoplnt_dl/codebase/exominer_pipeline/run_pipeline.py')

data_collection_mode = '2min'
num_processes = 8
num_jobs = 32
download_spoc_data_products = 'false'
external_data_repository = 'null'
stellar_parameters_source = 'ticv8'
ruwe_source = 'gaiadr2'
# whether to delete job directories and other data after the run; keep aggregated predictions only
delete_data_after_run = False  

#%% Iterate over each TICs table and run the shell script

for fp_i, fp in enumerate(tics_tbls_fps):
    
    logger.info(f"Processing {fp_i + 1}/{len(tics_tbls_fps)}: {fp.name}")
    
    # generate a unique run name using the table name and timestamp
    timestamp = datetime.datetime.now().strftime("%m-%d-%Y_%H%M%S")
    run_name = f"exominer_pipeline_run_{fp.stem}_{timestamp}"
    run_dir = runs_root_dir / run_name
    
    if process_target_sectors_fp is not None:
        tics_tbl = pd.read_csv(fp)
        # exclude already processed target/sector run pairs
        common_pairs = tics_tbl.merge(process_target_sectors, on=['tic_id', 'sector_run'], how='inner')
        logger.info(f"Excluding {len(common_pairs)}/{len(tics_tbl)} already processed target/sector run pairs from {fp.name}.")
        filtered_tics_tbl = tics_tbl[~tics_tbl.set_index(['tic_id', 'sector_run']).index.isin(common_pairs.set_index(['tic_id', 'sector_run']).index)]
        if len(filtered_tics_tbl) == 0:
            logger.info(f"All target/sector run pairs in {fp.name} have already been processed. Skipping this run...")
            continue
        # Save the result if needed
        run_dir.mkdir(parents=True, exist_ok=True)
        filtered_tics_tbl_fp = run_dir / 'tics_tbl.csv'
        filtered_tics_tbl.to_csv(filtered_tics_tbl_fp, index=False)
        fp = filtered_tics_tbl_fp

    # # Run the shell script with the specified arguments
    # subprocess.run([
    #     str(run_pipeline_sh_script_fp),
    #     "--pipeline_python_script_fp", str(run_pipeline_python_script_fp),
    #     "--tics_tbl_fp", str(fp),
    #     "--exominer_pipeline_run_dir", str(run_dir),
    #     "--data_collection_mode", data_collection_mode,
    #     "--num_processes", str(num_processes),
    #     "--num_jobs", str(num_jobs),
    #     "--download_spoc_data_products", download_spoc_data_products,
    #     "--external_data_repository", external_data_repository,
    #     "--stellar_parameters_source", stellar_parameters_source,
    #     "--ruwe_source", ruwe_source,
    # ])
    run_exominer_pipeline_main(
        output_dir=str(run_dir),
        tic_ids_fp=str(fp),
        data_collection_mode=data_collection_mode,
        num_processes=num_processes,
        num_jobs=num_jobs,
        download_spoc_data_products=download_spoc_data_products,
        external_data_repository=None if external_data_repository == 'null' else external_data_repository,
        stellar_parameters_source=stellar_parameters_source,
        ruwe_source=ruwe_source,
    )
    
    logger.info(f"Run completed for {fp.name}. Results saved to {run_dir}.")
    
    if delete_data_after_run:
        # delete jobs directories and other data after the run; keep aggregated predictions only
        logger.info(f"Cleaning up job directories and data in {run_dir}.")
        (run_dir / fp.name).unlink()  # delete the input table from inside the run directory
        job_dirs = run_dir.glob("job_*")    
        for job_dir in job_dirs:
            shutil.rmtree(job_dir)
        logger.info(f"Cleanup complete for {run_dir}.")
