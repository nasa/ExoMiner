""" Run the ExoMiner Pipeline Podman application for different TICs tables."""

# 3rd party
import subprocess
import datetime
from pathlib import Path
import logging
import datetime
import shutil
import os

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

tics_tbls_fps = list(Path(inputs_dir).glob("test*.csv"))

logger.info(f"Found {len(tics_tbls_fps)} TICs tables in {inputs_dir}.")

run_podman_sh_script_fp = Path('/data3/exoplnt_dl/codebase/exominer_pipeline/run_podman_application.sh')

data_collection_mode = '2min'
num_processes = 1
num_jobs = 1
download_spoc_data_products = 'true'
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

    # Run the shell script with the specified arguments
    subprocess.run([
        str(run_podman_sh_script_fp),
        "--tics_tbl_fp", str(fp),
        "--exominer_pipeline_run_dir", str(run_dir),
        "--data_collection_mode", data_collection_mode,
        "--num_processes", str(num_processes),
        "--num_jobs", str(num_jobs),
        "--download_spoc_data_products", download_spoc_data_products,
        "--external_data_repository", external_data_repository,
        "--stellar_parameters_source", stellar_parameters_source,
        "--ruwe_source", ruwe_source,
    ])
    
    logger.info(f"Run completed for {fp.name}. Results saved to {run_dir}.")
    
    if delete_data_after_run:
        # delete jobs directories and other data after the run; keep aggregated predictions only
        logger.info(f"Cleaning up job directories and data in {run_dir}.")
        (run_dir / fp.name).unlink()  # delete the input table from inside the run directory
        job_dirs = run_dir.glob("job_*")    
        for job_dir in job_dirs:
            shutil.rmtree(job_dir)
        logger.info(f"Cleanup complete for {run_dir}.")
    