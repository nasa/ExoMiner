"""
Main script used to generate TFRecords datasets.
"""

# 3rd party
from mpi4py import MPI
import socket
from pathlib import Path
import multiprocessing
import numpy as np
import yaml
import argparse
import logging

# local
from src_preprocessing.lc_preprocessing.utils_generate_input_records import get_tce_table
from utils.utils_dataio import is_yamlble
from src_preprocessing.lc_preprocessing.utils_manipulate_tfrecords import create_shards_table
from src_preprocessing.lc_preprocessing.utils_generate_input_records import process_file_shard, create_shards
from src_preprocessing.lc_preprocessing.utils_preprocessing_io import is_pfe

logger = logging.getLogger(__name__)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--rank', type=int, help='Rank', default=-1)
    parser.add_argument('--n_runs', type=int, help='Total number of runs', default=-1)
    parser.add_argument('--output_dir', type=str,
                        help='File path output directory for this preprocessing run',
                        default=None)
    parser.add_argument('--config_fp', type=str,
                        help='File path to yaml config file for this preprocessing run',
                        default='./config_preprocessing.yaml')
    args = parser.parse_args()

    # get the configuration parameters
    path_to_yaml = Path(args.config_fp).resolve()
    with(open(path_to_yaml, 'r')) as file:
        config = yaml.safe_load(file)

    if args.output_dir is not None:
        config['output_dir'] = Path(args.output_dir)
    else:
        config['output_dir'] = Path(config['output_dir'])

    # make the output directory if it doesn't already exist
    config['output_dir'].mkdir(exist_ok=True)

    # set up parameters
    config['bin_width_factor_glob'] = 1 / config['num_bins_glob']
    # config['tess_to_kepler_px_scale_factor'] = config['tess_px_scale'] / config['kepler_px_scale']
    config['tce_min_err']['tce_duration'] = config['tce_min_err']['tce_duration'] / 24
    # config['primary_buffer_time'] = (config['primary_buffer_nsamples'] /
    #                                  config['sampling_rate_h'][config['satellite']] / 24)

    if config['using_mpi']:  # using some sort of external library for parallelization
        if args.rank != -1:  # using parallel
            config['process_i'] = args.rank
            config['n_shards'] = args.n_runs
            config['n_processes'] = args.n_runs
        else:  # using mpi
            config['process_i'] = MPI.COMM_WORLD.rank
            config['n_shards'] = MPI.COMM_WORLD.size
            config['n_processes'] = MPI.COMM_WORLD.size

    if is_pfe():
        # get node id
        config['node_id'] = socket.gethostbyname(socket.gethostname()).split('.')[-1]

    # create logger
    config['preprocessing_logs_dir'] = config['output_dir'] / 'preprocessing_logs'
    config['preprocessing_logs_dir'].mkdir(exist_ok=True)
    logging.basicConfig(filename=config['output_dir'] / 'preprocessing_logs' /
                                 f'preprocessing_{config["process_i"]}.log',
                        level=logging.INFO,
                        format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        filemode='a',
                        )

    logger.info(f'Process shard {config["process_i"]} ({config["n_shards"]} total shards)')

    # make directory for exclusion logs
    config['exclusion_logs_dir'] = config['output_dir'] / 'exclusion_logs'
    config['exclusion_logs_dir'].mkdir(exist_ok=True)

    # make directory to save figures in different steps of the preprocessing pipeline
    if config['plot_figures']:
        config['plot_dir'] = config['output_dir'] / 'plots'
        config['plot_dir'].mkdir(exist_ok=True)

    # get TCE and gapping ephemeris tables
    shards_tce_tables, tce_table = get_tce_table(config)

    if config['process_i'] in [0, -1]:
        np.save(config['output_dir'] / 'preprocessing_params.npy', config)
        # save the YAML file with preprocessing parameters that are YAML serializable
        json_dict = {key: val for key, val in config.items() if is_yamlble(val)}
        with open(config['output_dir'] / 'preprocessing_params.yaml', 'w') as preproc_run_file:
            yaml.dump(json_dict, preproc_run_file)

    if config['using_mpi']:  # using some sort of external library for parallelization

        node_id = socket.gethostbyname(socket.gethostname()).split('.')[-1]
        shard_filename = f'shard-{config["process_i"]:05d}-of-{config["n_shards"]:05d}-node-{node_id:s}'
        shard_fp = config['output_dir'] / shard_filename

        logger.info(f'Started processing {len(shards_tce_tables[config["process_i"]])} items in shard {shard_filename}')

        process_file_shard(shards_tce_tables[config['process_i']], shard_fp, tce_table, config)

        logger.info(f'Finished processing {len(shards_tce_tables[config["process_i"]])} items in shard '
                    f'{shard_filename}')

    else:  # use multiprocessing.Pool

        file_shards = create_shards(config, shards_tce_tables, tce_table)

        # launch subprocesses for the file shards
        logger.info(f'Launching {config["n_processes"]} processes for {config["n_shards"]} total file shards.')

        pool = multiprocessing.Pool(processes=config['n_processes'])
        async_results = [pool.apply_async(process_file_shard, file_shard) for file_shard in file_shards]
        pool.close()

        # async_result.get() to ensure any exceptions raised by the worker processes are raised here
        for async_result in async_results:
            async_result.get()

        logger.info(f'Finished processing {config["n_shards"]} total file shards.')

        # concatenates shard tables into a single one
        create_shards_tbl_flag = create_shards_table(config['output_dir'])
        if not create_shards_tbl_flag:
            logger.info('Merged shard table not created.')


if __name__ == "__main__":

    main()
