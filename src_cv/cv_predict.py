""" Run inference using cross-validation experiment. """

# 3rd party
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path
import numpy as np
import logging
# from datetime import datetime
import tensorflow as tf
# from tensorflow.keras import callbacks
# import itertools
# import shutil
import time
import argparse
import sys
from mpi4py import MPI
import multiprocessing
import yaml

# local
from models.models_keras import ExoMiner
from src_hpo import utils_hpo
from utils.utils_dataio import is_yamlble
from src_cv.utils_cv import predict_ensemble, normalize_data
from paths import path_main


def cv_pred_run(run_params, cv_iter_dir):
    """ Run one iteration of CV.

    :param run_params: dict, configuration parameters for the CV ru
    :param cv_iter_dir: Path, CV iteration directory.

    :return:
    """

    cv_run_dir = run_params['paths']['experiment_dir'] / f'cv_iter_{run_params["cv_id"]}'
    cv_run_dir.mkdir(exist_ok=True)

    # process data before feeding it to the model (e.g., normalize data based on training set statistics
    if run_params['logger'] is not None:
        run_params['logger'].info(f'[cv_iter_{run_params["cv_id"]}] Normalizing data for CV iteration')

    norm_data_dir = cv_run_dir / 'norm_data'
    norm_data_dir.mkdir(exist_ok=True)

    # load normalization statistics
    norm_stats_dir = cv_iter_dir / 'norm_stats'
    norm_stats = {
        'scalar_params': np.load(norm_stats_dir / 'train_scalarparam_norm_stats.npy', allow_pickle=True).item(),
        'fdl_centroid': np.load(norm_stats_dir / 'train_fdlcentroid_norm_stats.npy', allow_pickle=True).item(),
        'centroid': np.load(norm_stats_dir / 'train_centroid_norm_stats.npy', allow_pickle=True).item()
    }
    # norm_stats['scalar_params']['tce_steff']['info']['dtype'] = 'float'
    # norm_stats['scalar_params']['tce_rb_tcount0']['info']['dtype'] = 'float'

    pool = multiprocessing.Pool(processes=run_params['n_processes_norm_data'])
    jobs = [(file, norm_stats, run_params['aux_params'], norm_data_dir)
            for file in run_params['data_fps']]
    async_results = [pool.apply_async(normalize_data, job) for job in jobs]
    pool.close()
    for async_result in async_results:
        async_result.get()

    data_shards_fps_norm = {'predict': [norm_data_dir / file.name for file in run_params['data_fps']]}

    if run_params['logger'] is not None:
        run_params['logger'].info(f'[cv_iter_{run_params["cv_id"]}] Running inference')
    # get the filepaths for the trained models
    models_dir = cv_iter_dir / 'models'
    models_filepaths = [model_dir / f'{model_dir.stem}.h5' for model_dir in models_dir.iterdir() if 'model' in
                        model_dir.stem]

    # evaluate ensemble=
    p = multiprocessing.Process(target=predict_ensemble,
                                args=(
                                    models_filepaths,
                                    run_params,
                                    data_shards_fps_norm,
                                    cv_run_dir,
                                ))
    p.start()
    p.join()

    # logger.info(f'[cv_iter_{run_params["cv_id"]}] Deleting normalized data')
    # # remove preprocessed data for this run
    # shutil.rmtree(cv_run_dir / 'norm_data')
    # # TODO: delete the models as well?


def cv_pred():

    path_to_yaml = Path(path_main + 'src_cv/config_cv_predict.yaml')
    with(open(path_to_yaml, 'r')) as file:
        config = yaml.safe_load(file)

    # used in job arrays
    parser = argparse.ArgumentParser()
    parser.add_argument('--job_idx', type=int, help='Job index', default=0)
    args = parser.parse_args()

    # uncomment for MPI multiprocessing
    rank = MPI.COMM_WORLD.rank
    config['rank'] = config['ngpus_per_node'] * args.job_idx + rank
    config['size'] = MPI.COMM_WORLD.size
    print(f'Rank = {config["rank"]}/{config["size"] - 1}')
    sys.stdout.flush()
    if config['rank'] != 0:
        time.sleep(2)

    try:
        print(f'[rank_{config["rank"]}] CUDA DEVICE ORDER: {os.environ["CUDA_DEVICE_ORDER"]}')
        print(f'[rank_{config["rank"]}] CUDA VISIBLE DEVICES: {os.environ["CUDA_VISIBLE_DEVICES"]}')
    except:
        print(f'[rank_{config["rank"]}] No CUDA environment variables exist.')

    # n_gpus = len(os.environ["CUDA_VISIBLE_DEVICES"].split(','))  # number of GPUs visible to the process
    if config["rank"] == 0:
        print(f'Number of GPUs selected per node = {config["ngpus_per_node"]}')
    config['gpu_id'] = config["rank"] % config['ngpus_per_node']

    # setting GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config['gpu_id'])  # "0, 1"

    print(f'[rank_{config["rank"]}] CUDA DEVICE ORDER: {os.environ["CUDA_DEVICE_ORDER"]}')
    print(f'[rank_{config["rank"]}] CUDA VISIBLE DEVICES: {os.environ["CUDA_VISIBLE_DEVICES"]}')

    # tf.debugging.set_log_device_placement(True)

    for path_name, path_str in config['paths'].items():
        config['paths'][path_name] = Path(path_str)
    config['paths']['experiment_dir'].mkdir(exist_ok=True)

    config['data_fps'] = [fp for fp in config['paths']['tfrec_dir'].iterdir() if fp.is_file()
                          and fp.name.startswith('predict-shard')]
    # cv iterations dictionary
    config['cv_iters'] = [fp for fp in config['paths']['cv_experiment_dir'].iterdir() if fp.is_dir()
                          and fp.name.startswith('cv_iter')]

    if config["rank"] >= len(config['cv_iters']):
        return

    # set up logger
    config['logger'] = logging.getLogger(name=f'cv_pred_run_rank_{config["rank"]}')
    logger_handler = logging.FileHandler(filename=config['paths']['experiment_dir'] / f'cv_run_{config["rank"]}.log',
                                         mode='w')
    # logger_handler_stream = logging.StreamHandler(sys.stdout)
    # logger_handler_stream.setLevel(logging.INFO)
    logger_formatter = logging.Formatter('%(asctime)s - %(message)s')
    config['logger'].setLevel(logging.INFO)
    logger_handler.setFormatter(logger_formatter)
    # logger_handler_stream.setFormatter(logger_formatter)
    config['logger'].addHandler(logger_handler)
    # logger.addHandler(logger_handler_stream)
    config['logger'].info(f'Starting run {config["paths"]["experiment_dir"].name}...')

    config['dev_train'] = f'/gpu:{config["gpu_id"]}'
    config['dev_predict'] = f'/gpu:{config["gpu_id"]}'

    if config['paths']['hpo_dir'] is not None:
        hpo_path = Path(config['paths']['hpo_dir'])
        res = utils_hpo.logged_results_to_HBS_result(hpo_path, f'_{hpo_path.name}')

        # get ID to config mapping
        id2config = res.get_id2config_mapping()
        # best config - incumbent
        incumbent = res.get_incumbent_id()
        config_id_hpo = incumbent
        config['config'].update(id2config[config_id_hpo]['config'])

        # select a specific config based on its ID
        # example - check config.json
        # config = id2config[(0, 0, 0)]['config']

        config['logger'].info(f'Using configuration from HPO study {hpo_path.name}')
        config['logger'].info(f'HPO Config {config_id_hpo}: {config["config"]}')

    # base model used - check estimator_util.py to see which models are implemented
    config['base_model'] = ExoMiner

    # choose features set
    for feature_name, feature in config['features_set'].items():
        if feature['dtype'] == 'float32':
            config['features_set'][feature_name]['dtype'] = tf.float32

    config['logger'].info(f'Feature set: {config["features_set"]}')

    config['logger'].info(f'Final configuration used: {config}')

    # save feature set used
    if rank == 0:
        np.save(config['paths']['experiment_dir'] / 'features_set.npy', config['features_set'])
        # save model configuration used
        np.save(config['paths']['experiment_dir'] / 'config.npy', config['config'])

        # save the YAML file with training-evaluation parameters that are YAML serializable
        json_dict = {key: val for key, val in config.items() if is_yamlble(val)}
        with open(config['paths']['experiment_dir'] / 'cv_params.yaml', 'w') as cv_run_file:
            yaml.dump(json_dict, cv_run_file)

    if config['train_parallel']:
        # run each CV iteration in parallel
        cv_id = config['rank']
        config['logger'].info(f'Running prediction for CV iteration {cv_id} (out of {len(config["cv_iters"])}): '
                    f'{config["cv_iters"][cv_id]}')
        config['cv_id'] = cv_id
        cv_pred_run(
            config,
            config['cv_iters'][cv_id],
        )
    else:
        # run each CV iteration sequentially
        for cv_id, cv_iter in enumerate(config['cv_iters']):
            config['logger'].info(f'[cv_iter_{cv_iter}] Running prediction for CV iteration {cv_id} '
                                  f'(out of {len(config["cv_iters"])}): {cv_iter}')
            config['cv_id'] = cv_id
            cv_pred_run(
                config,
                cv_iter
            )


if __name__ == '__main__':
    cv_pred()
