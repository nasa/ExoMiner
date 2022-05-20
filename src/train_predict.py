"""

"""

# 3rd party
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
from mpi4py import MPI
import time
from tensorflow.keras import callbacks
import argparse
from tensorflow.keras.utils import plot_model
from pathlib import Path
import logging
import yaml

# local
from src.utils_dataio import InputFnv2 as InputFn, get_data_from_tfrecord
from models.models_keras import ExoMiner, compile_model
from src.utils_metrics import get_metrics, get_metrics_multiclass, compute_precision_at_k
from src_hpo import utils_hpo
from src.utils_visualization import plot_class_distribution, plot_precision_at_k
from src.utils_train import save_metrics_to_file, print_metrics, plot_loss_metric, plot_roc, plot_pr_curve
from utils.utils_dataio import is_yamlble
from paths import path_main
from src.train_keras import run_main as train_fn
from src.predict_ensemble_keras import run_main as pred_fn


if __name__ == '__main__':


    with(open(path_to_yaml, 'r')) as file:
        config = yaml.safe_load(file)


    # used in job arrays
    parser = argparse.ArgumentParser()
    parser.add_argument('--job_idx', type=int, help='Job index', default=0)
    parser.add_argument('--train_config_file', type=str, help='File path to YAML train configuration file.', default=None)
    parser.add_argument('--pred_config_file', type=str, help='File path to YAML predict configuration file.', default=None)
    args = parser.parse_args()

    if args.config_file is None:  # use default config file in codebase
        path_to_yaml = Path(path_main + 'src/config_predict.yaml')
    else:  # use config file given as input
        path_to_yaml = Path(args.config_file)

    if config['rank'] == 0:  # save initial configuration
        with open(config['paths']['experiment_dir'] / 'train_params_init.yaml', 'w') as config_file:
            yaml.dump(config, config_file)

    # uncomment for MPI multiprocessing
    rank = MPI.COMM_WORLD.rank
    config['rank'] = config['ngpus_per_node'] * args.job_idx + rank
    config['size'] = MPI.COMM_WORLD.size
    print(f'Rank = {config["rank"]}/{config["size"] - 1}')
    sys.stdout.flush()
    if rank != 0:
        time.sleep(2)

    # select GPU to run the training on
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

    # SCRIPT PARAMETERS #############################################

    # experiment directory
    for path_name, path_str in config['paths'].items():
        config['paths'][path_name] = Path(path_str)
    config['paths']['experiment_dir'].mkdir(exist_ok=True)
    config['paths']['models_dir'] = config['paths']['experiment_dir'] / 'models'
    config['paths']['models_dir'].mkdir(exist_ok=True)

    # set up logger
    logger = logging.getLogger(name='train-eval_run')
    logger_handler = logging.FileHandler(filename=config['paths']['experiment_dir'] / f'train-eval_run_{rank}.log',
                                         mode='w')
    logger_formatter = logging.Formatter('%(asctime)s - %(message)s')
    logger.setLevel(logging.INFO)
    logger_handler.setFormatter(logger_formatter)
    logger.addHandler(logger_handler)
    logger.info(f'Starting run {config["paths"]["experiment_dir"].name}...')

    # TFRecord files directory
    logger.info(f'Using data from {config["paths"]["tfrec_dir"]}')

    # name of the HPO study from which to get a configuration; config needs to be set to None
    # set the configuration from an HPO study
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

        logger.info(f'Using configuration from HPO study {hpo_path.name}')
        logger.info(f'HPO Config {config_id_hpo}: {config["config"]}')

    # base model used - check estimator_util.py to see which models are implemented
    BaseModel = ExoMiner  # CNN1dPlanetFinderv2
    # config['config']['parameters'].update(baseline_configs.astronet)

    # choose features set
    for feature_name, feature in config['features_set'].items():
        if feature['dtype'] == 'float':
            config['features_set'][feature_name]['dtype'] = tf.float32
        elif feature['dtype'] == 'int':
            config['features_set'][feature_name]['dtype'] = tf.int64

    logger.info(f'Feature set: {config["features_set"]}')

    # early stopping callback
    config['callbacks']['early_stopping']['obj'] = callbacks.EarlyStopping(**config['callbacks']['early_stopping'])

    # TensorBoard callback
    config['callbacks']['tensorboard']['obj'] = callbacks.TensorBoard(**config['callbacks']['tensorboard'])

    logger.info(f'Final configuration used: {config}')

    if config['rank'] == 0:
        # save the YAML file with training-evaluation parameters that are YAML serializable
        json_dict = {key: val for key, val in config.items() if is_yamlble(val)}
        with open(config['paths']['experiment_dir'] / 'train_params.yaml', 'w') as cv_run_file:
            yaml.dump(json_dict, cv_run_file)

    if config['train_parallel']:  # train models in parallel
        if config['rank'] < config['training']['n_models']:
            print(f'Training model {config["rank"] + 1} out of {config["training"]["n_models"]} on '
                  f'{config["training"]["n_epochs"]}')
            sys.stdout.flush()
            train_fn(config=config,
                     base_model=BaseModel,
                     model_id=config['rank'] + 1,
                     )
    else:  # train models sequentially
        for model_i in range(config["training"]["n_models"]):
            print(f'Training model {model_i + 1} out of {config["training"]["n_models"]} on '
                  f'{config["training"]["n_epochs"]} epochs...')
            train_fn(config=config,
                     base_model=BaseModel,
                     model_id=model_i + 1,
                     )

    if args.config_file is None:  # use default config file in codebase
        path_to_yaml = Path(path_main + 'src/config_predict.yaml')
    else:  # use config file given as input
        path_to_yaml = Path(args.config_file)

    with(open(path_to_yaml, 'r')) as file:  # read YAML configuration file
        config = yaml.safe_load(file)

    # experiment directory
    for path_name, path_str in config['paths'].items():
        config['paths'][path_name] = Path(path_str)
    config['paths']['experiment_dir'].mkdir(exist_ok=True)

    # if config file is from the training experiment
    config['paths']['experiment_dir'].mkdir(exist_ok=True)

    # set up logger
    logger = logging.getLogger(name='predict_run')
    logger_handler = logging.FileHandler(filename=config['paths']['experiment_dir'] / f'predict_run.log', mode='w')
    logger_formatter = logging.Formatter('%(asctime)s - %(message)s')
    logger.setLevel(logging.INFO)
    logger_handler.setFormatter(logger_formatter)
    logger.addHandler(logger_handler)
    logger.info(f'Starting run {config["paths"]["experiment_dir"].name}...')

    logger.info(f'Using data from {config["paths"]["tfrec_dir"]}')

    logger.info(f'Datasets to be evaluated/tested: {config["datasets"]}')

    models_dir = config['paths']['models_dir'] / 'models'
    config['paths']['models_filepaths'] = [model_dir / f'{model_dir.stem}.h5' for model_dir in models_dir.iterdir()
                                           if 'model' in model_dir.stem]
    logger.info(f'Models\' file paths: {config["paths"]["models_filepaths"]}')

    # set the configuration from a HPO study to use extra architecture such as batch size
    if config['paths']['hpo_dir'] is not None:
        hpo_study_fp = Path(config['paths']['hpo_dir'])
        res = utils_hpo.logged_results_to_HBS_result(hpo_study_fp, f'_{hpo_study_fp.name}')

        # get ID to config mapping
        id2config = res.get_id2config_mapping()
        # best config - incumbent
        incumbent = res.get_incumbent_id()
        config_id_hpo = incumbent
        if 'config' in config:
            config['config'].update(id2config[config_id_hpo]['config'])
        else:
            config['config'] = id2config[config_id_hpo]['config']

        # select a specific config based on its ID
        # example - check config.json
        # config = id2config[(8, 0, 3)]['config']

        logger.info(f'Using configuration from HPO study {hpo_study_fp.name}')
        logger.info(f'HPO Config {config_id_hpo}: {config}')

    logger.info(f'Final configuration used: {config}')

    # choose features set
    for feature_name, feature in config['features_set'].items():
        if feature['dtype'] == 'float32':
            config['features_set'][feature_name]['dtype'] = tf.float32
        elif feature['dtype'] == 'int':
            config['features_set'][feature_name]['dtype'] = tf.int64

    logger.info(f'Feature set: {config["features_set"]}')

    # TensorBoard callback
    config['callbacks']['tensorboard']['obj'] = callbacks.TensorBoard(**config['callbacks']['tensorboard'])

    # save the YAML file with training-evaluation parameters that are YAML serializable
    json_dict = {key: val for key, val in config.items() if is_yamlble(val)}
    with open(config['paths']['experiment_dir'] / 'predict_params.yaml', 'w') as cv_run_file:
        yaml.dump(json_dict, cv_run_file)

    pred_fn(config=config)

    logger.info(f'Finished evaluation and prediction of the ensemble.')