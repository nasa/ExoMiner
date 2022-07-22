"""
Main script used to run hyperparameter optimization studies using BOHB, BO and RS implementation by Falkner et al [1].

[1] Falkner, Stefan, Aaron Klein, and Frank Hutter. "BOHB: Robust and efficient hyperparameter optimization at scale."
arXiv preprint arXiv:1807.01774 (2018).
"""

# 3rd party
# import logging
# logging.basicConfig(level=logging.DEBUG)
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tensorflow as tf
import time
from mpi4py import MPI
import numpy as np
import logging
logging.basicConfig(level=logging.WARNING)
from pathlib import Path
import yaml
import hpbandster.core.nameserver as hpns
from hpbandster.optimizers import BOHB, RandomSearch
import hpbandster.core.result as hpres

# local
from models.models_keras import Astronet, Exonet, TransformerExoMiner, ExoMiner
from src_hpo.worker_hpo_keras import TransitClassifier, get_configspace
from src_hpo.utils_hpo import analyze_results, json_result_logger, check_run_id
# import paths
from utils.utils_dataio import is_yamlble


def run_main(hpo_config, logger=None):
    """ Run HPO study.

    :param hpo_config: dict, HPO parameters
    :param logger: logger
    :return:
    """

    # for each rank, create a folder to save temporarily the models created for a given run
    hpo_config['model_dir_rank'] = hpo_config['paths']['experiment_dir'] / f'models_rank{hpo_config["rank"]}'
    hpo_config['model_dir_rank'].mkdir(exist_ok=True)

    host = hpns.nic_name_to_host(hpo_config['nic_name'])

    if hpo_config['rank'] != 0:  # workers go here
        # short artificial delay to make sure the nameserver is already running and current run_id is instantiated
        time.sleep(2 * hpo_config['rank'])
        # hpo_config['study'] = check_run_id(hpo_config['study'], hpo_config['paths']['experiment_dir'], worker=True)

        if logger is not None:
            logger.info(f'Starting worker {hpo_config["rank"]}')
        # printstr = "Starting worker %s" % rank
        # print('\n\x1b[0;33;33m' + printstr + '\x1b[0m\n')

        w = TransitClassifier(hpo_config, worker_id_custom=hpo_config['rank'], run_id=hpo_config['study'], host=host)
        w.load_nameserver_credentials(working_directory=hpo_config['paths']['experiment_dir'])
        w.run(background=False)
        exit(0)

    # args.run_id = check_run_id(args.run_id, shared_directory)
    # hpo_config['study'] = check_run_id(hpo_config['study'], hpo_config['paths']['experiment_dir'])

    # Start nameserver:
    name_server = hpns.NameServer(run_id=hpo_config['study'], host=host, port=0,
                                  working_directory=hpo_config['paths']['experiment_dir'])
    ns_host, ns_port = name_server.start()

    if logger is not None:
        logger.info(f'Starting worker {hpo_config["rank"]} on master node')
    # start worker on master node  ~optimizer is inexpensive, so can afford to run worker alongside optimizer
    w = TransitClassifier(hpo_config, worker_id_custom=hpo_config['rank'], run_id=hpo_config['study'], host=host,
                          nameserver=ns_host, nameserver_port=ns_port)
    w.run(background=True)

    # result_logger = json_result_logger(directory=hpo_config['paths']['experiment_dir'], run_id=hpo_config['study'],
    #                                    overwrite=False)
    result_logger = hpres.json_result_logger(directory=hpo_config['paths']['experiment_dir'], overwrite=True)

    # Let us load the src_old run now to use its results to warmstart a new run with slightly
    # different budgets in terms of data points and epochs.
    # Note that the search space has to be identical though!
    # directory must contain a config.json and results.json for the same configuration space.'
    if hpo_config['paths']['prev_run_dir'] is not None:
        if logger is not None:
            logger.info(f'Warm-start using results from HPO run in {hpo_config["paths"]["prev_run_dir"]}')
        previous_run = hpres.logged_results_to_HBS_result(hpo_config['paths']['prev_run_dir'])
    else:
        previous_run = None

    if hpo_config['optimizer'] == 'bohb':
        if logger is not None:
            logger.info(f'Starting BOHB/BO run')
        # instantiate BOHB or BO study
        hpo = BOHB(configspace=get_configspace(hpo_config['configuration_space']),
                   run_id=hpo_config['study'],
                   host=host,
                   nameserver=ns_host,
                   nameserver_port=ns_port,
                   working_directory=str(hpo_config['paths']['experiment_dir']),
                   result_logger=result_logger,
                   min_budget=hpo_config['min_budget'],
                   max_budget=hpo_config['max_budget'],
                   # min_points_in_model=18,
                   eta=hpo_config['eta'],
                   previous_result=previous_run,
                   **hpo_config['bohb_params'])

        # run BOHB/BO study
        res = hpo.run(n_iterations=hpo_config['n_iterations'])

        # save kde parameters
        if logger is not None:
            logger.info(f'Saving KDE parameters')
        if hpo_config['rank'] != 0:
            kde_models_bdgt = hpo.config_generator.kde_models
            kde_models_bdgt_params = {bdgt: dict() for bdgt in kde_models_bdgt}
            for bdgt in kde_models_bdgt:
                for est in kde_models_bdgt[bdgt]:
                    kde_models_bdgt_params[bdgt][est] = [kde_models_bdgt[bdgt][est].data,
                                                         kde_models_bdgt[bdgt][est].data_type,
                                                         kde_models_bdgt[bdgt][est].bw]
            kde_models_bdgt_params['hyperparameters'] = list(hpo.config_generator.configspace._hyperparameters.keys())

            np.save(hpo_config['paths']['experiment_dir'] / 'kde_models_params.npy', kde_models_bdgt_params)

    else:  # run random search
        if logger is not None:
            logger.info(f'Starting random search')
        hpo = RandomSearch(configspace=get_configspace(hpo_config['configuration_space']),
                           run_id=hpo_config['study'],
                           host=host,
                           nameserver=ns_host,
                           nameserver_port=ns_port,
                           working_directory=hpo_config['paths']['experiment_dir'],
                           result_logger=result_logger,
                           min_budget=hpo_config['min_budget'],
                           max_budget=hpo_config['max_budget'],
                           eta=hpo_config['eta'], )
        res = hpo.run(n_iterations=hpo_config['n_iterations'])

    # shutdown
    if logger is not None:
        logger.info(f'Shutting down workers')
    hpo.shutdown(shutdown_workers=True)
    name_server.shutdown()

    # analyse and save results
    if logger is not None:
        logger.info(f'Analyzing results from HPO run')
    analyze_results(res, hpo_config)


if __name__ == '__main__':

    path_to_yaml = Path('/Users/msaragoc/OneDrive - NASA/Projects/exoplanet_transit_classification/codebase/src_hpo/config_hpo.yaml')
    with(open(path_to_yaml, 'r')) as file:
        config = yaml.safe_load(file)

    for path_name, path_str in config['paths'].items():  # converting paths from string to Path
        if path_str is not None:
            config['paths'][path_name] = Path(path_str)

    if config['rank'] == 0 or config['rank'] is None:
        config['paths']['experiment_dir'].mkdir(exist_ok=True)

    # set up logger
    logger = logging.getLogger(name='hpo_run')
    logger_handler = logging.FileHandler(filename=config['paths']['experiment_dir'] /
                                                  f'hpo_run_{config["rank"]}.log', mode='w')
    logger_formatter = logging.Formatter('%(asctime)s - %(message)s')
    logger.setLevel(logging.INFO)
    logger_handler.setFormatter(logger_formatter)
    logger.addHandler(logger_handler)
    if config['rank'] == 0:
        logger.info(f'Starting run {config["paths"]["experiment_dir"].name}...')

    config['rank'] = MPI.COMM_WORLD.rank
    config['size'] = MPI.COMM_WORLD.size
    logger.info(f'Rank (size) = {config["rank"]} ({config["size"]})')

    if config['rank'] != 0:
        time.sleep(20)

    try:
        logger.info(f'[rank_{config["rank"]}] CUDA DEVICE ORDER: {os.environ["CUDA_DEVICE_ORDER"]}')
        logger.info(f'[rank_{config["rank"]}] CUDA VISIBLE DEVICES: {os.environ["CUDA_VISIBLE_DEVICES"]}')
    except:
        logger.info(f'[rank_{config["rank"]}] No CUDA environment variables exist.')

    # n_gpus = len(os.environ["CUDA_VISIBLE_DEVICES"].split(','))  # number of GPUs visible to the process
    if config["rank"] == 0:
        logger.info(f'Number of GPUs selected per node = {config["ngpus_per_node"]}')
    config['gpu_id'] = config["rank"] % config['ngpus_per_node']
    logger.info(f'GPU ID: {config["gpu_id"]}')

    # setting GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config['gpu_id'])  # "0, 1"

    logger.info(f'[rank_{config["rank"]}] CUDA DEVICE ORDER: {os.environ["CUDA_DEVICE_ORDER"]}')
    logger.info(f'[rank_{config["rank"]}] CUDA VISIBLE DEVICES: {os.environ["CUDA_VISIBLE_DEVICES"]}')

    # features set
    for feature_name, feature in config['features_set'].items():
        if feature['dtype'] == 'float':
            config['features_set'][feature_name]['dtype'] = tf.float32
        elif feature['dtype'] == 'int':
            config['features_set'][feature_name]['dtype'] = tf.int64

    # add fit parameters
    config['callbacks_list'] = []

    config['base_model'] = TransformerExoMiner

    config['study'] = config['paths']['experiment_dir'].name

    if config['rank'] == 0:
        (config['paths']['experiment_dir'] / 'logs').mkdir(exist_ok=True)

        np.save(config['paths']['experiment_dir'] / 'hpo_run_config.npy', config)

        # save the YAML file with training-evaluation parameters that are YAML serializable
        json_dict = {key: val for key, val in config.items() if is_yamlble(val)}
        with open(config['paths']['experiment_dir'] / 'hpo_run_config.yaml', 'w') as cv_run_file:
            yaml.dump(json_dict, cv_run_file)

    logger.info(f'HPO run parameters: {config}')
    sys.stdout.flush()

    run_main(config, logger)
