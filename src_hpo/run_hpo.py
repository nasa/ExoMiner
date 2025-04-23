"""
Main script used to run hyperparameter optimization studies using BOHB, BO and RS implementation by Falkner et al [1].

[1] Falkner, Stefan, Aaron Klein, and Frank Hutter. "BOHB: Robust and efficient hyperparameter optimization at scale."
arXiv preprint arXiv:1807.01774 (2018).
"""

# 3rd party
import time
import numpy as np
import logging
from pathlib import Path
import yaml
import hpbandster.core.nameserver as hpns
from hpbandster.optimizers import BOHB, RandomSearch
import hpbandster.core.result as hpres
import argparse

# local
from src_hpo.worker_hpo_keras import TransitClassifier, get_configspace
from src_hpo.utils_hpo import analyze_results


def run_main(hpo_config):
    """ Run HPO study.

    :param hpo_config: dict, HPO parameters
    :return:
    """

    host = hpns.nic_name_to_host(hpo_config['nic_name'])

    if hpo_config['worker_id'] != 0:  # workers go here
        # short artificial delay to make sure the nameserver is already running and current run_id is instantiated
        time.sleep(2 * hpo_config['worker_id'])

        # if logger is not None:
        #     logger.info(f'Starting worker {hpo_config["worker_id"]}')
        # printstr = "Starting worker %s" % worker_id
        # print('\n\x1b[0;33;33m' + printstr + '\x1b[0m\n')

        w = TransitClassifier(hpo_config, worker_id_custom=hpo_config['worker_id'], run_id=hpo_config['study'],
                              host=host)
        w.load_nameserver_credentials(working_directory=hpo_config['paths']['experiment_dir'])
        w.run(background=False)
        exit(0)

    # set up logger
    logger = logging.getLogger(name='hpo_main_run')
    logger_handler = logging.FileHandler(filename=config['paths']['experiment_dir'] /
                                                  f'main_worker_{config["worker_id"]}.log', mode='w')
    logger_formatter = logging.Formatter('%(asctime)s - %(message)s')
    logger.setLevel(logging.INFO)
    logger_handler.setFormatter(logger_formatter)
    logger.addHandler(logger_handler)

    # Start nameserver:
    name_server = hpns.NameServer(run_id=hpo_config['study'], host=host, port=0,
                                  working_directory=hpo_config['paths']['experiment_dir'])
    ns_host, ns_port = name_server.start()

    logger.info(f'Starting worker {hpo_config["worker_id"]} on master node')
    # start worker on master node  ~optimizer is inexpensive, so can afford to run worker alongside optimizer
    w = TransitClassifier(hpo_config, worker_id_custom=hpo_config['worker_id'], run_id=hpo_config['study'], host=host,
                          nameserver=ns_host, nameserver_port=ns_port)
    w.run(background=True)

    result_logger = hpres.json_result_logger(directory=hpo_config['paths']['experiment_dir'], overwrite=True)

    # let us load results from previous HPO run to use its results to warm-start a new run with slightly
    # different budgets in terms of data points and epochs.
    # note that the search space has to be identical though!
    # directory must contain a config.json and results.json for the same configuration space.'
    if hpo_config['paths']['prev_run_dir'] is not None:
        logger.info(f'Warm-start using results from HPO run in {hpo_config["paths"]["prev_run_dir"]}')
        previous_run = hpres.logged_results_to_HBS_result(hpo_config['paths']['prev_run_dir'])
    else:
        previous_run = None

    if hpo_config['optimizer'] == 'bohb':
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
        logger.info(f'Saving KDE parameters')
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
    logger.info(f'Shutting down workers')
    hpo.shutdown(shutdown_workers=True)
    name_server.shutdown()

    # analyse and save results
    logger.info(f'Analyzing results from HPO run')
    analyze_results(res, hpo_config)

    logger.info(f'Finished HPO run.')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--worker_id', type=int, help='Worker ID.',
                        default=0)
    parser.add_argument('--config_file', type=str, help='File path to YAML configuration file.',
                        default='/home6/msaragoc/work_dir/Kepler-TESS_exoplanet/codebase/src_hpo/config_hpo.yaml')
    parser.add_argument('--output_dir', type=str, help='File path to HPO run directory.',
                        default=None)

    args = parser.parse_args()

    # load HPO running parameters from configuration yaml file
    with open(args.config_file, 'r') as file:
        config = yaml.safe_load(file)

    # set worker id
    config['worker_id'] = args.worker_id

    # set output directory
    if args.output_dir is not None:
        config['paths']['experiment_dir'] = args.output_dir

    # convert paths to Path objects
    for path_name, path_str in config['paths'].items():
        if path_str is not None:
            config['paths'][path_name] = Path(path_str)

    # set HPO run name
    config['study'] = config['paths']['experiment_dir'].name

    # set TFRecords for training, and test from yaml file
    with open(config['paths']['datasets_fps'], 'r') as file:
        config['datasets_fps'] = yaml.unsafe_load(file)

    if config['worker_id'] == 0:
        np.save(config['paths']['experiment_dir'] / 'hpo_run_config.npy', config)

        # save the YAML file with configuration parameters that are YAML serializable
        json_dict = {key: val for key, val in config.items()}  # if is_yamlble(val)}
        with open(config['paths']['experiment_dir'] / 'hpo_run_config.yaml', 'w') as cv_run_file:
            yaml.dump(json_dict, cv_run_file, sort_keys=False)

    run_main(config)
