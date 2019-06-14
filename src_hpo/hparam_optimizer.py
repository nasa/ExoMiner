"""
Main script used to run hyperparameter optimization studies using BOHB, BO and RS implementation by Falkner et al.
"""

# 3rd party
import os
import sys
import argparse
import time
# from mpi4py import MPI
import numpy as np
import logging
# import matplotlib; matplotlib.use('agg')
# import matplotlib.pyplot as plt
logging.basicConfig(level=logging.WARNING)
# logging.basicConfig(level=logging.DEBUG)
# logging.propagate = False
# if 'nobackup' in os.path.dirname(__file__):
#     # For running on Pleiades without matplotlib fig() error
#     plt.switch_backend('agg')

import hpbandster.core.nameserver as hpns
from hpbandster.optimizers import BOHB, RandomSearch
import hpbandster.core.result as hpres

# local
# from src_hpo.worker_tf_locglob import TransitClassifier as TransitClassifier_tf
from src_hpo.worker_tf_locglob_ensemble import TransitClassifier as TransitClassifier_tf
# from src_hpo.hparam_optimizer import analyze_results, get_ce_weights, json_result_logger, check_run_id
from src_hpo.utils_hpo import analyze_results, get_ce_weights, json_result_logger, check_run_id
# from worker_tf_locglob import TransitClassifier as TransitClassifier_tf
# from worker_tf_locglob_ensemble import TransitClassifier as TransitClassifier_tf
# from hparam_optimizer import analyze_results, get_ce_weights, json_result_logger, check_run_id
# from utils_hpo import analyze_results, get_ce_weights, json_result_logger, check_run_id
import paths


def run_main(args, bohb_params=None):

    if 'tess' in args.tfrec_dir:
        args.label_map = {"PC": 1, "NTP": 0, "EB": 2, "BEB": 2} if args.multi_class else {"PC": 1, "NTP": 0, "EB": 0,
                                                                                          "BEB": 0}
    else:
        args.label_map = {"PC": 1, "NTP": 0, "AFP": 2} if args.multi_class else {"PC": 1, "NTP": 0, "AFP": 0}

    # shared_directory = os.path.dirname(os.path.abspath(__file__)) + '/logs'

    # print('shared directory:', shared_directory)

    if not args.worker:
        if not os.path.isdir(args.models_directory):
            os.mkdir(args.models_directory)
        if not os.path.isdir(args.results_directory):
            os.mkdir(args.results_directory)

    # shared_directory = os.path.dirname(__file__) + '/logs'

    worker = TransitClassifier_tf
    args.ce_weights, args.centr_flag = get_ce_weights(args.label_map, args.tfrec_dir)

    host = hpns.nic_name_to_host(args.nic_name)

    if args.worker:
        # short artificial delay to make sure the nameserver is already running and current run_id is instantiated
        time.sleep(2 * rank)
        # args.run_id = check_run_id(args.run_id, shared_directory, worker=True)
        args.run_id = check_run_id(args.studyid, args.results_directory, worker=True)

        # printstr = "Starting worker %s" % rank
        # print('\n\x1b[0;33;33m' + printstr + '\x1b[0m\n')

        # w = worker(args, worker_id_custom=rank, run_id=args.run_id, host=host)
        w = worker(args, worker_id_custom=rank, run_id=args.studyid, host=host)
        # w.load_nameserver_credentials(working_directory=shared_directory)
        w.load_nameserver_credentials(working_directory=args.results_directory)
        w.run(background=False)
        exit(0)

    # args.run_id = check_run_id(args.run_id, shared_directory)
    args.studyid = check_run_id(args.studyid, args.results_directory)

    # Start nameserver:
    # name_server = hpns.NameServer(run_id=args.run_id, host=host, port=0, working_directory=shared_directory)
    name_server = hpns.NameServer(run_id=args.studyid, host=host, port=0, working_directory=args.results_directory)
    ns_host, ns_port = name_server.start()

    # start worker on master node  ~optimizer is inexpensive, so can afford to run worker alongside optimizer
    # w = worker(args, worker_id_custom=rank, run_id=args.run_id, host=host, nameserver=ns_host, nameserver_port=ns_port)
    w = worker(args, worker_id_custom=rank, run_id=args.studyid, host=host, nameserver=ns_host, nameserver_port=ns_port)
    w.run(background=True)

    # result_logger = json_result_logger(directory=shared_directory, run_id=args.run_id, overwrite=False)
    result_logger = json_result_logger(directory=args.results_directory, run_id=args.studyid, overwrite=False)
    # result_logger = hpres.json_result_logger(directory=args.results_directory, overwrite=False)

    # Let us load the old run now to use its results to warmstart a new run with slightly
    # different budgets in terms of data points and epochs.
    # Note that the search space has to be identical though!
    # directory must contain a config.json and results.json for the same configuration space.'
    if args.prev_run_dir is not None:
        previous_run = hpres.logged_results_to_HBS_result(args.prev_run_dir)
    else:
        previous_run = None

    if args.optimizer == 'bohb':
        # instantiate BOHB or BO study
        hpo = BOHB(configspace=worker.get_configspace(),
                   run_id=args.studyid,  # args.run_id,
                   host=host,
                   nameserver=ns_host,
                   nameserver_port=ns_port,
                   working_directory=args.results_directory,  # shared_directory,
                   result_logger=result_logger,
                   min_budget=args.min_budget,
                   max_budget=args.max_budget,
                   # min_points_in_model=18,
                   eta=args.eta,
                   previous_result=previous_run,
                   **bohb_params)

        # run BOHB study
        res = hpo.run(n_iterations=args.n_iterations)

        # save kde parameters
        kde_models_bdgt = hpo.config_generator.kde_models
        kde_models_bdgt_params = {bdgt: dict() for bdgt in kde_models_bdgt}
        for bdgt in kde_models_bdgt:
            for est in kde_models_bdgt[bdgt]:
                kde_models_bdgt_params[bdgt][est] = [kde_models_bdgt[bdgt][est].data,
                                                     kde_models_bdgt[bdgt][est].data_type,
                                                     kde_models_bdgt[bdgt][est].bw]
        kde_models_bdgt_params['hyperparameters'] = list(hpo.config_generator.configspace._hyperparameters.keys())

        np.save(args.results_directory + '/kde_models_params.npy', kde_models_bdgt_params)

    else:  # run random search
        hpo = RandomSearch(configspace=worker.get_configspace(),
                           run_id=args.run_id,
                           host=host,
                           nameserver=ns_host,
                           nameserver_port=ns_port,
                           working_directory=args.results_directory,  # shared_directory,
                           result_logger=result_logger,
                           min_budget=args.min_budget,
                           max_budget=args.max_budget,
                           eta=args.eta,)
        res = hpo.run(n_iterations=args.n_iterations)

    # shutdown
    hpo.shutdown(shutdown_workers=True)
    name_server.shutdown()

    # Analyse and save results
    # analyze_results(res, args, shared_directory, args.run_id)
    analyze_results(res, args, args.results_directory, args.run_id)


if __name__ == '__main__':

    optimizer = 'bohb'  # 'random_search'  # 'bohb'

    min_budget = 1  # 16
    max_budget = 50  # 128
    n_iterations = 200  # 16

    ensemble_n = 2

    hpo_loss = 'pr auc'

    bohb_params = {'top_n_percent': 15, 'num_samples': 64, 'random_fraction': 1/3, 'bandwidth_factor': 3,
                   'min_bandwidth': 1e-3}

    eta = 2  # Down sampling rate, must be greater or equal to 2

    study = 'study_bo'

    # directory in which the models are saved
    models_directory = paths.path_hpomodels + study

    # directory in which the results are saved
    results_directory = paths.path_hpoconfigs + study

    # previous run directory  # used to warmup start model based optimizers
    prev_run_study = 'study_8'
    prev_run_dir = None  # paths.path_hpoconfigs + prev_run_study

    # data directory
    tfrec_dir = paths.tfrec_dir

    nic_name = 'lo'  # 'ib0'

    rank = 0  # MPI.COMM_WORLD.rank
    # size = MPI.COMM_WORLD.size    
    # print('Rank=', rank)
    sys.stdout.flush()

    parser = argparse.ArgumentParser(description='Transit classifier hyperparameter optimizer')
    parser.add_argument('--multi_class', type=bool, default=False)
    parser.add_argument('--satellite', type=str, default='kepler')
    parser.add_argument('--use_kepler_ce', type=bool, default=True)
    parser.add_argument('--test_frac', type=float, default=0.1, help='data fraction for model testing')
    parser.add_argument('--val_frac', type=float, default=0.1, help='model validation data fraction')
    parser.add_argument('--tfrec_dir', type=str, default=tfrec_dir)

    parser.add_argument('--min_budget', type=float, help='Minimum number of epochs for training.', default=min_budget)
    parser.add_argument('--max_budget', type=float, help='Maximum number of epochs for training.', default=max_budget)
    parser.add_argument('--n_iterations', type=int, help='Number of iterations performed by the optimizer',
                        default=n_iterations)
    parser.add_argument('--eta', type=int, help='Down sampling rate', default=eta)
    parser.add_argument('--worker', help='Flag to turn this into a Worker process',
                        default=False if rank == 0 else True)

    parser.add_argument('--nic_name', type=str, default=nic_name,
                        help='Which network interface to use for communication. \'lo\' for local, \'ib0\' '
                             'for Infinity Band.')

    parser.add_argument('--models_directory', type=str, default=models_directory,
                        help='Directory where the models are saved.')
    parser.add_argument('--results_directory', type=str, default=results_directory,
                        help='Directory where the results and logs are saved.')
    # parser.add_argument('--run_id', type=str, default=run_id,
    #                     help='Unique id for the BOHB study.')
    parser.add_argument('--studyid', type=str, default=study, help='Name id for the HPO study.')
    parser.add_argument('--prev_run_dir', type=str, help='A directory that contains a config.json and results.json for '
                                                         'the same configuration space.',
                        default=prev_run_dir)

    parser.add_argument('--optimizer', type=str, help='Optimizer used to conduct the HPO study. Choose between '
                                                      '\'bohb\' and \'random_search\'',
                        default=optimizer)

    parser.add_argument('--ensemble_n', type=int,
                        help='Number of models in ensemble when testing a given configuration.', default=ensemble_n)
    parser.add_argument('--hpo_loss', type=str,
                        help='Loss used by the hyperparameter optimization algorithm.', default=hpo_loss)

    args = parser.parse_args()

    run_main(args, bohb_params)
