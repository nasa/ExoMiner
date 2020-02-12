"""
Main script used to run hyperparameter optimization studies using BOHB, BO and RS implementation by Falkner et al [1].

[1] Falkner, Stefan, Aaron Klein, and Frank Hutter. "BOHB: Robust and efficient hyperparameter optimization at scale."
arXiv preprint arXiv:1807.01774 (2018).

"""

# 3rd party
import os
import sys
# sys.path.append('/home6/msaragoc/work_dir/HPO_Kepler_TESS/')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import time
from mpi4py import MPI
import numpy as np
import logging
logging.basicConfig(level=logging.WARNING)
# logging.propagate = False
import tensorflow as tf
# import itertools

import paths
if 'home6' in paths.path_hpoconfigs:
    import matplotlib; matplotlib.use('agg')

import hpbandster.core.nameserver as hpns
from hpbandster.optimizers import BOHB, RandomSearch
import hpbandster.core.result as hpres

# local
from src.estimator_util import get_ce_weights, CNN1dPlanetFinderv1, CNN1dModel, Exonet, Exonet_XS
from src_hpo.worker_tf_locglob import TransitClassifier
from src_hpo.utils_hpo import analyze_results, json_result_logger, check_run_id
import paths


def run_main(args, bohb_params=None):

    if 'tess' in args.satellite:
        args.label_map = {"PC": 1, "NTP": 0, "EB": 2, "BEB": 2} if args.multi_class else {"PC": 1, "NTP": 0, "EB": 0,
                                                                                          "BEB": 0}
    else:
        args.label_map = {"PC": 1, "NTP": 0, "AFP": 2} if args.multi_class else {"PC": 1, "NTP": 0, "AFP": 0}

    if not args.worker:
        if not os.path.isdir(args.results_directory):
            os.mkdir(args.results_directory)
        if not os.path.isdir(args.models_directory):
            os.mkdir(args.models_directory)

    args.ce_weights, args.n_train = get_ce_weights(args.label_map, args.tfrec_dir)

    host = hpns.nic_name_to_host(args.nic_name)

    if args.worker:
        # short artificial delay to make sure the nameserver is already running and current run_id is instantiated
        time.sleep(2 * rank)
        args.studyid = check_run_id(args.studyid, args.results_directory, worker=True)

        # printstr = "Starting worker %s" % rank
        # print('\n\x1b[0;33;33m' + printstr + '\x1b[0m\n')

        w = TransitClassifier(args, worker_id_custom=rank, run_id=args.studyid, host=host)
        w.load_nameserver_credentials(working_directory=args.results_directory)
        w.run(background=False)
        exit(0)

    # args.run_id = check_run_id(args.run_id, shared_directory)
    args.studyid = check_run_id(args.studyid, args.results_directory)

    # Start nameserver:
    name_server = hpns.NameServer(run_id=args.studyid, host=host, port=0, working_directory=args.results_directory)
    ns_host, ns_port = name_server.start()

    # start worker on master node  ~optimizer is inexpensive, so can afford to run worker alongside optimizer
    w = TransitClassifier(args, worker_id_custom=rank, run_id=args.studyid, host=host, nameserver=ns_host,
                          nameserver_port=ns_port)
    w.run(background=True)

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
        hpo = BOHB(configspace=TransitClassifier.get_configspace(),
                   run_id=args.studyid,
                   host=host,
                   nameserver=ns_host,
                   nameserver_port=ns_port,
                   working_directory=args.results_directory,
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
        hpo = RandomSearch(configspace=TransitClassifier.get_configspace(),
                           run_id=args.studyid,
                           host=host,
                           nameserver=ns_host,
                           nameserver_port=ns_port,
                           working_directory=args.results_directory,
                           result_logger=result_logger,
                           min_budget=args.min_budget,
                           max_budget=args.max_budget,
                           eta=args.eta,)
        res = hpo.run(n_iterations=args.n_iterations)

    # shutdown
    hpo.shutdown(shutdown_workers=True)
    name_server.shutdown()

    # Analyse and save results
    analyze_results(res, args, args.results_directory, args.studyid)


if __name__ == '__main__':

    optimizer = 'bohb'  # types of hyperparameter optimizers available: 'random_search', 'bohb'

    # if minimum and maximum budgets are set to the same value, then BOHB becomes BO (Bayesian optimization)
    min_budget = 6
    max_budget = 50
    n_iterations = 400

    # number of models trained per configuration evaluated on a given budget
    # used to decrease the variability due to random weights initialization
    ensemble_n = 3

    # metric used to evaluate the performance of a given configuration on the validation set
    hpo_loss = 'pr auc'

    # BOHB and BO parameters; check [1]
    bohb_params = {'top_n_percent': 15, 'num_samples': 64, 'random_fraction': 1/3, 'bandwidth_factor': 3,
                   'min_bandwidth': 1e-3}

    eta = 2  # Down sampling rate, must be greater or equal to 2

    study = ''  # name of the HPO study

    # base model used - check estimator_util.py to see which models are implemented
    BaseModel = CNN1dPlanetFinderv1

    num_gpus = 1  # numper of GPUs per node

    # directory in which the models are saved
    models_directory = os.path.join(paths.path_hpomodels, study)

    # directory in which the results are saved
    results_directory = os.path.join(paths.path_hpoconfigs, study)

    # previous run directory; used to warmup start model based optimizers
    prev_run_study = ''
    prev_run_dir = None  # os.path.join(paths.path_hpoconfigs, prev_run_study)

    # data directory
    tfrec_dir = os.path.join(paths.path_tfrecs, 'Kepler/tfrecordkeplerdr25_'
                                                'flux-centroid_selfnormalized-oddeven_nonwhitened_gapped_2001-201')

    # features to be extracted from the dataset - keywords used in the TFRecords
    # views = ['global_view', 'local_view']
    # #channels_centr = ['', '_centr']
    # channels_oddeven = ['', '_odd', '_even']
    # features_names = [''.join(feature_name_tuple)
    #                   for feature_name_tuple in itertools.product(views, channels_oddeven)]
    features_names = ['global_view', 'local_view']
    features_dim = {feature_name: 2001 if 'global' in feature_name else 201 for feature_name in features_names}
    features_dtypes = {feature_name: tf.float32 for feature_name in features_names}
    features_set = {feature_name: {'dim': features_dim[feature_name], 'dtype': features_dtypes[feature_name]}
                    for feature_name in features_names}
    # # example
    # features_set = {'global_view': {'dim': 2001, 'dtype': tf.float32},
    #                 'local_view': {'dim': 201, 'dtype': tf.float32}}

    nic_name = 'lo'  # 'ib0' or 'lo'

    rank = MPI.COMM_WORLD.rank
    # size = MPI.COMM_WORLD.size    
    print('Rank = ', rank)
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
                        help='Directory in which the models are saved.')
    parser.add_argument('--results_directory', type=str, default=results_directory,
                        help='Directory which the results and logs are saved.')

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

    parser.add_argument('--num_gpus', type=int,
                        help='Number of GPUs per node. If set to \'0\' the workers (configurations) are not distributed '
                             'by the available GPUs (1 worker/GPU).', default=num_gpus)


    args = parser.parse_args()

    # data used to filer the datasets; use None if not filtering
    args.filter_data = None  # np.load('/data5/tess_project/Data/tfrecords/filter_datasets/cmmn_kepids_spline-whitened.npy').item()

    args.features_set = features_set

    args.BaseModel = BaseModel

    run_main(args, bohb_params)
