"""
Main script used to run hyperparameter optimization studies using BOHB, BO and RS implementation by Falkner et al [1].

[1] Falkner, Stefan, Aaron Klein, and Frank Hutter. "BOHB: Robust and efficient hyperparameter optimization at scale."
arXiv preprint arXiv:1807.01774 (2018).
"""

# 3rd party
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import time
from mpi4py import MPI
import numpy as np
import logging
logging.basicConfig(level=logging.WARNING)
import tensorflow as tf

# import paths
# if 'home6' in paths.path_hpoconfigs:
#     import matplotlib; matplotlib.use('agg')

import hpbandster.core.nameserver as hpns
from hpbandster.optimizers import BOHB, RandomSearch
import hpbandster.core.result as hpres

# local
from src.models_keras import CNN1dPlanetFinderv1, CNN1dPlanetFinderv2
from src import config_keras
from src_hpo.worker_hpo_keras import TransitClassifier
from src_hpo.utils_hpo import analyze_results, json_result_logger, check_run_id
import paths


def run_main(args, bohb_params=None):
    """ Run HPO study.

    :param args:
    :param bohb_params: HPO parameters
    :return:
    """

    if not args.worker:  # create required folders if master node
        os.makedirs(args.results_directory, exist_ok=True)
        os.makedirs(os.path.join(args.results_directory, 'logs'), exist_ok=True)

    # for each rank, create a folder to save temporarily the models created for a given run
    args.model_dir_rank = os.path.join(args.results_directory, 'models_rank{}'.format(args.rank))
    os.makedirs(args.model_dir_rank, exist_ok=True)

    host = hpns.nic_name_to_host(args.nic_name)

    if args.worker:  # workers go here
        # short artificial delay to make sure the nameserver is already running and current run_id is instantiated
        time.sleep(2 * args.rank)
        args.studyid = check_run_id(args.studyid, args.results_directory, worker=True)

        # printstr = "Starting worker %s" % rank
        # print('\n\x1b[0;33;33m' + printstr + '\x1b[0m\n')

        w = TransitClassifier(args, worker_id_custom=args.rank, run_id=args.studyid, host=host)
        w.load_nameserver_credentials(working_directory=args.results_directory)
        w.run(background=False)
        exit(0)

    # args.run_id = check_run_id(args.run_id, shared_directory)
    args.studyid = check_run_id(args.studyid, args.results_directory)

    # Start nameserver:
    name_server = hpns.NameServer(run_id=args.studyid, host=host, port=0, working_directory=args.results_directory)
    ns_host, ns_port = name_server.start()

    # start worker on master node  ~optimizer is inexpensive, so can afford to run worker alongside optimizer
    w = TransitClassifier(args, worker_id_custom=args.rank, run_id=args.studyid, host=host, nameserver=ns_host,
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

        # run BOHB/BO study
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

    # analyse and save results
    analyze_results(res, args, args.results_directory, args.studyid)


if __name__ == '__main__':

    rank = MPI.COMM_WORLD.rank
    size = MPI.COMM_WORLD.size
    print('Rank (size) = {} ({})'.format(rank, size))
    
    num_gpus = 1  # number of GPUs per node

    if rank == 0:
        print('Number of GPUs selected per node = {}'.format(num_gpus))
    
    gpu_id = rank % num_gpus
    print('GPU ID for rank {}: {}'.format(rank, gpu_id))    

    # print('rank', rank, 'gpu_id', gpu_id)  # , os.environ['CUDA_VISIBLE_DEVICES'])
    # physical_devices = tf.config.list_physical_devices('GPU')
    # print('List of GPUs available to rank {}: {}'.format(rank, physical_devices))

    # tf.config.experimental.set_memory_growth(physical_devices[gpu_id], True)
    # tf.config.set_visible_devices(physical_devices[gpu_id], 'GPU')
    # logical_devices = tf.config.list_logical_devices('GPU')
    # print('GPU(s) selected for rank {}: {}'.format(rank, logical_devices))
    
    optimizer = 'bohb'  # types of hyperparameter optimizers available: 'random_search', 'bohb'

    # if minimum and maximum budgets are set to the same value, then BOHB becomes BO (Bayesian optimization)
    min_budget = 6
    max_budget = 50
    n_iterations = 400

    # number of models trained per configuration evaluated on a given budget
    # used to decrease the variability due to random weights initialization
    ensemble_n = 3

    # metric used to evaluate the performance of a given configuration on the validation set
    hpo_loss = 'auc_pr'

    # BOHB and BO parameters; check [1]
    bohb_params = {'top_n_percent': 15, 'num_samples': 64, 'random_fraction': 1/3, 'bandwidth_factor': 3,
                   'min_bandwidth': 1e-3}
    eta = 2  # Down sampling rate, must be greater or equal to 2

    study = 'hpo_test2'  # name of the HPO study

    # base model used - check estimator_util.py to see which models are implemented
    BaseModel = CNN1dPlanetFinderv2
    config = {'branches': ['global_flux_view_fluxnorm',
                           'local_flux_view_fluxnorm',
                           'local_flux_oddeven_views',
                           'global_centr_view_std_noclip',
                           'local_centr_view_std_noclip',
                           # 'local_weak_secondary_view_fluxnorm',
                           'local_weak_secondary_view_selfnorm'
                           ]
              }

    nic_name = 'lo'  # 'ib0' or 'lo'; 'ib0' to run on the supercomputer, 'lo' to run on a local host

    # features to be extracted from the dataset
    # # features names - keywords used in the TFRecords
    # features_names = ['global_flux_view_fluxnorm',
    #                   'local_flux_view_fluxnorm',
    #                   'local_weak_secondary_view_fluxnorm',
    #                   'local_flux_odd_view_fluxnorm',
    #                   'local_flux_even_view_fluxnorm',
    #                   'local_centr_view_std_noclip',
    #                   'global_centr_view_std_noclip'
    #                  ]  # time-series features names
    # # features dimension
    # # features_dim = {feature_name: (2001, 1) if 'global' in feature_name else (201, 1)
    # #                 for feature_name in features_names}
    # features_dim = {feature_name: (301, 1) if 'global' in feature_name else (31, 1)
    #                 for feature_name in features_names}
    # features_names.append('scalar_params')  # add scalar features
    # features_dim['scalar_params'] = (13,)
    # # features data types
    # features_dtypes = {feature_name: tf.float32 for feature_name in features_names}
    # features_set = {feature_name: {'dim': features_dim[feature_name], 'dtype': features_dtypes[feature_name]}
    #                 for feature_name in features_names}

    features_set = {
        'global_flux_view_fluxnorm': {'dim': (301, 1), 'dtype': tf.float32},
        'local_flux_view_fluxnorm': {'dim': (31, 1), 'dtype': tf.float32},
        'transit_depth_norm': {'dim': (1,), 'dtype': tf.float32},
        # 'global_centr_fdl_view_norm': {'dim': (301, 1), 'dtype': tf.float32},
        # 'local_centr_fdl_view_norm': {'dim': (31, 1), 'dtype': tf.float32},
        'local_flux_odd_view_fluxnorm': {'dim': (31, 1), 'dtype': tf.float32},
        'local_flux_even_view_fluxnorm': {'dim': (31, 1), 'dtype': tf.float32},
        'global_centr_view_std_noclip': {'dim': (301, 1), 'dtype': tf.float32},
        'local_centr_view_std_noclip': {'dim': (31, 1), 'dtype': tf.float32},
        # 'local_weak_secondary_view_fluxnorm': {'dim': (31, 1), 'dtype': tf.float32},
        'local_weak_secondary_view_selfnorm': {'dim': (31, 1), 'dtype': tf.float32},
        'wst_depth_norm': {'dim': (1,), 'dtype': tf.float32},
        'tce_maxmes_norm': {'dim': (1,), 'dtype': tf.float32},
        # 'tce_albedo_norm': {'dim': (1,), 'dtype': tf.float32},
        # 'tce_ptemp_norm': {'dim': (1,), 'dtype': tf.float32},
        'tce_albedo_stat_norm': {'dim': (1,), 'dtype': tf.float32},
        'tce_ptemp_stat_norm': {'dim': (1,), 'dtype': tf.float32},
        'tce_fwm_stat_norm': {'dim': (1,), 'dtype': tf.float32},
        'tce_dikco_msky_norm': {'dim': (1,), 'dtype': tf.float32},
        'tce_dikco_msky_err_norm': {'dim': (1,), 'dtype': tf.float32},
        'tce_dicco_msky_norm': {'dim': (1,), 'dtype': tf.float32},
        'tce_dicco_msky_err_norm': {'dim': (1,), 'dtype': tf.float32},
        'boot_fap_norm': {'dim': (1,), 'dtype': tf.float32},
        'tce_cap_stat_norm': {'dim': (1,), 'dtype': tf.float32},
        'tce_hap_stat_norm': {'dim': (1,), 'dtype': tf.float32},
        'tce_rb_tcount0_norm': {'dim': (1,), 'dtype': tf.float32},
        'tce_sdens_norm': {'dim': (1,), 'dtype': tf.float32},
        'tce_steff_norm': {'dim': (1,), 'dtype': tf.float32},
        'tce_smet_norm': {'dim': (1,), 'dtype': tf.float32},
        'tce_slogg_norm': {'dim': (1,), 'dtype': tf.float32},
        'tce_smass_norm': {'dim': (1,), 'dtype': tf.float32},
        'tce_sradius_norm': {'dim': (1,), 'dtype': tf.float32},
    }

    # extract from the scalar features Tensor only the features matching these indexes; if None uses all of them
    scalar_params_idxs = None  # [0, 1, 2, 3, 7, 8, 9, 10, 11, 12]

    # data directory
    tfrec_dir = os.path.join(paths.path_tfrecs, 
                             'Kepler',
                             'Q1-Q17_DR25',
                             'tfrecordskeplerdr25-dv_g301-l31_6tr_spline_nongapped_flux-loe-lwks-centroid-centroid_fdl-6stellar-bfap-ghost-rollingband-stdtimeseries_secsymphase_wksnorm_maxflux-wks_correctprimarygapping_data/tfrecordskeplerdr25-dv_g301-l31_6tr_spline_nongapped_flux-loe-lwks-centroid-centroidfdl-6stellar-bfap-ghost-rollingband_secsymphase_wksnorm_maxflux-wks_correctprimarygapping_starshuffle_experiment-labels-norm_nopps_ptempstat-albedostat-wstdepth-fwmstat'
                             )

    multi_class = False  # multiclass classification
    ce_weights_args = {'tfrec_dir': tfrec_dir, 'datasets': ['train'], 'label_fieldname': 'label', 'verbose': False}
    use_kepler_ce = False  # use weighted CE loss based on the class proportions in the training set
    satellite = 'kepler'  # if 'kepler' in tfrec_dir else 'tess'

    # add dataset parameters
    config = config_keras.add_dataset_params(satellite, multi_class, use_kepler_ce, ce_weights_args, config=config)

    # add fit parameters
    config['callbacks_list'] = []
    config['data_augmentation'] = False

    # add missing architecture parameters in hpo with default values
    config = config_keras.add_default_missing_params(config=config)
    config['non_lin_fn'] = 'prelu'

    # print('Base configuration used: ', config)

    # previous run directory; used to warmup start model based optimizers
    prev_run_study = ''
    prev_run_dir = None  # os.path.join(paths.path_hpoconfigs, prev_run_study)

    # directory in which the results are saved
    results_directory = os.path.join(paths.path_hpoconfigs, study)

    verbose = 1

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

    args.scalar_params_idxs = scalar_params_idxs

    args.BaseModel = BaseModel

    args.config = config

    args.rank = rank

    args.gpu_id = gpu_id

    args.verbose = verbose

    # args.num_gpus = num_gpus

    run_main(args, bohb_params)
