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
import time
from mpi4py import MPI
import numpy as np
import logging

logging.basicConfig(level=logging.WARNING)
import tensorflow as tf
from pathlib import Path
import json
import hpbandster.core.nameserver as hpns
from hpbandster.optimizers import BOHB, RandomSearch
import hpbandster.core.result as hpres

# local
from models.models_keras import Astronet
from models import config_keras
from src_hpo.worker_hpo_keras import TransitClassifier
from src_hpo.utils_hpo import analyze_results, json_result_logger, check_run_id
import paths
from utils.utils_dataio import is_jsonable


def run_main(hpo_config):
    """ Run HPO study.

    :param hpo_config: dict, HPO parameters
    :return:
    """

    # for each rank, create a folder to save temporarily the models created for a given run
    hpo_config['model_dir_rank'] = hpo_config['results_directory'] / f'models_rank{hpo_config["rank"]}'
    hpo_config['model_dir_rank'].mkdir(exist_ok=True)

    host = hpns.nic_name_to_host(hpo_config['nic_name'])

    if hpo_config['rank'] != 0:  # workers go here
        # short artificial delay to make sure the nameserver is already running and current run_id is instantiated
        time.sleep(2 * hpo_config['rank'])
        hpo_config['study'] = check_run_id(hpo_config['study'], hpo_config['results_directory'], worker=True)

        # printstr = "Starting worker %s" % rank
        # print('\n\x1b[0;33;33m' + printstr + '\x1b[0m\n')

        w = TransitClassifier(hpo_config, worker_id_custom=hpo_config['rank'], run_id=hpo_config['study'], host=host)
        w.load_nameserver_credentials(working_directory=hpo_config['results_directory'])
        w.run(background=False)
        exit(0)

    # args.run_id = check_run_id(args.run_id, shared_directory)
    hpo_config['study'] = check_run_id(hpo_config['study'], hpo_config['results_directory'])

    # Start nameserver:
    name_server = hpns.NameServer(run_id=hpo_config['study'], host=host, port=0,
                                  working_directory=hpo_config['results_directory'])
    ns_host, ns_port = name_server.start()

    # start worker on master node  ~optimizer is inexpensive, so can afford to run worker alongside optimizer
    w = TransitClassifier(hpo_config, worker_id_custom=hpo_config['rank'], run_id=hpo_config['study'], host=host,
                          nameserver=ns_host, nameserver_port=ns_port)
    w.run(background=True)

    result_logger = json_result_logger(directory=hpo_config['results_directory'], run_id=hpo_config['study'],
                                       overwrite=False)
    # result_logger = hpres.json_result_logger(directory=args.results_directory, overwrite=False)

    # Let us load the src_old run now to use its results to warmstart a new run with slightly
    # different budgets in terms of data points and epochs.
    # Note that the search space has to be identical though!
    # directory must contain a config.json and results.json for the same configuration space.'
    if hpo_config['prev_run_dir'] is not None:
        previous_run = hpres.logged_results_to_HBS_result(hpo_config['prev_run_dir'])
    else:
        previous_run = None

    if hpo_config['optimizer'] == 'bohb':
        # instantiate BOHB or BO study
        hpo = BOHB(configspace=TransitClassifier.get_configspace(),
                   run_id=hpo_config['study'],
                   host=host,
                   nameserver=ns_host,
                   nameserver_port=ns_port,
                   working_directory=str(hpo_config['results_directory']),
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
        if hpo_config['rank'] != 0:
            kde_models_bdgt = hpo.config_generator.kde_models
            kde_models_bdgt_params = {bdgt: dict() for bdgt in kde_models_bdgt}
            for bdgt in kde_models_bdgt:
                for est in kde_models_bdgt[bdgt]:
                    kde_models_bdgt_params[bdgt][est] = [kde_models_bdgt[bdgt][est].data,
                                                         kde_models_bdgt[bdgt][est].data_type,
                                                         kde_models_bdgt[bdgt][est].bw]
            kde_models_bdgt_params['hyperparameters'] = list(hpo.config_generator.configspace._hyperparameters.keys())

            np.save(hpo_config['results_directory'] / 'kde_models_params.npy', kde_models_bdgt_params)

    else:  # run random search
        hpo = RandomSearch(configspace=TransitClassifier.get_configspace(),
                           run_id=hpo_config['study'],
                           host=host,
                           nameserver=ns_host,
                           nameserver_port=ns_port,
                           working_directory=hpo_config['results_directory'],
                           result_logger=result_logger,
                           min_budget=hpo_config['min_budget'],
                           max_budget=hpo_config['max_budget'],
                           eta=hpo_config['eta'],)
        res = hpo.run(n_iterations=hpo_config['n_iterations'])

    # shutdown
    hpo.shutdown(shutdown_workers=True)
    name_server.shutdown()

    # analyse and save results
    analyze_results(res, hpo_config)


if __name__ == '__main__':

    rank = MPI.COMM_WORLD.rank
    size = MPI.COMM_WORLD.size
    print(f'Rank (size) = {rank} ({size})')

    if rank != 0:
        time.sleep(20)

    try:
        print(f'[rank_{rank}] CUDA DEVICE ORDER: {os.environ["CUDA_DEVICE_ORDER"]}')
        print(f'[rank_{rank}] CUDA VISIBLE DEVICES: {os.environ["CUDA_VISIBLE_DEVICES"]}')
    except:
        print(f'[rank_{rank}] No CUDA environment variables exist.')

    # n_gpus = len(os.environ["CUDA_VISIBLE_DEVICES"].split(','))  # number of GPUs visible to the process
    n_gpus = 1
    if rank == 0:
        print(f'Number of GPUs selected per node = {n_gpus}')
    gpu_id = rank % n_gpus

    # setting GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)  # "0, 1"

    print(f'[rank_{rank}] CUDA DEVICE ORDER: {os.environ["CUDA_DEVICE_ORDER"]}')
    print(f'[rank_{rank}] CUDA VISIBLE DEVICES: {os.environ["CUDA_VISIBLE_DEVICES"]}')

    study = 'hpo_test_astronet_oom_config_68_0_2'  # name of the HPO study

    config = {'branches':
        [
            'global_flux_view_fluxnorm',
            'local_flux_view_fluxnorm',
            # 'local_flux_oddeven_views',
            # 'global_centr_view_std_noclip',
            # 'local_centr_view_std_noclip',
            # 'global_centr_fdl_view_norm',
            # 'local_centr_fdl_view_norm',
            # 'local_weak_secondary_view_fluxnorm',
            # 'local_weak_secondary_view_selfnorm',
            # 'local_weak_secondary_view_max_flux-wks_norm',
        ]
    }

    # features to be extracted from the dataset
    features_set = {
        # flux features
        'global_flux_view_fluxnorm': {'dim': (2001, 1), 'dtype': tf.float32},
        'local_flux_view_fluxnorm': {'dim': (201, 1), 'dtype': tf.float32},
        # 'transit_depth_norm': {'dim': (1,), 'dtype': tf.float32},
        # odd and even flux features
        # 'local_flux_odd_view_fluxnorm': {'dim': (31, 1), 'dtype': tf.float32},
        # 'local_flux_even_view_fluxnorm': {'dim': (31, 1), 'dtype': tf.float32},
        # 'sigma_oot_odd': {'dim': (1,), 'dtype': tf.float32},
        # 'sigma_it_odd': {'dim': (1,), 'dtype': tf.float32},
        # 'sigma_oot_even': {'dim': (1,), 'dtype': tf.float32},
        # 'sigma_it_even': {'dim': (1,), 'dtype': tf.float32},
        # weak secondary flux features
        # 'local_weak_secondary_view_fluxnorm': {'dim': (31, 1), 'dtype': tf.float32},
        # 'local_weak_secondary_view_max_flux-wks_norm': {'dim': (31, 1), 'dtype': tf.float32},
        # 'wst_depth_norm': {'dim': (1,), 'dtype': tf.float32},
        # 'tce_maxmes_norm': {'dim': (1,), 'dtype': tf.float32},
        # 'tce_albedo_norm': {'dim': (1,), 'dtype': tf.float32},
        # 'tce_ptemp_norm': {'dim': (1,), 'dtype': tf.float32},
        # 'tce_albedo_stat_norm': {'dim': (1,), 'dtype': tf.float32},
        # 'tce_ptemp_stat_norm': {'dim': (1,), 'dtype': tf.float32},
        # centroid features
        # 'global_centr_view_std_noclip': {'dim': (301, 1), 'dtype': tf.float32},
        # 'local_centr_view_std_noclip': {'dim': (31, 1), 'dtype': tf.float32},
        # 'global_centr_fdl_view_norm': {'dim': (2001, 1), 'dtype': tf.float32},
        # 'local_centr_fdl_view_norm': {'dim': (201, 1), 'dtype': tf.float32},
        # 'tce_fwm_stat_norm': {'dim': (1,), 'dtype': tf.float32},
        # 'tce_dikco_msky_norm': {'dim': (1,), 'dtype': tf.float32},
        # 'tce_dikco_msky_err_norm': {'dim': (1,), 'dtype': tf.float32},
        # 'tce_dicco_msky_norm': {'dim': (1,), 'dtype': tf.float32},
        # 'tce_dicco_msky_err_norm': {'dim': (1,), 'dtype': tf.float32},
        # 'mag_norm': {'dim': (1,), 'dtype': tf.float32},
        # DV diagnostics
        # 'boot_fap_norm': {'dim': (1,), 'dtype': tf.float32},
        # 'tce_cap_stat_norm': {'dim': (1,), 'dtype': tf.float32},
        # 'tce_hap_stat_norm': {'dim': (1,), 'dtype': tf.float32},
        # 'tce_cap_hap_stat_diff_norm': {'dim': (1,), 'dtype': tf.float32},
        # 'tce_rb_tcount0n_norm': {'dim': (1,), 'dtype': tf.float32},
        # stellar parameters
        # 'tce_sdens_norm': {'dim': (1,), 'dtype': tf.float32},
        # 'tce_steff_norm': {'dim': (1,), 'dtype': tf.float32},
        # 'tce_smet_norm': {'dim': (1,), 'dtype': tf.float32},
        # 'tce_slogg_norm': {'dim': (1,), 'dtype': tf.float32},
        # 'tce_smass_norm': {'dim': (1,), 'dtype': tf.float32},
        # 'tce_sradius_norm': {'dim': (1,), 'dtype': tf.float32},
        # tce parameters
        # 'tce_prad_norm': {'dim': (1,), 'dtype': tf.float32},
        # 'tce_period_norm': {'dim': (1,), 'dtype': tf.float32},
    }

    # data directory
    tfrec_dir = os.path.join(paths.path_tfrecs, 
                             'Kepler',
                             'Q1-Q17_DR25',
                             'tfrecordskeplerdr25-dv_g2001-l201_9tr_spline_gapped1-5_flux-loe-lwks-centroid-centroidfdl-6stellar-bfap-ghost-rollband-stdts_secsymphase_correctprimarygapping_confirmedkoiperiod_data/tfrecordskeplerdr25-dv_g2001-l201_9tr_spline_gapped1-5_flux-loe-lwks-centroid-centroidfdl-6stellar-bfap-ghost-rollband-stdts_secsymphase_correctprimarygapping_confirmedkoiperiod_starshuffle_experiment-labels-norm_nopps'
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
    # config['non_lin_fn'] = 'prelu'
    # config['batch_size'] = 64  # for Astronet

    # previous run directory; used to warmup start model based optimizers
    prev_run_study = None

    # directory in which the results are saved
    results_directory = Path(paths.path_hpoconfigs) / study

    sys.stdout.flush()

    hpo_config = {
        'rank': rank,
        'gpu_id': gpu_id,
        'study': study,
        'optimizer': 'bohb',  # types of hyperparameter optimizers available: 'random_search', 'bohb'
        # if minimum and maximum budgets are set to the same value, then BOHB becomes BO (Bayesian optimization)
        'min_budget': 1,  # 6
        'max_budget': 10,  # 50
        'n_iterations': 1,
        # BOHB and BO parameters; check [1]
        'bohb_params': {'top_n_percent': 15,
                        'num_samples': 64,
                        'random_fraction': 1 / 3,
                        'bandwidth_factor': 3,
                        'min_bandwidth': 1e-3
                        },
        'eta': 2,  # down sampling rate, must be greater or equal to 2, hyperband parameter; check [1]
        # metric used to evaluate the performance of a given configuration on the validation set
        'hpo_loss': 'auc_pr',
        # number of models trained per configuration evaluated on a given budget
        # used to decrease the variability due to random weights initialization
        'ensemble_n': 2,
        'nic_name': 'lo',  # 'ib0' or 'lo'; 'ib0' to run on a cluster, 'lo' to run on a local host
        'verbose': 1,
        'experiment_config': config,
        'features_set': features_set,
        'base_model': Astronet,  # CNN1dPlanetFinderv2,
        # 'online_preproc_params': online_preproc_params,
        'filter_data': None,  # data used to filer the datasets; use None if not filtering
        'tfrec_dir': tfrec_dir,
        'results_directory': results_directory,
        # used to warmup start model based optimizers
        'prev_run_dir': None if prev_run_study is None else Path(paths['path_hpo_configs']) / prev_run_study
    }

    if rank == 0:
        hpo_config['results_directory'].mkdir(exist_ok=True)
        (hpo_config['results_directory'] / 'logs').mkdir(exist_ok=True)

        np.save(hpo_config['results_directory'] / 'hpo_config.npy', hpo_config)

        # save the JSON file with hpo parameters that are JSON serializable
        json_dict = {key: val for key, val in hpo_config.items() if is_jsonable(val)}
        with open(hpo_config['results_directory'] / 'hpo_config.json', 'w') as hpo_config_file:
            json.dump(json_dict, hpo_config_file)

    # set up logger
    logger = logging.getLogger(name='hpo_run')
    logger_handler = logging.FileHandler(filename=hpo_config['results_directory'] / f'hpo_run_{rank}.log', mode='w')
    logger_formatter = logging.Formatter('%(asctime)s - %(message)s')
    logger.setLevel(logging.INFO)
    logger_handler.setFormatter(logger_formatter)
    logger.addHandler(logger_handler)
    logger.info(f'Starting run {study}...')
    logger.info(f'HPO run parameters: {hpo_config}')

    run_main(hpo_config)
