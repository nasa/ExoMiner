""" Run cross-validation experiment. """

# 3rd party
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path
import numpy as np
import logging
# from datetime import datetime
import tensorflow as tf
from tensorflow.keras import callbacks
# import itertools
import copy
# import shutil
import time
import argparse
import sys
from mpi4py import MPI
import json
import multiprocessing

# local
import paths
from src.models_keras import CNN1dPlanetFinderv1, Astronet, Exonet, CNN1dPlanetFinderv2
import src.config_keras
from src_hpo import utils_hpo
from utils.utils_dataio import is_jsonable
from src_cv.utils_cv import processing_data_run, train_model, predict_ensemble


def cv_run(cv_dir, data_shards_fps, run_params, logger=None):
    """ Run one iteration of CV.

    :param cv_dir: Path, directory for the CV iteration
    :param data_shards_fps: dict, 'train' and 'test' keys with TFRecords folds used as training and test sets,
    respectively, for this CV iteration
    :param run_params: dict, configuration parameters for the CV ru
    :param logger: logger for the CV run
    :return:
    """

    cv_run_dir = cv_dir / f'cv_iter_{run_params["cv_id"]}'
    cv_run_dir.mkdir(exist_ok=True)

    # split training folds into training and validation sets by randomly selecting one of the folds as the validation
    # set
    data_shards_fps_eval = copy.deepcopy(data_shards_fps)
    data_shards_fps_eval['val'] = run_params['rng'].choice(data_shards_fps['train'], 1, replace=False)
    data_shards_fps_eval['train'] = np.setdiff1d(data_shards_fps['train'], data_shards_fps_eval['val'])

    logger.info(f'[cv_iter_{run_params["cv_id"]}] Split for CV iteration: {data_shards_fps_eval}')

    with open(cv_run_dir / 'fold_split.json', 'w') as cv_run_file:
        json.dump({dataset: [str(fp) for fp in data_shards_fps_eval[dataset]] for dataset in data_shards_fps_eval},
                  cv_run_file)

    # process data before feeding it to the model (e.g., normalize data based on training set statistics
    logger.info(f'[cv_iter_{run_params["cv_id"]}] Normalizing data for CV iteration')
    data_shards_fps_eval = processing_data_run(data_shards_fps_eval, run_params, cv_run_dir, logger)

    # sequential training
    for model_id in range(run_params['n_models']): # train 10 models
        logger.info(f'[cv_iter_{run_params["cv_id"]}] Training model {model_id + 1} out of {run_params["n_models"]} on '
                    f'{run_params["n_epochs"]} epochs...')

        # with tf.device(run_params['dev_train']):
        #     train_model(model_id,
        #                 run_params['base_model'],
        #                 run_params['n_epochs'],
        #                 run_params['config'],
        #                 run_params['features_set'],
        #                 {dataset: fps for dataset, fps in data_shards_fps_eval.items() if
        #                  dataset != 'test'},
        #                 cv_run_dir,
        #                 run_params['online_preproc_params'],
        #                 run_params['data_augmentation'],
        #                 run_params['callbacks_dict'],
        #                 run_params['opt_metric'],
        #                 run_params['min_optmetric'],
        #                 logger,
        #                 run_params['verbose'],
        #                 )
        p = multiprocessing.Process(target=train_model,
                                    args=(
                                        model_id,
                                        run_params['base_model'],
                                        run_params['n_epochs'],
                                        run_params['config'],
                                        run_params['features_set'],
                                        {dataset: fps for dataset, fps in data_shards_fps_eval.items() if
                                         dataset != 'test'},
                                        cv_run_dir,
                                        run_params['online_preproc_params'],
                                        run_params['data_augmentation'],
                                        run_params['callbacks_dict'],
                                        run_params['opt_metric'],
                                        run_params['min_optmetric'],
                                        logger,
                                        run_params['verbose'],
                                    ))
        p.start()
        p.join()
        # p.close()

    logger.info(f'[cv_iter_{run_params["cv_id"]}] Evaluating ensemble')
    # get the filepaths for the trained models
    models_dir = cv_run_dir / 'models'
    models_filepaths = [model_dir / f'{model_dir.stem}.h5' for model_dir in models_dir.iterdir() if 'model' in
                        model_dir.stem]

    # evaluate ensemble
    # with tf.device(run_params['dev_predict']):
        # predict_ensemble(models_filepaths,
        #                  run_params['config'],
        #                  run_params['features_set'],
        #                  data_shards_fps_eval,
        #                  run_params['data_fields'],
        #                  run_params['generate_csv_pred'],
        #                  cv_run_dir,
        #                  logger,
        #                  run_params['verbose'],
        #                  )
    p = multiprocessing.Process(target=predict_ensemble,
                                args=(
                                    models_filepaths,
                                    run_params['config'],
                                    run_params['features_set'],
                                    data_shards_fps_eval,
                                    run_params['data_fields'],
                                    run_params['generate_csv_pred'],
                                    cv_run_dir,
                                    logger,
                                    run_params['verbose'],
                                ))
    p.start()
    p.join()


    # logger.info(f'[cv_iter_{run_params["cv_id"]}] Deleting normalized data')
    # # remove preprocessed data for this run
    # shutil.rmtree(cv_run_dir / 'norm_data')
    # # TODO: delete the models as well?


def cv():

    # used in job arrays
    parser = argparse.ArgumentParser()
    parser.add_argument('--job_idx', type=int, help='Job index', default=0)
    args = parser.parse_args()

    ngpus_per_node = 1  # number of GPUs per node

    # uncomment for MPI multiprocessing
    rank = MPI.COMM_WORLD.rank
    size = MPI.COMM_WORLD.size
    print(f'Rank = {rank}/{size - 1}')

    rank = ngpus_per_node * args.job_idx + rank
    print(f'Rank* = {rank}')
    sys.stdout.flush()
    if rank != 0:
        time.sleep(2)

    try:
        print(f'[rank_{rank}] CUDA DEVICE ORDER: {os.environ["CUDA_DEVICE_ORDER"]}')
        print(f'[rank_{rank}] CUDA VISIBLE DEVICES: {os.environ["CUDA_VISIBLE_DEVICES"]}')
    except:
        print(f'[rank_{rank}] No CUDA environment variables exist.')

    # n_gpus = len(os.environ["CUDA_VISIBLE_DEVICES"].split(','))  # number of GPUs visible to the process
    if rank == 0:
        print(f'Number of GPUs selected per node = {ngpus_per_node}')
    gpu_id = rank % ngpus_per_node

    # setting GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)  # "0, 1"

    print(f'[rank_{rank}] CUDA DEVICE ORDER: {os.environ["CUDA_DEVICE_ORDER"]}')
    print(f'[rank_{rank}] CUDA VISIBLE DEVICES: {os.environ["CUDA_VISIBLE_DEVICES"]}')

    # tf.debugging.set_log_device_placement(True)

    run_params = {}

    # name of the experiment
    run_params['cv_experiment'] = 'test_cv_kepler'

    # cv root directory
    run_params['cv_dir'] = Path('/data5/tess_project/Data/tfrecords/Kepler/Q1-Q17_DR25/cv/cv_08-10-2021_06-47/')

    # data directory
    run_params['data_dir'] = run_params['cv_dir'] / 'tfrecords'

    # cv experiment directory
    run_params['cv_experiment_dir'] = Path(paths.path_cv_experiments) / run_params['cv_experiment']
    run_params['cv_experiment_dir'].mkdir(exist_ok=True)

    # cv iterations dictionary
    run_params['data_shards_fns'] = np.load(run_params['cv_dir'] / 'cv_folds_runs.npy', allow_pickle=True)
    run_params['data_shards_fps'] = [{dataset: [run_params['data_dir'] / fold for fold in cv_iter[dataset]]
                                      for dataset in cv_iter} for cv_iter in run_params['data_shards_fns']]

    if rank >= len(run_params['data_shards_fps']):
        return

    # set up logger
    logger = logging.getLogger(name=f'cv_run_rank_{rank}')
    logger_handler = logging.FileHandler(filename=run_params['cv_experiment_dir'] / f'cv_run_{rank}.log', mode='w')
    # logger_handler_stream = logging.StreamHandler(sys.stdout)
    # logger_handler_stream.setLevel(logging.INFO)
    logger_formatter = logging.Formatter('%(asctime)s - %(message)s')
    logger.setLevel(logging.INFO)
    logger_handler.setFormatter(logger_formatter)
    # logger_handler_stream.setFormatter(logger_formatter)
    logger.addHandler(logger_handler)
    # logger.addHandler(logger_handler_stream)
    logger.info(f'Starting run {run_params["cv_experiment"]}...')

    run_params['ngpus_per_node'] = ngpus_per_node
    run_params['gpu_id'] = rank % run_params['ngpus_per_node']
    run_params['dev_train'] = f'/gpu:{run_params["gpu_id"]}'
    run_params['dev_predict'] = f'/gpu:{run_params["gpu_id"]}'

    run_params['n_processes_norm_data'] = 15
    run_params['verbose'] = True

    run_params['rnd_seed'] = 2
    run_params['rng'] = np.random.default_rng(seed=run_params['rnd_seed'])

    # normalization information for the scalar parameters
    run_params['scalar_params'] = {
        # stellar parameters
        'tce_steff': {'missing_value': np.nan, 'log_transform': False, 'log_transform_eps': np.nan,
                      'clip_factor': 20, 'dtype': 'int', 'replace_value': None},
        'tce_slogg': {'missing_value': np.nan, 'log_transform': False, 'log_transform_eps': np.nan,
                      'clip_factor': 20, 'dtype': 'float', 'replace_value': None},
        'tce_smet': {'missing_value': np.nan, 'log_transform': False, 'log_transform_eps': np.nan,
                     'clip_factor': 20, 'dtype': 'float', 'replace_value': None},
        'tce_sradius': {'missing_value': np.nan, 'log_transform': False, 'log_transform_eps': np.nan,
                        'clip_factor': 20, 'dtype': 'float', 'replace_value': None},
        'tce_smass': {'missing_value': np.nan, 'log_transform': False, 'log_transform_eps': np.nan,
                      'clip_factor': 20, 'dtype': 'float', 'replace_value': None},
        'tce_sdens': {'missing_value': np.nan, 'log_transform': False, 'log_transform_eps': np.nan,
                      'clip_factor': 20, 'dtype': 'float', 'replace_value': None},
        # secondary
        'tce_maxmes': {'missing_value': np.nan, 'log_transform': False, 'log_transform_eps': np.nan,
                       'clip_factor': 20, 'dtype': 'float', 'replace_value': None},
        'wst_depth': {'missing_value': 0, 'log_transform': False, 'log_transform_eps': np.nan,
                      'clip_factor': 20, 'dtype': 'float', 'replace_value': 0},
        'tce_albedo_stat': {'missing_value': 0, 'log_transform': False, 'log_transform_eps': np.nan, 'clip_factor': 20,
                            'dtype': 'float', 'replace_value': None},
        'tce_ptemp_stat': {'missing_value': 0, 'log_transform': False, 'log_transform_eps': np.nan, 'clip_factor': 20,
                           'dtype': 'float', 'replace_value': None},
        # other diagnostics
        'boot_fap': {'missing_value': -1, 'log_transform': True, 'log_transform_eps': 1e-32,
                     'clip_factor': np.nan, 'dtype': 'float', 'replace_value': None},
        'tce_cap_stat': {'missing_value': np.nan, 'log_transform': False, 'log_transform_eps': np.nan,
                         'clip_factor': 20, 'dtype': 'float', 'replace_value': None},
        'tce_hap_stat': {'missing_value': np.nan, 'log_transform': False, 'log_transform_eps': np.nan,
                         'clip_factor': 20, 'dtype': 'float', 'replace_value': None},
        'tce_rb_tcount0': {'missing_value': np.nan, 'log_transform': False, 'log_transform_eps': np.nan,
                           'clip_factor': np.nan, 'dtype': 'int', 'replace_value': None},
        # centroid
        'tce_fwm_stat': {'missing_value': 0, 'log_transform': False, 'log_transform_eps': np.nan, 'clip_factor': 20,
                         'dtype': 'float', 'replace_value': None},
        'tce_dikco_msky': {'missing_value': 0, 'log_transform': False, 'log_transform_eps': np.nan,
                           'clip_factor': 20, 'dtype': 'float', 'replace_value': None},
        'tce_dicco_msky': {'missing_value': 0, 'log_transform': False, 'log_transform_eps': np.nan,
                           'clip_factor': 20, 'dtype': 'float', 'replace_value': None},
        'tce_dikco_msky_err': {'missing_value': -1, 'log_transform': False, 'log_transform_eps': np.nan,
                               'clip_factor': 20, 'dtype': 'float', 'replace_value': None},
        'tce_dicco_msky_err': {'missing_value': -1, 'log_transform': False, 'log_transform_eps': np.nan,
                               'clip_factor': 20, 'dtype': 'float', 'replace_value': None},
        # TCE parameters
        'transit_depth': {'missing_value': None, 'log_transform': False, 'log_transform_eps': np.nan,
                          'clip_factor': 20, 'dtype': 'float', 'replace_value': None},
        'tce_period': {'missing_value': None, 'log_transform': False, 'log_transform_eps': np.nan,
                       'clip_factor': 20, 'dtype': 'float', 'replace_value': None},
        'tce_prad': {'missing_value': None, 'log_transform': False, 'log_transform_eps': np.nan,
                     'clip_factor': 20, 'dtype': 'float', 'replace_value': None}
    }

    run_params['norm'] = {
        # 2 * 2.5 + 1,  # 2 * 4 + 1,  # number of transit durations (2*n+1, n on each side of the transit)
        'nr_transit_durations': 6,
        'num_bins_loc': 31,  # 31, 201
        'num_bins_glob': 301  # 301, 2001
    }

    # set configuration manually. Set to None to use a configuration from an HPO study
    config = {}

    # name of the HPO study from which to get a configuration
    hpo_study = 'ConfigK-bohb_keplerdr25-dv_g301-l31_spline_nongapped_starshuffle_norobovetterkois_glflux-' \
                'glcentr_std_noclip-loe-lwks-6stellar-bfap-ghost-rollingband-convscalars_loesubtract'
    # set the configuration from an HPO study
    if hpo_study is not None:
        # hpo_path = Path(paths.path_hpoconfigs) / hpo_study
        hpo_path = Path('/data5/tess_project/experiments/hpo_configs/') / 'experiments_paper(9-14-2020_to_1-19-2021)' / hpo_study
        res = utils_hpo.logged_results_to_HBS_result(hpo_path, f'_{hpo_study}')

        # get ID to config mapping
        id2config = res.get_id2config_mapping()
        # best config - incumbent
        incumbent = res.get_incumbent_id()
        config_id_hpo = incumbent
        config = id2config[config_id_hpo]['config']

        # select a specific config based on its ID
        # example - check config.json
        # config = id2config[(0, 0, 0)]['config']

        logger.info(f'Using configuration from HPO study {hpo_study}')
        logger.info(f'HPO Config {config_id_hpo}: {config}')

    # base model used - check estimator_util.py to see which models are implemented
    run_params['base_model'] = CNN1dPlanetFinderv2
    config.update({
        # 'num_loc_conv_blocks': 2,
        # 'num_glob_conv_blocks': 5,
        # 'init_fc_neurons': 512,
        # 'num_fc_layers': 4,
        # 'pool_size_loc': 7,
        # 'pool_size_glob': 5,
        # 'pool_stride': 2,
        # 'conv_ls_per_block': 2,
        # 'init_conv_filters': 4,
        # 'kernel_size': 5,
        # 'kernel_stride': 1,
        'non_lin_fn': 'prelu',  # 'relu',
        # 'optimizer': 'Adam',
        # 'lr': 1e-5,
        # 'batch_size': 64,
        # 'dropout_rate': 0,
    })

    # select convolutional branches
    config['branches'] = [
        'global_flux_view_fluxnorm',
        'local_flux_view_fluxnorm',
        # 'global_centr_fdl_view_norm',
        # 'local_centr_fdl_view_norm',
        'local_flux_oddeven_views',
        'global_centr_view_std_noclip',
        'local_centr_view_std_noclip',
        # 'local_weak_secondary_view_fluxnorm',
        # 'local_weak_secondary_view_selfnorm',
        'local_weak_secondary_view_max_flux-wks_norm'
    ]

    # choose features set
    run_params['features_set'] = {
        # flux related features
        'global_flux_view_fluxnorm': {'dim': (301, 1), 'dtype': tf.float32},
        'local_flux_view_fluxnorm': {'dim': (31, 1), 'dtype': tf.float32},
        'transit_depth_norm': {'dim': (1,), 'dtype': tf.float32},
        # odd-even views
        'local_flux_odd_view_fluxnorm': {'dim': (31, 1), 'dtype': tf.float32},
        'local_flux_even_view_fluxnorm': {'dim': (31, 1), 'dtype': tf.float32},
        # centroid views
        # 'global_centr_fdl_view_norm': {'dim': (301, 1), 'dtype': tf.float32},
        # 'local_centr_fdl_view_norm': {'dim': (31, 1), 'dtype': tf.float32},
        'global_centr_view_std_noclip': {'dim': (301, 1), 'dtype': tf.float32},
        'local_centr_view_std_noclip': {'dim': (31, 1), 'dtype': tf.float32},
        # secondary related features
        # 'local_weak_secondary_view_fluxnorm': {'dim': (31, 1), 'dtype': tf.float32},
        # 'local_weak_secondary_view_selfnorm': {'dim': (31, 1), 'dtype': tf.float32},
        'local_weak_secondary_view_max_flux-wks_norm': {'dim': (31, 1), 'dtype': tf.float32},
        'tce_maxmes_norm': {'dim': (1,), 'dtype': tf.float32},
        'wst_depth_norm': {'dim': (1,), 'dtype': tf.float32},
        # 'tce_albedo_norm': {'dim': (1,), 'dtype': tf.float32},
        'tce_albedo_stat_norm': {'dim': (1,), 'dtype': tf.float32},
        # 'tce_ptemp_norm': {'dim': (1,), 'dtype': tf.float32},
        'tce_ptemp_stat_norm': {'dim': (1,), 'dtype': tf.float32},
        # centroid related features
        'tce_fwm_stat_norm': {'dim': (1,), 'dtype': tf.float32},
        'tce_dikco_msky_norm': {'dim': (1,), 'dtype': tf.float32},
        'tce_dikco_msky_err_norm': {'dim': (1,), 'dtype': tf.float32},
        'tce_dicco_msky_norm': {'dim': (1,), 'dtype': tf.float32},
        'tce_dicco_msky_err_norm': {'dim': (1,), 'dtype': tf.float32},
        # other diagnostic parameters
        'boot_fap_norm': {'dim': (1,), 'dtype': tf.float32},
        'tce_cap_stat_norm': {'dim': (1,), 'dtype': tf.float32},
        'tce_hap_stat_norm': {'dim': (1,), 'dtype': tf.float32},
        'tce_rb_tcount0_norm': {'dim': (1,), 'dtype': tf.float32},
        # stellar parameters
        'tce_sdens_norm': {'dim': (1,), 'dtype': tf.float32},
        'tce_steff_norm': {'dim': (1,), 'dtype': tf.float32},
        'tce_smet_norm': {'dim': (1,), 'dtype': tf.float32},
        'tce_slogg_norm': {'dim': (1,), 'dtype': tf.float32},
        'tce_smass_norm': {'dim': (1,), 'dtype': tf.float32},
        'tce_sradius_norm': {'dim': (1,), 'dtype': tf.float32},
        # tce parameters
        'tce_prad_norm': {'dim': (1,), 'dtype': tf.float32},
        'tce_period_norm': {'dim': (1,), 'dtype': tf.float32},
    }

    logger.info(f'Feature set: {run_params["features_set"]}')

    run_params['data_augmentation'] = False  # if True, uses online data augmentation in the training set
    run_params['online_preproc_params'] = {'num_bins_global': 301, 'num_bins_local': 31, 'num_transit_dur': 6}

    run_params['n_models'] = 2  # number of models in the ensemble
    run_params['n_epochs'] = 2  # number of epochs used to train each model
    run_params['multi_class'] = False  # multi-class classification
    run_params['ce_weights_args'] = {
        'tfrec_dir': run_params['cv_experiment_dir'] / 'tfrecords',
        'datasets': ['train'],
        'label_fieldname': 'label',
        'verbose': False
    }
    run_params['use_kepler_ce'] = False  # use weighted CE loss based on the class proportions in the training set
    run_params['satellite'] = 'kepler'  # if 'kepler' in tfrec_dir else 'tess'

    run_params['opt_metric'] = 'auc_pr'  # choose which metric to plot side by side with the loss
    run_params['min_optmetric'] = False  # if lower value is better set to True

    # callbacks list
    run_params['callbacks_dict'] = {}

    # early stopping callback
    run_params['callbacks_dict']['early_stopping'] = callbacks.EarlyStopping(
        monitor=f'val_{run_params["opt_metric"]}',
        min_delta=0,
        patience=20,
        verbose=1,
        mode='max',
        baseline=None,
        restore_best_weights=True
    )

    # add dataset parameters
    config = src.config_keras.add_dataset_params(
        run_params['satellite'],
        run_params['multi_class'],
        run_params['use_kepler_ce'],
        run_params['ce_weights_args'],
        config
    )

    # add missing parameters in hpo with default values
    config = src.config_keras.add_default_missing_params(config=config)

    logger.info(f'Final configuration used: {config}')
    run_params['config'] = config

    run_params['data_fields'] = {
        'target_id': 'int_scalar',
              'tce_plnt_num': 'int_scalar',
              # 'oi': 'float_scalar',
              'label': 'string',
              # 'TESS Disposition': 'string',
              'tce_period': 'float_scalar',
              'tce_duration': 'float_scalar',
              'tce_time0bk': 'float_scalar',
              'original_label': 'string',
              'transit_depth': 'float_scalar',
              # 'tce_max_mult_ev': 'float_scalar',
              # 'tce_prad': 'float_scalar',
              # 'sigma_oot_odd': 'float_scalar',
              # 'sigma_it_odd': 'float_scalar',
              # 'sigma_oot_even': 'float_scalar',
              # 'sigma_it_even': 'float_scalar',
              }
    run_params['generate_csv_pred'] = True

    # save feature set used
    if rank == 0:
        np.save(run_params['cv_experiment_dir'] / 'features_set.npy', run_params['features_set'])
        # save configuration used
        np.save(run_params['cv_experiment_dir'] / 'config.npy', run_params['config'])

        # save the JSON file with training-evaluation parameters that are JSON serializable
        json_dict = {key: val for key, val in run_params.items() if is_jsonable(val)}
        with open(run_params['cv_experiment_dir'] / 'cv_params.json', 'w') as cv_run_file:
            json.dump(json_dict, cv_run_file)

    # run each CV iteration sequentially
    for cv_id, cv_iter in enumerate(run_params['data_shards_fps']):
        logger.info(f'[cv_iter_{cv_iter}] Running CV iteration {cv_id} (out of {len(run_params["data_shards_fps"])}): '
                    f'{cv_iter}')
        run_params['cv_id'] = cv_id
        cv_run(
               run_params['cv_experiment_dir'],
               cv_iter,
               run_params,
               logger
               )

    # # run each CV iteration in parallel
    # cv_id = rank
    # logger.info(f'Running CV iteration {cv_id} (out of {len(run_params["data_shards_fps"])}): '
    #             f'{run_params["data_shards_fps"][cv_id]}')
    # run_params['cv_id'] = cv_id
    # cv_run(
    #     run_params['cv_experiment_dir'],
    #     run_params['data_shards_fps'][cv_id],
    #     run_params,
    #     logger
    # )


if __name__ == '__main__':

    cv()
