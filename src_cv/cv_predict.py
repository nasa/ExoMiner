""" Run inference on a data set using trained models from cross-validation experiment. """

# 3rd party
import os
from pathlib import Path
import numpy as np
import logging
import tensorflow as tf
import time
import argparse
import sys
from mpi4py import MPI
import multiprocessing
import yaml
import pandas as pd

# local
from models.models_keras import ExoMiner_JointLocalFlux
from src_hpo import utils_hpo
from utils.utils_dataio import is_yamlble
from src.utils_train_eval_predict import predict_model
from src_preprocessing.normalize_data_tfrecords import normalize_examples
from src.utils_dataio import get_data_from_tfrecord


def cv_pred_run(run_params, cv_iter_dir):
    """ Run one iteration of CV.

    :param run_params: dict, configuration parameters for the CV ru
    :param cv_iter_dir: Path, CV iteration directory.

    :return:
    """

    run_params['paths']['experiment_dir'] = run_params['paths']['experiment_root_dir'] / \
                                            f'cv_iter_{run_params["cv_id"]}'
    run_params['paths']['experiment_dir'].mkdir(exist_ok=True)

    # process data before feeding it to the model (e.g., normalize data based on training set statistics
    if run_params['logger'] is not None:
        run_params['logger'].info(f'[cv_iter_{run_params["cv_id"]}] Normalizing the data')
    norm_data_dir = run_params['paths']['experiment_dir'] / 'norm_data'
    norm_data_dir.mkdir(exist_ok=True)

    # load normalization statistics
    norm_stats_dir = cv_iter_dir / 'norm_stats'
    norm_stats = {}
    if run_params['compute_norm_stats_params']['timeSeriesFDLList'] is not None:
        norm_stats.update({'fdl_centroid': np.load(norm_stats_dir /
                                'train_fdlcentroid_norm_stats.npy', allow_pickle=True).item()})
    if run_params['compute_norm_stats_params']['centroidList'] is not None:
        norm_stats.update({'centroid': np.load(norm_stats_dir /
                            'train_centroid_norm_stats.npy', allow_pickle=True).item()})
    if run_params['compute_norm_stats_params']['scalarParams'] is not None:
        scalar_params_norm_info = np.load(norm_stats_dir /
                                 'train_scalarparam_norm_stats.npy', allow_pickle=True).item()
        scalar_params_norm_info = {k: v for k, v in scalar_params_norm_info.items()
                                   if k in run_params['compute_norm_stats_params']['scalarParams']}
        norm_stats.update({'scalar_params': scalar_params_norm_info})

    # normalize the data
    if len(norm_stats) == 0:
        run_params['logger'].info(f'[cv_iter_{run_params["cv_id"]}] Data cannot be normalized since no normalization '
                                  f'statistics were loaded.')
        raise ValueError(f'[cv_iter_{run_params["cv_id"]}] Data cannot be normalized since no normalization '
                                  f'statistics were loaded.')

    pool = multiprocessing.Pool(processes=run_params['norm_examples_params']['n_processes_norm_data'])
    jobs = [(norm_data_dir, file, norm_stats, run_params['norm_examples_params']['aux_params'])
            for file in run_params['data_shards_fps']]
    async_results = [pool.apply_async(normalize_examples, job) for job in jobs]
    pool.close()
    for async_result in async_results:
        async_result.get()

    run_params['datasets_fps'] = {'predict': [norm_data_dir / data_fp.name
                                              for data_fp in run_params['data_shards_fps']]}

    # instantiate variable to get data from the TFRecords
    if run_params['logger'] is not None:
        run_params['logger'].info(f'[cv_iter_{run_params["cv_id"]}] Getting data from TFRecords to be added to '
                                  f'ranking table...')
    data = {'predict': {field: [] for field in run_params['data_fields']}}
    for tfrec_fp in run_params['datasets_fps']['predict']:
        # get dataset of the TFRecord
        data_aux = get_data_from_tfrecord(tfrec_fp, run_params['data_fields'], run_params['label_map'])
        for field in data_aux:
            data['predict'][field].extend(data_aux[field])

    if run_params['logger'] is not None:
        run_params['logger'].info(f'[cv_iter_{run_params["cv_id"]}] Running inference...')
    # get the file paths for the trained models
    models_dir = cv_iter_dir / 'models'
    run_params['paths']['models_filepaths'] = [model_dir / f'{model_dir.stem}.keras'
                                               for model_dir in models_dir.iterdir() if 'model' in model_dir.stem]

    # run inference for the data set defined in run_params['datasets']
    scores = predict_model(run_params)
    scores_classification = {dataset: np.zeros(scores[dataset].shape, dtype='uint8')
                             for dataset in run_params['datasets']}

    # threshold for classification
    if not run_params['config']['multi_class']:
        scores_classification['predict'][scores['predict'] >= run_params['metrics']['clf_thr']] = 1
    else:
        scores_classification['predict'] = [scores['predict'][i].argmax() for i in range(scores['predict'].shape[0])]

    # add predictions to the data dict
    if run_params['logger'] is not None:
        run_params['logger'].info(f'[cv_iter_{run_params["cv_id"]}] Create ranking table...')
    if not run_params['config']['multi_class']:
        data['predict']['score'] = scores['predict'].ravel()
        data['predict']['predicted class'] = scores_classification['predict'].ravel()
    else:
        for class_label, label_id in run_params['label_map'].items():
            data['predict'][f'score_{class_label}'] = scores['predict'][:, label_id]
        data['predict']['predicted class'] = scores_classification['predict']

    # write results to a csv file
    for dataset in run_params['datasets']:

        data_df = pd.DataFrame(data[dataset])
        # add label id
        data_df['label_id'] = data_df[run_params['label_field_name']].apply(lambda x: run_params['label_map'][x])

        # sort in descending order of output
        if not run_params['config']['multi_class']:
            data_df.sort_values(by='score', ascending=False, inplace=True)
        data_df.to_csv(run_params["paths"]["experiment_dir"] / f'ensemble_ranked_predictions_{dataset}set.csv',
                       index=False)

    if run_params['logger'] is not None:
        run_params['logger'].info(f'Finished CV iteration {run_params["cv_id"]}.')


def cv_pred():
    """ Main script. Run inference on a data set using trained models from a CV experiment. """

    # used in job arrays
    parser = argparse.ArgumentParser()
    parser.add_argument('--job_idx', type=int, help='Job index', default=0)
    parser.add_argument('--config_file', type=str, help='File path to YAML configuration file.',
                        default='/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/codebase/src_cv/config_cv_predict.yaml')
    args = parser.parse_args()

    with(open(args.config_file, 'r')) as file:
        config = yaml.safe_load(file)

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
    config['paths']['experiment_root_dir'].mkdir(exist_ok=True)

    # cv iterations dictionary
    config['cv_iters'] = [fp for fp in config['paths']['cv_experiment_dir'].iterdir() if fp.is_dir()
                          and fp.name.startswith('cv_iter')]

    # file paths to TFRecord data set to be run inference on
    # config['data_shards_fps'] = [fp for fp in config['paths']['tfrec_dir'].iterdir()
    #                              if fp.is_file() and fp.name.startswith('shard') and fp.suffix != '.csv']
    config['data_shards_fps'] = [fp for fp in config['paths']['tfrec_dir'].iterdir()
                                 if 'shard' in fp.name and fp.is_file() and fp.suffix != '.csv']
    if config["rank"] >= len(config['cv_iters']):
        return

    # set up logger
    config['logger'] = logging.getLogger(name=f'cv_pred_run_rank_{config["rank"]}')
    logger_handler = logging.FileHandler(filename=config['paths']['experiment_root_dir'] /
                                                  f'cv_run_{config["rank"]}.log',
                                         mode='w')
    logger_formatter = logging.Formatter('%(asctime)s - %(message)s')
    config['logger'].setLevel(logging.INFO)
    logger_handler.setFormatter(logger_formatter)
    config['logger'].addHandler(logger_handler)
    config['logger'].info(f'Starting run {config["paths"]["experiment_root_dir"].name}...')

    config['dev_train'] = f'/gpu:{config["gpu_id"]}'
    config['dev_predict'] = f'/gpu:{config["gpu_id"]}'

    # overwrite configuration in YAML file with configuration sampled from HPO run
    if config['paths']['hpo_dir'] is not None:
        hpo_path = Path(config['paths']['hpo_dir'])
        res = utils_hpo.logged_results_to_HBS_result(hpo_path, f'_{hpo_path.name}')

        # get ID to config mapping
        id2config = res.get_id2config_mapping()
        # best config - incumbent
        incumbent = res.get_incumbent_id()
        config_id_hpo = incumbent
        config_hpo_chosen = id2config[config_id_hpo]['config']

        # for legacy HPO runs
        config_hpo_chosen = utils_hpo.update_legacy_configs(config_hpo_chosen)

        config['config'].update(config_hpo_chosen)

        # select a specific config based on its ID
        # example - check config.json
        # config = id2config[(0, 0, 0)]['config']

        config['logger'].info(f'Using configuration from HPO study {hpo_path.name}')
        config['logger'].info(f'HPO Config chosen: {config_id_hpo}')
        # save the YAML file with the HPO configuration that was used
        with open(config['paths']['experiment_root_dir'] / 'hpo_config.yaml', 'w') as hpo_config_file:
            yaml.dump(config_hpo_chosen, hpo_config_file, sort_keys=False)

    # base model used - check estimator_util.py to see which models are implemented
    config['base_model'] = ExoMiner_JointLocalFlux

    # choose features set
    for feature_name, feature in config['features_set'].items():
        if feature['dtype'] == 'float32':
            config['features_set'][feature_name]['dtype'] = tf.float32

    if rank == 0:
        # save model configuration used
        np.save(config['paths']['experiment_root_dir'] / 'config.npy', config['config'])

        # save the YAML file with training-evaluation parameters that are YAML serializable
        json_dict = {key: val for key, val in config.items() if is_yamlble(val)}
        with open(config['paths']['experiment_root_dir'] / 'cv_params.yaml', 'w') as cv_run_file:
            yaml.dump(json_dict, cv_run_file, sort_keys=False)

    if config['train_parallel']:
        # run each CV iteration in parallel
        cv_id = config['rank']
        config['logger'].info(f'Running prediction for CV iteration {cv_id} (out of {len(config["cv_iters"])})')
        config['cv_id'] = cv_id
        cv_pred_run(
            config,
            config['cv_iters'][cv_id],
        )
    else:
        # run each CV iteration sequentially
        for cv_id, cv_iter in enumerate(config['cv_iters']):
            config['logger'].info(f'[cv_iter_{cv_id}] Running prediction for CV iteration {cv_id} '
                                  f'(out of {len(config["cv_iters"])})')
            config['cv_id'] = cv_id
            cv_pred_run(
                config,
                cv_iter
            )


if __name__ == '__main__':

    cv_pred()
