""" Run cross-validation experiment. """

# 3rd party
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path
import numpy as np
import logging
import tensorflow as tf
from tensorflow.keras import callbacks
import copy
import time
import argparse
import sys
from mpi4py import MPI
import json
import multiprocessing
import yaml
import pandas as pd

# local
from models.models_keras import ExoMiner, TransformerExoMiner
from src_hpo import utils_hpo
from utils.utils_dataio import is_yamlble
from src_cv.utils_cv import processing_data_run
from src.utils_train_eval_predict import train_model, evaluate_model, predict_model
from src.utils_dataio import get_data_from_tfrecord


def cv_run(cv_dir, data_shards_fps, run_params):
    """ Run one iteration of CV.

    :param cv_dir: Path, CV root directory
    :param data_shards_fps: dict, 'train' and 'test' keys with TFRecords folds used as training and test sets,
    respectively, for this CV iteration
    :param run_params: dict, configuration parameters for the CV run
    :return:
    """

    run_params['paths']['experiment_dir'] = cv_dir / f'cv_iter_{run_params["cv_id"]}'
    run_params['paths']['experiment_dir'].mkdir(exist_ok=True)

    # split training folds into training and validation sets by randomly selecting one of the folds as the validation
    # set
    data_shards_fps_eval = copy.deepcopy(data_shards_fps)
    data_shards_fps_eval['val'] = run_params['rng'].choice(data_shards_fps['train'], 1, replace=False)
    data_shards_fps_eval['train'] = np.setdiff1d(data_shards_fps['train'], data_shards_fps_eval['val'])

    if run_params['logger'] is not None:
        run_params['logger'].info(f'[cv_iter_{run_params["cv_id"]}] Split for CV iteration: {data_shards_fps_eval}')

    # save fold used
    with open(run_params['paths']['experiment_dir'] / 'fold_split.json', 'w') as cv_run_file:
        json.dump({dataset: [str(fp) for fp in data_shards_fps_eval[dataset]] for dataset in data_shards_fps_eval},
                  cv_run_file)

    # process data before feeding it to the model (e.g., normalize data based on training set statistics
    if run_params['config'] is not None:
        run_params['logger'].info(f'[cv_iter_{run_params["cv_id"]}] Processing data for CV iteration')
    run_params['datasets_fps'] = processing_data_run(data_shards_fps_eval, run_params,
                                                     run_params['paths']['experiment_dir'])

    # sequential training
    models_dir = run_params['paths']['experiment_dir'] / 'models'
    models_dir.mkdir(exist_ok=True)
    for model_id in range(run_params['training']['n_models']):  # train N models
        if run_params['config'] is not None:
            run_params['logger'].info(f'[cv_iter_{run_params["cv_id"]}] Training model {model_id + 1} '
                                      f'out of {run_params["training"]["n_models"]} on '
                                      f'{run_params["training"]["n_epochs"]} epochs...')
        model_dir = models_dir / f'model{model_id}'
        model_dir.mkdir(exist_ok=True)

        p = multiprocessing.Process(target=train_model,
                                    args=(
                                        run_params['base_model'],
                                        run_params,
                                        model_dir,
                                        model_id,
                                    ))
        p.start()
        p.join()
        # p.close()

    if run_params['config'] is not None:
        run_params['logger'].info(f'[cv_iter_{run_params["cv_id"]}] Evaluating ensemble')
    # get the filepaths for the trained models
    run_params['paths']['models_filepaths'] = [model_dir / f'{model_dir.stem}.h5'
                                               for model_dir in models_dir.iterdir() if 'model' in model_dir.stem]

    res_eval = evaluate_model(run_params)
    np.save(run_params['paths']['experiment_dir'] / 'res_eval.npy', res_eval)

    # write results to a txt file
    with open(run_params['paths']['experiment_dir'] / 'results_ensemble.txt', 'w') as res_file:

        str_aux = f'Performance metrics for the ensemble ({run_params["training"]["n_models"]} models)\n'
        res_file.write(str_aux)

        for dataset in run_params['datasets']:
            if dataset != 'predict':

                res_eval_dataset_metrics_names = [metric_name for metric_name in res_eval.keys()
                                                  if dataset in metric_name]

                str_aux = f'Dataset: {dataset}\n'
                res_file.write(str_aux)

                for metric in res_eval_dataset_metrics_names:
                    if not np.any([el in metric for el in ['prec_thr', 'rec_thr', 'tp', 'fn', 'tn', 'fp']]):
                        str_aux = f'{metric}: {res_eval[f"{metric}"]}\n'
                        res_file.write(str_aux)

            res_file.write('\n')

    scores = predict_model(run_params)
    scores_classification = {dataset: np.zeros(scores[dataset].shape, dtype='uint8')
                             for dataset in run_params['datasets']}
    for dataset in run_params['datasets']:
        # threshold for classification
        if not run_params['config']['multi_class']:
            scores_classification[dataset][scores[dataset] >= run_params['metrics']['clf_thr']] = 1

    # instantiate variable to get data from the TFRecords
    data = {dataset: {field: [] for field in run_params['data_fields']} for dataset in run_params['datasets']}
    for dataset in run_params['datasets']:
        for tfrec_fp in run_params['datasets_fps'][dataset]:
            # get dataset of the TFRecord
            data_aux = get_data_from_tfrecord(tfrec_fp, run_params['data_fields'], run_params['label_map'])
            for field in data_aux:
                data[dataset][field].extend(data_aux[field])

    # add predictions to the data dict
    for dataset in run_params['datasets']:
        if not run_params['config']['multi_class']:
            data[dataset]['score'] = scores[dataset].ravel()
            data[dataset]['predicted class'] = scores_classification[dataset].ravel()
        else:
            for class_label, label_id in run_params['label_map'].items():
                data[dataset][f'score_{class_label}'] = scores[dataset][:, label_id]
            data[dataset]['predicted class'] = scores_classification[dataset]

    # write results to a txt file
    for dataset in run_params['datasets']:
        # print(f'Saving ranked predictions in dataset {dataset} to '
        #       f'{run_params["paths"]["experiment_dir"] / f"ranked_predictions_{dataset}"}...')

        data_df = pd.DataFrame(data[dataset])

        # sort in descending order of output
        if not run_params['config']['multi_class']:
            data_df.sort_values(by='score', ascending=False, inplace=True)
        data_df.to_csv(run_params['paths']['experiment_dir'] / f'ensemble_ranked_predictions_{dataset}set.csv',
                       index=False)

    # if run_params['config'] is not None:
    # run_params['logger'].info(f'[cv_iter_{run_params["cv_id"]}] Deleting normalized data')
    # # remove preprocessed data for this run
    # shutil.rmtree(cv_run_dir / 'norm_data')
    # # TODO: delete the models as well?


def cv():
    """ Run CV experiment. """

    path_to_yaml = Path(
        '/Users/msaragoc/OneDrive - NASA/Projects/exoplanet_transit_classification/codebase/src_cv/config_cv_train.yaml')
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
        if path_str is not None:
            config['paths'][path_name] = Path(path_str)
    config['paths']['experiment_root_dir'].mkdir(exist_ok=True)

    # cv iterations dictionary
    config['data_shards_fns'] = np.load(config['paths']['cv_folds'], allow_pickle=True)
    config['data_shards_fps'] = [{dataset: [config['paths']['tfrec_dir'] / fold for fold in cv_iter[dataset]]
                                  for dataset in cv_iter} for cv_iter in config['data_shards_fns']]

    if config["rank"] >= len(config['data_shards_fps']):
        return

    # set up logger
    config['logger'] = logging.getLogger(name=f'cv_run_rank_{config["rank"]}')
    logger_handler = logging.FileHandler(filename=config['paths']['experiment_root_dir'] /
                                                  f'cv_run_{config["rank"]}.log', mode='w')
    # logger_handler_stream = logging.StreamHandler(sys.stdout)
    # logger_handler_stream.setLevel(logging.INFO)
    logger_formatter = logging.Formatter('%(asctime)s - %(message)s')
    config['logger'].setLevel(logging.INFO)
    logger_handler.setFormatter(logger_formatter)
    # logger_handler_stream.setFormatter(logger_formatter)
    config['logger'].addHandler(logger_handler)
    # logger.addHandler(logger_handler_stream)
    config['logger'].info(f'Starting run {config["paths"]["experiment_root_dir"].name}...')

    config['dev_train'] = f'/gpu:{config["gpu_id"]}'
    config['dev_predict'] = f'/gpu:{config["gpu_id"]}'

    config['rng'] = np.random.default_rng(seed=config['rnd_seed'])

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

    # base model used - check models/models_keras.py to see which models are implemented
    config['base_model'] = TransformerExoMiner  # ExoMiner

    # choose features set
    for feature_name, feature in config['features_set'].items():
        if feature['dtype'] == 'float':
            config['features_set'][feature_name]['dtype'] = tf.float32
        if feature['dtype'] == 'int':
            config['features_set'][feature_name]['dtype'] = tf.int64

    config['logger'].info(f'Feature set: {config["features_set"]}')

    # early stopping callback
    # config['callbacks']['early_stopping']['obj'] = callbacks.EarlyStopping(**config['callbacks']['early_stopping'])
    config['callbacks_list'] = {'train': [callbacks.EarlyStopping(**config['callbacks']['early_stopping'])]}

    config['logger'].info(f'Final configuration used: {config}')

    # save feature set used
    if config['rank'] == 0:
        # np.save(config['paths']['experiment_root_dir'] / 'features_set.npy', config['features_set'])
        # save configuration used
        np.save(config['paths']['experiment_root_dir'] / 'config.npy', config['config'])

        # save the YAML file with training-evaluation parameters that are YAML serializable
        json_dict = {key: val for key, val in config.items() if is_yamlble(val)}
        with open(config['paths']['experiment_root_dir'] / 'cv_params.yaml', 'w') as cv_run_file:
            yaml.dump(json_dict, cv_run_file)

    if config['train_parallel']:
        # run each CV iteration in parallel
        cv_id = config['rank']
        if config['logger'] is not None:
            config['logger'].info(f'Running CV iteration {cv_id} (out of {len(config["data_shards_fps"])}): '
                                  f'{config["data_shards_fps"][cv_id]}')
        config['cv_id'] = cv_id
        cv_run(
            config['paths']['experiment_root_dir'],
            config['data_shards_fps'][cv_id],
            config,
        )
    else:
        # run each CV iteration sequentially
        for cv_id, cv_iter in enumerate(config['data_shards_fps']):
            config['logger'].info(
                f'[cv_iter_{cv_iter}] Running CV iteration {cv_id} (out of {len(config["data_shards_fps"])}): '
                f'{cv_iter}')
            config['cv_id'] = cv_id
            cv_run(
                config['paths']['experiment_root_dir'],
                cv_iter,
                config,
            )


if __name__ == '__main__':
    cv()
