"""
Train models using a given configuration.
"""

# 3rd party
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
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
# import multiprocessing
import yaml
import pandas as pd

# local
from models.models_keras import ExoMiner_JointLocalFlux
from src_hpo import utils_hpo
from utils.utils_dataio import is_yamlble
from src.utils_train_eval_predict import train_model, evaluate_model, predict_model
from src.utils_dataio import get_data_from_tfrecord


def train_single_model(run_params):
    """ Train and evaluate a single model.

    :param run_params: dict, configuration parameters for the run
    :return:
    """

    # instantiate variable to get data from the TFRecords
    data = {dataset: {field: [] for field in run_params['data_fields']} for dataset in run_params['datasets']}
    for dataset in run_params['datasets']:
        for tfrec_fp in run_params['datasets_fps'][dataset]:
            # get dataset of the TFRecord
            data_aux = get_data_from_tfrecord(tfrec_fp, run_params['data_fields'])
            for field in data_aux:
                data[dataset][field].extend(data_aux[field])

    if run_params['logger'] is not None:
        run_params['logger'].info(f'[model_{run_params["model_id"]}] Training model {run_params["model_id"]} '
                                  f'out of {run_params["training"]["n_models"]} on '
                                  f'{run_params["training"]["n_epochs"]} epochs...')
    model_dir = run_params['models_dir'] / f'model{run_params["model_id"]}'
    model_dir.mkdir(exist_ok=True)

    train_model(
        run_params['base_model'],
        run_params,
        model_dir,
        run_params["model_id"],
        run_params['logger'],
    )

    run_params['paths']['models_filepaths'] = [model_dir / f'model{run_params["model_id"]}.keras']
    res_eval = evaluate_model(run_params)
    np.save(model_dir / 'res_eval.npy', res_eval)

    # write results to a txt file
    with open(model_dir / 'results.txt', 'w') as res_file:

        str_aux = f'Performance metrics for the model\n'
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

    # run inference on different data sets defined in run_params['datasets']
    if run_params['logger'] is not None:
        run_params['logger'].info(f'[model_{run_params["model_id"]}] Running inference...')
    scores = predict_model(run_params)
    scores_classification = {dataset: np.zeros(scores[dataset].shape, dtype='uint8')
                             for dataset in run_params['datasets']}
    for dataset in run_params['datasets']:
        # threshold for classification
        if not run_params['config']['multi_class']:  # binary classification
            scores_classification[dataset][scores[dataset] >= run_params['metrics']['clf_thr']] = 1
        else:  # multiclass
            scores_classification[dataset] = [scores[dataset][i].argmax() for i in range(scores[dataset].shape[0])]

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

        data_df = pd.DataFrame(data[dataset])
        # add label id
        data_df['label_id'] = data_df[run_params['label_field_name']].apply(lambda x: run_params['label_map'][x])

        # sort in descending order of output
        if not run_params['config']['multi_class']:
            data_df.sort_values(by='score', ascending=False, inplace=True)
        data_df.to_csv(model_dir / f'ensemble_ranked_predictions_{dataset}set.csv',
                       index=False)

    if run_params['logger'] is not None:
        run_params['logger'].info(f'Finished training model {run_params["model_id"]}.')


def main():
    """ Train and evaluate a single or a set of models on the same data set separately. Then run inference for each. """

    parser = argparse.ArgumentParser()
    parser.add_argument('--job_idx', type=int, help='Job index', default=0)
    parser.add_argument('--config_file', type=str, help='File path to YAML configuration file.',
                        default='/Users/msaragoc/OneDrive - NASA/Projects/exoplanet_transit_classification/codebase/src/config_train.yaml')
    args = parser.parse_args()

    with(open(args.config_file, 'r')) as file:
        config = yaml.safe_load(file)

    config['rank'] = config['ngpus_per_node'] * args.job_idx
    print(f'Rank = {config["rank"]}')
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

    for path_name, path_str in config['paths'].items():
        if path_str is not None:
            config['paths'][path_name] = Path(path_str)

    config['paths']['experiment_dir'].mkdir(exist_ok=True)

    # set dictionary of TFRecord file paths for the different data sets
    config['datasets_fps'] = {dataset: [fp for fp in config['paths']['tfrec_dir'].iterdir()
                                           if fp.name.startswith(dataset)]
                                 for dataset in config['datasets']}

    if config["rank"] >= config['training']['n_models']:
        print(f'Number of processes requested to train models ({config["rank"]}) is higher than the number models that '
              f'was set to be trained ({config["training"]["n_models"]}). Ending process.')
        return

    # set up logger
    config['logger'] = logging.getLogger(name=f'model_{config["rank"]}')
    logger_handler = logging.FileHandler(filename=config['paths']['experiment_dir'] /
                                                  f'model_{config["rank"]}.log', mode='w')
    logger_formatter = logging.Formatter('%(asctime)s - %(message)s')
    config['logger'].setLevel(logging.INFO)
    logger_handler.setFormatter(logger_formatter)
    config['logger'].addHandler(logger_handler)
    config['logger'].info(f'Starting run {config["paths"]["experiment_dir"].name}...')

    config['dev_train'] = f'/gpu:{config["gpu_id"]}'
    config['dev_predict'] = f'/gpu:{config["gpu_id"]}'

    config['rng'] = np.random.default_rng(seed=config['rnd_seed'])

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
        config['logger'].info(f'HPO Config chosen: {config_id_hpo}.')
        # save the YAML file with the HPO configuration that was used
        with open(config['paths']['experiment_dir'] / 'hpo_config.yaml', 'w') as hpo_config_file:
            yaml.dump(config_hpo_chosen, hpo_config_file, sort_keys=False)

    # base model used - check models/models_keras.py to see which models are implemented
    config['base_model'] = ExoMiner_JointLocalFlux

    # choose features set
    for feature_name, feature in config['features_set'].items():
        if feature['dtype'] == 'float':
            config['features_set'][feature_name]['dtype'] = tf.float32
        if feature['dtype'] == 'int':
            config['features_set'][feature_name]['dtype'] = tf.int64

    # early stopping callback
    config['callbacks_list'] = {
        'train': [
            callbacks.EarlyStopping(**config['callbacks']['early_stopping']),
        ],
    }

    config['models_dir'] = config['paths']['experiment_dir'] / 'models'
    config['models_dir'].mkdir(exist_ok=True)
    if config['rank'] == 0:
        # save configuration used
        np.save(config['paths']['experiment_dir'] / 'config.npy', config['config'])

        # save the YAML file with training-evaluation parameters that are YAML serializable
        json_dict = {key: val for key, val in config.items() if is_yamlble(val)}
        with open(config['paths']['experiment_dir'] / 'run_params.yaml', 'w') as cv_run_file:
            yaml.dump(json_dict, cv_run_file, sort_keys=False)

    if config['train_parallel']:  # run each model in parallel
        config['model_id'] = config['rank']
        if config['logger'] is not None:
            config['logger'].info(f'Running model {config["model_id"] + 1} '
                                  f'(out of {config["training"]["n_models"]})')
        train_single_model(
            config,
        )
    else:
        # run each model sequentially
        for model_id in range(config['training']['n_models']):
            config['model_id'] = config['rank']
            if config['logger'] is not None:
                config['logger'].info(
                    f'[model_{config["model_id"]}] Running model {config["model_id"] + 1} '
                    f'(out of {config["training"]["n_models"]})')
            config['model_id'] = model_id
            train_single_model(
                config,
            )


if __name__ == '__main__':

    main()
