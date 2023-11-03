"""
Train a single or multiple models on the same data set using a given configuration. Then evaluate them and run
inference.
"""

# 3rd party
import os
from pathlib import Path
import numpy as np
import logging
from tensorflow.keras import callbacks
import time
import argparse
import yaml
from keras.saving import load_model

# local
from utils.utils_dataio import is_yamlble
from src.utils_train_eval_predict import (evaluate_model, predict_model, write_performance_metrics_to_txt_file,
                                          set_tf_data_type_for_features)
from src.utils_dataio import get_data_from_tfrecords_for_predictions_table


def evaluate_inference_for_trained_model(run_params):
    """ Evaluate a model on labeled data. Run inference with model to generate scores for examples in different data
    sets.

    :param run_params: dict, configuration parameters for the run
    :return:
    """

    # get data from TFRecords files to be displayed in the table with predictions
    datasets_tbls = get_data_from_tfrecords_for_predictions_table(run_params['datasets'],
                                                                  run_params['data_fields'],
                                                                  run_params['datasets_fps'])

    # add label id
    for dataset in datasets_tbls:
        datasets_tbls[dataset]['label_id'] = (
            datasets_tbls[dataset][run_params['label_field_name']].apply(lambda x: run_params['label_map'][x]))

    if run_params['logger'] is not None:
        run_params['logger'].info(f'[model_{run_params["model_id"]}] Training model {run_params["model_id"]} '
                                  f'out of {run_params["training"]["n_models"]} on '
                                  f'{run_params["training"]["n_epochs"]} epochs...')
    # create model directory
    model_dir = run_params['models_dir'] / f'model{run_params["model_id"]}'
    model_dir.mkdir(exist_ok=True)

    # # load model
    # model = load_model(run_params['paths']['models_filepaths'][run_params["model_id"]])

    # evaluate model on the labeled existing data sets
    if run_params['logger'] is not None:
        run_params['logger'].info(f'[model_{run_params["model_id"]}] Evaluating model...')
    res_eval = evaluate_model(run_params, model_dir, run_params['logger'])
    # save performance metrics
    np.save(model_dir / 'res_eval.npy', res_eval)

    # write results to a txt file
    write_performance_metrics_to_txt_file(model_dir, run_params['datasets'], res_eval)

    # switch off plotting model or writing model summary when predicting
    run_params['plot_model'] = False
    run_params['write_model_summary'] = False

    # run inference on different data sets defined in run_params['datasets']
    if run_params['logger'] is not None:
        run_params['logger'].info(f'[model_{run_params["model_id"]}] Running inference...')
    scores = predict_model(run_params, run_params['logger'])

    # add predictions to the data dict
    for dataset in run_params['datasets']:
        if not run_params['config']['multi_class']:
            datasets_tbls[dataset]['score'] = scores[dataset].ravel()
            # data[dataset]['predicted_class'] = scores_classification[dataset].ravel()
        else:
            for class_label, label_id in run_params['label_map'].items():
                datasets_tbls[dataset][f'score_{class_label}'] = scores[dataset][:, label_id]
            # data[dataset]['predicted_class'] = scores_classification[dataset]

    # write predictions to the datasets tables
    for dataset in run_params['datasets']:
        # sort in descending order of output
        if not run_params['config']['multi_class']:
            datasets_tbls[dataset].sort_values(by='score', ascending=False, inplace=True)
        datasets_tbls[dataset].to_csv(model_dir / f'ranked_predictions_{dataset}set.csv', index=False)

    if run_params['logger'] is not None:
        run_params['logger'].info(f'[model_{run_params["model_id"]}] Finished training model {run_params["model_id"]}.')


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--rank', type=int, help='Rank index', default=0)
    parser.add_argument('--output_dir', type=str, help='Output directory', default=None)
    parser.add_argument('--config_fp', type=str, help='File path to YAML configuration file.',
                        default='/Users/msaragoc/OneDrive - NASA/Projects/exoplanet_transit_classification/codebase/src/config_predict_model.yaml')
    args = parser.parse_args()

    # load yaml file with run setup
    with(open(args.config_fp, 'r')) as file:
        config = yaml.safe_load(file)

    # set rank of process as function of number of GPUs per node and process id
    config['rank'] = args.rank
    if config['rank'] != 0:
        time.sleep(2)

    # rank should always be less than the number of models to be trained
    if config["rank"] >= len(config['paths']['models_filepaths']):
        print(f'Number of processes requested to evaluate and run inference with models ({config["rank"]}) is higher '
              f'than the number models that was set to be evaluated/used for prediction '
              f'({len(config["paths"]["models_filepaths"])}). Ending process.')
        return

    # set random generator
    config['rng'] = np.random.default_rng(seed=config['rnd_seed'])

    # set paths
    if args.output_dir is not None:
        config['paths']['experiment_dir'] = args.output_dir
    for path_name, path_str in config['paths'].items():
        if path_str is not None and path_name != 'models_filepaths':
            config['paths'][path_name] = Path(path_str)
    for fp_i, fp in enumerate(config['paths']['models_filepaths']):
        config['paths']['models_filepaths'][fp_i] = Path(fp)
    config['paths']['experiment_dir'].mkdir(exist_ok=True)
    # set models directory
    config['models_dir'] = config['paths']['experiment_dir'] / 'models'
    config['models_dir'].mkdir(exist_ok=True)

    # set up logger
    config['logger'] = logging.getLogger(name=f'model_{config["rank"]}')
    logger_handler = logging.FileHandler(filename=config['paths']['experiment_dir'] /
                                                  f'model_{config["rank"]}.log', mode='w')
    logger_formatter = logging.Formatter('%(asctime)s - %(message)s')
    config['logger'].setLevel(logging.INFO)
    logger_handler.setFormatter(logger_formatter)
    config['logger'].addHandler(logger_handler)
    config['logger'].info(f'Starting run {config["paths"]["experiment_dir"].name}...')
    config['logger'].info(f'Rank: {config["rank"]}')

    # setting GPU
    config['logger'].info(f'Number of GPUs selected per node = {config["ngpus_per_node"]}')
    config['gpu_id'] = config["rank"] % config['ngpus_per_node']
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config['gpu_id'])
    config['logger'].info(f'[rank_{config["rank"]}] CUDA DEVICE ORDER: {os.environ["CUDA_DEVICE_ORDER"]}')
    config['logger'].info(f'[rank_{config["rank"]}] CUDA VISIBLE DEVICES: {os.environ["CUDA_VISIBLE_DEVICES"]}')

    # set dictionary of TFRecord file paths for the different data sets
    # TFRECORDS MUST HAVE A PREFIX THAT MATCHES THE NAMING OF THE DATASETS IN CONFIG['DATASETS']
    config['datasets_fps'] = {dataset: [fp for fp in config['paths']['tfrec_dir'].iterdir()
                                        if fp.name.startswith(dataset)]
                              for dataset in config['datasets']}

    # set tensorflow data type for features in the feature set
    config['features_set'] = set_tf_data_type_for_features(config['features_set'])

    # set early stopping callback
    config['callbacks_list'] = {
        'train': [
            callbacks.EarlyStopping(**config['callbacks']['early_stopping']),
        ],
    }

    if config['rank'] == 0:
        # save all run parameters
        np.save(config['paths']['experiment_dir'] / 'run_params.npy', config)
        # save model's architecture and hyperparameters used

        # save the YAML file for this run that are YAML serializable
        json_dict = {key: val for key, val in config.items() if is_yamlble(val)}
        with open(config['paths']['experiment_dir'] / 'run_params.yaml', 'w') as file:
            yaml.dump(json_dict, file, sort_keys=False)

    if config['predict_parallel']:  # run each model in parallel
        config['model_id'] = config['rank']  # rank of process defines model id
        if config['logger'] is not None:
            config['logger'].info(f'Running model {config["model_id"] + 1} '
                                  f'(out of {len(config["paths"]["models_filepaths"])})')
        evaluate_inference_for_trained_model(config)
    else:
        # run each model sequentially
        for model_id in range(len(config['paths']['models_filepaths'])):
            config['model_id'] = model_id
            if config['logger'] is not None:
                config['logger'].info(
                    f'[model_{config["model_id"]}] Running model {config["model_id"] + 1} '
                    f'(out of {len(config["paths"]["models_filepaths"])})')
            evaluate_inference_for_trained_model(config)


if __name__ == '__main__':

    main()
