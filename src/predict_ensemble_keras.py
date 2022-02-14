"""
Perform inference using an ensemble of Tensorflow Keras models.
"""

# 3rd party
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import load_model
import pandas as pd
import logging
from pathlib import Path
from tensorflow.keras import callbacks
import yaml
import argparse

# local
from src.utils_dataio import get_data_from_tfrecord
from src.utils_dataio import InputFnv2 as InputFn
from models.models_keras import create_ensemble, compile_model
from src_hpo import utils_hpo
from src.utils_metrics import get_metrics, get_metrics_multiclass, compute_precision_at_k
from src.utils_visualization import plot_class_distribution, plot_precision_at_k
from src.utils_predict import save_metrics_to_file, plot_prcurve_roc
from utils.utils_dataio import is_yamlble
from paths import path_main


def run_main(config):
    """ Evaluate model on a given configuration in the specified datasets and also predict on them.

    :param config: configuration object from the Config class
    :return:
    """

    if 'label' not in config['data_fields']:
        config['data_fields']['label'] = 'string'

    # instantiate variable to get data from the TFRecords
    data = {dataset: {field: [] for field in config['data_fields']} for dataset in config['datasets']}

    tfrec_files = [file for file in config['paths']['tfrec_dir'].iterdir()
                   if file.name.split('-')[0] in config['datasets']]
    for tfrec_file in tfrec_files:

        # get dataset of the TFRecord
        dataset = tfrec_file.name.split('-')[0]

        data_aux = get_data_from_tfrecord(tfrec_file, config['data_fields'], config['label_map'])

        for field in data_aux:
            data[dataset][field].extend(data_aux[field])

    # convert from list to numpy array
    # TODO: should make this a numpy array from the beginning
    for dataset in config['datasets']:
        data[dataset]['label'] = np.array(data[dataset]['label'])
        data[dataset]['original_label'] = np.array(data[dataset]['original_label'])

    # create ensemble
    model_list = []
    for model_i, model_filepath in enumerate(config['paths']['models_filepaths']):
        model = load_model(filepath=model_filepath, compile=False)
        model._name = f'model{model_i}'

        model_list.append(model)

    if len(model_list) == 1:
        ensemble_model = model_list[0]
    else:
        ensemble_model = create_ensemble(features=config['features_set'],
                                         models=model_list,
                                         feature_map=config['feature_map'])

    ensemble_model.summary()

    # save model
    ensemble_model.save(config['paths']['experiment_dir'] / 'ensemble_model.h5')
    # plot ensemble model and save the figure
    keras.utils.plot_model(ensemble_model,
                           to_file=config['paths']['experiment_dir'] / 'ensemble.png',
                           show_shapes=False,
                           show_layer_names=True,
                           rankdir='TB',
                           expand_nested=False,
                           dpi=96)

    # set up metrics to be monitored
    if not config['config']['multi_class']:
        metrics_list = get_metrics(clf_threshold=config['metrics']['clf_thr'],
                                   num_thresholds=config['metrics']['num_thr'])
    else:  # metrics for multiclass setting
        metrics_list = get_metrics_multiclass()

    # compile model - set optimizer, loss and metrics
    ensemble_model = compile_model(ensemble_model, config, metrics_list)

    # initialize results dictionary for the evaluated datasets
    res = {}
    for dataset in config['datasets']:

        if dataset == 'predict':
            continue

        print(f'Evaluating on dataset {dataset}')

        # input function for evaluating on each dataset
        eval_input_fn = InputFn(file_paths=str(config['paths']['tfrec_dir']) + '/{}*'.format(dataset),
                                batch_size=config['evaluation']['batch_size'],
                                mode='EVAL',
                                label_map=config['label_map'],
                                features_set=config['features_set'],
                                data_augmentation=False,
                                online_preproc_params=None,
                                filter_data=None,
                                )

        callbacks_list = []
        # for callback_name in callbacks_dict:
        #     if 'layer' in callback_name:
        #         callbacks_dict[callback_name].input_fn = eval_input_fn()

        # evaluate model in the given dataset
        res_eval = ensemble_model.evaluate(x=eval_input_fn(),
                                           y=None,
                                           batch_size=None,
                                           verbose=config['verbose'],
                                           sample_weight=None,
                                           steps=None,
                                           callbacks=callbacks_list if dataset == 'train' else None,
                                           max_queue_size=10,
                                           workers=1,
                                           use_multiprocessing=False)

        # add evaluated dataset metrics to result dictionary
        for metric_name_i, metric_name in enumerate(ensemble_model.metrics_names):
            res[f'{dataset}_{metric_name}'] = res_eval[metric_name_i]

    # predict on given datasets - needed for computing the output distribution and produce a ranking
    scores = {dataset: [] for dataset in config['datasets']}
    for dataset in scores:
        print(f'Predicting on dataset {dataset}...')

        predict_input_fn = InputFn(file_paths=str(config['paths']['tfrec_dir']) + '/' + dataset + '*',
                                   batch_size=config['evaluation']['batch_size'],
                                   mode='PREDICT',
                                   label_map=config['label_map'],
                                   features_set=config['features_set'],
                                   multiclass=config['config']['multi_class'])

        scores[dataset] = ensemble_model.predict(predict_input_fn(),
                                                 batch_size=None,
                                                 verbose=config['verbose'],
                                                 steps=None,
                                                 callbacks=None,
                                                 max_queue_size=10,
                                                 workers=1,
                                                 use_multiprocessing=False)

    # initialize dictionary to save the classification scores for each dataset evaluated
    scores_classification = {dataset: np.zeros(scores[dataset].shape, dtype='uint8') for dataset in config['datasets']}
    for dataset in config['datasets']:
        # threshold for classification
        if not config['config']['multi_class']:
            scores_classification[dataset][scores[dataset] >= config['metrics']['clf_thr']] = 1
        else:  # multiclass - get label id of highest scoring class
            # scores_classification[dataset] = np.argmax(scores[dataset], axis=1)
            scores_classification[dataset] = np.repeat([np.unique(sorted(config['label_map'].values()))],
                                                       len(scores[dataset]),
                                                       axis=0)[np.arange(len(scores[dataset])),
                                                               np.argmax(scores[dataset], axis=1)]

    # sort predictions per class based on ground truth labels
    output_cl = {dataset: {} for dataset in config['datasets']}
    for dataset in output_cl:
        if dataset != 'predict':
            for original_label in config['label_map']:
                # get predictions for each original class individually to compute histogram
                if config['config']['multi_class']:
                    output_cl[dataset][original_label] = \
                        scores[dataset][np.where(data[dataset]['original_label'] ==
                                                 original_label)][:, config['label_map'][original_label]]
                else:
                    output_cl[dataset][original_label] = \
                        scores[dataset][np.where(data[dataset]['original_label'] == original_label)]
        else:
            output_cl[dataset]['NA'] = scores[dataset]

    # compute precision at top-k
    labels_sorted = {}
    for dataset in config['datasets']:
        if dataset == 'predict':
            continue
        if not config['config']['multi_class']:
            sorted_idxs = np.argsort(scores[dataset], axis=0).squeeze()
            labels_sorted[dataset] = data[dataset]['label'][sorted_idxs].squeeze()
            prec_at_k = compute_precision_at_k(labels_sorted[dataset], config['metrics']['top_k_arr'][dataset])
            res.update({f'{dataset}_precision_at_{k_val}': prec_at_k[f'precision_at_{k_val}']
                        for k_val in config['metrics']['top_k_arr'][dataset]})
        else:
            res.update({f'{dataset}_precision_at_{k_val}': np.nan
                        for k_val in config['metrics']['top_k_arr'][dataset]})

    # save evaluation metrics in a numpy file
    print('Saving metrics to a numpy file...')
    np.save(config['paths']['experiment_dir'] / 'results_ensemble.npy', res)

    print('Plotting evaluation results...')
    # draw evaluation plots
    for dataset in config['datasets']:
        plot_class_distribution(output_cl[dataset],
                                config['paths']['experiment_dir'] / f'ensemble_class_scoredistribution_{dataset}.png')

        if not config['config']['multi_class'] and dataset != 'predict':
            plot_prcurve_roc(res, config['paths']['experiment_dir'], dataset)
            k_curve_arr = np.linspace(**config['metrics']['top_k_curve'][dataset])
            plot_precision_at_k(labels_sorted[dataset],
                                k_curve_arr,
                                config['paths']['experiment_dir'] / f'{dataset}')

    print('Saving metrics to a txt file...')
    save_metrics_to_file(config['paths']['experiment_dir'], res, config['datasets'], ensemble_model.metrics_names,
                         config['metrics']['top_k_arr'], config['paths']['models_filepaths'],
                         print_res=True)

    # generate rankings for each evaluated dataset
    if config['generate_csv_pred']:

        print('Generating csv file(s) with ranking(s)...')

        # add predictions to the data dict
        for dataset in config['datasets']:
            if not config['config']['multi_class']:
                data[dataset]['score'] = scores[dataset].ravel()
                data[dataset]['predicted class'] = scores_classification[dataset].ravel()
            else:
                # for i in np.array(list(config['multiclass_class_labels'].keys()), dtype=np.int32):
                #     data[dataset][f'score_{config["multiclass_class_labels"][i]}'] = scores[dataset][:, i]
                for class_label, label_id in config['label_map'].items():
                    data[dataset][f'score_{class_label}'] = scores[dataset][:, label_id]
                data[dataset]['predicted class'] = scores_classification[dataset]

        # write results to a txt file
        for dataset in config['datasets']:
            print(f'Saving ranked predictions in dataset {dataset} to '
                  f'{config["paths"]["experiment_dir"] / f"ranked_predictions_{dataset}"}...')

            data_df = pd.DataFrame(data[dataset])

            # sort in descending order of output
            if not config['config']['multi_class']:
                data_df.sort_values(by='score', ascending=False, inplace=True)
            # else:
            # data_df.sort_values(by=f'score_{config["multiclass_class_labels"][np.max(np.array(list(config["multiclass_class_labels"].keys())))]}', ascending=False, inplace=True)
            data_df.to_csv(config['paths']['experiment_dir'] / f'ensemble_ranked_predictions_{dataset}set.csv',
                           index=False)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, help='File path to YAML configuration file.', default=None)
    args = parser.parse_args()

    if args.config_file is None:  # use default config file in codebase
        path_to_yaml = Path(path_main + 'src/config_predict.yaml')
    else:  # use config file given as input
        path_to_yaml = Path(args.config_file)

    with(open(path_to_yaml, 'r')) as file:  # read YAML configuration file
        config = yaml.safe_load(file)

    # experiment directory
    for path_name, path_str in config['paths'].items():
        config['paths'][path_name] = Path(path_str)
    config['paths']['experiment_dir'].mkdir(exist_ok=True)

    # if config file is from the training experiment
    config['paths']['experiment_dir'].mkdir(exist_ok=True)

    # set up logger
    logger = logging.getLogger(name='predict_run')
    logger_handler = logging.FileHandler(filename=config['paths']['experiment_dir'] / f'predict_run.log', mode='w')
    logger_formatter = logging.Formatter('%(asctime)s - %(message)s')
    logger.setLevel(logging.INFO)
    logger_handler.setFormatter(logger_formatter)
    logger.addHandler(logger_handler)
    logger.info(f'Starting run {config["paths"]["experiment_dir"].name}...')

    logger.info(f'Using data from {config["paths"]["tfrec_dir"]}')

    logger.info(f'Datasets to be evaluated/tested: {config["datasets"]}')

    models_dir = config['paths']['models_dir'] / 'models'
    config['paths']['models_filepaths'] = [model_dir / f'{model_dir.stem}.h5' for model_dir in models_dir.iterdir()
                                           if 'model' in model_dir.stem]
    logger.info(f'Models\' file paths: {config["paths"]["models_filepaths"]}')

    # set the configuration from a HPO study to use extra architecture such as batch size
    if config['paths']['hpo_dir'] is not None:
        hpo_study_fp = Path(config['paths']['hpo_dir'])
        res = utils_hpo.logged_results_to_HBS_result(hpo_study_fp, f'_{hpo_study_fp.name}')

        # get ID to config mapping
        id2config = res.get_id2config_mapping()
        # best config - incumbent
        incumbent = res.get_incumbent_id()
        config_id_hpo = incumbent
        if 'config' in config:
            config['config'].update(id2config[config_id_hpo]['config'])
        else:
            config['config'] = id2config[config_id_hpo]['config']

        # select a specific config based on its ID
        # example - check config.json
        # config = id2config[(8, 0, 3)]['config']

        logger.info(f'Using configuration from HPO study {hpo_study_fp.name}')
        logger.info(f'HPO Config {config_id_hpo}: {config}')

    logger.info(f'Final configuration used: {config}')

    # choose features set
    for feature_name, feature in config['features_set'].items():
        if feature['dtype'] == 'float32':
            config['features_set'][feature_name]['dtype'] = tf.float32
        elif feature['dtype'] == 'int':
            config['features_set'][feature_name]['dtype'] = tf.int64

    logger.info(f'Feature set: {config["features_set"]}')

    # TensorBoard callback
    config['callbacks']['tensorboard']['obj'] = callbacks.TensorBoard(**config['callbacks']['tensorboard'])

    # # Custom callbacks
    # file_writer = tf.summary.create_file_writer(logdir=os.path.join(save_path, 'logs'),
    #                                             filename_suffix='input_to_fcblock')
    # callbacks_dict['layer_conbranch_concat'] = LayerOutputCallback(
    #     input_fn=None,
    #     batch_size=config['batch_size'],
    #     layer_name='convbranch_concat',
    #     summary_writer=file_writer,
    #     buckets=30,
    #     description='Output convolutional branches',
    #     ensemble=True,
    #     log_dir=os.path.join(save_path, 'logs'),
    #     num_batches=None
    # )
    # callbacks_dict['layer_stellar_dv_scalars'] = LayerOutputCallback(
    #     input_fn=None,
    #     batch_size=config['batch_size'],
    #     layer_name='stellar_dv_scalar_input',
    #     summary_writer=file_writer,
    #     buckets=30,
    #     description='Input scalars',
    #     ensemble=True,
    #     log_dir=os.path.join(save_path, 'logs'),
    #     num_batches=None
    # )

    # save the YAML file with training-evaluation parameters that are YAML serializable
    json_dict = {key: val for key, val in config.items() if is_yamlble(val)}
    with open(config['paths']['experiment_dir'] / 'predict_params.yaml', 'w') as cv_run_file:
        yaml.dump(json_dict, cv_run_file)

    run_main(config=config)

    logger.info(f'Finished evaluation and prediction of the ensemble.')
