"""
Train models using a given configuration.
"""

# 3rd party
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
from mpi4py import MPI
import time
from tensorflow.keras import callbacks
import argparse
from tensorflow.keras.utils import plot_model
from pathlib import Path
import logging
import yaml
# import importlib

# local
from src.utils_dataio import InputFnv2 as InputFn, get_data_from_tfrecord
from models.models_keras import ExoMiner, compile_model, TransformerExoMiner
from src.utils_metrics import get_metrics, get_metrics_multiclass, compute_precision_at_k
from src_hpo import utils_hpo
from src.utils_visualization import plot_class_distribution, plot_precision_at_k
from src.utils_train import save_metrics_to_file, print_metrics, plot_loss_metric, plot_roc, plot_pr_curve
from utils.utils_dataio import is_yamlble
from paths import path_main


def run_main(config, base_model, model_id, logger=None):
    """ Train and evaluate a single model on a given configuration.

    :param config: dict, configuration parameters for the run
    :param base_model: model
    :param model_id: int, model id
    :param logger: logger
    :return:
    """

    # create directory for the model
    model_dir_sub = config['paths']['models_dir'] / f'model{model_id}'
    model_dir_sub.mkdir(exist_ok=True)

    if 'tensorboard' in config['callbacks']:
        config['callbacks']['tensorboard']['obj'].log_dir = model_dir_sub
    callbacks_list = [config['callbacks'][callback_config]['obj'] for callback_config in config['callbacks']]

    if config['training']['filter_data'] is None:
        filter_data = {dataset: None for dataset in config['datasets']}

    # get labels for each dataset
    labels = {dataset: [] for dataset in config['datasets']}
    if config['label_field_name'] == 'label':
        original_labels = {dataset: [] for dataset in config['datasets']}

    tfrec_files = [file for file in config['paths']['tfrec_dir'].iterdir() if file.name.split('-')[0]
                   in config['datasets']]
    for tfrec_file in tfrec_files:
        # find which dataset the TFRecord is from
        dataset = tfrec_file.name.split('-')[0]

        data_tfrec = get_data_from_tfrecord(tfrec_file,
                                            {config['label_field_name']: 'string', 'original_label': 'string'},
                                            config['label_map'])

        labels[dataset] += data_tfrec[config['label_field_name']]
        if config['label_field_name'] == 'label':
            original_labels[dataset] += data_tfrec['original_label']

    # convert from list to numpy array
    labels = {dataset: np.array(labels[dataset], dtype='uint8') for dataset in config['datasets']}
    if config['label_field_name'] == 'label':
        original_labels = {dataset: np.array(original_labels[dataset]) for dataset in config['datasets']}

    # instantiate Keras model
    model = base_model(config, config['features_set']).kerasModel

    # save model, features and config used for training this model
    if model_id == 1:
        # save plot of model
        plot_model(model,
                   to_file=config['paths']['experiment_dir'] / 'model.png',
                   show_shapes=True,
                   show_layer_names=True,
                   rankdir='TB',
                   expand_nested=False,
                   dpi=48)

    # print model summary
    if config['rank'] is None or config['rank'] == 0:
        model.summary()

    # setup metrics to be monitored
    if not config['config']['multi_class']:
        metrics_list = get_metrics(clf_threshold=config['metrics']['clf_thr'],
                                   num_thresholds=config['metrics']['num_thr'])
    else:
        metrics_list = get_metrics_multiclass(label_map=config['label_map'])

    # compile model - set optimizer, loss and metrics
    model = compile_model(model, config, metrics_list)

    # input function for training, validation and test
    train_input_fn = InputFn(
        file_paths=str(config['paths']['tfrec_dir']) + '/train*',
        batch_size=config['training']['batch_size'],
        mode='TRAIN',
        label_map=config['label_map'],
        data_augmentation=config['training']['data_augmentation'],
        online_preproc_params=config['training']['online_preprocessing_params'],
        filter_data=filter_data['train'],
        features_set=config['features_set'],
        category_weights=config['training']['category_weights'],
        multiclass=config['config']['multi_class'],
        use_transformer=config['config']['use_transformer'],
        feature_map=config['feature_map'],
        shuffle_buffer_size=config['training']['shuffle_buffer_size'],
        label_field_name=config['label_field_name'],
    )
    if 'val' in config['datasets']:
        val_input_fn = InputFn(
            file_paths=str(config['paths']['tfrec_dir']) + '/val*',
            batch_size=config['training']['batch_size'],
            mode='EVAL',
            label_map=config['label_map'],
            filter_data=filter_data['val'],
            features_set=config['features_set'],
            multiclass=config['config']['multi_class'],
            use_transformer=config['config']['use_transformer'],
            feature_map=config['feature_map'],
            label_field_name=config['label_field_name'],
        )
    else:
        val_input_fn = None
    if 'test' in config['datasets']:
        test_input_fn = InputFn(
            file_paths=str(config['paths']['tfrec_dir']) + '/test*',
            batch_size=config['training']['batch_size'],
            mode='EVAL',
            label_map=config['label_map'],
            filter_data=filter_data['test'],
            features_set=config['features_set'],
            multiclass=config['config']['multi_class'],
            use_transformer=config['config']['use_transformer'],
            feature_map=config['feature_map'],
            label_field_name=config['label_field_name']
        )
    else:
        test_input_fn = None

    # fit the model to the training data
    if logger is None:
        print('Training model...')
    else:
        logger.info('Training model...')
    history = model.fit(x=train_input_fn(),
                        y=None,
                        batch_size=None,
                        epochs=config['training']['n_epochs'],
                        verbose=config['verbose'],
                        callbacks=callbacks_list,
                        validation_split=0.,
                        validation_data=val_input_fn() if val_input_fn is not None else None,
                        shuffle=True,  # does the input function shuffle for every epoch?
                        class_weight=None,
                        sample_weight=None,
                        initial_epoch=0,
                        steps_per_epoch=None,
                        validation_steps=None,
                        max_queue_size=10,  # does not matter when using input function with tf.data API
                        workers=1,  # same
                        use_multiprocessing=False  # same
                        )

    # save model
    model.save(model_dir_sub / f'model{model_id}.h5')

    res = history.history

    if 'test' in config['datasets']:
        if logger is None:
            print('Evaluating model on the test set...')
        else:
            logger.info('Evaluating model on the test set...')

        res_eval = model.evaluate(x=test_input_fn(),
                                  y=None,
                                  batch_size=None,
                                  verbose=config['verbose'],
                                  sample_weight=None,
                                  steps=None,
                                  callbacks=None,
                                  max_queue_size=10,
                                  workers=1,
                                  use_multiprocessing=False)

        # add test set metrics to result
        for metric_name_i, metric_name in enumerate(model.metrics_names):
            res['test_{}'.format(metric_name)] = res_eval[metric_name_i]

    # predict on given datasets - needed for computing the output distribution
    predictions = {dataset: [] for dataset in config['datasets']}
    for dataset in predictions:
        if logger is None:
            print('Predicting on dataset {}...'.format(dataset))
        else:
            logger.info('Predicting on dataset {}...'.format(dataset))

        predict_input_fn = InputFn(
            file_paths=str(config['paths']['tfrec_dir']) + '/' + dataset + '*',
            batch_size=config['training']['batch_size'],
            mode='PREDICT',
            label_map=config['label_map'],
            filter_data=filter_data[dataset],
            features_set=config['features_set'],
            multiclass=config['config']['multi_class'],
            use_transformer=config['config']['use_transformer'],
            feature_map=config['feature_map'],
            label_field_name=config['label_field_name'],
        )

        predictions[dataset] = model.predict(predict_input_fn(),
                                             batch_size=None,
                                             verbose=config['verbose'],
                                             steps=None,
                                             callbacks=None,
                                             max_queue_size=10,
                                             workers=1,
                                             use_multiprocessing=False,
                                             )

    # sort predictions per class based on ground truth labels
    if config['label_field_name'] == 'label':
        output_cl = {dataset: {} for dataset in config['datasets']}
        for dataset in output_cl:
            for original_label in config['label_map']:
                # get predictions for each original class individually to compute histogram
                if config['config']['multi_class']:
                    output_cl[dataset][original_label] = \
                        predictions[dataset][np.where(original_labels ==
                                                      original_label)][:, config['label_map'][original_label]]
                else:
                    output_cl[dataset][original_label] = \
                        predictions[dataset][np.where(original_labels[dataset] == original_label)]

    # compute precision at top-k
    labels_sorted = {}
    for dataset in config['datasets']:
        if dataset == 'predict':
            continue
        if not config['config']['multi_class']:
            sorted_idxs = np.argsort(predictions[dataset], axis=0).squeeze()
            labels_sorted[dataset] = labels[dataset][sorted_idxs].squeeze()
            prec_at_k = compute_precision_at_k(labels_sorted[dataset], config['metrics']['top_k_arr'][dataset])
            res.update({f'{dataset}_precision_at_{k_val}': prec_at_k[f'precision_at_{k_val}']
                        for k_val in config['metrics']['top_k_arr'][dataset]})
        else:
            res.update({f'{dataset}_precision_at_{k_val}': np.nan
                        for k_val in config['metrics']['top_k_arr'][dataset]})

    # save results in a numpy file
    res_fp = model_dir_sub / 'results.npy'
    if logger is None:
        print(f'Saving metrics to {res_fp}...')
    else:
        logger.info(f'Saving metrics to {res_fp}...')
    np.save(res_fp, res)

    print('Plotting evaluation results...')
    epochs = np.arange(1, len(res['loss']) + 1)
    # choose epoch associated with the best value for the metric
    if 'early_stopping' in config['callbacks']:
        if config['callbacks']['early_stopping']['mode'] == 'min':
            ep_idx = np.argmin(res[config['callbacks']['early_stopping']['monitor']])
        else:
            ep_idx = np.argmax(res[config['callbacks']['early_stopping']['monitor']])
    else:
        ep_idx = -1

    # plot evaluation loss and metric curves
    plot_loss_metric(res,
                     epochs,
                     ep_idx,
                     model_dir_sub / f'model{model_id}_plotseval_epochs{epochs[-1]:.0f}.svg',
                     config['training']['opt_metric'])

    # plot class distribution
    for dataset in config['datasets']:
        plot_class_distribution(output_cl[dataset], model_dir_sub /
                                f'model{model_id}_class_predoutput_distribution_{dataset}.svg')

    # plot precision, recall, ROC AUC, PR AUC curves
    if not config['config']['multi_class']:
        # plot pr curve
        plot_pr_curve(res, ep_idx, model_dir_sub / f'model{model_id}_prec_rec.svg')
        # plot roc
        plot_roc(res, ep_idx, config['datasets'], model_dir_sub / f'model{model_id}_roc.svg')
        # plot precision-at-k and misclassfied-at-k examples curves
        for dataset in config['datasets']:
            if dataset != 'predict':
                k_curve_arr = np.linspace(**config['metrics']['top_k_curve'][dataset])
                plot_precision_at_k(labels_sorted[dataset], k_curve_arr, model_dir_sub / f'model{model_id}_{dataset}')

    if logger is None:
        print('Saving metrics to a txt file...')
    else:
        logger.info('Saving metrics to a txt file...')
    save_metrics_to_file(model_dir_sub, res, config['datasets'], ep_idx, model.metrics_names,
                         config['metrics']['top_k_arr'])

    print_metrics(model_id, res, config['datasets'], ep_idx, model.metrics_names, config['metrics']['top_k_arr'])


if __name__ == '__main__':

    # used in job arrays
    parser = argparse.ArgumentParser()
    parser.add_argument('--job_idx', type=int, help='Job index', default=0)
    parser.add_argument('--config_file', type=str, help='File path to YAML configuration file.', default=None)
    args = parser.parse_args()

    if args.config_file is None:  # use default config file in codebase
        path_to_yaml = Path('/Users/msaragoc/OneDrive - NASA/Projects/exoplanet_transit_classification/codebase/src/config_train.yaml')
    else:  # use config file given as input
        path_to_yaml = Path(args.config_file)

    # load YAML config file
    with(open(path_to_yaml, 'r')) as file:
        config = yaml.safe_load(file)

    # experiment directory
    for path_name, path_str in config['paths'].items():
        config['paths'][path_name] = Path(path_str)
    config['paths']['experiment_dir'].mkdir(exist_ok=True)

    if config['rank'] in [0, None]:  # save initial configuration
        with open(config['paths']['experiment_dir'] / 'train_params_init.yaml', 'w') as config_file:
            yaml.dump(config, config_file, sort_keys=False)

    # if config['train_parallel']:  # train models in parallel
    rank = MPI.COMM_WORLD.rank
    config['rank'] = config['ngpus_per_node'] * args.job_idx + rank
    config['size'] = MPI.COMM_WORLD.size
    print(f'Rank = {config["rank"]}/{config["size"] - 1}')
    sys.stdout.flush()
    if rank != 0:
        time.sleep(2)
    # else:
    #     config['rank'] = 0

    # # get list of physical GPU devices available to TF in this process
    # physical_devices = tf.config.list_physical_devices('GPU')
    # print(f'List of physical GPU devices available: {physical_devices}')
    #
    # # select GPU to be used
    # gpu_id = rank % ngpus_per_node
    # tf.config.set_visible_devices(physical_devices[gpu_id], 'GPU')
    # # tf.config.set_visible_devices(physical_devices[0], 'GPU')
    #
    # # get list of logical GPU devices available to TF in this process
    # logical_devices = tf.config.list_logical_devices('GPU')
    # print(f'List of logical GPU devices available: {logical_devices}')

    # select GPU to run the training on
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

    # SCRIPT PARAMETERS #############################################

    # experiment directory
    for path_name, path_str in config['paths'].items():
        config['paths'][path_name] = Path(path_str)
    config['paths']['experiment_dir'].mkdir(exist_ok=True)
    config['paths']['models_dir'] = config['paths']['experiment_dir'] / 'models'
    config['paths']['models_dir'].mkdir(exist_ok=True)

    # set up logger
    logger = logging.getLogger(name='train-eval_run')
    logger_handler = logging.FileHandler(filename=config['paths']['experiment_dir'] /
                                                  f'train-eval_run_{config["rank"]}.log',
                                         mode='w')
    logger_formatter = logging.Formatter('%(asctime)s - %(message)s')
    logger.setLevel(logging.INFO)
    logger_handler.setFormatter(logger_formatter)
    logger.addHandler(logger_handler)
    logger.info(f'Starting run {config["paths"]["experiment_dir"].name}...')

    # TFRecord files directory
    logger.info(f'Using data from {config["paths"]["tfrec_dir"]}')

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
        # for older HPO runs when kernel size was not optimized separately for local and global branches
        if 'kernel_size_glob' not in config_hpo_chosen:
            config_hpo_chosen['kernel_size_glob'] = config_hpo_chosen['kernel_size']
            config_hpo_chosen['kernel_size_loc'] = config_hpo_chosen['kernel_size']
        config['config'].update(config_hpo_chosen)

        # select a specific config based on its ID
        # example - check config.json
        # config = id2config[(0, 0, 0)]['config']

        logger.info(f'Using configuration from HPO study {hpo_path.name}')
        logger.info(f'HPO Config {config_id_hpo}: {config["config"]}')

    # base model used - check estimator_util.py to see which models are implemented
    BaseModel = TransformerExoMiner  # ExoMiner
    # config['config']['parameters'].update(baseline_configs.astronet)

    # choose features set
    for feature_name, feature in config['features_set'].items():
        if feature['dtype'] == 'float':
            config['features_set'][feature_name]['dtype'] = tf.float32
        elif feature['dtype'] == 'int':
            config['features_set'][feature_name]['dtype'] = tf.int64

    logger.info(f'Feature set: {config["features_set"]}')

    # early stopping callback
    if 'early_stopping' in config['callbacks']:
        config['callbacks']['early_stopping']['obj'] = callbacks.EarlyStopping(**config['callbacks']['early_stopping'])

    # TensorBoard callback
    if 'tensorboard' in config['callbacks']:
        config['callbacks']['tensorboard']['obj'] = callbacks.TensorBoard(**config['callbacks']['tensorboard'])

    logger.info(f'Final configuration used: {config}')

    if config['rank'] == 0:
        # save the YAML file with training-evaluation parameters that are YAML serializable
        json_dict = {key: val for key, val in config.items() if is_yamlble(val)}
        with open(config['paths']['experiment_dir'] / 'train_params.yaml', 'w') as cv_run_file:
            yaml.dump(json_dict, cv_run_file)

    if config['train_parallel']:  # train models in parallel
        if config['rank'] < config['training']['n_models']:
            logger.info(f'Training model {config["rank"] + 1} out of {config["training"]["n_models"]} on '
                        f'{config["training"]["n_epochs"]}')
            sys.stdout.flush()
            run_main(config=config,
                     base_model=BaseModel,
                     model_id=config['rank'] + 1,
                     )
    else:  # train models sequentially
        for model_i in range(config["training"]["n_models"]):
            logger.info(f'Training model {model_i + 1} out of {config["training"]["n_models"]} on '
                        f'{config["training"]["n_epochs"]} epochs...')
            run_main(config=config,
                     base_model=BaseModel,
                     model_id=model_i + 1,
                     )
