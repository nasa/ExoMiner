"""
Train models using a given configuration obtained on a hyperparameter optimization study.
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
from tensorflow.keras import callbacks, losses, optimizers
import argparse
from tensorflow.keras.utils import plot_model
from pathlib import Path
import logging
import yaml

# local
from models import baseline_configs
from src.utils_dataio import InputFnv2 as InputFn, get_data_from_tfrecord
from src.utils_dataio import get_data_from_tfrecord
from models.models_keras import CNN1dPlanetFinderv2
from src.utils_metrics import get_metrics, compute_precision_at_k
from src_hpo import utils_hpo
from src.utils_visualization import plot_class_distribution, plot_precision_at_k
from src.utils_train import save_metrics_to_file, print_metrics, plot_loss_metric, plot_roc, plot_pr_curve
from utils.utils_dataio import is_yamlble


def run_main(config, base_model, model_id):
    """ Train and evaluate model on a given configuration. Test set must also contain labels.

    :param config: dict, configuration parameters for the run
    :param base_model: model
    :return:
    """

    # create directory for the model
    model_dir_sub = config['paths']['models_dir'] / f'model{model_id}'
    model_dir_sub.mkdir(exist_ok=True)

    # debug_dir = os.path.join(model_dir_sub, 'debug')
    # os.makedirs(debug_dir, exist_ok=True)
    # tf.debugging.experimental.enable_dump_debug_info(debug_dir,
    #                                                  tensor_debug_mode='FULL_HEALTH',
    #                                                  circular_buffer_size=-1)

    if 'tensorboard' in config['callbacks']:
        config['callbacks']['tensorboard']['obj'].log_dir = model_dir_sub
    callbacks_list = [config['callbacks'][callback_config]['obj'] for callback_config in config['callbacks']]

    if config['training']['filter_data'] is None:
        filter_data = {dataset: None for dataset in config['datasets']}

    # get labels for each dataset
    labels = {dataset: [] for dataset in config['datasets']}
    original_labels = {dataset: [] for dataset in config['datasets']}

    tfrec_files = [file for file in config['paths']['tfrec_dir'].iterdir() if file.name.split('-')[0]
                   in config['datasets']]
    for tfrec_file in tfrec_files:
        # find which dataset the TFRecord is from
        dataset = tfrec_file.name.split('-')[0]

        data_tfrec = get_data_from_tfrecord(tfrec_file,
                                            {'label': 'string', 'original_label': 'string'},
                                            config['label_map'])

        labels[dataset] += data_tfrec['label']
        original_labels[dataset] += data_tfrec['original_label']

    # convert from list to numpy array
    labels = {dataset: np.array(labels[dataset], dtype='uint8') for dataset in config['datasets']}
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
    metrics_list = get_metrics(clf_threshold=config['metrics']['clf_thr'], num_thresholds=config['metrics']['num_thr'])

    if config['config']['optimizer'] == 'Adam':
        model.compile(optimizer=optimizers.Adam(learning_rate=config['config']['lr'],
                                                beta_1=0.9,
                                                beta_2=0.999,
                                                epsilon=1e-8,
                                                amsgrad=False,
                                                name='Adam'),  # optimizer
                      # loss function to minimize
                      loss=losses.BinaryCrossentropy(from_logits=False,
                                                     label_smoothing=0,
                                                     name='binary_crossentropy'),
                      # list of metrics to monitor
                      metrics=metrics_list)

    else:
        model.compile(optimizer=optimizers.SGD(learning_rate=config['config']['lr'],
                                               momentum=config['config']['sgd_momentum'],
                                               nesterov=False,
                                               name='SGD'),  # optimizer
                      # loss function to minimize
                      loss=losses.BinaryCrossentropy(from_logits=False,
                                                     label_smoothing=0,
                                                     name='binary_crossentropy'),
                      # list of metrics to monitor
                      metrics=metrics_list)

    # input function for training, validation and test
    train_input_fn = InputFn(file_pattern=str(config['paths']['tfrec_dir']) + '/train*',
                             batch_size=config['training']['batch_size'],
                             mode='TRAIN',
                             label_map=config['label_map'],
                             data_augmentation=config['training']['data_augmentation'],
                             online_preproc_params=config['training']['online_preprocessing_params'],
                             filter_data=filter_data['train'],
                             features_set=config['features_set'],
                             category_weights=config['training']['category_weights'])
    val_input_fn = InputFn(file_pattern=str(config['paths']['tfrec_dir']) + '/val*',
                           batch_size=config['training']['batch_size'],
                           mode='EVAL',
                           label_map=config['label_map'],
                           filter_data=filter_data['val'],
                           features_set=config['features_set'])
    test_input_fn = InputFn(file_pattern=str(config['paths']['tfrec_dir']) + '/test*',
                            batch_size=config['training']['batch_size'],
                            mode='EVAL',
                            label_map=config['label_map'],
                            filter_data=filter_data['test'],
                            features_set=config['features_set'],
                            )

    # fit the model to the training data
    print('Training model...')
    history = model.fit(x=train_input_fn(),
                        y=None,
                        batch_size=None,
                        epochs=config['training']['n_epochs'],
                        verbose=config['verbose'],
                        callbacks=callbacks_list,
                        validation_split=0.,
                        validation_data=val_input_fn(),
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

    print('Evaluating model on the test set...')

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
        print('Predicting on dataset {}...'.format(dataset))

        predict_input_fn = InputFn(file_pattern=str(config['paths']['tfrec_dir']) + '/' + dataset + '*',
                                   batch_size=config['training']['batch_size'],
                                   mode='PREDICT',
                                   label_map=config['label_map'],
                                   filter_data=filter_data[dataset],
                                   features_set=config['features_set'])

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
    output_cl = {dataset: {} for dataset in config['datasets']}
    for dataset in output_cl:
        for original_label in config['label_map']:
            # get predictions for each original class individually to compute histogram
            output_cl[dataset][original_label] = predictions[dataset][np.where(original_labels[dataset] ==
                                                                               original_label)]

    # compute precision at top-k
    labels_sorted = {}
    for dataset in config['datasets']:
        if dataset == 'predict':
            continue
        sorted_idxs = np.argsort(predictions[dataset], axis=0).squeeze()
        labels_sorted[dataset] = labels[dataset][sorted_idxs].squeeze()
        prec_at_k = compute_precision_at_k(labels_sorted[dataset], config['metrics']['top_k_arr'][dataset])
        res.update({f'{dataset}_precision_at_{k_val}': prec_at_k[f'precision_at_{k_val}']
                    for k_val in config['metrics']['top_k_arr'][dataset]})

    # save results in a numpy file
    res_fp = model_dir_sub / 'results.npy'
    print(f'Saving metrics to {res_fp}...')
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
    plot_loss_metric(res, epochs, ep_idx, config['training']['opt_metric'],
                     config['paths']['experiment_dir'] / f'model{model_id}_plotseval_epochs{epochs[-1]:.0f}.svg')
    # plot class distribution
    for dataset in config['datasets']:
        plot_class_distribution(output_cl[dataset],
                                config['paths'][
                                    'experiment_dir'] / f'model{model_id}_class_predoutput_distribution_{dataset}.svg')
    # plot precision, recall, ROC AUC, PR AUC curves
    # plot_prec_rec_roc_auc_pr_auc(res, epochs, ep_idx,
    # os.path.join(res_dir, 'model{}_prec_rec_auc.svg'.format(model_id)))
    # plot pr curve
    plot_pr_curve(res, ep_idx, config['paths']['experiment_dir'] / f'model{model_id}_prec_rec.svg')
    # plot roc
    plot_roc(res, ep_idx, config['paths']['experiment_dir'] / f'model{model_id}_roc.svg')
    # plot precision-at-k and misclassfied-at-k examples curves
    for dataset in config['datasets']:
        k_curve_arr = np.linspace(**config['metrics']['top_k_curve'][dataset])
        plot_precision_at_k(labels_sorted[dataset], k_curve_arr,
                            config['paths']['experiment_dir'] / f'model{model_id}_{dataset}')

    print('Saving metrics to a txt file...')
    save_metrics_to_file(model_dir_sub, res, config['datasets'], ep_idx, model.metrics_names,
                         config['metrics']['top_k_arr'])

    print_metrics(model_id, res, config['datasets'], ep_idx, model.metrics_names, config['metrics']['top_k_arr'])


if __name__ == '__main__':

    path_to_yaml = Path('/home/msaragoc/Projects/Kepler-TESS_exoplanet/codebase/src/config_train.yaml')
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
    if rank != 0:
        time.sleep(2)

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

    # SCRIPT PARAMETERS #############################################

    # experiment directory
    for path_name, path_str in config['paths'].items():
        config['paths'][path_name] = Path(path_str)
    config['paths']['experiment_dir'].mkdir(exist_ok=True)
    config['paths']['models_dir'] = config['paths']['experiment_dir'] / 'models'
    config['paths']['models_dir'].mkdir(exist_ok=True)

    # set up logger
    logger = logging.getLogger(name='train-eval_run')
    logger_handler = logging.FileHandler(filename=config['paths']['experiment_dir'] / f'train-eval_run_{rank}.log',
                                         mode='w')
    logger_formatter = logging.Formatter('%(asctime)s - %(message)s')
    logger.setLevel(logging.INFO)
    logger_handler.setFormatter(logger_formatter)
    logger.addHandler(logger_handler)
    logger.info(f'Starting run {config["paths"]["experiment_dir"].name}...')

    # TFRecord files directory
    logger.info(f'Using data from {config["paths"]["tfrec_dir"]}')

    # name of the HPO study from which to get a configuration; config needs to be set to None
    # set the configuration from an HPO study
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

        logger.info(f'Using configuration from HPO study {hpo_path.name}')
        logger.info(f'HPO Config {config_id_hpo}: {config["config"]}')

    # base model used - check estimator_util.py to see which models are implemented
    BaseModel = CNN1dPlanetFinderv2  # CNN1dPlanetFinderv2
    # config['config']['parameters'].update(baseline_configs.astronet)

    # choose features set
    for feature_name, feature in config['features_set'].items():
        if feature['dtype'] == 'float32':
            config['features_set'][feature_name]['dtype'] = tf.float32

    logger.info(f'Feature set: {config["features_set"]}')

    # early stopping callback
    config['callbacks']['early_stopping']['obj'] = callbacks.EarlyStopping(**config['callbacks']['early_stopping'])

    # TensorBoard callback
    config['callbacks']['tensorboard']['obj'] = callbacks.TensorBoard(**config['callbacks']['tensorboard'])

    logger.info(f'Final configuration used: {config}')

    if config['rank'] == 0:
        # save the YAML file with training-evaluation parameters that are YAML serializable
        json_dict = {key: val for key, val in config.items() if is_yamlble(val)}
        with open(config['paths']['experiment_dir'] / 'cv_params.yaml', 'w') as cv_run_file:
            yaml.dump(json_dict, cv_run_file)

    # comment for multiprocessing using MPI
    for model_i in range(config["training"]["n_models"]):
        print(
            f'Training model {model_i + 1} out of {config["training"]["n_models"]} on {config["training"]["n_epochs"]} epochs...')
        run_main(config=config,
                 base_model=BaseModel,
                 model_id=model_i + 1,
                 )

    # # uncomment for multiprocessing using MPI
    # if config['rank'] < config['training']['n_models']:
    #     print(f'Training model {config["rank"] + 1} out of {config["training"]["n_models"]} on {config["training"]["n_epochs"}')
    #     sys.stdout.flush()
    #     run_main(config=config,
    #              base_model=BaseModel,
    #              model_id = config['rank'] + 1,
    #              )
