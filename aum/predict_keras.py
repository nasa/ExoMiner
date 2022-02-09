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
from tensorflow.keras import callbacks
from tensorflow.keras.models import load_model
import argparse
from pathlib import Path
import logging
import yaml

# local
from src.utils_dataio import InputFnv2 as InputFn, get_data_from_tfrecord
from models.models_keras import compile_model, create_ensemble
from src.utils_metrics import get_metrics, get_metrics_multiclass, compute_precision_at_k
from src_hpo import utils_hpo
from src.utils_visualization import plot_class_distribution, plot_precision_at_k
from utils.utils_dataio import is_yamlble
from paths import path_main
from src.utils_predict import create_ranking, save_metrics_to_file


def run_main(config):
    """ Run inference using trained model.

    :param config: dict, configuration parameters for the run
    :param base_model: model
    :param model_id: int, model id
    :return:
    """

    # create directory for the model
    model_dir_sub = config['paths']['experiment_dir'] / 'models' / 'model1'

    if config['training']['filter_data'] is None:
        filter_data = {dataset: None for dataset in config['datasets']}

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
    for dataset in config['datasets']:
        data[dataset]['label'] = np.array(data[dataset]['label'])
        data[dataset]['original_label'] = np.array(data[dataset]['original_label'])

    # load model
    model_list = []
    config['paths']['models_filepaths'] = [model_dir_sub / 'model1.h5']
    for model_i, model_filepath in enumerate(config['paths']['models_filepaths']):
        model = load_model(filepath=model_dir_sub / 'model1.h5', compile=False)
        model._name = f'model{model_i}'
        model_list.append(model)

    if len(model_list) == 1:
        model = model_list[0]
    else:  # create ensemble
        model = create_ensemble(features=config['features_set'],
                                models=model_list,
                                feature_map=config['feature_map'])

    # print model summary
    if config['rank'] is None or config['rank'] == 0:
        model.summary()

    # setup metrics to be monitored
    if not config['config']['multi_class']:
        metrics_list = get_metrics(clf_threshold=config['metrics']['clf_thr'],
                                   num_thresholds=config['metrics']['num_thr'])
    else:
        metrics_list = get_metrics_multiclass(config['label_map'])

    # compile model - set optimizer, loss and metrics
    model = compile_model(model, config, metrics_list)

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
                                multiclass=config['config']['multi_class']
                                )

        callbacks_list = []

        # evaluate model in the given dataset
        res_eval = model.evaluate(x=eval_input_fn(),
                                  y=None,
                                  batch_size=None,
                                  verbose=config['verbose'],
                                  sample_weight=None,
                                  steps=None,
                                  callbacks=callbacks_list if dataset == 'train' else None,
                                  max_queue_size=10,
                                  workers=1,
                                  use_multiprocessing=False
                                  )

        # add evaluated dataset metrics to result dictionary
        for metric_name_i, metric_name in enumerate(model.metrics_names):
            res[f'{dataset}_{metric_name}'] = res_eval[metric_name_i]

    # predict on given datasets - needed for computing the output distribution
    predictions = {dataset: [] for dataset in config['datasets']}
    for dataset in predictions:
        print('Predicting on dataset {}...'.format(dataset))

        predict_input_fn = InputFn(file_paths=str(config['paths']['tfrec_dir']) + '/' + dataset + '*',
                                   batch_size=config['training']['batch_size'],
                                   mode='PREDICT',
                                   label_map=config['label_map'],
                                   filter_data=filter_data[dataset],
                                   features_set=config['features_set'],
                                   multiclass=config['config']['multi_class']
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
    output_cl = {dataset: {} for dataset in config['datasets']}
    for dataset in output_cl:
        for original_label in config['label_map_pred']:
            # get predictions for each original class individually to compute histogram
            if config['config']['multi_class']:
                output_cl[dataset][original_label] = \
                    predictions[dataset][np.where(data[dataset]['original_label'] ==
                                                  original_label)][:, config['label_map_pred'][original_label]]
            else:
                output_cl[dataset][original_label] = \
                    predictions[dataset][np.where(data[dataset]['original_label'] == original_label)]

    # # compute precision at top-k
    # labels_sorted = {}
    # for dataset in config['datasets']:
    #     if dataset == 'predict':
    #         continue
    #     if not config['config']['multi_class']:
    #         sorted_idxs = np.argsort(predictions[dataset], axis=0).squeeze()
    #         labels_sorted[dataset] = labels[dataset][sorted_idxs].squeeze()
    #         prec_at_k = compute_precision_at_k(labels_sorted[dataset], config['metrics']['top_k_arr'][dataset])
    #         res.update({f'{dataset}_precision_at_{k_val}': prec_at_k[f'precision_at_{k_val}']
    #                     for k_val in config['metrics']['top_k_arr'][dataset]})
    #     else:
    #         res.update({f'{dataset}_precision_at_{k_val}': np.nan
    #                     for k_val in config['metrics']['top_k_arr'][dataset]})

    # save results in a numpy file
    res_fp = config['paths']['experiment_dir'] / 'results.npy'
    print(f'Saving metrics to {res_fp}...')
    np.save(res_fp, res)

    # ep_idx = -1

    # plot class distribution
    for dataset in config['datasets']:
        plot_class_distribution(output_cl[dataset],
                                config['paths']['experiment_dir'] /
                                f'model1_class_predoutput_distribution_{dataset}.svg')

    # plot precision, recall, ROC AUC, PR AUC curves
    # # plot pr curve
    # if not config['config']['multi_class']:
    #     plot_pr_curve(res, ep_idx, config['paths']['experiment_dir'] / f'model1_prec_rec.svg')
    #     # plot roc
    #     plot_roc(res, ep_idx, config['paths']['experiment_dir'] / f'model1_roc.svg')
    # plot precision-at-k and misclassfied-at-k examples curves
    # for dataset in config['datasets']:
    #     k_curve_arr = np.linspace(**config['metrics']['top_k_curve'][dataset])
    #     plot_precision_at_k(labels_sorted[dataset], k_curve_arr,
    #                         config['paths']['experiment_dir'] / f'model{model_id}_{dataset}')

    print('Saving metrics to a txt file...')
    save_metrics_to_file(config['paths']['experiment_dir'],
                         res,
                         config['datasets'],
                         model.metrics_names,
                         config['metrics']['top_k_arr'],
                         config['paths']['models_filepaths'],
                         print_res=True
                         )

    # initialize dictionary to save the classification scores for each dataset evaluated
    scores_classification = {dataset: np.zeros(predictions[dataset].shape, dtype='uint8')
                             for dataset in config['datasets']}
    for dataset in config['datasets']:
        # threshold for classification
        if not config['config']['multi_class']:
            scores_classification[dataset][predictions[dataset] >= config['metrics']['clf_thr']] = 1
        else:  # multiclass - get label id of highest scoring class
            # scores_classification[dataset] = np.argmax(scores[dataset], axis=1)
            scores_classification[dataset] = np.repeat([np.array(sorted(config['label_map'].values()))],
                                                       len(predictions[dataset]),
                                                       axis=0)[np.arange(len(predictions[dataset])),
                                                               np.argmax(predictions[dataset], axis=1)]

    for dataset in config['datasets']:
        ranking_tbl = create_ranking(predictions[dataset],
                                     scores_classification[dataset],
                                     config['label_map_pred'],
                                     config['config']['multi_class'],
                                     data_examples=data[dataset])
        ranking_tbl.to_csv(config['paths']['experiment_dir'] / f'ranking_{dataset}.csv', index=False)


if __name__ == '__main__':

    # used in job arrays
    parser = argparse.ArgumentParser()
    parser.add_argument('--job_idx', type=int, help='Job index', default=0)
    parser.add_argument('--config_file', type=str, help='File path to YAML configuration file.', default=None)
    args = parser.parse_args()
    # args.config_file = '/home/msaragoc/Projects/Kepler-TESS_exoplanet/experiments/label_noise_detection_aum/run_02-03-2022_1052/configs/config_aum_run0.yaml'

    if args.config_file is None:  # use default config file in codebase
        path_to_yaml = Path('/home/msaragoc/Projects/Kepler-TESS_exoplanet/codebase/aum/config_train.yaml')
    else:  # use config file given as input
        path_to_yaml = Path(args.config_file)

    with(open(path_to_yaml, 'r')) as file:
        config = yaml.safe_load(file)

    if config['train_parallel']:  # train models in parallel
        rank = MPI.COMM_WORLD.rank
        config['rank'] = config['ngpus_per_node'] * args.job_idx + rank
        config['size'] = MPI.COMM_WORLD.size
        print(f'Rank = {config["rank"]}/{config["size"] - 1}')
        sys.stdout.flush()
        if rank != 0:
            time.sleep(2)
    else:
        config['rank'] = 0

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
    config['paths']['models_dir'] = config['paths']['experiment_dir'] / 'models'

    # set up logger
    logger = logging.getLogger(name='inference_run')
    logger_handler = logging.FileHandler(filename=config['paths']['experiment_dir'] /
                                                  f'inference_run_{config["rank"]}.log',
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

    # choose features set
    for feature_name, feature in config['features_set'].items():
        if feature['dtype'] == 'float':
            config['features_set'][feature_name]['dtype'] = tf.float32
        elif feature['dtype'] == 'int':
            config['features_set'][feature_name]['dtype'] = tf.int64

    logger.info(f'Feature set: {config["features_set"]}')

    config['filepaths'] = {dataset: [str(filepath) for filepath in config['paths']['tfrec_dir'].iterdir()
                                     if 'shard' in filepath.name and filepath.name.split('-')[0] == dataset]
                           for dataset in config['datasets']}
    config['filepaths']['all_datasets'] = [str(filepath) for filepath in config['paths']['tfrec_dir'].iterdir()
                                           if 'shard' in filepath.name]

    config['data_fields'] = {
        'target_id': 'int_scalar',
        'tce_plnt_num': 'int_scalar',
        'label': 'string',  # REQUIRED to run script
        'original_label': 'string',
        'tce_period': 'float_scalar',
        'tce_duration': 'float_scalar',
        'tce_time0bk': 'float_scalar',
        'transit_depth': 'float_scalar'
    }

    config['evaluation'] = {'batch_size': 32}
    config['label_map_pred']['UNK'] = 0

    logger.info(f'Final configuration used: {config}')

    if config['rank'] == 0:
        # save the YAML file with training-evaluation parameters that are YAML serializable
        json_dict = {key: val for key, val in config.items() if is_yamlble(val)}
        with open(config['paths']['experiment_dir'] / 'inference_params.yaml', 'w') as config_file:
            yaml.dump(json_dict, config_file)

    print(f'Running inference...')
    run_main(config=config)
