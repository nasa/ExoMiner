"""
Train models using a given configuration obtained on a hyperparameter optimization study.

TODO: allocate several models to the same GPU

"""

import os
# 3rd party
import sys

from models import baseline_configs

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

# local
import paths
from src.utils_dataio import InputFnv2 as InputFn
from src.utils_dataio import get_data_from_tfrecord
from models.models_keras import CNN1dPlanetFinderv2
import models.config_keras
from src.utils_metrics import get_metrics, compute_precision_at_k
from src_hpo import utils_hpo
from src.utils_visualization import plot_class_distribution, plot_precision_at_k
from src.utils_train import save_metrics_to_file, print_metrics, plot_loss_metric, plot_roc, plot_pr_curve


def run_main(config, n_epochs, data_dir, base_model, res_dir, model_id, opt_metric, min_optmetric,
             callbacks_dict, features_set, data_augmentation=False, online_preproc_params=None,
             filter_data=None, mpi_rank=None):
    """ Train and evaluate model on a given configuration. Test set must also contain labels.

    :param config: configuration object from the Config class
    :param n_epochs: int, number of epochs to train the models
    :param data_dir: str, path to directory with the tfrecord files
    :param res_dir: str, path to directory to save results
    :param opt_metric: str, optimization metric to be plotted alongside the model's loss
    :param min_optmetric: bool, if set to True, gets minimum value of the optimization metric and the respective epoch
    If False, gets the maximum value
    :param callbacks_dict: dict, callbacks
    :param features_set: dict, each key-value pair is feature_name: {'dim': feature_dim, 'dtype': feature_dtype}
    :param data_augmentation: bool, whether to use or not data augmentation
    :param online_preproc_params: dict, contains data used for preprocessing examples online for data augmentation
    :param mpi_rank: int, rank of the mpi process used to distribute the models for the GPUs; set to None when not
    training multiple models in multiple GPUs in parallel
    :return:
    """

    # if mpi_rank is None or mpi_rank == 0:
    #     print('Configuration used: ', config)

    verbose = False if 'home6' in paths.path_hpoconfigs else True

    # create directory for the model
    models_dir = res_dir / 'models'
    models_dir.mkdir(exist_ok=True)
    model_dir_sub = models_dir / f'model{model_id}'
    model_dir_sub.mkdir(exist_ok=True)

    # debug_dir = os.path.join(model_dir_sub, 'debug')
    # os.makedirs(debug_dir, exist_ok=True)
    # tf.debugging.experimental.enable_dump_debug_info(debug_dir,
    #                                                  tensor_debug_mode='FULL_HEALTH',
    #                                                  circular_buffer_size=-1)

    # datasets - same name convention as used for the TFRecords
    datasets = ['train', 'val', 'test']

    if 'tensorboard' in callbacks_dict:
        callbacks_dict['tensorboard'].log_dir = model_dir_sub

    if filter_data is None:
        filter_data = {dataset: None for dataset in datasets}

    # get labels for each dataset
    labels = {dataset: [] for dataset in datasets}
    original_labels = {dataset: [] for dataset in datasets}

    tfrec_files = [file for file in os.listdir(data_dir) if file.split('-')[0] in datasets]
    for tfrec_file in tfrec_files:

        # find which dataset the TFRecord is from
        dataset = tfrec_file.split('-')[0]

        # data = get_data_from_tfrecord(os.path.join(data_dir, tfrec_file), ['label', 'original_label'],
        #                               config['label_map'], filt=filter_data[dataset])
        data_tfrec = get_data_from_tfrecord(os.path.join(data_dir, tfrec_file),
                                      {'label': 'string', 'original_label': 'string'},
                                      config['label_map'])

        labels[dataset] += data_tfrec['label']
        original_labels[dataset] += data_tfrec['original_label']

    # convert from list to numpy array
    labels = {dataset: np.array(labels[dataset], dtype='uint8') for dataset in datasets}
    original_labels = {dataset: np.array(original_labels[dataset]) for dataset in datasets}

    # instantiate Keras model
    model = base_model(config, features_set).kerasModel

    # save model, features and config used for training this model
    if model_id == 1:
        # save plot of model
        plot_model(model,
                   to_file=res_dir / 'model.png',
                   show_shapes=True,
                   show_layer_names=True,
                   rankdir='TB',
                   expand_nested=False,
                   dpi=48)

    # print model summary
    if mpi_rank is None or mpi_rank == 0:
        model.summary()

    # setup metrics to be monitored
    metrics_list = get_metrics(clf_threshold=config['clf_thr'], num_thresholds=config['num_thr'])

    if config['optimizer'] == 'Adam':
        model.compile(optimizer=optimizers.Adam(learning_rate=config['lr'],
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
        model.compile(optimizer=optimizers.SGD(learning_rate=config['lr'],
                                               momentum=config['sgd_momentum'],
                                               nesterov=False,
                                               name='SGD'),  # optimizer
                      # loss function to minimize
                      loss=losses.BinaryCrossentropy(from_logits=False,
                                                     label_smoothing=0,
                                                     name='binary_crossentropy'),
                      # list of metrics to monitor
                      metrics=metrics_list)

    # input function for training, validation and test
    train_input_fn = InputFn(file_pattern=data_dir + '/train*',
                             batch_size=config['batch_size'],
                             mode=tf.estimator.ModeKeys.TRAIN,
                             label_map=config['label_map'],
                             data_augmentation=data_augmentation,
                             online_preproc_params=online_preproc_params,
                             filter_data=filter_data['train'],
                             features_set=features_set)
    val_input_fn = InputFn(file_pattern=data_dir + '/val*',
                           batch_size=config['batch_size'],
                           mode=tf.estimator.ModeKeys.EVAL,
                           label_map=config['label_map'],
                           filter_data=filter_data['val'],
                           features_set=features_set)
    test_input_fn = InputFn(file_pattern=data_dir + '/test*',
                            batch_size=config['batch_size'],
                            mode=tf.estimator.ModeKeys.EVAL,
                            label_map=config['label_map'],
                            filter_data=filter_data['test'],
                            features_set=features_set)

    # fit the model to the training data
    print('Training model...')
    history = model.fit(x=train_input_fn(),
                        y=None,
                        batch_size=None,
                        epochs=n_epochs,
                        verbose=verbose,
                        callbacks=list(callbacks_dict.values()),
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
                              verbose=verbose,
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
    predictions = {dataset: [] for dataset in datasets}
    for dataset in predictions:

        print('Predicting on dataset {}...'.format(dataset))

        predict_input_fn = InputFn(file_pattern=data_dir + '/' + dataset + '*',
                                   batch_size=config['batch_size'],
                                   mode=tf.estimator.ModeKeys.PREDICT,
                                   label_map=config['label_map'],
                                   filter_data=filter_data[dataset],
                                   features_set=features_set)

        predictions[dataset] = model.predict(predict_input_fn(),
                                             batch_size=None,
                                             verbose=verbose,
                                             steps=None,
                                             callbacks=None,
                                             max_queue_size=10,
                                             workers=1,
                                             use_multiprocessing=False,
                                             )

    # sort predictions per class based on ground truth labels
    output_cl = {dataset: {} for dataset in datasets}
    for dataset in output_cl:
        for original_label in config['label_map']:
            # get predictions for each original class individually to compute histogram
            output_cl[dataset][original_label] = predictions[dataset][np.where(original_labels[dataset] ==
                                                                               original_label)]

    # compute precision at top-k
    labels_sorted = {}
    for dataset in datasets:
        if dataset == 'predict':
            continue
        sorted_idxs = np.argsort(predictions[dataset], axis=0).squeeze()
        labels_sorted[dataset] = labels[dataset][sorted_idxs].squeeze()
        prec_at_k = compute_precision_at_k(labels_sorted[dataset], config['k_arr'][dataset])
        res.update({f'{dataset}_precision_at_{k_val}': prec_at_k[f'precision_at_{k_val}']
                    for k_val in config['k_arr'][dataset]})

    # save results in a numpy file
    res_fp = model_dir_sub / 'results.npy'
    print(f'Saving metrics to {res_fp}...')
    np.save(res_fp, res)

    print('Plotting evaluation results...')
    epochs = np.arange(1, len(res['loss']) + 1)
    # choose epoch associated with the best value for the metric
    if 'early_stopping' in callbacks_dict:
        if min_optmetric:
            ep_idx = np.argmin(res[f'val_{opt_metric}'])
        else:
            ep_idx = np.argmax(res[f'val_{opt_metric}'])
    else:
        ep_idx = -1
    # plot evaluation loss and metric curves
    plot_loss_metric(res, epochs, ep_idx, opt_metric,
                     res_dir / f'model{model_id}_plotseval_epochs{epochs[-1]:.0f}.svg')
    # plot class distribution
    for dataset in datasets:
        plot_class_distribution(output_cl[dataset],
                                res_dir / f'model{model_id}_class_predoutput_distribution_{dataset}.svg')
    # plot precision, recall, ROC AUC, PR AUC curves
    # plot_prec_rec_roc_auc_pr_auc(res, epochs, ep_idx,
                                 # os.path.join(res_dir, 'model{}_prec_rec_auc.svg'.format(model_id)))
    # plot pr curve
    plot_pr_curve(res, ep_idx, res_dir / f'model{model_id}_prec_rec.svg')
    # plot roc
    plot_roc(res, ep_idx, res_dir / f'model{model_id}_roc.svg')
    # plot precision-at-k and misclassfied-at-k examples curves
    for dataset in datasets:
        plot_precision_at_k(labels_sorted[dataset], config['k_curve_arr'][dataset],
                            res_dir / f'model{model_id}_{dataset}')

    print('Saving metrics to a txt file...')
    save_metrics_to_file(model_dir_sub, res, datasets, ep_idx, model.metrics_names, config['k_arr'])

    print_metrics(model_id, res, datasets, ep_idx, model.metrics_names, config['k_arr'])


if __name__ == '__main__':

    # used in job arrays
    parser = argparse.ArgumentParser()
    parser.add_argument('--job_idx', type=int, help='Job index', default=0)
    args = parser.parse_args()

    ngpus_per_node = 1  # number of GPUs per node

    # uncomment for MPI multiprocessing
    rank = MPI.COMM_WORLD.rank
    rank = ngpus_per_node * args.job_idx + rank
    size = MPI.COMM_WORLD.size
    print(f'Rank = {rank}/{size - 1}')
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

    # name of the study
    # study = 'keplerdr25-dv_g301-l31_6tr_spline_nongapped_norobovetterkois_starshuffle_configK_secsymphase_wksnormmaxflux-wks_corrprimgap_ptempstat_albedostat_wstdepth_fwmstat_nopps_ckoiper_secparams_prad_per'
    study = 'test_exominer_paper_astronet_config'

    # results directory
    save_path = Path(paths.pathtrainedmodels) / study
    save_path.mkdir(exist_ok=True)

    # set up logger
    logger = logging.getLogger(name='train-eval_run')
    logger_handler = logging.FileHandler(filename=save_path / f'train-eval_run_{rank}.log',
                                         mode='w')
    logger_formatter = logging.Formatter('%(asctime)s - %(message)s')
    logger.setLevel(logging.INFO)
    logger_handler.setFormatter(logger_formatter)
    logger.addHandler(logger_handler)
    logger.info(f'Starting run {study}...')

    # TFRecord files directory
    tfrec_dir = os.path.join(paths.path_tfrecs,
                             'Kepler',
                             'Q1-Q17_DR25',
                             # 'tfrecordskeplerdr25-se_std_oot_07-09-2021_data/tfrecordskeplerdr25-se_std_oot_07-09-2021_starshuffle_experiment-labels-normalized'
                             # 'tfrecordskeplerdr25-dv_g2001-l201_9tr_spline_gapped1-5_flux-loe-lwks-centroid-centroidfdl-6stellar-bfap-ghost-rollband-stdts_secsymphase_correctprimarygapping_confirmedkoiperiod_data/tfrecordskeplerdr25-dv_g2001-l201_9tr_spline_gapped1-5_flux-loe-lwks-centroid-centroidfdl-6stellar-bfap-ghost-rollband-stdts_secsymphase_correctprimarygapping_confirmedkoiperiod_starshuffle_experiment-labels-norm_nopps'
                             'tfrecordskeplerdr25-dv_g301-l31_6tr_spline_nongapped_flux-loe-lwks-centroid-centroidfdl-6stellar-bfap-ghost-rollband-stdts_secsymphase_correctprimarygapping_confirmedkoiperiod_data/tfrecordskeplerdr25-dv_g301-l31_6tr_spline_nongapped_flux-loe-lwks-centroid-centroidfdl-6stellar-bfap-ghost-rollband-stdts_secsymphase_correctprimarygapping_confirmedkoiperiod_starshuffle_experiment-labels-norm_nopps_secparams_prad_period'
                             )
    logger.info(f'Using data from {tfrec_dir}')

    # initialize configuration dictionary. Can set configuration manually
    config = {}

    # name of the HPO study from which to get a configuration; config needs to be set to None
    hpo_study = 'ConfigK-bohb_keplerdr25-dv_g301-l31_spline_nongapped_starshuffle_norobovetterkois_glflux-' \
                'glcentr_std_noclip-loe-lwks-6stellar-bfap-ghost-rollingband-convscalars_loesubtract'
    # hpo_study = 'bohb_keplerq1q17dr25_astronet_7-27-2021'
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
    BaseModel = CNN1dPlanetFinderv2  # CNN1dPlanetFinderv2
    config = baseline_configs.astronet
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
        # 'non_lin_fn': 'prelu',  # 'relu',
        # 'optimizer': 'Adam',
        # 'lr': 1e-5,
        # 'batch_size': 64,
        # 'dropout_rate': 0,
        'dropout_rate_fc_conv': 0,
        'num_fc_conv_units': 0,
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
    features_set = {
        # flux related features
        'global_flux_view_fluxnorm': {'dim': (301, 1), 'dtype': tf.float32},
        'local_flux_view_fluxnorm': {'dim': (31, 1), 'dtype': tf.float32},
        'transit_depth_norm': {'dim': (1,), 'dtype': tf.float32},
        # odd-even flux related features
        'local_flux_odd_view_fluxnorm': {'dim': (31, 1), 'dtype': tf.float32},
        'local_flux_even_view_fluxnorm': {'dim': (31, 1), 'dtype': tf.float32},
        # 'sigma_oot_odd': {'dim': (1,), 'dtype': tf.float32},
        # 'sigma_it_odd': {'dim': (1,), 'dtype': tf.float32},
        # 'sigma_oot_even': {'dim': (1,), 'dtype': tf.float32},
        # 'sigma_it_even': {'dim': (1,), 'dtype': tf.float32},
        # 'odd_se_oot': {'dim': (1,), 'dtype': tf.float32},
        # 'even_se_oot': {'dim': (1,), 'dtype': tf.float32},
        # centroid related features
        # 'global_centr_fdl_view_norm': {'dim': (301, 1), 'dtype': tf.float32},
        # 'local_centr_fdl_view_norm': {'dim': (31, 1), 'dtype': tf.float32},
        'global_centr_view_std_noclip': {'dim': (301, 1), 'dtype': tf.float32},
        'local_centr_view_std_noclip': {'dim': (31, 1), 'dtype': tf.float32},
        'tce_fwm_stat_norm': {'dim': (1,), 'dtype': tf.float32},
        'tce_dikco_msky_norm': {'dim': (1,), 'dtype': tf.float32},
        'tce_dikco_msky_err_norm': {'dim': (1,), 'dtype': tf.float32},
        'tce_dicco_msky_norm': {'dim': (1,), 'dtype': tf.float32},
        'tce_dicco_msky_err_norm': {'dim': (1,), 'dtype': tf.float32},
        # 'mag_norm': {'dim': (1,), 'dtype': tf.float32},
        # secondary related features
        # 'local_weak_secondary_view_fluxnorm': {'dim': (31, 1), 'dtype': tf.float32},
        # 'local_weak_secondary_view_selfnorm': {'dim': (31, 1), 'dtype': tf.float32},
        'local_weak_secondary_view_max_flux-wks_norm': {'dim': (31, 1), 'dtype': tf.float32},
        'tce_maxmes_norm': {'dim': (1,), 'dtype': tf.float32},
        'wst_depth_norm': {'dim': (1,), 'dtype': tf.float32},
        'tce_albedo_stat_norm': {'dim': (1,), 'dtype': tf.float32},
        'tce_ptemp_stat_norm': {'dim': (1,), 'dtype': tf.float32},
        # other diagnostic parameters
        'boot_fap_norm': {'dim': (1,), 'dtype': tf.float32},
        'tce_cap_stat_norm': {'dim': (1,), 'dtype': tf.float32},
        'tce_hap_stat_norm': {'dim': (1,), 'dtype': tf.float32},
        # 'tce_cap_hap_stat_diff_norm': {'dim': (1,), 'dtype': tf.float32},
        # 'tce_rb_tcount0n_norm': {'dim': (1,), 'dtype': tf.float32},
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
    logger.info(f'Feature set: {features_set}')

    data_augmentation = False  # if True, uses online data augmentation in the training set
    online_preproc_params = {'num_bins_global': 301, 'num_bins_local': 31, 'num_transit_dur': 5}

    n_models = 10  # number of models in the ensemble
    n_epochs = 300  # number of epochs used to train each model
    multi_class = False  # multi-class classification
    ce_weights_args = {'tfrec_dir': tfrec_dir, 'datasets': ['train'], 'label_fieldname': 'label', 'verbose': False}
    use_kepler_ce = False  # use weighted CE loss based on the class proportions in the training set
    satellite = 'kepler'  # if 'kepler' in tfrec_dir else 'tess'

    opt_metric = 'auc_pr'  # choose which metric to plot side by side with the loss
    min_optmetric = False  # if lower value is better set to True

    # callbacks list
    callbacks_dict = {}

    # early stopping callback
    callbacks_dict['early_stopping'] = callbacks.EarlyStopping(monitor=f'val_{opt_metric}',
                                                               min_delta=0,
                                                               patience=20,
                                                               verbose=1,
                                                               mode='max',
                                                               baseline=None,
                                                               restore_best_weights=True)

    # TensorBoard callback
    callbacks_dict['tensorboard'] = callbacks.TensorBoard(
                                                          histogram_freq=1,
                                                          write_graph=True,
                                                          write_images=False,
                                                          update_freq='epoch',  # either 'epoch' or 'batch'
                                                          profile_batch=2,
                                                          embeddings_metadata=None,
                                                          embeddings_freq=0
                                                          )

    # add dataset parameters
    config = models.config_keras.add_dataset_params(satellite, multi_class, use_kepler_ce, ce_weights_args, config)

    # add missing parameters in hpo with default values
    config = models.config_keras.add_default_missing_params(config=config)

    logger.info(f'Final configuration used: {config}')

    # comment for multiprocessing using MPI
    for model_i in range(n_models):
        print(f'Training model {model_i + 1} out of {n_models} on {n_epochs} epochs...')
        run_main(config=config,
                 n_epochs=n_epochs,
                 data_dir=tfrec_dir,
                 base_model=BaseModel,
                 res_dir=save_path,
                 model_id=model_i + 1,
                 opt_metric=opt_metric,
                 min_optmetric=min_optmetric,
                 callbacks_dict=callbacks_dict,
                 features_set=features_set,
                 data_augmentation=data_augmentation,
                 online_preproc_params=online_preproc_params,
                 )

    # # uncomment for multiprocessing using MPI
    # if rank < n_models:
    #     print(f'Training model {rank + 1} out of {n_models} on {n_epochs}')
    #     sys.stdout.flush()
    #     run_main(config=config,
    #              n_epochs=n_epochs,
    #              data_dir=tfrec_dir,
    #              base_model=BaseModel,
    #              res_dir=save_path,
    #              model_id = rank + 1,
    #              opt_metric=opt_metric,
    #              min_optmetric=min_optmetric,
    #              callbacks_dict=callbacks_dict,
    #              features_set=features_set,
    #              mpi_rank=rank,
    #              data_augmentation=data_augmentation,
    #              online_preproc_params=online_preproc_params,
    #              )
