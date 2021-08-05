"""
Custom TensorFlow Keras worker for the BOHB hyperparameter optimizer.
"""

# 3rd party
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras import losses, optimizers
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from hpbandster.core.worker import Worker
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
from pathlib import Path
# import multiprocessing
# from memory_profiler import profile

# local
from src.utils_dataio import InputFnv2 as InputFn
from src.utils_metrics import get_metrics
from src.models_keras import create_ensemble
# import paths


# @profile
def _delete_model_files(model_dir):
    """ Remove files from the model directory.

    :param model_dir: Path, model directory
    :return:
    """

    for model_filepath in model_dir.iterdir():
        # model_filepath.unlink(missing_ok=True)
        model_filepath.unlink()


# @profile
def _ensemble_run(config, config_id, worker_id, budget, results_directory, features_set, tfrec_dir,
                  gpu_id, verbose, model_dir_rank, ensemble_n):
    """ Evaluate the ensemble.

    :param config: dict,
    :param config_id: tuple, config ID for this run
    :param worker_id: int, worker ID (rank)
    :param budget: float, budget (epochs) used to train the ensemble
    :param results_directory: str, directory used to save the results
    :param features_set: dict, features used
    :param tfrec_dir: str, TFRecord directory
    :param gpu_id: int, ID of the GPU allocated to this run
    # :param q: queue, used to send the results to the parent process
    :param verbose: bool, verbose
    :param model_dir_rank: str, model directory for a given rank
    :param ensemble_n: int, number of models in the ensemble
    :return:
        res: dict, results
    """

    try:

        # setup monitoring metrics
        if verbose:
            print(f'[worker_{worker_id},config{config_id}] Setting up metrics for ensemble')
        metrics_list = get_metrics(clf_threshold=config['clf_thr'], num_thresholds=config['num_thr'])

        if verbose:
            print(f'[worker_{worker_id},config{config_id}] Setting up additional callbacks for ensemble')
        config['callbacks_list_temp'] = []
        # config['callbacks_list_temp'].append(
        #     callbacks.TensorBoard(log_dir=results_directory / 'logs' /
        #                                   f'config{config}_budget{int(budget):.0f}_ensemble_log'),
        #                           histogram_freq=1,
        #                           write_graph=True,
        #                           write_images=True,
        #                           update_freq='epoch',
        #                           profile_batch=0,
        #                           embeddings_freq=0,
        #                           embeddings_metadata=None)
        # )

        # create ensemble
        model_list = []
        models_filepaths = [model_fp for model_fp in model_dir_rank.iterdir()
                            if '.h5' in model_fp.name]
        for model_i, model_filepath in enumerate(models_filepaths):
            model = load_model(filepath=model_filepath, compile=False)
            model._name = f'model{model_i}'

            model_list.append(model)

        assert len(model_list) == ensemble_n

        ensemble_model = create_ensemble(features=features_set,
                                         models=model_list)

        # compile ensemble
        if verbose:
            print(f'[worker_{worker_id},config{config_id}] Compiling ensemble...')
        if config['optimizer'] == 'Adam':
            ensemble_model.compile(optimizer=optimizers.Adam(learning_rate=config['lr'],
                                                             beta_1=0.9,
                                                             beta_2=0.999,
                                                             epsilon=1e-8,
                                                             amsgrad=False,
                                                             name='Adam',  # optimizer
                                                             ),
                                   # loss function to minimize
                                   loss=losses.BinaryCrossentropy(from_logits=False,
                                                                  label_smoothing=0,
                                                                  name='binary_crossentropy'),
                                   # list of metrics to monitor
                                   metrics=metrics_list,
                                   )

        else:
            ensemble_model.compile(optimizer=optimizers.SGD(learning_rate=config['lr'],
                                                            momentum=config['sgd_momentum'],
                                                            nesterov=False,
                                                            name='SGD',  # optimizer
                                                            ),
                                   # loss function to minimize
                                   loss=losses.BinaryCrossentropy(from_logits=False,
                                                                  label_smoothing=0,
                                                                  name='binary_crossentropy'),
                                   # list of metrics to monitor
                                   metrics=metrics_list,
                                   )

        if verbose:
            print(f'[worker_{worker_id},config{config_id}] Evaluating ensemble')

        # evaluate on the test set at the end of the training only
        # initialize results dictionary for the evaluated datasets
        res = {
            'num_trainable_weights': float(np.sum([K.count_params(w) for w in model_list[0].trainable_weights])),
            'num_non_trainable_weights': float(np.sum([K.count_params(w)
                                                       for w in model_list[0].non_trainable_weights])),
            'total_num_weights': float(model_list[0].count_params()),
            'ensemble_size': float(len(model_list))
        }
        for dataset in config['datasets']:

            if verbose:
                print(f'[worker_{worker_id},config{config_id}] Evaluating ensemble on dataset {dataset}')

            # input function for evaluating on each dataset
            eval_input_fn = InputFn(file_pattern=tfrec_dir + f'/{dataset}*',
                                    batch_size=config['batch_size'],
                                    mode=tf.estimator.ModeKeys.EVAL,
                                    label_map=config['label_map'],
                                    features_set=features_set
                                    )

            # evaluate ensemble in the given dataset
            res_eval = ensemble_model.evaluate(x=eval_input_fn(),
                                               y=None,
                                               batch_size=None,
                                               verbose=verbose if '/home6' not in model_dir_rank else False,
                                               sample_weight=None,
                                               steps=None,
                                               callbacks=None,
                                               max_queue_size=10,
                                               workers=1,
                                               use_multiprocessing=False,
                                               return_dict=True)

            res.update({f'{dataset}_{metric_name}': metric_val
                        for metric_name, metric_val in res_eval.items()})

    except Exception as e:
        if verbose:
            print(f'[worker_{worker_id},config{config_id}] Error while evaluating ensemble: {e}')
            print(f'[worker_{worker_id},config{config_id}] Deleting model files...')
        _delete_model_files(Path(results_directory) / f'models_rank{worker_id}')
        sys.stdout.flush()
        # q.put({'Error while evaluating ensemble': e})
        # exit()
        return {f'[worker_{worker_id},config{config_id}] Error while evaluating ensemble': e}

    if verbose:
        print(f'[worker_{worker_id},config{config_id}] Deleting model files...')
    _delete_model_files(Path(results_directory) / f'models_rank{worker_id}')

    if verbose:
        print(f'[worker_{worker_id},config{config_id}] Finished ensemble evaluation.')

    sys.stdout.flush()

    # q.put(res)
    return res


# @profile
def _model_run(config, config_id, worker_id, budget, model_i, results_directory, features_set,
               tfrec_dir, BaseModel, gpu_id, verbose):
    """ Train and evaluate a model.

    :param config: dict, configuration
    :param config_id: tuple, config ID for this run
    :param worker_id: int, worker ID (rank)
    :param budget: float, budget (epochs) used to train the ensemble
    :param model_i: int, model ID
    :param results_directory: str, directory used to save the results
    :param features_set: dict, features used
    :param tfrec_dir: str, TFRecord directory
    :param gpu_id: int, ID of the GPU allocated to this run
    # :param q: queue, used to send the results to the parent process
    :param verbose: bool, verbose
    :return:
        res: dict, results
    """

    try:

        if verbose:
            print(f'[worker_{worker_id},config{config_id}] Setting up input functions...')
        # input function for training on the training set
        train_input_fn = InputFn(file_pattern=tfrec_dir + '/train*',
                                 batch_size=config['batch_size'],
                                 mode=tf.estimator.ModeKeys.TRAIN,
                                 label_map=config['label_map'],
                                 data_augmentation=config['data_augmentation'],
                                 features_set=features_set)

        # input functions for evaluating on the validation and test sets
        val_input_fn = InputFn(file_pattern=tfrec_dir + '/val*',
                               batch_size=config['batch_size'],
                               mode=tf.estimator.ModeKeys.EVAL,
                               label_map=config['label_map'],
                               features_set=features_set)
        test_input_fn = InputFn(file_pattern=tfrec_dir + '/test*',
                                batch_size=config['batch_size'],
                                mode=tf.estimator.ModeKeys.EVAL,
                                label_map=config['label_map'],
                                features_set=features_set)

        # setup monitoring metrics
        if verbose:
            print(f'[worker_{worker_id},config{config_id}] Setting up metrics...')
        metrics_list = get_metrics(clf_threshold=config['clf_thr'], num_thresholds=config['num_thr'])

        if verbose:
            print(f'[worker_{worker_id},config{config_id}] Setting up additional callbacks...')
        config['callbacks_list_temp'] = []

        # if model_i == 0:
        #     # Tensorboard callback
        #     config['callbacks_list_temp'].append(
        #         tf.keras.callbacks.TensorBoard(log_dir=os.path.join(results_directory, 'logs',
        #                                                             'config{}_budget{:.0f}_model{}_log'.format(
        #                                                                 config_id,
        #                                                                 int(budget),
        #                                                                 model_i)),
        #                                        histogram_freq=1,
        #                                        write_graph=True,
        #                                        write_images=True,
        #                                        update_freq='epoch',
        #                                        profile_batch=0,
        #                                        embeddings_freq=0,
        #                                        embeddings_metadata=None)
        #     )
        #


        # instantiate Keras model
        if verbose:
            print(f'[worker_{worker_id},config{config_id}] Instantiating model...')
        model = BaseModel(config, features_set).kerasModel

        # compile model
        if verbose:
            print(f'[worker_{worker_id},config{config_id}] Compiling model...')
        if config['optimizer'] == 'Adam':
            model.compile(optimizer=optimizers.Adam(learning_rate=config['lr'],
                                                    beta_1=0.9,
                                                    beta_2=0.999,
                                                    epsilon=1e-8,
                                                    amsgrad=False,
                                                    name='Adam',  # optimizer
                                                    ),
                          # loss function to minimize
                          loss=losses.BinaryCrossentropy(from_logits=False,
                                                         label_smoothing=0,
                                                         name='binary_crossentropy'),
                          # list of metrics to monitor
                          metrics=metrics_list,
                          )

        else:
            model.compile(optimizer=optimizers.SGD(learning_rate=config['lr'],
                                                   momentum=config['sgd_momentum'],
                                                   nesterov=False,
                                                   name='SGD',  # optimizer
                                                   ),
                          # loss function to minimize
                          loss=losses.BinaryCrossentropy(from_logits=False,
                                                         label_smoothing=0,
                                                         name='binary_crossentropy'),
                          # list of metrics to monitor
                          metrics=metrics_list,
                          )

        if verbose:
            print(f'[worker_{worker_id},config{config_id}] Training model {model_i}...')
        # fit the model to the training data
        history = model.fit(x=train_input_fn(),
                            y=None,
                            batch_size=None,
                            epochs=int(budget),
                            verbose=verbose if '/home6' not in str(results_directory) else False,
                            callbacks=config['callbacks_list'] + config['callbacks_list_temp'],
                            validation_split=0.,
                            validation_data=val_input_fn(),
                            shuffle=True,  # does the input function shuffle for every epoch?
                            class_weight=None,
                            sample_weight=None,
                            initial_epoch=0,
                            steps_per_epoch=None,
                            validation_steps=None,
                            max_queue_size=10,  # used for generator or keras.utils.Sequence input only
                            workers=1,  # same
                            use_multiprocessing=False  # same
                            )

        # get records of loss and metrics for the training and validation sets
        res = history.history

        if verbose:
            print(f'[worker_{worker_id},config{config_id}] Evaluating model on the test set')

        # evaluate on the test set at the end of the training only
        res_eval = model.evaluate(x=test_input_fn(),
                                  y=None,
                                  batch_size=None,
                                  verbose=False,  # verbose if '/home6' not in str(results_directory) else False,
                                  sample_weight=None,
                                  steps=None,
                                  callbacks=None,
                                  max_queue_size=10,
                                  workers=1,
                                  use_multiprocessing=False,
                                  return_dict=True)

        res.update({f'test_{metric_name}': metric_val for metric_name, metric_val in res_eval.items()})

        # plot model architecture
        if model_i == 0:
            if verbose:
                print(f'[worker_{worker_id},config{config_id}] Plotting model')
            plot_model(model,
                       to_file=results_directory / f'config{config_id}_model.png',
                       show_shapes=True,
                       show_layer_names=True,
                       rankdir='TB',
                       expand_nested=False,
                       dpi=48)

        model.save(results_directory / f'models_rank{worker_id}' / f'model{model_i}.h5')

    except Exception as e:
        if verbose:
            print(f'[worker_{worker_id},config{config_id}] Error during model fit: {e}')
        sys.stdout.flush()
        # q.put({'Error during model fit': e})
        # exit()
        return {f'[worker_{worker_id},config{config_id}] Error during model fit': e}

    if verbose:
        print(f'[worker_{worker_id},config{config_id}] Finished model fit.')

    sys.stdout.flush()

    # q.put(res)
    return res


# @profile
def _evaluate_config(ensemble_n, worker_id_custom, config_id, budget, gpu_id, config, results_directory, features_set,
                     tfrec_dir, BaseModel, verbose, model_dir_rank):
    # initialize results variable
    res_models = {}
    for model_i in range(ensemble_n):

        if verbose:
            printstr = f'[worker_{worker_id_custom},config{config_id}] Training model ' \
                       f'{model_i + 1}({ensemble_n}): {int(budget)} epochs'

            print(printstr)
            sys.stdout.flush()

        with tf.device(f'/gpu:{gpu_id}'):
            res_model_i = _model_run(config,
                                     config_id,
                                     worker_id_custom,
                                     budget,
                                     model_i,
                                     results_directory,
                                     features_set,
                                     tfrec_dir,
                                     BaseModel,
                                     gpu_id,
                                     verbose)

        if f'[worker_{worker_id_custom},config{config_id}] Error during model fit' in res_model_i:
            if verbose:
                print(f'[worker_{worker_id_custom},config{config_id}] Error in model fit.')
                sys.stdout.flush()

            return {'loss': np.inf,
                    'info': f'Error during model fit: '
                            f'{res_model_i[f"[worker_{worker_id_custom},config{config_id}] Error during model fit"]}'}

        # add results for this model to the results for the ensemble
        if len(res_models.keys()) == 0:
            res_models = {key: [val] for key, val in res_model_i.items()}
        else:
            for metric_name in res_models:
                res_models[metric_name].append(res_model_i[metric_name])

    # get ensemble average metrics and loss
    for metric in res_models:
        # res[metric] = {'all scores': res[metric],
        #                         'median': np.median(res[metric], axis=0),
        #                         'mad': np.median(np.abs(res[metric] -
        #                                                 np.median(res[metric], axis=0)), axis=0)}
        res_models[metric] = {'all scores': res_models[metric],
                              'central tendency': np.mean(res_models[metric], axis=0),
                              # TODO: check sqrt and decide if we want to print/save the model variability
                              'deviation': np.std(res_models[metric], axis=0, ddof=1) / np.sqrt(ensemble_n)}

    if ensemble_n > 1:
        with tf.device(f'/gpu:{gpu_id}'):
            res_ensemble = _ensemble_run(config,
                                         config_id,
                                         worker_id_custom,
                                         budget,
                                         results_directory,
                                         features_set,
                                         tfrec_dir,
                                         gpu_id,
                                         verbose,
                                         model_dir_rank,
                                         ensemble_n
                                         )

        if f'[worker_{worker_id_custom},config{config_id}] Error while evaluating ensemble' in res_ensemble:
            if verbose:
                print(f'[worker_{worker_id_custom},config{config_id}] Error in ensemble run subprocess')
                sys.stdout.flush()

            return {'loss': np.inf,
                    'info': 'Error while evaluating ensemble: '
                            f'{res_ensemble[f"[worker_{worker_id_custom},config{config_id}] Error while evaluating ensemble"]}'
                    }
    else:
        res_ensemble = {}
        for metric_name, metric_val in res_models.items():
            if 'test' in metric_name or 'val' in metric_name:
                metric_name_aux = metric_name
            else:
                metric_name_aux = f'train_{metric_name}'

            if isinstance(metric_val['all scores'][0], list):
                res_ensemble[metric_name_aux] = metric_val['all scores'][0][-1]
            else:
                res_ensemble[metric_name_aux] = metric_val['all scores'][0]

    # q.put([res_ensemble, res_models])
    return res_ensemble, res_models


class TransitClassifier(Worker):
    """ Custom BOHB worker. """

    def __init__(self, config_args, worker_id_custom=1, **kwargs):
        """ Initialize the worker.

        :param config_args: dict, configuration parameters
        :param worker_id_custom: int, worker ID
        :param kwargs:  dict, other parameters for the worker
        """

        # call the init of the inherited worker class
        super().__init__(**kwargs)

        self.worker_id_custom = str(worker_id_custom)  # worker id

        self.results_directory = config_args['results_directory']  # directory to save results
        if 'home6' in self.results_directory.as_posix():
            plt.switch_backend('agg')
        self.tfrec_dir = config_args['tfrec_dir']  # tfrecord directory
        self.model_dir_rank = config_args['model_dir_rank']  # model directory

        # base config that is fixed and it is not part of the evaluated configuration space
        self.base_config = config_args['experiment_config']

        self.hpo_loss = config_args['hpo_loss']  # HPO loss (it has to be one of the metrics/loss being evaluated

        self.ensemble_n = config_args['ensemble_n']  # number of models trained per configuration

        # # get cross-entropy class weights (weighted cross-entropy)
        # if config_args.satellite == 'kepler' and not config_args.use_kepler_ce:
        #     self.ce_weights = None
        # else:
        #     self.ce_weights = config_args.ce_weights

        self.features_set = config_args['features_set']

        self.BaseModel = config_args['base_model']

        self.gpu_id = config_args['gpu_id']

        self.verbose = config_args['verbose']

    # @profile
    def compute(self, config_id, config, budget, working_directory, *args, **kwargs):
        """ Evaluate a sampled configuration.

        :param config_id: tuple, configuration ID
        :param config: sampled configuration
        :param budget: int, budget (number of epochs to train the configuration sampled)
        :param working_directory: str, a name of a directory that is unique to this configuration. Use this to store
        intermediate results on lower budgets that can be reused later for a larger budget (for iterative algorithms,
        for example).
        :param args: tuple, other args
        :param kwargs: dict, other args
        :return:
            dict:
                needs to return a dictionary with two mandatory entries:
                - 'loss': a numerical value that is MINIMIZED
                - 'info': This can be pretty much any build in python type, e.g. a dict with lists as value.
                Due to Pyro4 handling the remote function calls, 3rd party types like numpy arrays are not supported!
        """

        # merge sampled and fixed configurations
        config.update(self.base_config)

        budget = int(budget)

        # q = multiprocessing.Queue()
        # p = multiprocessing.Process(target=_evaluate_config, args=(self.ensemble_n,
        #                                                            self.worker_id_custom,
        #                                                            config_id,
        #                                                            budget,
        #                                                            self.gpu_id,
        #                                                            config,
        #                                                            self.results_directory,
        #                                                            self.features_set,
        #                                                            self.tfrec_dir,
        #                                                            self.BaseModel,
        #                                                            self.verbose,
        #                                                            self.model_dir_rank,
        #                                                            q
        #                                                            ))
        # p.start()
        # res_ensemble, res_models = q.get()
        # sys.stdout.flush()
        # p.join()

        res_ensemble, res_models = _evaluate_config(
            self.ensemble_n,
            self.worker_id_custom,
            config_id,
            budget,
            self.gpu_id,
            config,
            self.results_directory,
            self.features_set,
            self.tfrec_dir,
            self.BaseModel,
            self.verbose,
            self.model_dir_rank,
        )

        # save metrics and loss for the ensemble
        res_total = {'ensemble': res_ensemble, 'single_models': res_models}
        np.save(self.results_directory / f'config{config_id}_budget{budget:.0f}_ensemblemetrics.npy', res_total)

        # draw loss and evaluation metric plots for the model on this given budget
        # self.draw_plots(res, config_id)
        self.draw_plots_ensemble(res_ensemble, res_models, config_id, budget)

        # report HPO loss and additional metrics and loss
        # hpo_loss_val = res['val_{}'.format(self.hpo_loss)]['central tendency'][-1]
        # # add test metrics and loss
        # info_dict = {metric: [float(res[metric]['central tendency']), float(res[metric]['deviation'])]
        #              for metric in res if 'test' in metric and len(res[metric]['central tendency'].shape) == 0}
        # # add train and validation metrics and loss
        # info_dict.update({metric: [float(res[metric]['central tendency'][-1]), float(res[metric]['deviation'][-1])]
        #                  for metric in res if 'test' not in metric and len(res[metric]['central tendency'].shape) == 1})
        hpo_loss_val = res_ensemble[f'val_{self.hpo_loss}']
        # add metrics and loss
        info_dict = {metric: res_ensemble[metric] for metric in res_ensemble if isinstance(res_ensemble[metric], float)}

        res_hpo = {'loss': 1 - hpo_loss_val,  # HPO loss to be minimized
                   'info': info_dict
                   }

        print('#' * 100)
        print(f'[worker_{self.worker_id_custom},config{config_id}] Finished evaluating configuration using a budget of '
              f'{int(budget)}')
        for k in res_hpo:
            if k != 'info':
                print(f'HPO {k}: {res_hpo[k]}')
            else:
                for l in res_hpo[k]:
                    print(f'{l}: {res_hpo[k][l]}')
        print('#' * 100)
        sys.stdout.flush()

        return res_hpo

    # @profile
    def draw_plots(self, res, config_id):
        """ Draw loss and evaluation metric plots. This function would plot the average performance of the models
        instead of the performance of the ensemble.

        :param res: dict, keys are loss and metrics on the training, validation and test set (for every epoch, except
        for the test set. Each metric/loss has a dictionary with 3 keys: 'central tendency', 'all scores' and
        'deviation'
        :param config_id: tuple, configuration ID
        :return:
        """

        # initialize epochs array
        epochs = np.arange(1, len(res['loss']['central tendency']) + 1)

        alpha = 0.1

        # plot loss and optimization metric as function of the epochs
        f, ax = plt.subplots(1, 2)
        for training_loss_i in res['loss']['all scores']:
            ax[0].plot(epochs, training_loss_i, color='b', alpha=alpha)
        ax[0].plot(epochs, res['loss']['central tendency'], label='Training', color='b')
        for validation_loss_i in res['val_loss']['all scores']:
            ax[0].plot(epochs, validation_loss_i, color='r', alpha=alpha)
        ax[0].plot(epochs, res['val_loss']['central tendency'], label='Validation', color='r')
        for test_loss_i in res['test_loss']['all scores']:
            ax[0].scatter(epochs[-1], test_loss_i, c='k', alpha=alpha)
        ax[0].scatter(epochs[-1], res['test_loss']['central tendency'], c='k', label='Test')
        ax[0].set_xlim([0, epochs[-1]])
        # ax[0].set_ylim(bottom=0)
        ax[0].set_xlabel('Epochs')
        ax[0].set_ylabel('Loss')
        ax[0].set_title(f'Val/Test {res["val_loss"]["central tendency"][-1]:.4f}/'
                        f'{res["test_loss"]["central tendency"]:.4f}')
        ax[0].legend(loc="upper right")
        ax[0].grid(True)

        for training_hpoloss_i in res[self.hpo_loss]['all scores']:
            ax[1].plot(epochs, training_hpoloss_i, color='b', alpha=alpha)
        ax[1].plot(epochs, res[self.hpo_loss]['central tendency'], label='Training', color='b')
        for validation_hpoloss_i in res[f'val_{self.hpo_loss}']['all scores']:
            ax[1].plot(epochs, validation_hpoloss_i, color='r', alpha=alpha)
        ax[1].plot(epochs, res[f'val_{self.hpo_loss}']['central tendency'], label='Validation', color='r')
        for test_hpoloss_i in res[f'test_{self.hpo_loss}']['all scores']:
            ax[1].scatter(epochs[-1], test_hpoloss_i, c='k', alpha=alpha)
        ax[1].scatter(epochs[-1], res[f'test_{self.hpo_loss}']['central tendency'], label='Test', c='k')
        ax[1].set_xlim([0, epochs[-1]])
        # ax[1].set_ylim([0.0, 1.05])
        ax[1].grid(True)
        ax[1].set_xlabel('Epochs')
        ax[1].set_ylabel(self.hpo_loss)
        ax[1].set_title(f'{self.hpo_loss}\nVal/Test '
                        f'{res[f"val_{self.hpo_loss}"]["central tendency"][-1]:.4f}/'
                        f'{res[f"test_{self.hpo_loss}"]["central tendency"]:.4f}')
        ax[1].legend(loc="lower right")
        f.suptitle(f'Config {config_id} | Budget = {epochs[-1]:.0f}')
        f.subplots_adjust(top=0.85, bottom=0.091, left=0.131, right=0.92, hspace=0.2, wspace=0.357)
        f.savefig(self.results_directory / f'config{config_id}_budget{epochs[-1]:.0f}_mloss-hpoloss.png')
        plt.close()

        # plot precision, recall, roc auc, pr auc curves for the validation and test sets
        # f, ax = plt.subplots()
        # ax.plot(epochs, res['val_precision']['central tendency'], label='Val Precision', color='b')
        # ax.plot(epochs, res['val_recall']['central tendency'], label='Val Recall', color='r')
        # ax.plot(epochs, res['val_auc_roc']['central tendency'], label='Val ROC AUC', color='g')
        # ax.plot(epochs, res['val_auc_pr']['central tendency'], label='Val PR AUC', color='k')
        # ax.scatter(epochs[-1], res['test_precision']['central tendency'], label='Test Precision', c='b')
        # ax.scatter(epochs[-1], res['test_recall']['central tendency'], label='Test Recall', c='r')
        # ax.scatter(epochs[-1], res['test_auc_roc']['central tendency'], label='Test ROC AUC', c='g')
        # ax.scatter(epochs[-1], res['test_auc_pr']['central tendency'], label='Test PR AUC', c='k')
        # ax.grid(True)
        # chartBox = ax.get_position()
        # ax.set_position([chartBox.x0, chartBox.y0, chartBox.width * 0.6, chartBox.height])
        # ax.legend(loc='upper center', bbox_to_anchor=(1.45, 0.8), shadow=True, ncol=1)
        # ax.set_xlim([0, epochs[-1]])
        # ax.set_ylim([0, 1])
        # ax.set_xlabel('Epochs')
        # ax.set_ylabel('Metric Value')
        # ax.set_title('Precision-Recall-ROC AUC-PR AUC\nVal/Test')
        # # ax[1].legend(loc="lower right")
        # # f.subplots_adjust(top=0.85, bottom=0.091, left=0.131, right=0.92, hspace=0.2, wspace=0.357)
        # f.savefig(os.path.join(self.results_directory,
        #                        'config{}_budget{}_prec-rec-auc.png'.format(config_id, epochs[-1])))
        # plt.close()

        # plot pr curve
        f, ax = plt.subplots()
        ax.plot(res['val_rec_thr']['central tendency'][-1], res['val_prec_thr']['central tendency'][-1],
                label=f'Val (AUC={res["val_auc_pr"]["central tendency"][-1]:.3f})', color='r')
        ax.plot(res['test_rec_thr']['central tendency'], res['test_prec_thr']['central tendency'],
                label=f'Test (AUC={res["test_auc_pr"]["central tendency"]:.3f})', color='k')
        ax.plot(res['rec_thr']['central tendency'][-1], res['prec_thr']['central tendency'][-1],
                label=f'Train (AUC={res["auc_pr"]["central tendency"][-1]:.3f})', color='b')
        # ax.scatter(res['val_rec_thr']['central tendency'][-1],
        #            res['val_prec_thr']['central tendency'][-1], c='r', s=5)
        # ax.scatter(res['test_rec_thr']['central tendency'],
        #            res['test_prec_thr']['central tendency'], c='k', s=5)
        # ax.scatter(res['rec_thr']['central tendency'][-1],
        #            res['prec_thr']['central tendency'][-1], c='b', s=5)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_xticks(np.linspace(0, 1, num=11, endpoint=True))
        ax.set_yticks(np.linspace(0, 1, num=11, endpoint=True))
        ax.grid(True)
        ax.legend(loc='lower left')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision Recall curve')
        f.savefig(self.results_directory / f'config{config_id}_budget{epochs[-1]}_prcurve.png')
        plt.close()

    # @profile
    def draw_plots_ensemble(self, res_ensemble, res_models, config_id, budget):
        """ Draw loss and evaluation metric plots for the ensemble.

        :param res_ensemble: dict, keys are loss and metrics on the different datasets for the ensemble
        :param res_models: dict, keys are loss and metrics on the training, validation and test set (for every epoch,
        except for the test set. Each metric/loss has a dictionary with 3 keys: 'central tendency', 'all scores' and
        'deviation'
        :param config_id: tuple, configuration ID
        :param budget: float, budget (number of epochs)
        :return:
        """

        # initialize epochs array
        epochs = np.arange(1, int(budget) + 1)

        alpha = 0.3

        # plot loss and optimization metric as function of the epochs
        f, ax = plt.subplots(1, 2)
        for training_loss_i in res_models['loss']['all scores']:
            ax[0].plot(epochs, training_loss_i, color='b', alpha=alpha)
        ax[0].scatter(epochs[-1], res_ensemble['train_loss'], label='Training', color='b')
        for validation_loss_i in res_models['val_loss']['all scores']:
            ax[0].plot(epochs, validation_loss_i, color='r', alpha=alpha)
        ax[0].scatter(epochs[-1], res_ensemble['val_loss'], label='Validation', color='r')
        for test_loss_i in res_models['test_loss']['all scores']:
            ax[0].scatter(epochs[-1], test_loss_i, c='k', alpha=alpha)
        ax[0].scatter(epochs[-1], res_ensemble['test_loss'], c='k', label='Test')
        ax[0].set_xlim([0, epochs[-1] + 1])
        # ax[0].set_ylim(bottom=0)
        ax[0].set_xlabel('Epochs')
        ax[0].set_ylabel('Loss')
        ax[0].set_title(f'Val/Test {res_ensemble["val_loss"]:.4f}/{res_ensemble["test_loss"]:.4f}')
        ax[0].legend(loc="upper right")
        ax[0].grid(True)

        for training_hpoloss_i in res_models[self.hpo_loss]['all scores']:
            ax[1].plot(epochs, training_hpoloss_i, color='b', alpha=alpha)
        ax[1].scatter(epochs[-1], res_ensemble[f'train_{self.hpo_loss}'], label='Training', color='b')
        for validation_hpoloss_i in res_models[f'val_{self.hpo_loss}']['all scores']:
            ax[1].plot(epochs, validation_hpoloss_i, color='r', alpha=alpha)
        ax[1].scatter(epochs[-1], res_ensemble[f'val_{self.hpo_loss}'], label='Validation', color='r')
        for test_hpoloss_i in res_models[f'test_{self.hpo_loss}']['all scores']:
            ax[1].scatter(epochs[-1], test_hpoloss_i, c='k', alpha=alpha)
        ax[1].scatter(epochs[-1], res_ensemble[f'test_{self.hpo_loss}'], label='Test', c='k')
        ax[1].set_xlim([0, epochs[-1] + 1])
        # ax[1].set_ylim([0.0, 1.05])
        ax[1].grid(True)
        ax[1].set_xlabel('Epochs')
        ax[1].set_ylabel(self.hpo_loss)
        ax[1].set_title(f'{self.hpo_loss}\nVal/Test '
                        f'{res_ensemble[f"val_{self.hpo_loss}"]:.4f}/{res_ensemble[f"test_{self.hpo_loss}"]:.4f}')
        ax[1].legend(loc="lower right")
        f.suptitle(f'Config {config_id} | Budget = {epochs[-1]}')
        f.subplots_adjust(top=0.85, bottom=0.091, left=0.131, right=0.92, hspace=0.2, wspace=0.357)
        f.savefig(self.results_directory / f'config{config_id}_budget{epochs[-1]}_mloss-hpoloss.png')
        plt.close()

        # # plot PR curve
        # f, ax = plt.subplots()
        # ax.plot(res_ensemble['val_rec_thr'], res_ensemble['val_prec_thr'],
        #         label=f'Val (AUC={res_ensemble["val_auc_pr"]:.3f})', color='r')
        # ax.plot(res_ensemble['test_rec_thr'], res_ensemble['test_prec_thr'],
        #         label=f'Test (AUC={res_ensemble["test_auc_pr"]:.3f})', color='k')
        # ax.plot(res_ensemble['train_rec_thr'], res_ensemble['train_prec_thr'],
        #         label=f'Train (AUC={res_ensemble["train_auc_pr"]:.3f})', color='b')
        # # ax.scatter(res['val_rec_thr']['central tendency'][],
        # #            res['val_prec_thr']['central tendency'][-1], c='r', s=5)
        # # ax.scatter(res['test_rec_thr']['central tendency'],
        # #            res['test_prec_thr']['central tendency'], c='k', s=5)
        # # ax.scatter(res['rec_thr']['central tendency'][-1],
        # #            res['prec_thr']['central tendency'][-1], c='b', s=5)
        # ax.set_xlim([0, 1])
        # ax.set_ylim([0, 1])
        # ax.set_xticks(np.linspace(0, 1, num=11, endpoint=True))
        # ax.set_yticks(np.linspace(0, 1, num=11, endpoint=True))
        # ax.grid(True)
        # ax.legend(loc='lower left')
        # ax.set_xlabel('Recall')
        # ax.set_ylabel('Precision')
        # ax.set_title('Precision Recall curve')
        # f.savefig(self.results_directory / f'config{config_id}_budget{epochs[-1]}_prcurve.png')
        # plt.close()

    # @profile
    @staticmethod
    def get_configspace():
        """ Build the hyperparameter configuration space.

        :return: ConfigurationsSpace-Object
        """

        config_space = CS.ConfigurationSpace()

        # use_softmax = CSH.CategoricalHyperparameter('use_softmax', [True, False])

        lr = CSH.UniformFloatHyperparameter('lr', lower=1e-6, upper=1e-1, default_value=1e-2, log=True)
        # lr_scheduler = CSH.CategoricalHyperparameter('lr_scheduler', ['constant', 'inv_exp_fast', 'inv_exp_slow'])

        optimizer = CSH.CategoricalHyperparameter('optimizer', ['Adam', 'SGD'])

        sgd_momentum = CSH.UniformFloatHyperparameter('sgd_momentum', lower=0.001, upper=0.99, default_value=0.9,
                                                      log=True)

        # batch_norm = CSH.CategoricalHyperparameter('batch_norm', [True, False])
        # non_lin_fn = CSH.CategoricalHyperparameter('non_lin_fn', ['relu', 'prelu'])
        # weight_initializer = CSH.CategoricalHyperparameter('weight_initializer', ['he', 'glorot'])

        # batch_size = CSH.CategoricalHyperparameter('batch_size', [4, 8, 16, 32, 64, 128, 256], default_value=32)

        # previous values: 0.001 to 0.7
        dropout_rate = CSH.UniformFloatHyperparameter('dropout_rate', lower=0.001, upper=0.2, default_value=0.2,
                                                      log=True)

        # l2_regularizer = CSH.CategoricalHyperparameter('l2_regularizer', [True, False])
        # l2_decay_rate = CSH.UniformFloatHyperparameter('decay_rate', lower=1e-4, upper=1e-1, default_value=1e-2,
        #                                                log=True)

        config_space.add_hyperparameters([
            sgd_momentum,
            dropout_rate,
            optimizer,
            lr
                                          # lr, optimizer, batch_size, l2_regularizer, l2_decay_rate, use_softmax, lr_scheduler,
                                          # batch_norm, non_lin_fn, weight_initializer
                                          ])

        cond = CS.EqualsCondition(sgd_momentum, optimizer, 'SGD')
        config_space.add_condition(cond)

        # cond = CS.EqualsCondition(l2_decay_rate, l2_regularizer, True)
        # config_space.add_condition(cond)

        init_conv_filters = CSH.UniformIntegerHyperparameter('init_conv_filters', lower=2, upper=6, default_value=4)
        kernel_size = CSH.UniformIntegerHyperparameter('kernel_size', lower=1, upper=8, default_value=2)
        kernel_stride = CSH.UniformIntegerHyperparameter('kernel_stride', lower=1, upper=2, default_value=1)
        conv_ls_per_block = CSH.UniformIntegerHyperparameter('conv_ls_per_block', lower=1, upper=3, default_value=1)

        pool_size_loc = CSH.UniformIntegerHyperparameter('pool_size_loc', lower=2, upper=8, default_value=2)
        pool_size_glob = CSH.UniformIntegerHyperparameter('pool_size_glob', lower=2, upper=8, default_value=2)
        pool_stride = CSH.UniformIntegerHyperparameter('pool_stride', lower=1, upper=2, default_value=1)

        num_loc_conv_blocks = CSH.UniformIntegerHyperparameter('num_loc_conv_blocks', lower=1, upper=5, default_value=2)
        num_glob_conv_blocks = CSH.UniformIntegerHyperparameter('num_glob_conv_blocks', lower=1, upper=5,
                                                                default_value=3)

        # num_fc_conv_units = CSH.CategoricalHyperparameter('num_fc_conv_units', [4, 8, 16, 32])
        dropout_ratefc_conv = CSH.UniformFloatHyperparameter('dropout_rate_fc_conv', lower=0.001, upper=0.2,
                                                             default_value=0.2, log=True)

        init_fc_neurons = CSH.CategoricalHyperparameter('init_fc_neurons', [32, 64, 128, 256, 512])
        num_fc_layers = CSH.UniformIntegerHyperparameter('num_fc_layers', lower=1, upper=4, default_value=2)

        config_space.add_hyperparameters([
            num_glob_conv_blocks,
            num_loc_conv_blocks,
            init_conv_filters,
            conv_ls_per_block,
            kernel_size,
            kernel_stride,
            pool_size_glob,
            pool_stride,
            pool_size_loc,
            # num_fc_conv_units,
            dropout_ratefc_conv,
            init_fc_neurons,
            num_fc_layers,
        ])

        return config_space
