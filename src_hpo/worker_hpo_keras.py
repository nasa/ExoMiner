"""
Custom TensorFlow Keras worker for the BOHB hyperparameter optimizer.
"""

# 3rd party
import numpy as np
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from hpbandster.core.worker import Worker
import matplotlib.pyplot as plt
import copy
import logging
import shutil
import multiprocessing
import traceback

# local
from src.train.train_model import train_model
from src.evaluate.evaluate_model import evaluate_model
from models.create_ensemble_avg_model import create_avg_ensemble_model
from models import models_keras
from src.utils.utils_dataio import set_tf_data_type_for_features

plt.switch_backend('agg')


def check_model_size(config):
    """ Check number of parameters in model.

    :param config: dict, HPO configuration run parameters

    :return: num_model_params: int, number of parameters in model
    """

    check_model_config = copy.deepcopy(config)
    check_model_config['features_set'] = set_tf_data_type_for_features(check_model_config['features_set'])
    base_model = getattr(models_keras, check_model_config['model_architecture'])
    check_model = base_model(config, check_model_config['features_set']).kerasModel
    num_model_params = check_model.count_params()

    return num_model_params


def delete_model_files(model_dir):
    """ Remove .keras files from the worker directory.

    :param model_dir: Path, model directory
    :return:
    """

    for model_filepath in [fp for fp in model_dir.rglob('*.keras')]:
        model_filepath.unlink(missing_ok=True)
    # model_dir.rmdir()


def evaluate_config_on_budget(worker_id_custom, config_id, config, verbose, logger):
    """ Evaluate a sampled configuration by first training a model (or a set of models) and then conducting evaluation.
    In the case of `ensemble_n` > 1, a set of `ensemble_n` models are trained and then the metrics are reported for the
    average score ensemble.

    :param worker_id_custom: int, worker id
    :param config_id: int, configuration id
    :param config: dict, configuration parameters
    :param verbose: bool, verbose if set to True
    :param logger: logger
    :return:
        # res_eval: dict, results for the ensemble during evaluation in 'train', 'val', and 'test' datasets
    """

    # check model size
    num_model_params = check_model_size(config)
    if num_model_params > float(config['max_num_model_params']):
        logger.info(f'Model is too large model ({num_model_params} parameters > {config["max_num_model_params"]}). '
                    f'Skipping model training and assigning infinite HPO loss.')
        raise ValueError(f'Model is too large model ({num_model_params} parameters > {config["max_num_model_params"]})')

    # train and evaluate only if number of parameters constraint is respected
    for model_i in range(config['ensemble_n']):  # train and evaluate an ensemble of `ensemble_n` models

        # set directory for models for the sampled config
        model_dir = config['paths']['config_models_dir'] / f'model_{model_i}'
        model_dir.mkdir(exist_ok=True)

        if verbose:
            logger.info(f'[worker_{worker_id_custom},config{config_id}] Started training model '
                        f'{model_i + 1} (out of {config["ensemble_n"]} models): '
                        f'using {config["training"]["n_epochs"]} '
                        f'epochs.')

        model_i_config = copy.deepcopy(config)

        # train single model for this configuration
        train_model(model_i_config, model_dir, logger=logger)

    models_fps = [fp / 'model.keras'
                  for fp in config['paths']['config_models_dir'].iterdir() if fp.stem.startswith('model_')
                  and fp.is_dir()]
    if config['ensemble_n'] > 1:  # create ensemble model
        if verbose:
            logger.info(f'[worker_{worker_id_custom},config{config_id}] Creating ensemble with {config["ensemble_n"]} '
                        f'trained models.')

        ensemble_fp = config['paths']['config_models_dir'] / 'ensemble_avg_model.keras'
        create_avg_ensemble_model(models_fps, config['features_set'], ensemble_fp)
        evaluate_model_fp = ensemble_fp
    else:
        evaluate_model_fp = models_fps[0]

    # evaluate model
    if verbose:
        logger.info(f'[worker_{worker_id_custom},config{config_id}] Started evaluating ensemble.')
    ensemble_model_config = copy.deepcopy(config)

    evaluate_model(ensemble_model_config, evaluate_model_fp, config['paths']['config_dir'], logger=logger)
    # res_eval = np.load(config['paths']['config_dir'] / 'res_eval.npy', allow_pickle=True).item()


def wrapper_evaluate_config_on_budget(worker_id, config_id, run_config, budget):
    """ Wrapper for `evaluate_config_on_budget()` function.

    :param worker_id: int, worker id
    :param config_id: int, configuration id
    :param run_config: dict, configuration parameters
    :param budget: int, budget allocated to configuration (number of epochs)

    :return:
    """

    # create logger for this config run
    logger = logging.getLogger(name=f'config_{config_id}_budget{budget}_log')
    logger_handler = logging.FileHandler(filename=run_config['paths']['config_dir'] /
                                                  f'config_run_budget{budget}.log', mode='a')
    logger_formatter = logging.Formatter('%(asctime)s - %(message)s')
    logger.setLevel(logging.INFO)
    logger_handler.setFormatter(logger_formatter)
    logger.addHandler(logger_handler)

    logger.info('Running evaluating configuration on budget...')

    try:
        evaluate_config_on_budget(
            worker_id,
            config_id,
            run_config,
            run_config['verbose'],
            logger
        )
    except Exception as error:  # return infinite HPO loss if there is any error during configuration evaluation
        with open(run_config['paths']['config_dir'] /
                  f'run_config_exception_budget_{budget}epochs.txt', "w") as excl_file:
            excl_file.write(f'Error: {error}\n')
            traceback.print_exc(file=excl_file)
        logger.info('Error when evaluating configuration on budget.')

    # rename results to add budget to the filename
    for fp in run_config['paths']['config_dir'].rglob('*train.npy'):
        shutil.move(fp, fp.parent / f'{fp.stem}_budget_{budget}epochs.npy')
    for fp in run_config['paths']['config_dir'].rglob('*eval.npy'):
        shutil.move(fp, fp.parent / f'{fp.stem}_budget_{budget}epochs.npy')
    results_txt_fp = run_config['paths']['config_dir'] / 'loss_and_performance_metrics.txt'
    if results_txt_fp.exists():
        shutil.move(results_txt_fp, results_txt_fp.parent / f'{results_txt_fp.stem}_budget_{budget}epochs.txt')

    logger.info('Finished evaluating configuration on budget.')


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

        self.results_directory = config_args['paths']['experiment_dir']  # directory to save results

        # base config that is fixed, and it is not part of the evaluated configuration space
        self.base_config = config_args

        self.hpo_loss = config_args['hpo_loss']  # HPO loss (it has to be one of the metrics/losses being evaluated)

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
        run_config = self.base_config.copy()
        run_config['config'].update(config)

        # set directory for sampled configuration
        run_config['paths']['config_dir'] = self.results_directory / f'config{config_id}'
        run_config['paths']['config_dir'].mkdir(exist_ok=True)
        run_config['paths']['config_models_dir'] = run_config['paths']['config_dir'] / 'models'
        run_config['paths']['config_models_dir'].mkdir(exist_ok=True)

        budget = int(budget)
        run_config['training']['n_epochs'] = int(budget)  # set budget for this config

        # create logger for this config run
        logger = logging.getLogger(name=f'config_{config_id}_budget{budget}_log')
        logger_handler = logging.FileHandler(filename=run_config['paths']['config_dir'] /
                                                      f'config_run_budget{budget}.log', mode='w')
        logger_formatter = logging.Formatter('%(asctime)s - %(message)s')
        logger.setLevel(logging.INFO)
        logger_handler.setFormatter(logger_formatter)
        logger.addHandler(logger_handler)

        logger.info(f'Sampled configuration {config_id} to be evaluated on budget {budget} epochs:\n {config}')

        p = multiprocessing.Process(target=wrapper_evaluate_config_on_budget,
                                    args=(self.worker_id_custom,
                                          config_id,
                                          run_config,
                                          budget
                                          )
                                    )
        p.start()
        p.join()

        res_ensemble_fp = run_config['paths']['config_dir'] / f'res_eval_budget_{budget}epochs.npy'
        if res_ensemble_fp.exists():
            res_ensemble = np.load(res_ensemble_fp, allow_pickle=True).item()

            # plots
            if run_config['draw_plots']:

                logger.info('Creating plots with results...')

                # get results for all trained models
                res_models = {model_i_dir.name: np.load(model_i_dir / f'res_train_budget_{budget}epochs.npy',
                                                        allow_pickle=True).item()
                              for model_i_dir in run_config['paths']['config_models_dir'].iterdir()
                              if model_i_dir.is_dir() and model_i_dir.name.startswith('model_')}

                plot_fp = (run_config['paths']['config_dir'] /
                           f'config{config_id}_budget_{budget}epochs_loss_metric_curves.png')
                self.draw_plots_ensemble(res_ensemble, res_models, config_id, budget, plot_fp)

            logger.info('Setting HPO loss and metrics...')

            # report HPO loss and additional metrics and loss
            hpo_loss = res_ensemble[f'test_{self.hpo_loss}']
            # add metrics and loss
            info_dict = {metric: res_ensemble[metric] for metric in res_ensemble if
                         isinstance(res_ensemble[metric], float)}

            res_hpo = {'loss': 1 - hpo_loss,  # HPO loss to be minimized
                       'info': info_dict
                       }
        else:
            logger.info('Error when evaluating configuration on budget. Set HPO loss to infinity.')
            run_config_error = ''
            with open(run_config['paths']['config_dir'] /
                      f'run_config_exception_budget_{budget}epochs.txt', "r") as excl_file:
                for line in excl_file:
                    run_config_error += line

            res_hpo = {'loss': np.inf, 'info': run_config_error}

        # delete trained models and ensemble
        logger.info('Deleting model Keras files.')
        delete_model_files(run_config['paths']['config_models_dir'])

        logger.info('Work for this configuration on this budget has been finished. Returning results to HPO.')

        return res_hpo

    def draw_plots_ensemble(self, res_ensemble, res_models, config_id, budget, plot_fp):
        """ Draw loss and evaluation metric plots for the ensemble.

        :param res_ensemble: dict, keys are loss and metrics on the different datasets for the ensemble
        :param res_models: dict, keys are loss and metrics on the training, validation and test set (for every epoch,
        except for the test set. Each metric/loss has a dictionary with 3 keys: 'central tendency', 'all scores' and
        'deviation'
        :param config_id: tuple, configuration ID
        :param budget: float, budget (number of epochs)
        :param plot_fp: Path, filepath to plot

        :return:
        """

        alpha = 0.3

        # initialize epochs array
        epochs = np.arange(budget)

        # plot loss and optimization metric as function of the epoch number
        f, ax = plt.subplots(1, 2)
        for model_i, (_, res_model_i) in enumerate(res_models.items()):
            ax[0].plot(epochs, res_model_i['loss'], color='b', alpha=alpha,
                       label=None if model_i != 0 else 'Train Single Model')
            # ax[0].plot(epochs, res_model_i['test_loss'], color='m', linestyle='dashed', alpha=alpha,
            #            label=None if model_i != 0 else 'Test Single Model')
        ax[0].scatter(epochs[-1], res_ensemble['train_loss'], s=8, label='Train Ensemble', color='r')
        ax[0].scatter(epochs[-1], res_ensemble['test_loss'], s=8, c='k', label='Test Ensemble')
        ax[0].set_xlim(left=epochs[0], right=epochs[-1] + 1)
        # ax[0].set_ylim(bottom=0)
        ax[0].set_xlabel('Epoch Number')
        ax[0].set_ylabel('Loss')
        ax[0].set_title(f'Train/Test {res_ensemble["train_loss"]:.4f}/{res_ensemble["test_loss"]:.4f}')
        ax[0].legend()  # loc="upper right")
        ax[0].grid(True)

        for model_i, (_, res_model_i) in enumerate(res_models.items()):
            ax[1].plot(epochs, res_model_i[self.hpo_loss], color='b', alpha=alpha,
                       label=None if model_i != 0 else 'Train Single Model')
            # ax[1].plot(epochs, res_model_i[f'test_{self.hpo_loss}'], color='m', linestyle='dashed', alpha=alpha,
            #            label=None if model_i != 0 else 'Test Single Model')
        ax[1].scatter(epochs[-1], res_ensemble[f'train_{self.hpo_loss}'], s=8, label='Train Ensemble', color='r')
        ax[1].scatter(epochs[-1], res_ensemble[f'test_{self.hpo_loss}'], s=8, label='Test Ensemble', c='k')
        ax[1].set_xlim(left=epochs[0], right=epochs[-1] + 1)
        # ax[1].set_ylim([0.0, 1.05])
        ax[1].grid(True)
        ax[1].set_xlabel('Epoch Number')
        ax[1].set_ylabel(self.hpo_loss)
        ax[1].set_title(f'Train/Test '
                        f'{res_ensemble[f"train_{self.hpo_loss}"]:.4f}/{res_ensemble[f"test_{self.hpo_loss}"]:.4f}')
        ax[1].legend()  # loc="lower right")
        f.suptitle(f'Config {config_id} | Budget = {budget} epochs')
        f.tight_layout()
        f.savefig(plot_fp)
        plt.close()


def get_configspace(config_space_setup):
    """ Build the hyperparameter configuration space.

    :param config_space_setup: dict, knk
    :return: ConfigurationsSpace-Object
    """

    config_space = CS.ConfigurationSpace()

    hyperparameters = {hyperparameter: None for hyperparameter in config_space_setup['hyperparameters']}

    for hyperparameter_name, hyperparameter_params in config_space_setup['hyperparameters'].items():
        if hyperparameter_params['type'] == 'uniform_float':
            hyperparameters[hyperparameter_name] = CSH.UniformFloatHyperparameter(name=hyperparameter_name,
                                                                                  **hyperparameter_params['parameters'])
        elif hyperparameter_params['type'] == 'categorical':
            hyperparameters[hyperparameter_name] = CSH.CategoricalHyperparameter(hyperparameter_name,
                                                                                 hyperparameter_params['parameters'])

        elif hyperparameter_params['type'] == 'uniform_integer':
            hyperparameters[hyperparameter_name] = CSH.UniformIntegerHyperparameter(name=hyperparameter_name,
                                                                                    **hyperparameter_params[
                                                                                        'parameters'])

    # add hyper-parameters
    config_space.add_hyperparameters(list(hyperparameters.values()))

    # add conditions
    for condition_name, condition_params in config_space_setup['conditions'].items():
        if condition_params['type'] == 'equal':
            cond = CS.EqualsCondition(hyperparameters[condition_params['child']],
                                      hyperparameters[condition_params['parent']],
                                      condition_params['value'])
            config_space.add_condition(cond)

    return config_space
