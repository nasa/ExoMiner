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
# import traceback

# local
from src.train_model import train_model
from src.evaluate_model import evaluate_model
from models.create_ensemble_avg_model import create_avg_ensemble_model

plt.switch_backend('agg')


def delete_model_files(model_dir):
    """ Remove .keras files from the worker directory.

    :param model_dir: Path, model directory
    :return:
    """

    for model_filepath in [fp for fp in model_dir.rglob('*.keras')]:
        model_filepath.unlink(missing_ok=True)
    # model_dir.rmdir()


def evaluate_config(worker_id_custom, config_id, config, verbose):
    """ Evaluate a sampled configuration by first training a model (or a set of models) and then conducting evaluation.
    In the case of `ensemble_n` > 1, a set of `ensemble_n` models are trained and then the metrics are reported for the
    average score ensemble.

    :param worker_id_custom: int, worker id
    :param config_id: int, configuration id
    :param config: dict, configuration parameters
    :param verbose:
    :return:
        res_eval: dict, results for the ensemble during evaluation in 'train', 'val', and 'test' datasets
    """

    for model_i in range(config['ensemble_n']):  # train and evaluate an ensemble of `ensemble_n` models

        # set directory for models for the sampled config
        model_dir = config['paths']['config_models_dir'] / f'model_{model_i}'
        model_dir.mkdir(exist_ok=True)

        if verbose:
            printstr = f'[worker_{worker_id_custom},config{config_id}] Started training model ' \
                       f'{model_i + 1}({config["ensemble_n"]} models): using {config["training"]["n_epochs"]} epochs.'
            print(printstr)
            # sys.stdout.flush()

        # choose random fold from the training set to use as validation set for this model
        model_i_config = copy.deepcopy(config)
        rng = np.random.default_rng(seed=config['rnd_seed'] + model_i)
        model_i_config['datasets_fps']['val'] = [rng.choice([fp for fp in config['datasets_fps']['train']])]
        model_i_config['datasets_fps']['train'].remove(model_i_config['datasets_fps']['val'][0])
        # train single model for this configuration
        train_model(model_i_config, model_dir, logger=None)

    models_fps = [fp / 'model.keras'
                  for fp in config['paths']['config_models_dir'].iterdir() if fp.stem.startswith('model_')
                  and fp.is_dir()]
    if config['ensemble_n'] > 1:  # create ensemble model
        if verbose:
            printstr = (f'[worker_{worker_id_custom},config{config_id}] Creating ensemble with {config["ensemble_n"]} '
                        f'trained models.')
            print(printstr)

        ensemble_fp = config['paths']['config_models_dir'] / 'ensemble_avg_model.keras'
        create_avg_ensemble_model(models_fps, config['features_set'], ensemble_fp)
        evaluate_model_fp = ensemble_fp
    else:
        evaluate_model_fp = models_fps[0]

    # evaluate model
    if verbose:
        printstr = f'[worker_{worker_id_custom},config{config_id}] Started evaluating ensemble.'
        print(printstr)
    ensemble_model_config = copy.deepcopy(config)
    # no evaluation on the validation set since each model in the ensemble gets a random training fold as validation set
    ensemble_model_config['datasets'].remove('val')
    evaluate_model(ensemble_model_config, evaluate_model_fp, config['paths']['config_dir'], logger=None)
    res_eval = np.load(config['paths']['config_dir'] / 'res_eval.npy', allow_pickle=True).item()

    for metric_name, metric_val in res_eval.items():

        if isinstance(metric_val, list):
            res_eval[metric_name] = metric_val[-1]
        else:
            res_eval[metric_name] = metric_val

    # delete trained models and ensemble
    delete_model_files(config['paths']['config_models_dir'])

    return res_eval


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

        try:
            res_ensemble = evaluate_config(
                self.worker_id_custom,
                config_id,
                run_config,
                run_config['verbose'],
            )

            # plots
            if run_config['draw_plots']:
                # get results for all trained models
                res_lst_models = {model_i_dir.name: np.load(model_i_dir / 'res_train.npy', allow_pickle=True).item()
                                  for model_i_dir in run_config['paths']['config_models_dir'].iterdir()
                                  if model_i_dir.is_dir() and model_i_dir.name.startswith('model_')}

                res_models = {}
                for metric_name in res_models['model_0']:
                    res_models[metric_name] = [res_model_i[metric_name]
                                               for model_i, res_model_i in res_lst_models.items()]

                plot_fp = (run_config['paths']['config_dir'] /
                           f'config{config_id}_budget{budget}epochs_loss_metric_curves.png')
                self.draw_plots_ensemble(res_ensemble, res_models, config_id, budget, plot_fp)

            # report HPO loss and additional metrics and loss
            hpo_loss_val = res_ensemble[f'val_{run_config["hpo_loss"]}']
            # add metrics and loss
            info_dict = {metric: res_ensemble[metric] for metric in res_ensemble if
                         isinstance(res_ensemble[metric], float)}

            res_hpo = {'loss': 1 - hpo_loss_val,  # HPO loss to be minimized
                       'info': info_dict
                       }

        except Exception as error:  # return infinite HPO loss if there is any error during configuration evaluation
            # with open(run_config['paths']['config_dir'] / 'run_config_exception.txt', "w") as excl_file:
            #     traceback.print_exception(error, file=excl_file)
            res_hpo = {'loss': np.inf, 'info': str(error)}

        # sys.stdout.flush()

        # delete single and ensemble models used to evaluate this configuration
        delete_model_files(run_config['paths']['config_models_dir'])

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
        epochs = np.arange(1, int(budget) + 1)

        # plot loss and optimization metric as function of the epochs
        f, ax = plt.subplots(1, 2)
        for training_loss_i in res_models['loss']:
            ax[0].plot(epochs, training_loss_i, color='b', alpha=alpha)
        ax[0].scatter(epochs[-1], res_ensemble['train_loss'], label='Training', color='b')
        # for validation_loss_i in res_models['val_loss']['all scores']:
        #     ax[0].plot(epochs, validation_loss_i, color='r', alpha=alpha)
        ax[0].scatter(epochs[-1], res_ensemble['val_loss'], label='Validation', color='r')
        # for test_loss_i in res_models['test_loss']['all scores']:
        #     ax[0].scatter(epochs[-1], test_loss_i, c='k', alpha=alpha)
        ax[0].scatter(epochs[-1], res_ensemble['test_loss'], c='k', label='Test')
        ax[0].set_xlim([0, epochs[-1] + 1])
        # ax[0].set_ylim(bottom=0)
        ax[0].set_xlabel('Epochs')
        ax[0].set_ylabel('Loss')
        ax[0].set_title(f'Val/Test {res_ensemble["val_loss"]:.4f}/{res_ensemble["test_loss"]:.4f}')
        ax[0].legend(loc="upper right")
        ax[0].grid(True)

        for training_hpoloss_i in res_models[self.hpo_loss]:
            ax[1].plot(epochs, training_hpoloss_i, color='b', alpha=alpha)
        ax[1].scatter(epochs[-1], res_ensemble[f'train_{self.hpo_loss}'], label='Training', color='b')
        # for validation_hpoloss_i in res_models[f'val_{self.hpo_loss}']['all scores']:
        #     ax[1].plot(epochs, validation_hpoloss_i, color='r', alpha=alpha)
        ax[1].scatter(epochs[-1], res_ensemble[f'val_{self.hpo_loss}'], label='Validation', color='r')
        # for test_hpoloss_i in res_models[f'test_{self.hpo_loss}']['all scores']:
        #     ax[1].scatter(epochs[-1], test_hpoloss_i, c='k', alpha=alpha)
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
