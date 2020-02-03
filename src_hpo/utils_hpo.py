"""
Utility functions for the hyperparameter optimizer script.
"""

# 3rd party
import os
# import tensorflow as tf
import numpy as np
import pickle
# import statsmodels.api as sm
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import hpbandster.core.result as hpres
import hpbandster.visualization as hpvis
import json
from hpbandster.core.base_iteration import Datum


def estimate_BOHB_runs(num_iterations, eta, bmin, bmax, nmodels=1, verbose=True):
    """ Outputs estimate of runs per budget per Successive Halving iteration. Used to get a quick understanding of
    how the BOHB study will run for a given number of SH iterations, eta and minimum and maximum budgets.

    :param num_iterations: int, number of Successing Halving iterations
    :param eta: int, equal or greater than 2
    :param bmin: float, minimum budget
    :param bmax: float, maximum budget
    :param nmodels: int, number of models trained in each run (performance averaging)
    :return:
        nruns: int, total number of runs for the BOHB study
        total_budget: int, total budget for the BOHB study
    """

    def compute_nSH(budgets, iteSH, s, smax, eta, nmodels=1, verbose=True):
        """ Computes number of runs per budget for a particular iteration of Successive Halving.

        :param budgets: array, budgets
        :param iteSH: int, iteration number of a run of Successive Halving
        :param s:
        :param smax: int, maximum number of inner iterations
        :param eta: int, equal or greater than 2
        :param nmodels: int, number of models trained in each run (performance averaging)
        :return:
            num_runsSH: int, number of runs for the Successive Halving iteration
            budgetSH: int, total budget used for the Successive Halving iteration
        """

        runs_per_budget = [np.floor(np.floor((smax + 1) / (s + 1) * eta ** s) * eta ** (- i))
                           for i in range(0, s + 1, 1)]
        runs_per_budget = np.array([0] * (smax - s) + runs_per_budget)
        if verbose:
            print('SH iter.: ', iteSH, '|s: ', s, '|', runs_per_budget)
        # floor instead of ceiling when computing number of configurations for SH as it is in the Hyperband paper
        # (Li et al) and also hpbandster paper
        # num_runsSH = np.sum([np.floor(np.floor((smax + 1) / (s + 1) * eta ** s) * eta ** (- i))
        #                      for i in range(0, s + 1, 1)])
        num_runsSH = np.sum(runs_per_budget)
        budgetSH = np.sum(nmodels * budgets * runs_per_budget)

        return num_runsSH, budgetSH

    smax = int(np.floor(np.log(bmax / bmin) / np.log(eta)))
    print('smax: {}'.format(smax))

    budgets = np.array([np.floor(eta ** (-i) * bmax) for i in range(smax, -1, -1)])
    print('Budgets: {}'.format(budgets))

    nruns, total_budget = np.sum([compute_nSH(budgets, s, smax - s % (smax + 1), smax, eta, nmodels=nmodels,
                                              verbose=True)
                                  for s in range(0, num_iterations, 1)], axis=0)

    return nruns, total_budget


def get_configspace():
    """ Build a hyperparameter configuration space

    :return: ConfigurationsSpace-Object
    """
    config_space = CS.ConfigurationSpace()

    lr = CSH.UniformFloatHyperparameter('lr', lower=1e-6, upper=1e-1, default_value='1e-2', log=True)

    optimizer = CSH.CategoricalHyperparameter('optimizer', ['Adam', 'SGD'])
    # lr_scheduler = CSH.CategoricalHyperparameter('lr_scheduler', ['constant', 'inv_exp', 'piecew_inv_exp'])
    sgd_momentum = CSH.UniformFloatHyperparameter('sgd_momentum', lower=0.001, upper=0.99, default_value=0.9,
                                                  log=True)
    # batch_size = CSH.UniformIntegerHyperparameter('batch_size', lower=4, upper=9, default_value=6)
    batch_size = CSH.CategoricalHyperparameter('batch_size', [4, 8, 16, 32, 64, 128, 256], default_value=32)
    # dropout_rate = CSH.UniformFloatHyperparameter('dropout_rate', lower=0.001, upper=0.7, default_value=0.2,
    #                                               log=True)
    dropout_rate = CSH.UniformFloatHyperparameter('dropout_rate', lower=0.001, upper=0.7, log=True)
    # decay_rate = CSH.UniformFloatHyperparameter('decay_rate', lower=1e-4, upper=1e-1, default_value=1e-2, log=True)

    # config_space.add_hyperparameters([lr, optimizer, sgd_momentum, batch_size, dropout_rate, decay_rate])
    config_space.add_hyperparameters([lr, optimizer, sgd_momentum, batch_size, dropout_rate])

    num_glob_conv_blocks = CSH.UniformIntegerHyperparameter('num_glob_conv_blocks', lower=2, upper=5)
    num_fc_layers = CSH.UniformIntegerHyperparameter('num_fc_layers', lower=0, upper=4)
    conv_ls_per_block = CSH.UniformIntegerHyperparameter('conv_ls_per_block', lower=1, upper=3, default_value=1)

    # init_fc_neurons = CSH.UniformIntegerHyperparameter('init_fc_neurons', lower=64, upper=512, default_value=256)
    init_fc_neurons = CSH.CategoricalHyperparameter('init_fc_neurons', [32, 64, 128, 256, 512])
    init_conv_filters = CSH.UniformIntegerHyperparameter('init_conv_filters', lower=2, upper=6, default_value=4)

    kernel_size = CSH.UniformIntegerHyperparameter('kernel_size', lower=1, upper=6)
    kernel_stride = CSH.UniformIntegerHyperparameter('kernel_stride', lower=1, upper=2, default_value=1)

    pool_size_glob = CSH.UniformIntegerHyperparameter('pool_size_glob', lower=2, upper=8)
    pool_stride = CSH.UniformIntegerHyperparameter('pool_stride', lower=1, upper=2, default_value=1)

    pool_size_loc = CSH.UniformIntegerHyperparameter('pool_size_loc', lower=2, upper=8)
    num_loc_conv_blocks = CSH.UniformIntegerHyperparameter('num_loc_conv_blocks', lower=1, upper=3)

    config_space.add_hyperparameters([num_glob_conv_blocks,
                                      num_fc_layers,
                                      conv_ls_per_block, kernel_size, kernel_stride,
                                      pool_size_glob, pool_stride,
                                      pool_size_loc, num_loc_conv_blocks,
                                      init_fc_neurons,
                                      init_conv_filters])

    cond = CS.EqualsCondition(sgd_momentum, optimizer, 'SGD')
    config_space.add_condition(cond)

    return config_space


def check_run_id(run_id, shared_directory, worker=False):
    """ Check if there is a previous study with that id and generates a new id.

    :param run_id: str, id for the study
    :param shared_directory:
    :param worker: bool, set to True when using multiple workers
    :return:
        run_id: str, new id for the study
    """

    def _gen_run_id(run_id):
        if run_id[-1].isnumeric():
            return run_id[:-1] + str(int(run_id[-1]) + 1)
        else:
            return run_id + str(1)

    while os.path.isfile(os.path.join(shared_directory, 'configs_%s.json' % run_id)):
        run_id = _gen_run_id(run_id)

    if worker:
        if run_id[-1] == '1':
            run_id = run_id[:-1]
        elif run_id[-1].isnumeric():
            run_id = run_id[:-1] + str(int(run_id[-1]) - 1)

    return run_id


def analyze_results(result, args, shared_dir, run_id):
    """ Save results from the HPO study in a pickle file and plot study analysis plots.

    :param result: HPO res object
    :param args:
    :param shared_dir: str, shared directory by the workers (result directory)
    :param run_id: str, study name id
    :return:
    """

    # save results in a pickle file
    with open(os.path.join(shared_dir, 'results_%s.pkl' % run_id), 'wb') as fh:
        pickle.dump(result, fh)

    # get the 'dict' that translates config ids to the actual configurations
    id2conf = result.get_id2config_mapping()

    # get he incumbent (best configuration)
    inc_id = result.get_incumbent_id()
    try:
        inc_config = id2conf[inc_id]['config']
        print('Best found configuration:', inc_config)
    except KeyError:
        print('No best found configuration!')

    all_runs = result.get_all_runs()

    print('A total of %i unique configurations were sampled.' % len(id2conf.keys()))
    print('A total of %i runs were executed.' % len(all_runs))
    print('Total budget corresponds to %.1f full function evaluations.'
          % (sum([r.budget for r in all_runs]) / args.max_budget))
    print('Total budget corresponds to %.1f full function evaluations.'
          % (sum([r.budget for r in all_runs]) / args.max_budget))
    print('The run took  %.1f seconds to complete.'
          % (all_runs[-1].time_stamps['finished'] - all_runs[0].time_stamps['started']))

    # let's grab the run on the highest budget
    inc_runs = result.get_runs_by_id(inc_id)
    inc_run = inc_runs[-1]

    # We have access to all information: the config, the loss observed during
    # optimization, and all the additional information
    inc_loss = inc_run.loss

    print('It achieved an optmization loss of {} in the validation set.'.format(1 - inc_loss))

    figs = {}

    # Observed losses grouped by budget
    figs['loss'], _ = hpvis.losses_over_time(all_runs)

    # Number of concurrent runs
    figs['concurrent_n'], _ = hpvis.concurrent_runs_over_time(all_runs)

    # Number of finished runs
    figs['finished_n'], _ = hpvis.finished_runs_over_time(all_runs)

    # Spearman rank correlation coefficients of the losses between different budgets
    figs['Spearman'], _ = hpvis.correlation_across_budgets(result)

    # For model based optimizers, one might wonder how much the model actually helped.
    # The next plot compares the performance of configs picked by the model vs. random ones
    figs['model_vs_rand'], _ = hpvis.performance_histogram_model_vs_random(all_runs, id2conf)

    for figname, fig in figs.items():
        fig.set_size_inches(10, 8)
        fig.savefig(os.path.join(shared_dir, figname + '.png'), bbox_inches='tight')

    # if 'nobackup' not in shared_dir:
    #     plt.show()
    #
    #     def realtime_learning_curves(runs):
    #         """
    #         example how to extract a different kind of learning curve.
    #
    #         The x values are now the time the runs finished, not the budget anymore.
    #         We no longer plot the validation loss on the y axis, but now the test accuracy.
    #
    #         This is just to show how to get different information into the interactive plot.
    #
    #         """
    #         sr = sorted(runs, key=lambda r: r.budget)
    #         lc = list(filter(lambda t: not t[1] is None,
    #                          [(r.time_stamps['finished'], r.info['validation accuracy']) for r in sr]))
    #         return [lc, ]
    #
    #     try:
    #         lcs = result.get_learning_curves(lc_extractor=realtime_learning_curves)
    #         hpvis.interactive_HBS_plot(lcs, tool_tip_strings=hpvis.default_tool_tips(result, lcs))
    #     except TypeError as e:
    #         print('\nGot TypeError: ', e)


class json_result_logger(hpres.json_result_logger):
    def __init__(self, directory, run_id, overwrite=False):
        """ This implementation saves the config and results json files appending also the study name id. The  function
        to load the results assumes that the files are named config.json and results.json.

        convenience logger for 'semi-live-results'

        Logger that writes job results into two files (configs.json and results.json).
        Both files contain proper json objects in each line.

        This version opens and closes the files for each result.
        This might be very slow if individual runs are fast and the
        filesystem is rather slow (e.g. a NFS).

        Parameters
        ----------

        directory: string
            the directory where the two files 'configs.json' and
            'results.json' are stored
        overwrite: bool
            In case the files already exist, this flag controls the
            behavior:

                * True:   The existing files will be overwritten. Potential risk of deleting previous results
                * False:  A FileExistsError is raised and the files are not modified.
        """
        os.makedirs(directory, exist_ok=True)

        self.config_fn = os.path.join(directory, 'configs_%s.json' % run_id)
        self.results_fn = os.path.join(directory, 'results_%s.json' % run_id)

        try:
            with open(self.config_fn, 'x') as fh:
                pass
        except FileExistsError:
            if overwrite:
                with open(self.config_fn, 'w') as fh:
                    pass
            else:
                raise FileExistsError('The file %s already exists.' % self.config_fn)
        except:
            raise

        try:
            with open(self.results_fn, 'x') as fh:
                pass
        except FileExistsError:
            if overwrite:
                with open(self.results_fn, 'w') as fh:
                    pass
            else:
                raise FileExistsError('The file %s already exists.' % self.config_fn)

        except:
            raise

        self.config_ids = set()


def logged_results_to_HBS_result(directory, run_id):
    """
    function to import logged 'live-results' and return a HB_result object

    You can load live run results with this function and the returned
    HB_result object gives you access to the results the same way
    a finished run would.

    Parameters
    ----------
    directory: str
        the directory containing the results.json and config.json files
    run_id: str
        the id of the study. If the study id is "study_example", then the json files are named
        "config_study_example.json" and "results_study_example.json"

    Returns
    -------
    hpbandster.core.result.Result: :object:

    TODO: the arguments are too convoluted...

    """
    data = {}
    time_ref = float('inf')
    budget_set = set()

    with open(os.path.join(directory, 'configs{}.json'.format(run_id))) as fh:
        for line in fh:

            line = json.loads(line)

            if len(line) == 3:
                config_id, config, config_info = line
            if len(line) == 2:
                config_id, config, = line
                config_info = 'N/A'

            data[tuple(config_id)] = Datum(config=config, config_info=config_info)

    with open(os.path.join(directory, 'results{}.json'.format(run_id))) as fh:
        for line in fh:
            config_id, budget, time_stamps, result, exception = json.loads(line)

            id = tuple(config_id)

            data[id].time_stamps[budget] = time_stamps
            data[id].results[budget] = result
            data[id].exceptions[budget] = exception

            budget_set.add(budget)
            time_ref = min(time_ref, time_stamps['submitted'])

    # infer the hyperband configuration from the data
    budget_list = sorted(list(budget_set))

    HB_config = {
        'eta': None if len(budget_list) < 2 else budget_list[1] / budget_list[0],
        'min_budget': min(budget_set),
        'max_budget': max(budget_set),
        'budgets': budget_list,
        'max_SH_iter': len(budget_set),
        'time_ref': time_ref
    }
    return hpres.Result([data], HB_config)


if __name__ == '__main__':

    num_iterations = 100
    eta = 2
    bmin, bmax = 5, 50
    print('Total number of runs,total budget: {}'.format(estimate_BOHB_runs(num_iterations, eta, bmin, bmax)))

    train_time = 0.5  # assuming that models on average take 30 minutes to train on 50 epochs
    nensemble = 3
    nnodes = 24
    niter = 400
    runtime = niter * train_time * nensemble / nnodes
    print('Estimate on the number of hours needed, number of configurations tested, total budget: ', runtime, niter,
          bmax * niter)

#
# # load configspace
# sample configurations
# evaluate them using l and g ratio
# choose 3 hyperparameters and plot the ratio

# bdgt = 8.0
#
# # load KDE parameters for a given budget
# kde_models_bdgt_params = np.load('').item()[bdgt]
# # generate the bad and good distributions
# kde_est_bn = sm.nonparametric.KDEMultivariate(data=kde_models_bdgt_params['bad'][0],
#                                               var_type=kde_models_bdgt_params['bad'][1],
#                                               bw=kde_models_bdgt_params['bad'][2])
# kde_est_gn = sm.nonparametric.KDEMultivariate(data=kde_models_bdgt_params['good'][0],
#                                               var_type=kde_models_bdgt_params['good'][1],
#                                               bw=kde_models_bdgt_params['good'][2])
#
# # get config space
# configspace = get_configspace()
# # sample configurations and compute ratio for each one
# num_samples =
# sample_configs = []
# bg_ratio = []
# for i in range(num_samples):
#     sample_configs.append(configspace.sample_configuration().get_array())
#     # print(kde_est_gn.pdf(sample_config), kde_est_bn.pdf(sample_config))
#     bg_ratio.append(max(1e-32, kde_est_bn.pdf(sample_configs[-1]))/max(kde_est_gn.pdf(sample_configs[-1]), 1e-32))
#
# sample_configs = np.array(sample_configs)
#
# # choose two hyperparameters to analyze in the 3D plot
# hparams = []
# hparams_idxs = [kde_models_bdgt_params['hyperparameters'].index(hparam) for hparam in hparams]
#
# # plot
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot(sample_configs[:, hparams_idxs[0]], sample_configs[:, hparams_idxs[1]], bg_ratio)
# ax.set_xlabel(hparams[0])
# ax.set_ylabel(hparams[1])
