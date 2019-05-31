import numpy as np
import statsmodels.api as sm
import matplotlib; matplotlib.use('agg')
import matplotlib.pyplot as plt
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH


def print_BOHB_runs(num_iterations, eta, bmin, bmax):
    """ Outputs estimate of runs per budget per Successive Halving iteration. Used to get a quick understanding of
    how the BOHB study will run for a given number of SH iterations, eta and minimum and maximum budgets.

    :param num_iterations: int, number of Successing Halving iterations
    :param eta: int, equal or greater than 2
    :param bmin: float, minimum budget
    :param bmax: float, maximum budget
    :return:
        nruns: int, total number of runs for the BOHB study
        total_budget: int, total budget for the BOHB study
    """

    def compute_nSH(budgets, iteSH, s, smax, eta):
        """ Computes number of runs per budget for a particular iteration of Successive Halving.

        :param budgets: array, budgets
        :param iteSH: int, iteration number of a run of Successive Halving
        :param s:
        :param smax: int, maximum number of inner iterations
        :param eta: int, equal or greater than 2
        :return:
            num_runsSH: int, number of runs for the Successive Halving iteration
            budgetSH: int, total budget used for the Successive Halving iteration
        """

        runs_per_budget = [np.floor(np.floor((smax + 1) / (s + 1) * eta ** s) * eta ** (- i))
                           for i in range(0, s + 1, 1)]
        runs_per_budget = np.array([0] * (smax - s) + runs_per_budget)
        print('SH iter.: ', iteSH, '|s: ', s, '|', runs_per_budget)
        # floor instead of ceiling when computing number of configurations for SH as it is in the Hyperband paper
        # (Li et al) and also hpbandster paper
        # num_runsSH = np.sum([np.floor(np.floor((smax + 1) / (s + 1) * eta ** s) * eta ** (- i))
        #                      for i in range(0, s + 1, 1)])
        num_runsSH = np.sum(runs_per_budget)
        budgetSH = np.sum(budgets * runs_per_budget)

        return num_runsSH, budgetSH

    smax = int(np.floor(np.log(bmax / bmin) / np.log(eta)))
    print('smax: {}'.format(smax))

    budgets = np.array([np.floor(eta ** (-i) * bmax) for i in range(smax, -1, -1)])
    print('Budgets: {}'.format(budgets))

    nruns, total_budget = np.sum([compute_nSH(budgets, s, smax - s % (smax + 1), smax, eta) for s in range(0, num_iterations, 1)], axis=0)

    return nruns, total_budget


def get_configspace():
    """ Build the hyperparameter configuration space

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


if __name__ == '__main__':

    num_iterations = 100
    eta = 2
    bmin, bmax = 5, 50
    print('Total number of runs,total budget: {}'.format(print_BOHB_runs(num_iterations, eta, bmin, bmax)))

    train_time = 0.5  # assuming that models on average take 30 minutes to train on 50 epochs
    nensemble = 3
    nnodes = 24
    niter = 400
    runtime = niter * train_time * nensemble / nnodes
    print('Estimate on the number of hours needed, number of configurations tested, total budget: ', runtime, niter, bmax * niter)

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
