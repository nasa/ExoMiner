"""
Evaluate a hyperparameter optimization study using BOHB, BO and RS implementation by Falkner et al.
"""

# 3rd party
# import hpbandster.core.result as hpres
import matplotlib.pyplot as plt
import hpbandster.visualization as hpvis
import numpy as np
from matplotlib import ticker, cm
import glob
# import random

# local
from src_hpo.utils_hpo import estimate_BOHB_runs, logged_results_to_HBS_result  # , json_result_logger
import paths

#%% check number of iterations per Successive Halving and per budget

num_iterations = 300
eta = 2
bmin, bmax = 5, 120
nmodels = 3
nruns, total_budget = estimate_BOHB_runs(num_iterations, eta, bmin, bmax, nmodels=nmodels)
print('Number of runs: {}\nTotal budget: {}'.format(nruns, total_budget))

#%% load results from a HPO study

study = 'bohb_dr25tcert_spline_gapped_g-lflux_selfnormalized'
# set to True if the optimizer is model based
model_based_optimizer = True
# set to True if the study trains multiple models for each configuration evaluated
ensemble_study = True
nmodels = 3
# set which metric to be used when ranking configurations evaluated
rankmetric = 'validation pr auc'
# set two performance metrics to plot in a 2D histogram
metrics_plot = ['test recall', 'test precision']
# minimum value of top configurations (used also for Parallel Coordinates Visualization)
min_val = 0.90

# res = hpres.logged_results_to_HBS_result(paths.path_hpoconfigs + study)
# res = logged_results_to_HBS_result(paths.path_hpoconfigs + study,
#                                    '_{}'.format(study)
#                                    )
# paths.path_hpoconfigs = '/data5/tess_project/Nikash_Walia/Kepler_planet_finder/res/Gapped_Splined/hpo_confs/'
# paths.path_hpoconfigs = '/data5/tess_project/Nikash_Walia/Kepler_planet_finder/res/Gapped_Splined_OddEven_Centroid/hpo_confs/'
res = logged_results_to_HBS_result(paths.path_hpoconfigs + study, '_{}'.format(study))

id2config = res.get_id2config_mapping()
all_runs = res.get_all_runs()
unique_configs = []
for run in all_runs:
    if run.config_id not in unique_configs:
        unique_configs.append(run.config_id)

print('Number of configurations submitted: {}'.format(len(id2config)))
print('Number of configurations evaluated (valid and nonviable): %i' % len(unique_configs))
total_runs = len(all_runs)
print('Total number of runs: {} (viable, nonviable and possibly non-evaluated)'.format(total_runs))

# remove invalid configs
all_runs = [run for run in all_runs if run.info is not None]
print('Number of valid/invalid runs: {}|{}'.format(len(all_runs), total_runs - len(all_runs)))

# extract budgets
budgets = []
for run in all_runs:
    if int(run.budget) not in budgets:
        budgets.append(int(run.budget))
print('Budgets: ', budgets)

total_budget_used = np.sum([int(run.budget) * nmodels for run in all_runs])
print('Total budget used: {}'.format(total_budget_used))

# extract runs and configurations per budget
configs_ftime = []
runs_per_budget = {key: [0, []] for key in budgets}
configs_per_budget = {key: [0, []] for key in budgets}
inv_models = 0
for run in all_runs:
    # if run.loss is None:  # invalid configs are not taken into account
    #     inv_models += 1
    #     continue
    configs_ftime.append([run.config_id, run.time_stamps['finished']])
    runs_per_budget[int(run.budget)][0] += 1
    if run.config_id not in configs_per_budget[int(run.budget)][1]:
        configs_per_budget[int(run.budget)][0] += 1
        configs_per_budget[int(run.budget)][1].append(run.config_id)

print('Runs per budget (viable): ', {k: runs_per_budget[k][0] for k in runs_per_budget})
print('Configs per budget (viable): ', {k: configs_per_budget[k][0] for k in configs_per_budget})

# extract model based and random picks per budget
modelspicks_per_budget = {key: {'model based': [0, []], 'random': [0, []]} for key in budgets}
for b in configs_per_budget:
    for config in configs_per_budget[b][1]:
        if model_based_optimizer:
            if id2config[config]['config_info']['model_based_pick']:
                modelspicks_per_budget[b]['model based'][0] += 1
                modelspicks_per_budget[b]['model based'][1].append(config)
            else:
                modelspicks_per_budget[b]['random'][0] += 1
                modelspicks_per_budget[b]['random'][1].append(config)
        else:
            modelspicks_per_budget[b]['random'][0] += 1
            modelspicks_per_budget[b]['random'][1].append(config)

print('Model/random based pick configurations: ',
      {b: {'model based': modelspicks_per_budget[b]['model based'][0], 'random': modelspicks_per_budget[b]['random'][0]}
       for b in modelspicks_per_budget})

# plot time ordered model based and random picked configurations
configs_ftime = [config_sorted[0] for config_sorted in sorted(configs_ftime, key=lambda x: x[1])]
configs_ftimeu = []
for config in configs_ftime:
    if config not in configs_ftimeu:
        configs_ftimeu.append(config)
if model_based_optimizer:
    idxs_modelbased = [1 if id2config[config]['config_info']['model_based_pick'] else 0 for config in configs_ftimeu]
else:
    idxs_modelbased = [0 for config in configs_ftimeu]
plt.figure()
plt.plot(idxs_modelbased)
plt.xlabel('Time Ordered Runs')
plt.ylabel('Model based (1)/Random (0) \npicked configs.')

nconfigs_valid = len(idxs_modelbased)
nconfigs_modelbased = len(np.nonzero(idxs_modelbased)[0])
print('Number of nonviable configurations: {}'. format(len(unique_configs) - nconfigs_valid))
print('Number of viable configurations: {}'.format(nconfigs_valid))
print('Number of model based pick configurations: {}'.format(nconfigs_modelbased))
print('Number of random picked configurations: {}'.format(nconfigs_valid - nconfigs_modelbased))

# plot 2D histograms for two chosen metrics
if ensemble_study:
    plt.hist2d([run.info[metrics_plot[0]][0] for run in all_runs],
               [run.info[metrics_plot[1]][0] for run in all_runs],
               range=[(0, 1), (0, 1)],
               bins=[np.linspace(0, 1, num=80, endpoint=True), np.linspace(0, 1, num=80, endpoint=True)])
else:
    plt.hist2d([run.info[metrics_plot[0]] for run in all_runs],
               [run.info[metrics_plot[1]] for run in all_runs],
               range=[(0, 1), (0, 1)],
               bins=[np.linspace(0, 1, num=80, endpoint=True), np.linspace(0, 1, num=80, endpoint=True)])
plt.xticks(np.linspace(0, 1, num=10, endpoint=True))
plt.xlabel(metrics_plot[0])
plt.ylabel(metrics_plot[1])
plt.title('2D histogram of configs performance metrics')

# rank configurations based on metric
if ensemble_study:
    ranked_allruns = sorted(all_runs, key=lambda x: x.info[rankmetric][0], reverse=True)
else:
    ranked_allruns = sorted(all_runs, key=lambda x: x.info[rankmetric], reverse=True)
# for run in ranked_allruns:
#     print(run)

# select top configurations based on minimum value for the ranking metric
top_configs_aux = []
top_configs = []
for run in ranked_allruns:
    if ensemble_study:
        if run.info[rankmetric][0] < min_val:
            break
    else:
        if run.info[rankmetric] < min_val:
            break
    if run.config_id not in top_configs_aux:
        top_configs.append(run)
        top_configs_aux.append(run.config_id)

print('Number of top configs {}'.format(len(top_configs)))

# Plot histogram of ranked metric for top configurations
bins = 'auto'
plt.figure()
if ensemble_study:
    _, bins, _ = plt.hist([run.info[rankmetric][0] for run in top_configs], bins=bins)
else:
    _, bins, _ = plt.hist([run.info[rankmetric] for run in top_configs], bins=bins)
plt.xlabel('{}'.format(rankmetric))
plt.ylabel('Counts')
plt.title('Histogram top configs ({:.2f})'.format(min_val))

#%% Configuration Visualization

# pick parameters to be analyzed
# discrete parameters
hparams_d = ['conv_ls_per_block', 'init_conv_filters', 'kernel_size', 'kernel_stride', 'num_fc_layers',
           'num_glob_conv_blocks', 'num_loc_conv_blocks', 'pool_size_glob', 'pool_size_loc', 'init_fc_neurons',
           'batch_size', 'pool_stride']
# continuous parameters
# hparams_c = ['lr', 'dropout_rate']
hparams_c = ['dropout_rate']

# hparams = ['conv_ls_per_block', 'init_conv_filters', 'num_fc_layers', 'init_fc_neurons', 'batch_size']
# hparams = ['kernel_size', 'kernel_stride', 'pool_stride', 'pool_size_glob', 'pool_size_loc', 'dropout_rate']
# hparams = ['kernel_size', 'pool_size_glob', 'pool_size_loc', 'dropout_rate', 'num_glob_conv_blocks', 'num_loc_conv_blocks', 'init_fc_neurons', 'conv_ls_per_block', 'init_conv_filters']
hparams = ['num_fc_layers', 'dropout_rate', 'conv_ls_per_block']

hparams_all = hparams_d + hparams_c

# for hparam in hparams_all:

# hparams = ['metric'] + hparams_all
hparams = hparams_all
# hparams = ['metric'] + hparams + ['metric']
#     hparams = ['metric'] + [hparam]

# get parameters' values for each configuration
data = []
metric_vals = []
for hparam in hparams:
    data_hparam = []
    for top_config in top_configs:
        data_hparam.append(id2config[top_config.config_id]['config'][hparam])
        # data_hparam.append(top_configs[config][hparam])
        if hparam == hparams[0]:
            if ensemble_study:
                metric_vals.append(top_config.info[rankmetric][0])
            else:
                metric_vals.append(top_config.info[rankmetric])
    data.append(data_hparam)

# hyperparameters boxplots
# for i, hparam in enumerate(hparams):
#     f, ax = plt.subplots()
#     # ax.set_title('{}'.format(hparam))
#     ax.set_title('Boxplot')
#     # whiskers [Q1 - whis * IQR; Q3 + whis * IQR]
#     ax.boxplot(data[i], showfliers=True, notch=False, labels=[hparam], whis=1.5)
#     ax.set_ylabel('Value')
#     # ax.set_xlabel('')
#     # ax.set_xticklabels([''])

data, metric_vals = np.array(data), np.array(metric_vals)

# Parallel Coordinates Visualization
# based on: http://benalexkeen.com/parallel-coordinates-in-matplotlib/

# lims = np.arange(min_val, 0.985, 0.0005)  # [0.981, 0.9815, 0.982, 0.9825, 0.983, 0.984]
lims = bins
cmap = cm.get_cmap('viridis')
colours = [cmap(el) for el in np.linspace(0, 1, endpoint=True, num=len(lims))]
# colours = ['#2e8ad8', '#cd3785', '#c64c00', '#889a00']
# colours = ["#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)])
#              for i in range(len(lims))]
colour_code = np.zeros(data.shape[1], dtype='int')
for i in range(1, len(lims)):
    if i != len(lims) - 1:
        # print(len(colour_code[(metric_vals >= lims[i]) & (metric_vals < lims[i + 1])]))
        colour_code[(metric_vals >= lims[i]) & (metric_vals < lims[i + 1])] = i
    else:
        # print(len(colour_code[(metric_vals >= lims[i])]))
        colour_code[metric_vals >= lims[i]] = i

# Create (X-1) sublots along x axis
fig, axes = plt.subplots(1, len(hparams) - 1, sharey=False, figsize=(15, 5))
if len(hparams) - 1 == 1:
    axes = [axes]

# Get min, max and range for each column
# Normalize the data for each column
min_max_range = {}
for i, hparam in enumerate(hparams):
    min_max_range[hparam] = [data[i].min(), data[i].max(), np.ptp(data[i])]
    data[i] = np.true_divide(data[i] - data[i].min(), (np.ptp(data[i]), data[i])[np.ptp(data[i]) == 0])

# Plot each row
for i in range(len(hparams) - 1):
    for j in range(data.shape[1]):  # plot each data point
        axes[i].plot(np.arange(len(hparams)), data[:, j], color=colours[colour_code[j]])
    axes[i].set_xlim([i, i + 1])


# Set the tick positions and labels on y axis for each plot
# Tick positions based on normalised data
# Tick labels are based on original data
def set_ticks_for_axis(dim, ax, ticks):
    min_val, max_val, val_range = min_max_range[hparams[dim]]
    step = val_range / float(ticks - 1)
    tick_labels = [round(min_val + step * i, 2) for i in range(ticks)]
    norm_min = data[dim].min()
    norm_range = np.ptp(data[dim])
    norm_step = norm_range / float(ticks - 1)
    ticks = [round(norm_min + norm_step * i, 2) for i in range(ticks)]
    ax.yaxis.set_ticks(ticks)
    ax.set_yticklabels(tick_labels)


for dim, ax in enumerate(axes):
    ax.xaxis.set_major_locator(ticker.FixedLocator([dim]))
    set_ticks_for_axis(dim, ax, ticks=6)
    ax.set_xticklabels([hparams[dim]])

# Move the final axis' ticks to the right-hand side
ax = plt.twinx(axes[-1])
dim = len(axes)
ax.xaxis.set_major_locator(ticker.FixedLocator([len(hparams) - 2, len(hparams) - 1]))
set_ticks_for_axis(dim, ax, ticks=6)
ax.set_xticklabels([hparams[-2], hparams[-1]])

# Remove space between subplots
plt.subplots_adjust(wspace=0)

# Add legend to plot
plt.legend(
    [plt.Line2D((0, 1), (0, 0), color=colours[i]) for i in range(len(lims))],
    ['[{:.4f},{:.4f}['.format(lims[l], lims[l + 1]) if l < len(lims) - 1 else '[{:.3f},1['.format(lims[l])
     for l in range(len(lims))],
    bbox_to_anchor=(1.2, 1), loc=2, borderaxespad=0.)

plt.suptitle("Parallel Coordinates\nMetric:{}".format(rankmetric))
if len(hparams) - 1 == 1:
    plt.subplots_adjust(top=0.914, bottom=0.068, left=0.042, right=0.703, hspace=0.195, wspace=0.0)

#%% Learning curves and tracking best configuration

# # plot learning curves for each configuration as function of the different budgets - it is a mess
# lc_configs = res.get_learning_curves(config_ids=None)
# fig, ax = plt.subplots()
# for config in lc_configs:
#     bdgt, lc = [], []
#     # print(config)
#     # print(lc_configs)
#     for run in lc_configs[config][0]:
#         bdgt.append(run[0])
#         lc.append(run[1])
#     ax.plot(lc, bdgt)
#     # ax.scatter(lc, bdgt)
# ax.set_ylabel('Budget [number of epochs]')
# ax.set_xlabel('Loss')
# ax.set_ylim(bottom=0)
# ax.set_xlim(left=0)

# returns the best configuration over time
# best_configtime = res.get_incumbent_trajectory(all_budgets=True, bigger_is_better=False, non_decreasing_budget=False)

# returns the best configuration over time/over cumulative budget
nmodels = 3
hpo_loss = 'pr auc'
lim_totalbudget = np.inf
timesorted_allruns = sorted(all_runs, key=lambda x: x.time_stamps['finished'], reverse=False)
if ensemble_study:
    ensmetrics_list = [file for file in glob.glob(paths.path_hpoconfigs + study + '/*.*') if 'ensemblemetrics' in file]
bconfig_loss, cum_budget = np.inf, 0
bmu_hpoloss, bsem_hpoloss = None, None
timestamps, cum_budget_vec, tinc_hpoloss = [], [], []
if ensemble_study:
    tinc_hpolossdev = []
for run_i, run in enumerate(timesorted_allruns):

    if cum_budget + int(run.budget) * nmodels > lim_totalbudget:
        print('break. pass the total budget limit')
        break

    cum_budget += int(run.budget) * nmodels

    if run.loss < bconfig_loss or run_i == len(timesorted_allruns) - 1:

        print('Best config so far: {}'. format(run.config_id))

        # add timestamp and cumulated budget to the arrays
        cum_budget_vec.append(cum_budget)
        timestamps.append(run.time_stamps['finished'])

        if not ensemble_study:
            tinc_hpoloss.append(run.loss)
        else:  # for ensemble studies, use the mean and std

            # search for the numpy file with the values for the given run
            for ensmetrics in ensmetrics_list:
                if str(run.config_id) + 'budget' + str(int(run.budget)) in ensmetrics:
                    censemetrics = ensmetrics
                    break
            else:  # did not find the file for that run
                raise ValueError('No saved metrics matched this run: config {} on budget {}'.format(run.config_id,
                                                                                                    run.budget))

            ensmetrics = np.array(np.load(censemetrics).item()['validation'][hpo_loss]['all scores'])
            mu_hpoloss = 1 - np.median(ensmetrics[:, -1])
            sem_hpoloss = np.std(ensmetrics[:, -1], ddof=1) / np.sqrt(ensmetrics.shape[0])

            if run.loss < bconfig_loss:
                bmu_hpoloss, bsem_hpoloss = mu_hpoloss, sem_hpoloss
                tinc_hpoloss.append(mu_hpoloss)
                tinc_hpolossdev.append(sem_hpoloss)
            else:
                tinc_hpoloss.append(bmu_hpoloss)
                tinc_hpolossdev.append(bsem_hpoloss)

        if run.loss < bconfig_loss:
            bconfig_loss = run.loss

f, ax = plt.subplots()
if not ensemble_study:
    ax.plot(timestamps, tinc_hpoloss)
else:
    ax.errorbar(timestamps, tinc_hpoloss,
                tinc_hpolossdev, label=study, capsize=5)
ax.scatter(timestamps, tinc_hpoloss, c='r')
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_ylim(top=1)
# ax.set_xlim(right=1e5)
ax.set_ylabel('Optimization loss')
ax.set_xlabel('Wall clock time [s]')
# ax.set_title('')
ax.grid(True, which='both')

f, ax = plt.subplots()
if not ensemble_study:
    ax.plot(cum_budget_vec, tinc_hpoloss)
else:
    ax.errorbar(cum_budget_vec, tinc_hpoloss,
                tinc_hpolossdev, label=study, capsize=5)
ax.scatter(cum_budget_vec, tinc_hpoloss, c='r')
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_ylim(top=1)
# ax.set_xlim(right=1e5)
ax.set_ylabel('Optimization loss')
ax.set_xlabel('Cumulative budget [Epochs]')
# ax.set_title('')
ax.grid(True, which='both')


#%% Study analysis plots

# metric used as optimization loss
hpo_loss = 'pr auc'

# extract best configuration
inc_id = res.get_incumbent_id()
inc_config = id2config[inc_id]['config']
print('Best config:', inc_id, inc_config)

inc_runs = res.get_runs_by_id(inc_id)
for run in inc_runs:
    print(run)

# We have access to all information: the config, the loss observed during
# optimization, and all the additional information
inc_run = inc_runs[-1]  # performance in the largest budget
inc_hpoloss = inc_run.loss
inc_val_hpoloss, inc_test_hpoloss = inc_run.info['validation ' + hpo_loss], inc_run.info['test ' + hpo_loss]
print('It achieved optimization loss ({}) of {} (validation) and {} (test).'.format(hpo_loss, inc_val_hpoloss,
                                                                                    inc_test_hpoloss))

# Let's plot the observed losses grouped by budget,
f, _ = hpvis.losses_over_time(all_runs)
f.set_size_inches(10, 6)

# the number of concurrent runs,
f, _ = hpvis.concurrent_runs_over_time(all_runs)
# f.set_size_inches(10, 6)

# and the number of finished runs.
hpvis.finished_runs_over_time(all_runs)

# This one visualizes the spearman rank correlation coefficients of the losses
# between different budgets.
f, _ = hpvis.correlation_across_budgets(res)
f.set_size_inches(10, 8)

# For model based optimizers, one might wonder how much the model actually helped.
# The next plot compares the performance of configs picked by the model vs. random ones
if model_based_optimizer:
    f, _ = hpvis.performance_histogram_model_vs_random(all_runs, id2config)
    f.set_size_inches(10, 8)

#%% Compare different HPO studies

# paths.path_hpoconfigs = ['/data5/tess_project/Nikash_Walia/Kepler_planet_finder/res/Gapped_Splined_OddEven/hpo_confs/bohb_dr25tcert_spline_gapped_oddeven_only',
#                          '/data5/tess_project/Nikash_Walia/Kepler_planet_finder/res/Gapped_Splined_Centroid/hpo_confs/bohb_dr25tcert_spline_gapped_centroid',
#                          '/data5/tess_project/Nikash_Walia/Kepler_planet_finder/res/Gapped_Splined/hpo_confs/bohb_dr25tcert_spline_gapped',
#                          '/data5/tess_project/Nikash_Walia/Kepler_planet_finder/res/Gapped_Splined_OddEven_Centroid/hpo_confs/bohb_dr25tcert_spline_oddeven']
paths.path_hpoconfigs = ['/home/msaragoc/Projects/Kepler-TESS_exoplanet/Kepler_planet_finder/hpo_configs/bohb_dr25tcert_spline_gapped_centroid_oddeven_normpair_ncoe',
                         '/home/msaragoc/Projects/Kepler-TESS_exoplanet/Kepler_planet_finder/hpo_configs/bohb_dr25tcert_spline_gapped_centroid_oddeven',
                         '/home/msaragoc/Projects/Kepler-TESS_exoplanet/Kepler_planet_finder/hpo_configs/bohb_dr25tcert_spline_gapped']

# load results from the BOHB study
# studies = ['study_bo', 'study_rs', 'study_bohb']
studies = ['Gapped+centroid+jointnorm_oddeven', 'Gapped+centroid+oddeven', 'Gapped']
# studies_name = {'study_bo': 'BO', 'study_rs': 'RS', 'study_bohb': 'BOHB'}
studies_name = {study: study for study in studies}
hpo_loss = 'pr auc'
lim_totalbudget = np.inf
nmodels = 3
ensemble_study = True
time_budget_studies = {study: {'hpo_loss': None, 'cum_budget': None, 'wall_clock_time': None} for study in studies}

# for study in studies:
for study_i in range(len(studies)):

    print('Study {}'.format(studies[study_i]))
    # if study == 'study_bohb_dr25_tcert_spline':
    #     res = logged_results_to_HBS_result(paths.path_hpoconfigs + study, '_' + study)
    # else:
    #     res = hpres.logged_results_to_HBS_result(paths.path_hpoconfigs + study)
    # res = logged_results_to_HBS_result(paths.path_hpoconfigs + study, '')
    res = logged_results_to_HBS_result(paths.path_hpoconfigs[study_i], '_' + paths.path_hpoconfigs[study_i].split('/')[-1])

    # extract best configuration
    inc_id = res.get_incumbent_id()
    id2config = res.get_id2config_mapping()
    inc_config = id2config[inc_id]['config']
    print('Best config:', inc_id, inc_config)

    # ensmetrics_list = [file for file in glob.glob(paths.path_hpoconfigs + study + '/*.*') if 'ensemblemetrics' in file]
    ensmetrics_list = [file for file in glob.glob(paths.path_hpoconfigs[study_i] + '/*.*') if 'ensemblemetrics' in file]
    study = studies[study_i]

    all_runs = res.get_all_runs()
    all_runs = [run for run in all_runs if run.info is not None]

    timesorted_allruns = sorted(all_runs, key=lambda x: x.time_stamps['finished'], reverse=False)

    bconfig_loss, cum_budget = np.inf, 0
    bmu_hpoloss, bsem_hpoloss = None, None
    timestamps, cum_budget_vec, tinc_hpoloss = [], [], []
    if ensemble_study:
        tinc_hpolossdev = []
    for run_i, run in enumerate(timesorted_allruns):

        if cum_budget + int(run.budget) * nmodels > lim_totalbudget:
            print('break {}'.format(study))
            break

        cum_budget += int(run.budget) * nmodels

        if run.loss < bconfig_loss or run_i == len(timesorted_allruns) - 1:

            # add timestamp and cumulated budget to the arrays
            timestamps.append(run.time_stamps['finished'])
            cum_budget_vec.append(cum_budget)

            # for ensemble studies, use the mean and std
            if ensemble_study:

                # search for the numpy file with the values for the given run
                for ensmetrics in ensmetrics_list:
                    if str(run.config_id) + 'budget' + str(int(run.budget)) in ensmetrics:
                        censemetrics = ensmetrics
                        break
                else:  # did not find the file for that run
                    raise ValueError('No saved metrics matched this run: config {} on budget {}'.format(run.config_id,
                                                                                                        run.budget))

                ensmetrics = np.array(np.load(censemetrics).item()['validation'][hpo_loss]['all scores'])
                mu_hpoloss = 1 - np.median(ensmetrics[:, -1])  # compute mean loss
                sem_hpoloss = np.std(ensmetrics[:, -1], ddof=1) / np.sqrt(ensmetrics.shape[0])  # compute std

                if run.loss < bconfig_loss:
                    bmu_hpoloss, bsem_hpoloss = mu_hpoloss, sem_hpoloss
                    tinc_hpoloss.append(mu_hpoloss)
                    tinc_hpolossdev.append(sem_hpoloss)
                else:
                    tinc_hpoloss.append(bmu_hpoloss)
                    tinc_hpolossdev.append(bsem_hpoloss)

            else:
                tinc_hpoloss.append(run.loss)

            # update best config loss
            if run.loss < bconfig_loss:
                bconfig_loss = run.loss

    time_budget_studies[study]['hpo_loss'] = tinc_hpoloss
    time_budget_studies[study]['hpo_loss_dev'] = tinc_hpolossdev
    time_budget_studies[study]['cum_budget'] = cum_budget_vec
    time_budget_studies[study]['wall_clock_time'] = timestamps

f, ax = plt.subplots()
for study in time_budget_studies:
    ax.errorbar(time_budget_studies[study]['wall_clock_time'], time_budget_studies[study]['hpo_loss'],
                yerr=time_budget_studies[study]['hpo_loss_dev'], label=studies_name[study], capsize=5)
    ax.scatter(time_budget_studies[study]['wall_clock_time'], time_budget_studies[study]['hpo_loss'], c='r')
    ax.set_yscale('log')
    ax.set_xscale('log')
ax.set_ylim(top=1)
ax.set_xlim(right=1e5)
ax.set_ylabel('Optimization loss')
ax.set_xlabel('Wall clock time [s]')
ax.set_title('')
ax.grid(True, which='both')
ax.legend()

f, ax = plt.subplots()
for study in time_budget_studies:
    ax.errorbar(time_budget_studies[study]['cum_budget'], time_budget_studies[study]['hpo_loss'],
                yerr=time_budget_studies[study]['hpo_loss_dev'], label=studies_name[study], capsize=5)
    ax.scatter(time_budget_studies[study]['cum_budget'], time_budget_studies[study]['hpo_loss'], c='r')
    ax.set_yscale('log')
    ax.set_xscale('log')
ax.set_ylim(top=1)
ax.set_xlim(right=1e5)
ax.set_ylabel('Optimization loss')
ax.set_xlabel('Cumulative budget [Epochs]')
# ax.set_title('')
ax.grid(True, which='both')
ax.legend()
