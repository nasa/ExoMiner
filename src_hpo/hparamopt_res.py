# 3rd party
import hpbandster.core.result as hpres
import matplotlib.pyplot as plt
import hpbandster.visualization as hpvis
import numpy as np
from matplotlib import ticker, cm
import random

# local
from src_hpo.utils_hpo import print_BOHB_runs

# check number of iterations per Successive Halving and per budget
num_iterations = 200
eta = 2
bmin, bmax = 5, 50
# nruns = print_BOHB_runs(num_iterations, eta, bmin, bmax)

# load results from the BOHB study
res = hpres.logged_results_to_HBS_result('/home/msaragoc/Kepler_planet_finder/configs/study_9')
id2config = res.get_id2config_mapping()
all_runs = res.get_all_runs()

print('Number of configurations tested: %i' % len(id2config))
print('Number of runs: {}'.format(len(all_runs)))

# extract budgets
budgets = []
for run in all_runs:
    if int(run.budget) not in budgets:
        budgets.append(int(run.budget))

# extract runs and configurations per budget
configs_ftime = []
runs_per_budget = {key: [0, []] for key in budgets}
configs_per_budget = {key: [0, []] for key in budgets}
inv_models = 0
for run in all_runs:
    if run.loss is None:  # invalid configs are not taken into account
        inv_models += 1
        continue
    configs_ftime.append([run.config_id, run.time_stamps['finished']])
    runs_per_budget[int(run.budget)][0] += 1
    if run.config_id not in configs_per_budget[int(run.budget)][1]:
        configs_per_budget[int(run.budget)][0] += 1
        configs_per_budget[int(run.budget)][1].append(run.config_id)

print('Runs per budget: ', {k: runs_per_budget[k][0] for k in runs_per_budget})
print('Configs per budget: ', {k: configs_per_budget[k][0] for k in configs_per_budget})

# extract model based and random picks per budget
modelspicks_per_budget = {key: {'model based': [0, []], 'random': [0, []]} for key in budgets}
for b in configs_per_budget:
    for config in configs_per_budget[b][1]:
        if id2config[config]['config_info']['model_based_pick']:
            modelspicks_per_budget[b]['model based'][0] += 1
            modelspicks_per_budget[b]['model based'][1].append(config)
        else:
            modelspicks_per_budget[b]['random'][0] += 1
            modelspicks_per_budget[b]['random'][1].append(config)

print({b: {'model based': modelspicks_per_budget[b]['model based'][0], 'random': modelspicks_per_budget[b]['random'][0]}
       for b in modelspicks_per_budget})

# plot time ordered model based and random picked configurations
configs_ftime = [config_sorted[0] for config_sorted in sorted(configs_ftime, key=lambda x: x[1])]
configs_ftimeu = []
for config in configs_ftime:
    if config not in configs_ftimeu:
        configs_ftimeu.append(config)
idxs_modelbased = [1 if id2config[config]['config_info']['model_based_pick'] else 0 for config in configs_ftimeu]
plt.figure()
plt.plot(idxs_modelbased)
plt.xlabel('Time Ordered Runs')
plt.ylabel('Model based (1)/Random (0) \npicked configs.')

print('Total number of models trained: {}'.format(len(all_runs)))
print('Number of nonviable configurations: {}'. format(inv_models))
print('Number of viable configurations: {}'.format(len(idxs_modelbased)))
print('Number of model based pick configurations: {}'.format(len(np.nonzero(idxs_modelbased)[0])))
print('Number of random picked configurations: {}'.format(len(idxs_modelbased) - len(np.nonzero(idxs_modelbased)[0])))

# get metrics for each configuration
# TODO: remove nan configs
# def printit(x):
#     # print(type(x.info['validation auc']))
#     return np.random.uniform(0, 1)
# a = sorted(all_runs, key=lambda x: printit(x))

# metrics = ['validation recall', 'validation precision']
metrics = ['validation recall', 'validation precision', 'validation auc', 'test recall', 'test precision', 'test auc']
metrics_rank = {}
for run in all_runs:
    if run.info is None:
        continue
    metrics_rank['{}_b{:.0f}'.format(run.config_id, run.budget)] = {}
    for metric in metrics:
        # print('config_id', run.config_id)
        # print('budget', run.budget)
        # print(metric, run.info[metric])
        metrics_rank['{}_b{:.0f}'.format(run.config_id, run.budget)][metric] = run.info[metric]
    metrics_rank['{}_b{:.0f}'.format(run.config_id, run.budget)]['id'] = run.config_id

# plot 2D histograms for two chosen metrics
# metrics_plot = ['validation recall', 'validation precision']
metrics_plot = ['test recall', 'test precision']
# plt.hist([metric_rank[key] for key in metric_rank], range=(0, 1), bins=np.linspace(0, 1, num=40, endpoint=True))
plt.hist2d([metrics_rank[key][metrics_plot[0]] for key in metrics_rank],
           [metrics_rank[key][metrics_plot[1]] for key in metrics_rank],
           range=[(0, 1), (0, 1)],
           bins=[np.linspace(0, 1, num=80, endpoint=True), np.linspace(0, 1, num=80, endpoint=True)])
plt.xticks(np.linspace(0, 1, num=10, endpoint=True))
plt.xlabel(metrics_plot[0])
plt.ylabel(metrics_plot[1])

# rank configurations based on metric
sort_metric = 'validation auc'
sorted_metrickeys = sorted(metrics_rank.items(), key=lambda x: x[1][sort_metric], reverse=True)
# for run in sorted_metrickeys:
#     print(run)

# select top configurations based on minimum value for the ranking metric
min_val = 0.982
vals = []
top_configs = {}
for run in sorted_metrickeys:
    if run[1][sort_metric] < min_val:
        break

    vals.append(run[1][sort_metric])

    if run[0].split('_')[0] not in top_configs.keys():
        top_configs[run[1]['id']] = id2config[run[1]['id']]['config']  # add config params
        top_configs[run[1]['id']]['metric'] = run[1][sort_metric]  # add metric value
    elif run[1][sort_metric] > top_configs[run[1]['id']]['metric']:  # update metric value with the best run for that config
        top_configs[run[1]['id']]['metric'] = run[1][sort_metric]

print('Number of top configs {}'.format(len(top_configs)))

plt.figure()
_, bins, _ = plt.hist(vals, bins='auto')
plt.xlabel('{}'.format(sort_metric))
plt.ylabel('Counts')
plt.title('Histogram top configs ({:.2f})'.format(min_val))

#%% Configuration Visualization

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

data = []
metric_vals = []
for hparam in hparams:
    data_hparam = []
    for config in top_configs:
        data_hparam.append(top_configs[config][hparam])
        if hparam == hparams[0]:
            metric_vals.append(top_configs[config]['metric'])
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

########
data, metric_vals = np.array(data), np.array(metric_vals)

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

plt.suptitle("Parallel Coordinates\nMetric:{}".format(sort_metric))
if len(hparams) - 1 == 1:
    plt.subplots_adjust(top=0.914, bottom=0.068, left=0.042, right=0.703, hspace=0.195, wspace=0.0)

#%%

# # plot learning curves for each configuration as function of the different budgets
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

# TODO: work on this
# returns the best configuration over time
best_configtime = res.get_incumbent_trajectory(all_budgets=True, bigger_is_better=True, non_decreasing_budget=True)

#%% Study analysis plots

# extract best configuration
inc_id = res.get_incumbent_id()
inc_config = id2config[inc_id]['config']
print('Best config:', inc_id, inc_config)

inc_runs = res.get_runs_by_id(inc_id)
inc_run = inc_runs[-1]

# We have access to all information: the config, the loss observed during
# optimization, and all the additional information
inc_loss = inc_run.loss
inc_test_auc = inc_run.info['test auc']

print('It achieved optimization metric of %f (validation) and %f (test).' % (1 - inc_loss, inc_test_auc))

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
f, _ = hpvis.performance_histogram_model_vs_random(all_runs, id2config)
f.set_size_inches(10, 8)
