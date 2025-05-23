"""
Evaluate a hyperparameter optimization study using BOHB, BO or RS implementation by Falkner et al.
"""

# 3rd party
import hpbandster.visualization as hpvis
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# local
from src_hpo.utils_hpo import logged_results_to_HBS_result

#%% Load results from a HPO study

# HPO run directory
hpo_dir = Path('/path/to/hpo')
# set to True if the optimizer is model based
model_based_optimizer = True
# set to True if the study trains multiple models for each configuration evaluated
ensemble_study = True
nmodels = 3  # number of models in the ensemble trained for each configuration
# set which metric to be used when ranking configurations evaluated
rankmetric = 'test_auc_pr'  # 'validation pr auc'
# set two performance metrics to plot in a 2D histogram
metrics_plot = ['test_recall', 'test_precision']  # ['test recall', 'test precision']
# minimum value of top configurations (used also for Parallel Coordinates Visualization)
min_val = 0.90

# load results json
res = logged_results_to_HBS_result(hpo_dir)

#%% Get all runs

id2config = res.get_id2config_mapping()
all_runs = res.get_all_runs()
# get configurations with reported results (even if invalid)
unique_configs = []
for run in all_runs:
    if run.config_id not in unique_configs:
        unique_configs.append(run.config_id)

print('Number of configurations submitted: {}'.format(len(id2config)))
print('Number of configurations evaluated (i.e., at least one result (valid or invalid) was logged): %i' %
      len(unique_configs))
total_runs = len(all_runs)
print('Total number of runs: {}'.format(total_runs))

# remove invalid configs
all_runs = [run for run in all_runs if run.info is not None]
all_runs = [run for run in all_runs if not isinstance(run.info, str)]
print('Number of valid/invalid runs: {}|{}'.format(len(all_runs), total_runs - len(all_runs)))

#%% Extract budgets

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

print('Runs per budget (valid): ', {k: runs_per_budget[k][0] for k in runs_per_budget})
print('Configs per budget (valid): ', {k: configs_per_budget[k][0] for k in configs_per_budget})

#%% Extract model based and random picks per budget

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

#%% Plot time ordered model based and random picked configurations

configs_ftime = [config_sorted[0] for config_sorted in sorted(configs_ftime, key=lambda x: x[1])]
configs_ftimeu = []
for config in configs_ftime:
    if config not in configs_ftimeu:
        configs_ftimeu.append(config)
if model_based_optimizer:
    idxs_modelbased = [1 if id2config[config]['config_info']['model_based_pick'] else 0 for config in configs_ftimeu]
else:
    idxs_modelbased = [0 for config in configs_ftimeu]

f, ax = plt.subplots()
ax.plot(idxs_modelbased)
ax.set_xlabel('Time Ordered Runs')
ax.set_ylabel('Model based (1)/Random (0) \npicked configs.')
f.tight_layout()
f.savefig(hpo_dir / 'time_model-random.png')

nconfigs_valid = len(idxs_modelbased)
nconfigs_modelbased = len(np.nonzero(idxs_modelbased)[0])
print('Number of invalid configurations (i.e., results are errors): {}'. format(len(unique_configs) - nconfigs_valid))
print('Number of valid configurations: {}'.format(nconfigs_valid))
print('Number of model based pick configurations: {}'.format(nconfigs_modelbased))
print('Number of random picked configurations: {}'.format(nconfigs_valid - nconfigs_modelbased))

#%% Plot 2D histograms for two chosen metrics

f, ax = plt.subplots()
if ensemble_study:
    # axh = ax.hist2d([run.info[metrics_plot[0]][0] for run in all_runs],
    #           [run.info[metrics_plot[1]][0] for run in all_runs],
    #           range=[(0, 1), (0, 1)],
    #           bins=[np.linspace(0, 1, num=41, endpoint=True), np.linspace(0, 1, num=41, endpoint=True)],
    #           norm=mcolors.LogNorm(),
    #           cmap='viridis')
    axh = ax.hist2d([run.info[metrics_plot[0]] for run in all_runs],
                    [run.info[metrics_plot[1]] for run in all_runs],
                    range=[(0, 1), (0, 1)],
                    bins=[np.linspace(0, 1, num=41, endpoint=True), np.linspace(0, 1, num=41, endpoint=True)],
                    norm=mcolors.LogNorm(),
                    cmap='viridis')
else:
    axh = ax.hist2d([run.info[metrics_plot[0]] for run in all_runs],
                    [run.info[metrics_plot[1]] for run in all_runs],
                    range=[(0, 1), (0, 1)],
                    bins=[np.linspace(0, 1, num=41, endpoint=True), np.linspace(0, 1, num=41, endpoint=True)])
ax.grid(True)
ax.set_xticks(np.linspace(0, 1, num=11, endpoint=True))
ax.set_yticks(np.linspace(0, 1, num=11, endpoint=True))
ax.set_xlabel(metrics_plot[0])
ax.set_ylabel(metrics_plot[1])
ax.set_title('2D histogram of configs performance metrics')
plt.colorbar(axh[3], label='Counts')
# cbar = f.colorbar(axh, ax=ax, cax=None, format="$%.2f$", orientation='vertical', location='right', fraction=0.05, label='Recall', aspect=8, ticks=np.arange(0, 1.1, 0.1))
# cbar.ax.tick_params(labelsize=10, rotation=0)
f.savefig(hpo_dir / '2dhist_precision-recall.png')

#%% rank configurations based on metric

if ensemble_study:
    # ranked_allruns = sorted(all_runs, key=lambda x: x.info[rankmetric][0], reverse=True)
    ranked_allruns = sorted(all_runs, key=lambda x: x.info[rankmetric], reverse=True)
else:
    ranked_allruns = sorted(all_runs, key=lambda x: x.info[rankmetric], reverse=True)
# for run in ranked_allruns:
#     print(run)

# select top configurations based on minimum value for the ranking metric
top_configs_aux = []
top_configs = []
for run in ranked_allruns:
    if ensemble_study:
        # if run.info[rankmetric][0] < min_val:
        if run.info[rankmetric] < min_val:
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
f, ax = plt.subplots()
if ensemble_study:
    # _, bins, _ = ax.hist([run.info[rankmetric][0] for run in top_configs], bins=bins, edgecolor='black')
    _, bins, _ = ax.hist([run.info[rankmetric] for run in top_configs], bins=bins, edgecolor='black')
else:
    _, bins, _ = ax.hist([run.info[rankmetric] for run in top_configs], bins=bins, edgecolor='black')
ax.set_xlabel('{}'.format(rankmetric))
ax.set_ylabel('Counts')
ax.set_title('Histogram top configs ({:.2f})'.format(min_val))
f.savefig(hpo_dir / 'hist_top{}_{}.png'.format(min_val, rankmetric))

#%% Check model's weights

bins = np.logspace(5, 8, 30)

configs_dirs = [fp for fp in hpo_dir.iterdir() if fp.is_dir() and fp.name.startswith('config')]

configs_model_weights = {}
for config_dir in configs_dirs:
    model_summary_fp = Path(config_dir / 'models' / 'model_0' / 'model_summary.txt')

    if model_summary_fp.exists():
        with open(model_summary_fp, 'r') as model_summary_file:
            for line in model_summary_file.readlines():
                if 'Total params' in line:
                    configs_model_weights[config_dir.name[6:]] = int(line.split(' ')[2])

# plot histogram of models' weights
f, ax = plt.subplots()
ax.hist(configs_model_weights.values(), bins=bins, edgecolor='k')
ax.set_xlabel('Number Model Weights')
ax.set_ylabel('Number Configurations Evaluated')
# ax.set_yscale('log')
ax.grid(axis='y')
ax.set_xscale('log')
f.savefig(hpo_dir / 'hist_num_weights.png')

#%% Learning curves and tracking best configuration

# plot learning curves for each configuration as function of the different budgets - it is a mess
lc_configs = res.get_learning_curves(config_ids=None)
fig, ax = plt.subplots()
for config in lc_configs:
    bdgt, lc = [], []
    # print(config)
    # print(lc_configs)
    for run in lc_configs[config][0]:
        bdgt.append(run[0])
        lc.append(run[1])
    ax.plot(bdgt, lc, linestyle='dashed', zorder=1, alpha=0.3)  # , label=f'{config}')
    ax.scatter(bdgt, lc, s=8, zorder=2)
ax.set_xlabel('Budget [number of training epochs]')
ax.set_ylabel('HPO loss')
# ax.set_ylim(bottom=0)
# ax.set_xlim(left=0)
ax.set_yscale('log')
# ax.legend()
# plt.close()

#%% Returns the best configuration over time

metric = 'auc_pr'
# hpo_loss = 'test_auc_pr'  # 'pr auc'
budget_chosen = 'all'  # 50.0  # 'all
lim_totalbudget = np.inf

best_configtime = res.get_incumbent_trajectory(all_budgets=True, bigger_is_better=False, non_decreasing_budget=False)

timesorted_allruns = sorted(all_runs, key=lambda x: x.time_stamps['finished'], reverse=False)
if ensemble_study:
    ensmetrics_list = [config_dir / f'{config_dir.name}_budget{budget_chosen}_ensemblemetrics.npy'
                       for config_dir in hpo_dir.iterdir()
                       if config_dir.is_dir() and config_dir.name.startswith('config')]
    # ensmetrics_list = [f'{config_dir.name}_budget{budget_chosen}_ensemblemetrics.npy'
    #                    for file in hpo_dir.iterdir()
    #                    if file.is_file() and file.name.startswith('config') and file.suffix == '.npy']
bconfig_loss, cum_budget = np.inf, 0
bmu_hpoloss, bsem_hpoloss = None, None
timestamps, cum_budget_vec, tinc_hpoloss, models_n_weights = [], [], [], []
if ensemble_study:
    tinc_hpoloss_centraltend = []
    tinc_hpoloss_dev = []

for run_i, run in enumerate(timesorted_allruns):

    if cum_budget + int(run.budget) * nmodels > lim_totalbudget:
        print('break. pass the total budget limit')
        break

    cum_budget += int(run.budget) * nmodels

    if budget_chosen != 'all':

        if int(run.budget) != budget_chosen:
            continue

    if run.loss < bconfig_loss:

        if run.loss < bconfig_loss:
            print('Best config so far: {} ({} on budget {})'. format(run.config_id, run.loss, int(run.budget)))

        # add timestamp and cumulated budget to the arrays
        cum_budget_vec.append(cum_budget)
        timestamps.append(run.time_stamps['finished'])

        if not ensemble_study:
            tinc_hpoloss.append(run.loss)
        else:  # for ensemble studies, use the mean and std

            # ensmetrics_fp = Path(hpo_dir / f'config{run.config_id}' /
            #                      f'config{run.config_id}_budget{int(run.budget)}_ensemblemetrics.npy')
            # if not ensmetrics_fp.exists():  # did not find the file for that run
            #     raise ValueError('No saved metrics matched this run: config {} on budget {}'.format(run.config_id,
            #                                                                                         int(run.budget)))
            # ensmetrics = np.array(np.load(ensmetrics_fp, allow_pickle=True).item()['single_models'][hpo_loss]['all scores'])
            # centraltend_hpoloss = 1 - np.mean(ensmetrics[:, -1])  # np.median(ensmetrics[:, -1])
            # dev_hpoloss = np.std(ensmetrics[:, -1], ddof=1) / np.sqrt(ensmetrics.shape[0])
            # ens_hpoloss = run.loss  # ensemble hpo loss

            ensmetrics_fp = Path(hpo_dir / f'config{run.config_id}' / f'res_eval_budget_{int(run.budget)}epochs.npy')

            if not ensmetrics_fp.exists():  # did not find the file for that run
                raise ValueError('No saved metrics matched this run: config {} on budget {}'.format(run.config_id,
                                                                                                    int(run.budget)))

            ensmetrics = np.load(ensmetrics_fp, allow_pickle=True).item()
            ens_hpoloss = ensmetrics[f'test_{metric}']
            centraltend_hpoloss = ensmetrics[f'train_{metric}']
            dev_hpoloss = np.nan

            if run.loss < bconfig_loss:  # found run with better HPO loss than the best run so far

                bens_hpoloss, bcentraltend_hpoloss, bdev_hpoloss = ens_hpoloss, centraltend_hpoloss, dev_hpoloss

                tinc_hpoloss.append(ens_hpoloss)
                tinc_hpoloss_centraltend.append(centraltend_hpoloss)
                tinc_hpoloss_dev.append(dev_hpoloss)
                models_n_weights.append(configs_model_weights[str(run.config_id)])
            else:
                tinc_hpoloss.append(bens_hpoloss)
                tinc_hpoloss_centraltend.append(bcentraltend_hpoloss)
                tinc_hpoloss_dev.append(bdev_hpoloss)
                models_n_weights.append(models_n_weights[-1])

        if run.loss < bconfig_loss:
            bconfig_loss = run.loss

f, ax = plt.subplots()
# ax.plot(timestamps, tinc_hpoloss, label='Ensemble' if ensemble_study else None)
ax.plot(timestamps, tinc_hpoloss, linestyle='dashed', zorder=2)
ax.scatter(timestamps, tinc_hpoloss, label='Test', s=8, zorder=2)
if ensemble_study:
    # ax.errorbar(timestamps, tinc_hpoloss_centraltend,
    #             tinc_hpoloss_dev, capsize=5, label='Single model variability')
    ax.plot(timestamps, tinc_hpoloss_centraltend, 'r', linestyle='dashed', alpha=0.5, zorder=1)
    ax.scatter(timestamps, tinc_hpoloss_centraltend, c='r', label='Train', alpha=0.5, zorder=1, s=8)
# ax.set_yscale('log')
ax.set_xscale('log')
# ax.set_ylim(top=1)
# ax.set_xlim(right=1e5)
ax.set_ylabel(f'{metric}')
ax.set_xlabel('Wall clock time [s]')
# ax.set_title('')
ax.grid(True, which='both')
ax.legend()
ax.set_title('Budget {} (Best: {:.5f})'.format(budget_chosen, tinc_hpoloss[-1]))
f.tight_layout()
f.savefig(hpo_dir / 'walltime-hpoloss_budget_{}.png'.format(budget_chosen))

f, ax = plt.subplots()
ax.plot(cum_budget_vec, tinc_hpoloss, zorder=2, linestyle='dashed')  # , label='Ensemble' if ensemble_study else None)
ax.scatter(cum_budget_vec, tinc_hpoloss, label='Test', zorder=2, s=8)
if ensemble_study:
    # ax.errorbar(cum_budget_vec, tinc_hpoloss_centraltend,
    #             tinc_hpoloss_dev, capsize=5, label='Single model variability')
    ax.plot(cum_budget_vec, tinc_hpoloss_centraltend, 'r', linestyle='dashed', alpha=0.5, zorder=1)
    ax.scatter(cum_budget_vec, tinc_hpoloss_centraltend, c='r', label='Train', zorder=1, s=8)
# ax.set_yscale('log')
ax.set_xscale('log')
# ax.set_ylim(top=1)
ax.legend()
# ax.set_xlim(right=1e5)
ax.set_ylabel(f'{metric}')
ax.set_xlabel('Cumulative budget [Epochs]')
# ax.set_title('')
ax.grid(True, which='both')
ax.set_title('Budget {} (Best: {:.5f})'.format(budget_chosen, tinc_hpoloss[-1]))
ax.legend()
f.tight_layout()
f.savefig(hpo_dir / 'cumbudget-hpoloss_budget_{}.png'.format(budget_chosen))

f, ax = plt.subplots()
ax.plot(cum_budget_vec, models_n_weights, linestyle='dashed')  # , label='Ensemble' if ensemble_study else None)
ax.scatter(cum_budget_vec, models_n_weights, s=8)
# ax.set_yscale('log')
ax.set_xscale('log')
# ax.set_ylim(top=1)
# ax.legend()
# ax.set_xlim(right=1e5)
ax.set_ylabel('Num model weights')
ax.set_xlabel('Cumulative budget [Epochs]')
# ax.set_title('')
ax.grid(True, which='both')
ax.set_title('Budget {} (Best: {} weights)'.format(budget_chosen, models_n_weights[-1]))
f.tight_layout()
f.savefig(hpo_dir / 'cumbudget-num_model_weights_budget_{}.png'.format(budget_chosen))

#%% Study analysis plots

# metric used as optimization loss
hpo_loss = 'auc_pr'  # 'pr auc'

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
# inc_val_hpoloss, inc_test_hpoloss = inc_run.info['validation ' + hpo_loss], inc_run.info['test ' + hpo_loss]
# inc_val_hpoloss, inc_test_hpoloss = inc_run.info['val_{}'.format(hpo_loss)], inc_run.info['test_{}'.format(hpo_loss)]
# print('It achieved optimization loss ({}) of {} (validation) and {} (test).'.format(hpo_loss, inc_val_hpoloss,
#                                                                                     inc_test_hpoloss))
inc_train_hpoloss = inc_run.info['train_{}'.format(hpo_loss)]
inc_test_hpoloss = inc_run.info['test_{}'.format(hpo_loss)]
print('It achieved optimization loss ({}) of {} (train) and {} (test).'.format(hpo_loss, inc_train_hpoloss, inc_test_hpoloss))

# Let's plot the observed losses grouped by budget,
f, ax = hpvis.losses_over_time(all_runs)
ax.set_yscale('log')
f.set_size_inches(10, 6)
f.savefig(hpo_dir / 'losses_over_time_budgets.png')

# the number of concurrent runs,
f, _ = hpvis.concurrent_runs_over_time(all_runs)
f.savefig(hpo_dir / 'concurrent_runs_over_time.png')
# f.set_size_inches(10, 6)

# and the number of finished runs.
f, _ = hpvis.finished_runs_over_time(all_runs)
f.savefig(hpo_dir / 'finished_runs_over_time.png')

# This one visualizes the spearman rank correlation coefficients of the losses
# between different budgets.
f, _ = hpvis.correlation_across_budgets(res)
f.set_size_inches(10, 8)
f.savefig(hpo_dir / 'spearman_corr_budgets.png')

# For model based optimizers, one might wonder how much the model actually helped.
# The next plot compares the performance of configs picked by the model vs. random ones
if model_based_optimizer:
    f, _ = hpvis.performance_histogram_model_vs_random(all_runs, id2config)
    f.set_size_inches(10, 8)
    f.savefig(hpo_dir / 'hist_model-random.png')
