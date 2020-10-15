import numpy as np
import matplotlib.pyplot as plt
import glob

import paths
from src_hpo.utils_hpo import estimate_BOHB_runs, logged_results_to_HBS_result  # , json_result_logger

#%% check number of iterations per Successive Halving and per budget

num_iterations = 10
eta = 2
bmin, bmax = 5, 50
nmodels = 3
nruns, total_budget = estimate_BOHB_runs(num_iterations, eta, bmin, bmax, nmodels=nmodels)
print('Number of runs: {}\nTotal budget: {}'.format(nruns, total_budget))

#%% Compare different HPO studies

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
                mu_hpoloss = 1 - np.mean(ensmetrics[:, -1])  # np.median(ensmetrics[:, -1])  # compute mean loss
                sem_hpoloss = np.std(ensmetrics[:, -1], ddof=1) / np.sqrt(ensmetrics.shape[0])  # compute standard error of the mean

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
