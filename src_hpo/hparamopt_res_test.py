import numpy as np
import matplotlib.pyplot as plt
import glob

import paths
from src_hpo.utils_hpo import estimate_BOHB_runs, logged_results_to_HBS_result  # , json_result_logger

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
