from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.stats import mad_std

# %% aggregate metrics and predictions for all sub runs

exp_dir = Path(
    '/data5/tess_project/experiments/current_experiments/label_noise/label_noise_keplerq1q17dr25_0-0.5_NTPswitch_9-6-2021')

runs_dirs = [fp for fp in exp_dir.iterdir() if fp.is_dir() and fp.name.startswith('run_iter')]

datasets = ['train', 'val', 'test']
cols_tbl = ['target_id', 'tce_plnt_num', 'label', 'tce_period', 'tce_duration', 'tce_time0bk', 'transit_depth',
            'original_label', 'score']
perf_metrics_lst = ['train_loss', 'train_binary_accuracy', 'train_precision', 'train_recall',
                    'train_auc_pr', 'train_auc_roc', 'val_loss',
                    'val_binary_accuracy', 'val_precision', 'val_recall', 'val_auc_pr', 'val_auc_roc',
                    'test_loss', 'test_binary_accuracy', 'test_precision', 'test_recall',
                    'test_auc_pr', 'test_auc_roc',
                    'train_precision_at_100', 'train_precision_at_1000', 'train_precision_at_1818',
                    'val_precision_at_50', 'val_precision_at_150', 'val_precision_at_222',
                    'test_precision_at_50', 'test_precision_at_150', 'test_precision_at_251']
n_subruns = 5

for run_dir in runs_dirs:

    perf_metrics_tbl = pd.DataFrame(data=np.nan * np.ones((n_subruns, len(perf_metrics_lst) + 1)),
                                    columns=['subrun'] + perf_metrics_lst)

    prob_mat_dict = np.load(run_dir / 'prob_transition_mat.npy', allow_pickle=True).item()

    sub_runs_dirs = [fp for fp in run_dir.iterdir() if fp.is_dir()]

    ensemble_pred_tbls = {dataset: None for dataset in datasets}
    flip_tbls = {dataset: None for dataset in ['train', 'val']}
    for sub_run_dir in sub_runs_dirs:

        # aggregate label flipping for the sub-runs
        flip_tbls_subrun = {dataset: pd.read_csv(sub_run_dir / f'flip_label_{dataset}set_tbl.csv')
                            for dataset in flip_tbls}

        for dataset in flip_tbls:
            if flip_tbls[dataset] is None:
                flip_tbls[dataset] = \
                    flip_tbls_subrun[dataset][['target_id', 'tce_plnt_num', 'label', 'new_label']].copy(deep=True)
                flip_tbls[dataset].rename(columns={'new_label': f'new_label_{sub_run_dir.name}'}, inplace=True)
            else:
                flip_tbls[dataset] = \
                    pd.merge(flip_tbls[dataset],
                             flip_tbls_subrun[dataset][['target_id', 'tce_plnt_num', 'new_label']],
                             on=['target_id', 'tce_plnt_num']
                             )
                flip_tbls[dataset].rename(columns={'new_label': f'new_label_{sub_run_dir.name}'}, inplace=True)

            # flip_tbls[dataset]['tau'] = prob_mat_dict['tau']

        # aggregate metrics for the sub-runs
        perf_metrics_sub_run = np.load(sub_run_dir / 'results_ensemble.npy', allow_pickle=True).item()
        subrun_idx = int(sub_run_dir.name.split('_')[-1])
        perf_metrics_tbl.loc[subrun_idx, 'subrun'] = subrun_idx
        perf_metrics_tbl.loc[subrun_idx, perf_metrics_lst] = [perf_metrics_sub_run[perf_metric] for perf_metric
                                                              in perf_metrics_lst]

        # aggregate predictions for the sub-runs
        for dataset in datasets:
            ensemble_pred_tbl_sub_run = pd.read_csv(sub_run_dir / f'ensemble_ranked_predictions_{dataset}set.csv',
                                                    usecols=cols_tbl)
            if ensemble_pred_tbls[dataset] is None:
                ensemble_pred_tbls[dataset] = ensemble_pred_tbl_sub_run
                ensemble_pred_tbls[dataset].rename(columns={'score': f'score_{sub_run_dir.name}'}, inplace=True)
            else:
                ensemble_pred_tbls[dataset] = \
                    pd.merge(ensemble_pred_tbls[dataset],
                             ensemble_pred_tbl_sub_run[['target_id', 'tce_plnt_num', 'score']],
                             on=['target_id', 'tce_plnt_num']
                             )
                ensemble_pred_tbls[dataset].rename(columns={'score': f'score_{sub_run_dir.name}'}, inplace=True)

    for dataset, flip_tbl in flip_tbls.items():
        flip_tbl.to_csv(run_dir /
                        f'flip_label_{dataset}set_allsubruns_tau_{prob_mat_dict["tau"]}.csv',
                        index=False)

    for dataset, ensemble_pred_tbl in ensemble_pred_tbls.items():
        ensemble_pred_tbl.to_csv(run_dir /
                                 f'ensemble_ranked_predictions_{dataset}set_allsubruns_tau_{prob_mat_dict["tau"]}.csv',
                                 index=False)

    # compute statistics for metrics across sub-runs
    perf_metrics_mean = perf_metrics_tbl.apply(lambda x: np.mean(x), axis=0)
    perf_metrics_mean['subrun'] = 'mean'
    perf_metrics_std = perf_metrics_tbl.apply(lambda x: np.std(x, ddof=1), axis=0)
    perf_metrics_std['subrun'] = 'std'
    perf_metrics_median = perf_metrics_tbl.apply(lambda x: np.median(x), axis=0)
    perf_metrics_median['subrun'] = 'median'
    perf_metrics_madstd = perf_metrics_tbl.apply(lambda x: mad_std(x), axis=0)
    perf_metrics_madstd['subrun'] = 'mad_std'
    perf_metrics_tbl = pd.concat([perf_metrics_tbl, perf_metrics_mean.to_frame().T, perf_metrics_std.to_frame().T,
                                  perf_metrics_median.to_frame().T, perf_metrics_madstd.to_frame().T],
                                 axis=0, ignore_index=True)
    perf_metrics_tbl['tau'] = prob_mat_dict['tau']
    perf_metrics_tbl.to_csv(run_dir / f'results_ensemble_allsubruns_tau_{prob_mat_dict["tau"]}.csv', index=False)

# %% aggregate metrics from all runs and plot metrics values as function of tau value

perf_metrics_lst = ['test_loss', 'test_binary_accuracy', 'test_precision', 'test_recall',
                    'test_auc_pr', 'test_auc_roc',
                    'test_precision_at_50', 'test_precision_at_150', 'test_precision_at_251']

exp_dir = Path(
    '/data5/tess_project/experiments/current_experiments/label_noise/label_noise_keplerq1q17dr25_0-0.5_NTPswitch_9-6-2021')
runs_dirs = [fp for fp in exp_dir.iterdir() if fp.is_dir() and fp.name.startswith('run_iter')]

for dataset in ['train', 'val']:
    flip_tbls = {}
    for run_dir in runs_dirs:
        tau = np.load(run_dir / 'prob_transition_mat.npy', allow_pickle=True).item()['tau']
        flip_label_fp = [fp for fp in run_dir.iterdir() if f'flip_label_{dataset}' in fp.name][0]
        flip_tbls[f'tau_{tau}'] = pd.read_csv(flip_label_fp).set_index(['target_id', 'tce_plnt_num', 'label'])

    flip_tbls = {key: flip_tbls[key] for key in sorted(flip_tbls)}
    flip_tbl_run_dir = pd.concat(flip_tbls, axis=1, names=['run', 'sub-run'])
    flip_tbl_run_dir.to_csv(exp_dir / f'flip_label_{dataset}set_allruns.csv')

# perf_metrics_tbls = {}
# for run_dir in runs_dirs:
#     tbl_fp = [fp for fp in run_dir.iterdir() if fp.name.startswith('results_ensemble')][0]
#     perf_metrics_tbls[float(tbl_fp.stem.split('_')[-1])] = pd.read_csv(tbl_fp)

perf_metrics_tbls = []
for run_dir in runs_dirs:
    tbl_fp = [fp for fp in run_dir.iterdir() if fp.name.startswith('results_ensemble')][0]
    perf_metrics_tbls.append(pd.read_csv(tbl_fp))
perf_metrics_tbl = pd.concat(perf_metrics_tbls, axis=0)

perf_metrics_tbl = perf_metrics_tbl.set_index(['tau', 'subrun']).sort_index()
perf_metrics_tbl.to_csv(exp_dir / 'results_ensemble_allruns.csv')

tau_arr = perf_metrics_tbl.index.get_level_values(0).unique().to_numpy()
metric_map = {

}
for perf_metric in perf_metrics_lst:
    y_arr = [perf_metrics_tbl.loc[(tau, 'mean')][perf_metric] for tau in tau_arr]
    y_err_arr = [perf_metrics_tbl.loc[(tau, 'std')][perf_metric] for tau in tau_arr]

    f, ax = plt.subplots()
    ax.errorbar(tau_arr, y_arr, yerr=y_err_arr, capsize=1)
    ax.set_ylabel(f'Metric {perf_metric}')
    ax.set_xlabel(r'$\tau$')
    f.savefig(exp_dir / f'plot_tau_{perf_metric}.png')
    plt.close()
