from pathlib import Path
import pandas as pd
import numpy as np
from astropy.stats import mad_std
import matplotlib.pyplot as plt
import json

# %% aggregate metrics and predictions for all sub runs

exp_dir = Path(
    '/data5/tess_project/experiments/current_experiments/training_set_size/trainset_size_keplerq1q17dr25_afps_matchedntps_9-23-2021')

runs_dirs = [fp for fp in exp_dir.iterdir() if fp.is_dir() and fp.name.startswith('run_iter')]

datasets = ['test']
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

    run_params = json.load(open(run_dir / 'run_params.json', 'r'))

    sub_runs_dirs = [fp for fp in run_dir.iterdir() if fp.is_dir()]

    ensemble_pred_tbls = {dataset: None for dataset in datasets}
    for sub_run_dir in sub_runs_dirs:

        perf_metrics_sub_run = np.load(sub_run_dir / 'results_ensemble.npy', allow_pickle=True).item()
        subrun_idx = int(sub_run_dir.name.split('_')[-1])
        perf_metrics_tbl.loc[subrun_idx, 'subrun'] = subrun_idx
        perf_metrics_tbl.loc[subrun_idx, perf_metrics_lst] = [perf_metrics_sub_run[perf_metric] for perf_metric
                                                              in perf_metrics_lst]

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

    for dataset, ensemble_pred_tbl in ensemble_pred_tbls.items():
        ensemble_pred_tbl.to_csv(run_dir /
                                 f'ensemble_ranked_predictions_{dataset}set_allsubruns_trainset_frac_'
                                 f'{run_params["train_set_frac"]}.csv',
                                 index=False)

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
    perf_metrics_tbl['train_set_frac'] = run_params['train_set_frac']
    perf_metrics_tbl.to_csv(run_dir / f'results_ensemble_allsubruns_trainset_frac_{run_params["train_set_frac"]}.csv',
                            index=False)

# %%

perf_metrics_lst = ['test_loss', 'test_binary_accuracy', 'test_precision', 'test_recall',
                    'test_auc_pr', 'test_auc_roc',
                    'test_precision_at_50', 'test_precision_at_150', 'test_precision_at_251']

exp_dir = Path(
    '/data5/tess_project/experiments/current_experiments/training_set_size/trainset_size_keplerq1q17dr25_afps_matchedntps_9-23-2021')

runs_dirs = [fp for fp in exp_dir.iterdir() if fp.is_dir() and fp.name.startswith('run_iter')]

# perf_metrics_tbls = {}
# for run_dir in runs_dirs:
#     tbl_fp = [fp for fp in run_dir.iterdir() if fp.name.startswith('results_ensemble')][0]
#     perf_metrics_tbls[float(tbl_fp.stem.split('_')[-1])] = pd.read_csv(tbl_fp)

perf_metrics_tbls = []
for run_dir in runs_dirs:
    tbl_fp = [fp for fp in run_dir.iterdir() if fp.name.startswith('results_ensemble')][0]
    perf_metrics_tbls.append(pd.read_csv(tbl_fp))
perf_metrics_tbl = pd.concat(perf_metrics_tbls, axis=0)

perf_metrics_tbl = perf_metrics_tbl.set_index(['train_set_frac', 'subrun']).sort_index()
perf_metrics_tbl.to_csv(exp_dir / 'results_ensemble_allruns.csv')

trainset_frac_arr = perf_metrics_tbl.index.get_level_values(0).unique().to_numpy()
metric_map = {

}
for perf_metric in perf_metrics_lst:
    y_arr = [perf_metrics_tbl.loc[(trainset_frac, 'mean')][perf_metric] for trainset_frac in trainset_frac_arr]
    y_err_arr = [perf_metrics_tbl.loc[(trainset_frac, 'std')][perf_metric] for trainset_frac in trainset_frac_arr]

    f, ax = plt.subplots()
    ax.errorbar(trainset_frac_arr, y_arr, yerr=y_err_arr, capsize=1)
    ax.set_ylabel(f'Metric {perf_metric}')
    ax.set_xlabel('Training set fraction used')
    f.savefig(exp_dir / f'plot_trainset_frac_{perf_metric}.png')
    plt.close()
