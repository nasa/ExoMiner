"""
Aggregate results across explainability experiments.
"""

# 3rd party
import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc as sk_auc

#%% aggregate top-n performance metrics across experiments

save_dir = Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/experiments/explainability/')

metrics_chosen = [
    # 'hamming',
    # 'exact_match_ratio',
    # 'jaccard_macro',

    'precision_macro',
    'recall_macro',

    # 'precision_micro',
    # 'recall_micro'
]

top_n_fps = {
    'random_analysis': Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/experiments/explainability/random_analysis/run_random_analysis_05-09-2023_1212/metrics_top_n_1-5.csv'),
    'shap': Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/experiments/explainability/shap/shap_1-11-2022/metrics_top_n_1-5.csv'),
    'shap_1': Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/experiments/explainability/shap1/shap1_1-10-2022/metrics_top_n_1-5.csv'),
    'occlusion': Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/experiments/explainability/occlusion/run_occlusion_zero_05-11-2023_1650/metrics_top_n_1-5.csv'),
    'ebrc_15pcs_max': Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/experiments/explainability/model_pc_replacement/run_num_model_pcs_15_05-02-2023_1748/max_eval/metrics_top_n_1-5_testset.csv')
}
top_n_tbl_all = []
for exp_name, fp in top_n_fps.items():
    if np.any([el in exp_name for el in ['random_analysis', 'ebrc']]):
        metrics_chosen_aux = [f'{metric}_mean' for metric in metrics_chosen]
    else:
        metrics_chosen_aux = list(metrics_chosen)

    if len(top_n_tbl_all) == 0:
        top_n_tbl = pd.read_csv(fp)[['top_n'] + metrics_chosen_aux]
    else:
        top_n_tbl = pd.read_csv(fp)[metrics_chosen_aux]

    top_n_tbl.rename(columns={metric: f'{exp_name}_{metric}' for metric in metrics_chosen_aux}, inplace=True)
    top_n_tbl_all.append(top_n_tbl)

top_n_tbl_all = pd.concat(top_n_tbl_all, axis=1)
top_n_tbl_all.to_csv(save_dir / 'metrics_top_n_1-5_allexperiments_macroavg_precision_recall.csv', index=False)

#%% aggregate threshold performance experiments across experiments

save_dir = Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/experiments/explainability/')

metrics_chosen = [
    'precision_macro',
    'recall_macro',

    'precision_micro',
    'recall_micro',

    'precision_weighted',
    'recall_weighted',

    'precision_samples',
    'recall_samples',

    'hamming',
    'exact_match_ratio',

    'jaccard_macro',
    'jaccard_micro',
    'jaccard_weighted',
    'jaccard_samples',
]

top_n_fps = {
    'random_analysis': Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/experiments/explainability/random_analysis/run_random_analysis_05-09-2023_1212/metrics_thr_-1.0-1.0.csv'),
    'shap': Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/experiments/explainability/shap/shap_1-11-2022/metrics_thr_-1.0-1.0.csv'),
    'shap_1': Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/experiments/explainability/shap1/shap1_1-10-2022/metrics_thr_-1.0-1.0.csv'),
    'occlusion': Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/experiments/explainability/occlusion/run_occlusion_zero_05-11-2023_1650/metrics_thr_-1.0-1.0.csv'),
    'ebrc_15pcs_max': Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/experiments/explainability/model_pc_replacement/run_num_model_pcs_15_05-02-2023_1748/max_eval/metrics_thr_-1.0-1.0.csv'),
    'ebrc_15pcs_min': Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/experiments/explainability/model_pc_replacement/run_num_model_pcs_15_05-02-2023_1748/min_eval/metrics_thr_-1.0-1.0.csv'),
    'ebrc_15pcs_median': Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/experiments/explainability/model_pc_replacement/run_num_model_pcs_15_05-02-2023_1748/median_eval/metrics_thr_-1.0-1.0.csv')
}
thr_tbl_all = []
for exp_name, fp in top_n_fps.items():
    if np.any([el in exp_name for el in ['random_analysis', 'ebrc']]):
        metrics_chosen_aux = [f'{metric}_mean' for metric in metrics_chosen]
    else:
        metrics_chosen_aux = list(metrics_chosen)

    if len(thr_tbl_all) == 0:
        thr_tbl = pd.read_csv(fp)[['thr'] + metrics_chosen_aux]
    else:
        thr_tbl = pd.read_csv(fp)[metrics_chosen_aux]

    thr_tbl.rename(columns={metric: f'{exp_name}_{metric}' for metric in metrics_chosen_aux}, inplace=True)
    thr_tbl_all.append(thr_tbl)

thr_tbl_all = pd.concat(thr_tbl_all, axis=1)
thr_tbl_all.to_csv(save_dir / 'metrics_thr_-1.0-1.0_allexperiments.csv', index=False)

#%% aggregate threshold performance for no flag across experiments

save_dir = Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/experiments/explainability/')

metrics_chosen = [
    'precision',
    'recall',
]

top_n_fps = {
    'random_analysis': Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/experiments/explainability/random_analysis/run_random_analysis_05-09-2023_1212/metrics_pc_thr_-1.0-1.0.csv'),
    'shap': Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/experiments/explainability/shap/shap_1-11-2022/metrics_pc_thr_-1.0-1.0.csv'),
    'shap_1': Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/experiments/explainability/shap1/shap1_1-10-2022/metrics_pc_thr_-1.0-1.0.csv'),
    'occlusion': Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/experiments/explainability/occlusion/run_occlusion_zero_05-11-2023_1650/metrics_pc_thr_-1.0-1.0.csv'),
    'ebrc_15pcs_max': Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/experiments/explainability/model_pc_replacement/run_num_model_pcs_15_05-02-2023_1748/max_eval/metrics_pc_thr_-1.0-1.0.csv'),
    'ebrc_15pcs_min': Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/experiments/explainability/model_pc_replacement/run_num_model_pcs_15_05-02-2023_1748/min_eval/metrics_pc_thr_-1.0-1.0.csv'),
    'ebrc_15pcs_median': Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/experiments/explainability/model_pc_replacement/run_num_model_pcs_15_05-02-2023_1748/median_eval/metrics_pc_thr_-1.0-1.0.csv')
}
thr_tbl_all = []
for exp_name, fp in top_n_fps.items():
    if np.any([el in exp_name for el in ['random_analysis', 'ebrc']]):
        metrics_chosen_aux = [f'{metric}_mean' for metric in metrics_chosen]
    else:
        metrics_chosen_aux = list(metrics_chosen)

    if len(thr_tbl_all) == 0:
        thr_tbl = pd.read_csv(fp)[['thr'] + metrics_chosen_aux]
    else:
        thr_tbl = pd.read_csv(fp)[metrics_chosen_aux]

    thr_tbl.rename(columns={metric: f'{exp_name}_{metric}' for metric in metrics_chosen_aux}, inplace=True)
    thr_tbl_all.append(thr_tbl)

thr_tbl_all = pd.concat(thr_tbl_all, axis=1)
thr_tbl_all.to_csv(save_dir / 'metrics_thr_-1.0-1.0_noflag_allexperiments.csv', index=False)

#%% aggregate top-n precision and recall per flag across experiments

save_dir = Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/experiments/explainability/')

top_n_fps = {
    'random_analysis': Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/experiments/explainability/random_analysis/run_random_analysis_05-09-2023_1212/precision_recall_per_flag_top_n_1-5.csv'),
    'shap': Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/experiments/explainability/shap/shap_1-11-2022/precision_recall_per_flag_top_n_1-5.csv'),
    'shap_1': Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/experiments/explainability/shap1/shap1_1-10-2022/precision_recall_per_flag_top_n_1-5.csv'),
    'occlusion': Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/experiments/explainability/occlusion/run_occlusion_zero_05-11-2023_1650/precision_recall_per_flag_top_n_1-5.csv'),
    'ebrc_15pcs_max': Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/experiments/explainability/model_pc_replacement/run_num_model_pcs_15_05-02-2023_1748/max_eval/precision_recall_per_flag_top_n_1-5_testset.csv'),
    'ebrc_15pcs_min': Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/experiments/explainability/model_pc_replacement/run_num_model_pcs_15_05-02-2023_1748/min_eval/precision_recall_per_flag_top_n_1-5_testset.csv'),
    'ebrc_15pcs_median': Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/experiments/explainability/model_pc_replacement/run_num_model_pcs_15_05-02-2023_1748/median_eval/precision_recall_per_flag_top_n_1-5_testset.csv'),
}
top_n_tbl_all = []
for exp_name, fp in top_n_fps.items():
    #     metrics_chosen_aux = [f'{metric}_mean' for metric in metrics_chosen]
    # else:
    #     metrics_chosen_aux = list(metrics_chosen)

    # if len(top_n_tbl_all) == 0:
    #     top_n_tbl = pd.read_csv(fp)[['top_n'] + metrics_chosen_aux]
    # else:
    #     top_n_tbl = pd.read_csv(fp)[metrics_chosen_aux]

    # top_n_tbl.rename(columns={metric: f'{exp_name}_{metric}' for metric in metrics_chosen_aux}, inplace=True)
    tbl = pd.read_csv(fp)
    if np.any([el in exp_name for el in ['random_analysis', 'ebrc']]):
        tbl.rename(columns={col: f'{col.replace("_mean", "")}' for col in tbl.columns if 'mean' in col}, inplace=True)
        tbl.drop(columns=[col for col in tbl.columns if 'std' in col], inplace=True)

    tbl.insert(0, f'experiment_name', [exp_name] * len(tbl))
    top_n_tbl_all.append(tbl)
    # top_n_tbl_all.append(top_n_tbl)

top_n_tbl_all = pd.concat(top_n_tbl_all, axis=0, ignore_index=True)
top_n_tbl_all.to_csv(save_dir / 'precision_recall_per_flag_top_n_1-5_allexperiments.csv', index=False)

#%% aggregate threshold precision and recall per flag across experiments

save_dir = Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/experiments/explainability/')

top_n_fps = {
    'random_analysis': Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/experiments/explainability/random_analysis/run_random_analysis_05-09-2023_1212/precision_recall_per_flag_thr_-1.0-1.0.csv'),
    'shap': Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/experiments/explainability/shap/shap_1-11-2022/precision_recall_per_flag_thr_-1.0-1.0.csv'),
    'shap_1': Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/experiments/explainability/shap1/shap1_1-10-2022/precision_recall_per_flag_thr_-1.0-1.0.csv'),
    'occlusion': Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/experiments/explainability/occlusion/run_occlusion_zero_05-11-2023_1650/precision_recall_per_flag_thr_-1.0-1.0.csv'),
    'ebrc_15pcs_max': Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/experiments/explainability/model_pc_replacement/run_num_model_pcs_15_05-02-2023_1748/max_eval/precision_recall_per_flag_thr_-1.0-1.0.csv'),
    'ebrc_15pcs_min': Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/experiments/explainability/model_pc_replacement/run_num_model_pcs_15_05-02-2023_1748/min_eval/precision_recall_per_flag_thr_-1.0-1.0.csv'),
    'ebrc_15pcs_median': Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/experiments/explainability/model_pc_replacement/run_num_model_pcs_15_05-02-2023_1748/median_eval/precision_recall_per_flag_thr_-1.0-1.0.csv'),
}
top_n_tbl_all = []
for exp_name, fp in top_n_fps.items():

    tbl = pd.read_csv(fp)
    if np.any([el in exp_name for el in ['random_analysis', 'ebrc']]):
        tbl.rename(columns={col: f'{col.replace("_mean", "")}' for col in tbl.columns if 'mean' in col}, inplace=True)
        tbl.drop(columns=[col for col in tbl.columns if 'std' in col], inplace=True)

    tbl.insert(0, f'experiment_name', [exp_name] * len(tbl))
    top_n_tbl_all.append(tbl)

top_n_tbl_all = pd.concat(top_n_tbl_all, axis=0, ignore_index=True)
top_n_tbl_all.to_csv(save_dir / 'precision_recall_per_flag_thr_-1.0-1.0_allexperiments.csv', index=False)

#%% plot PR curves for all experiments

save_dir = Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/experiments/explainability/')

exp_dir_fps = {
    'random_analysis': Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/experiments/explainability/random_analysis/run_random_analysis_05-09-2023_1212/'),
    'shap': Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/experiments/explainability/shap/shap_1-11-2022/'),
    'shap_1': Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/experiments/explainability/shap1/shap1_1-10-2022/'),
    'occlusion': Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/experiments/explainability/occlusion/run_occlusion_zero_05-11-2023_1650'),
    'ebrc_15pcs_min': Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/experiments/explainability/model_pc_replacement/run_num_model_pcs_15_05-02-2023_1748/min_eval'),
    'ebrc_15pcs_med': Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/experiments/explainability/model_pc_replacement/run_num_model_pcs_15_05-02-2023_1748/median_eval/'),
    'ebrc_15pcs_max': Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/experiments/explainability/model_pc_replacement/run_num_model_pcs_15_05-02-2023_1748/max_eval'),
}

exp_dir_plot_params = {
    'random_analysis': {'name': 'Random', 'color': 'b', 'linestyle': '-'},
    'shap': {'name': 'SHAP', 'color': 'r', 'linestyle': '--'},
    'shap_1': {'name': 'SHAP1', 'color': 'g', 'linestyle': '-.'},
    'occlusion': {'name': 'Occlusion', 'color': 'k', 'linestyle': '-'},
    'ebrc_15pcs_min': {'name': 'NOREX Min', 'color': 'c', 'linestyle': '-.'},
    'ebrc_15pcs_med': {'name': 'NOREX Med', 'color': 'y', 'linestyle': '-.'},
    'ebrc_15pcs_max': {'name': 'NOREX Max', 'color': 'm', 'linestyle': '-.'},

}

avg_type_arr = [
    'micro',
    'macro',
    'weighted',
    'samples',
]
thr_arr = np.linspace(-1, 1, 100, endpoint=True)

for avg_type in avg_type_arr:

    f, ax = plt.subplots(figsize=(10, 6))

    for exp_dir in exp_dir_fps:
        precision_arr = np.load(exp_dir_fps[exp_dir] / f'precision_{avg_type}_avg_arr.npy')
        recall_arr = np.load(exp_dir_fps[exp_dir] / f'recall_{avg_type}_avg_arr.npy')

        ax.scatter(recall_arr, precision_arr, color=exp_dir_plot_params[exp_dir]['color'], s=8)
        ax.plot(recall_arr, precision_arr, color=exp_dir_plot_params[exp_dir]['color'],
                linestyle=exp_dir_plot_params[exp_dir]['linestyle'],
                label=exp_dir_plot_params[exp_dir]['name'], linewidth=3)
    ax.set_xlabel('Recall', fontsize=20)
    ax.set_ylabel('Precision', fontsize=20)
    ax.grid()
    ax.legend(fontsize=16, loc=0)
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    # ax.set_xscale('log')
    # ax.set_yscale('log')
    f.savefig(save_dir / f'pr_curve_avg_{avg_type}_allexperiments.pdf')

for avg_type in avg_type_arr:

    f, ax = plt.subplots(2, 1, figsize=(10, 6))

    for exp_dir in exp_dir_fps:

        precision_arr = np.load(exp_dir_fps[exp_dir] / f'precision_{avg_type}_avg_arr.npy')
        recall_arr = np.load(exp_dir_fps[exp_dir] / f'recall_{avg_type}_avg_arr.npy')

        ax[0].scatter(thr_arr, precision_arr , color=exp_dir_plot_params[exp_dir]['color'], s=8)
        ax[0].plot(thr_arr, precision_arr, color=exp_dir_plot_params[exp_dir]['color'],
                linestyle=exp_dir_plot_params[exp_dir]['linestyle'],
                label=exp_dir_plot_params[exp_dir]['name'])
        ax[0].set_ylabel('Precision')
        ax[0].set_xlabel('Threshold')
        ax[0].grid(True)
        ax[0].legend()
        ax[0].set_xlim(thr_arr[[0, -1]])
        ax[0].set_ylim([0, 1])

        ax[1].scatter(thr_arr, recall_arr , color=exp_dir_plot_params[exp_dir]['color'], s=8)
        ax[1].plot(thr_arr, recall_arr, color=exp_dir_plot_params[exp_dir]['color'],
                linestyle=exp_dir_plot_params[exp_dir]['linestyle'],
                label=exp_dir_plot_params[exp_dir]['name'])
        ax[1].set_ylabel('Recall')
        ax[1].set_xlabel('Threshold')
        ax[1].grid(True)
        ax[1].legend()
        ax[1].set_xlim(thr_arr[[0, -1]])
        ax[1].set_ylim([0, 1])
    f.tight_layout()
    f.savefig(save_dir / f'precision-thr_recall-thr_curves_avg_{avg_type}_allexperiments.pdf')


f, ax = plt.subplots(figsize=(10, 6))
for exp_dir in exp_dir_fps:
    precision_arr = np.load(exp_dir_fps[exp_dir] / 'precision_arr_noflag.npy')
    recall_arr = np.load(exp_dir_fps[exp_dir] / 'recall_arr_noflag.npy')

    ax.scatter(recall_arr, precision_arr, color=exp_dir_plot_params[exp_dir]['color'], s=8)
    ax.plot(recall_arr, precision_arr, color=exp_dir_plot_params[exp_dir]['color'],
            linestyle=exp_dir_plot_params[exp_dir]['linestyle'],
            label=exp_dir_plot_params[exp_dir]['name'], linewidth=3)
ax.set_xlabel('Recall', fontsize=20)
ax.set_ylabel('Precision', fontsize=20)
ax.grid()
ax.legend(fontsize=16)
ax.set_xlim([0, 1.01])
ax.set_ylim([0, 1.01])
ax.tick_params(axis='both', which='major', labelsize=18)
# ax.set_xscale('log')
# ax.set_yscale('log')
f.savefig(save_dir / f'pr_curve_pcnoflag_allexperiments.pdf')

f, ax = plt.subplots(2, 1, figsize=(10, 6))
for exp_dir in exp_dir_fps:

    precision_arr = np.load(exp_dir_fps[exp_dir] / 'precision_arr_noflag.npy')
    recall_arr = np.load(exp_dir_fps[exp_dir] / 'recall_arr_noflag.npy')

    ax[0].scatter(thr_arr, precision_arr , color=exp_dir_plot_params[exp_dir]['color'], s=8)
    ax[0].plot(thr_arr, precision_arr, color=exp_dir_plot_params[exp_dir]['color'],
            linestyle=exp_dir_plot_params[exp_dir]['linestyle'],
            label=exp_dir_plot_params[exp_dir]['name'])
    ax[0].set_ylabel('Precision')
    ax[0].set_xlabel('Threshold')
    ax[0].grid(True)
    ax[0].legend()
    ax[0].set_xlim(thr_arr[[0, -1]])
    ax[0].set_ylim([0, 1])

    ax[1].scatter(thr_arr, recall_arr , color=exp_dir_plot_params[exp_dir]['color'], s=8)
    ax[1].plot(thr_arr, recall_arr, color=exp_dir_plot_params[exp_dir]['color'],
            linestyle=exp_dir_plot_params[exp_dir]['linestyle'],
            label=exp_dir_plot_params[exp_dir]['name'])
    ax[1].set_ylabel('Recall')
    ax[1].set_xlabel('Threshold')
    ax[1].grid(True)
    ax[1].legend()
    ax[1].set_xlim(thr_arr[[0, -1]])
    ax[1].set_ylim([0, 1])
f.tight_layout()
f.savefig(save_dir / f'precision-thr_recall-thr_curves_pcnoflag_allexperiments.pdf')

# thr_arr = np.linspace(0, 1, 100, endpoint=True)
# f, ax = plt.subplots(figsize=(10, 6))
# ax2 = ax.twinx()
# for exp_dir in exp_dir_fps:
#
#     precision_arr = np.load(exp_dir_fps[exp_dir] / 'precision_arr.npy')
#     recall_arr = np.load(exp_dir_fps[exp_dir] / 'recall_arr.npy')
#
#     # ax.scatter(thr_arr, precision_arr, color=exp_dir_plot_params[exp_dir]['color'], s=8)
#     ax.plot(thr_arr, precision_arr, color=exp_dir_plot_params[exp_dir]['color'],
#             linestyle='-',
#             label=exp_dir_plot_params[exp_dir]['name'])
#
#     ax.set_ylabel('Precision "-"')
#     ax.set_xlabel('Threshold')
#     ax.grid(True)
#     ax.legend()
#     ax.set_xlim([0, 1])
#     ax.set_ylim([-0.01, 1.01])
#
#     # ax2.scatter(thr_arr, recall_arr, color=exp_dir_plot_params[exp_dir]['color'], s=8)
#     ax2.plot(thr_arr, recall_arr, color=exp_dir_plot_params[exp_dir]['color'],
#             linestyle='--',
#             label=exp_dir_plot_params[exp_dir]['name'])
#     ax2.set_ylabel('Recall "--"')
#     ax2.set_ylim([-0.01, 1.01])
# f.tight_layout()
# f.savefig(save_dir / f'precision-thr_recall-thr_curves_allexperiments_sameplot.pdf')

#%% Plot precision and recall curves vs num pcs

exp_root_dir = Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/experiments/explainability/model_pc_replacement/')
exp_dirs = [fp for fp in exp_root_dir.iterdir() if fp.name.startswith('run_num_model_pcs')]
save_dir = exp_root_dir / 'n_pcs_analysis'
save_dir.mkdir(exist_ok=True)
agg_pcs_eval_type = 'max_eval'
# exp_dirs_plot_dict = {fp / agg_pcs_eval_type: {'num_pcs': int(fp.name.split('_')[4])} for fp in exp_dirs}
exp_dirs_plot_dict = {fp: {'num_pcs': int(fp.name.split('_')[4])} for fp in exp_dirs}

avg_type_arr = ['micro', 'macro', 'weighted', 'samples']

agg_pcs_eval_types = {
    'median_eval': {'label': 'Med', 'fmt': 'o', 'color': ['b', 'orange']},
    'max_eval': {'label': 'Max', 'fmt': '^', 'color': ['g', 'm']},
    'min_eval': {'label': 'Min', 'fmt': '*', 'color': ['k', 'c']},
}

thr_arr = np.linspace(-1, 1, 100, endpoint=True)
# precision and recall curves (mean +- std)
for avg_type in avg_type_arr:
    f, ax = plt.subplots(2, 1, figsize=(10, 6))
    for exp_dir in exp_dirs_plot_dict:

        precision_arr = np.load(exp_dir / f'precision_trials_10_{avg_type}_avg_arr.npy')
        recall_arr = np.load(exp_dir / f'precision_trials_10_{avg_type}_avg_arr.npy')

        precision_arr_mean = np.mean(precision_arr, axis=1)
        precision_arr_std = np.std(precision_arr, axis=1, ddof=1)

        recall_arr_mean = np.mean(recall_arr, axis=1)
        recall_arr_std = np.std(recall_arr, axis=1, ddof=1)

        # ax[0].scatter(thr_arr, precision_arr , color=exp_dir_plot_params[exp_dir]['color'], s=8)
        ax[0].plot(thr_arr, precision_arr_mean,  #  color=exp_fps[exp_dir]['color'],
                linestyle='-',
                label=exp_dirs_plot_dict[exp_dir]['num_pcs'])
        ax[0].plot(thr_arr, precision_arr_mean + precision_arr_std,  #  color=exp_fps[exp_dir]['color'],
                linestyle='--',)
        ax[0].plot(thr_arr, precision_arr_mean - precision_arr_std,  #  color=exp_fps[exp_dir]['color'],
                linestyle='--',
    )
        ax[0].set_ylabel('Precision', fontsize=20)
        ax[0].set_xlabel('Threshold', fontsize=20)
        ax[0].grid(True)
        # ax[0].legend(fontsize=12)
        ax[0].set_xlim([-1, 1])
        ax[0].set_ylim([0, 1])
        ax[0].tick_params(axis='both', which='major', labelsize=18)

        ax[1].plot(thr_arr, recall_arr_mean,  #  color=exp_fps[exp_dir]['color'],
                   linestyle='-',
                   label=exp_dirs_plot_dict[exp_dir]['num_pcs'])
        ax[1].plot(thr_arr, recall_arr_mean + recall_arr_std,  # color=exp_fps[exp_dir]['color'],
                   linestyle='--', )
        ax[1].plot(thr_arr, recall_arr_mean - recall_arr_std,  #  color=exp_fps[exp_dir]['color'],
                   linestyle='--',
                   )
        ax[1].set_ylabel('Recall', fontsize=20)
        ax[1].set_xlabel('Threshold', fontsize=20)
        ax[1].grid(True)
        # ax[1].legend(fontsize=12)
        ax[1].set_xlim([-1, 1])
        ax[1].set_ylim([0, 1])
        ax[1].tick_params(axis='both', which='major', labelsize=18)
    f.tight_layout()
    f.savefig(save_dir / f'precision-thr_recall-thr_curves_avg_{avg_type}_allebrcexperiments_{agg_pcs_eval_type}.pdf')

# pr curves
for avg_type in avg_type_arr:
    f, ax = plt.subplots(figsize=(10, 6))
    for exp_dir in exp_dirs_plot_dict:

        precision_arr = np.load(exp_dir / f'precision_trials_10_{avg_type}_avg_arr.npy')
        recall_arr = np.load(exp_dir / f'recall_trials_10_{avg_type}_avg_arr.npy')

        precision_arr_mean = np.mean(precision_arr, axis=1)
        recall_arr_mean = np.mean(recall_arr, axis=1)

        # ax[0].scatter(thr_arr, precision_arr , color=exp_dir_plot_params[exp_dir]['color'], s=8)
        ax.plot(recall_arr_mean, precision_arr_mean,  #  color=exp_fps[exp_dir]['color'],
                linestyle='-',
                label=exp_dirs_plot_dict[exp_dir]['num_pcs'], linewidth=3.0)
        ax.set_ylabel('Precision', fontsize=20)
        ax.set_xlabel('Recall', fontsize=20)
        ax.grid(True)
        # ax.legend(fontsize=16)
        ax.set_xlim([0, 1.01])
        ax.set_ylim([0, 1.01])
        ax.tick_params(axis='both', which='major', labelsize=18)
    f.tight_layout()
    f.savefig(save_dir / f'pr_curve_avg_{avg_type}_allebrcexperiments_{agg_pcs_eval_type}.pdf')

# precision and recall for a given threshold (mean +- std)

thr = 0.5
for avg_type in avg_type_arr:
    prec_avg, rec_avg = [], []
    prec_var, rec_var = [], []
    n_pcs_arr = []

    for exp_dir in exp_dirs_plot_dict:
        n_pcs_arr.append(exp_dirs_plot_dict[exp_dir]['num_pcs'])

        thr_metrics_df = pd.read_csv(exp_dir / 'metrics_thr_-1.0-1.0.csv')

        prec_avg.append(thr_metrics_df.loc[thr_metrics_df['thr'] == thr, f'precision_{avg_type}_mean'].values[0])
        prec_var.append(thr_metrics_df.loc[thr_metrics_df['thr'] == thr, f'precision_{avg_type}_std'].values[0])
        rec_avg.append(thr_metrics_df.loc[thr_metrics_df['thr'] == thr, f'recall_{avg_type}_mean'].values[0])
        rec_var.append(thr_metrics_df.loc[thr_metrics_df['thr'] == thr, f'recall_{avg_type}_std'].values[0])

    # f, ax = plt.subplots(1, 2, figsize=(14, 6))
    # ax[0].errorbar(n_pcs_arr, prec_avg, yerr=prec_var, fmt='o')
    # ax[0].set_xlabel('K', fontsize=20)
    # ax[0].set_ylabel('Precision', fontsize=20)
    # ax[0].set_xticks(n_pcs_arr)
    # ax[0].grid(True)
    # ax[0].tick_params(axis='both', which='major', labelsize=18)
    # ax[1].errorbar(n_pcs_arr, rec_avg, yerr=rec_var, fmt='o')
    # ax[1].set_xlabel('K', fontsize=20)
    # ax[1].set_ylabel('Recall', fontsize=20)
    # ax[1].set_xticks(n_pcs_arr)
    # ax[1].grid(True)
    # ax[1].tick_params(axis='both', which='major', labelsize=18)
    # # f.suptitle(f'Threshold={thr_arr[thr_idx]:.2f}')
    # f.tight_layout()
    # f.savefig(save_dir / f'precision_and_recall_vs_num_pcs_thr_{thr}_avg_{avg_type}_allebrcexperiments.pdf')

    f, ax = plt.subplots(figsize=(12, 6))
    ax.errorbar(n_pcs_arr, prec_avg, yerr=prec_var, fmt='o', label='Precision', linewidth=3.0)
    ax.set_xlabel('K', fontsize=24)
    ax.set_ylabel('Metric Value', fontsize=24)
    ax.set_xticks(n_pcs_arr)
    ax.set_xticklabels(n_pcs_arr, rotation=0)
    ax.grid(axis='y')
    ax.tick_params(axis='both', which='major', labelsize=22)
    ax.errorbar(n_pcs_arr, rec_avg, yerr=rec_var, fmt='o', label='Recall', linewidth=3.0)
    ax.legend(fontsize=22)
    f.tight_layout()
    f.savefig(save_dir / f'precision_and_recall_vs_num_pcs_thr_{thr}_avg_{avg_type}_allebrcexperiments_{agg_pcs_eval_type}_sameplot.pdf')

#%%
for avg_type in avg_type_arr:

    f, ax = plt.subplots(figsize=(10, 7))

    for agg_pcs_eval_type in agg_pcs_eval_types:
        prec_avg, rec_avg = [], []
        prec_var, rec_var = [], []
        n_pcs_arr = []

        for exp_dir in exp_dirs_plot_dict:
            n_pcs_arr.append(exp_dirs_plot_dict[exp_dir]['num_pcs'])

            thr_metrics_df = pd.read_csv(exp_dir / agg_pcs_eval_type / 'metrics_thr_-1.0-1.0.csv')

            prec_avg.append(thr_metrics_df.loc[thr_metrics_df['thr'] == thr, f'precision_{avg_type}_mean'].values[0])
            prec_var.append(thr_metrics_df.loc[thr_metrics_df['thr'] == thr, f'precision_{avg_type}_std'].values[0])
            rec_avg.append(thr_metrics_df.loc[thr_metrics_df['thr'] == thr, f'recall_{avg_type}_mean'].values[0])
            rec_var.append(thr_metrics_df.loc[thr_metrics_df['thr'] == thr, f'recall_{avg_type}_std'].values[0])

        idxs_sorted = np.argsort(n_pcs_arr)
        n_pcs_arr = np.array(n_pcs_arr)[idxs_sorted]
        prec_var = np.array(prec_var)[idxs_sorted]
        rec_avg, rec_var = np.array(rec_avg)[idxs_sorted], np.array(rec_var)[idxs_sorted]
        prec_avg, prec_var = np.array(prec_avg)[idxs_sorted], np.array(prec_var)[idxs_sorted]

        ax.errorbar(n_pcs_arr, prec_avg, yerr=prec_var, fmt=agg_pcs_eval_types[agg_pcs_eval_type]['fmt'], label=f'Precision {agg_pcs_eval_types[agg_pcs_eval_type]["label"]}', linewidth=3.0, color=agg_pcs_eval_types[agg_pcs_eval_type]['color'][0], markersize=10)
        ax.plot(n_pcs_arr, prec_avg, linestyle='dashed', color=agg_pcs_eval_types[agg_pcs_eval_type]['color'][0])
        ax.errorbar(n_pcs_arr, rec_avg, yerr=rec_var, fmt=agg_pcs_eval_types[agg_pcs_eval_type]['fmt'], label=f'Recall {agg_pcs_eval_types[agg_pcs_eval_type]["label"]}', linewidth=3.0, color=agg_pcs_eval_types[agg_pcs_eval_type]['color'][1], markersize=10)
        ax.plot(n_pcs_arr, rec_avg, color=agg_pcs_eval_types[agg_pcs_eval_type]['color'][1], linestyle='dashed')

    ax.set_xlabel('K', fontsize=24)
    ax.set_ylabel('Metric Value', fontsize=24)
    ax.set_xlim([0, 16])
    ax.grid(axis='y')
    ax.tick_params(axis='both', which='major', labelsize=22)
    ax.set_xticks(n_pcs_arr)
    ax.set_yticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    ax.set_ylim([0.3, 0.95])
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 1.075, box.height * 0.9])
    # Put a legend to the right of the current axis
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=3, fontsize=18)

    # f.tight_layout()
    f.savefig(save_dir / f'precision_and_recall_vs_num_pcs_thr_{thr}_avg_{avg_type}_allebrcexperiments_all_agg_pcs_eval_types_sameplot.pdf')

#%% precision and recall vs model PC threshold

save_dir = Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/experiments/explainability/model_pc_replacement/thr_analysis')
agg_pcs_eval_types = {
    'median_eval': {'label': 'Med', 'fmt': 'o', 'color': ['b', 'orange']},
    'max_eval': {'label': 'Max', 'fmt': '^', 'color': ['g', 'm']},
    'min_eval': {'label': 'Min', 'fmt': '*', 'color': ['k', 'c']},
}

# agg_pcs_eval_type = 'median_eval'

exp_dirs = [fp for fp in save_dir.iterdir() if fp.name.startswith('run_num_model_pcs')]
exp_dirs_plot_dict = {fp: {'thr_modelpcs': float(fp.name.split('_')[6])} for fp in exp_dirs}

avg_type_arr = ['micro', 'macro', 'weighted', 'samples']

thr = 0.5
for avg_type in avg_type_arr:
    f, ax = plt.subplots(figsize=(12, 6))
    ax2 = ax.twinx()
    for agg_pcs_eval_type in agg_pcs_eval_types:
        prec_avg, rec_avg = [], []
        prec_var, rec_var = [], []
        modelpcs_thr_arr = []

        for exp_dir in exp_dirs_plot_dict:
            modelpcs_thr_arr.append(exp_dirs_plot_dict[exp_dir]['thr_modelpcs'])

            thr_metrics_df = pd.read_csv(exp_dir / agg_pcs_eval_type / 'metrics_thr_-1.0-1.0.csv')

            prec_avg.append(thr_metrics_df.loc[thr_metrics_df['thr'] == thr, f'precision_{avg_type}_mean'].values[0])
            prec_var.append(thr_metrics_df.loc[thr_metrics_df['thr'] == thr, f'precision_{avg_type}_std'].values[0])
            rec_avg.append(thr_metrics_df.loc[thr_metrics_df['thr'] == thr, f'recall_{avg_type}_mean'].values[0])
            rec_var.append(thr_metrics_df.loc[thr_metrics_df['thr'] == thr, f'recall_{avg_type}_std'].values[0])

        idxs_sorted = np.argsort(modelpcs_thr_arr)
        prec_var = np.array(prec_var)[idxs_sorted]
        rec_avg, rec_var = np.array(rec_avg)[idxs_sorted], np.array(rec_var)[idxs_sorted]
        prec_avg, prec_var = np.array(prec_avg)[idxs_sorted], np.array(prec_var)[idxs_sorted]

        # ax.errorbar(np.arange(len(modelpcs_thr_arr)), prec_avg, yerr=prec_var, fmt='o', label='Precision', linewidth=3.0)
        ax.errorbar(modelpcs_thr_arr, prec_avg, yerr=prec_var, fmt=agg_pcs_eval_types[agg_pcs_eval_type]['fmt'], label=f'Precision {agg_pcs_eval_types[agg_pcs_eval_type]["label"]}', linewidth=3.0, color='b', markersize=10)
        # ax.errorbar(np.arange(len(modelpcs_thr_arr)), rec_avg, yerr=rec_var, fmt='o', label='Recall', linewidth=3.0)
        ax2.errorbar(modelpcs_thr_arr, rec_avg, yerr=rec_var, fmt=agg_pcs_eval_types[agg_pcs_eval_type]['fmt'], label=f'Recall {agg_pcs_eval_types[agg_pcs_eval_type]["label"]}', linewidth=3.0, color='orange', markersize=10)

    ax.set_xlabel('T', fontsize=24)
    ax.set_ylabel('Precision', fontsize=24, color='b')
    ax2.set_ylabel('Recall', fontsize=24, color='orange')
    # ax.set_xticks(modelpcs_thr_arr)
    ax.set_xlim([0.45, 1])
    # ax.set_xticks(np.arange(len(modelpcs_thr_arr)))
    # ax.set_xticklabels(modelpcs_thr_arr, rotation=0)
    ax.grid(axis='y')
    ax.tick_params(axis='both', which='major', labelsize=22)
    ax.legend(fontsize=22)
    ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1])
    ax.set_yticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    f.tight_layout()
    f.savefig(save_dir / f'precision_and_recall_vs_thrmodelpcs_thr_{thr}_avg_{avg_type}_allebrcexperiments_all_agg_pcs_eval_types_sameplot.pdf')

#%%

for avg_type in avg_type_arr:
    f, ax = plt.subplots(figsize=(10, 7))
    # ax2 = ax.twinx()
    for agg_pcs_eval_type in agg_pcs_eval_types:
        prec_avg, rec_avg = [], []
        prec_var, rec_var = [], []
        modelpcs_thr_arr = []

        for exp_dir in exp_dirs_plot_dict:
            modelpcs_thr_arr.append(exp_dirs_plot_dict[exp_dir]['thr_modelpcs'])

            thr_metrics_df = pd.read_csv(exp_dir / agg_pcs_eval_type / 'metrics_thr_-1.0-1.0.csv')

            prec_avg.append(thr_metrics_df.loc[thr_metrics_df['thr'] == thr, f'precision_{avg_type}_mean'].values[0])
            prec_var.append(thr_metrics_df.loc[thr_metrics_df['thr'] == thr, f'precision_{avg_type}_std'].values[0])
            rec_avg.append(thr_metrics_df.loc[thr_metrics_df['thr'] == thr, f'recall_{avg_type}_mean'].values[0])
            rec_var.append(thr_metrics_df.loc[thr_metrics_df['thr'] == thr, f'recall_{avg_type}_std'].values[0])

        idxs_sorted = np.argsort(modelpcs_thr_arr)
        modelpcs_thr_arr = np.array(modelpcs_thr_arr)[idxs_sorted]
        rec_avg, rec_var = np.array(rec_avg)[idxs_sorted], np.array(rec_var)[idxs_sorted]
        prec_avg, prec_var = np.array(prec_avg)[idxs_sorted], np.array(prec_var)[idxs_sorted]

        # ax.errorbar(np.arange(len(modelpcs_thr_arr)), prec_avg, yerr=prec_var, fmt='o', label='Precision', linewidth=3.0)
        # ax.errorbar(modelpcs_thr_arr, prec_avg, yerr=prec_var, fmt=agg_pcs_eval_types[agg_pcs_eval_type]['fmt'], label=f'Precision {agg_pcs_eval_types[agg_pcs_eval_type]["label"]}', linewidth=3.0, color='b', markersize=10)
        ax.errorbar(modelpcs_thr_arr, prec_avg, yerr=prec_var, fmt=agg_pcs_eval_types[agg_pcs_eval_type]['fmt'], label=f'Precision {agg_pcs_eval_types[agg_pcs_eval_type]["label"]}', linewidth=3.0, color=agg_pcs_eval_types[agg_pcs_eval_type]['color'][0], markersize=10)
        ax.plot(modelpcs_thr_arr, prec_avg, linestyle='dashed', color=agg_pcs_eval_types[agg_pcs_eval_type]['color'][0])
        # ax.errorbar(np.arange(len(modelpcs_thr_arr)), rec_avg, yerr=rec_var, fmt='o', label='Recall', linewidth=3.0)
        # ax.errorbar(modelpcs_thr_arr, rec_avg, yerr=rec_var, fmt=agg_pcs_eval_types[agg_pcs_eval_type]['fmt'], label=f'Recall {agg_pcs_eval_types[agg_pcs_eval_type]["label"]}', linewidth=3.0, color='orange', markersize=10)
        ax.errorbar(modelpcs_thr_arr, rec_avg, yerr=rec_var, fmt=agg_pcs_eval_types[agg_pcs_eval_type]['fmt'], label=f'Recall {agg_pcs_eval_types[agg_pcs_eval_type]["label"]}', linewidth=3.0, color=agg_pcs_eval_types[agg_pcs_eval_type]['color'][1], markersize=10)
        ax.plot(modelpcs_thr_arr, rec_avg, color=agg_pcs_eval_types[agg_pcs_eval_type]['color'][1], linestyle='dashed')

    ax.set_xlabel('T', fontsize=24)
    # ax.set_ylabel('Metric Value', fontsize=24, color='b')
    ax.set_ylabel('Metric Value', fontsize=24)
    # ax2.set_ylabel('Recall', fontsize=24, color='orange')
    # ax.set_xticks(modelpcs_thr_arr)
    ax.set_xlim([0.45, 1])
    # ax.set_xticks(np.arange(len(modelpcs_thr_arr)))
    # ax.set_xticklabels(modelpcs_thr_arr, rotation=0)
    ax.grid(axis='y')
    ax.tick_params(axis='both', which='major', labelsize=22)
    # ax.legend(fontsize=22)
    ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1])
    ax.set_yticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    ax.set_ylim([0.3, 1])
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 1.075, box.height * 0.9])

    # Put a legend to the right of the current axis
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=3, fontsize=18)

    # f.tight_layout()
    # f.subplots_adjust(bottom=0.1)
    f.savefig(save_dir / f'precision_and_recall_vs_thrmodelpcs_thr_{thr}_avg_{avg_type}_allebrcexperiments_all_agg_pcs_eval_types_sameplot.pdf')

#%% plot distribution of ExoMiner original scores

save_dir = Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/experiments/explainability/')

train_set_scores = pd.read_csv('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/experiments/explainability/shap1/shap1_1-10-2022/results_ensemble/keplerdr25-dv_g301-l31_5tr_spline_nongapped_configK_explainability_allfeatures_1-6-2022/ensemble_ranked_predictions_trainset.csv')

bins = np.linspace(0, 1, 100, endpoint=True)
f, ax = plt.subplots()
ax.hist(train_set_scores['score'], bins, edgecolor='k')
ax.set_xlabel('ExoMiner Score')
ax.set_ylabel('Counts')
ax.set_yscale('log')
ax.set_xticks(np.linspace(0, 1, 11, endpoint=True))
ax.set_xlim(bins[[0, -1]])
f.savefig(save_dir / 'hist_exominer_scores_trainset.pdf')

f, ax = plt.subplots()
ax.hist(train_set_scores.loc[train_set_scores['original_label'] == 'PC', 'score'], bins, edgecolor='k', label='PC', zorder=3, alpha=0.5)
ax.hist(train_set_scores.loc[train_set_scores['original_label'] == 'AFP', 'score'], bins, edgecolor='k', label='AFP', zorder=2, alpha=0.8)
ax.hist(train_set_scores.loc[train_set_scores['original_label'] == 'NTP', 'score'], bins, edgecolor='k', label='NTP', zorder=1)
ax.legend()
ax.set_xlabel('ExoMiner Score')
ax.set_ylabel('Counts')
ax.set_xticks(np.linspace(0, 1, 11, endpoint=True))
ax.set_yscale('log')
ax.set_xlim(bins[[0, -1]])
f.savefig(save_dir / 'hist_exominer_scores_PC-AFP-NTP_trainset.pdf')
#%% compute pr auc for all experiments overall fp flags

save_dir = Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/experiments/explainability/')
thr_metrics_df = pd.read_csv(save_dir / 'metrics_thr_-1.0-1.0_allexperiments.csv')

avg_types = [
    'macro',
    'micro',
    'weighted',
    'samples',
]

exps = [
    'random_analysis',
    'occlusion',
    'shap',
    'shap_1',
    'ebrc_15pcs_max',
    'ebrc_15pcs_median',
    'ebrc_15pcs_min',
]

pr_auc_dict = {'exps': exps}
pr_auc_dict.update({f'pr_auc_{avg_type}': np.nan * np.ones(len(exps)) for avg_type in avg_types})

for avg_type in avg_types:
    for exp_i, exp in enumerate(exps):
        if np.any([el in exp for el in ['random_analysis', 'ebrc']]):
            pr_auc_dict[f'pr_auc_{avg_type}'][exp_i] = sk_auc(thr_metrics_df[f'{exp}_recall_{avg_type}_mean'],
                                                              thr_metrics_df[f'{exp}_precision_{avg_type}_mean'])
        else:
            pr_auc_dict[f'pr_auc_{avg_type}'][exp_i] = sk_auc(thr_metrics_df[f'{exp}_recall_{avg_type}'], thr_metrics_df[f'{exp}_precision_{avg_type}'])

pr_auc_df = pd.DataFrame(pr_auc_dict)
pr_auc_df.to_csv(save_dir / 'pr_auc_allexperiments.csv', index=False)

#%% computte pr auc exoplanet class for all experiments

thr_metrics_df = pd.read_csv(save_dir / 'metrics_thr_-1.0-1.0_noflag_allexperiments.csv')
# thr_metrics_df = thr_metrics_df.loc[thr_metrics_df['thr'] > 0]
exps = [
    'random_analysis',
    'occlusion',
    'shap',
    'shap_1',
    'ebrc_15pcs_max',
    'ebrc_15pcs_median',
    'ebrc_15pcs_min',
]

pr_auc_dict = {'exps': exps, 'pr_auc': np.nan * np.ones(len(exps))}
for exp_i, exp in enumerate(exps):
    if np.any([el in exp for el in ['random_analysis', 'ebrc']]):
        pr_auc_dict[f'pr_auc'][exp_i] = sk_auc(thr_metrics_df[f'{exp}_recall_mean'],
                                                          thr_metrics_df[f'{exp}_precision_mean'])
    else:
        pr_auc_dict[f'pr_auc'][exp_i] = sk_auc(thr_metrics_df[f'{exp}_recall'], thr_metrics_df[f'{exp}_precision'])

pr_auc_df = pd.DataFrame(pr_auc_dict)
pr_auc_df.to_csv(save_dir / 'pr_auc_noflag_allexperiments.csv', index=False)

#%% compute pr auc for all experiments fp flags

save_dir = Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/experiments/explainability/')
thr_metrics_df = pd.read_csv(save_dir / 'precision_recall_per_flag_thr_-1.0-1.0_allexperiments.csv')

exps = [
    'random_analysis',
    'occlusion',
    'shap',
    'shap_1',
    'ebrc_15pcs_max',
    'ebrc_15pcs_median',
    'ebrc_15pcs_min',
]

flags = [
    'DV Flag',
    'Flux Flag',
    'Centroid Flag',
    'Odd Even Flag',
    'Secondary Flag',
]

pr_auc_dict = {'exps': exps}
pr_auc_dict.update({f'pr_auc_{flag}': np.nan * np.ones(len(exps)) for flag in flags})

for exp_i, exp in enumerate(exps):
    thr_metrics_df_exp = thr_metrics_df.loc[thr_metrics_df['experiment_name'] == exp]
    for flag in flags:
        pr_auc_dict[f'pr_auc_{flag}'][exp_i] = sk_auc(thr_metrics_df_exp[f'recall_{flag}'], thr_metrics_df_exp[f'precision_{flag}'])

pr_auc_df = pd.DataFrame(pr_auc_dict)
pr_auc_df.to_csv(save_dir / 'pr_auc_perflag_allexperiments.csv', index=False)

exp_dir_plot_params = {
    'random_analysis': {'name': 'Random', 'color': 'b', 'linestyle': '-'},
    'shap': {'name': 'SHAP', 'color': 'r', 'linestyle': '--'},
    'shap_1': {'name': 'SHAP1', 'color': 'g', 'linestyle': '-.'},
    'occlusion': {'name': 'Occlusion', 'color': 'k', 'linestyle': '-'},
    'ebrc_15pcs_min': {'name': 'NOREX Min', 'color': 'c', 'linestyle': '-.'},
    'ebrc_15pcs_median': {'name': 'NOREX Med', 'color': 'y', 'linestyle': '-.'},
    'ebrc_15pcs_max': {'name': 'NOREX Max', 'color': 'm', 'linestyle': '-.'},
}

for flag in flags:
    f, ax = plt.subplots(figsize=(10, 6))
    for exp_i, exp in enumerate(exps):
        thr_metrics_df_exp = thr_metrics_df.loc[thr_metrics_df['experiment_name'] == exp]
        ax.plot(thr_metrics_df_exp[f'recall_{flag}'], thr_metrics_df_exp[f'precision_{flag}'], color=exp_dir_plot_params[exp]['color'],
                linestyle=exp_dir_plot_params[exp]['linestyle'],
                label=exp_dir_plot_params[exp]['name'], linewidth=3)
        ax.scatter(thr_metrics_df_exp[f'recall_{flag}'], thr_metrics_df_exp[f'precision_{flag}'], color=exp_dir_plot_params[exp]['color'], s=8)
    ax.legend(fontsize=16)
    ax.grid()
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.set_ylabel('Precision', fontsize=20)
    ax.set_xlabel('Recall', fontsize=20)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])

    f.tight_layout()
    f.savefig(save_dir / f'pr_curves_{flag}_allexperiments.pdf')