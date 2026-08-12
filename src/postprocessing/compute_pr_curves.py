"""Compute precision and recall curves for several experiments.
"""

#%% imports

# 3rd party
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from pathlib import Path
import pandas as pd


def compute_precision_at_recall(recall_arr, score_col, label_id_col, num_thr, exp_name=''):
    
    prec_arr = np.nan * np.ones(len(recall_arr)) 
    for recall_val_i, recall_val in enumerate(recall_arr):
        prec_at_rec = tf.keras.metrics.PrecisionAtRecall(recall_val, num_thr, name=f'precision_at_recall_{exp_name}')
        _ = prec_at_rec.update_state(exp_pred_tbl[label_id_col].to_list(), exp_pred_tbl[score_col].to_list())
        prec_arr[recall_val_i] = prec_at_rec.result().numpy()
    
    return prec_arr


def plot_pr_curve(prcurve, save_fp):
    
    f, ax = plt.subplots()
    for exp_name, pr_dict in prcurve.items():
        ax.plot(pr_dict['recall'], pr_dict['precision'], label=exp_name)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_ylim([0.6, 1])
    ax.set_xlim([0.9, 1])
    ax.set_xticks(np.linspace(0.9, 1, 11))
    ax.grid(axis='x')
    ax.legend()
    f.savefig(save_fp)
    
    
if  __name__ == "__main__":
    #%% 

    save_fp = Path('/u/msaragoc/work_dir/Kepler-TESS_exoplanet/experiments/aux_labels/pr-curves_experiments.png')


    pred_tbls_fps = {
        'ExoMiner++ wo/ aux loss': '/u/msaragoc/work_dir/Kepler-TESS_exoplanet/experiments/tess_spoc_ffi_paper/aux_labels/cv_tfrecords_tess-spoc-tces_2min-s1-s88_ffi-s36-s72-s56s69_exominerpp_2-4-2026_1626/predictions_testset_allfolds.csv',
        'ExoMiner++ w/ aux loss (v1)': '/u/msaragoc/work_dir/Kepler-TESS_exoplanet/experiments/tess_spoc_ffi_paper/aux_labels/cv_tfrecords_tess-spoc-tces_2min-s1-s88_ffi-s36-s72-s56s69_exominerpp_aux-bg-0.2_2-3-2026_1150/predictions_testset_allfolds.csv',
        # 'ExoMiner++ w/ aux loss (v2)': '/u/msaragoc/work_dir/Kepler-TESS_exoplanet/experiments/tess_spoc_ffi_paper/aux_labels/cv_tfrecords_tess-spoc-tces_2min-s1-s88_ffi-s36-s72-s56s69_exominerpp_aux-bg-0.2_satexcluded_2-6-2026_1600/predictions_testset_allfolds.csv',
        # 'ExoMiner++ w/ aux loss (v3)': '/u/msaragoc/work_dir/Kepler-TESS_exoplanet/experiments/tess_spoc_ffi_paper/aux_labels/cv_tfrecords_tess-spoc-tces_2min-s1-s88_ffi-s36-s72-s56s69_exominerpp_aux-bg-0.2_evidence_2-9-2026_1101/predictions_testset_allfolds.csv',
        # 'ExoMiner++ w/ aux loss (v4)': '/u/msaragoc/work_dir/Kepler-TESS_exoplanet/experiments/tess_spoc_ffi_paper/aux_labels/cv_tfrecords_tess-spoc-tces_2min-s1-s88_ffi-s36-s72-s56s69_exominerpp_aux-bg-0.2_evidence-qmetric-nodisp_2-10-2026_1702/predictions_testset_allfolds.csv',
    }

    pred_tbls = {exp_name: pd.read_csv(fp, comment='#') for exp_name, fp in pred_tbls_fps.items()}

    # label_map = {
    #     'KP': 1,
    #     'CP': 1,
    #     'EB': 0,
    #     'FP': 0,
    #     'BD': 0,
    #     'NTP': 0,
    # }
    # label_col = 'label'
    # for exp_name, pred_tbl in pred_tbls.items():
    #     pred_tbls[exp_name]['label'] = pred_tbl[label_col].map(label_map)

    # for exp_name, pred_tbl in pred_tbls.items():
    #     pred_tbls[exp_name] = pred_tbl.loc[pred_tbl[label_col] != 'UNK']

    score_col = 'score'
    label_id_col = 'label_id'
    recall_arr = np.linspace(0, 1, 101, endpoint=True).tolist()
    num_thr = 1000

    prcurve = {exp_name: {'precision': np.nan * np.ones(len(recall_arr)), 'recall': recall_arr} for exp_name in pred_tbls}


    for exp_name, exp_pred_tbl in pred_tbls.items():

        prcurve[exp_name]['precision'] = compute_precision_at_recall(recall_arr, score_col, label_id_col, num_thr, exp_name=exp_name)


    #%%

    plot_pr_curve(prcurve, save_fp)

    # %%
