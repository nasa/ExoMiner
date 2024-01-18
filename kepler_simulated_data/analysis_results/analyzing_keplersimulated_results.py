"""

"""

# 3rd party
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow.keras.metrics import AUC, Precision, Recall
import shutil
from matplotlib.colors import LogNorm


#%% Plot score distribution for scrambling groups

predictions_tbl_fp = Path('/Users/msaragoc/Projects/exoplanet_transit_classification/experiments/kepler_simulated_data_exominer/exominer_train_realdata_10-13-2023_1530/cv_iter_0/ensemble_ranked_predictions_testset.csv')
predictions_tbl = pd.read_csv(predictions_tbl_fp)
save_dir = predictions_tbl_fp.parent  # Path('/Users/msaragoc/Projects/exoplanet_transit_classification/experiments/kepler_simulated_data_exominer/')

scrambling_preds = predictions_tbl.loc[predictions_tbl['label'].isin(['PC', 'AFP', 'NTP'])]

bins = np.linspace(0, 1, 11)

f, ax = plt.subplots()
_ = ax.hist(scrambling_preds.loc[scrambling_preds['label'] == 'NTP', 'score'], bins, edgecolor='k', label='NTP', zorder=1, alpha=1, align='mid')
ax.hist(scrambling_preds.loc[scrambling_preds['label'] == 'AFP', 'score'], bins, edgecolor='k', label='AFP', zorder=2, alpha=0.7, align='mid')
ax.hist(scrambling_preds.loc[scrambling_preds['label'] == 'PC', 'score'], bins, edgecolor='k', label='PC', zorder=3, alpha=0.5, align='mid')
ax.set_xlabel('Model Score')
ax.set_xticks(bins)
# ax.set_xticklabels(bins)
ax.legend()
ax.set_xlim(bins[[0, -1]])
ax.set_yscale('log')
ax.set_ylabel('Counts')
f.savefig(save_dir / 'hist_score_distribution.png')

#%% Compute precision-recall curve

predictions_tbl = pd.read_csv('/Users/msaragoc/Projects/exoplanet_transit_classification/experiments/kepler_simulated_data_exominer/exominer_trained_realdata_predict_simdata_11-6-2023_0931/models/model0/ranked_predictions_alldatasets.csv')
save_dir = Path('/Users/msaragoc/Projects/exoplanet_transit_classification/experiments/kepler_simulated_data_exominer/')

n_thr = 1000
thr_arr = list(np.linspace(0, 1, 1000))
precision = Precision(name='precision', thresholds=thr_arr)
recall = Recall(name='recall', thresholds=thr_arr)

_ = precision.update_state(predictions_tbl['label_id'], predictions_tbl['score'])
precision_arr = precision.result().numpy()
_ = recall.update_state(predictions_tbl['label_id'], predictions_tbl['score'])
recall_arr = recall.result().numpy()

auc_pr = AUC(num_thresholds=n_thr,
             summation_method='interpolation',
             curve='PR',
             name='auc_pr')
_ = auc_pr.update_state(predictions_tbl['label_id'], predictions_tbl['score'])
auc_pr = auc_pr.result().numpy()

#%% Plot PR curve and precision/recall vs classification threshold

xticks = np.linspace(0, 1, 11)

f, ax = plt.subplots()
ax.plot(recall_arr, precision_arr)
ax.scatter(recall_arr, precision_arr, s=8, c='k')
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_ylim([-0.01, 1.01])
ax.set_xlim([-0.01, 1.01])
ax.set_xticks(xticks)
ax.set_yticks(xticks)
ax.set_title(f'PR AUC={auc_pr:.3f}')
ax.grid()
f.savefig(save_dir / 'pr_curve.png')

f, ax = plt.subplots()
ax.plot(thr_arr, precision_arr, label='Precision')
ax.plot(thr_arr, recall_arr, label='Recall')
ax.set_xlabel('Threshold')
ax.set_xlim([thr_arr[0] - 0.01, thr_arr[-1] + 0.01])
ax.set_ylabel('Value')
ax.set_ylim([0, 1])
ax.legend()
ax.grid()
ax.set_xticks(xticks)
ax.set_yticks(xticks)
f.savefig(save_dir / 'plot_vs_precision-recall.png')

#%% Plot recall per period bin

thr_arr = np.arange(0, 1.1, 0.1)
period_bins = np.linspace(0, 500, 41, endpoint=True)
recall_mat = np.nan * np.ones((len(thr_arr), len(period_bins) - 1))

for period_bin_i, (period_bin_s, period_bin_end) in enumerate(zip(period_bins[:-1], period_bins[1:])):
    tbl_bin = predictions_tbl.loc[
        ((predictions_tbl['tce_period'] >= period_bin_s) & (predictions_tbl['tce_period'] < period_bin_end))]
    for thr_i, thr in enumerate(thr_arr):
        recall_bin_val = ((tbl_bin['label_id'] == 1) & (tbl_bin['score'] >= thr)).sum() / (tbl_bin['label_id'] == 1).sum()
        recall_mat[thr_i, period_bin_i] = recall_bin_val

f, ax = plt.subplots(figsize=(12, 8))
im = ax.imshow(recall_mat)  # , norm=LogNorm(vmin=1e-2, vmax=1, clip=True))
ax.set_ylabel('Threshold')
ax.set_yticks(np.arange(len(thr_arr)), np.around(thr_arr, 1))
# ax.set_yticklabels(np.round(thr_arr))
ax.set_xlabel('Period (day)')
ax.set_xticks(np.arange(len(period_bins))[::2] - 0.5, labels=period_bins[::2].astype('int'))
# axcolor = f.add_axes([0.90, 0.1, 0.03, 0.79])
# cbar = f.colorbar(im, ax=ax, cax=None, format="$%.2f$", orientation='vertical', location='right', fraction=0.05, label='Recall', aspect=8, ticks=np.concatenate([np.linspace(1e-2, 9e-2, 9), np.arange(0, 1.1, 0.1)]))
cbar = f.colorbar(im, ax=ax, cax=None, format="$%.2f$", orientation='vertical', location='right', fraction=0.05, label='Recall', aspect=8, ticks=np.arange(0, 1.1, 0.1))
cbar.ax.tick_params(labelsize=10, rotation=0)
f.tight_layout()
f.savefig(save_dir / 'recall_per_thr_vs_period.png')

#%% Plot recall per MES bin

thr_arr = np.arange(0, 1.1, 0.1)
mes_bins = np.linspace(7, 20, 14, endpoint=True)
recall_mat = np.nan * np.ones((len(thr_arr), len(mes_bins) - 1))

for mes_bin_i, (mes_bin_s, mes_bin_end) in enumerate(zip(mes_bins[:-1], mes_bins[1:])):
    tbl_bin = predictions_tbl.loc[
        ((predictions_tbl['tce_max_mult_ev'] >= mes_bin_s) & (predictions_tbl['tce_max_mult_ev'] < mes_bin_end))]
    for thr_i, thr in enumerate(thr_arr):
        recall_bin_val = ((tbl_bin['label_id'] == 1) & (tbl_bin['score'] >= thr)).sum() / (tbl_bin['label_id'] == 1).sum()
        recall_mat[thr_i, mes_bin_i] = recall_bin_val

f, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(recall_mat)  # , norm=LogNorm(vmin=1e-2, vmax=1, clip=True))
ax.set_ylabel('Threshold')
ax.set_yticks(np.arange(len(thr_arr)), np.around(thr_arr, 1))
# ax.set_yticklabels(np.round(thr_arr))
ax.set_xlabel('MES')
ax.set_xticks(np.arange(len(mes_bins)) - 0.5, labels=mes_bins.astype('int'))
# axcolor = f.add_axes([0.90, 0.1, 0.03, 0.79])
# cbar = f.colorbar(im, ax=ax, cax=None, format="$%.2f$", orientation='vertical', location='right', fraction=0.05, label='Recall', aspect=8, ticks=np.concatenate([np.linspace(1e-2, 9e-2, 9), np.arange(0, 1.1, 0.1)]))
cbar = f.colorbar(im, ax=ax, cax=None, format="$%.2f$", orientation='vertical', location='right', fraction=0.08, label='Recall', aspect=8, ticks=np.arange(0, 1.1, 0.1))
cbar.ax.tick_params(labelsize=10, rotation=0)
f.tight_layout()
f.savefig(save_dir / 'recall_per_thr_vs_mes.png')

#%% Plot recall per MES bin

predictions_tbl_fp = Path('/Users/msaragoc/Projects/exoplanet_transit_classification/experiments/kepler_simulated_data_exominer/exominer_trained_realdata_predict_simdata_11-6-2023_0931/models/model0/ranked_predictions_alldatasets.csv')
predictions_tbl = pd.read_csv(predictions_tbl_fp)
predictions_tbl['uid_n'] = predictions_tbl[['uid', 'label']].apply(lambda x: f'{x["uid"]}-{x["label"]}', axis=1)
# predictions_tbl = pd.concat([pd.read_csv(predictions_tbl_fp / f'ranked_predictions_{dataset}set.csv') for dataset in ['train', 'val', 'test']], axis=0)
tce_tbl = pd.read_csv('/Users/msaragoc/Projects/exoplanet_transit_classification/data/ephemeris_tables/kepler/q1-q17_dr25/simulated_data/dvOutputMatrix_allruns_renamed_updtstellar_preprocessed.csv')
tce_tbl['uid_n'] = tce_tbl[['uid', 'label']].apply(lambda x: f'{x["uid"]}-{x["label"]}', axis=1)
predictions_tbl = predictions_tbl.merge(tce_tbl[['uid_n', 'tce_depth']], on='uid_n', how='left', validate='one_to_one')
save_dir = predictions_tbl_fp.parent  # Path('/Users/msaragoc/Projects/exoplanet_transit_classification/experiments/kepler_simulated_data_exominer/')
# aaa

thr_arr = np.arange(0, 1.1, 0.1)
transit_depth_bins = np.array([0, 100, 250, 500, 750, 1000, 2500, 5000, 10000, 25000, 50000, 100000, 1000000])  # np.logspace(0, 5, 10, endpoint=True)
recall_mat = np.nan * np.ones((len(thr_arr), len(transit_depth_bins) - 1))

for depth_bin_i, (depth_bin_s, depth_bin_end) in enumerate(zip(transit_depth_bins[:-1], transit_depth_bins[1:])):
    tbl_bin = predictions_tbl.loc[
        ((predictions_tbl['tce_depth'] >= depth_bin_s) & (predictions_tbl['tce_depth'] < depth_bin_end))]
    for thr_i, thr in enumerate(thr_arr):
        recall_bin_val = ((tbl_bin['label_id'] == 1) & (tbl_bin['score'] >= thr)).sum() / (tbl_bin['label_id'] == 1).sum()
        recall_mat[thr_i, depth_bin_i] = recall_bin_val

f, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(recall_mat)  # , norm=LogNorm(vmin=1e-2, vmax=1, clip=True))
ax.set_ylabel('Threshold')
ax.set_yticks(np.arange(len(thr_arr)), np.around(thr_arr, 1))
# ax.set_yticklabels(np.round(thr_arr))
ax.set_xlabel('Transit Depth (ppm)')
ax.set_xticks(np.arange(len(transit_depth_bins)) - 0.5, labels=transit_depth_bins.astype('int'))
# axcolor = f.add_axes([0.90, 0.1, 0.03, 0.79])
# cbar = f.colorbar(im, ax=ax, cax=None, format="$%.2f$", orientation='vertical', location='right', fraction=0.05, label='Recall', aspect=8, ticks=np.concatenate([np.linspace(1e-2, 9e-2, 9), np.arange(0, 1.1, 0.1)]))
cbar = f.colorbar(im, ax=ax, cax=None, format="$%.2f$", orientation='vertical', location='right', fraction=0.08, label='Recall', aspect=8, ticks=np.arange(0, 1.1, 0.1))
cbar.ax.tick_params(labelsize=10, rotation=0)
f.tight_layout()
f.savefig(save_dir / 'recall_per_thr_vs_transitdepth.png')

#%% Create TFRecord data set by aggregating all shards for the Kepler Simulated data

src_datasets_fps = [
    Path('/Users/msaragoc/Projects/exoplanet_transit_classification/data/tfrecords/kepler/tfrecords_kepler_q1q17dr25_inj1_9-21-2023_1431'),
    Path('/Users/msaragoc/Projects/exoplanet_transit_classification/data/tfrecords/kepler/tfrecords_kepler_q1q17dr25_inj2_9-27-2023_1350'),
    Path('/Users/msaragoc/Projects/exoplanet_transit_classification/data/tfrecords/kepler/tfrecords_kepler_q1q17dr25_inj3_9-25-2023_1127'),
    Path('/Users/msaragoc/Projects/exoplanet_transit_classification/data/tfrecords/kepler/tfrecords_kepler_q1q17dr25_inv_8-18-2023_1344'),
    Path('/Users/msaragoc/Projects/exoplanet_transit_classification/data/tfrecords/kepler/tfrecords_kepler_q1q17dr25_scr2_10-30-2023_1552'),
    Path('/Users/msaragoc/Projects/exoplanet_transit_classification/data/tfrecords/kepler/tfrecords_kepler_q1q17dr25_scr1_10-30-2023_1154'),
]
dest_dir = Path('/Users/msaragoc/Projects/exoplanet_transit_classification/data/tfrecords/kepler/tfrecords_kepler_q1q17dr25_simdata_11-1-2023_1019_aggregated/agg_src_data/')
dest_dir.mkdir(exist_ok=True)

for src_datasets_fp in src_datasets_fps:
    suffix_shard = src_datasets_fp.name.split('_')[3]
    for shard_fp in src_datasets_fp.iterdir():
        shutil.copy(shard_fp, dest_dir / f'{shard_fp.name}_{suffix_shard}')

# get merged_shards files
merged_shards_fps = {
    'inj1': Path('/Users/msaragoc/Projects/exoplanet_transit_classification/data/tfrecords/kepler/tfrecords_kepler_q1q17dr25_inj1_9-21-2023_1431/merged_shards.csv'),
    'inj2': Path('/Users/msaragoc/Projects/exoplanet_transit_classification/data/tfrecords/kepler/tfrecords_kepler_q1q17dr25_inj2_9-27-2023_1350/merged_shards.csv'),
    'inj3': Path('/Users/msaragoc/Projects/exoplanet_transit_classification/data/tfrecords/kepler/tfrecords_kepler_q1q17dr25_inj3_9-25-2023_1127/merged_shards.csv'),
    'scr1': Path('/Users/msaragoc/Projects/exoplanet_transit_classification/data/tfrecords/kepler/tfrecords_kepler_q1q17dr25_scr1_10-30-2023_1154/merged_shards.csv'),
    'scr2': Path('/Users/msaragoc/Projects/exoplanet_transit_classification/data/tfrecords/kepler/tfrecords_kepler_q1q17dr25_scr2_10-30-2023_1552/merged_shards.csv'),
    'inv': Path('/Users/msaragoc/Projects/exoplanet_transit_classification/data/tfrecords/kepler/tfrecords_kepler_q1q17dr25_inv_9-13-2023_1122/merged_shards.csv')
}

merged_shards_tbls = []
for grp_dataset, fp in merged_shards_fps.items():
    merged_shards_grp_tbl = pd.read_csv(fp)
    merged_shards_grp_tbl['shard'] = merged_shards_grp_tbl['shard'].apply(lambda x: f'{x}_{grp_dataset}')
    merged_shards_tbls.append(merged_shards_grp_tbl)
merged_shards_tbl_all_grps = pd.concat(merged_shards_tbls, axis=0)
merged_shards_tbl_all_grps.to_csv(dest_dir / 'merged_shards.csv', index=False)
