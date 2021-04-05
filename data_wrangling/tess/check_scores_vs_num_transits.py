"""
Check score of model with respect to different parameters such as number of transits available, avg out-of-transit, ...
"""

from pathlib import Path
# import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#%% Draft to extract data from TFRecords, but not used

tfrec_dir = Path('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Data/tfrecords/TESS/plot_misclassified_KPs-CPs_TOIs')

tfrec_files = [fp for fp in tfrec_dir.iterdir() if fp.stem.startswith('shard') and fp.suffix != '.csv']

features_tfrec = [
    'flux'
    'flux_odd',
    'flux_even',
    'wks',
    'centroid',
    'centroid_fdl',
]
for tfrec_file in tfrec_files:

    # iterate through the source shard
    tfrecord_dataset = tf.data.TFRecordDataset(tfrec_file)

    for string_record in tfrecord_dataset.as_numpy_iterator():

        example = tf.train.Example()
        example.ParseFromString(string_record)

        toi = example.features.feature['oi'].int64_list.value[0]
        tic = example.features.feature['target_id'].int64_list.value[0]
#%%

tfrec_dir = Path('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Data/tfrecords/TESS/plot_misclassified_KPs-CPs_TOIs')
shards_tbl = pd.concat([pd.read_csv(fp) for fp in tfrec_dir.iterdir()
                        if fp.stem.startswith('shard') and fp.suffix == '.csv'])

ranking_tbl_cols = [
    'tce_period',
    'tce_duration',
    'tce_time0bk',
    'transit_depth',
    'Signal-to-noise',
    'original_label',
    'tfopwg_disp',
    'score',
    'predicted class',
    'matching_dist'
]
ranking_tbl_fp = Path('/data5/tess_project/git_repo/results_ensemble/experiments_tess(3-4-2021)/tess_g301-l31_6tr_spline_nongapped_spoctois_configK_wsphase/ensemble_ranked_predictions_predictset_tfopwg_disp_matchingdist.csv')
ranking_tbl = pd.read_csv(ranking_tbl_fp)
ranking_tbl = ranking_tbl.loc[(ranking_tbl['tfopwg_disp'].isin(['KP', 'CP'])) & (ranking_tbl['predicted class'] == 0)]

for toi_i, toi in shards_tbl.iterrows():

    shards_tbl.loc[toi_i, ranking_tbl_cols] = ranking_tbl.loc[ranking_tbl['oi'] == toi['oi'], ranking_tbl_cols].values

shards_tbl.to_csv(tfrec_dir / 'misclassified_KP_CP_shards_tbl.csv', index=False)

#%% check scores vs num_transits

bins_score = np.linspace(0, 1, 21, True)
bins_transits = np.linspace(0, 100, 11, True)
for num_tr in [
    'flux',
    'flux_odd',
    'flux_even',
    'wks',
    'centroid',
    'centroid_fdl',]:

    f, ax = plt.subplots()
    ax.scatter(shards_tbl['score'], shards_tbl[num_tr], s=10, edgecolors='k')
    ax.set_xlabel('Score')
    ax.set_ylabel('Number of transits')
    ax.set_title(f'{num_tr}')
    f.savefig(ranking_tbl_fp.parent / f'misclassified_KP_CP_score-num_transits_{num_tr}_scatter.png')

    f, ax = plt.subplots()
    h = ax.hist2d(shards_tbl['score'], shards_tbl[num_tr], bins=(bins_score, bins_transits), range=[(0, 1), None])
    f.colorbar(h[3], ax=ax)
    ax.set_xlabel('Score')
    ax.set_ylabel('Number of transits')
    ax.set_title(f'{num_tr}')
    f.savefig(ranking_tbl_fp.parent / f'misclassified_KP_CP_score-num_transits_{num_tr}_hist2d.png')

    # aaaa

#%% check scores vs other parameters

# bins_score = np.linspace(0, 1, 21, True)
# bins_transits = np.linspace(0, 100, 11, True)
for param in [
    'avg_oot_centroid',
    'peak_centroid',
    'transit_depth',
    'Signal-to-noise',
    'matching_dist',
    ]:

    f, ax = plt.subplots()
    ax.scatter(shards_tbl['score'], shards_tbl[param], s=10, edgecolors='k')
    ax.set_xlabel('Score')
    ax.set_ylabel(f'{param}')
    ax.set_title(f'{param}')
    f.savefig(ranking_tbl_fp.parent / f'misclassified_KP_CP_score-{param}_scatter.png')

    # f, ax = plt.subplots()
    # h = ax.hist2d(shards_tbl['score'], shards_tbl[num_tr], bins=(bins_score, bins_transits), range=[(0, 1), None])
    # f.colorbar(h[3], ax=ax)
    # ax.set_xlabel('Score')
    # ax.set_ylabel('Number of transits')
    # ax.set_title(f'{num_tr}')
    # f.savefig(ranking_tbl_fp.parent / f'misclassified_KP_CP_score-num_transits_{num_tr}_scatter.png')
