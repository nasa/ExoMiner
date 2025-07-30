"""
Create a TFRecord dataset to train ExoMiner difference image.
"""

# 3rd party
import pandas as pd
from pathlib import Path
import tensorflow as tf
import numpy as np
import multiprocessing

# local
from src_preprocessing.tf_util import example_util

tf.config.set_visible_devices([], 'GPU')

#%% set paths

# set TFRecord source directory
src_tfrec_dir = Path('/nobackupp19/msaragoc/work_dir/Kepler-TESS_exoplanet/data/tfrecords/TESS/tfrecords_tess_spoc_2min_s1-s88_4-25-2025_1536_data/tfrecords_tess_spoc_2min_s1-s88_4-25-2025_1536_agg_bdslabels_fixeduids_diffimg')
# destination TFRecord directory
dest_tfrec_dir = src_tfrec_dir.parent / f'{src_tfrec_dir.name}_transit_source_offset_flag'
dest_tfrec_dir.mkdir(exist_ok=True)
# read shards table
shards_tbl = pd.read_csv(src_tfrec_dir / 'shards_tbl.csv')
shards_tbl['example_i_tfrec'] = shards_tbl['example_i_tfrec'].astype(int)
# read TCE table
tce_tbl = pd.read_csv('/nobackupp19/msaragoc/work_dir/Kepler-TESS_exoplanet/data/Ephemeris_tables/TESS/tess_spoc_2min/tess_2min_tces_dv_s1-s88_3-27-2025_1316_label.csv')

#%% create labels using shards table

sigma_thr = 3  # sigma threshold
shards_tbl['transit_source_offset_sigma'] = np.nan
shards_tbl['transit_source_offset_label'] = ''

shards_tbl = shards_tbl.merge(
    tce_tbl[['uid', 'tce_dikco_msky_original', 'tce_dikco_msky_err_original', 'sg1_master_disp']],
    on='uid', how='left', validate='one_to_one')

# create labels based on DV stats
def _compute_transit_source_offset_from_target_sigma(row):

    if row['tce_dikco_msky_err_original'] != -1:
        # lower bound for error is 2.5, so no issue in dividing by zero
        tr_off_tic_sigma =  row['tce_dikco_msky_original'] / row['tce_dikco_msky_err_original']
    else:
        tr_off_tic_sigma = np.nan

    return tr_off_tic_sigma

# create transit source offset sigma
shards_tbl['tce_dikco_msky_sigma'] = shards_tbl.apply(_compute_transit_source_offset_from_target_sigma, axis=1)
# create label by thresholding transit source offset sigma
shards_tbl.loc[shards_tbl['tce_dikco_msky_sigma'] >= sigma_thr, 'transit_source_offset_label'] = 'off-target'
shards_tbl.loc[shards_tbl['tce_dikco_msky_sigma'] < sigma_thr, 'transit_source_offset_label'] = 'on-target'
shards_tbl.loc[shards_tbl['tce_dikco_msky_sigma'].isna(), 'transit_source_offset_label'] = ''

# exclude TCEs with issues when computing the DV transit source statistics
print(f'Number of TCEs before excluding those with transit source offset from TIC error equal to -1: {len(shards_tbl)})')
shards_tbl = shards_tbl.loc[shards_tbl['tce_dikco_msky_err_original'] != -1]
print(f'Number of TCEs after excluding those with transit source offset from TIC error equal to -1: {len(shards_tbl)}')

print(f'Disposition Counts:\n{shards_tbl["label"].value_counts()}')

print(f'Transit source offset label Counts:\n{shards_tbl["transit_source_offset_label"].value_counts()}')
print(f'Transit source offset missing label:\n{shards_tbl["transit_source_offset_label"].isna().sum()}')

#%% build new TFRecord dataset with transit source offset sigma threshold and corresponding label

def write_new_tfrecord_shard(src_shard_tbl, src_shard_fn, src_shard_fp_i, dest_tfrec_dir):
    """ Create new TFRecord shard with transit source offset sigma value and label.

        Args:
            src_shard_tbl: pandas DataFrame, source shard
            src_shard_fn: str, source shard filename
            src_shard_fp_i: int, source shard integer ID
            dest_tfrec_dir: Path, destination directory for new shards

        Returns: pandas DataFrame, TFRecord dataset csv file
    """

    src_shard_tbl.reset_index(inplace=True)
    src_shard_tbl.set_index('example_i_tfrec')

    print(f'Iterating through file {src_shard_fn} ({src_shard_fp_i + 1}')  #  out of {n_shards})')

    dest_tfrec_fp = dest_tfrec_dir / src_shard_fn

    with tf.io.TFRecordWriter(str(dest_tfrec_fp)) as writer:

        # iterate through the source shard
        tfrecord_dataset = tf.data.TFRecordDataset(str(src_tfrec_dir / src_shard_fn))

        for string_record_i, string_record in enumerate(tfrecord_dataset.as_numpy_iterator()):

            if string_record_i in src_shard_tbl['example_i_tfrec']:

                example = tf.train.Example()
                example.ParseFromString(string_record)

                example_util.set_float_feature(example, 'tce_dikco_msky_sigma',
                                               [src_shard_tbl.loc[string_record_i,
                                               'tce_dikco_msky_sigma']])
                example_util.set_bytes_feature(example, 'transit_source_offset_label',
                                               [src_shard_tbl.loc[string_record_i,
                                               'transit_source_offset_label']])

                writer.write(example.SerializeToString())

    print(f'Finished iterating through shard {src_shard_fn}.')


# src_shards_fps = list(src_tfrec_dir.glob('shard-*'))  # get source TFRecord shards
src_shards_tbls = shards_tbl.groupby('shard')
n_shards = len(src_shards_tbls)
print(f'Found {n_shards} shards.')

# parallelize
n_processes = 36
n_jobs = n_shards

jobs = [(src_shard_tbl, src_shard_fn, src_shard_i, dest_tfrec_dir)
        for src_shard_i, (src_shard_fn, src_shard_tbl) in enumerate(src_shards_tbls)]

pool = multiprocessing.Pool(processes=n_processes)
async_results = [pool.apply_async(write_new_tfrecord_shard, job) for job in jobs]
pool.close()
pool.join()
