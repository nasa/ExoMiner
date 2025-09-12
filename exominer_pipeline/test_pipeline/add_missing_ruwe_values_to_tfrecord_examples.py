"""Add RUWE values for examples in TFRecord dataset for cases where querying Gaia during ExoMiner Pipeline run(s) failed."""

# 3rd party
import pandas as pd
from pathlib import Path
import tensorflow as tf
import numpy as np
from tqdm import tqdm

# local
from src_preprocessing.tf_util import example_util

#%% setup

src_tfrec_dir = Path('/data3/exoplnt_dl/tfrecords/tess/tfrecords_tess-spoc-2min_s68s94_normalized_noneighborsimgs_aggexominerpipelineruns_9-9-2025_1155')
tce_tbl = pd.read_csv('/data3/exoplnt_dl/ephemeris_tables/tess/tess_spoc_2min/tess_2min_tces_dv_s1-s88_3-27-2025_1316_label_dv_mast_urls.csv')
pred_tbl = pd.read_csv('/data3/exoplnt_dl/experiments/exominer_pipeline/runs/predictions_exominer_pipeline_run_tics_aggregated_9-3-2025_1201_with_tois_in_tic_dv-mini_toi-ephem-matched_toi-dispositions.csv')
norm_stats = np.load('/data3/exoplnt_dl/codebase/exominer_pipeline/data/norm_stats/train_scalarparam_norm_stats.npy', allow_pickle=True).item()
std_eps = 1e-32

dest_tfrec_dir = src_tfrec_dir.parent / f'{src_tfrec_dir.stem}_with_ruwe'

dest_tfrec_dir.mkdir(parents=True, exist_ok=True)

shards_tbl = pd.read_csv(src_tfrec_dir / 'shards_tbl.csv')

med_tce_ruwe = norm_stats['ruwe']['median']
mad_std_tce_ruwe = norm_stats['ruwe']['mad_std']

#%% get TCEs with missing RUWE values

tces_missing_ruwe = pred_tbl.loc[pred_tbl['ruwe'].isna(), 'uid'].tolist()
print(f"Found {len(tces_missing_ruwe)} TCEs with missing RUWE values.")

tces_ruwe = tce_tbl.loc[tce_tbl['uid'].isin(tces_missing_ruwe), ['uid', 'ruwe']]
tces_ruwe = tces_ruwe.loc[~tces_ruwe['ruwe'].isna()]
print(f"Found {len(tces_ruwe)}/{len(tces_missing_ruwe)} TCEs with RUWE values in TCE table.")

tces_ruwe.set_index('uid', inplace=True)

#%% add RUWE values to TFRecord examples and write new TFRecord dataset

tfrec_fps = list(src_tfrec_dir.glob('*.tfrecord'))

for tfrec_fp_i, tfrec_fp in enumerate(tqdm(tfrec_fps, desc="Processing TFRecords")):
    
    if tfrec_fp_i % 50 == 0:
        print(f"Processing TFRecord file {tfrec_fp_i + 1}/{len(tfrec_fps)}: {tfrec_fp}...")
    
    dest_tfrec_fp = dest_tfrec_dir / tfrec_fp.name
    
    tfrecord_dataset = tf.data.TFRecordDataset(str(tfrec_fp))
    
    shard_tbl = shards_tbl.loc[shards_tbl['shard'] == str(tfrec_fp.name)]
    shard_tbl_tces_missing_ruwe = shard_tbl.loc[shard_tbl['uid'].isin(tces_ruwe.index)]   
    examples_lst = []
    for example_i, string_record in enumerate(tfrecord_dataset.as_numpy_iterator()):

        shard_tbl_tce = shard_tbl_tces_missing_ruwe.loc[shard_tbl_tces_missing_ruwe['example_i_tfrec'] == example_i]
        if len(shard_tbl_tce) == 0:  # TCE does not have missing RUWE value
            examples_lst.append(string_record)
        elif len(shard_tbl_tce) == 1:
            
            example = tf.train.Example()
            example.ParseFromString(string_record)
            
            example_uid = example.features.feature['uid'].bytes_list.value[0].decode('utf-8')   
            
            expected_uid = shard_tbl_tce['uid'].iloc[0]
            if example_uid != expected_uid:
                raise ValueError(f"Example index {example_i} in shard table for TFRecord file {tfrec_fp.name} has TCE UID {expected_uid}, but example has TCE UID {example_uid}.")
            
            # get RUWE value for this TCE
            tce_ruwe = float(tces_ruwe.loc[example_uid, 'ruwe'])
            # standardize RUWE value
            tce_ruwe_norm = (tce_ruwe - med_tce_ruwe) / (mad_std_tce_ruwe + std_eps)
            
            example_ruwe = example.features.feature['ruwe'].float_list.value[0]
            example_ruwe_norm = example.features.feature['ruwe_norm'].float_list.value[0]
            print(f'Example index {example_i} in shard table for TFRecord file {tfrec_fp.name} has TCE UID {example_uid} with missing RUWE value {example_ruwe:.3f} (standardized {example_ruwe_norm:.3f}). Adding RUWE value {tce_ruwe:.3f} (standardized: {tce_ruwe_norm:.3f}).')
            
            example_util.set_float_feature(example, 'ruwe', [tce_ruwe], allow_overwrite=True)
            example_util.set_float_feature(example, 'ruwe_norm', [tce_ruwe_norm], allow_overwrite=True)
            
            examples_lst.append(example.SerializeToString())
            
        else:
            raise ValueError(f"Multiple entries found for example index {example_i} in shard table for TFRecord file {tfrec_fp.name}.")
        
    with tf.io.TFRecordWriter(str(dest_tfrec_fp)) as writer:

            for example_str in examples_lst:
                writer.write(example_str)
