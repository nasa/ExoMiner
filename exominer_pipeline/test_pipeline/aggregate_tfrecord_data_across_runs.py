"""Aggregate TFRecord files across ExoMiner Pipeline runs."""

# 3rd party
from pathlib import Path
import pandas as pd
import tensorflow as tf

#%% setup

runs_root_dir = Path('/data3/exoplnt_dl/experiments/exominer_pipeline/runs/')
tfrec_dest_dir = Path('/data3/exoplnt_dl/tfrecords/tess/tfrecords_tess-spoc-2min_s68s94_normalized_noneighborsimgs_aggexominerpipelineruns_9-9-2025_1155')

tfrec_dest_dir.mkdir(parents=True, exist_ok=True)

#%% get list of TFRecord files across all runs

tfrec_fps = list(runs_root_dir.rglob('*/tfrecord_data_diffimg_normalized/shard-tess_diffimg_TESS_0'))
print(f"Found {len(tfrec_fps)} TFRecord files across ExoMiner Pipeline runs.")

#%% create table that maps all examples TCE UIDs to their source TFRecord filepath

data_dict = {field: [] for field in ['uid', 'tfrec_fp']}
for tfrec_fp_i, tfrec_fp in enumerate(tfrec_fps):
    
    if tfrec_fp_i % 50 == 0:
        print(f"Processing TFRecord file {tfrec_fp_i + 1}/{len(tfrec_fps)}: {tfrec_fps}...")
    
    tfrecord_dataset = tf.data.TFRecordDataset(str(tfrec_fp))

    for example_i, string_record in enumerate(tfrecord_dataset.as_numpy_iterator()):

        example = tf.train.Example()
        example.ParseFromString(string_record)

        example_uid = example.features.feature['uid'].bytes_list.value[0].decode('utf-8')

        data_dict['uid'].append(example_uid)
        data_dict['tfrec_fp'].append(str(tfrec_fp))
    
data_tbl = pd.DataFrame(data_dict)
print(f"Aggregated data table has {len(data_tbl)} examples.")

#%% remove duplicates

n_duplicates = len(data_tbl) - len(data_tbl['uid'].unique())
print(f"Found {n_duplicates} duplicate examples in aggregated data table.")

data_tbl = data_tbl.drop_duplicates(subset=['uid']).reset_index(drop=True)
data_tbl.to_csv(tfrec_dest_dir / 'source_tfrec_examples.csv', index=False)

#%% create new TFRecord dataset from source TFRecord files

n_examples_per_shard = 300

examples_grp_shard = data_tbl.groupby('tfrec_fp')
total_files = len(examples_grp_shard)

dest_tfrec_i = 0
examples_lst = []
for tfrec_i, (tfrec_fp, examples_grp) in enumerate(examples_grp_shard):

    if tfrec_i % 50 == 0:
        print(f"Processing TFRecord file {tfrec_i + 1}/{len(total_files)} with {len(examples_grp)} examples: {tfrec_fp}...")
        
    tfrecord_dataset = tf.data.TFRecordDataset(str(tfrec_fp))
    
    for example_i, string_record in enumerate(tfrecord_dataset.as_numpy_iterator()):
        
        example = tf.train.Example()
        example.ParseFromString(string_record)

        examples_lst.append(example.SerializeToString())
        
        if len(examples_lst) >= n_examples_per_shard:
            
            print(f"Writing shard {dest_tfrec_i} with {len(examples_lst)} examples...")
            
            dest_data_shard_fp = tfrec_dest_dir / f'shard_{dest_tfrec_i}.tfrecord'
    
            with tf.io.TFRecordWriter(str(dest_data_shard_fp)) as writer:
    
                for example_str in examples_lst:
                    writer.write(example_str)
                    
            examples_lst= []
            dest_tfrec_i += 1

if len(examples_lst) > 0:
    print(f"Writing final shard {dest_tfrec_i} with {len(examples_lst)} examples...")
    dest_data_shard_fp = tfrec_dest_dir / f'shard_{dest_tfrec_i}.tfrecord'
    with (tf.io.TFRecordWriter(str(dest_data_shard_fp)) as writer):
        for example_str in examples_lst:
            writer.write(example_str)
    