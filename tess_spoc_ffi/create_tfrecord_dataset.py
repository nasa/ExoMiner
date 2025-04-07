"""
Create TFRecord dataset for transfer learning for TESS SPOC FFI.
The TFRecord shards are built using three types of shards based on the type of observation and the type of target the
TCEs belong to:
- Shards with 2min TCEs whose targets are not shared with FFI data.
- Shards with 2min TCEs whose targets are shared with FFI data.
- Shards with FFI TCEs (from targets both shared and not shared with 2-min data).
"""

# 3rd party
import tensorflow as tf
from pathlib import Path
import pandas as pd

# local
from src_preprocessing.tf_util import example_util

#%% Set directories

src_data_dir = Path('/nobackupp19/msaragoc/work_dir/Kepler-TESS_exoplanet/data/tfrecords/TESS/tfrecords_tess_spoc_ffi_s36-s72_multisector_s56-s69_1-3-2025_1157_data/cv_tfrecords_tess_spoc_ffi_s36-s72_multisector_s56-s69_1-6-2025_1132/tfrecords/eval_with_2mindata_noduplicates')

dest_dir = Path('/nobackupp19/msaragoc/work_dir/Kepler-TESS_exoplanet/data/tfrecords/TESS/tfrecords_tess_spoc_ffi_s36-s72_multisector_s56-s69_1-3-2025_1157_data/cv_tfrecords_tess_spoc_ffi_s36-s72_multisector_s56-s69_1-6-2025_1132/tfrecords/eval_with_2mindata_transferlearning')
dest_dir.mkdir(exist_ok=True)

#%% Load shards tables

src_shards_tbl = pd.read_csv(src_data_dir / 'shards_tbl.csv')

ffi_targets = src_shards_tbl.loc[src_shards_tbl['obs_type'] == 'ffi', 'target_id'].unique()
print(f'Number of FFI targets: {len(ffi_targets)}')
twomin_only_targets = src_shards_tbl.loc[~src_shards_tbl['target_id'].isin(ffi_targets), 'target_id'].unique()
print(f'Number of 2-min only targets: {len(twomin_only_targets)}')

twomin_only_targets_tces = src_shards_tbl.loc[src_shards_tbl['target_id'].isin(twomin_only_targets)]
print(f'Number of 2-min only targets TCEs: {len(twomin_only_targets_tces)}\n {twomin_only_targets_tces["label"].value_counts()}')

# get shared FFI and 2-min targets TCEs + FFI-only targets TCEs
shared_target_tces = src_shards_tbl.loc[~src_shards_tbl['target_id'].isin(twomin_only_targets)]
print(f'Number of 2-min+FFI shared targets TCEs: {len(shared_target_tces)}\n {shared_target_tces["label"].value_counts()}')

# shared_ffi_target_tces = shared_target_tces.loc[shared_target_tces['obs_type'] == 'ffi']
# print(f'Number of FFI shared targets TCEs: {len(shared_ffi_target_tces)}\n {shared_ffi_target_tces["label"].value_counts()}')
# shared_twomin_target_tces = shared_target_tces.loc[shared_target_tces['obs_type'] == '2min']
# print(f'Number of 2-min shared targets TCEs: {len(shared_twomin_target_tces)}\n {shared_twomin_target_tces["label"].value_counts()}')

#%%

# Split into multiple DataFrames with about the same size
def split_dataframe(df_grouped, num_splits):

    # get the list of grouped dataframes by target id
    grouped_list = [group for _, group in df_grouped]
    n_targets = len(grouped_list)

    # calculate the size of each split
    split_size = n_targets // num_splits

    n_targets_left = n_targets % num_splits

    # Split the list into multiple DataFrames
    split_dfs = [pd.concat(grouped_list[i * split_size: (i + 1) * split_size]) for i in range(num_splits)]

    # Handle any remaining groups
    if n_targets_left != 0:
        if n_targets_left < 0.25 * split_size:
            split_dfs[-1] = pd.concat([split_dfs[-1]] + grouped_list[num_splits * split_size:], axis=0)
        else:
            split_dfs.append(pd.concat(grouped_list[num_splits * split_size:]))

    return split_dfs


# group by target id
twomin_only_targets_tces_grps = twomin_only_targets_tces.groupby('target_id')
shared_target_tces_grps = shared_target_tces.groupby('target_id')

# split into dataframes of about the same size
n_shards_twomin_only_targets, n_shards_shared_targets = 10 , 10
split_twomin_only_targets_tces = split_dataframe(twomin_only_targets_tces_grps, num_splits=n_shards_twomin_only_targets)
split_shared_target_tces = split_dataframe(shared_target_tces_grps, num_splits=n_shards_shared_targets)

# separate TCEs from shared targets based on observation type
split_shared_target_ffi_tces = [tbl_split.loc[tbl_split['obs_type'] == 'ffi']
                                for tbl_split in split_shared_target_tces]
split_shared_target_twomin_tces = [tbl_split.loc[tbl_split['obs_type'] == '2min']
                                   for tbl_split in split_shared_target_tces]

#%% Create new TFRecord dataset

def process_example(example):

    obs_type_map = {'2min': 0, 'ffi': 1}

    example_obs_type = example.features.feature['obs_type'].bytes_list.value[0].decode("utf-8")
    example_util.set_int64_feature(example, 'obs_type_int', [obs_type_map[example_obs_type]],
                                   allow_overwrite=True)

    return example


def write_examples_to_shard_from_src_shards(dest_tfrec_fp, shard_split, src_shards_fns, src_shard_dir):

    with (tf.io.TFRecordWriter(str(dest_tfrec_fp)) as writer):

        for src_shard_fn in src_shards_fns:  # iterate on source shards

            # get examples from source shard
            examples_src_shard_df = shard_split.get_group(src_shard_fn)
            examples_src_shard_df = examples_src_shard_df.sort_values(by=['example_i_tfrec'],
                                                                      ascending=True)

            # iterate through the source shard
            tfrecord_dataset = tf.data.TFRecordDataset(str(src_shard_dir / src_shard_fn))
            for example_i, string_record in enumerate(tfrecord_dataset.as_numpy_iterator()):

                if example_i in examples_src_shard_df['example_i_tfrec'].to_list():  # example found

                    example = tf.train.Example()
                    example.ParseFromString(string_record)

                    # manipulate example
                    example = process_example(example)

                    writer.write(example.SerializeToString())

                # stop when there are no more selected examples
                if example_i > examples_src_shard_df['example_i_tfrec'].max():
                    break

def write_tfrecord_from_src_dataset(split_shards, prefix_shard_fn, src_shards_dir):

    n_shards = len(split_shards)  # number of destination shards

    for shard_i in range(n_shards):  # iterate on destination shards

        dest_tfrec_fp = dest_dir / f'{prefix_shard_fn}_{shard_i}'

        print(f'Iterating on shard {dest_tfrec_fp} ({shard_i + 1}/{n_shards})')

        curr_split = split_shards[shard_i]
        curr_split_grps = curr_split.groupby('shard')
        src_shards_fns = curr_split['shard'].unique()  # get source shards that will map to the destination shard

        print(f'Number of source shards for {dest_tfrec_fp.name}: {len(src_shards_fns)}')

        write_examples_to_shard_from_src_shards(dest_tfrec_fp, curr_split_grps, src_shards_fns, src_shards_dir)


write_tfrecord_from_src_dataset(split_twomin_only_targets_tces, 'shard_twomin_only_targets', src_data_dir)
write_tfrecord_from_src_dataset(split_shared_target_twomin_tces, 'shard_twomin_shared_targets', src_data_dir)
write_tfrecord_from_src_dataset(split_shared_target_ffi_tces, 'shard_ffi_targets', src_data_dir)
