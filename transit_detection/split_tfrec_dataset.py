"""
Split tfrec dataset into training, validation, and test sets
"""


# 3rd party
import tensorflow as tf
from pathlib import Path
import numpy as np
import pandas as pd
from collections import defaultdict

# local
from src_preprocessing.tf_util import example_util

# function to process a single shard for a given set of tce uids
def compute_and_write_example_stats(all_set_feature_img_pixels_dict, dest_stats_dir):

    for feature_img in all_set_feature_img_pixels_dict.keys():
        median = all_set_feature_img_pixels_dict[feature_img].median()
        mean = all_set_feature_img_pixels_dict[feature_img].mean()
        std = all_set_feature_img_pixels_dict[feature_img].std()

        stats = {
            'mean' : mean,
            'std' : std,
            'median' : median,
            'total_examples' : len(all_set_feature_img_pixels_dict[feature_img])
        }

        np.save(f"{feature_img}_stats.npy", stats)
        print(f"Statistics for {feature_img} saved in a single .npy file")


# function to process a split and write the output
def process_split(target_ids, src_tfrec_fp, dest_tfrec_fp, dest_stats_dir=None):

    all_set_feature_img_pixels = {
        'diff_img' : np.array([]),
        'oot_img' : np.array([]),
        'snr_img' : np.array([]),
    }

    with tf.io.TFRecordWriter(str(dest_tfrec_fp)) as writer:
        src_tfrec_dataset = tf.data.TFRecordDataset(str(src_tfrec_fp))

        for str_record in src_tfrec_dataset:
            # load example
            example = tf.train.Example()
            example.ParseFromString(str_record)

            target_id = example.features.feature['target_id'].bytes_list.value[0].decode('utf-8')

            if target_id in target_ids:
                if dest_stats_dir:
                    # get pixel values for diff, oot, and snr imgs
                    for feature_img in all_set_feature_img_pixels.keys():
                        # load img feature
                        feature_img_pixels = tf.reshape(tf.io.parse_tensor(example.features.feature[feature_img].bytes_list.value[0], tf.float32),
                                                        [-1]).numpy() # flattened pixels in feature img
                        all_set_feature_img_pixels[feature_img].extend(feature_img_pixels)

                writer.write(example.SerializeToString())

        if dest_stats_dir:
            compute_and_write_example_stats(all_set_feature_img_pixels, dest_stats_dir)
        

if __name__ == "__main__":
    # load tce tbl
    tce_tbl = pd.read_csv('/nobackup/jochoa4/work_dir/data/tables/tess_2min_tces_dv_s1-s68_all_msectors_11-29-2023_2157_newlabels_nebs_npcs_bds_ebsntps_to_unks.csv')
    tce_tbl = tce_tbl.loc[tce_tbl['label'].isin(['EB','KP','CP','NTP','NEB','NPC'])] #filter for relevant labels

    #src directory for merged tfrecord shard
    src_tfrec_dir = Path("/nobackup/jochoa4/work_dir/data/datasets/TESS_exoplanet_dataset_11-25-2024")

    # destination directory for tfrecord shard splits
    dest_tfrec_dir = Path("/nobackup/jochoa4/work_dir/data/datasets/TESS_exoplanet_dataset_11-25-2024_split")

    # destination directory for computed training stats
    dest_stats_dir = Path("/nobackup/jochoa4/work_dir/data/stats/TESS_exoplanet_dataset_11-25-2024_stats")

    # set random seed
    np.random.seed(42) 

    # get unique targets from tce_tbl
    unique_targets = tce_tbl['target_id'].unique()

    # shuffle targets
    shuffled_targets = np.random.permutation(unique_targets) 

    # fractions for set sizes
    train_set_frac = 0.8
    val_set_frac = 0.1

    # compute exact sizes for each split
    num_train = int(len(shuffled_targets) * 0.8)
    num_val = int(len(shuffled_targets) * 0.1)
    num_test = len(shuffled_targets) - num_train - num_val

    # load targets per set from randomly shuffled targets
    train_targets = shuffled_targets[:num_train]
    val_targets = shuffled_targets[num_train:num_train + num_val]
    test_targets = shuffled_targets[num_train + num_val:]

    print(f'Using {len(train_targets)} targets in training set')
    print(f'Using {len(val_targets)} targets in validation set')
    print(f'Using {len(test_targets)} targets in test set')

    # use auxillary table that contains uid
    aux_table = pd.read_csv('data_tbl.csv')

    train_tfrec_fp = dest_tfrec_dir / 'training_set'
    val_tfrec_fp = dest_tfrec_dir / 'validation_set'
    test_tfrec_fp = dest_tfrec_dir / 'test_set'

    # Build training set
    process_split(train_targets, src_tfrec_dir, train_tfrec_fp, dest_stats_dir)

    # Build validation set
    process_split(val_targets, src_tfrec_dir, val_tfrec_fp)

    # Build test set
    process_split(test_targets, src_tfrec_dir, test_tfrec_fp)





    
    



