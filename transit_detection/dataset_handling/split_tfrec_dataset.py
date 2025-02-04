"""
Split tfrec dataset into training, validation, and test sets
"""


# 3rd party
import tensorflow as tf
from pathlib import Path
import numpy as np
import pandas as pd
from collections import defaultdict

# TODO: Add multiple shards for train/test/val
# can parallelize on multiple shards
import multiprocessing
from functools import partial
from copy import deepcopy

# local
from src_preprocessing.tf_util import example_util

def process_shard(input_shard_num, input_dir, output_dir, train_targets, val_targets, test_targets):

    input_shard_fp = input_dir / f'test_shard_0001-{input_shard_num}'

    train_tfrec_fp = output_dir / 'train_set'
    val_tfrec_fp = output_dir / 'val_set'
    test_tfrec_fp = output_dir / 'test_set'

    writers = {
        "train" : tf.io.TFRecordWriter(train_tfrec_fp / f"train_shard_0001-{input_shard_num}"),
        "val" : tf.io.TFRecordWriter(val_tfrec_fp / f"val_shard_0001-{input_shard_num}"),
        "test" : tf.io.TFRecordWriter(test_tfrec_fp / f"test_shard_0001-{input_shard_num}")
    }

    for str_record in tf.data.TFRecordDataset(input_shard_fp):
        example = tf.train.Example()
        example.ParseFromString(str_record)

        target_id = example.features.feature['target_id'].bytes_list.value[0].decode('utf-8')

        if target_id in train_targets:
            split = "train"
        elif target_id in val_targets:
            split = "val"
        elif target_id in test_targets:
            split = "test"
        else:
            raise ValueError(f"Target ID {target_id} not found in any set")
        
        writers[split].write(str_record.numpy())
    
    for writer in writers.values():
        writer.close()

    print(f"Processed shard: {input_shard_fp}")

    return f"Completed: {input_shard_fp}"


if __name__ == "__main__":
    # load tce tbl
    tce_tbl = pd.read_csv('/nobackup/jochoa4/work_dir/data/tables/tess_2min_tces_dv_s1-s68_all_msectors_11-29-2023_2157_newlabels_nebs_npcs_bds_ebsntps_to_unks.csv')
    tce_tbl = tce_tbl.loc[tce_tbl['label'].isin(['EB','KP','CP','NTP','NEB','NPC'])] #filter for relevant labels

    # src directory for raw tfrecord shards
    src_tfrec_dir = Path("/nobackup/jochoa4/work_dir/data/datasets/TESS_exoplanet_dataset_11-25-2024")

    # src fp for auxillary table
    src_aux_tbl_fp = Path("/nobackup/jochoa4/work_dir/data/datasets/TESS_exoplanet_dataset_11-25-2024/data_tbl.csv")

    # destination directory for tfrecord shard splits
    dest_tfrec_dir = Path("/nobackup/jochoa4/work_dir/data/datasets/TESS_exoplanet_dataset_11-25-2024_split")

    # destination directory for logging
    dest_log_dir = Path("/nobackup/jochoa4/work_dir/data/logging") #TODO: revisit and update dest_log_dir

    # set random seed
    np.random.seed(42) 

    # get unique targets from tce_tbl
    unique_targets = tce_tbl['target_id'].unique() # TODO: update with targets found in auxillary table

    # shuffle targets
    shuffled_targets = np.random.permutation(unique_targets) 

    # fractions for set sizes
    train_set_frac = 0.8
    val_set_frac = 0.1

    # number of unique targets
    N = len(shuffled_targets)

    # compute exact sizes for each split
    num_train = int(N * 0.8)
    num_val = int(N * 0.1)
    num_test = N - num_train - num_val # remaining fraction 

    # load targets per set from randomly shuffled targets
    train_targets = set(shuffled_targets[:num_train])
    val_targets = set(shuffled_targets[num_train:num_train + num_val])
    test_targets = set(shuffled_targets[num_train + num_val:])

    print(f'Using {len(train_targets)} targets in training set')
    print(f'Using {len(val_targets)} targets in validation set')
    print(f'Using {len(test_targets)} targets in test set')

    # Save the dataset targets to a npy file
    dataset_targets = {}
    dataset_targets['train_set'] = train_targets
    dataset_targets['val_set'] = val_targets
    dataset_targets['test_set'] = test_targets

    np.save('dataset_targets.npy', dataset_targets)

    # # define logging??

    # job_log_dir = log_dir / 'job_logs'
    # job_log_dir.mkdir(parents=True, exist_ok=True)


    #     # log chunk validation
    # logger = logging.getLogger(f"chunk_validation_logger")
    # logger.setLevel(logging.INFO)

    # log_path = Path(log_dir) / f"chunk_validation.log"
    # file_handler = logging.FileHandler(log_path)
    # logger_formatter = logging.Formatter('%(asctime)s - %(levelname)s- %(message)s')
    # file_handler.setFormatter(logger_formatter)
    # logger.addHandler(file_handler)

    # logger.info(f'Processing a total of {len(chunked_jobs)} chunks of size {job_chunk_size}')
    # print(f'Processing a total of {len(chunked_jobs)} chunks of size {job_chunk_size}')


    # logger.info(f'Skipping processing for {sum(processed_chunk_mask)} chunks that have already been processed.')



    # define partial func to predefine default values for directories and target sets
    partial_func = partial(process_shard,
                                # def input_shard_num per job
                                input_dir=src_tfrec_dir,
                                output_dir=dest_tfrec_dir,
                                train_targets=train_targets,
                                val_targets=val_targets,
                                test_targets=test_targets
                                )

    # using N-1 = 127
    pool = multiprocessing.Pool(processes=127)

    jobs = [shard_num for shard_num in range(1, 8612)] # chunk 1 to chunk 8611

    for shard_num in jobs:
        pool.apply_async(partial_func, args=shard_num)

    pool.close()
    pool.join()
    
    print('Finished processing train val test split')

