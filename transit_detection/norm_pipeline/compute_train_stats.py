"""
Split tfrec dataset into training, validation, and test sets
"""

# 3rd party
import tensorflow as tf
from pathlib import Path
import numpy as np
import pandas as pd
from collections import defaultdict
import multiprocessing
from functools import partial
import time
import logging

# local
from src_preprocessing.tf_util import example_util



# function to process a single shard for a given set of tce uids
def compute_and_write_example_stats(set_feature_img_pixels_dict, dest_stats_dir):
    """
    Compute median, mean, and std for a list of pixels belonging to a feature_img and write to .npy file
    
    Input:
        set_feature_img_pixels_dict: Dict with keys corresponding to features mapped to a flattened list
                    of pixels from all images in the set.
        dest_stats_dir: Path to the directory to save computed statistics to. 
    """

    for feature_img in set_feature_img_pixels_dict.keys():
        median = np.median(set_feature_img_pixels_dict[feature_img])
        mean = np.mean(set_feature_img_pixels_dict[feature_img])
        std = np.std(set_feature_img_pixels_dict[feature_img])

        stats = {
            'mean' : mean,
            'std' : std,
            'median' : median,
            'total_pixels' : len(set_feature_img_pixels_dict[feature_img]),
        }

        np.save(dest_stats_dir / f"train_set_{feature_img}_stats.npy", stats)
        print(f"Statistics for {feature_img} saved in a single .npy file")
        print(stats)

# function to process a split and write the output
def retrieve_shard_feature_img_pixels(src_tfrec_fp, src_aux_tbl_fp, feature_names, max_examples_per_tce):
    """
    Retrieve a shard's flattened feature_img pixels for all examples in the shard, based on features provided
    as keys in shard_feature_img_pixels

    Input:
        src_tfrec_fp : str or Path corresponding to shard filepath
        feature_names : List or other iterable of feature_names to search for in shard
    
    Returns:
        shard_feature_img_pixels : Dict of npy arrays holding all pixels in shard per feature_name
                                    ie) {feature1: [1,...N], feature2: [1....N]}
    """
    
    # set logger for each process
    logger = logging.getLogger(f"{str(src_tfrec_fp).split('/')[-1]}")
    logger.setLevel(logging.INFO)

    log_path = Path(log_dir) / f"{str(src_tfrec_fp).split('/')[-1]}.log"
    file_handler = logging.FileHandler(log_path)
    logger_formatter = logging.Formatter('%(asctime)s - %(levelname)s- %(message)s')
    file_handler.setFormatter(logger_formatter)

    # handle lingering handlers
    if logger.hasHandlers():
        logger.handlers.clear()
    
    logger.addHandler(file_handler)
    logger.info(f"Beginning data processing for {str(src_tfrec_fp).split('/')[-1]} using max of {max_examples_per_tce} examples per tce.")


    shard_feature_img_pixels = {feature_name : [] for feature_name in feature_names}

    src_tfrec_dataset = tf.data.TFRecordDataset(str(src_tfrec_fp))

    record_num = 0
    recorded_record_num = 0

    uids_to_process_set = set()

    split_df = pd.read_csv(src_aux_tbl_fp)
    split_df = split_df[split_df['shard'] == str(src_tfrec_fp.name)]

    # Transform uid with midpoint included (358-2-S38_t_2342.461771194361) -> to without (358-2-S38) for selecting randomly
    split_df['uid_prefix'] = split_df["uid"].apply(lambda x: x.split('_')[0])

    for uid_prefix in split_df['uid_prefix'].unique():
        df_subset = split_df[split_df['uid_prefix'] == uid_prefix]
        uids = df_subset['uid'].sample(min(len(df_subset), max_examples_per_tce))

        for uid in uids:
            uids_to_process_set.add(uid)

    if len(uids_to_process_set) == 0:
        return shard_feature_img_pixels # skip processing

    for str_record in src_tfrec_dataset:
        record_num += 1
        logger.info(f"Processing tfrecord example #{str(record_num).zfill(4)}")
        # load example
        example = tf.train.Example()
        example.ParseFromString(str_record.numpy())
        
        # Get only uid, without mid transit/oot point; expected to be in form 1129033-1-S1-36_t_1412.344...
        tce_uid = str(example.features.feature["uid"].bytes_list.value[0])

        if tce_uid in uids_to_process_set:


            recorded_record_num += 1
            # get pixel values for diff, oot, and snr imgs
            for feature_img in shard_feature_img_pixels.keys():
                logger.info(f"Processing feature: {feature_img} for example #{str(record_num).zfill(4)}")
                # load img feature
                feature_img_pixels = tf.reshape(tf.io.parse_tensor(example.features.feature[feature_img].bytes_list.value[0], tf.float32),
                                                [-1]).numpy() # flattened pixels in feature img
                
                shard_feature_img_pixels[feature_img].extend(feature_img_pixels.tolist())
        
    logger.info(f"Successfully finished processing {str(src_tfrec_fp).split('/')[-1]} for a total of {recorded_record_num}/{record_num} examples.")

    return shard_feature_img_pixels
        

if __name__ == "__main__":

    # src directory containing training set tfrecords
    train_set_tfrec_dir = Path("/nobackup/jochoa4/work_dir/data/datasets/TESS_exoplanet_dataset_11-12-2024_split/tfrecords/train_tfrecords")

    # src directory containing set aux_tbls
    src_aux_tbl_dir = Path("/nobackup/jochoa4/work_dir/data/datasets/TESS_exoplanet_dataset_11-12-2024_split/tfrecords/aux_tbls")
    
    # TRUE RUNS
    # destination directory for computed training stats
    dest_stats_dir = Path("/nobackup/jochoa4/work_dir/data/stats/TESS_exoplanet_dataset_11-25-2024_split_v2/")

    # destination directory for logging
    log_dir = Path("/nobackup/jochoa4/work_dir/data/logging/compute_train_stats_logs_v2")

    # # TESTING
    # dest_stats_dir = Path("/nobackupp27/jochoa4/work_dir/test_runs/compute_test_stats/stats/")
    # log_dir = Path("/nobackupp27/jochoa4/work_dir/test_runs/compute_test_stats/compute_train_stats_logs/")

    dest_stats_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # setting up logger
    logger = logging.getLogger(f"compute_train_stats_logger")
    logger.setLevel(logging.INFO)

    log_path = Path(log_dir) / f"compute_train_stats.log"
    file_handler = logging.FileHandler(log_path)
    logger_formatter = logging.Formatter('%(asctime)s - %(levelname)s- %(message)s')
    file_handler.setFormatter(logger_formatter)
    logger.addHandler(file_handler)

    # define feature imgs keys used in tfrecords to compute stats for
    train_set_feature_imgs = ['diff_img', 'oot_img', 'snr_img']

    train_set_feature_img_pixels = { feature_img : [] for feature_img in train_set_feature_imgs }

    # define partial func to predefine default values for directories and target sets
    partial_func = partial(retrieve_shard_feature_img_pixels,
                                feature_names=train_set_feature_imgs,
                                max_examples_per_tce=4,
                                )
    
    start = time.time()

    # using N-2 = 126 as tested working value
    pool = multiprocessing.Pool(processes=126)

    # jobs = [str(shard_num).zfill(4) for shard_num in range(1, 8611 + 1)] # chunk 1 to chunk 8611
    jobs = [str(shard_num).zfill(4) for shard_num in range(1, 8611 + 1)] # chunk 1 to chunk 8611

    logger.info(f"Beginning processing {len(jobs)} shards from range {jobs[0]} to {jobs[-1]}")
    print(f"Beginning processing {len(jobs)} shards from range {jobs[0]} to {jobs[-1]}")

    results = []
    for shard_num in jobs:
        shard_fp = train_set_tfrec_dir / f"train_shard_{shard_num}-8611"
        aux_tbl_fp = src_aux_tbl_dir / f"shards_tbl_{shard_num}-8611.csv"
        shard_pixels = pool.apply_async(partial_func, args=[shard_fp, aux_tbl_fp]).get()
        results.append(shard_pixels) # add flattened images to it

    logger.info(f"Succesfully finished retrieving pixels for {len(results)} shards.")

    num_results = 0
    for result in results:
        num_results += 1
        logger.info(f"Extending set pixels for result {str(num_results).zfill(4)}")
        for feature, pixels in result.items():
            train_set_feature_img_pixels[feature].extend(pixels)
            
    pool.close()
    pool.join()
    
    # Compute statistics based on train_set_img_pixels
    logger.info(f"Beginning computing train_set statistics.")
    compute_and_write_example_stats(set_feature_img_pixels_dict=train_set_feature_img_pixels, dest_stats_dir=dest_stats_dir)
    logger.info(f"Finished computing train_set statistics.")

    end = time.time()
    logger.info(f"Succesfully finished processing train_set shards in {end-start} seconds")




    


