import os
import tensorflow as tf
import pandas as pd
from pathlib import Path
import src_preprocessing.tf_util.example_util as example_util
#Supress Warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
"""
Print values within a shard to ensure that the shard is normalized, and contains the normalized features.
"""
#Kepler input

src_tfrec_dir = Path('/Users/agiri1/Desktop/ExoBD_Datasets/shard_tables/cv_iter_1/norm_data')
src_tfrec_shards = [shard for shard in src_tfrec_dir.iterdir() if "shard" in shard.name]
for src_shard in src_tfrec_shards:
    tfrecord_dataset = tf.data.TFRecordDataset(str(src_shard))
    for string_record in tfrecord_dataset.as_numpy_iterator():
        example = tf.train.Example()
        example.ParseFromString(string_record)
        example_features = example.features.feature['boot_fap']
        print(example_features)