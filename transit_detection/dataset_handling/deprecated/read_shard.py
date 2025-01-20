"""
Merge TFRecord shards, and concatenate auxillary tables, corresponding to chunks of the transit detection dataset.

Each entry in a shard and aux_tbl should be per TCE in order to merge with expected functionality.
"""

# 3rd party
import numpy as np
from pathlib import Path
import tensorflow as tf
import pandas as pd

# def read_shard(dest_tfrec_path, chunk_paths, update_col=True):

    # with tf.io.TFRecordWriter(str(dest_tfrec_path)) as writer:
    #     try:                
    #         for shard_path, aux_tbl_path in chunk_paths:

    #             dataset_chunk = tf.data.TFRecordDataset(str(shard_path))

    #             for example in dataset_chunk:
    #                 writer.write(example.numpy())


    #     except Exception as e:
    #         print(f"ERROR: For {shard_path.name} and {aux_tbl_path.name}: {e}")


if __name__ == '__main__':
    # for example in tf.python_io.tf_record_iterator("/Users/jochoa4/Desktop/test_transfers/test_shard_0001-0001"):


    raw_dataset = tf.data.TFRecordDataset("/Users/jochoa4/Desktop/test_transfers/test_shard_0001-0013")
    processed_dataset = tf.data.TFRecordDataset("/Users/jochoa4/Desktop/test_processed_tfrecords/test_shard_updated_0001-0013")
    zipped_datasets = zip(raw_dataset.take(150), processed_dataset.take(150))

    for raw_record, processed_record in zipped_datasets:
        raw_example, processed_example = tf.train.Example(), tf.train.Example()
        raw_example.ParseFromString(raw_record.numpy())
        processed_example.ParseFromString(processed_record.numpy())

        # zipped_features = zip(raw_example.features.feature, processed_example.features.feature)
        with open("/Users/jochoa4/Desktop/test_tfrecord/compared_shards.txt", "a") as f:
            # print(str(raw_example), str(processed_example))
            # print(str(str(raw_example) == str(processed_example)))

            # for raw_features, processed_features in zipped_features:
            raw_label, processed_label = raw_example.features.feature["label"].float_list.value[0], processed_example.features.feature["label"].bytes_list.value[0].decode('utf-8')
            raw_label, processed_label = int(raw_label), int(processed_label)
            print(str(raw_label) == str(processed_label))
            # f.write(str((raw_line) == str(processed_line)))
            f.write(str(str(raw_label) == str(processed_label)))
            f.write(str(raw_label) + ' ' + str(processed_label))
            f.write('\n\n')

    # for raw_record in raw_dataset.take(1):
    #     example = tf.train.Example()
    #     example.ParseFromString(raw_record.numpy())
    #     with open("/Users/jochoa4/Desktop/test_tfrecord/shard1.txt", "w") as f:
    #         print(example)
    #         f.write(str(example))

    # for processed_record in processed_dataset.take(1):
    #     example = tf.train.Example()
    #     example.ParseFromString(processed_record.numpy())
    #     with open("/Users/jochoa4/Desktop/test_tfrecord/shard1_processed.txt", "w") as f:
    #         print(example)
    #         f.write(str(example))