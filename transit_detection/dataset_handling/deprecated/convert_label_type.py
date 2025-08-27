"""Used to convert feature label from float to str for use in label mapping"""



# 3rd party
import numpy as np
from pathlib import Path
import tensorflow as tf
import pandas as pd

def convert_shard(tfrec_path, dest_tfrec_path):
    count = 1
    with tf.io.TFRecordWriter(str(dest_tfrec_path)) as writer:
        try:                

            serialized_dataset = tf.data.TFRecordDataset(str(tfrec_path))

            for serialized_example in serialized_dataset:
                # get label feature 
                example = tf.train.Example()
                updated_example = update_label(serialized_example.numpy())
                writer.write(updated_example)

        except Exception as e:
            print(f"ERROR: For {tfrec_path.name}: {e}")

def update_label(serialized_example):
    #parse serialized example into tf.trainExample
    example = tf.train.Example()
    example.ParseFromString(serialized_example)

    float_label = example.features.feature['label'].float_list.value[0]
    str_label = None
    if float_label == 0.0:
        str_label = str(0).encode('latin-1')
    elif float_label == 1.0:
        str_label = str(1).encode('latin-1')
    else:
        print("ERROR converting float label")

    example.features.feature['label'].Clear()
    example.features.feature['label'].bytes_list.value.append(str_label)

    return example.SerializeToString()

if __name__ == '__main__':
    # for example in tf.python_io.tf_record_iterator("/Users/jochoa4/Desktop/test_transfers/test_shard_0001-0001"):



    # tfrec_path = Path("/Users/jochoa4/Desktop/test_tfrecords/test_shard_0001-0013")
    # updated_tfrec_path = Path("//Users/jochoa4/Desktop/test_tfrecord/test_shard_updated_0001-0013")
    tfrec_path = Path("/Users/jochoa4/Desktop/test_tfrecords/test_shard_0001-0001")
    updated_tfrec_path = Path("//Users/jochoa4/Desktop/test_tfrecord/test_shard_updated_0001-0001")
    convert_shard(tfrec_path, updated_tfrec_path)
    # for raw_record in raw_dataset.take(16):
    #     example = tf.train.Example()
    #     example.ParseFromString(raw_record.numpy())
    #     with open("/Users/jochoa4/Desktop/test_tfrecord/shard1.txt", "a") as f:
    #         print(example)
    #         f.write(str(example))