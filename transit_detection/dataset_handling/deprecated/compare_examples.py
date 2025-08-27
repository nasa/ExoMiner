import numpy as np
from pathlib import Path
import tensorflow as tf
import pandas as pd

def parse_example(serialized_example):
    example = tf.train.Example()
    example.ParseFromString(serialized_example)
    return example

def compare_examples(original_example, updated_example):
    original_features = original_example.features.feature
    updated_features = updated_example.features.feature
    for key in original_features:
        if key == 'label':
            original_label = original_features[key].float_list.value[0]
            updated_label = updated_features[key].bytes_list.value[0].decode('latin-1')
            expected_label = '0' if original_label == 0.0 else '1'
            if updated_label != expected_label:
                print(f"Mismatch in label: Original={original_label}, Updated={updated_label}")
                return False
        else:
            if original_features[key] != updated_features[key]:
                print(f"Mismatch in feature {key} : Original={original_features[key]}, Updated={updated_features[key]}")
                return False
    return True


if __name__ == '__main__':

    original_tfrecord_path = "/Users/jochoa4/Desktop/test_transfers/test_shard_0001-0013"
    updated_tfrecord_path = "/Users/jochoa4/Desktop/test_processed_tfrecords/test_shard_updated_0001-0013"

    original_dataset = tf.data.TFRecordDataset(original_tfrecord_path)
    updated_dataset = tf.data.TFRecordDataset(updated_tfrecord_path)

    original_iterator = iter(original_dataset)
    updated_iterator = iter(updated_dataset)

    try:
        while True:
            original_record = next(original_iterator)
            updated_record = next(updated_iterator)

            
            original_example = parse_example(original_record.numpy())
            updated_example = parse_example(updated_record.numpy())
            if compare_examples(original_example, updated_example):
                print("Example match, including label conversion")
            else:
                print(f"There is a mismatch between original and updated example")
    except StopIteration:
        print("All examples have been processed")



    # for original_record, updated_record, in zip(original_dataset.take(1500), updated_dataset.take(1500)):

    #     original_example = parse_example(original_record.numpy())
    #     updated_example = parse_example(updated_record.numpy())

    #     if compare_examples(original_example, updated_example):
    #         print("Examples match, including label conversion")
    #     else:
    #         print(f"There is a mismatch between original and updated examples")

    