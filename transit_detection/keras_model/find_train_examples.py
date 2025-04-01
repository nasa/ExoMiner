import tensorflow as tf
import glob

tfrec_pattern = "/nobackupp27/jochoa4/work_dir/data/datasets/TESS_exoplanet_dataset_11-12-2024_split_norm/tfrecords/train/norm_train_shard_*-8611"

tfrec_files = glob.glob(tfrec_pattern)

def count_records_in_tfrecord(filename):
    count = 0
    dataset = tf.data.TFRecordDataset(filename)
    for _ in dataset:
        count += 1
    return count

if __name__ == "__main__":
    print(f"Found {len(tfrec_files)} files matching pattern")
    total_count = sum(count_records_in_tfrecord(file) for file in tfrec_files)
    print(f"Total examples in dataset: {total_count}")