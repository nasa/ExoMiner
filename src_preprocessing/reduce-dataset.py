import tensorflow as tf
from pathlib import Path
import random

total = 0
p_keep = 25

src_tfrec_dir = Path('/nobackup/cyates2/data/cvs/cv_merged_10-13-2022/tfrecords-normalized')
dest_tfrec_dir = Path('/nobackup/cyates2/data/cvs/cv_merged_10-13-2022/tfrecords-reduced')
dest_tfrec_dir.mkdir(exist_ok=True)
for src_tfrec_fp in src_tfrec_dir.iterdir():
    if "train" not in src_tfrec_fp.name:
        continue
    dest_tfrec_fp = dest_tfrec_dir / src_tfrec_fp.name

    with tf.io.TFRecordWriter(str(dest_tfrec_fp)) as writer:
        print(dest_tfrec_fp)

        tfrecord_dataset = tf.data.TFRecordDataset(str(src_tfrec_fp))
        for string_record in tfrecord_dataset.as_numpy_iterator():
            example = tf.train.Example()
            example.ParseFromString(string_record)
            if random.random() < p_keep/100:
                writer.write(example.SerializeToString())
                total += 1
print(total)


