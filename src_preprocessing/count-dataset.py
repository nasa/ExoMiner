import tensorflow as tf
from pathlib import Path
import random

total = 0
target = "train"

src_tfrec_dir = Path('/nobackup/cyates2/data/cvs/cv_merged_10-13-2022/tfrecords-reduced')
for src_tfrec_fp in src_tfrec_dir.iterdir():
    if target not in src_tfrec_fp.name:
        continue

    tfrecord_dataset = tf.data.TFRecordDataset(str(src_tfrec_fp))
    for string_record in tfrecord_dataset.as_numpy_iterator():
        total += 1
print(total)


