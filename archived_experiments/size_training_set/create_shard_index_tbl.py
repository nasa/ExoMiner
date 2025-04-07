import pandas as pd
from pathlib import Path
import tensorflow as tf

src_tfrec_dir = Path('/data5/tess_project/Data/tfrecords/Kepler/Q1-Q17_DR25/'
                     'tfrecordskeplerq1q17dr25-dv_g301-l31_5tr_spline_nongapped_all_features_paper_rbat0norm_8-20-2021_data/tfrecordskeplerq1q17dr25-dv_g301-l31_5tr_spline_nongapped_all_features_paper_rbat0norm_8-20-2021_starshuffle_experiment-labels')

src_tfrec_fps = [fp for fp in src_tfrec_dir.iterdir() if 'val-shard' in fp.name and fp.suffix != '.csv']

data_to_tbl = []
for src_tfrec_fp in src_tfrec_fps:

    tfrecord_dataset = tf.data.TFRecordDataset(str(src_tfrec_fp))
    shard_idx = 0

    for string_record in tfrecord_dataset.as_numpy_iterator():
        example = tf.train.Example()
        example.ParseFromString(string_record)

        target_id = example.features.feature['target_id'].int64_list.value[0]
        tce_plnt_num = example.features.feature['tce_plnt_num'].int64_list.value[0]

        data_to_tbl.append([target_id, tce_plnt_num, src_tfrec_fp.name, shard_idx])
        shard_idx += 1

shard_tbl = pd.DataFrame(data=data_to_tbl, columns=['target_id', 'tce_plnt_num', 'shard_name', 'shard_idx'])
shard_tbl.set_index('shard_idx', inplace=True)
shard_tbl.to_csv(src_tfrec_dir / 'val_set_shards.csv', index=True)
