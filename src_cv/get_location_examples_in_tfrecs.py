""" Create shard TCE table that makes tracking of location of TCEs in the shards easier. """

# 3rd party
from pathlib import Path

import pandas as pd
import tensorflow as tf

tfrec_dir_root = Path('/data5/tess_project/Data/tfrecords/Kepler/Q1-Q17_DR25/'
                      'tfrecordskeplerq1q17dr25-dv_g301-l31_5tr_spline_nongapped_all_features_paper_rbat0norm_8-20-2021_data')

src_tfrec_dir = tfrec_dir_root / \
                'tfrecordskeplerq1q17dr25-dv_g301-l31_5tr_spline_nongapped_all_features_paper_rbat0norm_8-20-2021_newval'

src_tfrec_fps = [file for file in src_tfrec_dir.iterdir() if 'shard' in file.stem and not file.suffix == '.csv']

data_to_df = []
for tfrec_fp in src_tfrec_fps:

    tfrecord_dataset = tf.data.TFRecordDataset(str(tfrec_fp))
    for string_i, string_record in enumerate(tfrecord_dataset.as_numpy_iterator()):
        example = tf.train.Example()
        example.ParseFromString(string_record)

        targetIdTfrec = example.features.feature['target_id'].int64_list.value[0]
        tceIdentifierTfrec = example.features.feature['tce_plnt_num'].int64_list.value[0]

        data_to_df.append([targetIdTfrec, tceIdentifierTfrec, tfrec_fp.name, string_i])

shards_tce_tbl = pd.DataFrame(data_to_df, columns=['target_id', 'tce_plnt_num', 'shard', 'example_i'])
shards_tce_tbl.to_csv(src_tfrec_dir / 'shards_tce_tbl.csv', index=False)

# %% test

shards_tce_tbl = pd.read_csv(
    '/data5/tess_project/Data/tfrecords/Kepler/Q1-Q17_DR25/tfrecordskeplerdr25-dv_g301-l31_6tr_spline_nongapped_flux-loe-lwks-centroid-centroidfdl-6stellar-bfap-ghost-rollband-stdts_secsymphase_correctprimarygapping_confirmedkoiperiod_data/tfrecordskeplerdr25-dv_g301-l31_6tr_spline_nongapped_flux-loe-lwks-centroid-centroidfdl-6stellar-bfap-ghost-rollband-stdts_secsymphase_correctprimarygapping_confirmedkoiperiod_starshuffle_experiment-labels-norm_nopps_secparams_prad_period/shards_tce_tbl.csv')

a = shards_tce_tbl.loc[(shards_tce_tbl['target_id'] == 1028246) & (shards_tce_tbl['tce_plnt_num'] == 2)]
