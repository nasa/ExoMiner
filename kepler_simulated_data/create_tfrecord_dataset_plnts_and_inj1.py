""" Filter TCEs in TFRecord """

# 3rd party
from pathlib import Path
import tensorflow as tf
import shutil

#%% Filter examples in source data sets

src_tfrec_dir = Path('/Users/msaragoc/Projects/exoplanet_transit_classification/data/tfrecords/kepler/tfrecordskeplerq1q17dr25-dv_g301-l31_5tr_spline_nongapped_all_features_phases_7-20-2022_1237_data/tfrecordskeplerq1q17dr25-dv_g301-l31_5tr_spline_nongapped_all_features_phases_7-20-2022_1237_updated_stellar_ruwe_confirmedkois_adddiffimg_perimg_normdiffimg_dataset_split')
dest_tfrec_dir = Path('/Users/msaragoc/Projects/exoplanet_transit_classification/data/tfrecords/kepler/tfrecordskeplerq1q17dr25-dv_g301-l31_5tr_spline_nongapped_all_features_phases_7-20-2022_1237_data/tfrecordskeplerq1q17dr25-dv_g301-l31_5tr_spline_nongapped_all_features_phases_7-20-2022_1237_updated_stellar_ruwe_confirmedkois_adddiffimg_perimg_normdiffimg_dataset_split_plntonly')
dest_tfrec_dir.mkdir(exist_ok=True)

src_fps = [fp for fp in src_tfrec_dir.iterdir() if fp.match("*-shard-*") and not fp.name.startswith('predict')]

for src_tfrec_fp in src_fps:

    dest_tfrec_fp = dest_tfrec_dir / src_tfrec_fp.name

    with tf.io.TFRecordWriter(str(dest_tfrec_fp)) as writer:

            # iterate through the source shard
            tfrecord_dataset = tf.data.TFRecordDataset(str(src_tfrec_fp))

            for string_record in tfrecord_dataset.as_numpy_iterator():

                example = tf.train.Example()
                example.ParseFromString(string_record)

                example_label = example.features.feature['label'].bytes_list.value[0].decode('UTF')

                if example_label == 'PC':

                    writer.write(example.SerializeToString())

#%% Aggregate simulated and observed data sets

obs_dir = Path('/Users/msaragoc/Projects/exoplanet_transit_classification/data/tfrecords/kepler/tfrecordskeplerq1q17dr25-dv_g301-l31_5tr_spline_nongapped_all_features_phases_7-20-2022_1237_data/tfrecordskeplerq1q17dr25-dv_g301-l31_5tr_spline_nongapped_all_features_phases_7-20-2022_1237_updated_stellar_ruwe_confirmedkois_adddiffimg_perimg_normdiffimg_dataset_split_plntonly')
sim_dir = Path('/Users/msaragoc/Projects/exoplanet_transit_classification/data/tfrecords/kepler/kepler_q1q17dr25_simulated_data/tfrecords_kepler_q1q17dr25_simdata_11-1-2023_1019_aggregated/agg_src_data_dataset_split_inj1only')
dest_dir = Path('/Users/msaragoc/Projects/exoplanet_transit_classification/data/tfrecords/kepler/kepler_q1q17dr25_simulated_data/tfrecords_kepler_q1q17dr25_simdata_11-1-2023_1019_aggregated/dataset_split_inj1_plnt')

for src_tfrec_fp in obs_dir.iterdir():
    shutil.copy(src_tfrec_fp, dest_dir / f'{src_tfrec_fp.name}_obs')

for src_tfrec_fp in sim_dir.iterdir():
    shutil.copy(src_tfrec_fp, dest_dir / f'{src_tfrec_fp.name}_sim')
