"""
Create data dictionary from TFRecord data set which is required for running the occlusion and model PC replacement
methods.
"""

# 3rd party
from pathlib import Path
import logging
from datetime import datetime
import numpy as np
import tensorflow as tf

# root directory for storing extracted data
root_dir = Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/experiments/explainability/tfrecord_to_dict_data')
# destination directory for this run
dest_data_dir = root_dir / f'run_{datetime.now().strftime("%m-%d-%Y_%H%M")}'
dest_data_dir.mkdir(exist_ok=True)

# set up logger
logger = logging.getLogger(name=f'log_data_from_tfrec')
logger_handler = logging.FileHandler(filename=dest_data_dir / f'data_from_tfrec_to_dict.log', mode='w')
logger_formatter = logging.Formatter('%(asctime)s - %(message)s')
logger.setLevel(logging.INFO)
logger_handler.setFormatter(logger_formatter)
logger.addHandler(logger_handler)
logger.info(f'Starting run {dest_data_dir.name}...')

# source TFRecord data set
src_tfrec_dir = Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/data/tfrecords/kepler/tfrecordskeplerdr25-dv_g301-l31_spline_nongapped_newvalpcs_tessfeaturesadjs_12-1-2021_experiment-normalized')
logger.info(f'Using as source TFRecord data set {src_tfrec_dir}')

# choose features to be extracted
features_to_extract = [
    # flux related features
  'global_flux_view_fluxnorm',
  'local_flux_view_fluxnorm',
   'transit_depth_norm',
  # odd-even flux related features
  'local_flux_odd_view_fluxnorm',
  'local_flux_even_view_fluxnorm',
  'odd_se_oot_norm',
  'even_se_oot_norm',
  # centroid related features
  'global_centr_view_std_noclip',
  'local_centr_view_std_noclip',
  'tce_fwm_stat_norm',
  'tce_dikco_msky_norm',
  'tce_dikco_msky_err_norm',
  'tce_dicco_msky_norm',
  'tce_dicco_msky_err_norm',
  'mag_cat',
  'local_weak_secondary_view_max_flux-wks_norm',
  'tce_maxmes_norm',
  'wst_depth_norm',
  'tce_albedo_stat_norm',
  'tce_ptemp_stat_norm',
  # other diagnostic parameters
  'boot_fap_norm',
  'tce_cap_stat_norm',
  'tce_hap_stat_norm',
  'tce_rb_tcount0n_norm',
  # stellar parameters
  'tce_sdens_norm',
  'tce_steff_norm',
  'tce_smet_norm',
  'tce_slogg_norm',
  'tce_smass_norm',
  'tce_sradius_norm',
  # tce parameters
  'tce_prad_norm',
  'tce_period_norm',
]

logger.info(f'Features extracted from TFRecord data set: {features_to_extract}')

examples_fields = [
  'target_id',
  'tce_plnt_num',
  'tce_period',
  'tce_duration',
  'original_label',
  'label',
]

label_map = {
  'PC': 1,
  'AFP': 0,
  'NTP': 0
}

datasets = [
  'train',
  'test',
  'val',
]

logger.info(f'Data sets for which to extract the data: {datasets}')

for dataset in datasets:
  logger.info(f'Iterating through data set {dataset}...')

  src_tfrec_fps = [fp for fp in src_tfrec_dir.iterdir() if fp.name.startswith(dataset)]
  n_src_tfrecs = len(src_tfrec_fps)

  logger.info(f'Found {n_src_tfrecs} files for data set {dataset}.')
  logger.info(f'Extracting data from those files...')

  data_dicts_for_dataset = []
  for src_tfrec_fp_i, src_tfrec_fp in enumerate(src_tfrec_fps):

    logger.info(f'Reading and extracting data from file {src_tfrec_fp} ({src_tfrec_fp_i + 1} out of {n_src_tfrecs})...')

    tfrecord_dataset = tf.data.TFRecordDataset(str(src_tfrec_fp))

    data_dict = {
      'example_info': {field: [] for field in examples_fields},
      'features': {feat: [] for feat in features_to_extract}
    }
    for string_record in tfrecord_dataset.as_numpy_iterator():

      example = tf.train.Example()
      example.ParseFromString(string_record)

      for field in examples_fields:

        if field == 'original_label':
          continue

        if field in ['label']:
          example_label = example.features.feature[field].bytes_list.value[0].decode('utf-8')
          example_label_id = label_map[example_label]
          data_dict['example_info'][field].append(example_label_id)
          data_dict['example_info']['original_label'].append(example_label)
        elif field in ['target_id', 'tce_plnt_num']:
          data_dict['example_info'][field].append(example.features.feature[field].int64_list.value[0])
        else:
          data_dict['example_info'][field].append(example.features.feature[field].float_list.value[0])

      for feat in features_to_extract:
        if feat in ['mag_cat']:
          data_dict['features'][feat].append(example.features.feature[feat].int64_list.value)
        else:
          data_dict['features'][feat].append(example.features.feature[feat].float_list.value)

    data_dicts_for_dataset.append(data_dict)

  # aggregate data from all the source files into a single dictionary
  logger.info(f'Aggregating data across TFRecord data set for data set {dataset}...')
  dataset_dict = {
      'example_info': {field: [] for field in examples_fields},
      'features': {feat: [] for feat in features_to_extract}
    }

  for field in examples_fields:
    for data_dict in data_dicts_for_dataset:
      dataset_dict['example_info'][field].extend(data_dict['example_info'][field])
  for feat in features_to_extract:
    for data_dict in data_dicts_for_dataset:
      dataset_dict['features'][feat].extend(data_dict['features'][feat])
    dataset_dict['features'][feat] = np.array(dataset_dict['features'][feat])

  # save data dict for data set as a NumPy file
  logger.info(f'Finished iterating and extracting data for data set {dataset}')
  dict_dataset_fp = dest_data_dir / f'all_{dataset}_data.npy'
  logger.info(f'Saving data for data set {dataset} in {dict_dataset_fp}...')
  np.save(dict_dataset_fp, dataset_dict)
  logger.info(f'Saved data for data set {dataset} in {dict_dataset_fp}...')
