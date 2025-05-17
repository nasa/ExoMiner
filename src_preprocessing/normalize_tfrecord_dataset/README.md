# Normalize a TFRecord Dataset Split

## Goal: To normalize a TFRecord dataset split by computing normalization statistics; to apply compute normalization 
statistics to nonnormalized TFRecord datasets. 
datasets.

## Applications
1. Compute normalization statistics from a TFRecord dataset.
2. Normalize a TFRecord dataset using computed normalization statistics.

## Nomenclature
- Training shards: TFRecord files with prefix 'train-'. Used to train the model.
- Validation shards: TFRecord files with prefix 'val-'. Used for model selection (e.g., early stopping).
- Test shards: TFRecord files with prefix 'test-'. Used as hold out set.
- Predict shards: TFRecord files with prefix 'predict-'. Used as set to run inference on (e.g., new set of examples).

## Requirements: An existing TFRecord dataset and a table for that dataset that can be generated using function 
`create_table_for_tfrecord_dataset()` in `src_preprocessing/utils_manipulate_tfrecords.py`. See the 
[README.md](../lc_preprocessing/README.md) for more information on TFRecord datasets.

## Steps

### 1. Compute normalization statistics

Using script `compute_normalization_stats_tfrecords.py` along with `config_compute_normalization_stats.yaml`, you can 
choose which TFRecord shards to use to compute statistics. Usually, the statistics are computed using the training 
shards.

**Outputs**: a set of files with normalization statistics for several features.

### 2. Normalize  TFRecord dataset

Using script `normalize_data_tfrecords.py` along with `config_normalize_data.yaml`, create a normalize TFRecord dataset 
based on a set of computed normalization statistics from Step 1.

**Outputs**: a normalized TFRecord dataset
 