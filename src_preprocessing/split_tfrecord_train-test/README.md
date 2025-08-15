# Creating a TFRecord Dataset Split

## Goal: To split a TFRecord dataset into training, validation, test, and prediction sets.

## Applications
1. Train and evaluate a model on a simple dataset split.
2. Run inference on a split.

## Nomenclature
- Training shards: TFRecord files with prefix 'train-'. Used to train the model.
- Validation shards: TFRecord files with prefix 'val-'. Used for model selection (e.g., early stopping).
- Test shards: TFRecord files with prefix 'test-'. Used as hold out set.
- Predict shards: TFRecord files with prefix 'predict-'. Used as set to run inference on (e.g., new set of examples).

## Requirements
An existing TFRecord dataset and a table for that dataset that can be generated using function 
`create_table_for_tfrecord_dataset()` in `src_preprocessing/utils_manipulate_tfrecords.py`. See the 
[README.md](../lc_preprocessing/README.md) for more information on TFRecord datasets.      

## Steps

### 1. Create tables for each dataset split

Using script `split_dataset_table.py`, you can define how to build the training, validation, test, and predict sets. 
This means choosing the splits among these different sets, and deciding whether to split at the target- or TCE-levels.

**Outputs**: CSV files for the dataset splits (e.g., 'trainset.csv', 'testset.csv')

### 2. Create  TFRecord split dataset

Using script `create_new_tfrecords.py` along with `config_create_new_tfrecords.yaml`, create the TFRecord dataset based 
on the dataset split tables built in Step 1 from the same source TFRecord dataset used to create those tables.

**Outputs**: TFRecord dataset with shards prefixed by dataset name (e.g., 'train-shard-0001-of-0010')
 