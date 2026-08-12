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
`create_table_for_tfrecord_dataset()` in [utils_manipulate_tfrecords.py](../utils_manipulate_tfrecords.py). See the 
[README.md](../lc_preprocessing/README.md) for more information on TFRecord datasets.      

## Steps

### 1. Create tables for each dataset split

Using script [split_dataset_table.py](.split_dataset_tcetable.py), you can define how to build the training, validation, test, and predict sets. 
This means choosing the splits among these different sets, and deciding whether to split at the target- or TCE-levels.

**Outputs**: CSV files for the dataset splits (e.g., 'trainset.csv', 'testset.csv')

### 2. Create  TFRecord split dataset

Using script [create_new_tfrecords.py](.create_new_tfrecords.py) along with [config_create_new_tfrecords.yaml](.config_create_new_tfrecords.yaml), create the TFRecord dataset based 
on the dataset split tables built in Step 1 from the same source TFRecord dataset used to create those tables.

**Outputs**: TFRecord dataset with shards prefixed by dataset name (e.g., 'train-shard-0001-of-0010')
 
### 3. Normalize  TFRecord split dataset

Using scripts from [normalize_tfrecord_dataset](../normalize_tfrecord_dataset/) (see [README.md](../normalize_tfrecord_dataset/README.md)), normalize examples across the splits using training set statistics.

**Outputs**: TFRecord dataset with same shards but now with normalized features.

### 4. Create YAML of filepaths for each split in dataset of TFRecord shards

Use script [create_data_yaml.py](.create_data_yaml.py) to create a YAML file that looks something like this:

```yaml
data_shards_fps:
- train:
  - !!python/object/apply:pathlib.PosixPath
    - /
    - path
    - to
    - tfrecord
    - directory
    - train-shard-00001-of-00129
  - !!python/object/apply:pathlib.PosixPath
    - /....
- val:
    - ....
- test:
    - ....
```