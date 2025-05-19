# Module *src_cv* 

### Goals

The code in `src_cv` is used for running cross-validation (CV) experiments. Contrary to the dataset built using a single 
train-test split as found in [`split_tfrecord_train-test`](../src_preprocessing/split_tfrecord_train-test), this module 
creates a CV dataset that a more complete analysis of model behavior. By running K-fold CV, we have a measure of the 
variance in performance metrics besides a more robust set of estimates obtained through averaging across CV folds.

## Code

- [`preprocessing`](preprocessing): contains scripts related to building a CV dataset for training/evaluation and for 
inference.
- [`train`](train): contains scripts related to running a training/evaluation experiment on a CV dataset. Returns 
metrics for both single and ensemble models in each CV iteration, and predictions table for ensemble model.
- [`predict`](predict): contains scripts related to running inference experiment on a dataset using trained models from 
a CV experiment. Returns predictions table for that dataset.
- [`postprocessing`](postprocessing): contains scripts related to aggregating predictions from the different CV 
iteration and computing performance metric estimates.

## Nomenclature
- Training shards: TFRecord files with prefix 'train-'. Used to train the model.
- Validation shards: TFRecord files with prefix 'val-'. Used for model selection (e.g., early stopping).
- Test shards: TFRecord files with prefix 'test-'. Used as hold out set.
- Predict shards: TFRecord files with prefix 'predict-'. Used as set to run inference on (e.g., new set of examples).

## Create a cross-validation dataset

### Requirements 
An existing TFRecord dataset and a table for that dataset that can be generated using function 
`create_table_for_tfrecord_dataset()` in `src_preprocessing/utils_manipulate_tfrecords.py`. See the 
[README.md](../src_preprocessing/lc_preprocessing/README.md) for more information on TFRecord datasets.      

### Steps

#### 1. Create tables for each dataset split

Use script [`create_cv_folds_tables.py`](preprocessing/create_cv_folds_tables.py) to create the tables of examples for 
each CV iteration. You can choose which examples will be part of the labeled and unlabeled datasets.

**Outputs**: CSV files for the CV iteration sets and prediction set

#### 2. Use CV iteration tables and source dataset to create a CV dataset

Use script [`create_cv_tfrecord_dataset_from_tables.py`](preprocessing/create_cv_tfrecord_dataset_from_tables.py) to 
create CV dataset using CV iteration tables from Step 1 and a source TFRecord dataset.

**Outputs**: CV TFRecord dataset

#### 3. Created normalized CV dataset for training and evaluation

At this point, the CV dataset consists of several CV folds. For each iteration, we need to choose at least a training 
and a test sets (validation is optional but usually useful for model selection, specially for neural networks). Use 
function [`create_cv_iterations_yaml_for_cv_dataset()`](preprocessing/create_cv_iterations_yaml_files.py) to create a YAML 
file that shows the train-(val optional)-test split for each CV iteration. Then 
[`Submitjob_create_data_for_cv.pbs`](preprocessing/Submitjob_create_data_for_cv.pbs), 
[`preprocessing_job.sh`](preprocessing/preprocessing_job.sh), [`preprocess_cv_folds_tfrecord_dataset.py`](preprocessing/preprocess_cv_folds_trecord_dataset.py) and 
[`config_preprocess_cv_folds_tfrecord_dataset.yaml`](preprocessing/config_preprocess_cv_folds_tfrecord_dataset.yaml) can 
be used to create a normalized CV TFRecord dataset (1. compute normalization statistics based on training set for each 
CV iteration; 2. normalize the data for the iteration based on those statistics). Use function 
[`create_cv_iterations_yaml_for_normalized_cv_dataset()`](preprocessing/create_cv_iterations_yaml_files.py) to create a 
YAML file that shows the train-(val optional)-test split for each CV iteration.

**Outputs**: normalized CV TFRecord dataset with shards prefixed by dataset name (e.g., 'train-shard-0001-of-0010'). In 
this case, the source CV dataset will have as many copies as CV iterations exist.
 
### Create a cross-validation dataset for inference

When running inference with a set of models obtained from a CV experiment, use scripts 
[`Submitjob_create_data_for_cv.pbs`](preprocessing/Submitjob_create_data_for_cv.pbs), 
[`preprocessing_jobs`](preprocessing/preprocessing_job.sh), 
[`preprocess_cv_folds_predict_tfrecord_dataset.py`](preprocessing/preprocess_cv_folds_predict_trecord_dataset.py), and 
[`config_preprocess_cv_folds_predict_tfrecord_dataset.yaml`](preprocessing/config_preprocess_cv_folds_predict_tfrecord_dataset.yaml).

**Outputs**: normalized CV TFRecord dataset with predict shards. In this case, the source predict dataset will have as many 
copies as CV iterations exist, since each CV iteration has their own statistics. Performance will be computed for each 
CV iteration.

#### Additional information

You can use script [`create_data_for_cv_local_sequential.sh`](preprocessing/create_data_for_cv_local_sequential.sh) if 
opting for no parallelization.
