


Basic process

Set up conditions in config_build_dataset.yaml
Run build_dataset.py
NOTE: build_dataset_t_sr.py is the logic for the previous dataset build approach - migrated to target level masking to prevent erroneous cases of example overlap that caused label noise.
    - Recommend not overwhelming number of targets per shard - can results in core dumps & interrupted building. 
    - Creates dataset in the form (assuming current config):
        dataset_name/
                    tfrecords/
                            data_tbl_0001-XXXX.csv
                                . . .
                            data_tbl_XXXX-XXXX.csv
                            raw_shard_0001-XXXX.tfrecord
                                . . .
                            raw_shard_XXXX-XXXX.tfrecord
### NOTE: At each further stage ensure the "num_shards" in relevant scripts corresponds to the XXXX value in the original dataset.
### NOTE: Ensure that if using multiprocessing - the num_processes <= number of available cores (ie matches desired number of processes)
Split dataset into train, val, test using dataset_handling/split_tfrec_dataset.py
    - Creates split dataset in the form
        dataset_name_split/
                            tfrecords/
                                        train/
                                            train_shard_0001-XXXX.tfrecord
                                        val/
                                        test/
Once split compute norm statistics on train set @ the target level using norm_pipeline/compute_train_stats.py
    - Note: Current max number of examples per tce = 4 for stats computation
    - Creates .npy file with training set statistics, used by normalization
Once stats have been computed, the split dataset can be normalized using the provided statistics using norm_pipeline/norm_tfrec_dataset_split.py
    - If using current setup, tfrecord shards will be transformed from:
        raw_train_shard_0001-XXXX.tfrecord
        ->
        norm_train_shard_0001-XXXX.tfrecord
    NOTE: The rest of the pipeline depends on shards being titled in this format - would need to accomodate for changes.
At this stage, a model can be trained using keras_model/train_model.py
    NOTE: keras_model/train_model.py is dependent on configuration parameters in keras_model/config_train.yaml.
Alternatively, the dataset can be further filtered to remove examples based on provided conditions, using dataset_handling/remove_examples_by_condition.py
    NOTE: conditions are manually defined within the process_shard function.
