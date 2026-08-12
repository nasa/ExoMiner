"""Create YAML file with datasets filepaths split into training, test, and optionally validation and predict sets."""

# imports
import yaml
import argparse
from pathlib import Path


def create_data_yaml_for_tfrecord_dataset(data_dir, validation_set_from_val_shards=True, prediction_set_from_predict_shards=False):
    """ Create dataset filepaths yaml file based on a TFRecord dataset of data split into at least training and test sets using prefixes 
    `train-shard` and `test-shard`. If `validation_set_from_val_shards` is True, then a validation set is also included assuming there are 
    validation shards with prefix `val-shard`. Similar for predict set if `prediction_set_from_predict_shards` is set to True and there are 
    shards with prefix `predict-shard`. A value error is raised if those shards are not found.

    Args:
        data_dir: Path, CV dataset directory with normalized data
        validation_set_from_val_shards: bool, if True it sets also path for validation set shards
        prediction_set_from_predict_shards: bool, if True it sets also path for predict set shards
    Returns:

    """
    
    # directory has to contain at least `train` and `test` shards
    datasets_fps = {dataset_name: list(data_dir.glob(f'{dataset_name}-shard*')) for dataset_name in ['train', 'test']}  
    
    if len(datasets_fps['train']) == 0:
        raise ValueError(f'No train shards with prefix `train-shard` were found in {str(data_dir)}')
    
    if len(datasets_fps['test']) == 0:
        raise ValueError(f'No test shards with prefix `test-shard` were found in {str(data_dir)}')
    
    if validation_set_from_val_shards:
        datasets_fps['val'] = list(data_dir.glob('val-shard*'))
        if len(datasets_fps['val']) == 0:
            raise ValueError(f'No validation shards with prefix `val-shard` were found in {str(data_dir)}')

    if prediction_set_from_predict_shards:
        datasets_fps['predict'] = list(data_dir.glob('predict-shard*'))
        if len(datasets_fps['test']) == 0:
            raise ValueError(f'No predict shards with prefix `predict-shard` were found in {str(data_dir)}')

    yaml_dict = {
        'data_shards_fps': [datasets_fps],
        'dataset_directory': str(data_dir),
    }
    with open(data_dir / 'datasets_fps.yaml', 'w') as file:
        yaml.dump(yaml_dict, file, sort_keys=False)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='TFRecord dataset directory')
    args = parser.parse_args()
    
    validation_set_from_val_shards = True
    prediction_set_from_predict_shards = True
    
    create_data_yaml_for_tfrecord_dataset(Path(args.data_dir), 
                                          validation_set_from_val_shards=validation_set_from_val_shards, 
                                          prediction_set_from_predict_shards=prediction_set_from_predict_shards)
    