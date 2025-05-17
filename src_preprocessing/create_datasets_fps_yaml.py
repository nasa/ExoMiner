"""
Creates yaml file that specifies which TFRecord shards should be used as training, validation, test, predict sets.
Assumes that shards were already split into these datasets and have prefix 'train-', 'val-', 'test-', 'predict-'.
"""

# 3rd party
from pathlib import Path
import tensorflow as tf
import yaml


def create_datasets_fps_yaml(tfrec_dir, datasets):
    """ Create yaml with list of TFRecord shards filepaths for each dataset in `datasets`. Assumes that shards have
    prefix {dataset}-shard- where dataset is an element of datasets (e.g. 'train', 'val', 'test', 'predict').

    Args:
        tfrec_dir: Path, TFRecord dataset directory
        datasets: list, datasets

    Returns:
    """

    datasets_fps_dict = {dataset: list(tfrec_dir.glob(f'{dataset}-shard*')) for dataset in datasets}

    with open(tfrec_dir / 'datasets_fps.yaml', 'w') as yml_file:
        yaml.dump(datasets_fps_dict, yml_file)

if __name__ == '__main__':

    tf.config.set_visible_devices([], 'GPU')

    dest_tfrec_dir = Path('/home6/msaragoc/work_dir/Kepler-TESS_exoplanet/data/tfrecords/TESS/tfrecords_tess_spoc_2min_s1-s88_4-25-2025_1536_data/tfrecords_tess_spoc_2min_s1-s88_4-25-2025_1536_agg_bdslabels_diffimg_targetsnotshared_train-test-split_5-16-2025_1157_normalized')
    datasets_in_tfrec_dir = ['train', 'test']

    print(f'Creating yaml file with datasets filepaths...')
    create_datasets_fps_yaml(dest_tfrec_dir, datasets_in_tfrec_dir)

    print('Done.')