"""
Generate the filepaths to the train, test, and validation shards with Cross-validation. Done after the cross-validation
shards are generated.
"""

# 3rd party
import yaml
from pathlib import Path

data_dir = Path('/Users/agiri1/Desktop/ExoBD_Datasets/shard_tables/')
config_path = Path('/Users/agiri1/Desktop/ExoBD_Datasets/shard_tables/tfrecords/cv_folds_tfrecords.yaml')

with open(config_path, 'r') as src_file:
    cv_iterations = yaml.unsafe_load(src_file)

for i, paths in enumerate(cv_iterations):

    cv_iter = {dataset: None for dataset in ['train', 'val', 'test']}

    cv_iter['test'] = [Path(f"/Users/agiri1/Desktop/ExoBD_Datasets/shard_tables/cv_iter_{i}/norm_data/{path.parts[-1]}")
                       for path in paths['test']]
    cv_iter['val'] = [Path(f"/Users/agiri1/Desktop/ExoBD_Datasets/shard_tables/cv_iter_{i}/norm_data/{path.parts[-1]}")
                      for path in paths['val']]

    no_test_fps = [Path(f"/Users/agiri1/Desktop/ExoBD_Datasets/shard_tables/cv_iter_{i}/norm_data/{path.parts[-1]}")
                   for path in paths['train']]

    cv_iter['train'] = no_test_fps

    with open(f'/Users/agiri1/Desktop/ExoBD_Datasets/shard_tables/cv_iter_{i}/dataset_fps.yaml', 'w+') as file:
        yaml.dump(cv_iter, file, sort_keys=False)
