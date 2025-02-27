"""
Generate the file containing all the filepaths for the training, test, and validation shards for datasets that don't
use cross-validation.
"""

# 3rd party
import yaml
from pathlib import Path

data_dir = Path('/Users/agiri1/Desktop/ExoBD_Datasets/tess_kepler_norm_shards_6-27-2024_1217')

cv_iter = {dataset: None for dataset in ['train', 'val', 'test']}

cv_iter['test'] = [shard for shard in data_dir.iterdir() if 'test' in shard.name]
cv_iter['val'] = [shard for shard in data_dir.iterdir() if 'val' in shard.name]
cv_iter['train'] = [shard for shard in data_dir.iterdir() if 'train' in shard.name]

with (open(f'/Users/agiri1/Desktop/ExoBD_Datasets/tess_kepler_norm_shards_6-27-2024_1217/dataset_fps.yaml', 'w+') as
      file):
    yaml.dump(cv_iter, file, sort_keys=False)