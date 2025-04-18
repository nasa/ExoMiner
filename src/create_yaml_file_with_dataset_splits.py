"""
Script used to create a yaml file with dataset splits that can be used to train, evaluate, and test the trained models.
The yaml has the following structure:

- train:
    - /path/to/train/shard1
    - /path/to/train/shard2
    - ...
- val:
    - /path/to/val/shard1
    - /path/to/val/shard2
    - ...
- test: ...
- predict: ...

"""

# 3rd party
import yaml
from pathlib import Path

yaml_fp = Path('')
# TFRecord directory with dataset shards
tfrec_dir = Path('')
# set datasets; comment if not needed
datasets = [
    'train',
    'val',
    'test',
    # 'predict',
]

# build dictionary
dataset_splits = {dataset: [str(fp) for fp in tfrec_dir.iterdir()
                            if fp.is_file() and fp.name.startswith(dataset)] for dataset in datasets}

# save the dictionary into a YAML file
with open(yaml_fp, 'w') as file:
    yaml.dump(dataset_splits, file, sort_keys=False)
