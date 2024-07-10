"""
Create a yaml file that maps 'train', 'val', 'test', and 'predict' to a list of Path objects that refer to the file
paths of TFRecord files to be used as training, validation, test, and prediction sets.
"""

# 3rd party
from pathlib import Path
import yaml

#%%

# path to directory with TFRecord data
source_tfrec_dir = Path('/Users/agiri1/Desktop/ExoBD_Datasets/tess_kepler_norm_shards_6-27-2024_1217')
# destination file path to yaml file
save_yaml_fp = source_tfrec_dir / 'datasets_fps.yaml'
# list of data sets; assumes that shards have the corresponding prefix (e.g., 'train-shard-...')
datasets_lst = ['train', 'val', 'test']

datasets_dict = {dataset: [fp for fp in source_tfrec_dir.iterdir() if fp.name.startswith(f'{dataset}-')] for dataset in datasets_lst}

with open(save_yaml_fp, 'w') as yml_file:

    yaml.dump(datasets_dict, yml_file)

print(f'Finished creating {save_yaml_fp}.')
