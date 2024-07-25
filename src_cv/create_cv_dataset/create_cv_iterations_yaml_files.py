"""
Create yaml CV iteration files for CV experiments.

1) Normalize CV dataset to prepare it to be used for a CV experiment.
2) Create CV iterations yaml for the CV experiment.
"""

# 3rd party
import yaml
from pathlib import Path

#%% Create yaml file to be used to create the normalized labeled dataset for the CV experiment from the nonnormalized dataset

data_dir = Path('/Users/msaragoc/Downloads/normalize_data_test/cv_bds_planets_keplerq1q17dr25_tess_data_7-10-2024_0951/tfrecords/eval')

cv_folds_fps = [fp for fp in data_dir.iterdir() if fp.name.startswith('shard')]

cv_iters = []  # aggregate CV iterations (each is a dictionary that maps to 'train', 'val', and 'test' sets)
for fp in cv_folds_fps:

    cv_iter = {dataset: None for dataset in ['train', 'val', 'test']}

    cv_iter['test'] = [fp]  # each CV fold shows up as test once; number of folds determines number of CV iterations

    no_test_fps = [fp_n for fp_n in cv_folds_fps if fp_n != fp]

    cv_iter['train'] = no_test_fps[:-1]
    cv_iter['val'] = [no_test_fps[-1]]  # get one of the training folds as validation fold

    cv_iters.append(cv_iter)

with open(data_dir / 'cv_iterations.yaml', 'w') as file:
    yaml.dump(cv_iters, file, sort_keys=False)

#%% Create yaml file to be used to run the CV experiment with the normalized labeled dataset

data_dir = Path('/Users/msaragoc/Downloads/normalize_data_test/cv_bds_planets_keplerq1q17dr25_tess_data_7-10-2024_0951/tfrecords/eval_normalized')
# use yaml file for CV iterations created when normalizing the data
cv_iterations_fp = Path('/Users/msaragoc/Downloads/normalize_data_test/cv_bds_planets_keplerq1q17dr25_tess_data_7-10-2024_0951/cv_folds.yaml')

with open(cv_iterations_fp, 'r') as file:
    cv_iterations = yaml.unsafe_load(file)

cv_iters = []  # aggregate CV iterations (each is a dictionary that maps to 'train', 'val', and 'test' sets)
for cv_iter_i, cv_iter in enumerate(cv_iterations):

    cv_iter = {dataset: [data_dir / f'cv_iter_{cv_iter_i}/norm_data' / dataset_fp.name for dataset_fp in dataset_fps]
               for dataset, dataset_fps in cv_iter.items()}

    cv_iters.append(cv_iter)

with open(data_dir / 'cv_iterations.yaml', 'w') as file:
    yaml.dump(cv_iters, file, sort_keys=False)
