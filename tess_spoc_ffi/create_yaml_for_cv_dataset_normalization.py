"""
Create yaml files that define the training, validation, and test sets for the different CV iterations for the two
training steps (before normalization). These yaml files are used to create a dataset of normalized data for running CV
experiments.
"""

# 3rd party
import yaml
from pathlib import Path
import numpy as np

#%% Set paths and other variables

src_data_dir = Path('/nobackupp19/msaragoc/work_dir/Kepler-TESS_exoplanet/data/tfrecords/TESS/tfrecords_tess_spoc_ffi_s36-s72_multisector_s56-s69_1-3-2025_1157_data/cv_tfrecords_tess_spoc_ffi_s36-s72_multisector_s56-s69_1-6-2025_1132/tfrecords/eval_with_2mindata_transferlearning')

rnd_seed = 42
rng = np.random.default_rng(seed=rnd_seed)

#%% Create CV iterations and save them into yaml files

src_shards_twomin_only = sorted(list(src_data_dir.glob('shard_twomin_only*')))
print(f'Found {len(src_shards_twomin_only)} shards for 2-min only targets dataset.')
src_shards_twomin_shared = sorted(list(src_data_dir.glob('shard_twomin_shared*')))
print(f'Found {len(src_shards_twomin_shared)} shards for 2-min shared targets dataset.')
src_shards_ffi = sorted(list(src_data_dir.glob('shard_ffi*')))
print(f'Found {len(src_shards_ffi)} shards for FFI dataset.')

n_cv_folds_twomin_shared = len(src_shards_twomin_shared)
n_cv_folds_ffi = len(src_shards_ffi)

n_train_shards_twomin_shared, n_test_shards_twomin_shared = 7, 1
n_train_shards_ffi, n_test_shards_ffi = 5, 1

n_val_shards_twomin_shared = n_cv_folds_ffi - n_train_shards_twomin_shared - n_test_shards_twomin_shared
n_val_shards_ffi = n_cv_folds_twomin_shared - n_train_shards_ffi - n_test_shards_ffi

datasets = ['train', 'val', 'test']
n_cv_iterations = len(src_shards_ffi)
cv_iterations_twomin, cv_iterations_ffi = [], []
for cv_iter in range(n_cv_iterations):

    print(f'Iterating on CV iteration {cv_iter}...')

    # set CV iteration for FFI
    cv_iteration_ffi_dict = {dataset: [] for dataset in datasets}
    cv_iteration_ffi_dict['test'] = [src_shards_ffi[cv_iter]]

    # choose validation folds
    cv_folds_aux = [cv_fold_fp for cv_fold_fp in src_shards_ffi if cv_fold_fp not in cv_iteration_ffi_dict['test']]
    cv_iteration_ffi_dict['val'] = list(rng.choice(cv_folds_aux, n_val_shards_ffi))

    # choose training folds
    cv_folds_aux = [cv_fold_fp for cv_fold_fp in src_shards_ffi if
                 all([cv_fold_fp not in cv_iteration_ffi_dict[dataset] for dataset in ['test', 'val']])]
    cv_iteration_ffi_dict['train'] = list(rng.choice(cv_folds_aux, n_train_shards_ffi))

    cv_iterations_ffi.append(cv_iteration_ffi_dict)

    print(f'FFI test fold: {cv_iteration_ffi_dict["test"][0].name}')
    print(f'FFI CV iteration {cv_iter} numbers:\n '
          f'Number of test folds = {len(cv_iteration_ffi_dict["test"])}\n '
          f'Number of training folds: {len(cv_iteration_ffi_dict["train"])}\n '
          f'Number of validation folds: {len(cv_iteration_ffi_dict["val"])}')

    # set CV iteration for 2-min
    cv_iteration_twomin_dict = {dataset: [] for dataset in datasets}

    # set test fold that contains the 2-min TCEs whose targets are shared with the FFI TCEs in the FFI test fold
    cv_fold_ffi_test_idx = cv_iteration_ffi_dict['test'][0].name.split('_')[-1]
    cv_iteration_twomin_dict['test'] = [cv_fold_fp for cv_fold_fp in src_shards_twomin_shared
                                        if cv_fold_fp.name.split('_')[-1] == cv_fold_ffi_test_idx]

    # choose validation folds
    cv_folds_aux = [cv_fold_fp for cv_fold_fp in src_shards_twomin_shared
                    if cv_fold_fp not in cv_iteration_twomin_dict['test']]
    cv_iteration_twomin_dict['val'] = list(rng.choice(cv_folds_aux, n_val_shards_twomin_shared))

    # choose training folds
    cv_folds_aux = [cv_fold_fp for cv_fold_fp in src_shards_twomin_shared if
                    all([cv_fold_fp not in cv_iteration_twomin_dict[dataset] for dataset in ['test', 'val']])]
    cv_iteration_twomin_dict['train'] = list(rng.choice(cv_folds_aux, n_train_shards_twomin_shared))
    # add 2-min only targets
    cv_iteration_twomin_dict['train'] += src_shards_twomin_only

    print(f'2-min test fold: {cv_iteration_twomin_dict["test"][0].name}')
    print(f'2-min CV iteration {cv_iter} numbers:\n '
          f'Number of test folds = {len(cv_iteration_twomin_dict["test"])}\n '
          f'Number of training folds: {len(cv_iteration_twomin_dict["train"])}\n '
          f'Number of validation folds: {len(cv_iteration_twomin_dict["val"])}')

    cv_iterations_twomin.append(cv_iteration_twomin_dict)

# save iterations into yaml files
with open(src_data_dir / 'cv_iterations_ffi.yaml', 'w') as yml_file:
    yaml.dump(cv_iterations_ffi, yml_file)

with open(src_data_dir / 'cv_iterations_twomin.yaml', 'w') as yml_file:
    yaml.dump(cv_iterations_twomin, yml_file)
