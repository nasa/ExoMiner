"""
Create yaml CV iteration files for CV experiments.

1) Create CV iterations yaml for the non-normalized dataset.
2) Create CV iterations yaml for the normalized dataset based on the CV iterations yaml for the non-normalized dataset.
3) Create CV iterations yaml for inference on a CV dataset already normalized and using a set of trained models from a
    CV experiment.
"""

# 3rd party
import yaml
from pathlib import Path
import numpy as np
from collections import defaultdict
import re


def create_cv_iterations_yaml_for_cv_dataset(data_dir, datasets, rnd_seed=21, choose_val_method='rotating_fold'):
    """ Create CV iterations yaml file for the non-normalized dataset. In each CV iteration, a different fold is 
        used as test set. If using a validation set, the strategy used to choose the validation fold is dependent on
        `choose_val_method` argument. TFRecord filenames must follow pattern "shard-fold-{X digits}-{Y digits}-{Z digits}.tfrecord".

    Args:
        data_dir: Path, CV dataset directory
        datasets: list, datasets to create for each CV iteration. Must contain at least 'train' and 'test'. If 'val' is
            present, then a random fold from the is chosen as validation fold from the training set folds.
        rnd_seed: int, random seed
        choose_val_method: str, method used to choose the validation fold for each CV iteration. Must be either: "rotating_fold" - 
            where each fold is the validation set once; "fixed_fold" where the validation fold is fixed (except when it becomes the test 
            set fold); "random_fold" - choose fold randomly.

    Returns:

    """

    # get filepaths for the CV folds tfrecord files
    # cv_folds_fps = sorted([fp for fp in data_dir.iterdir() if fp.name.startswith('shard-')])
    
    fold_shards = defaultdict(list)
    pattern = re.compile(r'shard-fold-(\d+)_\d+-\d+\.tfrecord')

    for fp in data_dir.iterdir():
        match = pattern.match(fp.name)
        if match:
            fold_id = int(match.group(1))  # Convert to int for sorting
            fold_shards[fold_id].append(fp)

    # Sort shards within each fold
    for fold_id in fold_shards:
        fold_shards[fold_id] = sorted(fold_shards[fold_id])

    sorted_fold_ids = sorted(fold_shards.keys())

    
    if 'val' in datasets:
        assert len(fold_shards) >= 3, "Need at least 3 folds for train/val/test split"
    else:
        assert len(fold_shards) >= 2, "Need at least 2 folds for train/test split"

    cv_iters = []  # aggregate CV iterations (each is a dictionary that maps to 'train', 'val', and 'test' sets)
    
    rng = np.random.default_rng(seed=rnd_seed)
    
    for fold_i, test_fold_id in enumerate(sorted_fold_ids):

        cv_iter = {dataset: None for dataset in datasets}

        cv_iter['test'] = fold_shards[test_fold_id]  # each CV fold shows up as test once; number of folds determines number of CV iterations

        remaining_folds_ids = [remaining_fold_id for remaining_fold_id in sorted_fold_ids if remaining_fold_id != test_fold_id]

        if 'val' in datasets:  # get one of the training folds as validation fold
            
            if choose_val_method == 'fixed_fold':  # fixed validation fold (different fold when the chosen fold is the test fold)
                rng = np.random.default_rng(seed=rnd_seed)
                val_fold_id = rng.choice(remaining_folds_ids)
            elif choose_val_method == 'rotating_fold':  # rotate validation fold: pick the i-th fold from remaining ones
                val_fold_id = remaining_folds_ids[fold_i % len(remaining_folds_ids)]
            elif choose_val_method == 'random_fold':  # choose validation fold randomly
                val_fold_id = rng.choice(remaining_folds_ids)
            else:
                raise ValueError(f'Method used to choose the validation fold in each iteration must be either: "fixed_fold", "rotating_fold", or "random_fold".')
                            
            train_fold_ids = [fid for fid in remaining_folds_ids if fid != val_fold_id]
            cv_iter['val'] = fold_shards[val_fold_id]
            cv_iter['train'] = [fp for fid in train_fold_ids for fp in fold_shards[fid]]

        else:
            cv_iter['train'] = [fp for fid in remaining_folds_ids for fp in fold_shards[fid]]

        cv_iters.append(cv_iter)

    # add just shard names for each CV iteration
    cv_iters_names = []
    for cv_iter in cv_iters:
        iter_names = {}
        for split, shard_paths in cv_iter.items():
            if shard_paths is not None:
                iter_names[split] = [fp.name for fp in shard_paths]
            else:
                iter_names[split] = None
        cv_iters_names.append(iter_names)

    cv_iters_dict = {
        'num_shards_per_fold': {fold_id: len(shards_fps) for fold_id, shards_fps in fold_shards.items()},
        'num_folds': len(cv_iters),
        'random_seed': rnd_seed,
        'validation_fold_strategy': choose_val_method,
        'dataset_directory': str(data_dir),
        'datasets': datasets,
        }
    cv_iters_dict['data_shards_fps'] = cv_iters
    cv_iters_dict['data_shards_names'] = cv_iters_names
    with open(data_dir / 'cv_iterations.yaml', 'w') as file:
        yaml.dump(cv_iters_dict, file, sort_keys=False)


def create_cv_iterations_yaml_for_normalized_cv_dataset(data_dir, src_cv_iterations_fp):
    """ Create CV iterations yaml file for the normalized CV dataset based on the CV iterations yaml file for the
    non-normalized dataset `src_cv_iterations_fp`.

    Args:
        data_dir: Path, CV dataset directory with normalized data
        src_cv_iterations_fp: Path, path to the yaml file with the CV iterations for the non-normalized dataset

    Returns:

    """
    
    # create yaml file to be used to run the CV experiment with the normalized labeled dataset
    with open(src_cv_iterations_fp, 'r') as file:
        cv_iters_dict = yaml.unsafe_load(file)

    cv_iters = []  # aggregate CV iterations (each is a dictionary that maps to 'train', 'val', and 'test' sets)
    for cv_iter_i, cv_iter in enumerate(cv_iters_dict['data_shards_fps']):

        cv_iter = {dataset: [data_dir / f'cv_iter_{cv_iter_i}/norm_data' / dataset_fp.name for dataset_fp in dataset_fps]
                   for dataset, dataset_fps in cv_iter.items()}

        cv_iters.append(cv_iter)

    cv_iters_dict['data_shards_fps'] = cv_iters
    cv_iters_dict['dataset_directory'] = str(data_dir)
    with open(data_dir / 'cv_iterations.yaml', 'w') as file:
        yaml.dump(cv_iters_dict, file, sort_keys=False)


def create_cv_iterations_yaml_for_inference_on_cv_dataset(data_dir, n_cv_iterations):
    """ Create CV iterations yaml file to be used to run CV trained models on a predict dataset (already normalized).
    Args:
        data_dir: Path, CV dataset directory
        n_cv_iterations: int, number of CV iterations trained models

    Returns:

    """

    data_fps = [fp for fp in data_dir.iterdir() if fp.name.startswith('shard-')]
    cv_iters = [{'predict': data_fps} for cv_i in range(n_cv_iterations)]

    with open(data_dir / 'cv_iterations.yaml', 'w') as file:
        yaml.dump(cv_iters, file, sort_keys=False)


if __name__ == "__main__":

    # # Create CV iterations yaml file for the non-normalized dataset
    # data_dir = Path('/u/msaragoc/work_dir/Kepler-TESS_exoplanet/data/tfrecords/TESS/cv_tfrecords_tess-spoc-tces_2min-s1-s94_ffi-s36-s72-s56s69_10-30-2025_1406/tfrecords/eval/')
    # datasets = ['train', 'test', 'val']
    # choose_val_method = 'rotating_fold'
    # create_cv_iterations_yaml_for_cv_dataset(data_dir, datasets, rnd_seed=21, choose_val_method=choose_val_method)

    # Create CV iterations yaml file to be used to run the CV experiment with the normalized labeled dataset
    data_dir = Path('/u/msaragoc/work_dir/Kepler-TESS_exoplanet/data/tfrecords/TESS/cv_tfrecords_tess-spoc-tces_2min-s1-s94_ffi-s36-s72-s56s69_10-30-2025_1406/tfrecords/eval_normalized/')
    # use yaml file for CV iterations created when normalizing the data
    src_cv_iterations_fp = Path('/u/msaragoc/work_dir/Kepler-TESS_exoplanet/data/tfrecords/TESS/cv_tfrecords_tess-spoc-tces_2min-s1-s94_ffi-s36-s72-s56s69_10-30-2025_1406/tfrecords/eval/cv_iterations.yaml')
    create_cv_iterations_yaml_for_normalized_cv_dataset(data_dir, src_cv_iterations_fp)

    # # Create CV iterations yaml file to be used to run the CV trained models on a predict dataset
    # data_dir = Path('')
    # n_cv_iterations = 5
    # create_cv_iterations_yaml_for_inference_on_cv_dataset(data_dir, n_cv_iterations)