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
import re


def create_cv_iterations_yaml_for_cv_dataset(data_dir, datasets, rnd_seed=21):
    """ Create CV iterations yaml file for the non-normalized dataset.

    Args:
        data_dir: Path, CV dataset directory
        datasets: list, datasets to create for each CV iteration. Must contain at least 'train' and 'test'. If 'val' is
            present, then a random fold from the training is chosen as validation fold from the training set folds.

    Returns:

    """

    # get filepaths for the CV folds tfrecord files
    cv_folds_fps = [fp for fp in data_dir.iterdir() if fp.name.startswith('shard-')]

    cv_iters = []  # aggregate CV iterations (each is a dictionary that maps to 'train', 'val', and 'test' sets)
    for fp in cv_folds_fps:

        cv_iter = {dataset: None for dataset in datasets}

        cv_iter['test'] = [fp]  # each CV fold shows up as test once; number of folds determines number of CV iterations

        no_test_fps = [fp_n for fp_n in cv_folds_fps if fp_n != fp]

        if 'val' in datasets:  # get one of the training folds as validation fold
            rng = np.random.default_rng(seed=rnd_seed)
            cv_iter['train'] = rng.choice(no_test_fps, len(no_test_fps) - 1, replace=False).tolist()
            cv_iter['val'] = [fp for fp in no_test_fps if fp not in cv_iter['train']]
        else:
            cv_iter['train'] = no_test_fps

        cv_iters.append(cv_iter)

    with open(data_dir / 'cv_iterations.yaml', 'w') as file:
        yaml.dump(cv_iters, file, sort_keys=False)


def create_cv_iterations_yaml_for_normalized_cv_dataset(data_dir, src_cv_iterations_fp):
    """ Create CV iterations yaml file for the normalized CV dataset based on the CV iterations yaml file for the
    non-normalized dataset  `src_cv_iterations_fp`.

    Args:
        data_dir: Path, CV dataset directory with normalized data
        src_cv_iterations_fp: Path, path to the yaml file with the CV iterations for the non-normalized dataset

    Returns:

    """
    # Create yaml file to be used to run the CV experiment with the normalized labeled dataset

    with open(src_cv_iterations_fp, 'r') as file:
        cv_iterations = yaml.unsafe_load(file)

    cv_iters = []  # aggregate CV iterations (each is a dictionary that maps to 'train', 'val', and 'test' sets)
    for cv_iter_i, cv_iter in enumerate(cv_iterations):

        cv_iter = {dataset: [data_dir / f'cv_iter_{cv_iter_i}/norm_data' / dataset_fp.name for dataset_fp in dataset_fps]
                   for dataset, dataset_fps in cv_iter.items()}

        cv_iters.append(cv_iter)

    with open(data_dir / 'cv_iterations.yaml', 'w') as file:
        yaml.dump(cv_iters, file, sort_keys=False)


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


def create_cv_iterations_yaml_update_root_dir(src_fp, dest_fp, dest_root_dir):
    """ Create CV dataset folds yaml file based on a source CV dataset folds yaml file. This is useful when running a
    new CV experiment using a CV dataset that was built in a different system/directory than the current copy.

    Args:
        src_fp: Path, source CV dataset folds yaml file
        dest_fp: Path, destination CV dataset folds yaml file
        dest_root_dir: Path, destination root directory for CV dataset

    Returns:

    """

    with open(src_fp, 'r') as file:
        cv_config = yaml.unsafe_load(file)

    new_cv_config = []
    for cv_iter in cv_config:
        new_cv_iter = {dataset: [dest_root_dir / str(dataset_fp)[re.search(r'cv_iter_[0-9]',
                                                                           str(dataset_fp)).start():]
                                 for dataset_fp in dataset_fps] for dataset, dataset_fps in cv_iter.items()}
        new_cv_config.append(new_cv_iter)

    with open(dest_fp, 'w') as config_file:
        yaml.dump(new_cv_config, config_file)


if __name__ == "__main__":

    # Create CV iterations yaml file for the non-normalized dataset
    data_dir = Path('')
    datasets = ['train', 'test', 'val']
    create_cv_iterations_yaml_for_cv_dataset(data_dir, datasets, rnd_seed=21)

    # Create CV iterations yaml file to be used to run the CV experiment with the normalized labeled dataset
    data_dir = Path( '')
    # use yaml file for CV iterations created when normalizing the data
    src_cv_iterations_fp = Path('')
    create_cv_iterations_yaml_for_normalized_cv_dataset(data_dir, src_cv_iterations_fp)

    # Create CV iterations yaml file to be used to run the CV trained models on a predict dataset
    data_dir = Path('')
    n_cv_iterations = 5
    create_cv_iterations_yaml_for_inference_on_cv_dataset(data_dir, n_cv_iterations)