"""
Create CV iterations yaml for the normalized CV dataset.
"""

# 3rd party
import yaml
from pathlib import Path
import numpy as np
import argparse


def create_cv_iterations_yaml_for_normalized_cv_dataset(data_dir: Path, src_cv_iterations_fp: Path):
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

    print(f"Created CV iterations yaml file: {data_dir / 'cv_iterations.yaml'}.")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, required=True, help='CV normalized data directory')
    parser.add_argument('--src_cv_iterations_fp', type=str, required=True, help='Path to CV iterations YAML file for the non-normalized CV dataset')

    args = parser.parse_args()

    create_cv_iterations_yaml_for_normalized_cv_dataset(Path(args.data_dir), src_cv_iterations_fp=args.src_cv_iterations_fp)
