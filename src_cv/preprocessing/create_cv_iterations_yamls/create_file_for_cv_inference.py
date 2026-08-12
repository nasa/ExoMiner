"""
Create CV iterations yaml to run inference on a new dataset.
"""

# 3rd party
import yaml
from pathlib import Path
import argparse


def create_cv_iterations_yaml_for_inference_on_cv_dataset(data_dir: Path, n_cv_iterations: int):
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
    
    print(f"Created CV iterations yaml file: {data_dir / 'cv_iterations.yaml'}.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, required=True, help='Data directory to run CV inference')
    parser.add_argument('--n_cv_iterations', type=int, required=True, help='Number of CV iterations')

    args = parser.parse_args()

    create_cv_iterations_yaml_for_inference_on_cv_dataset(Path(args.data_dir), args.n_cv_iterations)
