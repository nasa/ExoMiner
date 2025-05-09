"""
Create a TFRecord data set for K-fold CV from a source TFRecord data set of already defined folds with non-normalized
data.
"""

# 3rd party
import numpy as np
import yaml
import logging
import argparse
import multiprocessing
from pathlib import Path

# local
from src_preprocessing.normalize_tfrecord_dataset.normalize_data_tfrecords import normalize_examples


def create_cv_iteration_dataset(cv_fold_fp, run_params):
    """ Create a normalized data set for a single CV iteration.

    :param cv_fold_fp: Path, directory with normalization statistics files used to normalize the data.
    :param run_params: dict, configuration parameters for the CV run
    :return:
    """

    # create directory to store normalized data
    run_params['cv_iter_dir'] = (run_params['cv_dataset_dir'] / f'cv_iter_{run_params["cv_id"]}')
    run_params['cv_iter_dir'].mkdir(exist_ok=True)

    if run_params['logger'] is not None:
        run_params['logger'].info(f'[cv_iter_{run_params["cv_id"]}] Normalizing the data...')

    # load normalization statistics
    norm_stats = {feature_grp: np.load(Path(cv_fold_fp) / norm_stats_fn, allow_pickle=True).item()
                  for feature_grp, norm_stats_fn in run_params['norm_stats'].items()}

    # normalize data using the normalization statistics
    if len(norm_stats) == 0:
        run_params['logger'].info(f'[cv_iter_{run_params["cv_id"]}] Data cannot be normalized since no normalization '
                                  f'statistics were loaded.')
        raise ValueError(f'[cv_iter_{run_params["cv_id"]}] Data cannot be normalized since no normalization '
                         f'statistics were loaded.')

    pool = multiprocessing.Pool(processes=run_params['norm_examples_params']['n_processes_norm_data'])
    jobs = [(run_params['cv_iter_dir'], file, norm_stats, run_params['norm_examples_params']['aux_params'])
            for file in config['src_tfrec_fps']]
    async_results = [pool.apply_async(normalize_examples, job) for job in jobs]
    pool.close()
    for async_result in async_results:
        async_result.get()


def create_cv_dataset(config):
    """ Create a normalized data sets for CV iterations.

    :param config: dict, configuration parameters for the CV run
    :return:
    """

    # set up logger
    config['logger'] = logging.getLogger(name=f'cv_run_rank_{config["rank"]}')
    logger_handler = logging.FileHandler(filename=config['cv_dataset_dir'] / f'cv_iter_{config["rank"]}.log', mode='w')
    logger_formatter = logging.Formatter('%(asctime)s - %(message)s')
    config['logger'].setLevel(logging.INFO)
    logger_handler.setFormatter(logger_formatter)
    config['logger'].addHandler(logger_handler)
    config['logger'].info(f'Starting run {config["cv_dataset_dir"].name}...')

    if config['process_parallel']:
        # create each CV iteration in parallel
        cv_id = config['rank']
        if config['logger'] is not None:
            config['logger'].info(f'Running CV iteration {cv_id + 1} (out of {len(config["cv_folds_fps"])})')
        config['cv_id'] = cv_id
        create_cv_iteration_dataset(
            config['cv_folds_fps'][cv_id],
            config,
        )
    else:
        # create each CV iteration sequentially
        for cv_id, cv_fold_fp in enumerate(config['cv_folds_fps']):
            if config['logger'] is not None:
                config['logger'].info(
                    f'[cv_iter_{cv_id}] Running CV iteration {cv_id + 1} (out of {len(config["cv_folds_fps"])})')
            config['cv_id'] = cv_id
            create_cv_iteration_dataset(
                cv_fold_fp,
                config,
            )

    config['logger'].info(f'Finished creating CV data set in {config["cv_dataset_dir"].name}.')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--rank', type=int, help='Job index', default=0)
    parser.add_argument('--config_fp', type=str, help='File path to YAML configuration file.',
                        default='/nobackupp19/msaragoc/work_dir/Kepler-TESS_exoplanet/codebase/src_cv/preprocessing/config_preprocess_cv_folds_predict_tfrecord_dataset.yaml')
    parser.add_argument('--output_dir', type=str, help='Output directory', default=None)

    args = parser.parse_args()

    with(open(args.config_fp, 'r')) as file:
        config = yaml.safe_load(file)

    config['rank'] = args.rank
    config['rng'] = np.random.default_rng(seed=config['rnd_seed'])

    # set paths
    if args.output_dir is not None:
        config['cv_dataset_dir'] = Path(args.output_dir)

    for path_name in ['cv_dataset_dir', 'src_tfrec_dir']:
        config[path_name] = Path(config[path_name])
    config['cv_dataset_dir'].mkdir(exist_ok=True)

    # set list of paths to TFRecord files in source TFRec directory to be normalized
    config['src_tfrec_fps'] = [fp for fp in config['src_tfrec_dir'].iterdir()
                               if fp.name.startswith('shard') and fp.suffix != '.csv']

    if config['rank'] == 0:
        # save configuration used
        np.save(config['cv_dataset_dir'] / 'run_params.npy', config)

        # save the YAML file with parameters that are YAML serializable
        with open(config['cv_dataset_dir'] / 'run_params.yaml', 'w') as cv_run_file:
            yaml.dump(config, cv_run_file, sort_keys=False)

    if config["rank"] >= len(config['cv_folds_fps']):
        print(f'Number of processes requested to run CV ({config["rank"]}) is higher than the number CV of iterations'
              f'({len(config["cv_folds_fps"])}). Ending process.')
    else:
        create_cv_dataset(config)
