"""
Create a TFRecord data set for K-fold CV from a source TFRecord data set of already defined folds with non-normalized
data.
"""

# 3rd party
import numpy as np
import yaml
import logging
import copy
import argparse
import multiprocessing
from pathlib import Path
from tqdm import tqdm

# local
from src_preprocessing.normalize_tfrecord_dataset.compute_normalization_stats_tfrecords import compute_normalization_stats
from src_preprocessing.normalize_tfrecord_dataset.normalize_data_tfrecords import normalize_examples


def load_normalization_statistics(norm_dir, centroid_names=None, scalar_params_names=None, diff_imgs_names=None):
    """ Load normalization statistics for a single CV iteration.

    :param norm_dir: Path, directory containing NumPy files with normalization statistics files
    :parma centroid_names: list, centroid features names
    :param scalar_params_names: list, scalar features names
    :param diff_imgs_names: list, difference image features names
    :return: if the files exist and normalization statistics can be loaded from them, then it returns a dictionary; otherwise it returns `None`
    """
    
    try:
        norm_stats = {}
        
        if centroid_names is not None:
                norm_stats.update({'centroid': np.load(norm_dir/ 'train_centroid_norm_stats.npy', allow_pickle=True).item()})
        if scalar_params_names is not None:
            scalar_params_norm_info = np.load(norm_dir / 'train_scalarparam_norm_stats.npy', allow_pickle=True).item()
            scalar_params_norm_info = {k: v for k, v in scalar_params_norm_info.items() if k in scalar_params_names}
            norm_stats.update({'scalar_params': scalar_params_norm_info})
        if diff_imgs_names is not None:
            norm_stats.update({'diff_img': np.load(norm_dir / 'train_diffimg_norm_stats.npy', allow_pickle=True).item()})
    except Exception as e:
        print(f'Warning when loading normalization statistics:\n {e}')
        norm_stats = None
    
    return norm_stats

            
def create_cv_iteration_dataset(data_shards_fps, run_params):
    """ Create a normalized data set for a single CV iteration.

    :param data_shards_fps: dict, 'train' and 'test' keys with TFRecords folds used as training and test sets,
     respectively, for this CV iteration. 'val' is optional (if `run_params['val_from_train']` is set to True, then a
     random shard is chosen as validation fold from the training set folds).
    :param run_params: dict, configuration parameters for the CV run
    :return:
    """

    run_params['cv_iter_dir'] = (run_params['cv_dataset_dir'] / f'cv_iter_{run_params["cv_id"]}')
    run_params['cv_iter_dir'].mkdir(exist_ok=True)

    run_params['norm_dir'] = run_params['cv_iter_dir'] / 'norm_stats'
    run_params['norm_dir'].mkdir(exist_ok=True)

    run_params['norm_data_dir'] = run_params['cv_iter_dir'] / 'norm_data'  # create folder for normalized data set
    run_params['norm_data_dir'].mkdir(exist_ok=True)

    run_params['compute_norm_stats_params']['norm_dir'] = run_params['norm_dir']
    
    run_params['compute_norm_stats_params']['diff_img_data_shape'] = run_params['diff_img_data_shape']

    data_shards_fps_eval = copy.deepcopy(data_shards_fps)

    # process data before feeding it to the model (e.g., normalize data based on training set statistics
    if run_params['logger'] is not None:
        run_params['logger'].info(f'[cv_iter_{run_params["cv_id"]}] Processing data for CV iteration..')

    # load normalization statistics
    if run_params['compute_norm_stats_params']['precomputed']:
        norm_stats = {feature_grp: np.load(norm_stats_fp, allow_pickle=True).item()
                      for feature_grp, norm_stats_fp
                      in run_params['compute_norm_stats_params']['precomputed'].items()}

    else:
        if run_params['logger'] is not None:
            run_params['logger'].info(f'[cv_iter_{run_params["cv_id"]}] Checking if normalization statistics were already computed in {run_params["norm_dir"]}.')
        norm_stats = load_normalization_statistics(run_params['norm_dir'], 
                                                   run_params['compute_norm_stats_params']['centroidList'], 
                                                   run_params['compute_norm_stats_params']['scalarParams'], 
                                                   run_params['compute_norm_stats_params']['diff_imgList'])    
        if run_params['logger'] is not None and norm_stats is not None:
            run_params['logger'].info(f'[cv_iter_{run_params["cv_id"]}] Found normalization statistics already computed in {run_params["norm_dir"]}.')        
        
        if norm_stats is None:
            
            if run_params['logger'] is not None:
                run_params['logger'].info(f'[cv_iter_{run_params["cv_id"]}] Computing normalization statistics')
                
            # p = multiprocessing.Process(target=compute_normalization_stats,
            #                             args=(
            #                                 data_shards_fps['train'],
            #                                 run_params['compute_norm_stats_params'],
            #                             ))
            # p.start()
            # p.join()
            compute_normalization_stats(data_shards_fps['train'], run_params['compute_norm_stats_params'])


            norm_stats = load_normalization_statistics(run_params['norm_dir'], 
                                                   run_params['compute_norm_stats_params']['centroidList'], 
                                                   run_params['compute_norm_stats_params']['scalarParams'], 
                                                   run_params['compute_norm_stats_params']['diff_imgList'])            

    if run_params['logger'] is not None:
        run_params['logger'].info(f'[cv_iter_{run_params["cv_id"]}] Normalizing the data...')

    # normalize data using the normalization statistics
    if len(norm_stats) == 0:
        run_params['logger'].info(f'[cv_iter_{run_params["cv_id"]}] Data cannot be normalized since no normalization '
                                  f'statistics were loaded.')
        raise ValueError(f'[cv_iter_{run_params["cv_id"]}] Data cannot be normalized since no normalization '
                         f'statistics were loaded.')

    # prepare jobs
    files = np.concatenate(list(data_shards_fps_eval.values()))
    jobs = [(run_params['norm_data_dir'], file, norm_stats) for file in files]
    # for job in tqdm(jobs, desc='Normalizing TFRecord file', total=len(jobs)):
    #     normalize_examples(*job)
    # create the pool
    with multiprocessing.Pool(processes=run_params['norm_examples_params']['n_processes_norm_data']) as pool:
        
        # submit jobs asynchronously
        async_results = [pool.apply_async(normalize_examples, job) for job in jobs]

        # track progress with tqdm
        for result in tqdm(async_results, desc="Normalizing TFRecord file", total=len(jobs)):    
            try:
                result.get()
            except Exception as e:
                print(f"[ERROR] Subprocess failed: {e}")

    # # compute sample weights
    # if run_params['training']['sample_weights']:
    #     compute_sample_weights(data_shards_fps_norm, run_params)
    
    if run_params['logger'] is not None:
        run_params['logger'].info(f'[cv_iter_{run_params["cv_id"]}] Done normalizing data for CV iteration.')


def create_cv_dataset(config):
    """ Create a normalized data sets for CV iterations.

    :param config: dict, configuration parameters for the CV run
    :return:
    """

    # set up logger
    config['logger'] = logging.getLogger(name=f'cv_run_rank_{config["rank"]}')
    logger_handler = logging.FileHandler(filename=config['cv_log_dir'] / f'cv_iter_{config["rank"]}.log', mode='w')
    logger_formatter = logging.Formatter('%(asctime)s - %(message)s')
    config['logger'].setLevel(logging.INFO)
    logger_handler.setFormatter(logger_formatter)
    config['logger'].addHandler(logger_handler)
    config['logger'].info(f'Starting run {config["cv_dataset_dir"].name}...')

    if config['process_parallel']:
        # create each CV iteration in parallel
        cv_id = config['rank']
        if config['logger'] is not None:
            config['logger'].info(f'Running CV iteration {cv_id + 1} (out of {len(config["data_shards_fps"])})')
        config['cv_id'] = cv_id
        create_cv_iteration_dataset(
            config['data_shards_fps'][cv_id],
            config,
        )
    else:
        # create each CV iteration sequentially
        for cv_id, data_shards_fps in enumerate(config['data_shards_fps']):
            if config['logger'] is not None:
                config['logger'].info(
                    f'[cv_iter_{cv_id}] Running CV iteration {cv_id + 1} (out of {len(config["data_shards_fps"])})')
            config['cv_id'] = cv_id
            create_cv_iteration_dataset(
                data_shards_fps,
                config,
            )

    config['logger'].info(f'Finished creating CV data set in {config["cv_dataset_dir"].name}.')


if __name__ == '__main__':

    multiprocessing.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('--rank', type=int, help='Job index', default=0)
    parser.add_argument('--config_fp', type=str, help='File path to YAML configuration file.',
                        default=None)
    parser.add_argument('--output_dir', type=str, help='Output directory', default=None)
    parser.add_argument('--log_dir', type=str, help='Log directory', default=None)

    args = parser.parse_args()

    with(open(args.config_fp, 'r')) as file:
        config = yaml.safe_load(file)

    config['rank'] = args.rank
    config['rng'] = np.random.default_rng(seed=config['rnd_seed'])

    # set paths
    if args.output_dir is not None:
        config['cv_dataset_dir'] = Path(args.output_dir)
    if args.output_dir is not None:
        config['cv_log_dir'] = Path(args.log_dir)
    else:
        config['cv_log_dir'] = Path(args.output_dir)
    for path_name in ['cv_dataset_dir', 'cv_folds_fp']:
        config[path_name] = Path(config[path_name])
    config['cv_dataset_dir'].mkdir(exist_ok=True)

    # load cv iterations dictionary
    with(open(config['cv_folds_fp'], 'r')) as file:  # read default YAML configuration file
        config['data_shards_fps'] = yaml.unsafe_load(file)['data_shards_fps']

    if config['rank'] == 0:
        # save configuration used
        np.save(config['cv_dataset_dir'] / 'run_params.npy', config)

        # save the YAML file with parameters that are YAML serializable
        with open(config['cv_dataset_dir'] / 'run_params.yaml', 'w') as cv_run_file:
            yaml.dump(config, cv_run_file, sort_keys=False)

    if config["rank"] >= len(config['data_shards_fps']):
        print(f'Number of processes requested to run CV ({config["rank"]}) is higher than the number CV of iterations'
              f'({len(config["data_shards_fps"])}). Ending process.')
    else:
        create_cv_dataset(config)
