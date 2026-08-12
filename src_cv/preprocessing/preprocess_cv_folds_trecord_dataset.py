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
import traceback

# local
from src_preprocessing.normalize_tfrecord_dataset.compute_normalization_stats_tfrecords import compute_normalization_stats
from src_preprocessing.normalize_tfrecord_dataset.normalize_data_tfrecords import normalize_examples


def load_normalization_statistics(norm_dir: Path, centroid_names:list|None=None, scalar_params_names:list|None=None, diff_imgs_names:list|None=None) -> dict:
    """ Load normalization statistics for a single CV iteration.

    :param norm_dir: Path, directory containing NumPy files with normalization statistics files
    :param centroid_names: list, centroid features names
    :param scalar_params_names: list, scalar features names
    :param diff_imgs_names: list, difference image features names
    :return: if the files exist and normalization statistics can be loaded from them, then it returns a non-empty dictionary
    """
    
    NORM_MAP = {
        'centroid': {'fn': 'train_centroid_norm_stats.npy', 'features_names': centroid_names},
        'scalar_params': {'fn': 'train_scalarparam_norm_stats.npy', 'features_names': scalar_params_names},
        'diff_img': {'fn': 'train_diffimg_norm_stats.npy', 'features_names': diff_imgs_names},
    }

    norm_stats = {}
    any_stats_missing = False
    for feature_set_name, feature_dict in NORM_MAP.items():

        if feature_dict['features_names'] is not None:
            norm_features_fp = norm_dir / feature_dict['fn']
            if norm_features_fp.exists():
                stats_arr = np.load(norm_features_fp, allow_pickle=True).item()
                if feature_set_name == 'scalar_params':  # exclude parameters not used
                    stats_arr_filt = {k: v for k, v in stats_arr.items() if k in feature_dict['features_names']}
                    norm_stats.update({feature_set_name: stats_arr_filt})
                else:
                    norm_stats.update({feature_set_name: stats_arr})
                print(f'Normalized {feature_set_name} stats loaded from {norm_features_fp}.')
            else:
                print(f"Normalization features file {norm_features_fp} does not exist.")
                any_stats_missing = True
                break

    if any_stats_missing:
        print("Some normalization features files are missing. Skipping normalization.")
        norm_stats = {}

    return norm_stats

            
def create_cv_iteration_dataset(
    data_shards_fps: dict, 
    cv_id: int,
    cv_dataset_dir: Path,
    compute_norm_stats_params: dict,  
    diff_img_data_shape: tuple,
    n_processes_norm_data: int=1,
    logger: logging.Logger | None = None
) -> None:
    """ Create a normalized data set for a single CV iteration.

    :param data_shards_fps: dict, 'train' and 'test' keys with TFRecords folds used as training and test sets,
     respectively, for this CV iteration. 'val' is optional (if `run_params['val_from_train']` is set to True, then a
     random shard is chosen as validation fold from the training set folds).
    :param cv_id: int CV iteration ID
    :param cv_dataset_dir: Path, directory used to save data for CV iteration `cv_id`
    :param compute_norm_stats_params: dict, with parameters to compute normalization statistics for different features
    :param diff_img_data_shape: tuple, shape of the difference image data (used to reshape the flattened difference image)
    :param n_processes_norm_data: int, number of processes used to normalize the data
    :param logger: logging.Logger | None, logger object to log messages

    :return:
    """

    cv_iter_dir = (cv_dataset_dir / f'cv_iter_{cv_id}')
    cv_iter_dir.mkdir(exist_ok=True)

    norm_dir = cv_iter_dir / 'norm_stats'
    norm_dir.mkdir(exist_ok=True)

    norm_data_dir = cv_iter_dir / 'norm_data'  # create folder for normalized data set
    norm_data_dir.mkdir(exist_ok=True)

    compute_norm_stats_params['norm_dir'] = norm_dir
    
    compute_norm_stats_params['diff_img_data_shape'] = diff_img_data_shape

    data_shards_fps_eval = copy.deepcopy(data_shards_fps)

    # process data before feeding it to the model (e.g., normalize data based on training set statistics
    if logger is not None:
        logger.info(f'[cv_iter_{cv_id}] Processing data for CV iteration..')

    # load normalization statistics
    if compute_norm_stats_params['precomputed']:
        
        if logger is not None:
            logger.info(f'[cv_iter_{cv_id}] Using pre-computed normalization statistics.')

        norm_stats = {feature_grp: np.load(norm_stats_fp, allow_pickle=True).item()
                      for feature_grp, norm_stats_fp
                      in compute_norm_stats_params['precomputed'].items()}

    else:
        if logger is not None:
            logger.info(f'[cv_iter_{cv_id}] Checking if normalization statistics were already computed in {norm_dir}.')
        norm_stats = load_normalization_statistics(norm_dir, 
                                                   compute_norm_stats_params['centroidList'], 
                                                   compute_norm_stats_params['scalarParams'], 
                                                   compute_norm_stats_params['diff_imgList'])    
        if logger is not None:
            if len(norm_stats) > 0:
                logger.info(f'[cv_iter_{cv_id}] Found normalization statistics already computed in {norm_dir}.')        
            else:
                logger.info(f'[cv_iter_{cv_id}] Did not find normalization statistics already computed in {norm_dir}.')        
        
        if len(norm_stats) == 0:
            
            if logger is not None:
                logger.info(f'[cv_iter_{cv_id}] Computing normalization statistics')
            
            if 'train' not in data_shards_fps.keys():
                if logger is not None:
                    logger.error(f'[cv_iter_{cv_id}] No training data found in the provided data shards in {cv_iter_dir} to compute normalization statistics.')

                raise ValueError(f' [cv_iter_{cv_id}] No training data found in the provided data shards in {cv_iter_dir} to compute normalization statistics.')
            
            compute_normalization_stats(data_shards_fps['train'], compute_norm_stats_params)

            if logger is not None:
                logger.info(f'[cv_iter_{cv_id}] Computed normalization statistics. Loading those statistics from {norm_dir}...')

            norm_stats = load_normalization_statistics(norm_dir, 
                                                   compute_norm_stats_params['centroidList'], 
                                                   compute_norm_stats_params['scalarParams'], 
                                                   compute_norm_stats_params['diff_imgList'])  

            if logger is not None:
                logger.info(f'[cv_iter_{cv_id}] Loaded normalization statistics from {norm_dir}.')          

    if logger is not None:
        logger.info(f'[cv_iter_{cv_id}] Normalizing the data...')

    # normalize data using the normalization statistics
    if len(norm_stats) == 0:
        if logger is not None:
            logger.info(f'[cv_iter_{cv_id}] Data cannot be normalized since no normalization '
                                  f'statistics were loaded.')
        raise ValueError(f'[cv_iter_{cv_id}] Data cannot be normalized since no normalization '
                         f'statistics were loaded.')

    # prepare jobs
    files = np.concatenate(list(data_shards_fps_eval.values()))
    jobs = [(norm_data_dir, file, norm_stats) for file in files]

    # create the pool
    with multiprocessing.Pool(processes=n_processes_norm_data) as pool:
        
        # submit jobs asynchronously
        async_results = [pool.apply_async(normalize_examples, job) for job in jobs]

        # track progress with tqdm
        for result in tqdm(async_results, desc="Normalizing TFRecord file", total=len(jobs)):    
            try:
                result.get()
            except Exception as e:
                if logger is not None:
                    logger.exception(f'[cv_iter_{cv_id}] Process failed for one of the jobs:\n{e}')
                else:
                    print(f"[cv_iter_{cv_id}] Process failed for one of the jobs: {e}")
                    traceback.print_exc()
               
                raise

    # # compute sample weights
    # if run_params['training']['sample_weights']:
    #     compute_sample_weights(data_shards_fps_norm, run_params)
    
    if logger is not None:
        logger.info(f'[cv_iter_{cv_id}] Done normalizing data for CV iteration.')


def create_cv_dataset(config: dict) -> None:
    """ Create a normalized data sets for CV iterations.

    :param config: dict, configuration parameters for the CV run
    :return:
    """

    # set up logger
    logger = logging.getLogger(name=f'cv_run_rank_{config["rank"]}')
    logger_handler = logging.FileHandler(filename=config['cv_log_dir'] / f'cv_iter_{config["rank"]}.log', mode='w')
    logger_formatter = logging.Formatter('%(asctime)s - %(message)s')
    logger.setLevel(logging.INFO)
    logger_handler.setFormatter(logger_formatter)
    logger.addHandler(logger_handler)
    logger.info(f'Starting run {config["cv_dataset_dir"].name}...')

    if config['process_parallel']:  # create each CV iteration in parallel
        
        cv_id = config['rank'] 
        
        logger.info(f'Running CV iteration {cv_id + 1} (out of {len(config["data_shards_fps"])})')
        
        config['cv_id'] = cv_id
        
        create_cv_iteration_dataset(
            data_shards_fps=config['data_shards_fps'][cv_id],
            cv_id=cv_id,
            cv_dataset_dir=config['cv_dataset_dir'],
            compute_norm_stats_params=config['compute_norm_stats_params'],
            diff_img_data_shape=config['diff_img_data_shape'],
            n_processes_norm_data=config['norm_examples_params']['n_processes_norm_data'],
            logger=logger,
        )
    else:
        # create each CV iteration sequentially
        for cv_id, data_shards_fps in enumerate(config['data_shards_fps']):
            
            logger.info(f'[cv_iter_{cv_id}] Running CV iteration {cv_id + 1} (out of {len(config["data_shards_fps"])})')
                
            config['cv_id'] = cv_id

            create_cv_iteration_dataset(
                data_shards_fps,
                cv_id=cv_id,
                cv_dataset_dir=config['cv_dataset_dir'],
                compute_norm_stats_params=config['compute_norm_stats_params'],
                diff_img_data_shape=config['diff_img_data_shape'],
                n_processes_norm_data=config['norm_examples_params']['n_processes_norm_data'],
                logger=logger,
        )

    logger.info(f'Finished creating CV data set in {config["cv_dataset_dir"].name}.')


def check_shards_fps(data_shards_fps: list) -> None:
    """ Checks the structure of the list of CV iterations dicts containing the list of TFRecord filepaths for each dataset.

    :params data_shards_fps: list of dicts, one per CV iteration. Each dict contains dataset keys 'train', 'val', 'test' that map 
        the list of TFRecord filepaths for that corresponding dataset.
    """

    for cv_id, data_shards in enumerate(data_shards_fps):
        if 'train' not in data_shards:
            raise KeyError(f'CV iteration {cv_id} does not have a "train" dataset.')
        
        for dataset, dataset_fps in data_shards.items():
            for dataset_fp in dataset_fps:
                if not Path(dataset_fp).exists():
                    raise FileNotFoundError(f'File {dataset_fp} does not exist for dataset {dataset} in CV iteration {cv_id}.')

def main() -> None:

    multiprocessing.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('--rank', type=int, help='Job index', default=0)
    parser.add_argument('--config_fp', type=str, help='File path to YAML configuration file.', required=True)
    parser.add_argument('--output_dir', type=str, help='Output directory', required=True)
    parser.add_argument('--log_dir', type=str, help='Log directory', default=None)

    args = parser.parse_args()

    with open(args.config_fp, 'r') as file:
        config = yaml.safe_load(file)

    config['rank'] = args.rank
    config['rng'] = np.random.default_rng(seed=config['rnd_seed'])

    # set paths
    config['cv_dataset_dir'] = Path(args.output_dir)
    if args.log_dir is not None:
        config['cv_log_dir'] = Path(args.log_dir)
    else:
        config['cv_log_dir'] = Path(args.output_dir)
        
    for path_name in ['cv_dataset_dir', 'cv_folds_fp']:
        config[path_name] = Path(config[path_name])
    config['cv_dataset_dir'].mkdir(exist_ok=True)

    # load cv iterations dictionary
    with open(config['cv_folds_fp'], 'r') as file:
        config['data_shards_fps'] = yaml.unsafe_load(file)['data_shards_fps']

    check_shards_fps(config['data_shards_fps'])

    if config['rank'] == 0:
        # save configuration used
        np.save(config['cv_dataset_dir'] / 'run_params.npy', config)

        # save the YAML file with parameters that are YAML serializable
        # (Remember to sanitize this dictionary if yaml.dump throws an error on Path/rng objects!)
        with open(config['cv_dataset_dir'] / 'run_params.yaml', 'w') as cv_run_file:
            yaml.dump(config, cv_run_file, sort_keys=False)

    if config["rank"] >= len(config['data_shards_fps']):
        print(f'Number of processes requested to run CV ({config["rank"]}) is higher than the number CV of iterations '
              f'({len(config["data_shards_fps"])}). Ending process.')
    else:
        create_cv_dataset(config)


if __name__ == '__main__':

    main()

