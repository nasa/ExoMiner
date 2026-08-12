""" Prepare dataset TCE tables into folds for cross-validation. 

Requires shards CSVtable for source TFRecord dataset with following columns:
- target_id: int, target star ID
- uid; str, TCE unique ID
- label: str, TCE label
"""

# 3rd party
import logging
from pathlib import Path
import numpy as np
import pandas as pd


def create_table_folds_statistics(folds_tbls_fps, save_fp, metadata=None):
    """Create table with folds statistics at the disposition level. Both counts and percentages for each disposition.

    :param list folds_tbls_fps: list of Paths to the fold tables
    :param Path save_fp: filepath where this table is saved to
    :param dict metadata: metadata to add to header of table
    :return:
    """
    
    folds_tbls_lst = []
    for tbl_fp in folds_tbls_fps:

        fold_id = tbl_fp.stem.split('_')[-1]
        
        fold_tbl = pd.read_csv(tbl_fp)
        
        fold_counts = fold_tbl['label'].value_counts().reset_index(name=fold_id).set_index('label')
        fold_counts.loc['total'] = fold_counts[fold_id].sum()
        fold_counts_normalized = fold_tbl['label'].value_counts(normalize=True).reset_index(name=f'{fold_id} %').set_index('label') * 100
        fold_counts_normalized.loc['total'] = fold_counts_normalized[f'{fold_id} %'].sum()
        
        fold_counts_agg = pd.concat([fold_counts, fold_counts_normalized], axis=1).fillna(0) 
        
        folds_tbls_lst.append(fold_counts_agg)
    
    folds_cnts = pd.concat(folds_tbls_lst, axis=1).fillna(0)
    
    # add metadata
    folds_cnts.attrs['description'] = 'Contains statistics about the distribution of examples across cross-validation folds'
    folds_cnts.attrs['created'] = str(pd.Timestamp.now().floor('min'))
    folds_cnts.attrs['created by'] = 'src_cv/preprocessing/create_cv_folds_tables.py'
    if metadata:
        for k, v in metadata.items():
            folds_cnts.attrs[k] = v
    
    with open(save_fp, "w") as f:
        for key, value in folds_cnts.attrs.items():
            f.write(f"# {key}: {value}\n")
        folds_cnts.to_csv(f, index=True)

     
def split_tces(tce_tbl, n_folds, rng=None):
    """Split TCEs without taking into account their targets.

    :param pandas DataFrame tce_tbl: TCE table
    :param int n_folds: number of folds
    :param NumPy rng: random generator
    :return list: each item in the list is a list of pandas DataFrames containing the tables for each fold
    """
    
    # shuffle TCEs so they are not ordered by target and can end up in different folds
    tce_tbl = tce_tbl.sample(frac=1, random_state=rng).reset_index(drop=True)
    
    tce_splits = np.array_split(range(len(tce_tbl)), n_folds, axis=0)
    folds = [[tce_tbl.iloc[tce_split]] for tce_split in tce_splits]
        
    return folds

def split_tces_by_target(tce_tbl, n_folds):
    """Split TCEs by target into folds. Each fold contains all TCEs that share the same target.

    :param pandas DataFrame tce_tbl: TCE table
    :param int n_folds: number of folds
    :return list: each item in the list is a list of pandas DataFrames containing the target tables with 
        their corresponding TCEs
    """
    
    target_star_grps = [df for _, df in tce_tbl.groupby('target_id')]
    target_stars_splits = np.array_split(range(len(target_star_grps)), n_folds, axis=0)
    
    folds = [[target_star_grps[i] for i in target_stars_split] for target_stars_split in target_stars_splits]
    
    return folds

def split_tces_by_target_balanced(tce_tbl, n_folds, planet_disps=['CP', 'KP']):
    """Split TCEs by target into folds. Each fold contains all TCEs that share the same target.
        To ensure balanced folds, a greedy approach is used based on the number of planet TCEs 
        that are found for each target

    :param pandas DataFrame tce_tbl: TCE table
    :param int n_folds: number of folds
    :param list planet_disps: planet dispositions
    :return list: each item in the list is a list of pandas DataFrames containing the target tables with 
        their corresponding TCEs
    """
    
    if tce_tbl['label'].isin(planed_disps).sum() == 0:
        raise ValueError(f'No TCEs found with planet disposition: {planed_disps}')
    
    target_groups = []
    for target_id, group in tce_tbl.groupby('target_id'):
        planet_score = group['label'].isin(planet_disps).sum()  # count planet-labeled TCEs
        target_groups.append((target_id, group, planet_score))

    # initialize folds and their planet score totals
    folds = [[] for _ in range(n_folds)]
    fold_scores = np.array([0] * n_folds)
    # greedy assignment: place each target group in the fold with the lowest current planet score
    for target_id, group, score in target_groups:
        # get folds with smallest number of planets
        min_score = np.min(fold_scores)
        min_score_fold_idxs = np.where(fold_scores == min_score)[0]
        # get fold with the smallest size
        min_score_folds_small_len_idx = np.argmin([len(folds[idx]) for idx in min_score_fold_idxs])
        # add to this fold
        folds[min_score_fold_idxs[min_score_folds_small_len_idx]].append(group)
        fold_scores[min_score_fold_idxs[min_score_folds_small_len_idx]] += score

    return folds

def create_cv_folds_tables(shard_tbls_dir, dataset_tbl_fp, dataset_tbl_cols, n_folds_eval, rnd_seed, unlabeled_cats,
                           split_by='target greedy balanced', planet_disps=['KP', 'CP']):
    """ Create cross-validation folds tables based on a dataset table.

    :param shard_tbls_dir: Path, directory used to save CV folds tables.
    :param dataset_tbl_fp: Path, source table
    :param dataset_tbl_cols: dict, required columns in dataset table and their data types
    :param n_folds_eval: int, number of folds to create for CV.
    :param rnd_seed: int, random seed
    :param unlabeled_cats: bool, unlabeled categories; examples with this label are excluded from the CV dataset and
        put into a predict set.
    :param split_by: str, if 'target' examples are split by their target id, ensuring examples from same target are put
        into the same fold to minimize data leakage. If 'target greedy balanced', a greedy approach is used to try to 
        balance the number of planet TCEs in each fold. If 'tce', then splits by example.
    :param list planet_disps: planet dispositions. Needed when splitting by target using a greedy approach

    :return:
    """

    shard_tbls_root_dir.mkdir(exist_ok=True, parents=True)

    # set up logger
    logger = logging.getLogger(name='prepare_cv_tables_run')
    logger_handler = logging.FileHandler(filename=shard_tbls_root_dir / 'prepare_cv_tables_run.log', mode='w')
    logger_formatter = logging.Formatter('%(asctime)s - %(message)s')
    logger.setLevel(logging.INFO)
    logger_handler.setFormatter(logger_formatter)
    logger.addHandler(logger_handler)
    logger.info('Starting run...')
    logger.info(f'Setting random seed to {rnd_seed}')

    rng = np.random.default_rng(rnd_seed)

    logger.info(f'Number of folds used for CV: {n_folds_eval}')

    # load table with TCEs used in the dataset
    if dataset_tbl_fp.exists():
        dataset_tbl = pd.read_csv(dataset_tbl_fp, usecols=dataset_tbl_cols.keys(), dtype=dataset_tbl_cols)
    else:
        raise FileNotFoundError(f'Dataset table not found in {str(dataset_tbl_fp)}. Use `create_table_for_tfrecord_dataset()` in '
                                f'src_preprocessing/utils_manipulate_tfrecords.py to create dataset table for the TFRecord dataset.')
    
    logger.info(f'Using data set table: {dataset_tbl_fp}')

    # remove TCEs not used in the dataset
    logger.info(f'Removing examples not in the data set... (total number of examples in dataset before removing '
                f'examples: {len(dataset_tbl)})')

    # removing unlabeled examples or examples not used for evaluation
    used_tces = ~dataset_tbl['label'].isin(unlabeled_cats)
    working_tce_tbl = dataset_tbl.loc[used_tces]

    logger.info(f'Removing unlabeled examples or examples not used for evaluation: {used_tces.sum()} examples left.')

    logger.info(f'Total number of examples in dataset after removing examples: {len(working_tce_tbl)}')

    # shuffle per target stars
    logger.info('Shuffling TCE table per target stars...')
    target_star_grps = [df for _, df in working_tce_tbl.groupby('target_id')]
    logger.info(f'Number of target stars: {len(target_star_grps)}')
    rng.shuffle(target_star_grps)
    working_tce_tbl = pd.concat(target_star_grps).reset_index(drop=True)

    working_tce_tbl.to_csv(shard_tbls_dir / 'labeled_dataset_tbl_shuffled.csv', index=False)

    # split TCE table into n fold tables (all TCEs from the same target star should be in the same table)
    logger.info(f'Split TCE table into {n_folds_eval} fold TCE tables...')

    shard_tbls_eval_dir = shard_tbls_dir / 'eval'
    shard_tbls_eval_dir.mkdir(exist_ok=True, parents=True)

    if split_by == 'target':  # split at the target star level
    
        logger.info('Splitting at target star level...')

        folds = split_tces_by_target(working_tce_tbl, n_folds_eval)
    
    elif split_by == 'target greedy balanced':  # split at the target star level balanced planets
        
        logger.info('Splitting at target star level using a greedy approach to balance number of planets in each fold...')
        
        folds = split_tces_by_target_balanced(working_tce_tbl, n_folds_eval, planet_disps)
                        
    elif split_by == 'tce':  # split at the TCE level
        
        logger.info('Splitting at TCE level...')
        
        folds = split_tces(working_tce_tbl, n_folds_eval, rng)
        
    else:
        raise ValueError(f'Invalid split method: {split_by}.')

    # save each fold
    for fold_i, fold_groups in enumerate(folds):
        # shuffle TCEs within the fold
        fold_tce_tbl = pd.concat(fold_groups).sample(frac=1, random_state=rng).reset_index(drop=True)
        fold_tce_tbl.to_csv(shard_tbls_eval_dir / f'labeled_dataset_tbl_fold{fold_i}.csv', index=False)
        
    # tces not in the eval set
    unused_tces = ~dataset_tbl['uid'].isin(working_tce_tbl['uid'])
    if len(unused_tces) > 0:
        logger.info(f'Working with unlabeled subset: {unused_tces.sum()} examples.')
        shard_tbls_predict_dir = shard_tbls_dir / 'predict'
        shard_tbls_predict_dir.mkdir(exist_ok=True, parents=True)
        dataset_noteval = dataset_tbl.loc[unused_tces]
        dataset_noteval.to_csv(shard_tbls_dir / 'unlabeled_dataset.csv', index=False)
        tce_splits = np.array_split(range(len(dataset_noteval)), n_folds_predict, axis=0)
        for fold_i, tce_split in enumerate(tce_splits):
            fold_tce_tbl = dataset_noteval[tce_split[0]:tce_split[-1] + 1]
            fold_tce_tbl.to_csv(shard_tbls_predict_dir / f'unlabeled_dataset_tbl_fold{fold_i}.csv', index=False)

    # check distribution of examples per fold
    logger.info('Checking distribution of dispositions per fold in the labeled dataset...')
    folds_tbls_fps = list(shard_tbls_eval_dir.glob('labeled_dataset_tbl_fold*.csv'))
    
    metadata = {
        'source dataset table': str(dataset_tbl_fp),
        'random seed': rnd_seed,
        'split by method': split_by
    }
    create_table_folds_statistics(folds_tbls_fps, shard_tbls_eval_dir / 'folds_statistics.csv', metadata=metadata)


if __name__ == '__main__':

    shard_tbls_root_dir = Path('/home6/msaragoc/work_dir/Kepler-TESS_exoplanet/data/tfrecords/TESS/cv_tfrecords_tess-spoc-2min_tces_s1-s98_stricter-phot-pcs_7-14-2026_1417/shard_tables')
    dataset_tbl_fp = Path('/home6/msaragoc/work_dir/Kepler-TESS_exoplanet/data/tfrecords/TESS/tfrecords_tess-spoc-2min_tces_s1-s98_stricter-phot-pcs_7-14-2026_1417/shards_tbl.csv')
    rnd_seed = 24
    n_folds_eval = 10  # which is also the number of shards
    n_folds_predict = 100
    # unlabeled cats TCEs become part of the predict set; not evaluation
    unlabeled_cats = [
        'UNK',
    ]
    split_by = 'target greedy balanced'  # 'target split'
    planed_disps = ['PC']  # needed when splitting using a target greedy balanced approach
    dataset_tbl_cols = {
        'uid': str,
        # 'uid_obs_type': str,
        'label': str,
        'target_id': int,
        'sectors_observed': str,
    }

    create_cv_folds_tables(shard_tbls_root_dir, dataset_tbl_fp, dataset_tbl_cols, n_folds_eval, rnd_seed,
                           unlabeled_cats=unlabeled_cats, split_by=split_by, planet_disps=planed_disps)
