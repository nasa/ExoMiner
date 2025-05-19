"""
Script used to split TCE table into training, validation, test, and predict sets.

Input: TCE table
Output: TCE table split into different sets (e.g., training, validation, test, predict)
"""

# 3rd party
import numpy as np
import pandas as pd
from pathlib import Path
import logging


def split_tce_table_by_target_stars(tce_tbl, dataset_splits, rng, logger):
    """ Split TCE table `tce_tbl` into different sets according to `dataset_splits`. The TCEs are split at the target
    level (using column 'target_id' in `tce_tbl`).

    Args:
        tce_tbl: pandas DataFrame, TCE table
        dataset_splits: dict, maps a set (e.g., 'train') to a fraction of the TCE table
        rng: NumPy random number generator
        logger: logging.Logger object

    Returns: dict, dataset split by target id
    """

    # shuffle and split per target stars
    target_id_lst = tce_tbl['target_id'].unique()
    n_targets = len(target_id_lst)
    logger.info(f'Number of target starts in labeled set: {n_targets}')

    logger.info('Shuffling labeled set at target star level...')
    rng.shuffle(target_id_lst)

    logger.info(f'Split TCEs by target stars into {", ".join(list(dataset_splits.keys()))} set(s)...')
    curr_idx = 0
    targets_datasets_split = {dataset: np.nan * np.ones(int(n_targets * dataset_frac))
                              for dataset, dataset_frac in dataset_splits.items()}
    for dataset_i, (dataset, dataset_frac) in enumerate(dataset_splits.items()):
        start_target_idx = curr_idx
        if dataset_i == len(dataset_splits) - 1:
            last_target_idx = n_targets
        else:
            last_target_idx = int(n_targets * dataset_frac)

        targets_datasets_split[dataset] = target_id_lst[start_target_idx:last_target_idx]
        curr_idx = start_target_idx + last_target_idx

    datasets_tbls = {dataset: tce_tbl.loc[tce_tbl['target_id'].isin(targets_datasets_split[dataset])]
                     for dataset in dataset_splits}

    return datasets_tbls


if __name__ == '__main__':

    # saving directory
    dest_tfrec_dir = Path(f'/path/to/dest/tfrecord')
    # shards table for your source TFRecord dataset
    shards_tbl_fp = Path('/path/shards_tbl.csv')
    rnd_seed = 24  # random seed
    # split ratio
    dataset_splits = {
        'train': 0.8,
        # 'val': 0.1,
        'test': 0.2
    }
    # TCEs with these labels are put into the predict set
    cat_unlabeled_tces = [
        # 'UNK',
    ]

    # create dataset tables
    if sum(dataset_splits.values()) != 1:
        raise ValueError('Dataset splits should sum to 1.')

    dest_tfrec_dir.mkdir(exist_ok=True)

    # set up logger
    logger = logging.getLogger(name='split_dataset_run')
    logger_handler = logging.FileHandler(filename=dest_tfrec_dir / f'split_dataset.log', mode='w')
    logger_formatter = logging.Formatter('%(asctime)s - %(message)s')
    logger.setLevel(logging.INFO)
    logger_handler.setFormatter(logger_formatter)
    logger.addHandler(logger_handler)
    logger.info(f'Starting run...')

    logger.info(f'Setting random seed to {rnd_seed}')
    rng = np.random.default_rng(rnd_seed)

    logger.info(f'Using as source table {shards_tbl_fp}')
    shards_tbl = pd.read_csv(shards_tbl_fp)
    shards_tbl.sort_values(['target_id', 'tce_plnt_num'], ascending=True, inplace=True)
    shards_tbl.reset_index(drop=True, inplace=True)

    # get TCEs with label and without
    predict_tces = shards_tbl.loc[shards_tbl['label'].isin(cat_unlabeled_tces)]
    labeled_tces = shards_tbl.loc[~shards_tbl['label'].isin(cat_unlabeled_tces)]

    logger.info(f'TCE disposition count in labeled set:\n {labeled_tces["label"].value_counts()}')

    # labeled_tces.to_csv(dest_tfrec_dir / f'{shards_tbl_fp.stem}_labeled_tces.csv', index=False)
    if len(predict_tces) > 0:
        logger.info(f'Number of TCEs in unlabeled set: {len(predict_tces)}')
        predict_tces.to_csv(dest_tfrec_dir / f'predictset.csv', index=False)

    datasets_tbls = split_tce_table_by_target_stars(labeled_tces, dataset_splits, rng, logger)

    # shuffle TCEs in each dataset
    logger.info('Shuffling TCEs inside each labeled set...')
    for dataset, dataset_tbl in datasets_tbls.items():
        datasets_tbls[dataset] = dataset_tbl.sample(frac=1, random_state=rng, replace=False)

    for dataset, dataset_tbl in datasets_tbls.items():
        logger.info(f'Number of TCEs in dataset {dataset}: {len(dataset_tbl)}')
        logger.info(f'Label counts for dataset {dataset}:\n{dataset_tbl["label"].value_counts()}')
        logger.info(f'Saving TCE table for dataset {dataset}...')
        dataset_tbl.to_csv(dest_tfrec_dir / f'{dataset}set.csv', index=False)

    logger.info('Finished splitting data in table.')
