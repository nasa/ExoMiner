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
import math


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
            last_target_idx = start_target_idx + int(n_targets * dataset_frac)

        targets_datasets_split[dataset] = target_id_lst[start_target_idx:last_target_idx]
        curr_idx = last_target_idx

    datasets_tbls = {dataset: tce_tbl.loc[tce_tbl['target_id'].isin(targets_datasets_split[dataset])]
                     for dataset in dataset_splits}

    return datasets_tbls


if __name__ == '__main__':

    # saving directory
    dest_tfrec_dir = Path(f'/nobackupp19/msaragoc/work_dir/Kepler-TESS_exoplanet/data/tfrecords/TESS/tfrecords_tess-spoc-2min_tces_s1-s94_10-11-2025_0858_agg_diffimg_train-test-split_10-22-2025_1458_split-tables')
    # shards table for your source TFRecord dataset
    shards_tbl_fp = Path('/nobackupp19/msaragoc/work_dir/Kepler-TESS_exoplanet/data/tfrecords/TESS/tfrecords_tess-spoc-2min_tces_s1-s94_10-11-2025_0858_agg_diffimg/shards_tbl.csv')
    rnd_seed = 24  # random seed
    # split ratio
    dataset_splits = {
        'train': 0.7,
        'val': 0.2,
        'test': 0.1,
    }
    # TCEs with these labels are put into the predict set
    cat_unlabeled_tces = [
        'UNK',
    ]

    # create dataset tables
    if not math.isclose(sum(dataset_splits.values()), 1.0, rel_tol=1e-9):
        raise ValueError(f'Dataset splits should sum to 1: currently summing up to {sum(dataset_splits.values())}.')

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
    
    # saving counts of examples per label into CSV file
    dataset_counts = []
    for dataset, dataset_tbl in datasets_tbls.items():
        dataset_counts.append(dataset_tbl['label'].value_counts().rename(dataset))
    dataset_counts = pd.concat(dataset_counts, axis=1).fillna(0).astype(int)
    
    # add metadata about dataset split
    dataset_counts.attrs['dataset'] = str(shards_tbl_fp)
    dataset_counts.attrs['split'] =  str(dataset_splits)
    dataset_counts.attrs['random_seed'] =  rnd_seed
    dataset_counts.attrs['created'] = str(pd.Timestamp.now().floor('min'))
    with open(dest_tfrec_dir / 'datasets_counts.csv', "w") as f:
        for key, value in dataset_counts.attrs.items():
            f.write(f"# {key}: {value}\n")
        dataset_counts.to_csv(f)

    logger.info('Finished splitting data in table.')
