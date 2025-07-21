""" Prepare dataset TCE tables into folds for cross-validation. """

# 3rd party
import logging
from pathlib import Path
import numpy as np
import pandas as pd


def create_cv_folds_tables(shard_tbls_dir, dataset_tbl_fp, n_folds_eval, rnd_seed, logger, unlabeled_cats,
                           split_by='target'):
    """ Create cross-validation folds tables based on a dataset table.

    :param shard_tbls_dir: Path, directory used to save CV folds tables.
    :param dataset_tbl_fp: Path, source table
    :param n_folds_eval: int, number of folds to create for CV.
    :param rnd_seed: int, random seed
    :param logger: logging.Logger instance
    :param unlabeled_cats: bool, unlabeled categories; examples with this label are excluded from the CV dataset and
        put into a predict set.
    :param split_by: str, if 'target' examples are split by their target id, ensuring examples from same target are put
        into the same fold to minimize data leakage. If 'tce', then splits by example

    :return:
    """

    logger.info(f'Setting random seed to {rnd_seed}')

    rng = np.random.default_rng(rnd_seed)

    logger.info(f'Number of folds used for CV: {n_folds_eval}')

    # load table with TCEs used in the dataset
    dataset_tbl = pd.read_csv(dataset_tbl_fp)

    logger.info(f'Using data set table: {dataset_tbl_fp}')

    # remove TCEs not used in the dataset
    logger.info(f'Removing examples not in the data set... (total number of examples in dataset before removing '
                f'examples: {len(dataset_tbl)})')

    # # removing unlabeled examples or examples not used for evaluation
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

    working_tce_tbl.to_csv(shard_tbls_dir / f'labeled_dataset_tbl_shuffled.csv', index=False)

    # split TCE table into n fold tables (all TCEs from the same target star should be in the same table)
    logger.info(f'Split TCE table into {n_folds_eval} fold TCE tables...')

    shard_tbls_eval_dir = shard_tbls_dir / 'eval'
    shard_tbls_eval_dir.mkdir(exist_ok=True, parents=True)

    # split at the target star level
    if split_by == 'target':
        logger.info('Splitting at target star level...')
        target_star_grps = [df for _, df in working_tce_tbl.groupby('target_id')]
        target_stars_splits = np.array_split(range(len(target_star_grps)), n_folds_eval, axis=0)
        for fold_i, target_stars_split in enumerate(target_stars_splits):
            fold_tce_tbl = pd.concat(target_star_grps[target_stars_split[0]:target_stars_split[-1] + 1])
            # shuffle TCEs in each fold
            fold_tce_tbl = fold_tce_tbl.sample(frac=1, replace=False, random_state=rng,
                                               axis=0).reset_index(drop=True)
            fold_tce_tbl.to_csv(shard_tbls_eval_dir / f'labeled_dataset_tbl_fold{fold_i}.csv', index=False)
    elif split_by == 'tce':
        # split at the TCE level
        logger.info('Splitting at TCE level...')
        tce_splits = np.array_split(range(len(working_tce_tbl)), n_folds_eval, axis=0)
        for fold_i, tce_split in enumerate(tce_splits):
            fold_tce_tbl = working_tce_tbl[tce_split[0]:tce_split[-1] + 1]
            # shuffle TCEs in each fold
            fold_tce_tbl = fold_tce_tbl.sample(frac=1, replace=False, random_state=rng,
                                               axis=0).reset_index(drop=True)
            fold_tce_tbl.to_csv(shard_tbls_eval_dir / f'labeled_dataset_tbl_fold{fold_i}.csv', index=False)
    else:
        raise ValueError(f'Invalid split method: {split_by}. Choose between `target` or `tce`.')

    # tces not in the eval set
    unused_tces = ~dataset_tbl['uid'].isin(working_tce_tbl['uid'])
    if len(unused_tces) > 0:
        logger.info(f'Working with unlabeled subset: {unused_tces.sum()} examples.')
        shard_tbls_predict_dir = shard_tbls_dir / 'predict'
        shard_tbls_predict_dir.mkdir(exist_ok=True, parents=True)
        dataset_noteval = dataset_tbl.loc[unused_tces]
        dataset_noteval.to_csv(shard_tbls_dir / f'unlabeled_dataset.csv', index=False)
        tce_splits = np.array_split(range(len(dataset_noteval)), n_folds_predict, axis=0)
        for fold_i, tce_split in enumerate(tce_splits):
            fold_tce_tbl = dataset_noteval[tce_split[0]:tce_split[-1] + 1]
            fold_tce_tbl.to_csv(shard_tbls_predict_dir / f'unlabeled_dataset_tbl_fold{fold_i}.csv', index=False)

    # Check distribution of examples per fold
    logger.info(f'Checking distribution of dispositions per fold in the labeled dataset...')
    for tbl_fp in shard_tbls_eval_dir.iterdir():

        tbl = pd.read_csv(tbl_fp)
        logger.info(f'Fold {tbl_fp.name} ({len(tbl)/len(working_tce_tbl) * 100:.3f} %)')
        logger.info(f'Number of examples: {len(tbl)}')
        logger.info(f'Disposition distribution (counts):\n{tbl["label"].value_counts().to_string()}')
        logger.info(f'Disposition distribution (%):\n{(tbl["label"].value_counts(normalize=True) * 100).to_string()}')


if __name__ == '__main__':

    data_dir = Path(f'/data/directory')
    dataset_tbl_fp = Path('/tfrec_dir/shards_tbl.csv')
    rnd_seed = 24
    n_folds_eval = 5  # which is also the number of shards
    n_folds_predict = 100
    # unlabeled cats TCEs become part of the predict set; not evaluation
    unlabeled_cats = [
        'UNK',
    ]
    split_by = 'target'

    # prepare the shards for CV by splitting the TCE table into shards (n folds)
    data_dir.mkdir(exist_ok=True)

    shard_tbls_root_dir = data_dir / 'shard_tables'
    shard_tbls_root_dir.mkdir(exist_ok=True, parents=True)

    # set up logger
    logger = logging.getLogger(name='prepare_cv_tables_run')
    logger_handler = logging.FileHandler(filename=shard_tbls_root_dir / f'prepare_cv_tables_run.log', mode='w')
    logger_formatter = logging.Formatter('%(asctime)s - %(message)s')
    logger.setLevel(logging.INFO)
    logger_handler.setFormatter(logger_formatter)
    logger.addHandler(logger_handler)
    logger.info(f'Starting run...')

    create_cv_folds_tables(shard_tbls_root_dir, dataset_tbl_fp, n_folds_eval, rnd_seed, logger,
                           unlabeled_cats=unlabeled_cats, split_by=split_by)
