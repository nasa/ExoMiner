"""
Script used to split dataset TCE table into training, validation, test, and (optional) predict sets.

Input: dataset table
Output: dataset table split into different sets (e.g., training, validation, test, predict)
"""

# 3rd party
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import math


def split_tces_by_target_and_temporal_split(dataset_tbl, temporal_split='mission_phases'):
    """ Assign each target (TIC) to a temporal split based on the LAST sector observed. 
    Ensures all TCEs from the same target remain in the same split.

    Expected columns:
        - target_id (int)
        - sectors_observed (str): binary string indicating observed sectors

    using mission phases (i.e., PM, EM1, EM2, EM3) splits:
        - 0: max sector 1–26
        - 1:   max sector 27–38
        - 2:  max sector 39-96
    using forward-expanding rolling windows splits:
        e.g., train PM, validate early EM1, test late EM1
        - 0: max sector 1–26
        - 1:   max sector 27–32
        - 2:  max sector 33-38
    
    :param pandas DataFrame dataset_tbl: TCEs dataset table 
    :param str temporal_split: either `mission_phases` or `forward_expanding_rolling_window`
    :return dict: train/val/test map to corresponding pandas DataFrame
    """
    
    mission_phases = ['PM', 'EM1', 'EM2']
    rolling_windows_dict = {
        0: [26, 32, 38],  # train PM, val early EM1, test late EM1
        1: [38, 52, 70],  # train PM+EM1, val early EM2, test mid EM2
        2: [52, 70, 96]  # train PM+EM1+early EM2, val mid EM2, test late EM2
    }
    rolling_window_idx = 0
    
    for req_col in ['sectors_observed', 'target_id']:
        if req_col not in dataset_tbl.columns:
            raise ValueError(f'Column `{req_col}` not found in dataset table.')

    # compute max sector from binary observation string
    def get_last_sector(binary_str):
        # rightmost '1' corresponds to highest sector index (0-based)
        idx = binary_str.rfind('1')
        return idx + 1 if idx != -1 else 0

    # compute per-target last sector
    # target_last_sector = (
    #     dataset_tbl
    #     .groupby("target_id")["sectors_observed"]
    #     .apply(lambda x: get_last_sector(x.iloc[0]))  # all rows for this target have same sectors_observed
    # )
    target_last_sector = (
        dataset_tbl
        .groupby("target_id")["sectors_observed"]
        .apply(lambda col: max(get_last_sector(s) for s in col)))

    # assign split based on last observed sector
    def assign_split_mission_phases(last_sector):
        if last_sector <= 26:  # PM
            return 'PM'
        elif last_sector <= 38:  # EM 1
            return 'EM1'
        elif last_sector <= 96:  # EM 2
            return 'EM2'
        else:  
            raise ValueError(f'Last sector {last_sector} observed is out of bounds.')
    
    def assign_split_rolling_window(last_sector):
        
        for end_sector_i, end_sector in enumerate(rolling_windows_dict[rolling_window_idx]):
            if last_sector <= end_sector: 
                return end_sector_i
        else:  
            return -1
            # raise ValueError(f'Last sector {last_sector} observed is out of bounds for: {rolling_windows_dict[rolling_window_idx]}')
        
    if temporal_split == 'mission_phases':
        target_split = target_last_sector.apply(assign_split_mission_phases)
        split_mapping = mission_phases
    elif temporal_split == 'forward_expanding_rolling_window':
        target_split = target_last_sector.apply(assign_split_rolling_window)
        split_mapping = sorted(range(len(rolling_windows_dict[rolling_window_idx])))
        
    dataset_tbl["temporal_split"] = dataset_tbl["target_id"].map(target_split)
    
    # set datasets tables
    datasets_tbls = {
        dataset: dataset_tbl.loc[dataset_tbl['temporal_split'] == split] 
        for dataset, split in zip(['train', 'val', 'test'], split_mapping)
        }
    
    return datasets_tbls


def split_tce_table_by_target_stars(dataset_tbl, dataset_splits, rng, logger):
    """ Split TCE table `tce_tbl` into different sets according to `dataset_splits`. The TCEs are split at the target
    level (using column 'target_id' in `tce_tbl`).

    Args:
        dataset_tbl: pandas DataFrame, TCE table
        dataset_splits: dict, maps a set (e.g., 'train') to a fraction of the TCE table
        rng: NumPy random number generator
        logger: logging.Logger object

    Returns: dict, dataset split by target id
    """
    
    for req_col in ['target_id']:
        if req_col not in dataset_tbl.columns:
            raise ValueError(f'Column `{req_col}` not found in dataset table.')

    # shuffle and split per target stars
    target_id_lst = dataset_tbl['target_id'].unique()
    n_targets = len(target_id_lst)
    logger.info(f'Number of stars in labeled set: {n_targets}')

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

    datasets_tbls = {dataset: dataset_tbl.loc[dataset_tbl['target_id'].isin(targets_datasets_split[dataset])]
                     for dataset in dataset_splits}

    return datasets_tbls


def main(dest_tfrec_dir, shards_tbl_fp, split_method, dataset_splits=None, cat_unlabeled_tces=None, rnd_seed=42):
    """Main function used to create train/val/test splits.

    :param Path dest_tfrec_dir: destination directory
    :param Path shards_tbl_fp: filepath to source dataset table
    :param str split_method: split method used
    :param dict dataset_splits: maps 'train', 'val', 'test to corresponding fraction splits. Not used when split method is 'temporal'. Defaults to None
    :param list cat_unlabeled_tces: labels considered for the unlabeled dataset. If empty list or `None`, the unlabeled set is not created. Defaults to None
    :param int rnd_seed: random seed number used for shuffling, defaults to 42
    :raises ValueError: if `dataset_splits` fractions do not sum up to 1.
    :raises ValueError: if `split_method` is not valid
    """
    
    if cat_unlabeled_tces is None:
        cat_unlabeled_tces = []
        
    # create dataset tables
    if dataset_splits is not None:
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
    if len(shards_tbl) == 0:
        raise ValueError('Dataset table has no examples.')

    # shards_tbl.sort_values(['target_id', 'tce_plnt_num'], ascending=True, inplace=True)
    shards_tbl.reset_index(drop=True, inplace=True)
    
    for req_col in ['label']:
        if req_col not in shards_tbl.columns:
            raise ValueError(f'Column `{req_col}` not found in dataset table.')

    # get TCEs with label and without
    predict_tces = shards_tbl.loc[shards_tbl['label'].isin(cat_unlabeled_tces)].copy()
    labeled_tces = shards_tbl.loc[~shards_tbl['label'].isin(cat_unlabeled_tces)].copy()
    
    if len(labeled_tces) == 0:
        raise ValueError('No examples available for the labeled set after filtering examples for the unlabeled set')

    logger.info(f'TCE disposition count in labeled set:\n {labeled_tces["label"].value_counts()}')

    # labeled_tces.to_csv(dest_tfrec_dir / f'{shards_tbl_fp.stem}_labeled_tces.csv', index=False)
    if len(predict_tces) > 0:
        logger.info(f'Number of TCEs in unlabeled set: {len(predict_tces)}')
        predict_tces.to_csv(dest_tfrec_dir / f'predictset.csv', index=False)

    if split_method == 'by_target_star':
        datasets_tbls = split_tce_table_by_target_stars(labeled_tces, dataset_splits, rng, logger)
    elif split_method in ['mission_phases', 'forward_expanding_rolling_window']:
        datasets_tbls = split_tces_by_target_and_temporal_split(labeled_tces, split_method)
    else:
        raise NotImplementedError(f'Split method {split_method} is not valid.')

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
    if len(predict_tces) > 0:
        dataset_counts.append(predict_tces['label'].value_counts().rename('predict'))
    dataset_counts = pd.concat(dataset_counts, axis=1).fillna(0).astype(int)
    
    # add metadata about dataset split
    dataset_counts.attrs['dataset'] = str(shards_tbl_fp)
    dataset_counts.attrs['split_method'] = split_method
    if dataset_splits is not None:
        dataset_counts.attrs['split_fractions'] =  str(dataset_splits)
    if cat_unlabeled_tces is not None:
        dataset_counts.attrs['unlabeled_categories'] = cat_unlabeled_tces
    dataset_counts.attrs['random_seed'] =  rnd_seed
    dataset_counts.attrs['created'] = str(pd.Timestamp.now().floor('min'))
    with open(dest_tfrec_dir / 'datasets_counts.csv', "w") as f:
        for key, value in dataset_counts.attrs.items():
            f.write(f"# {key}: {value}\n")
        dataset_counts.to_csv(f, index=True)

    logger.info('Finished splitting data in table.')


if __name__ == '__main__':

    # saving directory
    dest_tfrec_dir = Path(f'/nobackupp19/msaragoc/work_dir/Kepler-TESS_exoplanet/data/tfrecords/TESS/tfrecords_tess-spoc-2min_tces_s1-s88_1-19-2026_1459_agg_diffimg_fixedtointps_photdisps_strong-confidence_train-val-test-splits')
    # shards table for your source TFRecord dataset
    shards_tbl_fp = Path('/nobackupp19/msaragoc/work_dir/Kepler-TESS_exoplanet/data/tfrecords/TESS/tfrecords_tess-spoc-2min_tces_s1-s88_1-19-2026_1459_agg_diffimg_fixedtointps_photdisps_strong-confidence/shards_tbl.csv')
    rnd_seed = 24  # random seed
    # split ratio; set to None if using other split methods
    dataset_splits = {
        'train': 0.7,
        'val': 0.2,
        'test': 0.1,
    }
    # TCEs with these labels are put into the predict set
    cat_unlabeled_tces = [
        'UNK',
    ]
    # either 'by_target_star', 'mission_phases', 'forward_expanding_rolling_window'
    split_method = 'by_target_star'

    main(
        dest_tfrec_dir, 
        shards_tbl_fp,
        split_method,
        dataset_splits=dataset_splits,
        cat_unlabeled_tces=cat_unlabeled_tces, 
        rnd_seed=rnd_seed,
        )
