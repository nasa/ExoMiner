""" Compute area under margin (AUM). """

# 3rd party
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from pathlib import Path
import numpy as np
from functools import reduce
import yaml
import multiprocessing

# local
from aum.create_mislabeled_dataset import create_tfrec_dataset_tbl


def compute_margin(logit_true, logit_max_other):
    """ Compute margin for an example.

    :param logit_true: float, logit for true class
    :param logit_max_other: flot, logit for class with maximum logit that is not the true class
    :return:
        float, margin
    """

    return logit_true - logit_max_other


def compute_margin_ex_tbl(example, label_ids):
    """ Compute margin for an example from a table.

    :param example: pandas Series, example
    :param label_ids: list, label ids for the different classes
    :return:
        float, margin for example
    """

    logit_true = example[str(example['label_id'])]
    other_label_ids = [str(label_id) for label_id in label_ids if label_id != example['label_id']]
    if len(other_label_ids) == 0:
        return np.nan

    logit_max_other = max(example[other_label_ids])

    return compute_margin(logit_true, logit_max_other)


def set_dataset(row):
    """ Get dataset for example from shard name.

    :param row: pandas Series, example
    :return:
        str, dataset
    """

    if 'train' in row['shard_name']:
        return 'train'
    elif 'val' in row['shard_name']:
        return 'val'
    elif 'test' in row['shard_name']:
        return 'test'
    elif 'predict' in row['shard_name']:
        return 'predict'


def compute_aum_for_run(experiment_dir, run, features, idx_cols, tce_tbl):
    """ Compute area under margins (AUM) for a given run.

    :param experiment_dir: Path, experiment root directory
    :param run: Path, run directory
    :param features: dict, features to extract from the TFRecord dataset
    :param idx_cols: list, columns to keep in the AUM table from the margin tables
    :param tce_tbl: pandas DataFrame, TCE table
    :return:
    """

    print(f'Computing AUM for run {run}...')

    # create TFRecord dataset table
    tfrec_dir = experiment_dir / 'tfrecords' / run.name
    tfrec_fps = [fp for fp in tfrec_dir.iterdir() if 'shard' in fp.name]

    tfrec_tbl = create_tfrec_dataset_tbl(tfrec_fps, features=features)
    # tfrec_tbl.to_csv(experiment_dir / 'tfrec_tbl.csv', index=False)
    tfrec_tbl = tfrec_tbl.merge(tce_tbl, on=['target_id', 'tce_plnt_num'], how='left', validate='one_to_one')

    # logit directory
    logit_dir = run / 'models' / 'model1'

    train_config_fp = experiment_dir / 'config_train.yaml'
    with(open(train_config_fp, 'r')) as file:  # read default YAML configuration file
        train_config = yaml.safe_load(file)
    train_config['label_map_pred']['UNK'] = 0  # assume UNK examples are non-PC

    # create directory to save computed margins
    margins_dir = logit_dir / 'margins'
    margins_dir.mkdir(exist_ok=True)

    logits_tbls = {fp.stem.split('-')[1]: pd.read_csv(fp) for fp in logit_dir.iterdir() if 'logits' in fp.name}
    # aggregate logits across all epochs
    for logits_tbl_name, logits_tbl in logits_tbls.items():
        logits_tbls[logits_tbl_name] = tfrec_tbl.merge(logits_tbl, how='left', left_index=True, right_index=True,
                                                       validate='one_to_one')

    label_ids = np.unique(list(train_config['label_map'].values()))  # .astype('str')

    # map labels to label ids
    for logits_tbl_name, logits_tbl in logits_tbls.items():
        logits_tbls[logits_tbl_name]['label_id'] = \
            logits_tbl.apply(lambda row: train_config['label_map_pred'][row['label']], axis=1)
        # logits_tbls[logits_tbl_name] = logits_tbls[logits_tbl_name].astype(dtype={'label_id': str})

    # compute margin
    for logits_tbl_name, logits_tbl in logits_tbls.items():
        logits_tbls[logits_tbl_name]['margin'] = logits_tbl.apply(compute_margin_ex_tbl, args=(label_ids,), axis=1)

    # save margin tables
    for logits_tbl_name, logits_tbl in logits_tbls.items():
        logits_tbls[logits_tbl_name].to_csv(margins_dir / f'margins_epoch-{logits_tbl_name}.csv', index=False)

    # compute AUM
    margin_tbls = [pd.read_csv(fp).set_index(idx_cols)['margin'] for fp in margins_dir.iterdir()]
    n_epochs = len(margin_tbls)

    # sum margins across epochs and divide by number of epochs
    aum_tbl = reduce(lambda df1, df2: df1.add(df2), margin_tbls) / n_epochs

    # fill in dataset
    aum_tbl = aum_tbl.to_frame().reset_index()
    aum_tbl['dataset'] = aum_tbl.apply(set_dataset, axis=1)

    aum_tbl.to_csv(logit_dir / 'aum.csv', index=False)

    # aum_tbl.loc[aum_tbl['label'] == noise_label, 'margin'] = np.nan
    #
    # aum_tbl.to_csv(logit_dir / 'aum_nan_mislabeled.csv', index=False)


if __name__ == '__main__':

    features = {
        'target_id': {'dtype': 'int64'},
        'tce_plnt_num': {'dtype': 'int64'},
        'label': {'dtype': 'str'},
        # 'original_label': {'dtype': 'str'},
        # 'label_id': {'dtype': 'int64'}
    }

    # choose columns to keep in the AUM table
    idx_cols = ['target_id', 'tce_plnt_num', 'label', 'label_id', 'original_label', 'shard_name']

    # noise_label = 'MISLABELED'

    experiment_dir = Path(
        '/data5/tess_project/experiments/current_experiments/label_noise_detection_aum/label_noise_detection_aum/run_02-03-2022_1444')

    tce_tbl = pd.read_csv(
        '/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17_DR25/11-17-2021_1243/q1_q17_dr25_tce_2020.09.28_10.36.22_stellar_koi_cfp_norobovetterlabels_renamedcols_nomissingval_symsecphase_cpkoiperiod_rba_cnt0n_valpc_modelchisqr.csv',
        usecols=['target_id', 'tce_plnt_num', 'label'])
    tce_tbl.rename(columns={'label': 'original_label'}, inplace=True)

    # # get original labels from original TFRecord dataset
    # og_tfrec_dir = Path('/data5/tess_project/Data/tfrecords/Kepler/Q1-Q17_DR25/tfrecordskeplerdr25-dv_g301-l31_spline_nongapped_newvalpcs_tessfeaturesadjs_12-1-2021_data/tfrecordskeplerdr25-dv_g301-l31_spline_nongapped_newvalpcs_tessfeaturesadjs_12-1-2021_experiment-normalized')
    # og_tfrec_tbl = pd.read_csv(og_tfrec_dir / 'tfrec_tbl.csv')
    # og_tfrec_tbl.rename(columns={'label': 'original_label'}, inplace=True)

    runs_dir = experiment_dir / 'runs'
    runs = sorted([run for run in runs_dir.iterdir() if run.is_dir()])

    n_processes = min(len(runs), 10)
    pool = multiprocessing.Pool(processes=n_processes)
    jobs = [(experiment_dir, run, features, idx_cols, tce_tbl) for run in runs]
    async_results = [pool.apply_async(compute_aum_for_run, job) for job in jobs]
    pool.close()
    for async_result in async_results:
        async_result.get()

    # for run in runs:
    # compute_aum_for_run(experiment_dir, run, features, idx_cols, tce_tbl)
    #     print(f'Computing AUM for run {run}...')
    #
    #     # create TFRecord dataset table
    #     tfrec_dir = experiment_dir / 'tfrecords' / run.name
    #     tfrec_fps = [fp for fp in tfrec_dir.iterdir() if 'shard' in fp.name]
    #
    #     tfrec_tbl = create_tfrec_dataset_tbl(tfrec_fps, features=features)
    #     # tfrec_tbl.to_csv(experiment_dir / 'tfrec_tbl.csv', index=False)
    #     tfrec_tbl = tfrec_tbl.merge(tce_tbl, on=['target_id', 'tce_plnt_num'], how='left', validate='one_to_one')
    #
    #     # compute margin for each epoch
    #     logit_dir = run / 'models' / 'model1'
    #
    #     train_config_fp = experiment_dir / 'config_train.yaml'
    #     with(open(train_config_fp, 'r')) as file:  # read default YAML configuration file
    #         train_config = yaml.safe_load(file)
    #     train_config['label_map_pred']['UNK'] = 0  # assume UNK examples are non-PC
    #
    #     margins_dir = logit_dir / 'margins'
    #     margins_dir.mkdir(exist_ok=True)
    #
    #     logits_tbls = {fp.stem.split('-')[1]: pd.read_csv(fp) for fp in logit_dir.iterdir() if 'logits' in fp.name}
    #
    #     for logits_tbl_name, logits_tbl in logits_tbls.items():
    #         logits_tbls[logits_tbl_name] = tfrec_tbl.merge(logits_tbl, how='left', left_index=True, right_index=True,
    #                                                        validate='one_to_one')
    #
    #     label_ids = np.unique(list(train_config['label_map'].values()))  # .astype('str')
    #
    #     # map labels to label ids
    #     for logits_tbl_name, logits_tbl in logits_tbls.items():
    #         logits_tbls[logits_tbl_name]['label_id'] = \
    #             logits_tbl.apply(lambda row: train_config['label_map_pred'][row['label']], axis=1)
    #         # logits_tbls[logits_tbl_name] = logits_tbls[logits_tbl_name].astype(dtype={'label_id': str})
    #
    #     # compute margin
    #     for logits_tbl_name, logits_tbl in logits_tbls.items():
    #         logits_tbls[logits_tbl_name]['margin'] = logits_tbl.apply(compute_margin_ex_tbl, args=(label_ids,), axis=1)
    #
    #     # save margin tables
    #     for logits_tbl_name, logits_tbl in logits_tbls.items():
    #         logits_tbls[logits_tbl_name].to_csv(margins_dir / f'margins_epoch-{logits_tbl_name}.csv', index=False)
    #
    #     # compute AUM
    #     margin_tbls = [pd.read_csv(fp).set_index(idx_cols)['margin'] for fp in margins_dir.iterdir()]
    #     n_epochs = len(margin_tbls)
    #
    #     # sum margins across epochs and divide by number of epochs
    #     aum_tbl = reduce(lambda df1, df2: df1.add(df2), margin_tbls) / n_epochs
    #
    #     # fill in dataset
    #     aum_tbl = aum_tbl.to_frame().reset_index()
    #     aum_tbl['dataset'] = aum_tbl.apply(set_dataset, axis=1)
    #
    #     aum_tbl.to_csv(logit_dir / 'aum.csv', index=False)
    #
    #     # aum_tbl.loc[aum_tbl['label'] == noise_label, 'margin'] = np.nan
    #     #
    #     # aum_tbl.to_csv(logit_dir / 'aum_nan_mislabeled.csv', index=False)
