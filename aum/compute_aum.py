""" Compute margins and area under margin (AUM) using the logits output by the model for each epoch. """

# 3rd party
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from pathlib import Path
import numpy as np
# from functools import reduce
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
    :param idx_cols: list, columns to keep in the AUM and margin tables from the margin tables
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
    tfrec_tbl['dataset'] = tfrec_tbl.apply(set_dataset, axis=1)

    # model directory
    model_dir = run / 'models' / 'model1'

    train_config_fp = experiment_dir / 'config_train.yaml'
    with(open(train_config_fp, 'r')) as file:  # read default YAML configuration file
        train_config = yaml.safe_load(file)
    train_config['label_map_pred']['UNK'] = 0  # assume UNK examples are non-PC

    # create directory to save computed margins
    margins_dir = model_dir / 'margins'
    margins_dir.mkdir(exist_ok=True)

    logits_tbls = {fp.stem.split('-')[1]: pd.read_csv(fp) for fp in model_dir.iterdir() if 'logits' in fp.name}
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

    n_epochs = len(logits_tbls)

    # compute margin
    for logits_tbl_name, logits_tbl in logits_tbls.items():
        logits_tbls[logits_tbl_name]['margin'] = logits_tbl.apply(compute_margin_ex_tbl, args=(label_ids,), axis=1)

    # save margin tables
    for logits_tbl_name, logits_tbl in logits_tbls.items():
        logits_tbls[logits_tbl_name].to_csv(margins_dir / f'margins_epoch-{logits_tbl_name}.csv', index=False)
    margins_tbl = []
    for epoch_i in range(len(logits_tbls)):
        margins_tbl.append(logits_tbls[str(epoch_i)][idx_cols + ['margin']] if epoch_i == 0
                           else logits_tbls[str(epoch_i)][['margin']])
        margins_tbl[-1] = margins_tbl[-1].rename(columns={'margin': f'epoch_{epoch_i}'})
    margins_tbl = pd.concat(margins_tbl, axis=1)
    margins_tbl.to_csv(margins_dir / 'margins_allepochs.csv', index=False)

    # compute AUM;     # sum margins across epochs and divide by number of epochs
    aum_tbl = margins_tbl.copy(deep=True)
    aum_tbl[[f'epoch_{epoch_i}' for epoch_i in range(n_epochs)]] = \
        aum_tbl[[f'epoch_{epoch_i}' for epoch_i in range(n_epochs)]].cumsum(axis=1) / np.arange(1, n_epochs + 1)
    aum_tbl.to_csv(model_dir / 'aum.csv', index=False)


if __name__ == '__main__':

    features = {
        'target_id': {'dtype': 'int64'},
        'tce_plnt_num': {'dtype': 'int64'},
        'label': {'dtype': 'str'},
        # 'original_label': {'dtype': 'str'},
        # 'label_id': {'dtype': 'int64'}
    }

    # choose columns to keep in the AUM table
    idx_cols = ['target_id', 'tce_plnt_num', 'label', 'label_id', 'original_label', 'shard_name', 'dataset']

    # noise_label = 'MISLABELED'

    experiment_dir = Path(
        '/home/msaragoc/Projects/Kepler-TESS_exoplanet/experiments/label_noise_detection_aum/run_03-24-2022_1044')

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

    print('Finished computing AUM.')
