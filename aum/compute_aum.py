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
# from aum.create_mislabeled_dataset import create_tfrec_dataset_tbl
from utils.utils_dataio import is_yamlble


def compute_margin_np_arr(logits_arr, label_id_arr):
    """ Compute margins for all examples across epochs.

    :param logits_arr: NumPy array, logits for each example over epochs [logits_ids, examples, epochs]
    :param label_id_arr: NumpY array, label ids for the examples [examples]
    :return:
        margins_arr, Numpy array that contains margins for all examples over epochs
    """

    # mask = np.zeros(logits_arr.shape, dtype='bool')
    # mask[label_id_arr, :, :] = True  # mask logits for labeled class
    # logits_true = logits_arr[:, ]
    margins_arr = np.nan * np.ones((logits_arr.shape[1], logits_arr.shape[2]), dtype='float')
    for example_i in range(logits_arr.shape[1]):
        logits_example = logits_arr[:, example_i, :]
        logits_label = logits_example[label_id_arr[example_i], :]
        mask = np.ones(logits_example.shape, dtype='bool')
        mask[label_id_arr[example_i], :] = False  # mask logits for labeled class
        # get logits for class with max logit that is not the labeled class
        logits_max_not_label = \
            logits_example[mask].reshape([logits_example.shape[0] - 1, logits_example.shape[1]]).max(axis=0)

        margins_arr[example_i, :] = logits_label - logits_max_not_label

    return margins_arr


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


def compute_aum_for_run(experiment_dir, run, features, idx_cols, tce_tbl, n_epochs):
    """ Compute area under margins (AUM) for a given run.

    :param experiment_dir: Path, experiment root directory
    :param run: Path, run directory
    :param features: dict, features to extract from the TFRecord dataset
    :param idx_cols: list, columns to keep in the AUM and margin tables from the margin tables
    :param tce_tbl: pandas DataFrame, TCE table
    :param n_epochs: int, number of epochs the model was trained for
    :return:
    """

    print(f'Computing AUM for run {run}...')

    train_config_fp = experiment_dir / 'config_train.yaml'
    with(open(train_config_fp, 'r')) as file:  # read default YAML configuration file
        train_config = yaml.safe_load(file)
    train_config['label_map_pred']['UNK'] = 0  # assume UNK examples are non-PC

    # train_config_fp2 = run / 'train_params.yaml'
    # with(open(train_config_fp2, 'r')) as file:  # read default YAML configuration file
    #     # train_config2 = yaml.safe_load(file)
    #     train_config2 = yaml.load(file, Loader=yaml.Loader)
    #     # train_config = yaml.safe_load(file)
    #     # train_config = yaml.danger_load(file)
    #
    # # create TFRecord dataset table
    # tfrec_dir = experiment_dir / 'tfrecords' / run.name
    # tfrec_fps = [fp for fp in tfrec_dir.iterdir() if 'shard' in fp.name]

    # get table of examples in the TFRecord dataset used to get the logits (order is preserved)
    tfrec_tbl_fp = Path(run / 'tfrec_tbl.csv')
    # if tfrec_tbl_fp.exists():
    tfrec_tbl = pd.read_csv(tfrec_tbl_fp)
    # else:
    #     tfrec_tbl = create_tfrec_dataset_tbl(tfrec_fps, features=features)
    tfrec_tbl = tfrec_tbl.merge(tce_tbl, on=['target_id', 'tce_plnt_num'], how='left', validate='one_to_one')
    tfrec_tbl['dataset'] = tfrec_tbl.apply(set_dataset, axis=1)  # add dataset column
    #     tfrec_tbl.to_csv(run / 'tfrec_tbl.csv', index=False)

    # model directory
    model_dir = run / 'models' / 'model1'

    # map labels to label ids
    label_ids = np.unique(list(train_config['label_map'].values()))  # .astype('str')
    tfrec_tbl['label_id'] = \
        tfrec_tbl.apply(lambda row: train_config['label_map_pred'][row['label']], axis=1)

    # # create directory to save computed margins
    # margins_dir = model_dir / 'margins'
    # margins_dir.mkdir(exist_ok=True)

    # aggregate logits
    # logits_tbls = {fp.stem.split('-')[1]: pd.read_csv(fp) for fp in model_dir.iterdir() if 'logits' in fp.name}
    # for logits_tbl_name, logits_tbl in logits_tbls.items():  # add extra information on examples such as their id
    #     logits_tbls[logits_tbl_name] = tfrec_tbl.merge(logits_tbl, how='left', left_index=True, right_index=True,
    #                                                    validate='one_to_one')
    logits_tbls = {logit_i: pd.concat([tfrec_tbl[idx_cols]] + [pd.read_csv(model_dir / f'logits_epoch-{epoch_i}.csv',
                                                   usecols=[logit_i],
                                                   names=[f'epoch_{epoch_i}'],
                                                   header=0)
                                       for epoch_i in range(n_epochs)], axis=1) for logit_i in label_ids
                   }
    # # map labels to label ids
    # label_ids = np.unique(list(train_config['label_map'].values()))  # .astype('str')
    # for logits_tbl_name, logits_tbl in logits_tbls.items():
    #     logits_tbls[logits_tbl_name]['label_id'] = \
    #         logits_tbl.apply(lambda row: train_config['label_map_pred'][row['label']], axis=1)
    #     # logits_tbls[logits_tbl_name] = logits_tbls[logits_tbl_name].astype(dtype={'label_id': str})

    # n_epochs = len(logits_tbls)  # number of epochs for which the logits were logged

    # save logits tables; each logit has a csv for values over epochs
    # logits_tbls = \
    #     {logit_i: pd.concat([logits_tbls[f'{logit_i}'][idx_cols]] +
    #                         [logits_tbls[f'{epoch_i}'][[f'{logit_i}']].rename(columns={f'{logit_i}': f'epoch_{epoch_i}'})
    #                          for epoch_i in range(n_epochs)], axis=1) for logit_i in label_ids}
    for logit_i, logits_tbl in logits_tbls.items():
        logits_tbl.to_csv(model_dir / f'logit{logit_i}_allepochs.csv', index=False)

    # compute margin
    logits_arr = np.array([logit_tbl[[f'epoch_{epoch_i}' for epoch_i in range(n_epochs)]]
                           for logit_tbl in logits_tbls.values()])
    margins_arr = compute_margin_np_arr(logits_arr, tfrec_tbl['label_id'].to_numpy())
    # for logits_tbl_name, logits_tbl in logits_tbls.items():
    #     logits_tbls[logits_tbl_name]['margin'] = logits_tbl.apply(compute_margin_ex_tbl, args=(label_ids,), axis=1)

    # save margin tables
    # for logits_tbl_name, logits_tbl in logits_tbls.items():
    #     logits_tbls[logits_tbl_name].to_csv(margins_dir / f'margins_epoch-{logits_tbl_name}.csv', index=False)
    # margins_tbl = []
    # for epoch_i in range(len(logits_tbls)):
    #     margins_tbl.append(logits_tbls[str(epoch_i)][idx_cols + ['margin']] if epoch_i == 0
    #                        else logits_tbls[str(epoch_i)][['margin']])
    #     margins_tbl[-1] = margins_tbl[-1].rename(columns={'margin': f'epoch_{epoch_i}'})
    # margins_tbl = pd.concat(margins_tbl, axis=1)
    # margins_tbl.to_csv(margins_dir / 'margins_allepochs.csv', index=False)
    margins_tbl = pd.DataFrame(margins_arr, columns=[f'epoch_{epoch_i}' for epoch_i in range(n_epochs)])
    margins_tbl = pd.concat([tfrec_tbl[idx_cols], margins_tbl], axis=1)
    margins_tbl.to_csv(model_dir / 'margins_allepochs.csv', index=False)

    # compute AUM; sum margins across epochs and divide by number of epochs
    aum_tbl = margins_tbl.copy(deep=True)
    aum_tbl[[f'epoch_{epoch_i}' for epoch_i in range(n_epochs)]] = \
        aum_tbl[[f'epoch_{epoch_i}' for epoch_i in range(n_epochs)]].cumsum(axis=1) / np.arange(1, n_epochs + 1)
    aum_tbl.to_csv(model_dir / 'aum.csv', index=False)


if __name__ == '__main__':

    path_to_yaml = Path('/Users/msaragoc/OneDrive - NASA/Projects/exoplanet_transit_classification/codebase/aum/config_compute_aum.yaml')
    with(open(path_to_yaml, 'r')) as file:
        config = yaml.safe_load(file)

    experiment_dir = Path(config['experiment_dir_fp'])

    # load TCE table to get additional information for the examples
    tce_tbl = pd.read_csv(config['tce_tbl_fp'], usecols=['target_id', 'tce_plnt_num', 'label'])
    tce_tbl.rename(columns={'label': 'original_label'}, inplace=True)

    # # get original labels from original TFRecord dataset
    # og_tfrec_dir = Path('/data5/tess_project/Data/tfrecords/Kepler/Q1-Q17_DR25/tfrecordskeplerdr25-dv_g301-l31_spline_nongapped_newvalpcs_tessfeaturesadjs_12-1-2021_data/tfrecordskeplerdr25-dv_g301-l31_spline_nongapped_newvalpcs_tessfeaturesadjs_12-1-2021_experiment-normalized')
    # og_tfrec_tbl = pd.read_csv(og_tfrec_dir / 'tfrec_tbl.csv')
    # og_tfrec_tbl.rename(columns={'label': 'original_label'}, inplace=True)

    runs_dir = experiment_dir / 'runs'
    runs = sorted([run for run in runs_dir.iterdir() if run.is_dir()])

    n_processes = min(len(runs), config['min_num_processes'])  # number of processes used to parallelize the computation of AUM
    pool = multiprocessing.Pool(processes=n_processes)
    jobs = [(experiment_dir, run, config['features'], config['idx_cols'], tce_tbl, config['n_epochs']) for run in runs]
    async_results = [pool.apply_async(compute_aum_for_run, job) for job in jobs]
    pool.close()
    for async_result in async_results:
        async_result.get()

    # save the YAML file with the configuration parameters that are YAML serializable
    json_dict = {key: val for key, val in config.items() if is_yamlble(val)}
    with open(experiment_dir / 'dataset_params.yaml', 'w') as yml_file:
        yaml.dump(json_dict, yml_file)

    print('Finished computing AUM.')
