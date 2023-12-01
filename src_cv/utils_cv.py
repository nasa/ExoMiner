"""
Utility functions for CV.
"""

# 3rd party
import multiprocessing
import shutil
import numpy as np
import pandas as pd
import tensorflow as tf
import logging
from pathlib import Path

# local
from src_preprocessing.compute_normalization_stats_tfrecords import compute_normalization_stats
from src_preprocessing.normalize_data_tfrecords import normalize_examples
from src_preprocessing.tf_util import example_util


def create_shard_fold(shard_tbl_fp, dest_tfrec_dir, fold_i, src_tfrec_dir, src_tfrec_tbl, log=False):
    """ Create a TFRecord fold for cross-validation based on source TFRecords and fold TCE table.

    :param shard_tbl_fp: Path, shard TCE table file path
    :param dest_tfrec_dir: Path, destination TFRecord directory
    :param fold_i: int, fold id
    :param src_tfrec_dir: Path, source TFRecord directory
    :param src_tfrec_tbl: pandas Dataframe, maps example to a given TFRecord shard
    :param log: bool, if True creates a logger
    :return:
        tces_not_found: list, each sublist contains the target id and tce planet number for TCEs not found in the source
        TFRecords
    """

    if log:
        # set up logger
        logger = logging.getLogger(name=f'log_fold_{fold_i}')
        logger_handler = logging.FileHandler(filename=dest_tfrec_dir.parent / f'create_tfrecord_fold_{fold_i}.log',
                                             mode='w')
        logger_formatter = logging.Formatter('%(asctime)s - %(message)s')
        logger.setLevel(logging.INFO)
        logger_handler.setFormatter(logger_formatter)
        logger.addHandler(logger_handler)
        logger.info(f'Starting run...')

    tces_not_found = []

    if log:
        logger.info(f'Reading fold TCE table: {shard_tbl_fp}')
    fold_tce_tbl = pd.read_csv(shard_tbl_fp)

    tfrec_new_fp = dest_tfrec_dir / f'shard-{f"{fold_i}".zfill(4)}'
    if log:
        logger.info(f'Destination TFRecord file: {tfrec_new_fp}')

    if log:
        logger.info('Start iterating over the fold TCE table...')
    n_tces_in_shard = 0
    # write examples in a new TFRecord shard
    with tf.io.TFRecordWriter(str(tfrec_new_fp)) as writer:

        for tce_i, tce in fold_tce_tbl.iterrows():

            if tce_i + 1 % 50 == 0 and log:
                logger.info(f'Iterating over fold table {fold_i} {shard_tbl_fp.name} '
                            f'({tce_i + 1} out of {len(fold_tce_tbl)})\nNumber of TCEs in the shard: {n_tces_in_shard}...')

            # look for TCE in the source TFRecords table
            tce_found = src_tfrec_tbl.loc[src_tfrec_tbl['uid'] == tce['uid'],  ['shard', 'example_i_tfrec']]

            if len(tce_found) == 0:
                tces_not_found.append([tce['uid']])
                continue

            src_tfrec_name, example_i = tce_found.values[0]

            tfrecord_dataset = tf.data.TFRecordDataset(str(src_tfrec_dir / src_tfrec_name))
            for string_i, string_record in enumerate(tfrecord_dataset.as_numpy_iterator()):
                if string_i == example_i:
                    example = tf.train.Example()
                    example.ParseFromString(string_record)

                    example_uid = example.features.feature['uid'].bytes_list.value[0].decode("utf-8")

                    assert f'{example_uid}' == f'{tce["uid"]}'

                    writer.write(example.SerializeToString())
                    n_tces_in_shard += 1

        writer.close()

    if log:
        logger.info(f'Finished iterating over fold table {fold_i} {shard_tbl_fp.name}.')

    return tces_not_found


def check_normalized_features(norm_features, norm_data_dir, tceid):
    """ Check if normalized features have any non-finite values.

    :param norm_features: dict, normalized features
    :param norm_data_dir: Path, normalized TFRecord directory
    :param tceid: str, TCE ID
    :return:
    """

    feature_str_arr = []
    for feature_name, feature_val in norm_features.items():
        if np.any(~np.isfinite(feature_val)):
            feature_str_arr.append(f'Feature {feature_name} has non-finite values.')

    if len(feature_str_arr) > 0:
        with open(norm_data_dir / f'check_normalized_features_{tceid}.txt', 'w') as file:
            for feature_str in feature_str_arr:
                file.write(feature_str)


def add_sample_weights_to_tfrecords(data_shards_fps, weights_cats, temp_data_dir):
    """ Add sample weights to examples in the training set TFRecords.

    Args:
        data_shards_fps: list, TFRecords folds used as training set for this CV iteration
        weights_cats: dict, categories weights
        temp_data_dir: Path, temporary data directory in which the new data is stored

    Returns:

    """

    for data_shards_fp in data_shards_fps:
        with tf.io.TFRecordWriter(str(temp_data_dir / data_shards_fp.name)) as writer:
            tfrecord_dataset = tf.data.TFRecordDataset(data_shards_fp)
            for string_i, string_record in enumerate(tfrecord_dataset.as_numpy_iterator()):
                example = tf.train.Example()
                example.ParseFromString(string_record)
                example_label = example.features.feature['label'].bytes_list.value[0].decode("utf-8")
                example_util.set_float_feature(example, 'sample_weight', [weights_cats[example_label]])
                writer.write(example.SerializeToString())
            writer.close()


def compute_sample_weights(data_shards_fps, run_params):
    """ Compute sample weights and add them to the examples in the TFRecords for the training set.

    Args:
        data_shards_fps: dict, 'train', 'val', and 'test' keys with TFRecords folds used as training and test sets,
    respectively, for this CV iteration
        param run_params: dict, configuration parameters for the CV run

    Returns:

    """

    curr_data_dir = data_shards_fps['train'][0].parent.parent
    temp_data_dir = curr_data_dir / 'temp_data'
    temp_data_dir.mkdir(exist_ok=True)

    sample_weights_dict = {'uid': [], 'label': []}
    for train_data_shards_fp in data_shards_fps['train']:
        tfrecord_dataset = tf.data.TFRecordDataset(str(train_data_shards_fp))
        for string_i, string_record in enumerate(tfrecord_dataset.as_numpy_iterator()):
            example = tf.train.Example()
            example.ParseFromString(string_record)
            sample_weights_dict['uid'].append(example.features.feature['uid'].bytes_list.value[0].decode("utf-8"))
            sample_weights_dict['label'].append(example.features.feature['label'].bytes_list.value[0].decode("utf-8"))

    sample_weights_df = pd.DataFrame(sample_weights_dict)
    CATS = [['PC', 'T-KP', 'T-CP'], ['AFP', 'T-EB', 'T-FP'], ['NTP', 'T-NTP', 'T-FA']]
    n_cats = len(CATS)  # len(sample_weights_df['label'].unique())
    n_total_samples = len(sample_weights_df)
    # n_samples_per_cat = sample_weights_df['label'].value_counts()
    weights_cats = {cat: np.nan for cat in sample_weights_df['label'].unique()}
    for cat_grp in CATS:
        n_samples_per_cat_grp = (sample_weights_df['label'].isin(cat_grp)).sum()
        for cat in cat_grp:
            weights_cats[cat] = n_total_samples / (n_samples_per_cat_grp * n_cats)

    weights_cats_df = pd.DataFrame({key: [val]
                                    for key, val in weights_cats.items()}).T.rename(columns={0: 'category_weight'})
    weights_cats_df.to_csv(curr_data_dir / 'samples_weights.csv')

    pool = multiprocessing.Pool(processes=run_params['norm_examples_params']['n_processes_norm_data'])
    data_shards_fps_jobs = np.array_split(data_shards_fps['train'],
                                          run_params['norm_examples_params']['n_processes_norm_data'])
    jobs = [(fps, weights_cats, temp_data_dir) for fps in data_shards_fps_jobs]
    _ = [pool.apply_async(add_sample_weights_to_tfrecords, job) for job in jobs]
    pool.close()
    for async_result in _:
        async_result.get()

    for dataset in data_shards_fps:
        if dataset != 'train':
            for fp in data_shards_fps[dataset]:
                shutil.copy(fp, temp_data_dir / fp.name)

    shutil.rmtree(curr_data_dir / 'norm_data')
    shutil.move(temp_data_dir, curr_data_dir / 'norm_data')


def processing_data_run(data_shards_fps, run_params, cv_run_dir):
    """ Processing data for a given CV run. This step involves computing normalization statistics from the training set,
    and then normalizing and clipping the data using those statistics.

    :param data_shards_fps: dict, 'train', 'val', and 'test' keys with TFRecords folds used as training and test sets,
    respectively, for this CV iteration
    :param run_params: dict, configuration parameters for the CV run
    :param cv_run_dir: Path, CV run directory file path
    :return:
        data_shards_fps_norm: dict, 'train', 'val', and 'test' keys with TFRecords folds used as training and test sets,
    respectively, for this CV iteration. Examples are normalized.
    """

    run_params['compute_norm_stats_params']['norm_dir'] = cv_run_dir / 'norm_stats'
    run_params['compute_norm_stats_params']['norm_dir'].mkdir(exist_ok=True)

    if run_params['logger'] is not None:
        run_params['logger'].info(f'[cv_iter_{run_params["cv_id"]}] Computing normalization statistics')

    p = multiprocessing.Process(target=compute_normalization_stats,
                                args=(
                                      data_shards_fps['train'],
                                      run_params['compute_norm_stats_params'],
                                      ))
    p.start()
    p.join()
    p.terminate()

    if run_params['logger'] is not None:
        run_params['logger'].info(f'[cv_iter_{run_params["cv_id"]}] Normalizing the data...')
    norm_data_dir = cv_run_dir / 'norm_data'  # create folder for normalized data set
    norm_data_dir.mkdir(exist_ok=True)

    # load normalization statistics
    norm_stats = {}
    if run_params['compute_norm_stats_params']['timeSeriesFDLList'] is not None:
        norm_stats.update({'fdl_centroid': np.load(run_params['compute_norm_stats_params']['norm_dir'] /
                                'train_fdlcentroid_norm_stats.npy', allow_pickle=True).item()})
    if run_params['compute_norm_stats_params']['centroidList'] is not None:
        norm_stats.update({'centroid': np.load(run_params['compute_norm_stats_params']['norm_dir'] /
                            'train_centroid_norm_stats.npy', allow_pickle=True).item()})
    if run_params['compute_norm_stats_params']['scalarParams'] is not None:
        scalar_params_norm_info = np.load(run_params['compute_norm_stats_params']['norm_dir'] /
                                 'train_scalarparam_norm_stats.npy', allow_pickle=True).item()
        scalar_params_norm_info = {k: v for k, v in scalar_params_norm_info.items()
                                   if k in run_params['compute_norm_stats_params']['scalarParams']}
        norm_stats.update({'scalar_params': scalar_params_norm_info})
    if run_params['compute_norm_stats_params']['diff_imgList'] is not None:
        norm_stats.update({'diff_img': np.load(run_params['compute_norm_stats_params']['norm_dir'] /
                                               'train_diffimg_norm_stats.npy', allow_pickle=True).item()})

    # normalize data using the normalization statistics
    if len(norm_stats) == 0:
        run_params['logger'].info(f'[cv_iter_{run_params["cv_id"]}] Data cannot be normalized since no normalization '
                                  f'statistics were loaded.')
        raise ValueError(f'[cv_iter_{run_params["cv_id"]}] Data cannot be normalized since no normalization '
                         f'statistics were loaded.')

    pool = multiprocessing.Pool(processes=run_params['norm_examples_params']['n_processes_norm_data'])
    jobs = [(norm_data_dir, file, norm_stats, run_params['norm_examples_params']['aux_params'])
            for file in np.concatenate(list(data_shards_fps.values()))]
    async_results = [pool.apply_async(normalize_examples, job) for job in jobs]
    pool.close()
    pool.join()
    for async_result in async_results:
        async_result.get()

    data_shards_fps_norm = {dataset: [norm_data_dir / data_fp.name for data_fp in data_fps]
                            for dataset, data_fps in data_shards_fps.items()}

    # compute sample weights
    if run_params['training']['sample_weights']:
        compute_sample_weights(data_shards_fps_norm, run_params)

    return data_shards_fps_norm


def create_table_shard_example_location(tfrec_dir):
    """ Create shard TCE table that makes tracking of location of TCEs in the shards easier. The table consists  of 4
    columns: target_id, tce_plnt_num, shard filename that contains the example, sequential id for example in the shard
    starting at 0.

    :param tfrec_dir: Path, TFRecord directory
    :return:
    """

    tfrec_fps = [file for file in tfrec_dir.iterdir() if 'shard' in file.stem and not file.suffix == '.csv']

    data_to_df = []
    for tfrec_fp in tfrec_fps:

        tfrecord_dataset = tf.data.TFRecordDataset(str(tfrec_fp))
        for string_i, string_record in enumerate(tfrecord_dataset.as_numpy_iterator()):
            example = tf.train.Example()
            example.ParseFromString(string_record)

            targetIdTfrec = example.features.feature['target_id'].int64_list.value[0]
            tceIdentifierTfrec = example.features.feature['tce_plnt_num'].int64_list.value[0]

            data_to_df.append([targetIdTfrec, tceIdentifierTfrec, tfrec_fp.name, string_i])

    shards_tce_tbl = pd.DataFrame(data_to_df, columns=['target_id', 'tce_plnt_num', 'shard', 'example_i'])
    shards_tce_tbl.to_csv(tfrec_dir / 'shards_tce_tbl.csv', index=False)


if __name__ == '__main__':

    tfrec_dir = Path(
        '/data5/tess_project/Data/tfrecords/Kepler/Q1-Q17_DR25/tfrecordskeplerdr25-dv_g301-l31_spline_nongapped_newvalpcs_tessfeaturesadjs_12-1-2021_data/tfrecordskeplerdr25-dv_g301-l31_spline_nongapped_newvalpcs_tessfeaturesadjs_12-1-2021_koi_fpflagec')

    create_table_shard_example_location(tfrec_dir)
