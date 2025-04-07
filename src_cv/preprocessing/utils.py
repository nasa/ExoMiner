"""
Utility functions for CV  dataset preprocessing.
"""

# 3rd party
import pandas as pd
import tensorflow as tf
import logging
from pathlib import Path


def create_shard_fold(shard_tbl_fp, dest_tfrec_dir, fold_i, src_tfrec_dir, src_tfrec_tbl, log_fp=None):
    """ Create a TFRecord fold for cross-validation based on source TFRecords and fold TCE table.

    :param shard_tbl_fp: Path, shard TCE table filepath
    :param dest_tfrec_dir: Path, destination TFRecord directory
    :param fold_i: int, fold id
    :param src_tfrec_dir: Path, source TFRecord directory
    :param src_tfrec_tbl: pandas Dataframe, maps example to a given TFRecord shard
    :param log_fp: Path, log file path; if None, does not create a logger
    :return:
        tces_not_found: list, each sublist contains the target id and tce planet number for TCEs not found in the source
        TFRecords
    """

    if log_fp:
        # set up logger
        logger = logging.getLogger(name=f'log_fold_{fold_i}')
        logger_handler = logging.FileHandler(filename=log_fp,  # dest_tfrec_dir.parent / f'create_tfrecord_fold_{fold_i}.log',
                                             mode='w')
        logger_formatter = logging.Formatter('%(asctime)s - %(message)s')
        logger.setLevel(logging.INFO)
        logger_handler.setFormatter(logger_formatter)
        logger.addHandler(logger_handler)
        logger.info(f'Starting run...')

    tces_not_found = []

    if log_fp:
        logger.info(f'Reading fold TCE table: {shard_tbl_fp}')
    fold_tce_tbl = pd.read_csv(shard_tbl_fp)

    tfrec_new_fp = dest_tfrec_dir / f'shard-{f"{fold_i}".zfill(4)}'
    if log_fp:
        logger.info(f'Destination TFRecord file: {tfrec_new_fp}')

    if log_fp:
        logger.info('Iterating over the fold TCE table...')
    n_tces_in_shard = 0
    # write examples in a new TFRecord shard
    with tf.io.TFRecordWriter(str(tfrec_new_fp)) as writer:

        for tce_i, tce in fold_tce_tbl.iterrows():

            if (tce_i + 1) % 50 == 0 and log_fp:
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

    if log_fp:
        logger.info(f'Finished iterating over fold table {fold_i} {shard_tbl_fp.name}.')

    return tces_not_found


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

    tfrec_dir = Path('/data5/tess_project/Data/tfrecords/Kepler/Q1-Q17_DR25/tfrecordskeplerdr25-dv_g301-l31_spline_nongapped_newvalpcs_tessfeaturesadjs_12-1-2021_data/tfrecordskeplerdr25-dv_g301-l31_spline_nongapped_newvalpcs_tessfeaturesadjs_12-1-2021_koi_fpflagec')
    create_table_shard_example_location(tfrec_dir)