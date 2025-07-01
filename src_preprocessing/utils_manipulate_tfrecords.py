""" Utility functions used to manipulate TFRecords. """

# 3rd party
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import shutil


def create_shard(shardFilename, shardTbl, srcTbl, srcTfrecDir, destTfrecDir, omitMissing=True, verbose=False):
    """ Create a TFRecord file (shard) based on a set of existing TFRecord files.

    :param shardFilename: str, shard filename
    :param shardTbl: pandas DataFrame, shard TCE table
    :param srcTbl: pandas DataFrame, source TCE table
    :param srcTfrecDir: str, filepath to directory with the source TFRecords
    :param destTfrecDir: str, filepath to directory in which to save the new TFRecords
    :param omitMissing: bool, omit missing TCEs in teh source TCE table
    :return:
    """

    with tf.io.TFRecordWriter(str(destTfrecDir / shardFilename)) as writer:

        # iterate through TCEs in the shard TCE table
        for tce_i, tce in shardTbl.iterrows():

            # check if TCE is in the source TFRecords TCE table
            foundTce = srcTbl.loc[(srcTbl['uid'] == tce['uid'])]['shard']

            if len(foundTce) > 0:

                tceIdx = foundTce.index[0]
                tceFoundInTfrecordFlag = False

                tfrecord_dataset = tf.data.TFRecordDataset(str(srcTfrecDir / foundTce.values[0]))

                for string_i, string_record in enumerate(tfrecord_dataset.as_numpy_iterator()):

                    # index in the source TFRecords TCE table follow the writing order in the source TFRecords, so it
                    # can be used to access a specific TCE
                    if string_i == tceIdx:
                        example = tf.train.Example()
                        example.ParseFromString(string_record)

                        example_uid = example.features.feature['uid'].bytes_list.value[0].decode("utf-8")

                        if example_uid != tce['uid']:
                            raise ValueError(f'Example {tce["uid"]} not found at respective index {tceIdx} in source '
                                             f'shard {foundTce.values[0]} (destination shard: {shardFilename}). '
                                             f'Example found instead: {example_uid}')

                        tceFoundInTfrecordFlag = True
                        break

                if not tceFoundInTfrecordFlag:
                    raise ValueError(f'Example {tce["uid"]} for shard {shardFilename} not found in source shard '
                                     f'{foundTce.values[0]} (destination shard: {shardFilename}).')

                writer.write(example.SerializeToString())

            else:
                if omitMissing:  # omit missing TCEs in the source TCE table
                    if verbose:
                        print(f'Example {tce["uid"]} for shard {shardFilename} not found in the TFRecords merged '
                              f'table.')
                    continue

                raise ValueError(f'Example {tce["uid"]} for shard {shardFilename} not found in the TFRecords merged '
                                 f'table.')


def plot_features_example(viewsDict, scalarParamsStr, tceid, labelTfrec, plotDir, scheme, basename='', display=False):
    """ Plot example (TCE/OI) stored into a shard (TFRecord).

    :param viewsDict: dict, time series views to be plotted
    :param scalarParamsStr: str, string with scalar parameters to be displayed as title
    :param tceid: TCE/OI ID
    :param labelTfrec: str, disposition/label
    :param plotDir: Path, plot directory
    :param scheme: tuple/list, plots configuration
    :param basename: str, additional name for the figure
    :param display: bool, if True displays figure using Matplotlib
    :return:
    """

    f, ax = plt.subplots(scheme[0], scheme[1], figsize=(22, 12))
    k = 0
    views_list = list(viewsDict.keys())
    for i in range(scheme[0]):
        for j in range(scheme[1]):
            if k < len(views_list):
                ax[i, j].plot(viewsDict[views_list[k]])
                ax[i, j].scatter(np.arange(len(viewsDict[views_list[k]])), viewsDict[views_list[k]], s=5, c='k',
                                 alpha=0.3)
                ax[i, j].set_title(views_list[k], pad=20)
            if i == scheme[0] - 1:
                ax[i, j].set_xlabel('Bin number')
            if j == 0:
                ax[i, j].set_ylabel('Amplitude')
            k += 1

    f.suptitle(f'{tceid} {labelTfrec}\n{scalarParamsStr}')
    plt.subplots_adjust(top=0.795, bottom=0.075, left=0.045, right=0.98, hspace=0.435, wspace=0.315)
    plt.savefig(plotDir / f'{tceid}_{labelTfrec}_{basename}.png')
    # f.tight_layout(rect=[0, 0.03, 1, 0.95])
    # f.tight_layout()
    if not display:
        plt.close()


def create_shards_table(srcTfrecDir):

    """ Create table that merges tables for each TFRecord file. Keeps track of examples stored in each file.

    :param srcTfrecDir: str, source TFRecord directory filepath
    :return:
        bool, True if the table was created successfully
    """

    srcTfrecDir = Path(srcTfrecDir)

    # get csv files for each TFRecord file
    srcTfrecTblsFps = sorted([file for file in srcTfrecDir.iterdir() if file.suffix == '.csv' and
                              file.stem.startswith('shard')])

    if len(srcTfrecTblsFps) == 0:  # no shard csv files found in the directory
        return False

    # concatenate TFRecord tables
    srcTfrecTblMerge = pd.concat([pd.read_csv(srcTfrecTblFp) for srcTfrecTblFp in srcTfrecTblsFps])

    srcTfrecTblMerge.to_csv(srcTfrecDir / 'merged_shards.csv', index=False)

    return True


def create_table_with_tfrecord_examples(tfrec_fp, data_fields=None):
    """ Create table with examples from the TFRecord file with scalar features/attributes defined in `data_fields`.

    Args:
        tfrec_fp: Path, TFRecord file path
        data_fields: dict, 'data_field_name': 'data_type'

    Returns:
        data_tbl, pandas DataFrame
    """

    # iterate through the source shard
    tfrecord_dataset = tf.data.TFRecordDataset(str(tfrec_fp))

    data_to_tbl = {'example_i_tfrec': [], 'shard': []}
    if data_fields:
        data_to_tbl.update({field: [] for field in data_fields})
    else:
        data_fields = {}

    for string_i, string_record in enumerate(tfrecord_dataset.as_numpy_iterator()):

        example = tf.train.Example()
        example.ParseFromString(string_record)

        data_to_tbl['example_i_tfrec'].append(string_i)
        data_to_tbl['shard'].append(tfrec_fp.name)

        for data_field, data_type in data_fields.items():
            if data_type == 'int':
                example_feature = example.features.feature[data_field].int64_list.value[0]
            elif data_type == 'str':
                example_feature = example.features.feature[data_field].bytes_list.value[0].decode("utf-8")
            elif data_type == 'float':
                example_feature = example.features.feature[data_field].float_list.value[0]
            else:
                raise ValueError(f'Data type not expected: {data_type}')

            data_to_tbl[data_field].append(example_feature)

    data_tbl = pd.DataFrame(data_to_tbl)

    return data_tbl


def merge_tfrecord_datasets(dest_tfrec_dir, src_tfrecs):
    """ Merge TFRecord datasets.

    Args:
        dest_tfrec_dir: Path, destination TFRecord directory
        src_tfrecs: dict, where the keys are the suffix of the source TFRecord file and the value is the path to the
        source file

    Returns:
    """

    dest_tfrec_dir.mkdir(exist_ok=True)

    for src_tfrecs_suffix, src_tfrec_fps in src_tfrecs.items():

        for src_tfrec_fp in src_tfrec_fps:
            shutil.copy(src_tfrec_fp, dest_tfrec_dir / f'{src_tfrec_fp.name}_{src_tfrecs_suffix}')


def create_table_for_tfrecord_dataset(tfrec_fps, data_fields, delete_corrupted_tfrec_files=False, verbose=True,
                                      logger=None):
    """ Create table with examples from the TFRecord file with scalar features/attributes defined in `data_fields`.

        Args:
            tfrec_fps: list of Paths, TFRecord shards filepaths
            data_fields: dict, 'data_field_name': 'data_type'
            delete_corrupted_tfrec_files: bool, if True it will delete corrupted TFRecord files
            verbose: bool, verbose

        Returns: pandas DataFrame, TFRecord dataset csv file
    """

    tfrec_tbls = []
    for fp in tfrec_fps:
        if verbose:
            if logger:
                logger.info(f'Iterating over {fp}...')
            else:
                print(f'Iterating over {fp}...')

        try:
            tfrec_tbls.append(create_table_with_tfrecord_examples(fp, data_fields))

        except Exception as e:
            if logger:
                logger.info(f'Failed to read {fp}.\n {e}')
            else:
                print(f'Failed to read {fp}.\n {e}')
            if delete_corrupted_tfrec_files:
                if logger:
                    logger.info(f'Deleting corrupted {fp}...')
                else:
                    print(f'Deleting {fp}...')
                fp.unlink()
                (fp.parent / f'{fp.name}.csv').unlink()

    # concatenate tables for all TFRecord shards
    if len(tfrec_fps) > 1:
        tfrec_tbl = pd.concat(tfrec_tbls, axis=0)
    else:
        tfrec_tbl = tfrec_tbls[0]

    return tfrec_tbl


if __name__ == '__main__':

    tf.config.set_visible_devices([], 'GPU')

    # create shards table for a tfrecord data set
    tfrec_dir = Path('/home6/msaragoc/work_dir/Kepler-TESS_exoplanet/data/tfrecords/TESS/tfrecords_tess_spoc_2min_s1-s88_4-25-2025_1536_data/tfrecords_tess_spoc_2min_s1-s88_4-25-2025_1536_agg_bdslabels_diffimg_targetsnotshared')
    # get filepaths for TFRecord shards
    tfrec_fps = list(tfrec_dir.glob('shard-*'))
    data_fields = {  # extra data fields that you want to see in the table
        'uid': 'str',
        'target_id': 'int',
        'tce_plnt_num': 'int',
        'sector_run': 'str',  # COMMENT FOR KEPLER!!
        'label': 'str',
    }
    delete_corrupted_tfrec_files = False
    verbose = True

    tfrec_tbl = create_table_for_tfrecord_dataset(tfrec_fps, data_fields,
                                                  delete_corrupted_tfrec_files=delete_corrupted_tfrec_files,
                                                  verbose=verbose)
    tfrec_tbl.to_csv(tfrec_dir / 'shards_tbl.csv', index=False)
