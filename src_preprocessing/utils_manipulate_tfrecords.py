""" Utility functions used to manipulate TFRecords. """

# 3rd party
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import shutil
import multiprocessing
from tqdm import tqdm


def parse_feature(serialized_example, feature_name):
    """Parse a specified string feature from a serialized TFRecord example.

    :param serialized_example: Serialized TFRecord example
    :param feature_name: Name of the string feature to parse
    :return: Tuple of feature value and serialized example
    """
    
    feature_spec = {
        feature_name: tf.io.FixedLenFeature([], tf.string)
    }
    
    parsed_features = tf.io.parse_single_example(serialized_example, feature_spec)
    
    return parsed_features[feature_name], serialized_example

def make_filter_by_feature_fn(chosen_values):
    """Create a filter function for TFRecord dataset based on inclusion in chosen string values.

    :param chosen_values: Tensor of string values to filter by
    :return: Filtering function
    """
    
    def filter_fn(feature_value):
        return tf.reduce_any(tf.equal(feature_value, chosen_values))
    
    return filter_fn


def parse_uid(serialized_example):
    """Parse only TCE unique IDs 'uid' from the examples in the TFRecord datset.

    :param TF serialized_example: serialized TFRecord example 
    :return tuple: tuple of uid and serialized example
    """
    
    # define the feature spec for just the UuidID
    feature_spec = {
        'uid': tf.io.FixedLenFeature([], tf.string)
    }
    parsed_features = tf.io.parse_single_example(serialized_example, feature_spec)
    
    return parsed_features['uid'], serialized_example


def make_filter_by_uid_fn(chosen_uids):
    """Create filter for TFRecord dataset that excludes TCE examples whose uid is not included in `chosen_uids`.

    :param TF Tensor chosen_uids: chosen uids to filter examples
    :return: filtering function
    """
    
    def filter_uid_tf(uid):
        """Filter function of examples in TFRecord dataset based on 'uid' feature

        :param Tensor string uid: chosen uid
        :return Tensor bool: boolean values that check for uid inclusion
        """
        
        return tf.reduce_any(tf.equal(uid, chosen_uids))
    
    return filter_uid_tf


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

    if not isinstance(tfrec_fps, list):
        raise TypeError("tfrec_fps must be a list file paths.")
    if not isinstance(data_fields, dict):
        raise TypeError("data_fields must be a list of field names.")
    
    tfrec_tbls = []
    for fp_i, fp in tqdm(enumerate(tfrec_fps), desc=f'Iterating over TFRecord files', total=len(tfrec_fps)):
        if verbose:
            if logger:
                logger.info(f'Iterating over {fp} ({fp_i + 1}/{len(tfrec_fps)})...')
            else:
                print(f'Iterating over {fp} ({fp_i + 1}/{len(tfrec_fps)})...')

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


def log_msg(msg, logger=None):
    
    if logger:
        logger.info(msg)
    elif verbose:
        print(msg)
        
def create_table_for_tfrecord_dataset_mp_pool(tfrec_fps, data_fields, delete_corrupted_tfrec_files=False, verbose=True, logger=None, n_procs=1, n_jobs=1):
    """Creates auxiliary shards table for TFRecord dataset containing information on the examples and their data.

    :param list tfrec_fps: list of TFRecord filepaths in the dataset
    :param list data_fields: list of attributes/features to add to the table
    :param bool delete_corrupted_tfrec_files: delete corrupted TFRecord shards and their companion CSV files, defaults to False
    :param bool verbose: verbose, defaults to True
    :param Logger logger: log, defaults to None
    :param int n_procs: number of processes to use, defaults to 1
    :param int n_jobs: split work across these number of jobs, defaults to 1
    :return pandas DataFrame: auxiliary shards table
    """
    
    n_procs = max(1, int(n_procs)) if isinstance(n_procs, int) else 1
    n_jobs = max(1, int(n_jobs)) if isinstance(n_jobs, int) else 1

    if n_jobs > len(tfrec_fps):
        n_jobs = len(tfrec_fps)
    
    if n_procs > n_jobs:
        n_procs = n_jobs
    
    if not isinstance(tfrec_fps, list):
        raise TypeError("tfrec_fps must be a list file paths.")
    if not isinstance(data_fields, dict):
        raise TypeError("data_fields must be a list of field names.")
    
    log_msg(f'Running {n_jobs} jobs in {n_procs} processes.', logger)
        
    results = []
    
    log_msg(f'Found {len(tfrec_fps)} TFRecord files.', logger)
        
    if not tfrec_fps:
        log_msg("No TFRecord files provided.")
        return pd.DataFrame()
        
    tfrec_fps_jobs = np.array_split(tfrec_fps, n_jobs)
    log_msg(f'Split {len(tfrec_fps)} TFRecord files into {n_jobs} chunks.', logger)
    
    log_msg(f'Starting jobs to extract data...', logger)
    with multiprocessing.Pool(processes=n_procs) as pool:
        async_results = [pool.apply_async(create_table_for_tfrecord_dataset, 
                                          (list(tfrec_fps_job), data_fields, delete_corrupted_tfrec_files, verbose, logger)) 
                        for tfrec_fps_job in tfrec_fps_jobs]
    
        for i, async_result in enumerate(async_results):
            try:
                result = async_result.get()
                results.append(result)
            except Exception as e:
                msg = f"Job {i} failed with error: {e}"
                if logger:
                    logger.error(msg)
                elif verbose:
                    print(msg)

    log_msg(f'Finished extracting data. Concatenating tables...', logger)
    tfrec_tbl = pd.concat(results, axis=0) if results else pd.DataFrame()
    
    log_msg(f'Done.', logger)
    
    return tfrec_tbl


if __name__ == '__main__':

    tf.config.set_visible_devices([], 'GPU')

    # create shards table for a tfrecord data set
    tfrec_dir = Path('/home6/msaragoc/work_dir/Kepler-TESS_exoplanet/data/tfrecords/TESS/tfrecords_tess-spoc-tces_2min-s1-s94_ffi-s36-s72-s56s69_10-30-2025_1406')
    # get filepaths for TFRecord shards
    tfrec_fps = list([fp for fp in tfrec_dir.glob('shard-*') if fp.suffix != '.csv'])
    data_fields = {  # extra data fields that you want to see in the table
        'uid': 'str',
        'target_id': 'int',
        'tce_plnt_num': 'int',
        'sector_run': 'str',  # COMMENT FOR KEPLER!!
        'label': 'str',
        'obs_type': 'str',
        'uid_obs_type': 'str',
    }
    delete_corrupted_tfrec_files = False
    verbose = True

    # tfrec_tbl = create_table_for_tfrecord_dataset(tfrec_fps, data_fields,
    #                                               delete_corrupted_tfrec_files=delete_corrupted_tfrec_files,
    #                                               verbose=verbose)
    tfrec_tbl = create_table_for_tfrecord_dataset_mp_pool(tfrec_fps, 
                                                          data_fields,
                                                          delete_corrupted_tfrec_files=delete_corrupted_tfrec_files,
                                                          verbose=verbose,
                                                          logger=None,
                                                          n_procs=36,
                                                          n_jobs=36,
                                                          )
    
    tfrec_tbl.to_csv(tfrec_dir / 'shards_tbl.csv', index=False)
