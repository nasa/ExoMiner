""" Utility functions used to manipulate TFRecords. """
import shutil

# 3rd party
import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# local
from src_preprocessing.tf_util import example_util
from src_preprocessing.utils_preprocessing import get_out_of_transit_idxs_glob, get_out_of_transit_idxs_loc
from src_preprocessing.preprocess import centering_and_normalization


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


def normalize_scalar_features(row, normStats):
    """ Standardize parameters for a TCE.

    :param row: pandas Series, TCE parameters
    :param normStats: dict, normalization statistics
    :return:
        pandas Series with TCE parameters standardized
    """

    for feature in row.index:

        # TODO: make these checks more flexible and clearer
        # replace missing values

        # special case for weak secondary depth (<=); depth should not be negative
        if feature == 'wst_depth' and row[feature] <= normStats[feature]['missing_value']:
            row[feature] = normStats[feature]['replace_value']
        # replace by median of the training set
        elif row[feature] == normStats[feature]['missing_value']:

            row[feature] = normStats[feature]['median']
        else:
            # log transform the data
            if normStats[feature]['log_transform']:

                # add constant value to deal with zero for log transform
                if not np.isnan(normStats[feature]['log_transform_eps']):
                    row[feature] += normStats[feature]['log_transform_eps']

                row[feature] = np.log10(row[feature])

            # clipping the data between median +- clip_factor * mad_std
            if not np.isnan(normStats[feature]['clip_factor']):
                row[feature] = np.clip([row[feature]],
                                       normStats[feature]['median'] -
                                       normStats[feature]['clip_factor'] * normStats[feature]['mad_std'],
                                       normStats[feature]['median'] +
                                       normStats[feature]['clip_factor'] * normStats[feature]['mad_std']
                                       )[0]

        # standardization
        row[feature] -= normStats[feature]['median']
        row[feature] /= normStats[feature]['mad_std']

    return row


def normalize_timeseries_features(destTfrecDir, srcTfrecFile, normStats, auxParams):
    """ Normalize time series features in TFRecords.

    :param destTfrecDir:  str, destination TFRecord directory for the normalized data
    :param srcTfrecFile: str, source TFRecord directory with the non-normalized data
    :param normStats: dict, normalization statistics used for normalizing the data
    :return:
    """

    # get out-of-transit indices for the local views
    idxs_nontransitcadences_loc = get_out_of_transit_idxs_loc(auxParams['num_bins_loc'],
                                                              auxParams['nr_transit_durations'])  # same for all TCEs

    with tf.io.TFRecordWriter(os.path.join(destTfrecDir, srcTfrecFile.split('/')[-1])) as writer:

        # iterate through the source shard
        tfrecord_dataset = tf.data.TFRecordDataset(srcTfrecFile)

        for string_record in tfrecord_dataset.as_numpy_iterator():

            example = tf.train.Example()
            example.ParseFromString(string_record)

            normalizedFeatures = {}

            # normalize FDL centroid time series
            # get out-of-transit indices for the global views
            transitDuration = example.features.feature['tce_duration'].float_list.value[0]
            orbitalPeriod = example.features.feature['tce_period'].float_list.value[0]
            idxs_nontransitcadences_glob = get_out_of_transit_idxs_glob(auxParams['num_bins_glob'],
                                                                        transitDuration,
                                                                        orbitalPeriod)
            # compute oot global and local flux views std
            glob_flux_view_std = \
                np.std(
                    np.array(
                        example.features.feature['global_flux_view_fluxnorm'].float_list.value)[idxs_nontransitcadences_glob],
                    ddof=1)
            loc_flux_view_std = \
                np.std(
                    np.array(
                        example.features.feature['local_flux_view_fluxnorm'].float_list.value)[idxs_nontransitcadences_loc],
                    ddof=1)
            # center and normalize FDL centroid time series
            glob_centr_fdl_view = np.array(example.features.feature['global_centr_fdl_view'].float_list.value)
            glob_centr_fdl_view_norm = \
                centering_and_normalization(glob_centr_fdl_view,
                                            normStats['fdl_centroid']['global_centr_fdl_view']['oot_median'],
                                            normStats['fdl_centroid']['global_centr_fdl_view']['oot_std']
                                                                   )
            glob_centr_fdl_view_norm *= glob_flux_view_std / \
                                        np.std(glob_centr_fdl_view_norm[idxs_nontransitcadences_glob], ddof=1)
            loc_centr_fdl_view = np.array(example.features.feature['local_centr_fdl_view'].float_list.value)
            loc_centr_fdl_view_norm = \
                centering_and_normalization(loc_centr_fdl_view,
                                            normStats['fdl_centroid']['local_centr_fdl_view']['oot_median'],
                                            normStats['fdl_centroid']['local_centr_fdl_view']['oot_std']
                                                                  )
            loc_centr_fdl_view_norm *= loc_flux_view_std / np.std(loc_centr_fdl_view_norm[idxs_nontransitcadences_loc],
                                                                  ddof=1)

            # normalize centroid time series; first clip time series to a maximum value for each view
            glob_centr_view = np.array(example.features.feature['global_centr_view'].float_list.value)
            loc_centr_view = np.array(example.features.feature['local_centr_view'].float_list.value)

            # 1) clipping to physically meaningful distance in arcsec
            glob_centr_view_std_clip = np.clip(glob_centr_view,
                                               a_max=normStats['centroid']['global_centr_view']['clip_value'],
                                               a_min=None)
            glob_centr_view_std_clip = centering_and_normalization(glob_centr_view_std_clip,
                                                                   # normStats['centroid']['global_centr_view']['median_'
                                                                   #                                            'clip'],
                                                                   np.median(glob_centr_view_std_clip),
                                                                   normStats['centroid']['global_centr_view']['std_'
                                                                                                              'clip'])

            loc_centr_view_std_clip = np.clip(loc_centr_view, a_max=normStats['centroid']['local_centr_view']['clip_'
                                                                                                              'value'],
                                              a_min=None)
            loc_centr_view_std_clip = centering_and_normalization(loc_centr_view_std_clip,
                                                                   # normStats['centroid']['local_centr_view']['median_'
                                                                   #                                           'clip'],
                                                                  np.median(loc_centr_view_std_clip),
                                                                   normStats['centroid']['local_centr_view']['std_'
                                                                                                             'clip'])

            # 2) no clipping
            glob_centr_view_std_noclip = centering_and_normalization(glob_centr_view,
                                                                     # normStats['centroid']['global_centr_'
                                                                     #                       'view']['median'],
                                                                     np.median(glob_centr_view),
                                                                     normStats['centroid']['global_centr_view']['std'])
            loc_centr_view_std_noclip = centering_and_normalization(loc_centr_view,
                                                                    # normStats['centroid']['local_centr_view']['median'],
                                                                    np.median(loc_centr_view),
                                                                    normStats['centroid']['local_centr_view']['std'])

            # 3) center each centroid individually using their median and divide by the standard deviation of the
            # training set
            glob_centr_view_medind_std = glob_centr_view - np.median(glob_centr_view)
            glob_centr_view_medind_std /= normStats['centroid']['global_centr_view']['std']
            loc_centr_view_medind_std = loc_centr_view - np.median(loc_centr_view)
            loc_centr_view_medind_std /= normStats['centroid']['local_centr_view']['std']

            # add features to the example in the TFRecord
            normalizedFeatures.update({
                'local_centr_fdl_view_norm': loc_centr_fdl_view_norm,
                'global_centr_fdl_view_norm': glob_centr_fdl_view_norm,
                'global_centr_view_std_clip': glob_centr_view_std_clip,
                'local_centr_view_std_clip': loc_centr_view_std_clip,
                'global_centr_view_std_noclip': glob_centr_view_std_noclip,
                'local_centr_view_std_noclip': loc_centr_view_std_noclip,
                'global_centr_view_medind_std': glob_centr_view_medind_std,
                'local_centr_view_medind_std': loc_centr_view_medind_std
            })

            for normalizedFeature in normalizedFeatures:
                example_util.set_float_feature(example, normalizedFeature, normalizedFeatures[normalizedFeature],
                                               allow_overwrite=True)

            writer.write(example.SerializeToString())


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
    """ Create table with examples from the TFRecords with scalar features/attributes defined in `data_fields`.

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

    dest_tfrec_dir.mkdir(exist_ok=True)

    for src_tfrecs_suffix, src_tfrec_fps in src_tfrecs.items():

        for src_tfrec_fp in src_tfrec_fps:
            shutil.copy(src_tfrec_fp, dest_tfrec_dir / f'{src_tfrec_fp.name}_{src_tfrecs_suffix}')


if __name__ == '__main__':

    # create shards table for a tfrecord data set
    tfrec_dir = Path('/home6/msaragoc/work_dir/Kepler-TESS_exoplanet/data/tfrecords/TESS/tfrecords_tess_splinedetrending_s1-s67_4-5-2024_1513_merged')
    tfrec_fps = [fp for fp in tfrec_dir.iterdir() if fp.name.startswith('shard') and fp.suffix != '.csv']
    data_fields = {
        'uid': 'str',
        'target_id': 'int',
        'tce_plnt_num': 'int',
        'sector_run': 'str',
        'label': 'str',
    }
    tfrec_tbls = []
    for fp in tfrec_fps:
        # print(f'Iterating over {fp}...')
        try:
            tfrec_tbls.append(create_table_with_tfrecord_examples(fp, data_fields))
        except Exception as e:
            print(f'Failed to read {fp}.')
            # print(f'Deleting {fp}...')
            # fp.unlink()
            # (fp.parent / f'{fp.name}.csv').unlink()

    tfrec_tbl = pd.concat(tfrec_tbls, axis=0)

    tfrec_tbl.to_csv(tfrec_dir / 'shards_tbl.csv', index=False)
