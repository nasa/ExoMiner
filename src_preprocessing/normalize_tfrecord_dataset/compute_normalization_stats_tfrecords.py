"""
Compute normalization statistics for scalar parameters, time series, ... This usually involves iterating through the
TFRecords of a given set (e.g., training set), get all valid values and compute stats such as mean, median,
std, ...
"""

# 3rd party
from pathlib import Path
import pandas as pd
import numpy as np
from astropy import stats
import tensorflow as tf
import multiprocessing
import yaml

# local
from src_preprocessing.lc_preprocessing.utils_preprocessing import (get_out_of_transit_idxs_glob,
                                                                    get_out_of_transit_idxs_loc)
from src_preprocessing.tf_util.example_util import get_feature


def compute_scalar_params_norm_stats(scalarParamsDict, config):
    """  Compute scalar parameters normalization statistics.

    :param scalarParamsDict: dict, scalar parameters
    :param config: dict, configuration parameters

    :return:
        scalarNormStatsDf, pandas DataFrame with normalization statistics for the data
    """

    # save normalization statistics for the scalar parameters (median and robust estimator of std)
    scalarParamsDict = {scalarParam: np.array(scalarParamVals) for scalarParam, scalarParamVals in
                        scalarParamsDict.items()}
    scalarNormStats = {
        scalarParam: {'median': np.nan, 'mad_std': np.nan, 'info': config['scalarParams'][scalarParam]}
        for scalarParam in config['scalarParams']}
    for scalarParam in config['scalarParams']:

        scalarParamVals = scalarParamsDict[scalarParam]

        # remove missing values so that they do not contribute to the normalization statistics
        if config['scalarParams'][scalarParam]['missing_value'] is not None:
            if scalarParam == 'wst_depth':
                scalarParamVals = scalarParamVals[
                    np.where(scalarParamVals > config['scalarParams'][scalarParam]['missing_value'])]
            else:
                scalarParamVals = scalarParamVals[
                    np.where(scalarParamVals != config['scalarParams'][scalarParam]['missing_value'])]

        # remove non-finite values
        scalarParamVals = scalarParamVals[np.isfinite(scalarParamVals)]

        # log transform the data (assumes data is non-negative after adding eps)
        if config['scalarParams'][scalarParam]['log_transform']:

            # add constant value
            if not np.isnan(config['scalarParams'][scalarParam]['log_transform_eps']):
                scalarParamVals += config['scalarParams'][scalarParam]['log_transform_eps']

            scalarParamVals = np.log10(scalarParamVals)

        # compute median as robust estimate of central tendency
        scalarNormStats[scalarParam]['median'] = np.median(scalarParamVals)
        # compute MAD std as robust estimate of deviation from central tendency
        scalarNormStats[scalarParam]['mad_std'] = stats.mad_std(scalarParamVals) \
            if scalarParam not in ['tce_rb_tcount0n'] else np.std(scalarParamVals)

    # save normalization statistics for scalar parameters
    np.save(config['norm_dir'] / 'train_scalarparam_norm_stats.npy', scalarNormStats)

    # create additional csv file with normalization statistics
    scalarNormStatsDataForDf = {}
    for scalarParam in config['scalarParams']:
        scalarNormStatsDataForDf[f'{scalarParam}_median'] = [scalarNormStats[scalarParam]['median']]
        scalarNormStatsDataForDf[f'{scalarParam}_mad_std'] = [scalarNormStats[scalarParam]['mad_std']]

    scalarNormStatsDf = pd.DataFrame(data=scalarNormStatsDataForDf)

    return scalarNormStatsDf


def compute_centroid_norm_stats(centroidDict, config):
    """  Compute centroid time series data normalization statistics.

    :param centroidDict: dict, centroid time series
    :param config: dict, configuration parameters

    :return:
        normStatsCentroidDf, pandas DataFrame with normalization statistics for the data
    """

    # save normalization statistics for centroid time series
    normStatsCentroid = {timeSeries: {
        'median': np.median(centroidDict[timeSeries]),
        'std': stats.mad_std(centroidDict[timeSeries]),
        'clip_value': config['clip_value_centroid']
        # 'clip_value': np.percentile(centroidMat[timeSeries], 75) +
        #               1.5 * np.subtract(*np.percentile(centroidMat[timeSeries], [75, 25]))
    }
        for timeSeries in config['centroidList']}
    for timeSeries in config['centroidList']:
        centroidMatClipped = np.clip(centroidDict[timeSeries], a_max=config['clip_value_centroid'], a_min=None)
        clipStats = {
            'median_clip': np.median(centroidMatClipped),
            'std_clip': stats.mad_std(centroidMatClipped)
        }
        normStatsCentroid[timeSeries].update(clipStats)
    np.save(config['norm_dir'] / 'train_centroid_norm_stats.npy', normStatsCentroid)
    # create additional csv file with normalization statistics
    normStatsCentroidDataForDf = {}
    for timeSeries in config['centroidList']:
        normStatsCentroidDataForDf[f'{timeSeries}_median'] = [normStatsCentroid[timeSeries]['median']]
        normStatsCentroidDataForDf[f'{timeSeries}_std'] = [normStatsCentroid[timeSeries]['std']]
        # normStatsCentroidDataForDf['{}_clip_value'.format(timeSeries)] = [normStatsCentroid[timeSeries]['clip_value']]
        normStatsCentroidDataForDf[f'{timeSeries}_clip_value'] = config['clip_value_centroid']
        normStatsCentroidDataForDf[f'{timeSeries}_median_clip'] = [normStatsCentroid[timeSeries]['median_clip']]
        normStatsCentroidDataForDf[f'{timeSeries}_std_clip'] = [normStatsCentroid[timeSeries]['std_clip']]

    normStatsCentroidDf = pd.DataFrame(data=normStatsCentroidDataForDf)

    return normStatsCentroidDf


def compute_diff_img_data_norm_stats(diff_imgDict, config):
    """  Compute difference image data normalization statistics.

    :param diff_imgDict: dict, difference image data for the images in the data
    :param config: dict, configuration parameters

    :return:
        normStatsDiff_imgDF, pandas DataFrame with normalization statistics for the data
    """

    normStatsDiff_img = {diffimgs: {
        'median': np.nanmedian(diff_imgDict[diffimgs]),  # need to flatten each entry
        'std': stats.mad_std(diff_imgDict[diffimgs], ignore_nan=True),
        'min': np.nanmin(diff_imgDict[diffimgs]),
        'max': np.nanmax(diff_imgDict[diffimgs]),
    }
        for diffimgs in config['diff_imgList']}

    for diffimgs in config['diff_imgList']:
        if 'neighbor' in config['diff_imgList']:
            normStatsDiff_img[diffimgs]['std'] = stats.mad_std(diff_imgDict[diffimgs][diff_imgDict[diffimgs] != 0],
                                                               ignore_nan=True)

    np.save(config['norm_dir'] / 'train_diffimg_norm_stats.npy', normStatsDiff_img)

    # create additional csv file with normalization statistics
    normStatsDiff_imgForDf = {}
    for diffimgs in config['diff_imgList']:
        normStatsDiff_imgForDf[f'{diffimgs}_median'] = [normStatsDiff_img[diffimgs]['median']]
        normStatsDiff_imgForDf[f'{diffimgs}_std'] = [normStatsDiff_img[diffimgs]['std']]
        normStatsDiff_imgForDf[f'{diffimgs}_min'] = [normStatsDiff_img[diffimgs]['min']]
        normStatsDiff_imgForDf[f'{diffimgs}_max'] = [normStatsDiff_img[diffimgs]['max']]
    normStatsDiff_imgDf = pd.DataFrame(data=normStatsDiff_imgForDf)

    return normStatsDiff_imgDf


def get_values_from_tfrecord(tfrec_file, scalar_params=None, centroidList=None, diff_imgList=None, **kwargs):
    """  Extracts feature values from a TFRecord file for computing normalization statistics.

    :param tfrec_file: path to source TFRecord file
    :param scalar_params: dict, scalar parameters to be normalized and normalization info for each
    :param centroidList: list, name of centroid time series
    :param diff_imgList: list, name of difference image features
    :param kwargs: dict, auxiliary parameters
    :return:
        scalarParamsDict: dict, list of values for each scalar parameters used to compute normalization statistics
        centroidDict: dict, list of values for the centroid time series used to compute normalization statistics
        diff_imgDict: dict, list of values for the difference image features used to compute normalization statistics
    """

    if scalar_params is not None:
        scalarParamsDict = {scalarParam: [] for scalarParam in scalar_params}
    else:
        scalarParamsDict = None

    # our centroid time series normalization statistics parameters
    if centroidList is not None:
        centroidDict = {timeSeries: [] for timeSeries in centroidList}
    else:
        centroidDict = None

    if diff_imgList is not None:
        diff_imgDict = {diffimgs: [] for diffimgs in diff_imgList}
    else:
        diff_imgDict = None

    # iterate through the shard
    tfrecord_dataset = tf.data.TFRecordDataset(str(tfrec_file))

    for string_record in tfrecord_dataset.as_numpy_iterator():

        example = tf.train.Example()
        example.ParseFromString(string_record)

        # get scalar parameters data
        if scalar_params is not None:
            for scalarParam in scalar_params:
                scalarParamsDict[scalarParam].append(get_feature(example, scalarParam)[0])

        # get centroid time series data
        if centroidList is not None:
            for timeSeries in centroidList:
                timeSeriesTce = np.array(example.features.feature[timeSeries].float_list.value)
                if 'glob' in timeSeries:
                    transitDuration = example.features.feature['tce_duration'].float_list.value[0]
                    orbitalPeriod = example.features.feature['tce_period'].float_list.value[0]
                    idxs_nontransitcadences_glob = get_out_of_transit_idxs_glob(kwargs['num_bins_glob'],
                                                                                transitDuration,
                                                                                orbitalPeriod)
                    centroidDict[timeSeries].extend(timeSeriesTce[idxs_nontransitcadences_glob])
                else:
                    centroidDict[timeSeries].extend(timeSeriesTce[kwargs['idxs_nontransitcadences_loc']])

        # get diff img data
        if diff_imgList is not None:
            for diffimgs in diff_imgList:
                diffimgsTce = tf.io.parse_tensor(serialized=example.features.feature[diffimgs].bytes_list.value[0],
                                                 out_type='float').numpy()
                diff_imgDict[diffimgs].extend(diffimgsTce)

    return scalarParamsDict, centroidDict, diff_imgDict


def get_values_from_tfrecords(tfrec_files, scalar_params=None, centroidList=None, diff_imgList=None, **kwargs):
    """ Extracts feature values from a list of TFRecord files for computing normalization statistics.

    :param tfrec_files: list, paths to source TFRecord files
    :param scalar_params: dict, scalar parameters to be normalized and normalization info for each
    :param centroidList: list, name of centroid time series
    :param diff_imgList: list, name of difference image features
    :param kwargs: dict, auxiliary parameters needed for normalization
    :return:
        scalarParamsDict: dict, list of values for each scalar parameters used to compute normalization statistics
        centroidDict: dict, list of values for the centroid time series used to compute normalization statistics
        diff_imgDict: dict, list of values for the difference image features used to compute normalization statistics
    """

    if scalar_params is not None:
        scalarParamsDict = {scalarParam: [] for scalarParam in scalar_params}
    else:
        scalarParamsDict = None

    # our centroid time series normalization statistics parameters
    if centroidList is not None:
        centroidDict = {timeSeries: [] for timeSeries in centroidList}
    else:
        centroidDict = None

    if diff_imgList is not None:
        diff_imgDict = {diffimgs: [] for diffimgs in diff_imgList}
    else:
        diff_imgDict = None

    for tfrec_i, tfrecFile in enumerate(tfrec_files):

        print(f'[{multiprocessing.current_process().name}] Getting data from {tfrecFile.name} '
              f'({tfrec_i / len(tfrec_files) * 100} %)')

        scalarParamsDict_tfrecord, centroidDict_tfrecord, diff_imgDict_tfrecord = \
            get_values_from_tfrecord(tfrecFile, scalar_params, centroidList, diff_imgList, **kwargs)

        if scalar_params is not None:
            for param in scalar_params:
                scalarParamsDict[param].extend(scalarParamsDict_tfrecord[param])

        if centroidList is not None:
            for param in centroidList:
                centroidDict[param].extend(centroidDict_tfrecord[param])

        if diff_imgList is not None:
            for param in diff_imgList:
                diff_imgDict[param].extend(diff_imgDict_tfrecord[param])

    print(f'[{multiprocessing.current_process().name}] Finished extracting data (100 %)')

    return scalarParamsDict, centroidDict, diff_imgDict


def compute_normalization_stats(tfrec_fps, config):
    """ Compute normalization statistics for different features from data in a set of TFRecord files specified by the
    file paths in `tfrec_fps`.

    Args:
        tfrec_fps: list, Path objects of TFRecord file paths used to compute the normalization statistics
        config: dict, auxiliary configuration parameters for normalization methods

    Returns:

    """

    if config['scalarParams'] is not None:
        scalarParamsDict = {scalarParam: [] for scalarParam in config['scalarParams']}
    else:
        scalarParamsDict = None

        # our centroid time series normalization statistics parameters
    if config['centroidList'] is not None:
        centroidDict = {timeSeries: [] for timeSeries in config['centroidList']}
    else:
        centroidDict = None

    # diff img normalization statistics parameters
    if config['diff_imgList'] is not None:
        diff_imgDict = {diffimgs: [] for diffimgs in config['diff_imgList']}
    else:
        diff_imgDict = None

    idxs_nontransitcadences_loc = get_out_of_transit_idxs_loc(config['num_bins_loc'],
                                                              config['nr_transit_durations'])  # same for all TCEs

    print(f'Started extracting data from {len(tfrec_fps)} TFRecord files...')

    if config['n_processes_compute_norm_stats'] > 1:
        pool = multiprocessing.Pool(processes=config['n_processes_compute_norm_stats'])

        # tfrecTrainFiles_split = np.array_split(tfrec_fps, config['n_processes_compute_norm_stats'])
        # number of jobs is equal to number of TFRecord files
        jobs = [([tfrec_fp], config['scalarParams'], config['centroidList'], config['diff_imgList'])
                for tfrec_fp in tfrec_fps]
        async_results = [pool.apply_async(get_values_from_tfrecords, job,
                                          kwds={'idxs_nontransitcadences_loc': idxs_nontransitcadences_loc,
                                                'num_bins_loc': config['num_bins_loc'],
                                                'num_bins_glob': config['num_bins_glob']})
                         for job in jobs]
        for async_result in async_results:
            partial_values = async_result.get()
            if config['scalarParams'] is not None:
                for param in config['scalarParams']:
                    scalarParamsDict[param].extend(partial_values[0][param])
            if config['centroidList'] is not None:
                for param in config['centroidList']:
                    centroidDict[param].extend(partial_values[1][param])
            if config['diff_imgList'] is not None:
                for param in config['diff_imgList']:
                    diff_imgDict[param].extend(partial_values[2][param])
        pool.close()
        pool.join()
        print('Aggregated extracted data.')
    else:
        scalarParamsDict, centroidDict, diff_imgDict = \
            get_values_from_tfrecords(tfrec_fps,
                                      config['scalarParams'],
                                      config['centroidList'],
                                      config['diff_imgList'],
                                      idxs_nontransitcadences_loc=idxs_nontransitcadences_loc,
                                      num_bins_loc=config['num_bins_loc'],
                                      num_bins_glob=config['num_bins_glob']
                                      )

    print('Finished extracting data. Started computing normalization statistics...')

    if config['scalarParams'] is not None:

        print('Computing normalization statistics for scalar parameters...')

        scalar_norm_stats_df = compute_scalar_params_norm_stats(scalarParamsDict, config)
        scalar_norm_stats_df.to_csv(config['norm_dir'] / 'train_scalarparam_norm_stats.csv', index=False)

        print('Done.')

    if config['centroidList'] is not None:

        print('Computing normalization statistics for centroid motion data...')

        centroid_norm_stats_df = compute_centroid_norm_stats(centroidDict, config)
        centroid_norm_stats_df.to_csv(config['norm_dir'] / 'train_centroid_norm_stats.csv', index=False)

        print('Done.')

    if config['diff_imgList'] is not None:

        print('Computing normalization statistics for difference image data...')

        diff_img_data_norm_stats = compute_diff_img_data_norm_stats(diff_imgDict, config)
        diff_img_data_norm_stats.to_csv(config['norm_dir'] / 'train_diffimg_norm_stats.csv', index=False)

        print('Done.')

    print('Finished computing normalization statistics for the data.')


if __name__ == '__main__':

    tf.config.set_visible_devices([], 'GPU')

    # get the configuration parameters
    path_to_yaml = Path('/nobackupp19/msaragoc/work_dir/Kepler-TESS_exoplanet/codebase/src_preprocessing/normalize_tfrecord_dataset/config_compute_normalization_stats.yaml')

    with(open(path_to_yaml, 'r')) as file:
        config = yaml.unsafe_load(file)

    # get only training set TFRecords
    tfrecTrainFiles = list(Path(config['tfrecDir']).glob('train-shard*'))

    config['norm_dir'] = Path(config['norm_dir'])
    config['norm_dir'].mkdir(exist_ok=True)

    compute_normalization_stats(tfrecTrainFiles, config)

    print('Normalization statistics computed.')
