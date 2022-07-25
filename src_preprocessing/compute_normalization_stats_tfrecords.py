"""
Compute normalization statistics for scalar parameters, timeseries, ... This usually involves iterating through the
TFRecords of a given set (e.g., Kepler training set), get all valid values and compute stats such as mean, median,
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
from src_preprocessing.utils_preprocessing import get_out_of_transit_idxs_glob, get_out_of_transit_idxs_loc
# from paths import path_main


# %%

def get_values_from_tfrecord(tfrec_file, scalar_params=None, timeSeriesFDLList=None, centroidList=None, **kwargs):
    """  Extracts feature values from a TFRecord file for computing normalization statistics.

    :param tfrec_file: path to source TFRecord file
    :param scalar_params: dict, scalar parameters to be normalized and normalization info for each
    :param timeSeriesFDLList: list, name of FDL centroid time series
    :param centroidList: list, name of centroid time series
    :param kwargs: dict, auxiliary parameters
    :return:
        scalarParamsDict: dict, list of values for each scalar parameters used to compute normalization statistics
        timeSeriesFDLDict: dict, list of values for  the FDL centroid  time series used to compute normalization
        statistics
        centroidDict: dict, list of values for the centroid time series used to compute normalization statistics
    """

    if scalar_params is not None:
        scalarParamsDict = {scalarParam: [] for scalarParam in scalar_params}
    else:
        scalarParamsDict = None

    # FDL centroid time series normalization statistics parameters
    if timeSeriesFDLList is not None:
        timeSeriesFDLDict = {timeSeries: [] for timeSeries in timeSeriesFDLList}
    else:
        timeSeriesFDLDict = None

    # our centroid time series normalization statistics parameters
    if centroidList is not None:
        centroidDict = {timeSeries: [] for timeSeries in centroidList}
    else:
        centroidDict = None

    # iterate through the shard
    tfrecord_dataset = tf.data.TFRecordDataset(str(tfrec_file))

    for string_record in tfrecord_dataset.as_numpy_iterator():

        example = tf.train.Example()
        example.ParseFromString(string_record)

        # get scalar parameters data
        if scalar_params is not None:
            for scalarParam in scalar_params:
                if scalar_params[scalarParam]['dtype'] == 'int':
                    scalarParamsDict[scalarParam].append(example.features.feature[scalarParam].int64_list.value[0])
                elif scalar_params[scalarParam]['dtype'] == 'float':
                    scalarParamsDict[scalarParam].append(example.features.feature[scalarParam].float_list.value[0])

        # get FDL centroid time series data
        if timeSeriesFDLList is not None:
            transitDuration = example.features.feature['tce_duration'].float_list.value[0]
            orbitalPeriod = example.features.feature['tce_period'].float_list.value[0]
            idxs_nontransitcadences_glob = get_out_of_transit_idxs_glob(kwargs['num_bins_glob'],
                                                                        transitDuration,
                                                                        orbitalPeriod)
            for timeSeries in timeSeriesFDLList:
                timeSeriesTce = np.array(example.features.feature[timeSeries].float_list.value)
                if 'glob' in timeSeries:
                    timeSeriesFDLDict[timeSeries].extend(timeSeriesTce[idxs_nontransitcadences_glob])
                else:
                    timeSeriesFDLDict[timeSeries].extend(timeSeriesTce[kwargs['idxs_nontransitcadences_loc']])

        # get centroid time series data
        if centroidList is not None:
            for timeSeries in centroidList:
                timeSeriesTce = np.array(example.features.feature[timeSeries].float_list.value)
                if 'glob' in timeSeries:
                    centroidDict[timeSeries].extend(timeSeriesTce[idxs_nontransitcadences_glob])
                else:
                    centroidDict[timeSeries].extend(timeSeriesTce[kwargs['idxs_nontransitcadences_loc']])

    return scalarParamsDict, timeSeriesFDLDict, centroidDict


def get_values_from_tfrecords(tfrec_files, scalar_params=None, timeSeriesFDLList=None, centroidList=None, **kwargs):
    """ Extracts feature values from a list of TFRecord files for computing normalization statistics.

    :param tfrec_files: list, paths to source TFRecord files
    :param scalar_params: dict, scalar parameters to be normalized and normalization info for each
    :param timeSeriesFDLList: list, name of FDL centroid time series
    :param centroidList: list, name of centroid time series
    :param kwargs: dict, auxiliary parameters needed for normalization
    :return:
        scalarParamsDict: dict, list of values for each scalar parameters used to compute normalization statistics
        timeSeriesFDLDict: dict, list of values for  the FDL centroid  time series used to compute normalization
        statistics
        centroidDict: dict, list of values for the centroid time series used to compute normalization statistics
    """

    if scalar_params is not None:
        scalarParamsDict = {scalarParam: [] for scalarParam in scalar_params}
    else:
        scalarParamsDict = None

    # FDL centroid time series normalization statistics parameters
    if timeSeriesFDLList is not None:
        timeSeriesFDLDict = {timeSeries: [] for timeSeries in timeSeriesFDLList}
    else:
        timeSeriesFDLDict = None

    # our centroid time series normalization statistics parameters
    if centroidList is not None:
        centroidDict = {timeSeries: [] for timeSeries in centroidList}
    else:
        centroidDict = None

    for tfrec_i, tfrecFile in enumerate(tfrec_files):

        print(f'[{multiprocessing.current_process().name}] Getting data from {tfrecFile.name} '
              f'({tfrec_i / len(tfrec_files) * 100} %)')

        scalarParamsDict_tfrecord, timeSeriesFDLDict_tfrecord, centroidDict_tfrecord = \
            get_values_from_tfrecord(tfrecFile, scalar_params, timeSeriesFDLList, centroidList, **kwargs)

        if scalar_params is not None:
            for param in scalar_params:
                scalarParamsDict[param].extend(scalarParamsDict_tfrecord[param])

        if timeSeriesFDLList is not None:
            for param in timeSeriesFDLList:
                timeSeriesFDLDict[param].extend(timeSeriesFDLDict_tfrecord[param])

        if centroidList is not None:
            for param in centroidList:
                centroidDict[param].extend(centroidDict_tfrecord[param])

    print(f'[{multiprocessing.current_process().name}] Got data from {tfrecFile.name} '
          f'({tfrec_i / len(tfrec_files) * 100} %)')

    return scalarParamsDict, timeSeriesFDLDict, centroidDict


def compute_normalization_stats(tfrec_fps, config):

    if config['scalarParams'] is not None:
        scalarParamsDict = {scalarParam: [] for scalarParam in config['scalarParams']}
    else:
        scalarParamsDict = None

        # FDL centroid time series normalization statistics parameters
    if config['timeSeriesFDLList'] is not None:
        timeSeriesFDLDict = {timeSeries: [] for timeSeries in config['timeSeriesFDLList']}
    else:
        timeSeriesFDLDict = None

        # our centroid time series normalization statistics parameters
    if config['centroidList'] is not None:
        centroidDict = {timeSeries: [] for timeSeries in config['centroidList']}
    else:
        centroidDict = None

    idxs_nontransitcadences_loc = get_out_of_transit_idxs_loc(config['num_bins_loc'],
                                                              config['nr_transit_durations'])  # same for all TCEs

    if config['n_processes_compute_norm_stats'] is not None:
        pool = multiprocessing.Pool(processes=config['n_processes_compute_norm_stats'])

        tfrecTrainFiles_split = np.array_split(tfrec_fps, config['n_processes_compute_norm_stats'])
        jobs = [(files, config['scalarParams'], config['timeSeriesFDLList'], config['centroidList'])
                for files in tfrecTrainFiles_split]
        async_results = [pool.apply_async(get_values_from_tfrecords, job,
                                          kwds={'idxs_nontransitcadences_loc': idxs_nontransitcadences_loc,
                                                'num_bins_loc': config['num_bins_loc'],
                                                'num_bins_glob': config['num_bins_glob']})
                         for job in jobs]
        pool.close()
        for async_result in async_results:
            partial_values = async_result.get()
            if config['scalarParams'] is not None:
                for param in config['scalarParams']:
                    scalarParamsDict[param].extend(partial_values[0][param])
            if config['timeSeriesFDLList'] is not None:
                for param in config['timeSeriesFDLList']:
                    timeSeriesFDLDict[param].extend(partial_values[1][param])
            if config['centroidList'] is not None:
                for param in config['centroidList']:
                    centroidDict[param].extend(partial_values[2][param])
    else:
        scalarParamsDict, timeSeriesFDLDict, centroidDict = \
            get_values_from_tfrecords(tfrec_fps,
                                      config['scalarParams'],
                                      config['timeSeriesFDLList'],
                                      config['centroidList'],
                                      idxs_nontransitcadences_loc=idxs_nontransitcadences_loc,
                                      num_bins_loc=config['num_bins_loc'],
                                      num_bins_glob=config['num_bins_glob']
                                      )

    if config['scalarParams'] is not None:
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
        scalarNormStatsDf.to_csv(config['norm_dir'] / 'train_scalarparam_norm_stats.csv', index=False)

    if config['timeSeriesFDLList'] is not None:
        # save normalization statistics for FDL centroid time series
        normStatsFDL = {timeSeries: {
            # 'values': timeSeriesFDLMat,
            'oot_median': np.median(timeSeriesFDLDict[timeSeries]),
            'oot_std': stats.mad_std(timeSeriesFDLDict[timeSeries])
        }
            for timeSeries in config['timeSeriesFDLList']}
        np.save(config['norm_dir'] / 'train_fdlcentroid_norm_stats.npy', normStatsFDL)
        # create additional csv file with normalization statistics
        normStatsFDLDataForDf = {}
        for timeSeries in config['timeSeriesFDLList']:
            normStatsFDLDataForDf[f'{timeSeries}_oot_median'] = [normStatsFDL[timeSeries]['oot_median']]
            normStatsFDLDataForDf[f'{timeSeries}_oot_std'] = [normStatsFDL[timeSeries]['oot_std']]
        normStatsFDLDf = pd.DataFrame(data=normStatsFDLDataForDf)
        normStatsFDLDf.to_csv(config['norm_dir'] / 'train_fdlcentroid_norm_stats.csv', index=False)

    if config['centroidList'] is not None:
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
        normStatsCentroidDf.to_csv(config['norm_dir'] / 'train_centroid_norm_stats.csv', index=False)


if __name__ == '__main__':

    # get the configuration parameters
    path_to_yaml = Path('/Users/msaragoc/OneDrive - NASA/Projects/exoplanet_transit_classification/codebase/src_preprocessing/config_compute_normalization_stats.yaml')

    with(open(path_to_yaml, 'r')) as file:
        config = yaml.safe_load(file)

    # get only training set TFRecords
    tfrecTrainFiles = [file for file in Path(config['tfrecDir']).iterdir() if 'train-shard' in file.stem]

    compute_normalization_stats(tfrecTrainFiles, config)

    # # FDL centroid time series normalization statistics parameters
    # timeSeriesFDLList = ['global_centr_fdl_view', 'local_centr_fdl_view']
    #
    # # our centroid time series normalization statistics parameters
    # centroidList = ['global_centr_view', 'local_centr_view']  # , 'global_centr_view_adjscl', 'local_centr_view_adjscl']
    #
    # idxs_nontransitcadences_loc = get_out_of_transit_idxs_loc(config['num_bins_loc'],
    #                                                           config['nr_transit_durations'])  # same for all TCEs
    # pool = multiprocessing.Pool(processes=config['nProcesses'])
    # tfrecTrainFiles_split = np.array_split(tfrecTrainFiles, config['nProcesses'])
    # jobs = [(files, config['scalarParams'], timeSeriesFDLList, centroidList)
    #         for files in tfrecTrainFiles_split]
    # async_results = [pool.apply_async(get_values_from_tfrecords, job,
    #                                   kwds={'idxs_nontransitcadences_loc': idxs_nontransitcadences_loc,
    #                                         'num_bins_loc': config['num_bins_loc'],
    #                                         'num_bins_glob': config['num_bins_glob']})
    #                  for job in jobs]
    # pool.close()
    #
    # if config['scalarParams'] is not None:
    #     scalarParamsDict = {scalarParam: [] for scalarParam in config['scalarParams']}
    # else:
    #     scalarParamsDict = None
    #
    #     # FDL centroid time series normalization statistics parameters
    # if timeSeriesFDLList is not None:
    #     timeSeriesFDLDict = {timeSeries: [] for timeSeries in timeSeriesFDLList}
    # else:
    #     timeSeriesFDLDict = None
    #
    #     # our centroid time series normalization statistics parameters
    # if centroidList is not None:
    #     centroidDict = {timeSeries: [] for timeSeries in centroidList}
    # else:
    #     centroidDict = None
    #
    # for async_result in async_results:
    #     partial_values = async_result.get()
    #     if config['scalarParams'] is not None:
    #         for param in config['scalarParams']:
    #             scalarParamsDict[param].extend(partial_values[0][param])
    #     if timeSeriesFDLList is not None:
    #         for param in timeSeriesFDLList:
    #             timeSeriesFDLDict[param].extend(partial_values[1][param])
    #     if centroidList is not None:
    #         for param in centroidList:
    #             centroidDict[param].extend(partial_values[2][param])
    #
    # if config['scalarParams'] is not None:
    #     # save normalization statistics for the scalar parameters (median and robust estimator of std)
    #     scalarParamsDict = {scalarParam: np.array(scalarParamVals) for scalarParam, scalarParamVals in
    #                         scalarParamsDict.items()}
    #     scalarNormStats = {
    #         scalarParam: {'median': np.nan, 'mad_std': np.nan, 'info': config['scalarParams'][scalarParam]}
    #         for scalarParam in config['scalarParams']}
    #     for scalarParam in config['scalarParams']:
    #
    #         scalarParamVals = scalarParamsDict[scalarParam]
    #
    #         # remove missing values so that they do not contribute to the normalization statistics
    #         if config['scalarParams'][scalarParam]['missing_value'] is not None:
    #             if scalarParam == 'wst_depth':
    #                 scalarParamVals = scalarParamVals[
    #                     np.where(scalarParamVals > config['scalarParams'][scalarParam]['missing_value'])]
    #             else:
    #                 scalarParamVals = scalarParamVals[
    #                     np.where(scalarParamVals != config['scalarParams'][scalarParam]['missing_value'])]
    #
    #         # remove non-finite values
    #         scalarParamVals = scalarParamVals[np.isfinite(scalarParamVals)]
    #
    #         # log transform the data (assumes data is non-negative after adding eps)
    #         if config['scalarParams'][scalarParam]['log_transform']:
    #
    #             # add constant value
    #             if not np.isnan(config['scalarParams'][scalarParam]['log_transform_eps']):
    #                 scalarParamVals += config['scalarParams'][scalarParam]['log_transform_eps']
    #
    #             scalarParamVals = np.log10(scalarParamVals)
    #
    #         # compute median as robust estimate of central tendency
    #         scalarNormStats[scalarParam]['median'] = np.median(scalarParamVals)
    #         # compute MAD std as robust estimate of deviation from central tendency
    #         scalarNormStats[scalarParam]['mad_std'] = stats.mad_std(scalarParamVals) \
    #             if scalarParam not in ['tce_rb_tcount0n'] else np.std(scalarParamVals)
    #
    #     # save normalization statistics for scalar parameters
    #     np.save(tfrecDir / 'train_scalarparam_norm_stats.npy', scalarNormStats)
    #     # create additional csv file with normalization statistics
    #     scalarNormStatsDataForDf = {}
    #     for scalarParam in config['scalarParams']:
    #         scalarNormStatsDataForDf[f'{scalarParam}_median'] = [scalarNormStats[scalarParam]['median']]
    #         scalarNormStatsDataForDf[f'{scalarParam}_mad_std'] = [scalarNormStats[scalarParam]['mad_std']]
    #     scalarNormStatsDf = pd.DataFrame(data=scalarNormStatsDataForDf)
    #     scalarNormStatsDf.to_csv(tfrecDir / 'train_scalarparam_norm_stats.csv', index=False)
    #
    # if timeSeriesFDLList is not None:
    #     # save normalization statistics for FDL centroid time series
    #     normStatsFDL = {timeSeries: {
    #         # 'values': timeSeriesFDLMat,
    #         'oot_median': np.median(timeSeriesFDLDict[timeSeries]),
    #         'oot_std': stats.mad_std(timeSeriesFDLDict[timeSeries])
    #     }
    #         for timeSeries in timeSeriesFDLList}
    #     np.save(tfrecDir / 'train_fdlcentroid_norm_stats.npy', normStatsFDL)
    #     # create additional csv file with normalization statistics
    #     normStatsFDLDataForDf = {}
    #     for timeSeries in timeSeriesFDLList:
    #         normStatsFDLDataForDf[f'{timeSeries}_oot_median'] = [normStatsFDL[timeSeries]['oot_median']]
    #         normStatsFDLDataForDf[f'{timeSeries}_oot_std'] = [normStatsFDL[timeSeries]['oot_std']]
    #     normStatsFDLDf = pd.DataFrame(data=normStatsFDLDataForDf)
    #     normStatsFDLDf.to_csv(tfrecDir / 'train_fdlcentroid_norm_stats.csv', index=False)
    #
    # if centroidList is not None:
    #     # save normalization statistics for centroid time series
    #     normStatsCentroid = {timeSeries: {
    #         'median': np.median(centroidDict[timeSeries]),
    #         'std': stats.mad_std(centroidDict[timeSeries]),
    #         'clip_value': config['clip_value_centroid']
    #         # 'clip_value': np.percentile(centroidMat[timeSeries], 75) +
    #         #               1.5 * np.subtract(*np.percentile(centroidMat[timeSeries], [75, 25]))
    #     }
    #         for timeSeries in centroidList}
    #     for timeSeries in centroidList:
    #         centroidMatClipped = np.clip(centroidDict[timeSeries], a_max=config['clip_value_centroid'], a_min=None)
    #         clipStats = {
    #             'median_clip': np.median(centroidMatClipped),
    #             'std_clip': stats.mad_std(centroidMatClipped)
    #         }
    #         normStatsCentroid[timeSeries].update(clipStats)
    #     np.save(tfrecDir / 'train_centroid_norm_stats.npy', normStatsCentroid)
    #     # create additional csv file with normalization statistics
    #     normStatsCentroidDataForDf = {}
    #     for timeSeries in centroidList:
    #         normStatsCentroidDataForDf[f'{timeSeries}_median'] = [normStatsCentroid[timeSeries]['median']]
    #         normStatsCentroidDataForDf[f'{timeSeries}_std'] = [normStatsCentroid[timeSeries]['std']]
    #         # normStatsCentroidDataForDf['{}_clip_value'.format(timeSeries)] = [normStatsCentroid[timeSeries]['clip_value']]
    #         normStatsCentroidDataForDf[f'{timeSeries}_clip_value'] = config['clip_value_centroid']
    #         normStatsCentroidDataForDf[f'{timeSeries}_median_clip'] = [normStatsCentroid[timeSeries]['median_clip']]
    #         normStatsCentroidDataForDf[f'{timeSeries}_std_clip'] = [normStatsCentroid[timeSeries]['std_clip']]
    #     normStatsCentroidDf = pd.DataFrame(data=normStatsCentroidDataForDf)
    #     normStatsCentroidDf.to_csv(tfrecDir / 'train_centroid_norm_stats.csv', index=False)
