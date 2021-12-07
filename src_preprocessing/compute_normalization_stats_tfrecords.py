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

# local
from src_preprocessing.utils_preprocessing import get_out_of_transit_idxs_glob, get_out_of_transit_idxs_loc


# %%

def get_values_from_tfrecord(tfrec_file, scalar_params=None, timeSeriesFDLList=None, centroidList=None, **kwargs):

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
            for scalarParam in scalarParams:
                if scalarParams[scalarParam]['dtype'] == 'int':
                    scalarParamsDict[scalarParam].append(example.features.feature[scalarParam].int64_list.value[0])
                elif scalarParams[scalarParam]['dtype'] == 'float':
                    try:
                        scalarParamsDict[scalarParam].append(example.features.feature[scalarParam].float_list.value[0])
                    except:
                        aaa

        # get FDL centroid time series data
        if timeSeriesFDLList is not None:
            transitDuration = example.features.feature['tce_duration'].float_list.value[0]
            orbitalPeriod = example.features.feature['tce_period'].float_list.value[0]
            idxs_nontransitcadences_glob = get_out_of_transit_idxs_glob(num_bins_glob, transitDuration, orbitalPeriod)
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
                    centroidDict[timeSeries].extend(timeSeriesTce[idxs_nontransitcadences_loc])

        return scalarParamsDict, timeSeriesFDLDict, centroidDict


def get_values_from_tfrecords(tfrec_files, scalar_params=None, timeSeriesFDLList=None, centroidList=None, **kwargs):

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

    return scalarParamsDict, timeSeriesFDLDict, centroidDict


# %% comment/add parts related to features not present/present in the TFRecords that one wants to normalize

if __name__ == '__main__':

    # source TFRecord directory
    tfrecDir = Path(
        '/data5/tess_project/Data/tfrecords/Kepler/Q1-Q17_DR25/tfrecordskeplerdr25-dv_g301-l31_spline_nongapped_newvalpcs_tessfeaturesadjs_12-1-2021_data/tfrecordskeplerdr25-dv_g301-l31_spline_nongapped_newvalpcs_tessfeaturesadjs_12-1-2021_experiment')

    nProcesses = 15

    # get only training set TFRecords
    tfrecTrainFiles = [file for file in tfrecDir.iterdir() if
                       'train-shard' in file.stem]  # if file.stem.startswith('train-shard')]

    clip_value_centroid = 30  # in arcsec; 115.5 and 30 arcsec for TESS and Kepler, respectively

    # get out-of-transit indices for the local and global views
    nr_transit_durations = 5  # number of transit durations in the local view
    num_bins_loc = 31  # number of bins for local view
    num_bins_glob = 301  # number of bins for global view

    # dictionary that provides information on the normalization specificities for each scalar parameters (e.g., placeholder
    # for missing value other than NaN, perform log-transform, clipping factor, data type, replace value for missing value)
    # comment/add scalar parameters that are not/are part of the source TFRecords
    scalarParams = {
        # stellar parameters
        'tce_steff': {'missing_value': None, 'log_transform': False, 'log_transform_eps': np.nan,
                      'clip_factor': 20, 'dtype': 'float', 'replace_value': None},
        'tce_slogg': {'missing_value': None, 'log_transform': False, 'log_transform_eps': np.nan,
                      'clip_factor': 20, 'dtype': 'float', 'replace_value': None},
        'tce_smet': {'missing_value': None, 'log_transform': False, 'log_transform_eps': np.nan,
                     'clip_factor': 20, 'dtype': 'float', 'replace_value': None},
        'tce_sradius': {'missing_value': None, 'log_transform': False, 'log_transform_eps': np.nan,
                        'clip_factor': 20, 'dtype': 'float', 'replace_value': None},
        'tce_smass': {'missing_value': None, 'log_transform': False, 'log_transform_eps': np.nan,
                      'clip_factor': 20, 'dtype': 'float', 'replace_value': None},
        'tce_sdens': {'missing_value': None, 'log_transform': False, 'log_transform_eps': np.nan,
                      'clip_factor': 20, 'dtype': 'float', 'replace_value': None},
        'mag': {'missing_value': None, 'log_transform': False, 'log_transform_eps': np.nan,
                'clip_factor': 20, 'dtype': 'float', 'replace_value': None},
        # secondary
        'tce_maxmes': {'missing_value': None, 'log_transform': False, 'log_transform_eps': np.nan,
                       'clip_factor': 20, 'dtype': 'float', 'replace_value': None},
        'wst_depth': {'missing_value': 0, 'log_transform': False, 'log_transform_eps': np.nan,
                      'clip_factor': 20, 'dtype': 'float', 'replace_value': 0},
        'tce_albedo_stat': {'missing_value': 0, 'log_transform': False, 'log_transform_eps': np.nan, 'clip_factor': 20,
                            'dtype': 'float', 'replace_value': None},
        'tce_ptemp_stat': {'missing_value': 0, 'log_transform': False, 'log_transform_eps': np.nan, 'clip_factor': 20,
                           'dtype': 'float', 'replace_value': None},
        # other diagnostics
        'boot_fap': {'missing_value': -1, 'log_transform': True, 'log_transform_eps': 1e-32,
                     'clip_factor': np.nan, 'dtype': 'float', 'replace_value': None},
        'tce_cap_stat': {'missing_value': 0, 'log_transform': False, 'log_transform_eps': np.nan,
                         'clip_factor': 20, 'dtype': 'float', 'replace_value': None},
        'tce_hap_stat': {'missing_value': 0, 'log_transform': False, 'log_transform_eps': np.nan,
                         'clip_factor': 20, 'dtype': 'float', 'replace_value': None},
        'tce_cap_hap_stat_diff': {'missing_value': None, 'log_transform': False, 'log_transform_eps': np.nan,
                                  'clip_factor': 20, 'dtype': 'float', 'replace_value': None},
        'tce_rb_tcount0n': {'missing_value': None, 'log_transform': False, 'log_transform_eps': np.nan,
                            'clip_factor': np.nan, 'dtype': 'float', 'replace_value': None},
        # centroid
        'tce_fwm_stat': {'missing_value': 0, 'log_transform': False, 'log_transform_eps': np.nan, 'clip_factor': 20,
                         'dtype': 'float', 'replace_value': None},
        'tce_dikco_msky': {'missing_value': 0, 'log_transform': False, 'log_transform_eps': np.nan,
                           'clip_factor': 20, 'dtype': 'float', 'replace_value': None},
        'tce_dicco_msky': {'missing_value': 0, 'log_transform': False, 'log_transform_eps': np.nan,
                           'clip_factor': 20, 'dtype': 'float', 'replace_value': None},
        'tce_dikco_msky_err': {'missing_value': -1, 'log_transform': False, 'log_transform_eps': np.nan,
                               'clip_factor': 20, 'dtype': 'float', 'replace_value': None},
        'tce_dicco_msky_err': {'missing_value': -1, 'log_transform': False, 'log_transform_eps': np.nan,
                               'clip_factor': 20, 'dtype': 'float', 'replace_value': None},
        # flux
        'transit_depth': {'missing_value': None, 'log_transform': False, 'log_transform_eps': np.nan,
                          'clip_factor': 20, 'dtype': 'float', 'replace_value': None},
        'tce_max_mult_ev': {'missing_value': None, 'log_transform': False, 'log_transform_eps': np.nan,
                            'clip_factor': 20, 'dtype': 'float', 'replace_value': None},
        'tce_robstat': {'missing_value': None, 'log_transform': False, 'log_transform_eps': np.nan,
                        'clip_factor': 20, 'dtype': 'float', 'replace_value': None},
        'tce_period': {'missing_value': None, 'log_transform': False, 'log_transform_eps': np.nan,
                       'clip_factor': 20, 'dtype': 'float', 'replace_value': None},
        'tce_prad': {'missing_value': None, 'log_transform': False, 'log_transform_eps': np.nan,
                     'clip_factor': 20, 'dtype': 'float', 'replace_value': None},
        # odd-even
        'odd_se_oot': {'missing_value': None, 'log_transform': False, 'log_transform_eps': np.nan,
                       'clip_factor': 20, 'dtype': 'float', 'replace_value': None},
        'even_se_oot': {'missing_value': None, 'log_transform': False, 'log_transform_eps': np.nan,
                        'clip_factor': 20, 'dtype': 'float', 'replace_value': None},
        'odd_std_oot_bin': {'missing_value': None, 'log_transform': False, 'log_transform_eps': np.nan,
                            'clip_factor': 20, 'dtype': 'float', 'replace_value': None},
        'even_std_oot_bin': {'missing_value': None, 'log_transform': False, 'log_transform_eps': np.nan,
                             'clip_factor': 20, 'dtype': 'float', 'replace_value': None},
        'tce_bin_oedp_stat': {'missing_value': 0, 'log_transform': False, 'log_transform_eps': np.nan,
                              'clip_factor': 400, 'dtype': 'float', 'replace_value': None},
    }

    # FDL centroid time series normalization statistics parameters
    timeSeriesFDLList = ['global_centr_fdl_view', 'local_centr_fdl_view']

    # our centroid time series normalization statistics parameters
    centroidList = ['global_centr_view', 'local_centr_view']  # , 'global_centr_view_adjscl', 'local_centr_view_adjscl']

    idxs_nontransitcadences_loc = get_out_of_transit_idxs_loc(num_bins_loc, nr_transit_durations)  # same for all TCEs
    pool = multiprocessing.Pool(processes=nProcesses)
    tfrecTrainFiles_split = np.array_split(tfrecTrainFiles, nProcesses)
    jobs = [(files, scalarParams, timeSeriesFDLList, centroidList)
            for files in tfrecTrainFiles_split]
    async_results = [pool.apply_async(get_values_from_tfrecords, job,
                                      kwds={'idxs_nontransitcadences_loc': idxs_nontransitcadences_loc}) for job in
                     jobs]
    pool.close()

    if scalarParams is not None:
        scalarParamsDict = {scalarParam: [] for scalarParam in scalarParams}
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

    for async_result in async_results:
        partial_values = async_result.get()
        if scalarParams is not None:
            for param in scalarParams:
                scalarParamsDict[param].extend(partial_values[0][param])
        if timeSeriesFDLList is not None:
            for param in timeSeriesFDLList:
                timeSeriesFDLDict[param].extend(partial_values[1][param])
        if centroidList is not None:
            for param in centroidList:
                centroidDict[param].extend(partial_values[2][param])

    if scalarParams is not None:
        # save normalization statistics for the scalar parameters (median and robust estimator of std)
        scalarParamsDict = {scalarParam: np.array(scalarParamVals) for scalarParam, scalarParamVals in
                            scalarParamsDict.items()}
        scalarNormStats = {scalarParam: {'median': np.nan, 'mad_std': np.nan, 'info': scalarParams[scalarParam]}
                           for scalarParam in scalarParams}
        for scalarParam in scalarParams:

            scalarParamVals = scalarParamsDict[scalarParam]

            # remove missing values so that they do not contribute to the normalization statistics
            if scalarParams[scalarParam]['missing_value'] is not None:
                if scalarParam == 'wst_depth':
                    scalarParamVals = scalarParamVals[
                        np.where(scalarParamVals > scalarParams[scalarParam]['missing_value'])]
                else:
                    scalarParamVals = scalarParamVals[
                        np.where(scalarParamVals != scalarParams[scalarParam]['missing_value'])]

            # remove non-finite values
            scalarParamVals = scalarParamVals[np.isfinite(scalarParamVals)]

            # log transform the data (assumes data is non-negative after adding eps)
            if scalarParams[scalarParam]['log_transform']:

                # add constant value
                if not np.isnan(scalarParams[scalarParam]['log_transform_eps']):
                    scalarParamVals += scalarParams[scalarParam]['log_transform_eps']

                scalarParamVals = np.log10(scalarParamVals)

            # compute median as robust estimate of central tendency
            scalarNormStats[scalarParam]['median'] = np.median(scalarParamVals)
            # compute MAD std as robust estimate of deviation from central tendency
            scalarNormStats[scalarParam]['mad_std'] = stats.mad_std(scalarParamVals) if scalarParam not in [
                'tce_rb_tcount0n'] \
                else np.std(scalarParamVals)

        # save normalization statistics for scalar parameters
        np.save(tfrecDir / 'train_scalarparam_norm_stats.npy', scalarNormStats)
        # create additional csv file with normalization statistics
        scalarNormStatsDataForDf = {}
        for scalarParam in scalarParams:
            scalarNormStatsDataForDf[f'{scalarParam}_median'] = [scalarNormStats[scalarParam]['median']]
            scalarNormStatsDataForDf[f'{scalarParam}_mad_std'] = [scalarNormStats[scalarParam]['mad_std']]
        scalarNormStatsDf = pd.DataFrame(data=scalarNormStatsDataForDf)
        scalarNormStatsDf.to_csv(tfrecDir / 'train_scalarparam_norm_stats.csv', index=False)

    if timeSeriesFDLList is not None:
        # save normalization statistics for FDL centroid time series
        normStatsFDL = {timeSeries: {
            # 'values': timeSeriesFDLMat,
            'oot_median': np.median(timeSeriesFDLDict[timeSeries]),
            'oot_std': stats.mad_std(timeSeriesFDLDict[timeSeries])
        }
            for timeSeries in timeSeriesFDLList}
        np.save(tfrecDir / 'train_fdlcentroid_norm_stats.npy', normStatsFDL)
        # create additional csv file with normalization statistics
        normStatsFDLDataForDf = {}
        for timeSeries in timeSeriesFDLList:
            normStatsFDLDataForDf[f'{timeSeries}_oot_median'] = [normStatsFDL[timeSeries]['oot_median']]
            normStatsFDLDataForDf[f'{timeSeries}_oot_std'] = [normStatsFDL[timeSeries]['oot_std']]
        normStatsFDLDf = pd.DataFrame(data=normStatsFDLDataForDf)
        normStatsFDLDf.to_csv(tfrecDir / 'train_fdlcentroid_norm_stats.csv', index=False)

    if centroidList is not None:
        # save normalization statistics for centroid time series
        normStatsCentroid = {timeSeries: {
            'median': np.median(centroidDict[timeSeries]),
            'std': stats.mad_std(centroidDict[timeSeries]),
            'clip_value': clip_value_centroid
            # 'clip_value': np.percentile(centroidMat[timeSeries], 75) +
            #               1.5 * np.subtract(*np.percentile(centroidMat[timeSeries], [75, 25]))
        }
            for timeSeries in centroidList}
        for timeSeries in centroidList:
            centroidMatClipped = np.clip(centroidDict[timeSeries], a_max=clip_value_centroid, a_min=None)
            clipStats = {
                'median_clip': np.median(centroidMatClipped),
                'std_clip': stats.mad_std(centroidMatClipped)
            }
            normStatsCentroid[timeSeries].update(clipStats)
        np.save(tfrecDir / 'train_centroid_norm_stats.npy', normStatsCentroid)
        # create additional csv file with normalization statistics
        normStatsCentroidDataForDf = {}
        for timeSeries in centroidList:
            normStatsCentroidDataForDf[f'{timeSeries}_median'] = [normStatsCentroid[timeSeries]['median']]
            normStatsCentroidDataForDf[f'{timeSeries}_std'] = [normStatsCentroid[timeSeries]['std']]
            # normStatsCentroidDataForDf['{}_clip_value'.format(timeSeries)] = [normStatsCentroid[timeSeries]['clip_value']]
            normStatsCentroidDataForDf[f'{timeSeries}_clip_value'] = clip_value_centroid
            normStatsCentroidDataForDf[f'{timeSeries}_median_clip'] = [normStatsCentroid[timeSeries]['median_clip']]
            normStatsCentroidDataForDf[f'{timeSeries}_std_clip'] = [normStatsCentroid[timeSeries]['std_clip']]
        normStatsCentroidDf = pd.DataFrame(data=normStatsCentroidDataForDf)
        normStatsCentroidDf.to_csv(tfrecDir / 'train_centroid_norm_stats.csv', index=False)
