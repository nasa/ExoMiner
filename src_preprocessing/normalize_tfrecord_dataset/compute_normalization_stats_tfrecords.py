"""
Compute normalization statistics for scalar parameters, time series, ... This usually involves iterating through the
TFRecords of a given set (e.g., training set), get all valid values and compute stats such as mean, median,
std, ...
"""

# 3rd party
from pathlib import Path
import pandas as pd
import numpy as np
from astropy.stats import mad_std
import tensorflow as tf
import multiprocessing
import yaml
import argparse
from tqdm import tqdm

# local
from src_preprocessing.lc_preprocessing.utils_preprocessing import (get_out_of_transit_idxs_glob,
                                                                    get_out_of_transit_idxs_loc)
from src_preprocessing.tf_util.example_util import get_feature



def robust_scale_per_image(arr, eps=1e-10):
    """Per-image robust scaling: (x - median) / (MAD-based sigma).
    
    param arr: np.ndarray, input array
    param eps: float, small value to avoid division-by-zero
    return: np.ndarray, robustly scaled array
    """
    
    med = np.nanmedian(arr)
    sigma = mad_std(arr, ignore_nan=True) + eps
    return (arr - med) / sigma

def compute_global_percentiles_after_per_image_scaling(tfrec_fps, channels,
                                                       sample_per_image=1024,
                                                       eps=1e-6):
    """
    Iterate TFRecord files; for each example and channel:
      - parse the image
      - per-image robust-scale it
      - sample normalized pixels to build a pooled distribution
    Then compute p1/p99 and 'a' (= max(|p1|, |p99|)) per channel.

    param tfrec_fps: list, paths to TFRecord files
    param channels: list, names of channels to compute percentiles for
    param sample_per_image: int, number of pixels to sample per image after per-image robust scaling; set to -1 to use all pixels
    param eps: float, small value to avoid division-by-zero
    return: dict, percentiles per channel
    """
    
    pools = {ch: [] for ch in channels}

    for tfrecFile in tfrec_fps:
        ds = tf.data.TFRecordDataset(str(tfrecFile))
        for string_record in ds.as_numpy_iterator():
            ex = tf.train.Example()
            ex.ParseFromString(string_record)

            for ch in channels:
                if ch not in ex.features.feature:
                    continue
                vals = np.array(ex.features.feature[ch].float_list.value, dtype=np.float32)
                # Per-image robust scale
                z = robust_scale_per_image(vals, eps=eps)
                z = z[np.isfinite(z)]
                if z.size == 0:
                    continue
                n = min(sample_per_image, z.size)
                if sample_per_image == -1:
                    pools[ch].append(z)
                else:
                    idx = np.random.choice(z.size, n, replace=False)
                    pools[ch].append(z[idx])

    # Aggregate & compute percentiles per channel
    percentiles = {}
    for ch, chunks in pools.items():
        if len(chunks) == 0:
            # No samples collected for this channel; skip
            continue
        pool = np.concatenate(chunks)
        p1 = np.percentile(pool, 1.0)
        p99 = np.percentile(pool, 99.0)
        # a = max(abs(p1), abs(p99)) + eps
        percentiles[ch] = {'p1': float(p1), 'p99': float(p99)}  # , 'a': float(a)}
        
    return percentiles


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

        print(f'Computing normalization statistics for {scalarParam}...')

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
        scalarNormStats[scalarParam]['mad_std'] = mad_std(scalarParamVals) \
            if scalarParam not in ['tce_rb_tcount0n'] else np.std(scalarParamVals)
        # fallback to std if MAD std is zero to prevent explosion of values
        if scalarNormStats[scalarParam]['mad_std'] == 0:
            scalarNormStats[scalarParam]['mad_std'] = np.std(scalarParamVals)
    
    save_fp = config['norm_dir'] / 'train_scalarparam_norm_stats.npy'
    print(f'Saving computed normalization statistics for scalar parameters to {save_fp.resolve()}')
    # save normalization statistics for scalar parameters
    np.save(save_fp, scalarNormStats)

    # create additional csv file with normalization statistics
    scalarNormStatsDataForDf = {}
    for scalarParam in config['scalarParams']:
        scalarNormStatsDataForDf[f'{scalarParam}_median'] = scalarNormStats[scalarParam]['median']
        scalarNormStatsDataForDf[f'{scalarParam}_mad_std'] = scalarNormStats[scalarParam]['mad_std']

    scalarNormStatsDf = pd.Series(data=scalarNormStatsDataForDf)

    return scalarNormStatsDf


def compute_centroid_norm_stats(centroidDict, config):
    """  Compute centroid time series data normalization statistics.

    :param centroidDict: dict, centroid time series
    :param config: dict, configuration parameters

    :return:
        normStatsCentroidDf, pandas DataFrame with normalization statistics for the data
    """

    print(f"Computing normalization statistics for centroid time series: {config['centroidList']}")

    # save normalization statistics for centroid time series
    normStatsCentroid = {timeSeries: {
        'median': np.median(centroidDict[timeSeries]),
        'std': mad_std(centroidDict[timeSeries]),
        'clip_value': config['clip_value_centroid']
        # 'clip_value': np.percentile(centroidMat[timeSeries], 75) +
        #               1.5 * np.subtract(*np.percentile(centroidMat[timeSeries], [75, 25]))
    }
        for timeSeries in config['centroidList']}
    
    for timeSeries in config['centroidList']:
        centroidMatClipped = np.clip(centroidDict[timeSeries], a_max=config['clip_value_centroid'], a_min=None)
        clipStats = {
            'median_clip': np.median(centroidMatClipped),
            'std_clip': mad_std(centroidMatClipped)
        }
        normStatsCentroid[timeSeries].update(clipStats)
    
    save_fp = config['norm_dir'] / 'train_centroid_norm_stats.npy'
    print(f'Saving computed normalization statistics for centroid time series to {save_fp.resolve()}')
    np.save(save_fp, normStatsCentroid)
    
    # create additional csv file with normalization statistics
    normStatsCentroidDataForDf = {}
    for timeSeries in config['centroidList']:
        normStatsCentroidDataForDf[f'{timeSeries}_median'] = normStatsCentroid[timeSeries]['median']
        normStatsCentroidDataForDf[f'{timeSeries}_std'] = normStatsCentroid[timeSeries]['std']
        # normStatsCentroidDataForDf['{}_clip_value'.format(timeSeries)] = normStatsCentroid[timeSeries]['clip_value']
        normStatsCentroidDataForDf[f'{timeSeries}_clip_value'] = config['clip_value_centroid']
        normStatsCentroidDataForDf[f'{timeSeries}_median_clip'] = normStatsCentroid[timeSeries]['median_clip']
        normStatsCentroidDataForDf[f'{timeSeries}_std_clip'] = normStatsCentroid[timeSeries]['std_clip']

    normStatsCentroidDf = pd.Series(data=normStatsCentroidDataForDf)

    return normStatsCentroidDf


def compute_diff_img_data_norm_stats(diff_imgDict, config):
    """  Compute difference image data normalization statistics.

    :param diff_imgDict: dict, difference image data for the images in the data
    :param config: dict, configuration parameters

    :return:
        normStatsDiff_imgDF, pandas DataFrame with normalization statistics for the data
    """
    
    print(f"Computing normalization statistics for difference image data: {config['diff_imgList']}")

    normStatsDiff_img = {diffimgs: {
        'median': np.nanmedian(diff_imgDict[diffimgs]),  # need to flatten each entry
        'std': mad_std(diff_imgDict[diffimgs], ignore_nan=True),
        'min': np.nanmin(diff_imgDict[diffimgs]),
        'max': np.nanmax(diff_imgDict[diffimgs]),
    }
        for diffimgs in config['diff_imgList']}

    # # recompute std only for neighbor images ignoring zero values
    # for diffimgs in config['diff_imgList']:
    #     if 'neighbor' in config['diff_imgList']:
    #         normStatsDiff_img[diffimgs]['std'] = stats.mad_std(diff_imgDict[diffimgs][diff_imgDict[diffimgs] != 0],
    #                                                            ignore_nan=True)
    # get min and max only for neighbor images from config so min-max normalization maps to [-1, 1]
    for diffimgs in config['diff_imgList']:
        if 'neighbor' in diffimgs:
            normStatsDiff_img[diffimgs]['max'] = config['neighbors_img_tmag_diff_range'][1]  # max of the range
            normStatsDiff_img[diffimgs]['min'] = (config['neighbors_img_tmag_diff_range'][1] + config['neighbors_img_tmag_diff_range'][0]) / 2  # midpoint of the range

    save_fp = config['norm_dir'] / 'train_diffimg_norm_stats.npy'
    print(f'Saving computed normalization statistics for difference image data to {save_fp.resolve()}')
    np.save(save_fp, normStatsDiff_img)

    # create additional csv file with normalization statistics
    normStatsDiff_imgForDf = {}
    for diffimgs in config['diff_imgList']:
        normStatsDiff_imgForDf[f'{diffimgs}_median'] = normStatsDiff_img[diffimgs]['median']
        normStatsDiff_imgForDf[f'{diffimgs}_std'] = normStatsDiff_img[diffimgs]['std']
        normStatsDiff_imgForDf[f'{diffimgs}_min'] = normStatsDiff_img[diffimgs]['min']
        normStatsDiff_imgForDf[f'{diffimgs}_max'] = normStatsDiff_img[diffimgs]['max']
    normStatsDiff_imgDf = pd.Series(data=normStatsDiff_imgForDf)

    return normStatsDiff_imgDf


def get_values_from_tfrecord(tfrec_file, scalar_params=None, centroidList=None, diff_imgList=None, max_n_examples_shard=-1, **kwargs):
    """  Extracts feature values from a TFRecord file for computing normalization statistics.

    :param tfrec_file: path to source TFRecord file
    :param scalar_params: dict, scalar parameters to be normalized and normalization info for each
    :param centroidList: list, name of centroid time series
    :param diff_imgList: list, name of difference image features
    :param max_n_examples_shard: int, maximum number of examples used to compute statistics per TFRecord file. Computing the normalization 
        statistics relies on having enough memory to load all the data for each feature. Use a smaller set of the data to get an 
        approximation of the dataset statistics.
    :param kwargs: dict, auxiliary parameters
    :return:
        scalarParamsDict: dict, list of values for each scalar parameters used to compute normalization statistics
        centroidDict: dict, list of values for the centroid time series used to compute normalization statistics
        diff_imgDict: dict, list of values for the difference image features used to compute normalization statistics
    """

    # iterate through the shard
    tfrecord_dataset = tf.data.TFRecordDataset(str(tfrec_file))
    
    n_examples_in_dataset = sum(1 for _ in tf.data.TFRecordDataset(tfrec_file))
    
    if max_n_examples_shard == -1:
        max_n_examples_shard = np.inf

    n_examples_in_dataset = min(n_examples_in_dataset, max_n_examples_shard)

    if scalar_params is not None:
        # scalarParamsDict = {scalarParam: [] for scalarParam in scalar_params}
        scalarParamsDict = {scalarParam: np.empty(n_examples_in_dataset) for scalarParam in scalar_params}
    else:
        scalarParamsDict = None

    # our centroid time series normalization statistics parameters
    if centroidList is not None:
        # flattened_size_centroid_ts = {timeSeries: np.prod(kwargs['centroid_ts_shape'][timeSeries]) for timeSeries in centroidList}
        # centroidDict = {timeSeries: np.empty(n_examples_in_dataset * flattened_size_centroid_ts[timeSeries]) for timeSeries in centroidList}
        centroidDict = {timeSeries: [] if 'glob' in timeSeries else np.empty(n_examples_in_dataset * len(kwargs['idxs_nontransitcadences_loc'][0])) for timeSeries in centroidList}
    else:
        centroidDict = None

    if diff_imgList is not None:
        flattened_size_diff_img = np.prod(kwargs['diff_img_data_shape'])
        diff_imgDict = {diffimgs: np.empty(n_examples_in_dataset * flattened_size_diff_img) for diffimgs in diff_imgList}
    else:
        diff_imgDict = None
        
    # for string_i, string_record in tqdm(enumerate(tfrecord_dataset.as_numpy_iterator()), desc=f'Get data from TCEs in TFRecord {tfrec_file.name}', total=n_examples_in_dataset):
    for string_i, string_record in enumerate(tfrecord_dataset.as_numpy_iterator()):
        
        if string_i == max_n_examples_shard:
            print(f"Reached the maximum number of examples allowed ({max_n_examples_shard}) in {tfrec_file} for computing statistics. Stopping iteration over this file...")
            break

        example = tf.train.Example()
        example.ParseFromString(string_record)

        # get scalar parameters data
        if scalar_params is not None:
            for scalarParam in scalar_params:
                # scalarParamsDict[scalarParam].append(get_feature(example, scalarParam)[0])
                scalarParamsDict[scalarParam][string_i] = get_feature(example, scalarParam)[0]

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
                    # centroidDict[timeSeries][string_i * len(idxs_nontransitcadences_glob[0]):(string_i + 1) * len(idxs_nontransitcadences_glob[0])] = \
                    #     timeSeriesTce[idxs_nontransitcadences_glob]
                else:
                    # centroidDict[timeSeries].extend(timeSeriesTce[kwargs['idxs_nontransitcadences_loc']])
                    centroidDict[timeSeries][string_i * len(kwargs['idxs_nontransitcadences_loc'][0]):(string_i + 1) * len(kwargs['idxs_nontransitcadences_loc'][0])] = \
                        timeSeriesTce[kwargs['idxs_nontransitcadences_loc']]

        # get diff img data
        if diff_imgList is not None:
            for diffimgs in diff_imgList:
                # diffimgsTce = tf.io.parse_tensor(serialized=example.features.feature[diffimgs].bytes_list.value[0],
                                                #  out_type='float').numpy()
                diffimgsTce = np.array(example.features.feature[diffimgs].float_list.value, dtype=np.float32)
                # diff_imgDict[diffimgs].extend(diffimgsTce)
                diff_imgDict[diffimgs][string_i * flattened_size_diff_img: (string_i + 1) * flattened_size_diff_img] = diffimgsTce

    return scalarParamsDict, centroidDict, diff_imgDict


def get_values_from_tfrecords(tfrec_files, scalar_params=None, centroidList=None, diff_imgList=None, max_n_examples_shard=-1, **kwargs):
    """ Extracts feature values from a list of TFRecord files for computing normalization statistics.

    :param tfrec_files: list, paths to source TFRecord files
    :param scalar_params: dict, scalar parameters to be normalized and normalization info for each
    :param centroidList: list, name of centroid time series
    :param diff_imgList: list, name of difference image features
    :param max_n_examples_shard: int, maximum number of examples used to compute statistics per TFRecord file. Computing the normalization 
        statistics relies on having enough memory to load all the data for each feature. Use a smaller set of the data to get an 
        approximation of the dataset statistics.
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

    # for tfrecFile in tqdm(tfrec_files, desc='Finished getting data from TFRecord file', total=len(tfrec_files), unit='job'):
    for tfrecFile in tfrec_files:

        scalarParamsDict_tfrecord, centroidDict_tfrecord, diff_imgDict_tfrecord = \
            get_values_from_tfrecord(tfrecFile, scalar_params, centroidList, diff_imgList, max_n_examples_shard, **kwargs)

        if scalar_params is not None:
            for param in scalar_params:
                scalarParamsDict[param].append(scalarParamsDict_tfrecord[param])  # append array

        if centroidList is not None:
            for param in centroidList:
                centroidDict[param].append(centroidDict_tfrecord[param])

        if diff_imgList is not None:
            for param in diff_imgList:
                diff_imgDict[param].append(diff_imgDict_tfrecord[param])

    # concatenate all arrays at the end
    if scalar_params is not None:
        for param in scalar_params:
            scalarParamsDict[param] = np.concatenate(scalarParamsDict[param])

    if centroidList is not None:
        for param in centroidList:
            centroidDict[param] = np.concatenate(centroidDict[param])

    if diff_imgList is not None:
        for param in diff_imgList:
            diff_imgDict[param] = np.concatenate(diff_imgDict[param])

    return scalarParamsDict, centroidDict, diff_imgDict


def compute_normalization_stats(tfrec_fps, config):
    """ Compute normalization statistics for different features from data in a set of TFRecord files specified by the
    file paths in `tfrec_fps`.

    Args:
        tfrec_fps: list, Path objects of TFRecord file paths used to compute the normalization statistics
        config: dict, auxiliary configuration parameters for normalization methods

    Returns:

    """
    
    config['norm_dir'] = Path(config['norm_dir'])
    config['norm_dir'].mkdir(exist_ok=True)
    
    with open(config['norm_dir'] / 'config_compute_normalization_stats.yaml', 'w') as config_file:
        yaml.dump(config, config_file)

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
        
        # number of jobs is equal to number of TFRecord files
        jobs = [([tfrec_fp], config['scalarParams'], config['centroidList'], config['diff_imgList'], config['max_n_examples_shard'])
                for tfrec_fp in tfrec_fps]
        
        with multiprocessing.Pool(processes=config['n_processes_compute_norm_stats']) as pool, tqdm(desc='Finished computing norm stats job', total=len(jobs), unit='job') as pbar:

            def _on_end(_):
                pbar.update(1)
                
            async_results = []
            for job in jobs:
                ar_apply= pool.apply_async(
                    get_values_from_tfrecords, 
                    job,
                    callback=_on_end,
                    kwds={'idxs_nontransitcadences_loc': idxs_nontransitcadences_loc,
                          'num_bins_loc': config['num_bins_loc'],
                          'num_bins_glob': config['num_bins_glob'],
                          'diff_img_data_shape': config['diff_img_data_shape'],
                          'centroid_ts_shape': config['centroid_ts_shape'],
                          }
                )
                async_results.append(ar_apply)
            
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
            
        print('Aggregated extracted data.')
    else:
        scalarParamsDict, centroidDict, diff_imgDict = \
            get_values_from_tfrecords(tfrec_fps,
                                      config['scalarParams'],
                                      config['centroidList'],
                                      config['diff_imgList'],
                                      idxs_nontransitcadences_loc=idxs_nontransitcadences_loc,
                                      num_bins_loc=config['num_bins_loc'],
                                      num_bins_glob=config['num_bins_glob'],
                                      diff_img_data_shape=config['diff_img_data_shape'],
                                      centroid_ts_shape=config['centroid_ts_shape'],
                                      )

    print('Finished extracting data. Started computing normalization statistics...')

    if config['scalarParams'] is not None:

        print('Computing normalization statistics for scalar parameters...')

        scalar_norm_stats_df = compute_scalar_params_norm_stats(scalarParamsDict, config)
        scalar_norm_stats_df.to_csv(config['norm_dir'] / 'train_scalarparam_norm_stats.csv')

        print('Done.')

    if config['centroidList'] is not None:

        print('Computing normalization statistics for centroid motion data...')

        centroid_norm_stats_df = compute_centroid_norm_stats(centroidDict, config)
        centroid_norm_stats_df.to_csv(config['norm_dir'] / 'train_centroid_norm_stats.csv')

        print('Done.')

    if config['diff_imgList'] is not None:

        print('Computing normalization statistics for difference image data...')

        diff_img_data_norm_stats = compute_diff_img_data_norm_stats(diff_imgDict, config)
        diff_img_data_norm_stats.to_csv(config['norm_dir'] / 'train_diffimg_norm_stats.csv')

        print('Done.')
        
        # compute global percentiles after per-image scaling
        # skip neighbors here (they're already clipped to use affine mapping)
        channels_for_pcts = [ch for ch in config['diff_imgList'] if 'neighbors' not in ch]

        print('Computing global percentiles in normalized space (per-image robust scaling)...')
        pcts = compute_global_percentiles_after_per_image_scaling(
            tfrec_fps=tfrec_fps,
            channels=channels_for_pcts,
            sample_per_image=config.get('sample_per_image_pcts', 1024),
        )
        print('Done. Merging percentiles into train_diffimg_norm_stats.npy ...')

        # Load the previously saved dict and update it with p1/p99/a per channel
        diff_stats_fp = config['norm_dir'] / 'train_diffimg_norm_stats.npy'
        normStatsDiff_img = np.load(diff_stats_fp, allow_pickle=True).item()

        for ch, stats_dict in pcts.items():
            if ch not in normStatsDiff_img:
                normStatsDiff_img[ch] = {}
            normStatsDiff_img[ch].update(stats_dict)

        np.save(diff_stats_fp, normStatsDiff_img)


    print('Finished computing normalization statistics for the data.')


if __name__ == '__main__':

    tf.config.set_visible_devices([], 'GPU')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_fp', type=str, help='File path to YAML configuration file')
    args = parser.parse_args()
    
    with(open(args.config_fp, 'r')) as file:
        config = yaml.unsafe_load(file)
    
    # config = config['compute_norm_stats_params']
    # config['norm_dir'] = Path('/u/msaragoc/work_dir/Kepler-TESS_exoplanet/data/tfrecords/TESS/cv_tfrecords_tess-spoc-tces_2min-s1-s94_ffi-s36-s72-s56s69_10-30-2025_1406/tfrecords/eval_normalized/cv_iter_0/norm_stats')

    # get only training set TFRecords
    # with open('/u/msaragoc/work_dir/Kepler-TESS_exoplanet/data/tfrecords/TESS/cv_tfrecords_tess-spoc-tces_2min-s1-s94_ffi-s36-s72-s56s69_10-30-2025_1406/tfrecords/eval/cv_iterations.yaml', 'r') as dataset_fps:
    #     tfrec_fps = yaml.unsafe_load(dataset_fps)['data_shards_fps'][0]['train']
    tfrec_fps = list(Path(config['tfrecDir']).glob('train-shard*'))
    print(f'Found {len(tfrec_fps)} TFRecord shards')

    compute_normalization_stats(tfrec_fps, config)

    print('Normalization statistics computed.')
