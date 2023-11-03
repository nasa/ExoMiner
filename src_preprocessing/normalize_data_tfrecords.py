"""
Perform normalization of source TFRecords files using computed normalization statistics.

Input: TFRecord data set.
Output: new TFRecord data set with normalized data.

`normStats` dictionary controls which features are normalized.

"""

# 3rd party
from pathlib import Path
import multiprocessing
import numpy as np
import tensorflow as tf
import yaml

# local
from src_preprocessing.tf_util import example_util
from src_preprocessing.utils_preprocessing import get_out_of_transit_idxs_glob, get_out_of_transit_idxs_loc
from src_preprocessing.preprocess import centering_and_normalization


def normalize_fdl_centroid(example, normStatsFDLCentroid, auxParams, idxs_nontransitcadences_loc):
    """ Normalize FDL centroid for example.

    :param example: serialized example
    :param normStatsFDLCentroid: dict, normalization statistics and respective normalization protocol
    :param auxParams: dict, auxiliary parameters needed for normalization
    :param idxs_nontransitcadences_loc: NumPy array, list with out-of-transit indices for the local views
    :return:
        norm_centr_fdl_feat: dict, normalized FDL centroid views for the example
    """

    # get out-of-transit indices for the global views
    transitDuration = example.features.feature['tce_duration'].float_list.value[0]
    orbitalPeriod = example.features.feature['tce_period'].float_list.value[0]
    idxs_nontransitcadences_glob = get_out_of_transit_idxs_glob(auxParams['num_bins_glob'],
                                                                transitDuration,
                                                                orbitalPeriod)
    # compute oot global and local flux views std
    glob_flux_view_std = \
        np.std(
            np.array(example.features.feature['global_'
                                              'flux_view_'
                                              'fluxnorm'].float_list.value)[idxs_nontransitcadences_glob],
            ddof=1)
    loc_flux_view_std = \
        np.std(
            np.array(
                example.features.feature['local_'
                                         'flux_view_'
                                         'fluxnorm'].float_list.value)[idxs_nontransitcadences_loc], ddof=1)

    # center and normalize FDL centroid time series
    glob_centr_fdl_view = np.array(example.features.feature['global_centr_fdl_view'].float_list.value)
    assert normStatsFDLCentroid['global_centr_fdl_view']['oot_std'] > 0
    glob_centr_fdl_view_norm = \
        centering_and_normalization(glob_centr_fdl_view,
                                    normStatsFDLCentroid['global_centr_fdl_view']['oot_median'],
                                    normStatsFDLCentroid['global_centr_fdl_view']['oot_std']
                                    )
    std_oot_glob = np.std(glob_centr_fdl_view_norm[idxs_nontransitcadences_glob], ddof=1)
    assert std_oot_glob > 0
    glob_centr_fdl_view_norm *= glob_flux_view_std / std_oot_glob
    loc_centr_fdl_view = np.array(example.features.feature['local_centr_fdl_view'].float_list.value)
    assert normStatsFDLCentroid['local_centr_fdl_view']['oot_std'] > 0
    loc_centr_fdl_view_norm = \
        centering_and_normalization(loc_centr_fdl_view,
                                    normStatsFDLCentroid['local_centr_fdl_view']['oot_median'],
                                    normStatsFDLCentroid['local_centr_fdl_view']['oot_std']
                                    )
    std_oot_loc = np.std(loc_centr_fdl_view_norm[idxs_nontransitcadences_loc], ddof=1)
    assert std_oot_loc > 0
    loc_centr_fdl_view_norm *= loc_flux_view_std / std_oot_loc

    norm_centr_fdl_feat = {
        'local_centr_fdl_view_norm': loc_centr_fdl_view_norm,
        'global_centr_fdl_view_norm': glob_centr_fdl_view_norm,
    }

    return norm_centr_fdl_feat


def normalize_scalar_parameters(example, normStatsScalars, epsilon=1e-32):
    """ Normalize scalar features for example.

    :param example: serialized example
    :param normStatsScalars: dict, normalization statistics and respective normalization protocol for each feature
    :param epsilon: float, factor added to the denominator of the normalization to avoid division by zero
    :return:
        norm_scalar_feat: dict, normalized scalar features for the example
    """

    norm_scalar_feat = {f'{scalar_param}_norm': np.nan for scalar_param in normStatsScalars}

    # normalize scalar parameters
    for scalarParam in normStatsScalars:
        # get the scalar value from the example
        if normStatsScalars[scalarParam]['info']['dtype'] == 'int':
            scalarParamVal = np.array(example.features.feature[scalarParam].int64_list.value)
        elif normStatsScalars[scalarParam]['info']['dtype'] == 'float':
            scalarParamVal = np.array(example.features.feature[scalarParam].float_list.value)

        replace_flag = False
        # check if there is a placeholder for missing value
        if normStatsScalars[scalarParam]['info']['missing_value'] is not None:
            if scalarParam in ['wst_depth']:
                replace_flag = scalarParamVal < \
                               normStatsScalars[scalarParam]['info']['missing_value']
            else:
                replace_flag = scalarParamVal == \
                               normStatsScalars[scalarParam]['info']['missing_value']

        if ~np.isfinite(scalarParamVal):  # always replace if value is non-finite
            replace_flag = True

        if replace_flag:
            # replace by a defined value
            if normStatsScalars[scalarParam]['info']['replace_value'] is not None:
                scalarParamVal = normStatsScalars[scalarParam]['info']['replace_value']
            else:  # replace by the median
                scalarParamVal = normStatsScalars[scalarParam]['median']

        # log transform the data (assumes data is non-negative after adding eps)
        # assumes that the median is already log-transformed, but not any possible replace value
        if normStatsScalars[scalarParam]['info']['log_transform'] and \
                not (replace_flag and scalarParamVal == normStatsScalars[scalarParam]['median']):

            # add constant value
            if not np.isnan(normStatsScalars[scalarParam]['info']['log_transform_eps']):
                scalarParamVal += normStatsScalars[scalarParam]['info']['log_transform_eps']

            scalarParamVal = np.log10(scalarParamVal)

        # clipping the data to median +- clip_factor * MAD std to remove outliers
        if not np.isnan(normStatsScalars[scalarParam]['info']['clip_factor']):
            scalarParamVal = np.clip([scalarParamVal],
                                     normStatsScalars[scalarParam]['median'] -
                                     normStatsScalars[scalarParam]['info']['clip_factor'] *
                                     normStatsScalars[scalarParam]['mad_std'],
                                     normStatsScalars[scalarParam]['median'] +
                                     normStatsScalars[scalarParam]['info']['clip_factor'] *
                                     normStatsScalars[scalarParam]['mad_std']
                                     )[0]

        # standardization
        if normStatsScalars[scalarParam]['info']['standardize']:
            # TODO: add value to avoid division by zero?
            # try:
            #     assert normStatsScalars[scalarParam]['mad_std'] > 0
            # except:
            #     print(f'{scalarParam} has zero MAD std.')
            scalarParamVal = (scalarParamVal - normStatsScalars[scalarParam]['median']) / \
                             (normStatsScalars[scalarParam]['mad_std'] + epsilon)

        # add normalized feature to dictionary of normalized features
        norm_scalar_feat[f'{scalarParam}_norm'] = [scalarParamVal]

    return norm_scalar_feat


def normalize_centroid(example, normStatsCentroid):
    """ Normalize the centroid views for the example.

    :param example: serialized example
    :param normStatsCentroid: dict, normalization statistics for the centroid views
    :return:
        norm_centroid_feat: dict, normalized centroid views for the example
    """

    norm_centroid_feat = {}

    glob_centr_view = np.array(example.features.feature['global_centr_view'].float_list.value)
    loc_centr_view = np.array(example.features.feature['local_centr_view'].float_list.value)
    loc_centr_view_var = np.array(example.features.feature['local_centr_view_var'].float_list.value)

    # 1) clipping to physically meaningful distance in arcsec
    glob_centr_view_std_clip = np.clip(glob_centr_view,
                                       a_max=normStatsCentroid['global_centr_view']['clip_value'],
                                       a_min=None)
    assert normStatsCentroid['global_centr_view']['std_clip'] > 0
    glob_centr_view_std_clip = centering_and_normalization(glob_centr_view_std_clip,
                                                           # normStats['centroid']['global_centr_view']['median_'
                                                           #                                            'clip'],
                                                           np.median(glob_centr_view_std_clip),
                                                           normStatsCentroid['global_centr_view']['std_clip'])

    loc_centr_view_std_clip = np.clip(loc_centr_view,
                                      a_max=normStatsCentroid['local_centr_view']['clip_value'],
                                      a_min=None)
    assert normStatsCentroid['local_centr_view']['std_clip'] > 0
    loc_centr_view_std_clip = centering_and_normalization(loc_centr_view_std_clip,
                                                          # normStats['centroid']['local_centr_view']['median_'
                                                          #                                           'clip'],
                                                          np.median(loc_centr_view_std_clip),
                                                          normStatsCentroid['local_centr_view']['std_clip'])
    norm_centroid_feat.update({'global_centr_view_std_clip': glob_centr_view_std_clip,
                               'local_centr_view_std_clip': loc_centr_view_std_clip})

    # 2) no clipping
    assert normStatsCentroid['global_centr_view']['std'] > 0
    glob_centr_view_std_noclip = centering_and_normalization(glob_centr_view,
                                                             # normStats['centroid']['global_centr_'
                                                             #                       'view']['median'],
                                                             np.median(glob_centr_view),
                                                             normStatsCentroid['global_centr_view']['std'])
    assert normStatsCentroid['local_centr_view']['std'] > 0
    loc_centr_view_std_noclip = centering_and_normalization(loc_centr_view,
                                                            # normStats['centroid']['local_centr_view']['median'],
                                                            np.median(loc_centr_view),
                                                            normStatsCentroid['local_centr_view']['std'])
    loc_centr_view_std_noclip_var = loc_centr_view_var / normStatsCentroid['local_centr_view']['std']
    norm_centroid_feat.update({'global_centr_view_std_noclip': glob_centr_view_std_noclip,
                               'local_centr_view_std_noclip': loc_centr_view_std_noclip,
                               'local_centr_view_std_noclip_var': loc_centr_view_std_noclip_var})

    # # 3) center each centroid individually using their median and divide by the standard deviation of the
    # # training set
    # glob_centr_view_medind_std = glob_centr_view - np.median(glob_centr_view)
    # glob_centr_view_medind_std /= normStatsCentroid['global_centr_view']['std']
    # loc_centr_view_medind_std = loc_centr_view - np.median(loc_centr_view)
    # loc_centr_view_medind_std /= normStatsCentroid['local_centr_view']['std']

    # 4) normalize adjusted scale centroid for TESS
    if 'global_centr_view_adjscl_median' in normStatsCentroid:
        glob_centr_view_adjscl = np.array(example.features.feature['global_centr_view_adjscl'].float_list.value)
        loc_centr_view_adjscl = np.array(example.features.feature['local_centr_view_adjscl'].float_list.value)
        assert normStatsCentroid['global_centr_view']['std_adjscl'] > 0
        glob_centr_view_std_noclip_adjscl = centering_and_normalization(glob_centr_view_adjscl,
                                                                        # normStats['centroid']['global_centr_'
                                                                        #                       'view']['median'],
                                                                        np.median(glob_centr_view_adjscl),
                                                                        normStatsCentroid['global_centr_view'][
                                                                            'std_adjscl'])
        assert normStatsCentroid['local_centr_view']['std_adjscl'] > 0
        loc_centr_view_std_noclip_adjscl = centering_and_normalization(loc_centr_view_adjscl,
                                                                       # normStats['centroid']['local_centr_view']['median'],
                                                                       np.median(loc_centr_view_adjscl),
                                                                       normStatsCentroid['local_centr_view'][
                                                                           'std_adjscl'])
        norm_centroid_feat.update({'global_centr_view_std_noclip_adjscl': glob_centr_view_std_noclip_adjscl,
                                   'local_centr_view_std_noclip_adjscl': loc_centr_view_std_noclip_adjscl})

    return norm_centroid_feat


def normalize_diff_img(example, normStatsDiff_img):
    """ Normalize the difference image features for the example.

    Args:
        example: serialized example
        normStatsDiff_img: dict, normalization statistics for the centroid views

    Returns: norm_diff_img_feat, dict with normalized difference image features for the example
    """

    # initialize dictionary to store the normalized features
    norm_diff_img_feat = {}

    # read non-normalized difference image features for example
    diff_imgs = tf.reshape(tf.io.parse_tensor(example.features.feature['diff_imgs'].bytes_list.value[0], tf.float32),
                           normStatsDiff_img['diff_img']['feature_dims']).numpy()
    oot_imgs = tf.reshape(tf.io.parse_tensor(example.features.feature['oot_imgs'].bytes_list.value[0], tf.float32),
                           normStatsDiff_img['oot_img']['feature_dims']).numpy()

    # OPTION 2: min-max normalization
    diff_imgs_opt2 = np.array(diff_imgs)
    oot_imgs_opt2 = np.array(oot_imgs)

    # set NaNs to zero for diff img
    diff_imgs_opt2[np.isnan(diff_imgs_opt2)] = 0

    # set NaNs to -1 for oot img
    oot_imgs_opt2[np.isnan(oot_imgs_opt2)] = -1

    # x_n = (x - min(x)) / (max(x) - min(x))
    diff_imgs_opt2 = ((diff_imgs_opt2 - normStatsDiff_img['diff_img']['min']) /
                      (normStatsDiff_img['diff_img']['max'] - normStatsDiff_img['diff_img']['min'] +
                       normStatsDiff_img['zero_division_eps']))
    oot_imgs_opt2 = ((oot_imgs_opt2 - normStatsDiff_img['oot_img']['min']) /
                     (normStatsDiff_img['oot_img']['max'] - normStatsDiff_img['oot_img']['min'] +
                      normStatsDiff_img['zero_division_eps']))

    norm_diff_img_feat.update({'diff_imgs_opt2': diff_imgs_opt2,
                               'oot_imgs_opt2': oot_imgs_opt2})

    # OPTION 4: standardization
    diff_imgs_opt4 = np.array(diff_imgs)
    oot_imgs_opt4 = np.array(oot_imgs)

    # set NaNs to zero for diff img
    diff_imgs_opt4[np.isnan(diff_imgs_opt4)] = 0

    # set missing values to 25th quantile for oot images
    quantile_oot_img = np.nanquantile(oot_imgs_opt4, normStatsDiff_img['oot_img']['quantile_oot'], axis=(1, 2))
    for img_i, img in enumerate(oot_imgs_opt4):
        # when all values are missing, quantile chosen will also be NaN
        if np.isnan(quantile_oot_img[img_i]):
            img[np.isnan(img)] = -1
        else:
            img[np.isnan(img)] = quantile_oot_img[img_i]

    # x_n = (x - med(x)) / (std(x) + eps)
    diff_imgs_opt4 = ((diff_imgs_opt4 - normStatsDiff_img['diff_img']['median']) /
                      (normStatsDiff_img['diff_img']['std'] + normStatsDiff_img['zero_division_eps']))

    oot_imgs_opt4 = ((oot_imgs_opt4 - normStatsDiff_img['oot_img']['median']) /
                     (normStatsDiff_img['oot_img']['std'] + normStatsDiff_img['zero_division_eps']))

    norm_diff_img_feat.update({'diff_imgs_opt4': diff_imgs_opt4,
                               'oot_imgs_opt4': oot_imgs_opt4})

    return norm_diff_img_feat


def normalize_examples(destTfrecDir, srcTfrecFile, normStats, auxParams):
    """ Normalize examples in TFRecords.

    :param destTfrecDir:  Path, destination TFRecord directory for the normalized data
    :param srcTfrecFile: Path, source TFRecord directory with the non-normalized data
    :param normStats: dict, normalization statistics used for normalizing the data
    :param auxParams: dict, auxiliary parameters needed for normalization
    :return:
    """

    # get out-of-transit indices for the local views
    idxs_nontransitcadences_loc = get_out_of_transit_idxs_loc(auxParams['num_bins_loc'],
                                                              auxParams['nr_transit_durations'])  # same for all TCEs

    with tf.io.TFRecordWriter(str(destTfrecDir / srcTfrecFile.name)) as writer:

        # iterate through the source shard
        tfrecord_dataset = tf.data.TFRecordDataset(str(srcTfrecFile))

        for string_record in tfrecord_dataset.as_numpy_iterator():

            example = tf.train.Example()
            example.ParseFromString(string_record)

            normalizedFeatures = {}

            # normalize scalar features
            if 'scalar_params' in normStats:
                norm_scalar_feat = normalize_scalar_parameters(example, normStats['scalar_params'])
                normalizedFeatures.update(norm_scalar_feat)

            # normalize FDL centroid time series
            if 'fdl_centroid' in normStats:
                norm_centr_fdl_feat = normalize_fdl_centroid(example,
                                                             normStats['fdl_centroid'],
                                                             auxParams,
                                                             idxs_nontransitcadences_loc)
                normalizedFeatures.update(norm_centr_fdl_feat)

            # normalize centroid time series
            if 'centroid' in normStats:
                norm_centr_feat = normalize_centroid(example, normStats['centroid'])
                normalizedFeatures.update(norm_centr_feat)

            # normalize diff img and oot img
            if 'diff_img' in normStats:
                norm_diff_img_feat = normalize_diff_img(example, normStats['diff_img'])
                normalizedFeatures.update(norm_diff_img_feat)

            for normalizedFeature in normalizedFeatures:

                if isinstance(normalizedFeatures[normalizedFeature], list):  # check for 1-D lists
                    example_util.set_float_feature(example, normalizedFeature,
                                                   normalizedFeatures[normalizedFeature],
                                                   allow_overwrite=True)
                elif len(normalizedFeatures[normalizedFeature].shape) < 2:  # check for 1-D NumPy arrays
                    example_util.set_float_feature(example, normalizedFeature, normalizedFeatures[normalizedFeature],
                                                   allow_overwrite=True)
                elif len(normalizedFeatures[normalizedFeature].shape) >= 2:  # check for N-D Numpy arrays with N >= 2
                    example_util.set_tensor_feature(example, normalizedFeature,
                                                    np.array(normalizedFeatures[normalizedFeature]),
                                                    allow_overwrite=True)

            writer.write(example.SerializeToString())


if __name__ == '__main__':

    # get the configuration parameters
    path_to_yaml = Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/codebase/src_preprocessing/config_normalize_data.yaml')

    with(open(path_to_yaml, 'r')) as file:
        config = yaml.safe_load(file)

    # source TFRecord directory
    srcTfrecDir = Path(config['srcTfrecDir'])

    # destination TFRecord
    destTfrecDir = srcTfrecDir.parent / f'{srcTfrecDir.name}-{config["destTfrecDirName"]}'
    destTfrecDir.mkdir(exist_ok=True)

    # get source TFRecords file paths to be normalized
    srcTfrecFiles = [file for file in srcTfrecDir.iterdir() if 'shard' in file.stem and file.suffix != '.csv']

    # load normalization statistics; the keys in the 'scalar_params' dictionary define which statistics are normalized
    normStatsDir = Path(config['normStatsDir'])
    normStats = {
        'scalar_params': np.load(normStatsDir / 'train_scalarparam_norm_stats.npy', allow_pickle=True).item(),
    #     # 'fdl_centroid': np.load(normStatsDir / 'train_fdlcentroid_norm_stats.npy', allow_pickle=True).item(),
        'centroid': np.load(normStatsDir / 'train_centroid_norm_stats.npy', allow_pickle=True).item(),
    #     'diff_img': np.load(normStatsDir / 'train_diff_img_stats.npy', allow_pickle=True).item()
    }
    # del normStats['scalar_params']['tce_depth']
    # normStats = {
    #     'diff_img': {
    #         'diff_img': {'median': 1, 'std': 1e-1, 'min': 1, 'max': 2, 'feature_dims': (5, 11, 11)},
    #         'oot_img': {'median': 1, 'std': 1e-1, 'quantile_oot': 0.25, 'min': 1, 'max': 2, 'feature_dims': (5, 11, 11)},
    #         'zero_division_eps': 1e-10,
    #     }
    # }

    # WRITE CODE HERE TO ADJUST NORMALIZATION STATISTICS (E.G., ADD SCALAR FEATURES THAT YOU  WANT TO NORMALIZE AND ARE
    # NOT IN  THE NORMALIZATION STATS DICTIONARY
    # for TPS
    # normStats['scalar_params'] = {key: val for key, val in normStats['scalar_params'].items()
    #                               if key in ['tce_steff', 'tce_slogg', 'tce_smet', 'tce_sradius', 'tce_smass', 'tce_sdens',
    #                                          'transit_depth', 'tce_duration', 'tce_period', 'tce_max_mult_ev']}
    # for TESS
    # normStats['scalar_params'] = {key: val for key, val in normStats['scalar_params'].items()
    #                               if key in ['tce_steff', 'tce_slogg', 'tce_smet', 'tce_sradius', 'tce_smass', 'tce_sdens',
    #                                          'transit_depth', 'tce_duration', 'tce_period']}
    # normStats['scalar_params'] = {key: val for key, val in normStats['scalar_params'].items()
    #                               if key not in ['tce_fwm_stat', 'tce_rb_tcount0n']}
    # for param in ['global_centr_view', 'local_centr_view']:  # add scaled centroid needed for TESS
    #     normStats['centroid'][param]['median_adsjcl'] = normStats['centroid'][param]['median']
    #     normStats['centroid'][param]['std_adsjcl'] = normStats['centroid'][param]['std']

    # normStats['scalar_params']['tce_steff']['info']['dtype'] = 'float'

    pool = multiprocessing.Pool(processes=config['nProcesses'])
    jobs = [(destTfrecDir, file, normStats, config['auxParams']) for file in srcTfrecFiles]
    async_results = [pool.apply_async(normalize_examples, job) for job in jobs]
    pool.close()

    for async_result in async_results:
        async_result.get()

    print('Normalization finished.')
