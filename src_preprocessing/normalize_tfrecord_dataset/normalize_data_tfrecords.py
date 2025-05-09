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
from src_preprocessing.lc_preprocessing.utils_preprocessing import (get_out_of_transit_idxs_glob,
                                                                    get_out_of_transit_idxs_loc)
from src_preprocessing.lc_preprocessing.preprocess import centering_and_normalization
from src_preprocessing.tf_util.example_util import get_feature


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
        # TODO: no need to have the data type in the config yaml file
        scalarParamVal = np.array(get_feature(example, scalarParam))
        # if normStatsScalars[scalarParam]['info']['dtype'] == 'int':
        #     scalarParamVal = np.array(example.features.feature[scalarParam].int64_list.value)
        # elif normStatsScalars[scalarParam]['info']['dtype'] == 'float':
        #     scalarParamVal = np.array(example.features.feature[scalarParam].float_list.value)

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


def normalize_diff_img(example, normStatsDiff_img, imgs_dims, zero_division_eps=1e-10):
    """ Normalize the difference image features for the example.

    Args:
        example: serialized example
        normStatsDiff_img: dict, normalization statistics for the centroid views
        imgs_dims: tuple with the dimensions of the images features.
        zero_division_eps: float, term added to denominator to avoid division by zero in normalization processes

    Returns: norm_diff_img_feat, dict with normalized difference image features for the example
    """

    # initialize dictionary to store the normalized features
    norm_diff_img_feat = {}

    target_centered_prefix_feature_lst = ['', '_tc']

    for target_centered_prefix_feature in target_centered_prefix_feature_lst:

        data_example = {}

        # read non-normalized difference image features for example
        if f'diff_imgs{target_centered_prefix_feature}' in example.features.feature:
            data_example[f'diff_imgs{target_centered_prefix_feature}'] = tf.reshape(tf.io.parse_tensor(
                example.features.feature[f'diff_imgs{target_centered_prefix_feature}'].bytes_list.value[0], tf.float32),
                imgs_dims).numpy()
        if f'oot_imgs{target_centered_prefix_feature}' in example.features.feature:
            data_example[f'oot_imgs{target_centered_prefix_feature}'] = tf.reshape(tf.io.parse_tensor(
                example.features.feature[f'oot_imgs{target_centered_prefix_feature}'].bytes_list.value[0], tf.float32),
                              imgs_dims).numpy()
        if f'snr_imgs{target_centered_prefix_feature}' in example.features.feature:
            data_example[f'snr_imgs{target_centered_prefix_feature}'] = tf.reshape(tf.io.parse_tensor(
                example.features.feature[f'snr_imgs{target_centered_prefix_feature}'].bytes_list.value[0], tf.float32),
                              imgs_dims).numpy()
        if f'neighbors_imgs{target_centered_prefix_feature}' in example.features.feature:
            data_example[f'neighbors_imgs{target_centered_prefix_feature}'] = tf.reshape(tf.io.parse_tensor(
                example.features.feature[f'neighbors_imgs{target_centered_prefix_feature}'].bytes_list.value[0],
                tf.float32), imgs_dims).numpy()


        # min-max normalization
        # x_n = (x - min(x)) / (max(x) - min(x))
        for img_type, img_data in data_example.items():

            img_data_minmaxn = np.array(img_data)
            img_data_minmaxn =  ((img_data_minmaxn - normStatsDiff_img[img_type]['min']) /
                                 (normStatsDiff_img[img_type]['max'] - normStatsDiff_img[img_type]['min'] +
                                  zero_division_eps))

            norm_diff_img_feat[f'{img_type}_minmaxnorm_trainset'] = img_data_minmaxn

        # diff_imgs_minmaxn = np.array(diff_imgs)
        # oot_imgs_minmaxn = np.array(oot_imgs)
        #
        # diff_imgs_minmaxn = ((diff_imgs_minmaxn - normStatsDiff_img['diff_imgs']['min']) /
        #                      (normStatsDiff_img['diff_imgs']['max'] - normStatsDiff_img['diff_imgs']['min'] +
        #                       zero_division_eps))
        # oot_imgs_minmaxn = ((oot_imgs_minmaxn - normStatsDiff_img['oot_imgs']['min']) /
        #                     (normStatsDiff_img['oot_imgs']['max'] - normStatsDiff_img['oot_imgs']['min'] +
        #                      zero_division_eps))
        #
        # norm_diff_img_feat.update({'diff_imgs_minmaxnorm_trainset': diff_imgs_minmaxn,
        #                            'oot_imgs_minmaxnorm_trainset': oot_imgs_minmaxn})

        # standardization
        # x_n = (x - med(x)) / (std(x) + eps)
        for img_type, img_data in data_example.items():
            img_data_std = np.array(img_data)
            img_data_std = ((img_data_std - normStatsDiff_img[img_type]['median']) /
                            (normStatsDiff_img[img_type]['std'] + zero_division_eps))

            norm_diff_img_feat[f'{img_type}_std_trainset'] = img_data_std

        # diff_imgs_std = np.array(diff_imgs)
        # oot_imgs_std = np.array(oot_imgs)
        #
        # diff_imgs_std = ((diff_imgs_std - normStatsDiff_img['diff_imgs']['median']) /
        #                  (normStatsDiff_img['diff_imgs']['std'] + zero_division_eps))
        #
        # oot_imgs_std = ((oot_imgs_std - normStatsDiff_img['oot_imgs']['median']) /
        #                 (normStatsDiff_img['oot_imgs']['std'] + zero_division_eps))
        #
        # norm_diff_img_feat.update({'diff_imgs_std_trainset': diff_imgs_std,
        #                            'oot_imgs_std_trainset': oot_imgs_std})

    return norm_diff_img_feat


def normalize_examples(destTfrecDir, srcTfrecFile, normStats):
    """ Normalize examples in TFRecords.

    :param destTfrecDir:  Path, destination TFRecord directory for the normalized data
    :param srcTfrecFile: Path, source TFRecord directory with the non-normalized data
    :param normStats: dict, normalization statistics used for normalizing the data
    :return:
    """

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

            # normalize centroid time series
            if 'centroid' in normStats:
                norm_centr_feat = normalize_centroid(example, normStats['centroid'])
                normalizedFeatures.update(norm_centr_feat)

            # normalize diff img and oot img
            if 'diff_img' in normStats:
                norm_diff_img_feat = normalize_diff_img(example, normStats['diff_img'], config['diff_img_data_shape'])
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

    tf.config.set_visible_devices([], 'GPU')

    # get the configuration parameters
    path_to_yaml = Path('/nobackupp19/msaragoc/work_dir/Kepler-TESS_exoplanet/codebase/src_preprocessing/normalize_tfrecord_dataset/config_normalize_data.yaml')

    with(open(path_to_yaml, 'r')) as file:
        config = yaml.safe_load(file)

    # source TFRecord directory
    srcTfrecDir = Path(config['srcTfrecDir'])

    # destination TFRecord
    destTfrecDir = Path(config['destTfrecDir'])
    destTfrecDir.mkdir(exist_ok=True)

    # save run setup
    with open(destTfrecDir / 'normalization_run.yaml', 'w') as file:
        yaml.dump(config, file, sort_keys=False)

    # get source TFRecords file paths to be normalized
    srcTfrecFiles = list(srcTfrecDir.glob('*-shard-*'))

    # load normalization statistics; the keys in each sub-dictionary define which statistics are normalized
    normStats = {param: np.load(stats_fp, allow_pickle=True).item() for param, stats_fp in config['normStats'].items()}

    pool = multiprocessing.Pool(processes=config['nProcesses'])
    jobs = [(destTfrecDir, file, normStats) for file in srcTfrecFiles]
    async_results = [pool.apply_async(normalize_examples, job) for job in jobs]
    pool.close()
    pool.join()

    for async_result in async_results:
        async_result.get()

    print('Normalization finished.')
