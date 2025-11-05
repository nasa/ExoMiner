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
import argparse
from tqdm import tqdm

# local
from src_preprocessing.tf_util import example_util
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


def normalize_diff_img(example, normStatsDiff_img, zero_division_eps=1e-10):
    """Normalize the difference image features for the example.

    Args:
        example: serialized example
        normStatsDiff_img: dict, normalization statistics for the centroid views
        # imgs_dims: tuple with the dimensions of the image features
        zero_division_eps: float, small value to avoid division by zero
    Returns:
        dict: normalized difference image features
    """

    MAX_MAG_RATIO = 5
    norm_diff_img_feat = {}

    def parse_feature(example, feature_name):
        if feature_name in example.features.feature:
            # return tf.reshape(
            #     example.features.feature[feature_name].float_list.value,
            #     imgs_dims
            # ).numpy()
            return example.features.feature[feature_name].float_list.value
        return None

    def min_max_normalize(data, stats):
        return (data - stats['min']) / (stats['max'] - stats['min'] + zero_division_eps)

    def standardize(data, stats):
        return (data - stats['median']) / (stats['std'] + zero_division_eps)

    def fixed_min_max_normalize(data):
        return data / MAX_MAG_RATIO
    
    for prefix in ['', '_tc']:
    
        data_example = {}
    
        for img_type in ['diff_imgs', 'oot_imgs', 'snr_imgs', 'neighbors_imgs']:
            
            feature_name = f'{img_type}{prefix}'
            parsed_data = parse_feature(example, feature_name)
            
            if parsed_data is not None:
                data_example[feature_name] = parsed_data

        for img_type, img_data in data_example.items():
            
            if 'neighbors_imgs' not in img_type:  # normalize difference image data
                
                norm_diff_img_feat[f'{img_type}_minmaxnorm_trainset'] = min_max_normalize(
                    np.array(img_data), normStatsDiff_img[img_type]
                )
                
                norm_diff_img_feat[f'{img_type}_std_trainset'] = standardize(
                    np.array(img_data), normStatsDiff_img[img_type]
                )
            
            else:  # normalize neighbors images
                
                norm_diff_img_feat[f'{img_type}_fixed_min_max_norm'] = fixed_min_max_normalize(
                    np.array(img_data)
                )

    return norm_diff_img_feat


def normalize_examples(destTfrecDir, srcTfrecFile, normStats):
    """ Normalize examples in TFRecords.

    :param destTfrecDir:  Path, destination TFRecord directory for the normalized data
    :param srcTfrecFile: Path, source TFRecord directory with the non-normalized data
    :param normStats: dict, normalization statistics used for normalizing the data
    # :param config: dict, configuration parameters for the normalized data
    :return:
    """

    with tf.io.TFRecordWriter(str(destTfrecDir / srcTfrecFile.name)) as writer:

        # # iterate through the source shard
        # n_examples_in_dataset = sum(1 for _ in tf.data.TFRecordDataset(str(srcTfrecFile)))
        
        tfrecord_dataset = tf.data.TFRecordDataset(str(srcTfrecFile))

        # for string_record in tqdm(tfrecord_dataset.as_numpy_iterator(), desc=f'Normalizing TCEs in {srcTfrecFile.name}', total=n_examples_in_dataset):
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
                norm_diff_img_feat = normalize_diff_img(example, normStats['diff_img'])
                normalizedFeatures.update(norm_diff_img_feat)

            for normalizedFeature in normalizedFeatures:

                if isinstance(normalizedFeatures[normalizedFeature], list) or len(normalizedFeatures[normalizedFeature].shape) < 2:  # check for 1-D lists and 1-D NumPy arryas
                    example_util.set_float_feature(example, normalizedFeature,
                                                   normalizedFeatures[normalizedFeature],
                                                   allow_overwrite=True)
                # elif len(normalizedFeatures[normalizedFeature].shape) >= 2:  # check for N-D Numpy arrays with N >= 2
                #     example_util.set_tensor_feature(example, normalizedFeature,
                #                                     np.array(normalizedFeatures[normalizedFeature]),
                #                                     allow_overwrite=True)
                else:
                    raise ValueError(f'Normalized feature {normalizedFeature} is neither a list or a 1-D NumPy array.')

            writer.write(example.SerializeToString())


def normalize_examples_main(config_fp, src_tfrec_dir=None, dest_tfrec_dir=None):
    """ Wrapper for `normalize_examples()`.

    Args:
        config_fp: str, configuration parameters for data normalization
        src_tfrec_dir: Path, source TFRecord directory for the data to be normalized
        dest_tfrec_dir: Path, destination TFRecord directory for the normalized data

    Returns:

    """

    with open(config_fp, 'r') as file:
        config = yaml.safe_load(file)

    if src_tfrec_dir is not None:
        config['srcTfrecDir'] = src_tfrec_dir
    if dest_tfrec_dir is not None:
        config['destTfrecDir'] = dest_tfrec_dir

    # source TFRecord directory
    srcTfrecDir = Path(config['srcTfrecDir'])

    # destination TFRecord
    destTfrecDir = Path(config['destTfrecDir'])
    destTfrecDir.mkdir(exist_ok=True)

    # save run setup
    with open(destTfrecDir / 'normalization_run.yaml', 'w') as file:
        yaml.dump(config, file, sort_keys=False)

    # get source TFRecords file paths to be normalized
    srcTfrecFiles = list(srcTfrecDir.glob('*shard-*'))
    srcTfrecFiles = [fp for fp in srcTfrecFiles if fp.suffix != '.csv']

    # load normalization statistics; the keys in each sub-dictionary define which statistics are normalized
    normStats = {param: np.load(stats_fp, allow_pickle=True).item() for param, stats_fp in config['normStats'].items()}

    jobs = [(destTfrecDir, file, normStats) for file in srcTfrecFiles]
    if config['nProcesses'] > 1:
        pool = multiprocessing.Pool(processes=config['nProcesses'])
        async_results = [pool.apply_async(normalize_examples, job) for job in jobs]
        pool.close()
        pool.join()

        for async_result in async_results:
            async_result.get()

    else:
        for job in jobs:
            normalize_examples(*job)


if __name__ == '__main__':

    tf.config.set_visible_devices([], 'GPU')

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_fp', type=str, help='File path to YAML configuration file')
    args = parser.parse_args()
    
    normalize_examples_main(args.config_fp)
