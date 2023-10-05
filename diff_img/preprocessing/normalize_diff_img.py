"""
Normalize difference image data in TFRecords using per-image statistics (i.e., no statistics computed based on a
training set.

Input: TFRecord data set with difference image data.
Output: new TFRecord data set with per-image normalized difference image data.
"""

# 3rd party
import tensorflow as tf
from pathlib import Path
import numpy as np
from astropy.stats import mad_std
import multiprocessing

# local
from src_preprocessing.tf_util import example_util


def perimage_normalization_diffimg(example, q_oot, zero_division_eps, features):
    """ Perform per-image normalization of difference image features for a single example in a TFRecord data set.

    Args:
        example: TFRecord example with difference image features to be normalized.
        zero_division_eps: float, term added to denominator to avoid division by zero in normalization processes
        q_oot: float, quantile used to fill missing values in out-of-transit images
        features: dict, key is feature name of difference image feature in the source data that is to be normalized;
        value is a tuple with the dimensions of said feature.

    Returns: example, with added normalized difference image features

    """

    for feature_name, feature_dims in features.items():  # iterate through features to be normalized
        # extract feature of interest
        feature_data_example = tf.reshape(
            tf.io.parse_tensor(example.features.feature[feature_name].bytes_list.value[0], tf.float32),
            feature_dims).numpy()

        # OPTION 1: min-max normalization
        feature_data_example_opt1 = np.array(feature_data_example)
        feature_min_per_img = np.nanmin(feature_data_example_opt1, axis=(1, 2), keepdims=True)
        feature_max_per_img = np.nanmax(feature_data_example_opt1, axis=(1, 2), keepdims=True)

        # set missing values to zero for difference images
        if feature_name == 'diff_imgs':
            feature_data_example_opt1[np.isnan(feature_data_example_opt1)] = 0

        # set missing values to -1 for oot images
        if feature_name == 'oot_imgs':
            feature_data_example_opt1[np.isnan(feature_data_example_opt1)] = -1

        # perform min-max normalization
        # img_n = (img - min(img)) / (max(img) - min(img) + zero_division_eps)
        feature_data_example_opt1 = (feature_data_example_opt1 - feature_min_per_img) / \
                                    (feature_max_per_img - feature_min_per_img + zero_division_eps)

        # add normalized feature to example
        example_util.set_tensor_feature(example, f'{feature_name}_opt1', feature_data_example_opt1)

        # OPTION 3: standardization
        feature_data_example_opt3 = np.array(feature_data_example)
        feature_median_per_img = np.nanmedian(feature_data_example_opt3, axis=(1, 2), keepdims=True)
        feature_madstd_per_img = \
            np.expand_dims(mad_std(feature_data_example_opt3, axis=(1, 2), ignore_nan=True), axis=(1, 2))

        # set missing values to zero for difference images
        if feature_name == 'diff_imgs':
            feature_data_example_opt3[np.isnan(feature_data_example_opt3)] = 0

        # set missing values to 25th quantile for oot images
        if feature_name == 'oot_imgs':
            quantile_oot_img = np.nanquantile(feature_data_example_opt3, q_oot, axis=(1, 2))

            for img_i, img in enumerate(feature_data_example_opt3):
                # when all values are missing, quantile chosen will also be NaN
                if np.isnan(quantile_oot_img[img_i]):
                    img[np.isnan(img)] = -1
                else:
                    img[np.isnan(img)] = quantile_oot_img[img_i]

        # perform standardization
        # img_n = (img - med(img)) / (mad_std(img) + zero_division_eps)
        feature_data_example_opt3 = (feature_data_example_opt3 - feature_median_per_img) / \
                                    (feature_madstd_per_img + zero_division_eps)

        # add normalized feature to example
        example_util.set_tensor_feature(example, f'{feature_name}_opt3', feature_data_example_opt3)

    return example


def preprocess_tfrecs_perimage_normalization_diffimg(src_tfrec_fps, dest_tfrec_dir, zero_division_eps, q_oot, features):
    """ Perform per-image normalization of difference image features for a set of TFRecord files. The TFRecord files
    with the normalized data will be under the directory defined by `dest_tfrec_dir`. The files have the same name as
    the source ones.

    Args:
        src_tfrec_fps: list of Paths, source TFRecord files to be preprocessed (i.e., for which the difference image
        data is going to be normalized and create new TFRecord files
        dest_tfrec_dir: Path, destination directory for the preprocessed data
        zero_division_eps: float, term added to denominator to avoid division by zero in normalization processes
        q_oot: float, quantile used to fill missing values in out-of-transit images
        features: dict, key is feature name of difference image feature in the source data that is to be normalized;
        value is a tuple with the dimensions of said feature.

    Returns:

    """

    for src_tfrec_fp in src_tfrec_fps:  # iterate through source TFRecord files

        print(f'Iterating through shard {src_tfrec_fp}...')

        # set file path to destination TFRecord file to which the examples with the normalized data are written to
        dest_tfrec_fp = dest_tfrec_dir / src_tfrec_fp.name

        with tf.io.TFRecordWriter(str(dest_tfrec_fp)) as writer:

            # iterate through the source shard
            tfrecord_dataset = tf.data.TFRecordDataset(str(src_tfrec_fp))

            for string_record in tfrecord_dataset.as_numpy_iterator():  # iterate through examples in the shard

                example = tf.train.Example()
                example.ParseFromString(string_record)  # parse features

                example = perimage_normalization_diffimg(example, q_oot, zero_division_eps, features)

                # write example with added normalized features to new TFRecord file
                writer.write(example.SerializeToString())


if __name__ == '__main__':

    src_tfrec_dir = Path('/Users/msaragoc/Projects/exoplanet_transit_classification/data/tfrecords/kepler/tfrecordskeplerq1q17dr25-dv_g301-l31_5tr_spline_nongapped_all_features_phases_7-20-2022_1237_data/tfrecordskeplerq1q17dr25-dv_g301-l31_5tr_spline_nongapped_all_features_phases_7-20-2022_1237_updated_stellar_ruwe_confirmedkois_adddiffimg')
    dest_tfrec_dir = src_tfrec_dir.parent / f'{src_tfrec_dir.name}_perimg_normdiffimg'
    zero_division_eps = 1e-10  # term added to denominator to avoid division by zero
    q_oot = 0.25  # set quantile used to fill missing values in oot images
    features = {
        'diff_imgs': (5, 11, 11),  # feature dimension
        'oot_imgs': (5, 11, 11),
    }
    n_procs = 10
    n_jobs = 10

    # get file paths for source TFRecord files under the source directory
    src_tfrec_fps = [fp for fp in src_tfrec_dir.iterdir() if fp.name.startswith('shard-')]
    # create destination directory
    dest_tfrec_dir.mkdir(exist_ok=True)

    # parallel processing
    src_tfrec_fps_jobs = np.array_split(src_tfrec_fps, n_jobs)
    pool = multiprocessing.Pool(processes=n_procs)
    jobs = [(src_tfrec_fps_job, dest_tfrec_dir, zero_division_eps, q_oot, features)
            for src_tfrec_fps_job in src_tfrec_fps_jobs]
    async_results = [pool.apply_async(preprocess_tfrecs_perimage_normalization_diffimg, job) for job in jobs]
    pool.close()

    for async_result in async_results:
        async_result.get()

    print('Normalization finished.')
