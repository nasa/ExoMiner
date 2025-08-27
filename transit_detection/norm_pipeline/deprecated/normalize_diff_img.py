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


def perimage_normalization_diffimg(example, zero_division_eps, features):
    """ Perform per-image normalization of difference image features for a single example in a TFRecord data set.

    Args:
        example: TFRecord example with difference image features to be normalized.
        zero_division_eps: float, term added to denominator to avoid division by zero in normalization processes
        # q_oot: float, quantile used to fill missing values in out-of-transit images
        features: dict, key is feature name of difference image feature in the source data that is to be normalized;
        value is a tuple with the dimensions of said feature.

    Returns: example, with added normalized difference image features

    """

    for feature_name, feature_dims in features.items():  # iterate through features to be normalized
        # extract feature of interest
        feature_data_example = tf.reshape(
            tf.io.parse_tensor(example.features.feature[feature_name].bytes_list.value[0], tf.float32),
            feature_dims).numpy()

        # min-max normalization
        feature_data_example_minmaxn = np.array(feature_data_example)
        feature_min_per_img = np.nanmin(feature_data_example_minmaxn, axis=(1, 2), keepdims=True)
        feature_max_per_img = np.nanmax(feature_data_example_minmaxn, axis=(1, 2), keepdims=True)

        # perform min-max normalization
        # img_n = (img - min(img)) / (max(img) - min(img) + zero_division_eps)
        feature_data_example_minmaxn = ((feature_data_example_minmaxn - feature_min_per_img) /
                                        (feature_max_per_img - feature_min_per_img + zero_division_eps))

        # add normalized feature to example
        example_util.set_tensor_feature(example, f'{feature_name}_minmaxnorm', feature_data_example_minmaxn)

        # standardization
        feature_data_example_std = np.array(feature_data_example)
        feature_median_per_img = np.nanmedian(feature_data_example_std, axis=(1, 2), keepdims=True)
        feature_madstd_per_img = \
            np.expand_dims(mad_std(feature_data_example_std, axis=(1, 2), ignore_nan=True), axis=(1, 2))

        # perform standardization
        # img_n = (img - med(img)) / (mad_std(img) + zero_division_eps)
        feature_data_example_std = ((feature_data_example_std - feature_median_per_img) /
                                    (feature_madstd_per_img + zero_division_eps))

        # add normalized feature to example
        example_util.set_tensor_feature(example, f'{feature_name}_stdnorm', feature_data_example_std)

    return example


def preprocess_tfrecs_perimage_normalization_diffimg(src_tfrec_fps, dest_tfrec_dir, zero_division_eps, features):
    """ Perform per-image normalization of difference image features for a set of TFRecord files. The TFRecord files
    with the normalized data will be under the directory defined by `dest_tfrec_dir`. The files have the same name as
    the source ones.

    Args:
        src_tfrec_fps: list of Paths, source TFRecord files to be preprocessed (i.e., for which the difference image
        data is going to be normalized and create new TFRecord files
        dest_tfrec_dir: Path, destination directory for the preprocessed data
        zero_division_eps: float, term added to denominator to avoid division by zero in normalization processes
        # q_oot: float, quantile used to fill missing values in out-of-transit images
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

                example = perimage_normalization_diffimg(example, zero_division_eps, features)

                # write example with added normalized features to new TFRecord file
                writer.write(example.SerializeToString())


if __name__ == '__main__':

    src_tfrec_dir = Path('/home6/msaragoc/work_dir/Kepler-TESS_exoplanet/data/tfrecords/TESS/tfrecords_tess_sgdetrending_s1-s67_3-27-2024_1127_merged_adddiffimg')
    dest_tfrec_dir = src_tfrec_dir.parent / f'{src_tfrec_dir.name}_perimgnormdiffimg'
    zero_division_eps = 1e-12  # term added to denominator to avoid division by zero

    features = {
        'diff_imgs': (33, 33),  # feature dimension
        'oot_imgs': (33, 33),
        'snr_imgs' : (33, 33)
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
    jobs = [(src_tfrec_fps_job, dest_tfrec_dir, zero_division_eps, features)
            for src_tfrec_fps_job in src_tfrec_fps_jobs]
    async_results = [pool.apply_async(preprocess_tfrecs_perimage_normalization_diffimg, job) for job in jobs]
    pool.close()

    for async_result in async_results:
        async_result.get()

    print('Normalization finished.')