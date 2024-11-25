"""
Normalization pipeline for flux windows and img features (diff_img, oot_img, snr_img) from TESS transit_detection dataset

Normalize flux window data in TFRecords using per-window statistics (i.e., no statistics computed based
on a training set.)

Normalize img feature data in TFRecords using per dataset statistics (i.e statistics computed based on a 
training set)

Input: TFRecord data set with flux window and difference image data.
Output: new TFRecord data set with per-image normalized flux window and per dataset difference image data.
"""

# 3rd party
import tensorflow as tf
from pathlib import Path
import numpy as np

# local
from src_preprocessing.tf_util import example_util


def normalize_flux(flux_window, zero_division_eps):
    """
    Normalize a flux window based on per-window statistics

    Input: flux_window : List[float]
           zero_division_eps : small float to prevent division by zero

    Output: flux_window_norm : List[float]
    """
    #vectorized
    #load as np array
    flux_window = np.array(flux_window)
    median = np.median(flux_window)
    minimum = np.min(flux_window)

    norm_flux_window = (flux_window - median) / (minimum + zero_division_eps)

    return norm_flux_window.tolist()


def normalize_img_feature(img_feature, img_set_med, img_set_std, zero_division_eps):
    """
    Normalize an img feature based on per-set statistics (diff_img, oot_img, snr_img)

    Input: img_feature : np.array of shape (X, X)
           img_set_med : median computed for img feature based on set (i.e. training set)
           img_set_std : std computed for img feature based on set (i.e. training set)
           zero_division_eps : small float to prevent division by zero

    Output: norm_img_feature: normalized np.array of shape (X, X)
    """
    #X_n = [x - median(X_train) ] / [std(X_train) + eps]
    norm_img_feature = (img_feature - img_set_med) / (img_set_std + zero_division_eps)

    return norm_img_feature


if __name__ == "__main__":
    src_tfrec_dir = Path('/nobackup/jochoa4/work_dir/data/datasets/TESS_exoplanet_dataset_11-12-2024/tfrecords')
    dest_tfrec_dir = src_tfrec_dir.parent / f'{src_tfrec_dir.name}_perset_normdiffimg_perimg_normfluxwindow'
    zero_division_eps = 1e-12  # term added to denominator to avoid division by zero

    src_tfrec_fp = Path('/nobackup/jochoa4/work_dir/data/datasets/TESS_exoplanet_dataset_11-12-2024/tfrecords_split')

    norm_stats_dir = Path('/nobackup/jochoa4/work_dir/data/stats/TESS_exoplanet_dataset_11-12-2024_stats')

    print(f'Iterating over source file: {src_tfrec_fp.name}')

    flux_window_set_vals = []


    
    for set_type in ['training', 'validation', 'test']:
        dest_tfrec_fp = src_tfrec_dir.parent / f'{src_tfrec_dir.name}_perset_normdiffimg_perimg_normfluxwindow' / 'tfrecords' / set_type + '_set'

        with tf.io.TFRecordWriter(path=str(dest_tfrec_fp)) as writer:

            # Load source dataset
            src_tfrecord_dataset = tf.data.TFRecordDataset(str(src_tfrec_fp / set_type + '_set'))

            for string_record in src_tfrecord_dataset.as_numpy_iterator():

                example = tf.train.Example()

                example.ParseFromString(string_record)

                # normalize flux window
                example_flux = example.features.feature['flux'].float_list.value
                example_norm_flux = normalize_flux(example_flux, zero_division_eps)

                example_util.set_float_feature(ex=example, name='flux_norm', value=example_norm_flux, allow_overwrite=False)

                # normalize diff img, oot_img, snr_img
                img_dims = (33, 33)

                for img_feature in ['diff_img', 'oot_img', 'snr_img']:
                    # get median/std etc from somewhere for training / val set
                    norm_stats = np.load(norm_stats_dir / f"{img_feature}_stats.npy")
                    img_feature_set_med, img_feature_set_std = norm_stats['median'], norm_stats['std']

                    example_img_feature = tf.reshape(tf.io.parse_tensor(example.features.feature[img_feature].bytes_list.value[0], tf.float32),
                                                        img_dims).numpy()
                    example_norm_img_feature = normalize_img_feature(example_img_feature, img_feature_set_med, img_feature_set_std, zero_division_eps)

                    example_util.set_tensor_feature(ex=example, name=f"{img_feature}_stdnorm", value=example_norm_img_feature,
                                                allow_overwrite=False)

                # write to new TFRecord
                writer.write(example.SerializeToString())



