"""
Script to visualize the output from applying online data augmentation techniques
"""

import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import os

from src.utils_train import add_whitegaussiannoise, phase_inversion, phase_shift
from src.utils_dataio import get_out_of_transit_idxs_loc, get_out_of_transit_idxs_glob

#%%

saveDir = '/home/msaragoc/Projects/Kepler-TESS_exoplanet/Analysis/online_data_augmentation'

tfrecord = '/data5/tess_project/Data/tfrecords/Kepler/Q1-Q17_DR25/tfrecordskeplerdr25_g2001-l201_spline_nongapped_flux-centroid_selfnormalized-oddeven-wks-scalar_data/tfrecordskeplerdr25_g2001-l201_spline_nongapped_flux-centroid_selfnormalized-oddeven-wks-scalar_centrmedcmaxncorr_starshuffle_experiment-labels-norm/val-shard-00015-of-00030'

tfrecord_dataset = tf.data.TFRecordDataset(tfrecord)

for string_record in tfrecord_dataset.as_numpy_iterator():

    example = tf.train.Example()
    example.ParseFromString(string_record)

    features_dict = {'tce_period': tf.io.FixedLenFeature([], tf.float32),
                     'tce_duration': tf.io.FixedLenFeature([], tf.float32),
                     'local_flux_view': tf.io.FixedLenFeature((201, 1), tf.float32),
                     'target_id': tf.io.FixedLenFeature([], tf.int64),
                     'tce_plnt_num': tf.io.FixedLenFeature([], tf.int64),
                     'label': tf.io.FixedLenFeature([], tf.string)
                     }

    tensor_dict = tf.io.parse_example(string_record, features=features_dict)

    should_reverse = tf.less(x=tf.random.uniform([], minval=0, maxval=1), y=0.5, name="should_reverse")
    # bin shifting
    bin_shift = [-5, 5]
    shift = tf.random.uniform(shape=(), minval=bin_shift[0], maxval=bin_shift[1],
                              dtype=tf.dtypes.int32, name='randuniform')

    # get oot indices for Gaussian noise augmentation added to oot indices
    # boolean tensor for oot indices for global view
    idxs_nontransitcadences_glob = get_out_of_transit_idxs_glob(2001,
                                                                tensor_dict['tce_duration'],
                                                                tensor_dict['tce_period'])
    # boolean tensor for oot indices for local view
    idxs_nontransitcadences_loc = get_out_of_transit_idxs_loc(201, 9)

    view = tensor_dict['local_flux_view']
    original_view = tensor_dict['local_flux_view']

    # add Gaussian noise using oot statistics
    oot_values = tf.boolean_mask(view, idxs_nontransitcadences_loc)
    oot_median = tfp.stats.percentile(oot_values, 50, axis=0, name='oot_median')
    oot_std = tf.math.reduce_std(oot_values, axis=0, name='oot_std')
    view = add_whitegaussiannoise(view, oot_median, oot_std)

    # invert phase
    # view = phase_inversion(view, should_reverse)

    # phase shift some bins
    # view = phase_shift(view, shift)

    # plot view after data augmentation
    f, ax = plt.subplots()
    ax.plot(original_view.numpy(), label='original')
    ax.plot(view.numpy(), label='augmented')
    ax.legend()
    ax.set_xlabel('Phase')
    ax.set_ylabel('Amplitude')
    ax.set_title('{}-{} {}'.format(tensor_dict['target_id'], tensor_dict['tce_plnt_num'], tensor_dict['label']))
    f.savefig(os.path.join(saveDir, '{}.png'.format('addootgaussiannoise')))

    aaa
