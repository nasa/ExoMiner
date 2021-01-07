"""
Testing transit model fit to the flux views
"""

from pytransit import QuadraticModel
import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt

#%%

tceOfInterest = (8937762, 1)

tfrecDir = '/data5/tess_project/Data/tfrecords/Kepler/Q1-Q17_DR25/tfrecordskeplerdr25_g2001-l201_spline_nongapped_flux-centroid-oddeven-wks-6stellar-bfap-ghost-rollingband_data/tfrecordskeplerdr25_g2001-l201_spline_nongapped_flux-centroid-oddeven-wks-6stellar-bfap-ghost-rollingband_starshuffle_experiment-labels-norm'
tfrecFiles = [os.path.join(tfrecDir, file) for file in os.listdir(tfrecDir) if 'shard' in file]

tceIdentifier = 'tce_plnt_num'
for tfrecFile in tfrecFiles:

    tfrecord_dataset = tf.data.TFRecordDataset(tfrecFile)

    for string_record in tfrecord_dataset.as_numpy_iterator():

        example = tf.train.Example()
        example.ParseFromString(string_record)

        tceIdentifierTfrec = example.features.feature[tceIdentifier].int64_list.value[0]
        targetIdTfrec = example.features.feature['target_id'].int64_list.value[0]

        durTfrec = example.features.feature['tce_duration'].float_list.value[0]
        perTfrec = example.features.feature['tce_period'].float_list.value[0]

        if targetIdTfrec != tceOfInterest[0] or tceIdentifierTfrec != tceOfInterest[1]:
            continue
        else:
            loc_flux_view = np.array(example.features.feature['local_flux_view'].float_list.value)
            glob_flux_view = np.array(example.features.feature['global_flux_view'].float_list.value)
            aa

#%%

# times = np.linspace(0, 2001, 2001, endpoint=False)
times = np.linspace(0, perTfrec, 2001, endpoint=False)

tm = QuadraticModel()
tm.set_data(times)

modelTransitParams = {
    'k': 0.0157,  # 0.09,
    'ldc': [0.4635112, 0.1161916],  # , 0.4499052, -0.2809696],  # [0.2, 0.1],
    't0': perTfrec / 2,
    'p': perTfrec,
    'a': 57.614511,
    'i': 89.38 * np.pi / 180,  # 89.38 * np.pi / 180,
    'e': 0,  # assumed to be zero by SPOC processing pipeline
    'w': 0  # assumed to be zero by SPOC processing pipeline
}

modelTransit = tm.evaluate(**modelTransitParams)
modelTransit -= 1

#%% fit model transit

phase = times - perTfrec / 2
f, ax = plt.subplots(2, 1, sharex=True, sharey=True)
ax[0].plot(phase, glob_flux_view)
ax[0].set_xlim([phase[0], phase[-1]])
ax[0].set_ylabel('Normalized flux')
ax[0].set_ylim(bottom=-1)
ax[1].plot(phase, modelTransit/np.abs(np.min(modelTransit)))
ax[1].set_xlim([phase[0], phase[-1]])
ax[1].set_ylim(bottom=-1)
ax[1].set_xlabel('Phase [day]')
ax[1].set_ylabel('Normalized flux')
f.suptitle('TCE {}-{}'.format(*tceOfInterest))
f.savefig('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Analysis/model_transit_fit/'
          'global_flux_view_TCE_{}-{}.png'.format(*tceOfInterest))

#%%

numPoints = 100
times = np.linspace(0, perTfrec, 2001, endpoint=False)
timesInterp = np.linspace(0, perTfrec, numPoints, endpoint=False)
glob_flux_view_interp = np.interp(timesInterp, times, glob_flux_view)

f, ax = plt.subplots()
ax.plot(times, glob_flux_view)
ax.plot(timesInterp, glob_flux_view_interp)
ax.scatter(timesInterp, glob_flux_view_interp, c='r')
ax.set_xlabel('Time [day]')
ax.set_ylabel('Normalized flux')

