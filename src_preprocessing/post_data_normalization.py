"""
Auxiliary script designed to extract statistics from the non-normalized preprocessed training set which are then used
to normalized the data.
"""

import numpy as np
import os
import matplotlib.pyplot as plt

from src.estimator_util import get_data_from_tfrecords

#%% Get extrema for the non-normalized centroid training set

tfrec_dir = '/home/msaragoc/Projects/Kepler-TESS_exoplanet/Data/tfrecords/' \
            'tfrecord_keplerdr25_centroidnonnormalized_radec_nonwhitened_gapped_2001-201'

tfrecs = [os.path.join(tfrec_dir, tfrec_filename) for tfrec_filename in os.listdir(tfrec_dir)
          if 'train-' in tfrec_filename]

data_fields = ['global_view_centr', 'local_view_centr', 'kepid', 'tce_n']

data = get_data_from_tfrecords(tfrecs, data_fields, label_map=None, filt=None, coupled=False)

data = {key: np.array(data[key]) for key in data}

# kepid, tce_n = 2437505, 1
# idx = np.intersect1d(np.where(data['kepid'] == kepid), np.where(data['tce_n'] == tce_n))
# plt.figure()
# plt.plot(data['local_view_centr'][idx[0]])

extrema_centr = {'global_view_centr': {'max': np.max(data['global_view_centr']),
                                       'min': np.min(data['global_view_centr'])},
                 'local_view_centr': {'max': np.max(data['local_view_centr']),
                                      'min': np.min(data['local_view_centr'])}}

print('Extrema: ', extrema_centr)
np.save('{}/extrema_centr_trainingset.npy'.format(tfrec_dir), extrema_centr)

#%% Get maximum, minimum and out-of-transit median and standard deviation for the non-normalized centroid training set

savedir = '/home/msaragoc/Downloads/centroid_preprocessing/data_normalization'

tfrec_dir = '/home/msaragoc/Projects/Kepler-TESS_exoplanet/Data/tfrecords/' \
            'tfrecordkeplerdr25_centroidnonnormalized_nonwhitened_gapped_2001-201'
tfrecs = [os.path.join(tfrec_dir, tfrec_filename) for tfrec_filename in os.listdir(tfrec_dir)
          if 'train-' in tfrec_filename]

data_fields = ['global_view_centr', 'local_view_centr', 'kepid', 'tce_n', 'label']
time_series_fields = ['global_view_centr', 'local_view_centr']
idxs_nontransitcadences = {field: [] for field in time_series_fields}
# stats_fields = ['max', 'min', 'median_oot', 'std_oot']
stats_fields = ['mean_oot', 'std_oot']
stats = {field: {stats_field: None for stats_field in stats_fields} for field in time_series_fields}

data = get_data_from_tfrecords(tfrecs, data_fields, label_map=None, filt=None, coupled=False)
data = {key: np.array(data[key]) for key in data}

nr_loc_bins = 201
nr_glob_bins = 2001

nr_transit_durations = 2 * 4 + 1
transit_duration_bins_loc = nr_loc_bins / nr_transit_durations
nontransitcadences_loc = np.array([True] * nr_loc_bins)
nontransitcadences_loc[np.arange(int(np.floor((nr_loc_bins - transit_duration_bins_loc) / 2)),
                                 int(np.ceil((nr_loc_bins + transit_duration_bins_loc) / 2)))] = False
idxs_nontransitcadences_loc = np.where(nontransitcadences_loc)
# bidx = np.where(~nontransitcadences_loc)[0][0] - 1
# aidx = np.where(~nontransitcadences_loc)[0][-1] + 1
idxs_nontransitcadences['local_view_centr'] = idxs_nontransitcadences_loc

glob_bidx = 749
glob_aidx = 1249
nontransitcadences_glob = np.array([True] * nr_glob_bins)
nontransitcadences_glob[glob_bidx:glob_aidx] = False
idxs_nontransitcadences_loc = np.where(nontransitcadences_glob)
idxs_nontransitcadences['global_view_centr'] = idxs_nontransitcadences_loc

b, e, s = 0, 1e2, 1e0
bins = None  # np.linspace(b, e, int((e - b) / s), endpoint=True)  # np.arange(0, 1e1, 1e-1)
cumulative = False
density = False
# plot_fields = {'global_view_centr': ''}
for field in time_series_fields:

    # data[field][np.where(data[field] > 30)] = 30
    valid_timeseries_idxs = np.where(np.all(data[field] <= 30, axis=1))
    print('Number of valid time-series: {} (out of {})'.format(len(valid_timeseries_idxs[0]), data[field].shape[0]))
    # data_flat = data[field][valid_timeseries_idxs].flatten()
    # data_oot_flat = data[field][:, idxs_nontransitcadences[field][0]].flatten()

    # q75, q25 = np.percentile(data_flat, [75, 25])
    # iqr = q75 - q25
    # data_flat[(data_flat <= q75 + 1.5 * iqr) & (data_flat >= q25 - 1.5 * iqr)].shape

    # f, ax = plt.subplots(1, 2)
    # ax[0].hist(data_flat, bins=bins, cumulative=True, density=True, edgecolor='k')
    # ax[0].set_title('All cadences')
    # ax[0].set_xlabel('Amplitude')
    # ax[0].set_ylabel('Counts')
    # ax[0].set_xlim(left=bins[0], right=bins[-1])
    # ax[1].hist(data_oot_flat, bins=bins, cumulative=True, density=True, edgecolor='k')
    # ax[1].set_title('Oot cadences')
    # ax[1].set_xlabel('Amplitude')
    # ax[1].set_xlim(left=bins[0], right=bins[-1])
    # f.suptitle(field)
    # f, ax = plt.subplots(figsize=(10, 8))
    # ax.hist(data_oot_flat, bins=bins, cumulative=cumulative, density=density, edgecolor='k')
    # ax.set_title('Out-of-transit cadences\n'
    #              '{}x{} time-series -> {} data points\n'
    #              '{} out-of-transit data points'.format(data[field].shape[0], data[field].shape[1], data_flat.shape[0],
    #                                                     data_oot_flat.shape[0]))
    # ax.set_xlabel('Amplitude')
    # ax.set_ylabel(('Counts', 'Density')[density])
    # ax.set_xlim(left=bins[0], right=bins[-1])
    # ax.set_yscale('log')
    # ax.set_xscale('log')
    # f.suptitle(field)
    # f.savefig('{}'.format(os.path.join(savedir, '{}_{}.png'.format(field, 'ootcadences'))))
    # plt.close()
    # aaa

    # f, ax = plt.subplots(2, 2, figsize=(12, 10))
    # ax[0, 0].hist(np.max(data[field], axis=1), bins=bins, cumulative=cumulative, density=density, edgecolor='k')
    # ax[0, 0].set_title('Max ({})'.format(np.max(data_flat)))
    # ax[0, 0].set_ylabel(('Counts', 'Density')[density])
    # ax[0, 0].set_xlim(left=bins[0], right=bins[-1])
    # ax[0, 0].set_yscale('log')
    # ax[0, 0].set_xscale('log')
    # ax[1, 0].hist(np.min(data[field], axis=1), bins=bins, cumulative=cumulative, density=density, edgecolor='k')
    # ax[1, 0].set_ylabel(('Counts', 'Density')[density])
    # ax[1, 0].set_xlabel('Amplitude')
    # ax[1, 0].set_xlim(left=bins[0], right=bins[-1])
    # ax[1, 0].set_title('Min ({})'.format(np.min(data_flat)))
    # ax[1, 0].set_yscale('log')
    # ax[1, 0].set_xscale('log')
    # ax[0, 1].hist(np.squeeze(np.std(data[field][:, idxs_nontransitcadences[field][0]], axis=1)), bins=bins,
    #               cumulative=cumulative, density=density, edgecolor='k')
    # ax[0, 1].set_xlim(left=bins[0], right=bins[-1])
    # ax[0, 1].set_title('Out-of-transit Median ({})'.format(np.median(data_oot_flat)))
    # ax[0, 1].set_yscale('log')
    # ax[0, 1].set_xscale('log')
    # ax[1, 1].hist(np.median(data[field][:, idxs_nontransitcadences[field][0]], axis=1), bins=bins,
    #               cumulative=cumulative, density=density, edgecolor='k')
    # ax[1, 1].set_xlabel('Amplitude')
    # ax[1, 1].set_xlim(left=bins[0], right=bins[-1])
    # ax[1, 1].set_title('Out-of-transit Std ({})'.format(np.std(data_oot_flat)))
    # ax[1, 1].set_yscale('log')
    # ax[1, 1].set_xscale('log')
    # f.suptitle('{}\nStatistics per time-series (training set value)'.format(field))
    # f.savefig('{}'.format(os.path.join(savedir, '{}_{}.png'.format(field, 'minmaxmedstd'))))
    # plt.close()

    # stats[field]['max'] = np.max(data[field])
    # stats[field]['min'] = np.min(data[field])
    stats[field]['std_oot'] = np.std(data[field][valid_timeseries_idxs, idxs_nontransitcadences[field]])
    # stats[field]['median_oot'] = np.median(data[field][:, idxs_nontransitcadences[field]])
    stats[field]['mean_oot'] = np.mean(data[field][valid_timeseries_idxs, idxs_nontransitcadences[field]])

print('Computed stats: ', stats)
# print('Saving stats...')
# np.save('{}/stats_trainingset.npy'.format(tfrec_dir), stats)
