import os
import tensorflow as tf
tf.enable_eager_execution()
import numpy as np
# import matplotlib; matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt
import itertools

from src_preprocessing.preprocess import normalize_view
from src.estimator_util import get_data_from_tfrecords, get_data_from_tfrecord

#%%

tfrec_dir = '/home/msaragoc/Projects/Kepler-TESS_exoplanet/Kepler_planet_finder/src_preprocessing/tfrecords/' \
            'tfrecord_dr25_manual_2d_few180k_keplernonwhitened'
tfrecs = [os.path.join(tfrec_dir, tfrec_filename) for tfrec_filename in os.listdir(tfrec_dir)]

features_set = {'global_view': {'dim': 2001, 'dtype': tf.float32},
                                 'local_view': {'dim': 201, 'dtype': tf.float32}}
data_fields = {feature_name: tf.FixedLenFeature([feature_info['dim']], feature_info['dtype'])
               for feature_name, feature_info in features_set.items()}

for tfrec_filepath in tfrecs:
    tf_dataset = tf.data.TFRecordDataset([tfrec_filepath])

    for record in tf_dataset:
        print('########33')
        print(repr(record))
        tf.parse_single_example(record, features=data_fields)
        aa
#
# for raw_record in tf_dataset.take(100):
#   print(repr(raw_record))

# writing a TFRecord file
for tfrec_filepath in tfrecs:
    with tf.python_io.TFRecordWriter(tfrec_filepath) as writer:
        for i in range(n_observations):
            example = serialize_example(feature0[i], feature1[i], feature2[i], feature3[i])
            writer.write(example)

# reading a TFRecord file
for tfrec_filepath in tfrecs:
    record_iterator = tf.python_io.tf_record_iterator(path=tfrec_filepath)

    for string_record in record_iterator:
      example = tf.train.Example()
      example.ParseFromString(string_record)

      print(example)

#%% Get maximum value of the non-normalized centroid

tfrec_dir = '/home/msaragoc/Projects/Kepler-TESS_exoplanet/Kepler_planet_finder/src_preprocessing/tfrecords/' \
            'tfrecordkepler_dr25_centroidnonnormalizednonwhitened_gapped'
tfrecs = [os.path.join(tfrec_dir, tfrec_filename) for tfrec_filename in os.listdir(tfrec_dir)]

data_fields = ['global_view_centr', 'local_view_centr']

data = get_data_from_tfrecords(tfrecs, data_fields, label_map=None, filt=None, coupled=False)

data = {key: np.array(data[key]) for key in data}

max_gvcentr = np.max(data['global_view_centr'])
max_lvcentr = np.max(data['local_view_centr'])

max_centr = {'global_view': max_gvcentr, 'local_view': max_lvcentr}

np.save('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Kepler_planet_finder/src_preprocessing/'
        'max_centr_trainingset.npy', max_centr)



#%% Get std and median from the training set for the centroid time series and median for the flux for the global and
# local views

tfrec_dir = '/home/msaragoc/Projects/Kepler-TESS_exoplanet/Kepler_planet_finder/src_preprocessing/tfrecords/' \
            'tfrecordkepler_dr25_flux_centroidnonnormalized_nonwhitened_gapped_2001-201'
tfrecs = [os.path.join(tfrec_dir, tfrec_filename) for tfrec_filename in os.listdir(tfrec_dir) if 'train-' in
          tfrec_filename]

nr_loc_bins = 201
nr_glob_bins = 2001
nr_transit_durations = 2 * 4 + 1
transit_duration_bins_loc = nr_loc_bins / nr_transit_durations
nontransitcadences_loc = np.array([True] * 201)
nontransitcadences_loc[np.arange(int(np.floor((nr_loc_bins - transit_duration_bins_loc) / 2)),
                                 int(np.ceil((nr_loc_bins + transit_duration_bins_loc) / 2)))] = False
bidx = np.where(~nontransitcadences_loc)[0][0] - 1
aidx = np.where(~nontransitcadences_loc)[0][-1] + 1
glob_bidx = 749
glob_aidx = 1249

time_series_fields = ['global_view', 'local_view', 'global_view_centr', 'local_view_centr']
stellar_fields = ['kMag', 'sTEff', 'sLogG', 'sMet', 'sRad', 'sMass', 'sDens']
data_fields = time_series_fields + stellar_fields
# data_fields = ['local_view_centr']

# ['global_view_med', 'global_view_std',
#                                    'local_view_std', 'local_view_med',
#                                    'global_view_centr_med', 'global_view_centr_std',
#                                    'local_view_centr_med', 'local_view_centr_std',
#                                    'global_view_centr_max', 'global_view_centr_min',
#                                    'local_view_centr_max', 'local_view_centr_min',
#                                    ]
stats_vec = {key: None for key in [''.join(el)
                                   for el in itertools.product(time_series_fields, ['_std', '_med' + '_min', '_max'])]
             + [''.join(el)  for el in itertools.product(stellar_fields, ['_std', '_med'])]}
# stats_vec = {key: None for key in ['local_view_centr_med', 'local_view_centr_std',
#                                    'local_view_centr_max', 'local_view_centr_min']}

# data = {field: [] for field in data_fields}
data = {field: np.array([], dtype='float64') for field in data_fields}
oot_cadences = {'global_view': np.array([], dtype='bool'), 'local_view': np.array([], dtype='bool')}
for i_tfrec, tfrec in enumerate(tfrecs):

    print('Getting data from tfrecord {}/{} ({})...'.format(i_tfrec + 1, len(tfrec), tfrec))

    # data_aux = get_data_from_tfrecord(tfrec, data_fields + ['tce_period', 'tce_duration'], label_map=None, filt=None,
    #                                   coupled=False)
    data_aux = get_data_from_tfrecord(tfrec, data_fields, label_map=None, filt=None, coupled=False)

    # get only oot values for flux and centroid
    for i_tce in range(len(data_aux[data_fields[0]])):

        # normalize the flux views
        if 'global_view' in data_fields:
            if i_tce == 0:
                print('Normalizing flux global view...')
            data_aux['global_view'][i_tce] /= max(np.abs(min(data_aux['global_view'][i_tce])), 1e-11)

        if 'local_view' in data_fields:
            if i_tce == 0:
                print('Normalizing flux local view...')
            try:
                data_aux['local_view'][i_tce] /= max(np.abs(min(data_aux['local_view'][i_tce])), 1e-11)
            except Exception as e:
                print(e)
                data_aux['local_view'][i_tce] = np.array(data_aux['local_view'][i_tce]) / \
                                                max(np.abs(min(data_aux['local_view'][i_tce])), 1e-11)

            # # center the centroid views
            # if 'global_view_centr' in data_fields:
            #     if i_tce == 0:
            #         print('Centering centroid global view...')
            #     data_aux['global_view_centr'][i_tce] -= np.median(data_aux['global_view_centr'][i_tce])
            #
            # if 'local_view_centr' in data_fields:
            #     if i_tce == 0:
            #         print('Centering centroid local view...')
            #     try:
            #         data_aux['local_view_centr'][i_tce] -= np.median(data_aux['local_view_centr'][i_tce])
            #     except Exception as e:
            #         print(e)
            #         data_aux['local_view_centr'][i_tce] = np.array(data_aux['local_view_centr'][i_tce]) - \
            #                                               np.median(data_aux['local_view_centr'][i_tce])

        # plot data
        if np.random.random() < -1:  # 0.01:
            f, ax = plt.subplots(2, 2, figsize=(10, 6))

            ax[0, 0].set_title('Local view')
            ax[0, 0].plot(data_aux['local_view'][i_tce])
            # ax[0].axvline(x=bidx, ymin=min(data_aux['local_view'][i_tce]),
            #               ymax=max(data_aux['local_view'][i_tce]), c='r')
            ax[0, 0].axvline(x=bidx, ymin=-10, ymax=10, c='r')
            # ax[0].axvline(x=aidx, ymin=min(data_aux['local_view'][i_tce]),
            #               ymax=max(data_aux['local_view'][i_tce]), c='r')
            ax[0, 0].axvline(x=aidx, ymin=-10, ymax=10, c='r')
            ax[0, 0].set_ylabel('Amplitude')
            # ax[0].set_xlabel('Bin number')
            ax[1, 0].plot(data_aux['local_view_centr'][i_tce])
            # ax[1].axvline(x=bidx, ymin=min(data_aux['local_view_centr'][i_tce]),
            #               ymax=max(data_aux['local_view_centr'][i_tce]), c='r')
            ax[1, 0].axvline(x=bidx, ymin=-10, ymax=10, c='r')
            # ax[1].axvline(x=bidx, ymin=min(data_aux['local_view_centr'][i_tce]),
            #               ymax=max(data_aux['local_view_centr'][i_tce]), c='r')
            ax[1, 0].axvline(x=aidx, ymin=-10, ymax=10, c='r')
            ax[1, 0].set_xlabel('Bin number')
            ax[1, 0].set_ylabel('Amplitude')

            ax[0, 1].set_title('Global view')
            ax[0, 1].plot(data_aux['global_view'][i_tce])
            ax[0, 1].axvline(x=glob_bidx, ymin=-10, ymax=10, c='r')
            ax[0, 1].axvline(x=glob_aidx, ymin=-10, ymax=10, c='r')
            ax[1, 1].plot(data_aux['global_view_centr'][i_tce])
            ax[1, 1].axvline(x=glob_bidx, ymin=-10, ymax=10, c='r')
            ax[1, 1].axvline(x=glob_aidx, ymin=-10, ymax=10, c='r')
            ax[1, 1].set_xlabel('Bin number')

            plt.subplots_adjust(top=0.88, bottom=0.11, left=0.135, right=0.9, hspace=0.2, wspace=0.335)
            f.savefig('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Kepler_planet_finder/visualize_inputs/'
                      'non-normalized_fluxandcentroid/local_view_interval/tfrec{}_tce{}.svg'.format(i_tfrec, i_tce))
            # aaaa
            plt.close()

        # local view
        # data_aux['local_view'][i_tce] = np.array(data_aux['local_view'][i_tce])[nontransitcadences_loc]
        # data_aux['local_view_centr'][i_tce] = np.array(data_aux['local_view_centr'][i_tce])[nontransitcadences_loc]
        data_aux['local_view'][i_tce] = np.array(data_aux['local_view'][i_tce])
        data_aux['local_view_centr'][i_tce] = np.array(data_aux['local_view_centr'][i_tce])

        # global view
        # transit_duration_bins_glob = data_aux['tce_duration'][i_tce] / data_aux['tce_period'][i_tce] * nr_glob_bins
        #
        nontransitcadences_glob = np.array([True] * nr_glob_bins)
        # nontransitcadences_glob[np.arange(int(np.floor((nr_glob_bins - transit_duration_bins_glob) / 2)),
        #                                  int(np.ceil((nr_glob_bins + transit_duration_bins_glob) / 2)))] = False
        #
        nontransitcadences_glob[np.arange(glob_bidx, glob_aidx + 1)] = False
        # data_aux['global_view'][i_tce] = np.array(data_aux['global_view'][i_tce])[nontransitcadences_glob]
        # data_aux['global_view_centr'][i_tce] = np.array(data_aux['global_view_centr'][i_tce])[nontransitcadences_glob]
        data_aux['global_view'][i_tce] = np.array(data_aux['global_view'][i_tce])
        data_aux['global_view_centr'][i_tce] = np.array(data_aux['global_view_centr'][i_tce])

        # data_aux['global_view'][i_tce] = np.array(data_aux['global_view'][i_tce])
        # data_aux['global_view_centr'][i_tce] = np.array(data_aux['global_view_centr'][i_tce])

        oot_cadences['global_view'] = np.concatenate((oot_cadences['global_view'],
                                                  nontransitcadences_glob))
        oot_cadences['local_view'] = np.concatenate((oot_cadences['local_view'],
                                                 nontransitcadences_loc))

    for field in data_fields:
        # data[field].extend(list(np.concatenate(data_aux[field])))
        data[field] = np.concatenate((data[field], np.concatenate(data_aux[field])))
        del data_aux[field]

for stat_el in stats_vec:
    print(stat_el)
    if 'std' in stat_el:
        # oot dataset std
        stats_vec[stat_el] = np.std(data['_'.join(stat_el.split('_')[:-1])][
                                        oot_cadences[('local_view', 'global_view')['global_view' in stat_el]]])
    elif 'med' in stat_el:
        # oot dataset median
        stats_vec[stat_el] = np.median(data['_'.join(stat_el.split('_')[:-1])][
                                           oot_cadences[('local_view', 'global_view')['global_view' in stat_el]]])
    elif 'min' in stat_el:
        # dataset min
        stats_vec[stat_el] = np.min(data['_'.join(stat_el.split('_')[:-1])])
    elif 'max' in stat_el:
        # dataset max
        stats_vec[stat_el] = np.max(data['_'.join(stat_el.split('_')[:-1])])

print('Computed stats: ', stats_vec)
print('Saving stats...')
np.save('{}/stats_trainingset.npy'.format(tfrec_dir), stats_vec)
