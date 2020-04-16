import os
import tensorflow as tf
tf.enable_eager_execution()
import numpy as np
# import matplotlib; matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt
import itertools

from src_preprocessing.preprocess import normalize_view
from src.estimator_util import get_data_from_tfrecords, get_data_from_tfrecord


def gap_other_tces(all_time, all_flux, all_centroids, tce, table, config, conf_dict, gap_pad=0):
    """ Remove from the time series the cadences that belong to other TCEs in the light curve. These values are set to
    NaN. Old version that all kept data for the imputed cadences.

    :param all_time: list of numpy arrays, cadences
    :param all_flux: list of numpy arrays, flux time series
    :param all_centroids: list of numpy arrays, centroid time series
    :param tce: row of pandas DataFrame, main TCE ephemeris
    :param table: pandas DataFrame, TCE ephemeris table
    :param config: Config object, preprocessing parameters
    :param conf_dict: dict, keys are a tuple (Kepler ID, TCE planet number) and the values are the confidence level used
     when gapping (between 0 and 1)
    :param gap_pad: extra pad on both sides of the gapped TCE transit duration
    :return:
        all_time: list of numpy arrays, cadences
        all_flux: list of numpy arrays, flux time series
        all_centroids: list of numpy arrays, flux centroid time series
        imputed_time: list of numpy arrays,
    """

    # get gapped TCEs ephemeris
    if config.satellite == 'kepler':
        gap_ephems = table.loc[(table['kepid'] == tce.kepid) &
                               (table['tce_plnt_num'] != tce.tce_plnt_num)][['tce_period', 'tce_duration',
                                                                             'tce_time0bk']]
    else:
        # raise NotImplementedError('Gapping is still not implemented for TESS.')

        gap_ephems = table.loc[(table['tic'] == tce.tic) &
                                  (table['sector'] == tce.sector) &
                                  (table['tce_plnt_num'] != tce.tce_plnt_num)]

    # rename column names to same as Kepler - it would be easier if the uniformization was done in the TCE tables
    if config.satellite == 'tess':
        gap_ephems = gap_ephems.rename(columns={'transitDurationHours': 'tce_duration', 'orbitalPeriodDays':
            'tce_period', 'transitEpochBtjd': 'tce_time0bk'})

    # if gapping with confidence level, remove those gapped TCEs that are not in the confidence dict
    if config.gap_with_confidence_level:
        poplist = []
        for index, gapped_tce in gap_ephems.iterrows():
            if (tce.kepid, gapped_tce.tce_plnt_num) not in conf_dict or \
                    conf_dict[(tce.kepid, gapped_tce.tce_plnt_num)] < config.gap_confidence_level:
                poplist += [gapped_tce.tce_plnt_num]

        gap_ephems = gap_ephems.loc[gap_ephems['tce_plnt_num'].isin(poplist)]

    imputed_time = [] if config.gap_imputed else None

    begin_time, end_time = all_time[0][0], all_time[-1][-1]

    # find gapped cadences for each TCE
    for ephem_i, ephem in gap_ephems.iterrows():

        if ephem['tce_time0bk'] < begin_time:  # when the epoch of the gapped TCE occurs before the first cadence
            ephem['tce_time0bk'] = ephem['tce_time0bk'] + \
                                   ephem['tce_period'] * np.ceil((begin_time - ephem['tce_time0bk']) /
                                                                 ephem['tce_period'])
        else:
            ephem['tce_time0bk'] = ephem['tce_time0bk'] - ephem['tce_period'] * np.floor(
                (ephem['tce_time0bk'] - begin_time) / ephem['tce_period'])

        ephem['tce_duration'] = ephem['tce_duration'] * (1 + 2 * gap_pad)

        if ephem['tce_time0bk'] <= end_time:
            midTransitTimes = np.arange(ephem['tce_time0bk'], end_time, ephem['tce_period'])
            midTransitTimeBefore = midTransitTimes[0] - ephem['tce_period']
            midTransitTimeAfter = midTransitTimes[-1] + ephem['tce_period']
        else:
            midTransitTimes = []
            midTransitTimeBefore = ephem['tce_time0bk'] - ephem['tce_period']
            midTransitTimeAfter = ephem['tce_time0bk']

        extendedMidTransitTimes = np.concatenate([[midTransitTimeBefore], midTransitTimes, [midTransitTimeAfter]])

        startTransitTimes = (extendedMidTransitTimes - 0.5 * ephem['tce_duration'])
        endTransitTimes = (extendedMidTransitTimes + 0.5 * ephem['tce_duration'])
        nTransits = len(startTransitTimes)

        for quarter_i, time_i in enumerate(all_time):
            transit_boolean = np.full(time_i.shape[0], False)

            index = 0
            for i in range(time_i.shape[0]):
                for j in [index, index + 1]:
                    if startTransitTimes[j] <= time_i[i] <= endTransitTimes[j]:
                        transit_boolean[i] = True
                        if j > index:
                            index += 1
                        break
                if index > nTransits - 1:
                    break

            if config.gap_imputed and np.any(transit_boolean):  # if gaps need to be imputed and True in transit_boolean
                imputed_time += [[all_time[quarter_i][transit_boolean], all_flux[quarter_i][transit_boolean],
                                  {all_centroids[coord][quarter_i][transit_boolean] for coord in all_centroids}]]

            all_flux[quarter_i][transit_boolean] = np.nan
            all_centroids['x'][quarter_i][transit_boolean] = np.nan
            all_centroids['y'][quarter_i][transit_boolean] = np.nan

    return all_time, all_flux, all_centroids, imputed_time


def get_gap_ephems_for_DR25readouttbl(table, tce):
    """ Get ephemeris for the gapped TCEs.

    :param table: pandas DataFrame, TCE ephemeris table
    :param tce: row of pandas DataFrame, TCE of interest
    :return:
        gap_ephems: dict, each item contains ephemeris information for a gapped TCE
    """

    ephem_keys = {'epoch': 'tce_time0bk', 'period': 'tce_period', 'duration': 'tce_duration'}

    # initialize empty dictionary for gapped TCEs
    gap_ephems = {}

    # FIXME: is it already in days?
    d_factor = 1.0 / 24.0  # if satellite_id == 'kepler' else 1.0  # for tess, duration is already in [day] units

    # search for TCEs belonging to the same Kepler ID but with a different TCE planet number
    for tce_i, tce_i_ephem in table[tce.kepid].items():
        if tce.tce_plnt_num != tce_i:  # if it is not the TCE of interest
            # gap_ephems[len(gap_ephems)] = {'epoch': tce_i_ephem['epoch_corr'],
            #                                'period': tce_i_ephem['period'],
            #                                'duration': tce_i_ephem['duration'] * d_factor,
            #                                'tce_n': tce_i}
            gap_ephems[len(gap_ephems)] = {'epoch': tce_i_ephem[ephem_keys['epoch']],
                                           'period': tce_i_ephem[ephem_keys['period']],
                                           'duration': tce_i_ephem[ephem_keys['duration']] * d_factor,
                                           'tce_n': tce_i}

    return gap_ephems


def get_centr_oot_rms(all_centroids, all_time, tce, table, config):
    gap_pad = 0

    gap_ephems = {}
    d_factor = 1.0 / 24.0 if config.satellite == 'kepler' else 1.0  # for tess, duration is already in [day] units

    for tce_i, tce_i_ephem in table[tce.kepid].items():
        gap_ephems[len(gap_ephems)] = {'epoch': tce_i_ephem['epoch_corr'],
                                       'period': tce_i_ephem['period'],
                                       'duration': tce_i_ephem['duration'] * d_factor,
                                       'tce_n': tce_i}

    begin_time, end_time = all_time[0][0], all_time[-1][-1]
    for ephem in gap_ephems.values():
        if ephem['epoch'] < begin_time:
            ephem['epoch'] = ephem['epoch'] + ephem['period'] * np.ceil((begin_time - ephem['epoch']) / ephem['period'])
        else:
            ephem['epoch'] = ephem['epoch'] - ephem['period'] * np.floor(
                (ephem['epoch'] - begin_time) / ephem['period'])
        ephem['duration'] = ephem['duration'] * (1 + 2 * gap_pad)

        if ephem['epoch'] <= end_time:
            midTransitTimes = np.arange(ephem['epoch'], end_time, ephem['period'])
            midTransitTimeBefore = midTransitTimes[0] - ephem['period']
            midTransitTimeAfter = midTransitTimes[-1] + ephem['period']
        else:
            midTransitTimes = []
            midTransitTimeBefore = ephem['epoch'] - ephem['period']
            midTransitTimeAfter = ephem['epoch']

        extendedMidTransitTimes = np.concatenate([[midTransitTimeBefore], midTransitTimes, [midTransitTimeAfter]])

        startTransitTimes = (extendedMidTransitTimes - 0.5 * ephem['duration'])
        endTransitTimes = (extendedMidTransitTimes + 0.5 * ephem['duration'])
        nTransits = len(startTransitTimes)

        for quarter_i, time_i in enumerate(all_time):
            transit_boolean = np.full(time_i.shape[0], False)

            index = 0
            for i in range(time_i.shape[0]):
                for j in [index, index + 1]:
                    if startTransitTimes[j] <= time_i[i] <= endTransitTimes[j]:
                        transit_boolean[i] = True
                        if j > index:
                            index += 1
                        break
                if index > nTransits - 1:
                    break

            all_centroids['x'][quarter_i][transit_boolean] = np.nan
            all_centroids['y'][quarter_i][transit_boolean] = np.nan

    all_centroids_2 = {}
    for dim, array in all_centroids.items():
        all_centroids_2[dim] = np.concatenate(array)

    def _get_rms(array):
        return np.sqrt(np.square(np.nanmean(array)) + np.nanvar(array))

    return {key: _get_rms(centr_array) for key, centr_array in all_centroids_2.items()}


def transit_points(all_time, tce):
    gap_pad = 0
    d_factor = 1.0 / 24.0
    ephem = {}
    # for tce_i, tce_i_ephem in table[tce.kepid].items():
    #     if tce.tce_plnt_num == tce_i:
    #         ephem = {'epoch': tce_i_ephem['epoch_corr'],
    #                  'period': tce_i_ephem['period'],
    #                  'duration': tce_i_ephem['duration'] * d_factor,
    #                  'tce_n': tce_i
    #                  }
    ephem = {'epoch': tce['tce_time0bk'],
             'period': tce['tce_period'],
             'duration': tce['tce_duration'] * d_factor,
             }

    begin_time, end_time = all_time[0][0], all_time[-1][-1]

    if ephem['epoch'] < begin_time:
        ephem['epoch'] = ephem['epoch'] + ephem['period'] * np.ceil((begin_time - ephem['epoch']) / ephem['period'])
    else:
        ephem['epoch'] = ephem['epoch'] - ephem['period'] * np.floor((ephem['epoch'] - begin_time) / ephem['period'])
    ephem['duration'] = ephem['duration'] * (1 + 2 * gap_pad)

    if ephem['epoch'] <= end_time:
        midTransitTimes = np.arange(ephem['epoch'], end_time, ephem['period'])
        midTransitTimeBefore = midTransitTimes[0] - ephem['period']
        midTransitTimeAfter = midTransitTimes[-1] + ephem['period']
    else:
        midTransitTimes = []
        midTransitTimeBefore = ephem['epoch'] - ephem['period']
        midTransitTimeAfter = ephem['epoch']

    extendedMidTransitTimes = np.concatenate([[midTransitTimeBefore], midTransitTimes, [midTransitTimeAfter]])

    return extendedMidTransitTimes

#%% ancient code snippet from process_light_curve...

# # patch quarter centroid time series
# all_centroids = patch_centroid_curve(all_centroids)
#
# plot_centroids(all_centroids, add_info, tce, '4patchedquarters')

# quarter_points = []
# idx = 0
# for centroids in all_centroids['x']:
#     quarter_points.append(idx + len(centroids))
#     idx += len(centroids)

# concatenate quarters
# for dim, array in all_centroids.items():
#     all_centroids[dim] = np.concatenate(array)
# centroid_quadr = np.concatenate(centroid_quadr)

# # compute global median and the euclidean distance to it
# median_x = np.nanmedian(all_centroids['x'])
# median_y = np.nanmedian(all_centroids['y'])
# centroid_quadr = np.sqrt(np.square(all_centroids['x'] - median_x) + np.square(all_centroids['y'] - median_y))
#
# plot_dist_centroids(centroid_quadr, quarter_points, add_info, tce, '5distcentr')

# # spline preprocessing
# if not config.whitened:
#     # Fit a piecewise-cubic spline with default arguments.
#     spline = kepler_spline.fit_kepler_spline(all_time, all_flux, verbose=False)[0]
#     spline = np.concatenate(spline)
#
#     # In rare cases the piecewise spline contains NaNs in places the spline could
#     # not be fit. We can't normalize those points if the spline isn't defined
#     # there. Instead we just remove them.
#     finite_i_flux = np.isfinite(spline)
#     finite_i = np.concatenate((finite_i, finite_i_centr))
#     if not np.all(finite_i):
#         # time = time[finite_i]
#         # flux = flux[finite_i]
#         all_flux = all_flux[finite_i]
#         spline = spline[finite_i]
#
#     flux = np.concatenate(all_flux)
#
#     # "Flatten" the light curve (remove low-frequency variability) by dividing by
#     # the spline.
#     # flux /= spline
#     flux /= spline

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

#%% Tests with tfrecords
#
# import tensorflow as tf
# import src_preprocessing.tf_util.example_util as example_util
#
# arr = np.random.normal(4e-5, 1e-17, 2000)
# arr = np.array([np.min(arr), np.max(arr)])
# print(np.mean(arr))
# # np.save('/home/msaragoc/Downloads/arr.npy', arr)
# # saved_arr = np.load('/home/msaragoc/Downloads/arr.npy')
# # arr_py = [float(el) for el in arr]
# # arr_sqrt = np.sqrt(arr)
# #
# # plt.figure()
# # plt.subplot(121)
# # plt.plot(arr)
# # plt.subplot(122)
# # plt.plot(arr_py)
#
# example_tf = tf.train.Example()
#
# example_util.set_float_feature(example_tf, "example_arr", arr)
#
# arr_from_example = example_tf.features.feature['example_arr'].float_list.value
#
# plt.figure()
# plt.subplot(121)
# plt.suptitle('From example in memory')
# plt.plot(arr)
# plt.subplot(122)
# plt.plot(arr_from_example)
#
# print('max:', np.max(arr), np.max(np.array(arr_from_example, dtype='float64')))
# print('min:', np.min(arr), np.min(np.array(arr_from_example, dtype='float64')))
#
# tfrecord_fp = '/home/msaragoc/Downloads/tfrecord_example'
# with tf.python_io.TFRecordWriter(tfrecord_fp) as writer:
#     writer.write(example_tf.SerializeToString())
#
# record_iterator = tf.python_io.tf_record_iterator(path=tfrecord_fp)
# for string_record in record_iterator:
#     new_example_tf = tf.train.Example()
#     new_example_tf.ParseFromString(string_record)
#     new_arr = new_example_tf.features.feature["example_arr"].float_list.value
#
# # plt.figure()
# # plt.subplot(121)
# # plt.title('From saved TFRecord')
# # plt.plot(arr)
# # plt.subplot(122)
# # plt.plot(new_arr)
#
# # plt.figure()
# # plt.plot(np.array(new_arr) * 1e17)
