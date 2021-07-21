""" Preprocessing module adapted from:

# Copyright 2018 The Exoplanet ML Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

Functions for reading and preprocessing light curves and related data.

Authors:
- Laurent Wilkens
- Nikash Walia
- Miguel Martinho
- Sam Donald

"""

# 3rd party
import numpy as np
import tensorflow as tf
import os
from astropy.stats import mad_std

# local
from src_preprocessing.light_curve import kepler_io
from src_preprocessing.light_curve import median_filter
from src_preprocessing.light_curve import util
from src_preprocessing.tf_util import example_util
from src_preprocessing.third_party.kepler_spline import kepler_spline
from src_preprocessing import utils_visualization
from src_preprocessing.utils_centroid_preprocessing import kepler_transform_pxcoordinates_mod13
from src_preprocessing.utils_ephemeris import create_binary_time_series, find_first_epoch_after_this_time
from src_preprocessing import tess_io
from src_preprocessing.utils_odd_even import create_odd_even_views, phase_fold_and_sort_light_curve_odd_even
from src_preprocessing.utils_imputing import impute_binned_ts, imputing_gaps
from src_preprocessing.utils_preprocessing import count_transits
from src_preprocessing.utils_preprocessing_io import report_exclusion
from src_preprocessing.utils_gapping import gap_this_tce, gap_other_tces


def read_light_curve(tce, config):
    """ Reads the FITS files pertaining to a Kepler/TESS target.

    Args:
        tce: row of DataFrame, information on the TCE (ID, ephemeris, ...).
        config: Config object, preprocessing parameters

    Returns:
        all_time: A list of numpy arrays; the time values of the raw light curve
        all_flux: A list of numpy arrays corresponding to the PDC flux time series
        all_centroid: A list of numpy arrays corresponding to the raw centroid time series
        add_info: A dict with additional data extracted from the FITS files; 'quarter' is a list of quarters for each
        NumPy
        array of the light curve; 'module' is the same but for the module in which the target is in every quarter;
        'target position' is a list of two elements which correspond to the target star position, either in world
        (RA, Dec) or local CCD (x, y) pixel coordinates

    Raises:
        IOError: If the light curve files for this target cannot be found.
    """

    # gets data from the lc FITS files for the TCE's target star
    if config['satellite'] == 'kepler':  # Kepler

        # get lc FITS files for the respective target star
        file_names = kepler_io.kepler_filenames(config['lc_data_dir'],
                                                tce.target_id,
                                                injected_group=config['injected_group'])

        if not file_names:
            if not config['omit_missing']:
                raise IOError(f'Failed to find .fits files in {config["lc_data_dir"]} for Kepler ID {tce.target_id}')
            else:
                report_exclusion(config, tce, 'No available lightcurve FITS files')
                return None

        data, fits_files_not_read = kepler_io.read_kepler_light_curve(file_names,
                                                                      centroid_radec=not config['px_coordinates'],
                                                                      prefer_psfcentr=config['prefer_psfcentr'],
                                                                      light_curve_extension=config[
                                                                          'light_curve_extension'],
                                                                      scramble_type=config['scramble_type'],
                                                                      invert=config['invert'])

        if len(fits_files_not_read) > 0:
            report_exclusion(config, tce, 'FITS files not read correctly')
            if len(fits_files_not_read) == len(file_names):
                return None

        return data

    else:  # TESS

        # get sectors for the run
        sectors = [int(sect) for sect in tce['sectors'].split(' ')]

        # get lc FITS files for the respective target star if it was observed for that modality in the given sectors
        file_names = tess_io.tess_filenames(config['lc_data_dir'], tce.target_id, sectors)

        if not file_names:
            if not config['omit_missing']:
                raise IOError(f'Failed to find .fits files in {config["lc_data_dir"]} for TESS ID {tce.target_id}')
            else:
                report_exclusion(config, tce, 'No available lightcurve FITS files')
                return None

        fits_data, fits_files_not_read = tess_io.read_tess_light_curve(file_names,
                                                                       centroid_radec=not config['px_coordinates'],
                                                                       prefer_psfcentr=config['prefer_psfcentr'],
                                                                       light_curve_extension=config[
                                                                           'light_curve_extension'],
                                                                       invert=config['invert']
                                                                       )

        if len(fits_files_not_read) > 0:
            report_exclusion(config, tce, 'FITS files not read correctly')
            if len(fits_files_not_read) == len(file_names):
                return None

        return fits_data


def lininterp_transits(timeseries, transit_pulse_train, centroid=False):
    """ Linearly interpolate the timeseries across the in-transit cadences. The interpolation is performed between the
    boundary points using their values.

    :param timeseries: list of numpy arrays, time-series; if centroid is True, then it is a dictionary with a list of
    numpy arrays for each coordinate ('x' and 'y')
    :param transit_pulse_train: list of numpy arrays, binary arrays that are 1's for in-transit cadences and 0 otherwise
    :param centroid: bool, treats time-series as centroid time-series ('x' and 'y') if True
    :return:
        timeseries_interp: list of numpy arrays, time-series linearly interpolated at the transits; if centroid is True,
         then it is a dictionary with a list of numpy arrays for each coordinate ('x' and 'y')
    """

    # initialize variables
    if centroid:
        num_arrs = len(timeseries['x'])
        timeseries_interp = {'x': [], 'y': []}
    else:
        num_arrs = len(timeseries)
        timeseries_interp = []

    for i in range(num_arrs):

        if centroid:
            timeseries_interp['x'].append(np.array(timeseries['x'][i]))
            timeseries_interp['y'].append(np.array(timeseries['y'][i]))
        else:
            timeseries_interp.append(np.array(timeseries[i]))

        idxs_it = np.where(transit_pulse_train[i] == 1)[0]
        if len(idxs_it) == 0:  # no transits in the array
            continue

        idxs_lim = np.where(np.diff(idxs_it) > 1)[0] + 1

        start_idxs = np.insert(idxs_lim, 0, 0)
        end_idxs = np.append(idxs_lim, len(idxs_it))

        for start_idx, end_idx in zip(start_idxs, end_idxs):

            # boundary issue - do nothing, since the whole array is a transit; does this happen?
            if idxs_it[start_idx] == 0 and idxs_it[end_idx - 1] == len(transit_pulse_train[i]) - 1:
                continue

            if idxs_it[start_idx] == 0:  # boundary issue start - constant value end
                if centroid:
                    timeseries_interp['x'][i][idxs_it[start_idx]:idxs_it[end_idx - 1] + 1] = \
                        timeseries['x'][i][idxs_it[end_idx - 1] + 1]
                    timeseries_interp['y'][i][idxs_it[start_idx]:idxs_it[end_idx - 1] + 1] = \
                        timeseries['y'][i][idxs_it[end_idx - 1] + 1]
                else:
                    timeseries_interp[i][idxs_it[start_idx]:idxs_it[end_idx - 1] + 1] = \
                        timeseries[i][idxs_it[end_idx - 1] + 1]

            elif idxs_it[end_idx - 1] == len(transit_pulse_train[i]) - 1:  # boundary issue end - constant value start
                if centroid:
                    timeseries_interp['x'][i][idxs_it[start_idx]:idxs_it[end_idx - 1] + 1] = timeseries['x'][i][
                        idxs_it[start_idx] - 1]
                    timeseries_interp['y'][i][idxs_it[start_idx]:idxs_it[end_idx - 1] + 1] = timeseries['y'][i][
                        idxs_it[start_idx] - 1]
                else:
                    timeseries_interp[i][idxs_it[start_idx]:idxs_it[end_idx - 1] + 1] = \
                        timeseries[i][idxs_it[start_idx] - 1]

            else:  # linear interpolation
                idxs_interp = np.array([idxs_it[start_idx] - 1, idxs_it[end_idx - 1] + 1])
                idxs_to_interp = np.arange(idxs_it[start_idx], idxs_it[end_idx - 1] + 1)

                if centroid:
                    timeseries_interp['x'][i][idxs_it[start_idx]:idxs_it[end_idx - 1] + 1] = \
                        np.interp(idxs_to_interp, idxs_interp, timeseries['x'][i][idxs_interp])
                    timeseries_interp['y'][i][idxs_it[start_idx]:idxs_it[end_idx - 1] + 1] = \
                        np.interp(idxs_to_interp, idxs_interp, timeseries['y'][i][idxs_interp])
                else:
                    timeseries_interp[i][idxs_it[start_idx]:idxs_it[end_idx - 1] + 1] = \
                        np.interp(idxs_to_interp, idxs_interp, timeseries[i][idxs_interp])

    return timeseries_interp


def _process_tce(tce, table, config, conf_dict):
    """ Processes the time-series and scalar features for the TCE and returns an Example proto.

    :param tce: row of the input TCE table.
    :param table: pandas DataFrame, ephemeris information on other TCEs used when gapping
    :param config: dict, holds preprocessing parameters
    :param conf_dict: dict, keys are a tuple (Kepler ID, TCE planet number) and the values are the confidence level used
    when gapping (between 0 and 1)

    return:
        a tensorflow.train.Example proto containing TCE features
    """

    # import pandas as pd
    # rankingTbl = pd.read_csv('/data5/tess_project/Data/tfrecords/Kepler/Q1-Q17_DR25/tfrecordskeplerdr25-dv_g301-l31_spline_nongapped_flux-loe-lwks-centroid-centroid_fdl-scalars_rbanorm_oecheck_oestd_extrastats_koiephemonlydiff_data/tfrecordskeplerdr25-dv_g301-l31_spline_nongapped_flux-loe-lwks-centroid-centroid_fdl-scalars_rbanorm_oecheck_oestd_extrastats_koiephemonlydiff/merged_shards_disp_set.csv')
    # rankingTbl = rankingTbl.loc[(rankingTbl['mid_global_flux_shift'].abs() >= 15) & (rankingTbl['label'] == 'PC')]
    # rankingTbl = rankingTbl.loc[(rankingTbl['odd_even_flag'] != 'ok') & (rankingTbl['label'].isin(['PC', 'AFP']))]
    # rankingTbl = rankingTbl.loc[(rankingTbl['tfopwg_disp'].isin(['KP', 'CP'])) & (rankingTbl['predicted class'] == 0)]
    # cfpTbl = pd.read_csv('/home/msaragoc/Downloads/fpwg_2020.10.02_15.25.08.csv', header=10)
    # koiTTVs = cfpTbl.loc[cfpTbl['fpwg_comment'].str.contains('TTV', na=False)]['kepoi_name'].values

    # if len(rankingTbl.loc[(rankingTbl['target_id'] == tce['target_id']) &
    #                       (rankingTbl['tce_plnt_num'] == tce['tce_plnt_num'])]) == 1:
    # if len(rankingTbl.loc[rankingTbl['oi'] == tce['oi']]) == 1:
    # if tce['kepoi_name'] in koiTTVs:
    # # if tce['target_id'] in rankingTbl[0:30]['target_id'].values:
    # # if tce['target_id'] in rankingTbl['KICID'].values and tce['tce_plnt_num'] == 1:
    # if tce['target_id'] in rankingTbl[0:10]['target_id'].values:
    # if tce['target_id'] == 9705459 and tce['tce_plnt_num'] == 2:  # tce['av_training_set'] == 'PC' and
    # if '{}-{}'.format(tce['target_id'], tce['tce_plnt_num']) in ['97486585-1']:  # , '3239945-1', '6933567-1', '8416523-1', '9663113-2']:
    # if tce['oi'] in [1774.01]:
    # tce['tce_time0bk'] = 1325.726
    # tce['tce_period'] = 0.941451
    # tce['sectors'] = '1'
    #     print(tce)
    # else:
    #     return None

    # check if preprocessing pipeline figures are saved for the TCE
    plot_preprocessing_tce = False  # False
    if np.random.random() < 0.01:
        plot_preprocessing_tce = config['plot_figures']

    # sample TCE ephemeris using uncertainty interval
    if tce['augmentation_idx'] != 0 and config['augmentation']:  # one of the TCEs does not use uncertainty values

        # tce_tbl_ephem = {ephemeris: tce[ephemeris] for ephemeris in ['tce_period', 'tce_duration', 'tce_time0bk']}
        #
        # # fill missing values in uncertainty
        # for ephemeris in ['tce_period', 'tce_duration', 'tce_time0bk']:
        #     if tce['{}_err'.format(ephemeris)] == -1:
        #         tce['{}_err'.format(ephemeris)] = config['tce_min_err[ephemeris]']
        #
        #     tce[ephemeris] = np.random.normal(tce[ephemeris], tce['{}_err'.format(ephemeris)])
        #
        # # sample ephemerides until they are positive
        # for ephemeris in ['tce_period', 'tce_duration']:
        #     while tce[ephemeris] <= 0:
        #         tce[ephemeris] = np.random.normal(tce_tbl_ephem[ephemeris], tce['{}_err'.format(ephemeris)])

        # add noise to the ephemerides; assuming augmenting data with 2 more TCEs (ephemeris * (1 +- aug_noise_factor))
        for ephemeris in ['tce_period', 'tce_duration', 'tce_time0bk']:
            tce[ephemeris] = tce[ephemeris] * (1 - config['aug_noise_factor'],
                                               1 + config['aug_noise_factor'])[tce['augmentation_idx'] == 1]

    # initialize data dict
    data = {}

    # get cadence, flux and centroid data for the tce
    data_fits = read_light_curve(tce, config)

    # TODO: do we need this check?
    if data_fits is None:
        report_exclusion(config, tce, 'Empty arrays')
        return None

    data.update(data_fits)

    if config['satellite'] == 'kepler':
        add_info = {'quarter': data['quarter'], 'module': data['module']}
    else:
        add_info = {'sectors': data['sectors']}

    # # TODO: do the same thing for TESS
    # if config['get_denoised_centroids']:
    #     all_centroids, idxs_nan_centroids = get_denoised_centroids_kepler(tce.target_id, config['denoised_centroids_dir'])

    if plot_preprocessing_tce:
        utils_visualization.plot_centroids(data['all_time'],
                                           data['all_centroids'],
                                           None,
                                           tce,
                                           config,
                                           os.path.join(config['output_dir'], 'plots'),
                                           f'1_raw_aug{tce["augmentation_idx"]}',
                                           add_info=add_info,
                                           target_position=None
                                           )

        if not config['px_coordinates']:
            utils_visualization.plot_centroids(data['all_time'],
                                               data['all_centroids'],
                                               None,
                                               tce,
                                               config,
                                               os.path.join(config['output_dir'], 'plots'),
                                               f'1_raw_target_aug{tce["augmentation_idx"]}',
                                               add_info=add_info,
                                               target_position=data['target_position']
                                               )

        # utils_visualization.plot_centroids(data['all_time'], data['all_centroids_px'], None, tce, config,
        #                                    os.path.join(config['output_dir'], 'plots'),
        #                                    '1_raw_fdl_aug{}'.format(tce['augmentation_idx']),
        #                                    add_info=add_info,
        #                                    pxcoordinates=True)

    # gap TCE to get weak secondary test time-series
    # get gapped indices
    data['gap_time_noprimary'] = None
    # initialize arrays for gapped time series
    timeseries_gapped = ['all_time_noprimary', 'all_flux_noprimary']
    data['all_time_noprimary'] = [np.array(arr) for arr in data['all_time']]
    data['all_flux_noprimary'] = [np.array(arr) for arr in data['all_flux']]
    gapped_idxs_noprimary = gap_this_tce(data['all_time_noprimary'], tce, gap_pad=config['gap_padding_primary'])
    if config['gap_imputed']:
        data['gap_time_noprimary'] = []
    # set to NaN gapped cadences
    for arr_i in range(len(gapped_idxs_noprimary)):
        if len(gapped_idxs_noprimary[arr_i]) == 0:
            continue
        for timeseries in timeseries_gapped:
            data[timeseries][arr_i][gapped_idxs_noprimary[arr_i]] = np.nan
        if config['gap_imputed'] > 0:
            data['gap_time_noprimary'].append(data['all_time'][arr_i][gapped_idxs_noprimary[arr_i]])

    # non-gapped time array required for FDL centroid time-series
    data['all_time_nongapped'] = [np.array(el) for el in data['all_time']]

    # estimate the global (across quarters) average out-of-transit centroid position before gapping TCEs
    if 'tce_maxmesd' in tce:
        duration_gapped = min((1 + 2 * config['gap_padding']) * tce['tce_duration'], np.abs(tce['tce_maxmesd']),
                              tce['tce_period'])
    else:
        duration_gapped = min((1 + 2 * config['gap_padding']) * tce['tce_duration'], tce['tce_period'])
    # remove NaNs
    # centroid time-series
    joint_timeseries = ['all_time', 'all_centroids']
    finite_idxs = []
    for arr_i in range(len(data[joint_timeseries[0]])):
        finite_idxs_aux = []
        for timeseries in joint_timeseries:
            if 'time' not in timeseries:
                finite_idxs_aux.append(np.isfinite(data[timeseries]['x'][arr_i]))
                finite_idxs_aux.append(np.isfinite(data[timeseries]['y'][arr_i]))
            else:
                finite_idxs_aux.append(np.isfinite(data[timeseries][arr_i]))
        finite_idxs.append(np.logical_and.reduce(finite_idxs_aux))

    # create binary time series for each time array in which in-transit points are labeled as 1's, otherwise as 0's
    binary_time_all = [create_binary_time_series(time[finite_idx],
                                                 find_first_epoch_after_this_time(tce['tce_time0bk'],
                                                                                  tce['tce_duration'],
                                                                                  time[finite_idx][0]),
                                                 duration_gapped,
                                                 tce['tce_period']) if np.any(finite_idx) else []
                       for time, finite_idx in zip(data['all_time'], finite_idxs)]

    # get out-of-transit indices for the centroid time series
    centroid_oot = {coord: [centroids[finite_idx][np.where(binary_time == 0)] for binary_time, centroids, finite_idx in
                            zip(binary_time_all, data['all_centroids'][coord], finite_idxs)]
                    for coord in data['all_centroids']}
    # estimate average out-of-transit centroid as the median across quarters - same as they do in the Kepler pipeline
    # TODO: how to compute the average oot? mean, median, other...
    data['avg_centroid_oot'] = {coord: np.nanmedian(np.concatenate(centroid_oot[coord])) for coord in centroid_oot}

    # gap cadences belonging to the transits of other TCEs in the same target star
    # FIXME: what if removes a whole quarter? need to adjust all_additional_info to it
    data['gap_time'] = None
    if config['gapped']:
        gapped_idxs = gap_other_tces(data['all_time'],
                                     add_info,
                                     tce,
                                     table,
                                     config,
                                     conf_dict,
                                     gap_pad=config['gap_padding'],
                                     keep_overlap=config['gap_keep_overlap'])
        timeseries_gapped = ['all_time', 'all_flux', 'all_centroids']
        if config['gap_imputed']:
            data['gap_time'] = []
        # set to NaN gapped cadences in the time series
        for arr_i in range(len(gapped_idxs)):
            if len(gapped_idxs[arr_i]) == 0:
                continue
            for timeseries in timeseries_gapped:
                if 'centroids' in timeseries:
                    data[timeseries]['x'][arr_i][gapped_idxs[arr_i]] = np.nan
                    data[timeseries]['y'][arr_i][gapped_idxs[arr_i]] = np.nan
                else:
                    data[timeseries][arr_i][gapped_idxs[arr_i]] = np.nan
            if config['gap_imputed']:
                data['gap_time'].append(data['all_time'][arr_i][gapped_idxs[arr_i]])

    data['all_time_centroids'] = [np.array(time) for time in data['all_time']]

    # remove timestamps with NaN or infinite time, or time-series value in each quarter
    # at least some of these NaNs come from gapping the time series
    # other NaNs can come from missing time values from the FITS files
    # flux time-series
    joint_timeseries = ['all_time', 'all_flux']
    for arr_i in range(len(data[joint_timeseries[0]])):
        finite_idxs = []
        for timeseries in joint_timeseries:
            finite_idxs.append(np.isfinite(data[timeseries][arr_i]))
        finite_idxs = np.logical_and.reduce(finite_idxs)
        for timeseries in joint_timeseries:
            data[timeseries][arr_i] = data[timeseries][arr_i][finite_idxs]

    # centroid time-series
    joint_timeseries = ['all_time_centroids', 'all_centroids']
    for arr_i in range(len(data[joint_timeseries[0]])):
        finite_idxs = []
        for timeseries in joint_timeseries:
            if 'time' not in timeseries:
                finite_idxs.append(np.isfinite(data[timeseries]['x'][arr_i]))
                finite_idxs.append(np.isfinite(data[timeseries]['y'][arr_i]))
            else:
                finite_idxs.append(np.isfinite(data[timeseries][arr_i]))
        finite_idxs = np.logical_and.reduce(finite_idxs)
        for timeseries in joint_timeseries:
            if 'time' not in timeseries:
                data[timeseries]['x'][arr_i] = data[timeseries]['x'][arr_i][finite_idxs]
                data[timeseries]['y'][arr_i] = data[timeseries]['y'][arr_i][finite_idxs]
            else:
                data[timeseries][arr_i] = data[timeseries][arr_i][finite_idxs]

    # weak secondary flux time-series
    joint_timeseries = ['all_time_noprimary', 'all_flux_noprimary']
    for arr_i in range(len(data[joint_timeseries[0]])):
        finite_idxs = []
        for timeseries in joint_timeseries:
            finite_idxs.append(np.isfinite(data[timeseries][arr_i]))
        finite_idxs = np.logical_and.reduce(finite_idxs)
        for timeseries in joint_timeseries:
            data[timeseries][arr_i] = data[timeseries][arr_i][finite_idxs]

    # same for FDL centroid time-series and non-gapped time array
    joint_timeseries = ['all_time_nongapped', 'all_centroids_px']
    for arr_i in range(len(data[joint_timeseries[0]])):
        finite_idxs = []
        for timeseries in joint_timeseries:
            if 'time' not in timeseries:
                finite_idxs.append(np.isfinite(data[timeseries]['x'][arr_i]))
                finite_idxs.append(np.isfinite(data[timeseries]['y'][arr_i]))
            else:
                finite_idxs.append(np.isfinite(data[timeseries][arr_i]))
        finite_idxs = np.logical_and.reduce(finite_idxs)
        for timeseries in joint_timeseries:
            if 'time' not in timeseries:
                data[timeseries]['x'][arr_i] = data[timeseries]['x'][arr_i][finite_idxs]
                data[timeseries]['y'][arr_i] = data[timeseries]['y'][arr_i][finite_idxs]
            else:
                data[timeseries][arr_i] = data[timeseries][arr_i][finite_idxs]

    # if plot_preprocessing_tce:
    #     utils_visualization.plot_centroids(data['all_time_centroids'],
    #                                        data['all_centroids'],
    #                                        None,
    #                                        tce,
    #                                        config,
    #                                        os.path.join(config['output_dir'], 'plots'),
    #                                        '2_rawwithoutnans_aug{}'.format(tce['augmentation_idx']),
    #                                        add_info=add_info)

    # preprocess the flux and centroid time series
    data_processed = process_light_curve(data, config, tce, plot_preprocessing_tce)

    # generate TCE example based on the preprocessed data
    return generate_example_for_tce(data_processed, tce, config, plot_preprocessing_tce)


def flux_preprocessing(all_time, all_flux, gap_time, tce, config, plot_preprocessing_tce):
    """ Preprocess the flux time series.

    :param all_time: list of NumPy arrays, timestamps
    :param all_flux: list of NumPy arrays, flux time series
    :param gap_time: list of NumPy arrays, gapped timestamps
    :param tce: pandas Series, TCE parameters
    :param config: dict, preprocessing parameters
    :param plot_preprocessing_tce: bool, set to True to plot figures related to different preprocessing steps
    :return:
        time: list of NumPy arrays, timestamps for preprocessed flux time seriesed
        flux: list of NumPy arrays, preprocessed flux time series
    """

    # split on gaps
    # flux time-series
    all_time, all_flux, _ = util.split(all_time, all_flux, gap_width=config['gapWidth'])

    # add gap after and before transit based on transit duration
    if 'tce_maxmesd' in tce:
        duration_gapped = min((1 + 2 * config['gap_padding']) * tce['tce_duration'], np.abs(tce['tce_maxmesd']),
                              tce['tce_period'])
    else:
        duration_gapped = min((1 + 2 * config['gap_padding']) * tce['tce_duration'], tce['tce_period'])

    # get epoch of first transit for each time array
    first_transit_time_all = [find_first_epoch_after_this_time(tce['tce_time0bk'], tce['tce_period'], time[0])
                              for time in all_time]

    # create binary time series for each time array in which in-transit points are labeled as 1's, otherwise as 0's
    binary_time_all = [create_binary_time_series(time, first_transit_time, duration_gapped, tce['tce_period'])
                       for first_transit_time, time in zip(first_transit_time_all, all_time)]

    if plot_preprocessing_tce:
        utils_visualization.plot_binseries_flux(all_time, all_flux, binary_time_all, tce, config,
                                                os.path.join(config['output_dir'], 'plots'),
                                                '2_binarytimeseriesandflux')

    # linearly interpolate across TCE transits
    all_flux_lininterp = lininterp_transits(all_flux, binary_time_all, centroid=False)

    # fit a spline to the flux time-series
    spline_flux = kepler_spline.fit_kepler_spline(all_time, all_flux_lininterp, verbose=False)[0]

    if plot_preprocessing_tce:
        utils_visualization.plot_flux_fit_spline(all_time,
                                                 all_flux,
                                                 spline_flux,
                                                 tce,
                                                 config,
                                                 os.path.join(config['output_dir'], 'plots'),
                                                 f'3_smoothingandnormalizationflux_aug{tce["augmentation_idx"]}',
                                                 flux_interp=all_flux_lininterp)

    # get indices for which the spline has finite values
    finite_i = [np.isfinite(spline_flux[i]) for i in range(len(spline_flux))]
    # normalize flux time-series by the fitted spline
    all_flux = [all_flux[i][finite_i[i]] / spline_flux[i][finite_i[i]] for i in range(len(spline_flux))
                if len(finite_i[i]) > 0]

    all_time = [all_time[i][finite_i[i]] for i in range(len(all_time)) if len(finite_i[i]) > 0]

    # impute the time series with Gaussian noise based on global estimates of median and std
    if config['gap_imputed']:
        all_time, all_flux = imputing_gaps(all_time, all_flux, gap_time)

    time = np.concatenate(all_time)
    flux = np.concatenate(all_flux)

    return time, flux


def weak_secondary_flux_preprocessing(all_time, all_flux_noprimary, gap_time, tce, config, plot_preprocessing_tce):
    """ Preprocess the weak secondary flux time series.

    :param all_time: list of NumPy arrays, timestamps
    :param all_flux_noprimary: list of NumPy arrays, weak secondary flux time series
    :param gap_time: list of NumPy arrays, gapped timestamps
    :param tce: Pandas Series, TCE parameters
    :param config: dict, preprocessing parameters
    :param plot_preprocessing_tce: bool, set to True to plot figures related to different preprocessing steps
    :return:
        time: list of NumPy arrays, timestamps for preprocessed flux time seriesed
        flux_noprimary: list of NumPy arrays, preprocessed weak secondary flux time series
    """

    all_time, all_flux_noprimary, _ = util.split(all_time, all_flux_noprimary, gap_width=config['gapWidth'])

    # add gap after and before transit based on transit duration
    duration_gapped = min((1 + 2 * config['gap_padding']) * tce['tce_duration'], np.abs(tce['tce_maxmesd']),
                          tce['tce_period'])

    first_transit_time_all_noprimary = [find_first_epoch_after_this_time(tce['tce_time0bk'] + tce['tce_maxmesd'],
                                                                         tce['tce_period'], time[0])
                                        for time in all_time]
    binary_time_all_noprimary = [create_binary_time_series(time, first_transit_time, duration_gapped, tce['tce_period'])
                                 for first_transit_time, time in
                                 zip(first_transit_time_all_noprimary, all_time)]

    if plot_preprocessing_tce:
        utils_visualization.plot_binseries_flux(all_time, all_flux_noprimary, binary_time_all_noprimary,
                                                tce, config,
                                                os.path.join(config['output_dir'], 'plots'),
                                                f'2_binarytimeseries_wksflux_aug{tce["augmentation_idx"]}')

    # spline fitting for the secondary flux time-series
    all_flux_noprimary_lininterp = lininterp_transits(all_flux_noprimary, binary_time_all_noprimary, centroid=False)

    spline_flux_noprimary = kepler_spline.fit_kepler_spline(all_time, all_flux_noprimary_lininterp, verbose=False)[0]

    if plot_preprocessing_tce:
        utils_visualization.plot_flux_fit_spline(all_time,
                                                 all_flux_noprimary,
                                                 spline_flux_noprimary,
                                                 tce,
                                                 config,
                                                 os.path.join(config['output_dir'], 'plots'),
                                                 f'3_smoothingandnormalization_wksflux_aug{tce["augmentation_idx"]}',
                                                 flux_interp=all_flux_noprimary_lininterp)

    finite_i = [np.isfinite(spline_flux_noprimary[i]) for i in range(len(spline_flux_noprimary))]

    all_time = [all_time[i][finite_i[i]] for i in range(len(all_time)) if len(finite_i[i]) > 0]

    all_flux_noprimary = [all_flux_noprimary[i][finite_i[i]] /
                          spline_flux_noprimary[i][finite_i[i]]
                          for i in range(len(spline_flux_noprimary)) if len(finite_i[i]) > 0]

    # impute the time series with Gaussian noise based on global estimates of median and std
    if config['gap_imputed']:
        all_time, all_flux_noprimary = imputing_gaps(all_time, all_flux_noprimary, gap_time)

    time = np.concatenate(all_time)
    flux_noprimary = np.concatenate(all_flux_noprimary)

    return time, flux_noprimary


def centroid_preprocessing(all_time, all_centroids, avg_centroid_oot, target_position, add_info, gap_time, tce, config,
                           plot_preprocessing_tce):
    """ Preprocess the centroid time series.

    :param all_time: list of NumPy arrays, timestamps
    :param all_centroids: dictionary for the two centroid coordinates coded as 'x' and 'y'. Each key maps to a list of
    NumPy arrays for the respective centroid coordinate time series
    :param avg_centroid_oot: dictionary for the two centroid coordinates coded as 'x' and 'y'. Each key maps to the
    estimate of the average out-of-transit centroid
    :param target_position: list, target star position in 'x' and 'y'
    :param add_info: dictionary, additional information such as quarters and modules
    :param gap_time: list of NumPy arrays, gapped timestamps
    :param tce: Pandas Series, TCE parameters
    :param config: dict, preprocessing parameters
    :param plot_preprocessing_tce: bool, set to True to plot figures related to different preprocessing steps
    :return:
        time: NumPy array, timestamps for preprocessed centroid time series
        centroid_dist: NumPy array, preprocessed centroid time series which is an estimate of the distance of the
        transit to the target
    """

    all_time, all_centroids, add_info = util.split(all_time, all_centroids, add_info=add_info, centroid=True,
                                                   gap_width=config['gapWidth'])

    # pixel coordinate transformation for targets on module 13 for Kepler
    if config['px_coordinates'] and config['satellite'] == 'kepler':
        if add_info['module'][0] == 13:
            all_centroids = kepler_transform_pxcoordinates_mod13(all_centroids, add_info)

            if plot_preprocessing_tce:
                utils_visualization.plot_centroids(all_time,
                                                   all_centroids,
                                                   None,
                                                   tce,
                                                   config,
                                                   os.path.join(config['output_dir'], 'plots'),
                                                   f'2_rawaftertransformation_aug{tce["augmentation_idx"]}',
                                                   add_info=add_info)

    # add gap after and before transit based on transit duration
    if 'tce_maxmesd' in tce:
        duration_gapped = min((1 + 2 * config['gap_padding']) * tce['tce_duration'], np.abs(tce['tce_maxmesd']),
                              tce['tce_period'])
    else:
        duration_gapped = min((1 + 2 * config['gap_padding']) * tce['tce_duration'], tce['tce_period'])

    # get epoch of first transit for each time array
    first_transit_time_all = [find_first_epoch_after_this_time(tce['tce_time0bk'], tce['tce_period'], time[0])
                              for time in all_time]

    # create binary time series for each time array in which in-transit points are labeled as 1's, otherwise as 0's
    binary_time_all = [create_binary_time_series(time, first_transit_time, duration_gapped, tce['tce_period'])
                       for first_transit_time, time in zip(first_transit_time_all, all_time)]

    if plot_preprocessing_tce:
        utils_visualization.plot_binseries_flux(all_time, all_centroids, binary_time_all,
                                                tce, config,
                                                os.path.join(config['output_dir'], 'plots'),
                                                f'2_binarytimeseries_centroid_aug{tce["augmentation_idx"]}',
                                                centroid=True)

    # spline fitting and normalization - fit a piecewise-cubic spline with default arguments
    # FIXME: wouldn't it be better to fit a spline to oot values or a Savitzky Golay filter
    #  (as Jeff and Doug mentioned)?
    all_centroids_lininterp = lininterp_transits(all_centroids, binary_time_all, centroid=True)
    spline_centroid = {coord: kepler_spline.fit_kepler_spline(all_time, all_centroids_lininterp[coord],
                                                              verbose=False)[0] for coord in all_centroids_lininterp}

    if plot_preprocessing_tce:
        utils_visualization.plot_centroids(all_time,
                                           all_centroids,
                                           spline_centroid,
                                           tce,
                                           config,
                                           os.path.join(config['output_dir'], 'plots'),
                                           f'3_smoothingandnormalizationcentroid_aug{tce["augmentation_idx"]}')

    # In rare cases the piecewise spline contains NaNs in places the spline could not be fit. We can't normalize those
    # points if the spline isn't defined there. Instead we just remove them.
    finite_i = [np.logical_and(np.isfinite(spline_centroid['x'][i]), np.isfinite(spline_centroid['y'][i]))
                for i in range(len(spline_centroid['x']))]

    # # get average oot per quarter
    # oot_idxs = [np.where(~binary_time) for binary_time in binary_time_all]  # compute it using only finite and oot values
    # oot_idxs = [np.union1d(oot_idxs[i], finite_i[i]) for i in range(len(oot_idxs))]
    # avg_centroid_oot = {coord: [np.median(all_centroids[coord][i][oot_idxs[i]])
    #                             for i in range(len(all_centroids[coord]))] for coord in all_centroids}

    # normalize by the spline
    all_centroids = {coord: [all_centroids[coord][i][finite_i[i]] / spline_centroid[coord][i][finite_i[i]] *
                             avg_centroid_oot[coord] for i in range(len(spline_centroid[coord]))
                             if len(finite_i[i]) > 0]
                     for coord in all_centroids}
    # all_centroids = {coord: [all_centroids[coord][i][finite_i[i]] - spline_centroid[coord][i][finite_i[i]]
    #                          for i in range(len(spline_centroid[coord]))
    #                          if len(finite_i[i]) > 0]
    #                  for coord in all_centroids}
    # normalize by the fitted splines and recover the range by multiplying by the average oot for each quarter
    # all_centroids = {coord: [all_centroids[coord][i][finite_i[i]] / spline_centroid[coord][i][finite_i[i]] *
    #                          avg_centroid_oot[coord][i] for i in range(len(spline_centroid[coord]))]
    #                  for coord in all_centroids}
    # all_centroids = {coord: [all_centroids[coord][i][finite_i[i]] - spline_centroid[coord][i][finite_i[i]]
    #                          for i in range(len(spline_centroid[coord]))]
    #                  for coord in all_centroids}

    binary_time_all = [binary_time_all[i][finite_i[i]] for i in range(len(binary_time_all)) if len(finite_i[i]) > 0]
    all_time = [all_time[i][finite_i[i]] for i in range(len(all_time)) if len(finite_i[i]) > 0]

    # # set outliers to zero using Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    # q25_75 = {'x': {'q25': np.percentile(np.concatenate(all_centroids['x']), 25),
    #                 'q75': np.percentile(np.concatenate(all_centroids['x']), 75)},
    #           'y': {'q25': np.percentile(np.concatenate(all_centroids['y']), 25),
    #                 'q75': np.percentile(np.concatenate(all_centroids['y']), 75)}
    #           }
    # iqr = {'x': q25_75['x']['q75'] - q25_75['x']['q25'],
    #        'y': q25_75['y']['q75'] - q25_75['y']['q25']}
    # outlier_thr = 1.5
    # for coord in all_centroids:
    #     for i in range(len(all_centroids[coord])):
    #         all_centroids[coord][i][np.where(all_centroids[coord][i] > q25_75[coord]['q75'] + outlier_thr * iqr[coord])] = avg_centroid_oot[coord]
    #         all_centroids[coord][i][np.where(all_centroids[coord][i] < q25_75[coord]['q25'] - outlier_thr * iqr[coord])] = avg_centroid_oot[coord]

    avg_centroid_oot = {coord: np.median(np.concatenate(all_centroids[coord])) for coord in all_centroids}

    # compute the new average oot after the spline fitting and normalization
    # TODO: how to compute the average oot? mean, median, other...
    if plot_preprocessing_tce:
        utils_visualization.plot_centroids_it_oot(all_time,
                                                  binary_time_all,
                                                  all_centroids,
                                                  avg_centroid_oot,
                                                  target_position,
                                                  tce, config,
                                                  os.path.join(config['output_dir'], 'plots'),
                                                  f'4_centroidtimeseries_it-ot-target_aug{tce["augmentation_idx"]}',
                                                  target_center=True)

    # compute the corrected centroid time-series normalized by the transit depth fraction and centered on the avg oot
    # centroid position
    transit_depth = tce['transit_depth'] + 1
    transitdepth_term = (1e6 - transit_depth) / transit_depth
    # avg_centroid_oot = {coord: avg_centroid_oot[coord] * 1.15 for coord in avg_centroid_oot}
    # all_centroids_corr = {coord: [-((all_centroids[coord][i] - avg_centroid_oot[coord]) * transitdepth_term) /
    #                               (1, np.cos(target_position[1] * np.pi / 180))[coord == 'x'] +
    #                               avg_centroid_oot[coord] for i in range(len(all_centroids[coord]))]
    #                       for coord in all_centroids}
    # only oot centroid
    # all_centroids_corr = {coord: [avg_centroid_oot[coord] * np.ones(len(all_centroids[coord][i])) for i in range(len(all_centroids[coord]))]
    #                       for coord in all_centroids}
    # only it centroid
    all_centroids_corr = {coord: [-(all_centroids[coord][i] - avg_centroid_oot[coord]) * transitdepth_term /
                                  (1, np.cos(all_centroids['y'][i] * np.pi / 180))[coord == 'x'] +
                                  avg_centroid_oot[coord]
                                  for i in range(len(all_centroids[coord]))]
                          for coord in all_centroids}
    # all_centroids_corr = {coord: [all_centroids[coord][i] for i in range(len(all_centroids[coord]))]
    #                       for coord in all_centroids}
    # correction performed using quarter average oot estimates
    # all_centroids_corr = {coord: [-((all_centroids[coord][i] - avg_centroid_oot[coord][i]) * transitdepth_term) /
    #                               (1, np.cos(all_centroids['y'][i] * np.pi / 180))[coord == 'x'] +
    #                               avg_centroid_oot[coord][i] for i in range(len(all_centroids[coord]))]
    #                       for coord in all_centroids}
    # all_centroids_corr = {coord: [-(all_centroids[coord][i] * transitdepth_term) /
    #                               (1, np.cos((all_centroids['y'][i]) * np.pi / 180))[coord == 'x'] +
    #                               avg_centroid_oot[coord] for i in range(len(all_centroids[coord]))]
    #                       for coord in all_centroids}

    if plot_preprocessing_tce:
        utils_visualization.plot_corrected_centroids(all_time,
                                                     all_centroids_corr,
                                                     avg_centroid_oot,
                                                     target_position,
                                                     tce,
                                                     config,
                                                     os.path.join(config['output_dir'], 'plots'),
                                                     f'5_correctedcentroidtimeseries_aug{tce["augmentation_idx"]}')

    if config['px_coordinates']:
        # TODO: map target position from celestial coordinates to CCD frame
        # compute the euclidean distance of the corrected centroid time series to the target star position
        all_centroid_dist = [np.sqrt(np.square(all_centroids['x'][i] - target_position[0]) +
                                     np.square(all_centroids['y'][i] - target_position[1]))
                             for i in range(len(all_centroids['x']))]
    else:
        # compute the angular distance of the corrected centroid time series to the target star position
        # all_centroid_dist = [np.sqrt(np.square((all_centroids['x'][i] - target_position[0])) +
        #                              np.square(all_centroids['y'][i] - target_position[1]))
        #                      for i in range(len(all_centroids['x']))]
        all_centroid_dist = [np.sqrt(np.square((all_centroids['x'][i] - tce['ra'])) +
                                     np.square(all_centroids['y'][i] - tce['dec']))
                             for i in range(len(all_centroids['x']))]

    # # get the across quarter average oot estimate using only finite and oot values
    # avg_centroid_oot_dist_global = np.median(np.concatenate([all_centroid_dist[i][oot_idxs[i]]
    #                                                          for i in range(len(all_centroid_dist))]))
    # spline_centroid_dist = kepler_spline.fit_kepler_spline(all_time, all_centroid_dist, verbose=False)[0]
    # # center the offset centroid distance to the across quarter average oot estimate
    # all_centroid_dist = [all_centroid_dist[i] / np.median(all_centroid_dist[i][oot_idxs[i]]) *
    #                      avg_centroid_oot_dist_global
    #                      for i in range(len(all_centroid_dist))]

    # convert from degree to arcsec
    if not config['px_coordinates']:
        all_centroid_dist = [centroid_dist_arr * 3600 for centroid_dist_arr in all_centroid_dist]

    if plot_preprocessing_tce:
        utils_visualization.plot_dist_centroids(all_time,
                                                all_centroid_dist,
                                                None,
                                                None,
                                                tce,
                                                config,
                                                os.path.join(config['output_dir'], 'plots'),
                                                f'6_distcentr_aug{tce["augmentation_idx"]}')

    # impute the time series with Gaussian noise based on global estimates of median and std
    if config['gap_imputed']:
        time, all_centroid_dist = imputing_gaps(all_time, all_centroid_dist, gap_time)

    time = np.concatenate(all_time)
    centroid_dist = np.concatenate(all_centroid_dist)

    return time, centroid_dist  # , add_info


def centroidFDL_preprocessing(all_time, all_centroids, add_info, gap_time, tce, config, plot_preprocessing_tce):
    """ Preprocess the centroid time series following FDL preprocessing pipeline [1].

     :param all_time: list of NumPy arrays, timestamps
     :param all_centroids: dictionary for the two centroid coordinates coded as 'x' and 'y'. Each key maps to a list of
     NumPy arrays for the respective centroid coordinate time series
     :param add_info: dictionary, additional information such as quarters and modules
     :param gap_time: list of NumPy arrays, gapped timestamps
     :param tce: pandas Series, TCE parameters
     :param config: dict, preprocessing parameters
     :param plot_preprocessing_tce: bool, set to True to plot figures related to different preprocessing steps
     :return:
         time: NumPy array, timestamps for preprocessed centroid time series
         centroid_dist: NumPy array, preprocessed centroid time series according to FDL

     [1] Ansdell, Megan, et al. "Scientific Domain Knowledge Improves Exoplanet Transit Classification with Deep
     Learning." The Astrophysical Journal Letters 869.1 (2018): L7.
     """

    all_time, all_centroids, add_info = util.split(all_time, all_centroids, gap_width=config['gapWidth'], centroid=True,
                                                   add_info=add_info)

    if config['satellite'] == 'kepler' and add_info['quarter'][0] == 13:
        all_centroids = kepler_transform_pxcoordinates_mod13(all_centroids, add_info)

    first_transit_time_all = [find_first_epoch_after_this_time(tce['tce_time0bk'], tce['tce_period'], time[0])
                              for time in all_time]

    if 'tce_maxmesd' in tce:
        duration_gapped = min((1 + 2 * config['gap_padding']) * tce['tce_duration'], np.abs(tce['tce_maxmesd']),
                              tce['tce_period'])
    else:
        duration_gapped = min((1 + 2 * config['gap_padding']) * tce['tce_duration'], tce['tce_period'])

    binary_time_all = [create_binary_time_series(time, first_transit_time, duration_gapped, tce['tce_period'])
                       for first_transit_time, time in zip(first_transit_time_all, all_time)]

    all_centroids_lininterp = lininterp_transits(all_centroids, binary_time_all, centroid=True)

    spline_centroid = {coord: kepler_spline.fit_kepler_spline(all_time,
                                                              all_centroids_lininterp[coord],
                                                              verbose=False)[0] for coord in all_centroids_lininterp}

    finite_i_centroid = [np.logical_and(np.isfinite(spline_centroid['x'][i]), np.isfinite(spline_centroid['y'][i]))
                         for i in range(len(spline_centroid['x']))]

    # all_centroids = {coord: [all_centroids[coord][i][finite_i_centroid[i]] /
    #                          spline_centroid[coord][i][finite_i_centroid[i]]
    #                          for i in range(len(spline_centroid[coord])) if len(finite_i_centroid[i] > 0)]
    #                  for coord in all_centroids}
    all_centroids = {coord: [all_centroids[coord][i][finite_i_centroid[i]] -
                             spline_centroid[coord][i][finite_i_centroid[i]]
                             for i in range(len(spline_centroid[coord])) if len(finite_i_centroid[i] > 0)]
                     for coord in all_centroids}


    all_time = [all_time[i][finite_i_centroid[i]] for i in range(len(all_time)) if len(finite_i_centroid[i]) > 0]

    all_centroid_dist = [np.sqrt(np.square(all_centroids['x'][i]) + np.square(all_centroids['y'][i]))
                         for i in range(len(all_centroids['x']))]

    if plot_preprocessing_tce:
        utils_visualization.plot_dist_centroids(all_time,
                                                all_centroid_dist,
                                                None,
                                                None,
                                                tce,
                                                config,
                                                os.path.join(config['output_dir'], 'plots'),
                                                f'6_distcentrfdl_aug{tce["augmentation_idx"]}',
                                                pxcoordinates=True)

    # impute the time series with Gaussian noise based on global estimates of median and std
    if config['gap_imputed']:
        all_time, all_centroid_dist = imputing_gaps(all_time, all_centroid_dist, gap_time)

    time = np.concatenate(all_time)
    centroid_dist = np.concatenate(all_centroid_dist)

    return time, centroid_dist


def process_light_curve(data, config, tce, plot_preprocessing_tce=False):
    """ Removes low-frequency variability from the flux and centroid time-series.

    Args:
      data: dictionary containing data related to the time-series to be preprocessed before generating views
      config: dict, holds preprocessing parameters
      tce: Pandas Series, row of the TCE table for the TCE that is being processed; it contains data such as the
      ephemeris
      plot_preprocessing_tce: bool, if True plots figures for several steps in the preprocessing pipeline

    Returns:
      data_views: dict, containing data used to generate the views
    """

    if config['satellite'] == 'kepler':
        add_info_centr = {'quarter': data['quarter'], 'module': data['module']}
    else:
        add_info_centr = None

    # preprocess flux time series
    try:
        time, flux = flux_preprocessing(data['all_time'],
                                        data['all_flux'],
                                        data['gap_time'],
                                        tce,
                                        config,
                                        plot_preprocessing_tce)
    except:
        time, flux = None, None

    # # preprocess weak secondary flux time series
    try:
        time_wksecondaryflux, wksecondaryflux = weak_secondary_flux_preprocessing(data['all_time_noprimary'],
                                                                                  data['all_flux_noprimary'],
                                                                                  data['gap_time_noprimary'],
                                                                                  tce,
                                                                                  config,
                                                                                  plot_preprocessing_tce)
    except:
        time_wksecondaryflux, wksecondaryflux = None, None

    # preprocess centroid time series
    try:
        time_centroid, centroid_dist = centroid_preprocessing(data['all_time_centroids'],
                                                              data['all_centroids'],
                                                              data['avg_centroid_oot'],
                                                              data['target_position'],
                                                              add_info_centr,
                                                              data['gap_time'],
                                                              tce, config, plot_preprocessing_tce)
    except:
        time_centroid, centroid_dist = None, None

    # preprocess FDL centroid time series
    try:
        time_centroidFDL, centroid_distFDL = centroidFDL_preprocessing(data['all_time_nongapped'],
                                                                       data['all_centroids_px'],
                                                                       add_info_centr,
                                                                       [],
                                                                       tce,
                                                                       config,
                                                                       plot_preprocessing_tce)
    except:
        time_centroidFDL, centroid_distFDL = None, None

    # dictionary with preprocessed time series used to generate views
    data_views = {}
    data_views['time'] = time
    data_views['flux'] = flux
    data_views['time_centroid_dist'] = time_centroid
    data_views['centroid_dist'] = centroid_dist
    data_views['time_wksecondaryflux'] = time_wksecondaryflux
    data_views['wksecondaryflux'] = wksecondaryflux
    data_views['time_centroid_distFDL'] = time_centroidFDL
    data_views['centroid_distFDL'] = centroid_distFDL

    return data_views


def min_max_normalization(arr, max_val, min_val):
    """ Min-max normalization.

    :param arr: array
    :param max_val: float, max val
    :param min_val: float, min val
    :return:
        normalized array
    """

    return (arr - min_val) / (max_val - min_val)


def phase_fold_and_sort_light_curve(time, timeseries, period, t0, augmentation=False):
    """ Phase folds a light curve and sorts by ascending time.

    Args:
      time: 1D NumPy array of time values.
      timeseries: 1D NumPy array of time series values.
      period: A positive real scalar; the period to fold over.
      t0: The center of the resulting folded vector; this value is mapped to 0.

    Returns:
      time: 1D NumPy array of phase folded time values in
          [-period / 2, period / 2), where 0 corresponds to t0 in the original
          time array. Values are sorted in ascending order.
      timeseries: 1D NumPy array. Values are the same as the original input
          array, but sorted by folded_time.
      num_transits: int, number of transits in the time series.
    """

    # Phase fold time.
    if not augmentation:
        time = util.phase_fold_time(time, period, t0)
    else:
        time, sampled_idxs, num_transits = util.phase_fold_time_aug(time, period, t0)
        timeseries = timeseries[sampled_idxs]

    # count number of transits in the phase domain
    num_transits = np.sum(np.diff(time) < 0) + 1

    # Sort by ascending time.
    sorted_i = np.argsort(time)
    time = time[sorted_i]
    timeseries = timeseries[sorted_i]

    return time, timeseries, num_transits


def generate_view(time, flux, num_bins, bin_width, t_min, t_max, tce,
                  centering=True, normalize=True, centroid=False, **kwargs):
    """ Generates a view of a phase-folded light curve using a median filter.

    Args:
      time: 1D array of time values, sorted in ascending order.
      flux: 1D array of flux/centroid values.
      num_bins: The number of intervals to divide the time axis into.
      bin_width: The width of each bin on the time axis.
      t_min: The inclusive leftmost value to consider on the time axis.
      t_max: The exclusive rightmost value to consider on the time axis.
      tce: Pandas Series, TCE parameters
      centering: bool, whether to center the view by subtracting the median
      normalize: Whether to perform normalization
      centroid: bool, if True considers these view a centroid time series

    Returns:
      1D NumPy array of size num_bins containing the median flux values of
      uniformly spaced bins on the phase-folded time axis.
    """

    # binning using median
    view, time_bins, view_var, bin_counts = median_filter.median_filter(time, flux, num_bins, bin_width, t_min, t_max)

    # impute missing bin values
    view, inds_nan = impute_binned_ts(time_bins, view, time, flux, tce['tce_period'], tce['tce_duration'])

    # variability of bins without any values is set to the std of the phase folded time series
    view_var[np.isnan(view_var)] = mad_std(flux, ignore_nan=True)

    # global median centering
    if centering:
        view -= np.median(view)

    # normalization
    if normalize:
        view = normalize_view(view, val=None, centroid=centroid, **kwargs)

    return view, time_bins, view_var, inds_nan, bin_counts


def normalize_view(view, val=None, centroid=False, **kwargs):
    """ Normalize the phase-folded time series.

    :param view: array, phase-folded time series
    :param val: float, value used to normalize the time series
    :param centroid: bool, True for centroid time series
    :param kwargs: dict, extra keyword parameters
    :return:
        array, normalized phase-folded time series
    """

    # for the centroid time series
    # range [new_min_val, 1], assuming max is positive, which should be since we are removing the median from a
    # non-negative time series
    # for the flux time series
    # range [-1, new_max_val], assuming min is negative, if not [1, new_max_val]
    if val is None:
        val = np.abs(np.max(view)) if centroid else np.abs(np.min(view))

    if val == 0:
        print(f'Dividing view by 0. Returning the non-normalized view {kwargs["report"]["view"]}.')
        report_exclusion(kwargs['report']['config'], kwargs['report']['tce'],
                         f'Dividing view by 0. Returning the non-normalized view {kwargs["report"]["view"]}.')

        return view

    return view / val


def centering_and_normalization(view, val_centr, val_norm, **kwargs):
    """ Center and normalize a 1D time series.

    :param view: array, 1D time series
    :param val_centr: float, value used to center the time series
    :param val_norm: float, value used normalize the time series
    :param kwargs: dict, extra keyword parameters
    :return:
        array, centered and normalized time series
    """

    if val_norm == 0:
        print(f'Dividing view by 0. Returning the non-normalized view {kwargs["report"]["view"]}.')
        report_exclusion(kwargs['report']['config'], kwargs['report']['tce'],
                         f'Dividing view by 0. Returning the non-normalized view {kwargs["report"]["view"]}.')

        return view - val_centr

    return (view - val_centr) / val_norm


def global_view(time, flux, period, tce, num_bins=2001, bin_width_factor=1/2001, centroid=False, normalize=True,
                centering=True, **kwargs):
    """Generates a 'global view' of a phase folded light curve.

    See Section 3.3 of Shallue & Vanderburg, 2018, The Astronomical Journal.
    http://iopscience.iop.org/article/10.3847/1538-3881/aa9e09/meta

    Args:
      time: 1D array of time values, sorted in ascending order.
      flux: 1D array of flux values.
      period: The period of the event (in days).
      tce: Pandas Series, TCE parameters
      num_bins: The number of intervals to divide the time axis into.
      bin_width_factor: Width of the bins, as a fraction of period.
      centering: bool, whether to center the view by subtracting the median
      normalize: Whether to perform normalization
      centroid: bool, if True considers these view a centroid time series


    Returns:
      1D NumPy array of size num_bins containing the median flux values of
      uniformly spaced bins on the phase-folded time axis.
    """
    return generate_view(
        time,
        flux,
        num_bins=num_bins,
        bin_width=max(period * bin_width_factor, kwargs['tce_duration'] * 0.16),
        t_min=-period / 2,
        t_max=period / 2,
        tce=tce,
        centroid=centroid,
        normalize=normalize,
        centering=centering,
        **kwargs)


def local_view(time,
               flux,
               period,
               duration,
               tce,
               num_bins=201,
               bin_width_factor=0.16,
               num_durations=4,
               centroid=False,
               normalize=True,
               centering=True,
               **kwargs):
    """Generates a 'local view' of a phase folded light curve.

    See Section 3.3 of Shallue & Vanderburg, 2018, The Astronomical Journal.
    http://iopscience.iop.org/article/10.3847/1538-3881/aa9e09/meta

    Args:
      time: 1D array of time values, sorted in ascending order.
      flux: 1D array of flux values.
      period: The period of the event (in days).
      duration: The duration of the event (in days).
      tce: Pandas Series, TCE parameters
      num_bins: The number of intervals to divide the time axis into.
      bin_width_factor: Width of the bins, as a fraction of duration.
      num_durations: The number of durations to consider on either side of 0 (the event is assumed to be centered at 0).
      centering: bool, whether to center the view by subtracting the median
      normalize: Whether to perform normalization
      centroid: bool, if True considers these view a centroid time series


    Returns:
      1D NumPy array of size num_bins containing the median flux values of
      uniformly spaced bins on the phase-folded time axis.
    """
    return generate_view(
        time,
        flux,
        tce=tce,
        num_bins=num_bins,
        bin_width=duration * bin_width_factor,
        t_min=max(-period / 2, -duration * num_durations),
        t_max=min(period / 2, duration * num_durations),
        centroid=centroid,
        normalize=normalize,
        centering=centering,
        **kwargs)


def remove_positive_outliers(time, ts, sigma, fill=False):
    """ Remove positive outliers in the flux time series using a threshold in the standard deviation. Option to fill
    outlier cadences using Gaussian noise with global statistics of the time series.

    :param time: NumPy array, time stamps
    :param ts: NumPy array, time series
    :param sigma: float, sigma factor
    :param fill: bool, if True fills outliers
    :return:
        time: NumPy array, time stamps for the new time series
        ts: NumPy array, time series without outliers
        idxs_out: NumPy array, outlier indices
    """

    rob_std = mad_std(ts)

    idxs_out = np.where(ts >= 1 + sigma * rob_std)

    if fill:
        # fill with mean
        # fill_val = np.median(ts)
        # ts[idxs_out] = fill_val

        # fill with Gaussian noise with global time series statistics
        ts[idxs_out] = np.random.normal(np.median(ts), rob_std, idxs_out[0].shape)
    else:
        ts[idxs_out] = np.nan
        idxs_in = ~np.isnan(ts)
        ts = ts[idxs_in]
        time = time[idxs_in]

    return time, ts, idxs_out


def generate_example_for_tce(data, tce, config, plot_preprocessing_tce=False):
    """ Generates a tf.train.Example representing an input TCE.

    Args:
      data: dictionary containing preprocessed time-series used to generate the views
      tce: Dict-like object containing at least 'tce_period', 'tce_duration', and
        'tce_time0bk'. Additional items are included as features in the output.
      config: dict; preprocessing parameters.
      plot_preprocessing_tce: bool, if True plots figures for some steps while generating the inputs.

    Returns:
      A tf.train.Example containing features. These features can be time series, stellar parameters, statistical
      quantities, .... it returns None if some exception while creating these features occurs
    """

    # if len(data['time']) == 0:
    #     print(tce)
    #     report_exclusion(config, tce, 'No data[time] before creating views.')
    #     return None
    #
    # if len(data['time_centroid_dist']) == 0:
    #     report_exclusion(config, tce, 'No preprocessed time series for centroid.')
    #     return None

    # check if data for all views is valid after preprocessing the raw time series
    for key, val in data.items():
        if val is None:
            report_exclusion(config, tce, f'No data before creating views for {key}.')
            return None

    # time interval for the transit
    t_min_transit, t_max_transit = max(-tce['tce_period'] / 2, -tce['tce_duration'] / 2), \
                                   min(tce['tce_period'] / 2, tce['tce_duration'] / 2)

    # initialize dictionary with data for the preprocessing table
    example_stats = {}
    # initialize number of transits per time series dictionary
    num_transits = {}
    # initialize dictionary with number of empty bins per view
    inds_bin_nan = {}

    # phase folding for odd and even time series
    (odd_time, odd_flux, _, odd_time_nophased), \
    (even_time, even_flux, _, even_time_nophased) = phase_fold_and_sort_light_curve_odd_even(data['time'],
                                                                                             data['flux'],
                                                                                             tce['tce_period'],
                                                                                             tce['tce_time0bk'],
                                                                                             augmentation=False)
    num_transits['flux_odd'] = count_transits(odd_time_nophased,
                                              tce['tce_period'],
                                              tce['tce_time0bk'],
                                              tce['tce_duration'])
    num_transits['flux_even'] = count_transits(even_time_nophased,
                                               tce['tce_period'],
                                               tce['tce_time0bk'],
                                               tce['tce_duration'])

    # phase folding for flux time series
    time, flux, _ = phase_fold_and_sort_light_curve(data['time'],
                                                    data['flux'],
                                                    tce['tce_period'],
                                                    tce['tce_time0bk'],
                                                    augmentation=False)
    num_transits['flux'] = count_transits(data['time'],
                                          tce['tce_period'],
                                          tce['tce_time0bk'],
                                          tce['tce_duration'])

    # phase folding for centroid time series
    time_centroid_dist, centroid_dist, \
    num_transits['centroid'] = phase_fold_and_sort_light_curve(data['time_centroid_dist'],
                                                               data['centroid_dist'],
                                                               tce['tce_period'],
                                                               tce['tce_time0bk'],
                                                               augmentation=False)
    num_transits['centroid'] = count_transits(data['time_centroid_dist'],
                                              tce['tce_period'],
                                              tce['tce_time0bk'],
                                              tce['tce_duration'])

    # phase folding for the weak secondary flux time series
    time_noprimary, flux_noprimary, num_transits['wks'] = phase_fold_and_sort_light_curve(data['time_wksecondaryflux'],
                                                                                          data['wksecondaryflux'],
                                                                                          tce['tce_period'],
                                                                                          tce['tce_time0bk'] +
                                                                                          tce['tce_maxmesd'],
                                                                                          augmentation=False)
    num_transits['wks'] = count_transits(data['time_wksecondaryflux'],
                                         tce['tce_period'],
                                         tce['tce_time0bk'] + tce['tce_maxmesd'],
                                         tce['tce_duration'])

    # same for FDL centroid time-series
    # phase folding for flux and centroid time series
    time_centroid_dist_fdl, centroid_dist_fdl, \
    num_transits['centroid_fdl'] = phase_fold_and_sort_light_curve(data['time_centroid_distFDL'],
                                                                   data['centroid_distFDL'],
                                                                   tce['tce_period'],
                                                                   tce['tce_time0bk'],
                                                                   augmentation=False)
    num_transits['centroid_fdl'] = count_transits(data['time_centroid_distFDL'],
                                                  tce['tce_period'],
                                                  tce['tce_time0bk'],
                                                  tce['tce_duration'])

    phasefolded_timeseries = {'Flux': (time, flux),
                              'Odd Flux': (odd_time, odd_flux),
                              'Even Flux': (even_time, even_flux),
                              'Weak Secondary Flux': (time_noprimary, flux_noprimary),
                              'Centroid Offset Distance': (time_centroid_dist, centroid_dist),
                              'Centroid Offset Distance FDL': (time_centroid_dist_fdl, centroid_dist_fdl)
                              }

    # remove positive outliers from the phase folded flux time series
    if config['pos_outlier_removal']:
        phasefolded_timeseries_posout = {ts: (np.array(ts_arr[0]), np.array(ts_arr[1]))
                                         for ts, ts_arr in phasefolded_timeseries.items() if 'Flux' in ts}
        for ts in phasefolded_timeseries:
            if 'Flux' in ts:
                ts_time, ts_arr, idxs_out = remove_positive_outliers(
                    phasefolded_timeseries[ts][0],
                    phasefolded_timeseries[ts][1],
                    sigma=config['pos_outlier_removal_sigma'],
                    fill=config['pos_outlier_removal_fill']
                )
                phasefolded_timeseries[ts] = (ts_time, ts_arr)
                phasefolded_timeseries_posout[ts] = (phasefolded_timeseries_posout[ts][0][idxs_out],
                                                     phasefolded_timeseries_posout[ts][1][idxs_out])
    else:
        phasefolded_timeseries_posout = None

    if plot_preprocessing_tce:
        # CHANGE NUMBER OF PLOTS AS FUNCTION OF THE TIME SERIES PREPROCESSED
        utils_visualization.plot_all_phasefoldedtimeseries(phasefolded_timeseries,
                                                           tce,
                                                           config,
                                                           (3, 2),
                                                           os.path.join(config['output_dir'], 'plots'),
                                                           f'7_phasefoldedbinned_timeseries_'
                                                           f'aug{tce["augmentation_idx"]}',
                                                           phasefolded_timeseries_posout
                                                           )

    try:

        # make output proto
        ex = tf.train.Example()

        # initialize binned time series
        binned_timeseries = {}

        # preprocess odd and even flux time series to create local views and related processed data
        odd_data, even_data, odd_even_flag = \
            create_odd_even_views(odd_time,
                                  odd_flux,
                                  even_time,
                                  even_flux,
                                  num_transits['flux_odd'],
                                  num_transits['flux_even'],
                                  tce,
                                  config)

        phasefolded_timeseries['Odd Flux'] = (odd_time, odd_flux)
        phasefolded_timeseries['Even Flux'] = (even_time, even_flux)

        binned_timeseries['Local Odd Flux'] = (odd_data['binned_time'], odd_data['local_flux_view'],
                                               odd_data['local_flux_view_se'])
        binned_timeseries['Local Even Flux'] = (even_data['binned_time'], even_data['local_flux_view'],
                                                even_data['local_flux_view_se'])

        # create global flux view
        glob_flux_view, binned_time, glob_flux_view_var, inds_bin_nan['global_flux'], bin_counts = \
            global_view(time,
                        flux,
                        tce['tce_period'],
                        tce=tce,
                        normalize=False,
                        centering=False,
                        num_bins=config['num_bins_glob'],
                        bin_width_factor=config['bin_width_factor_glob'],
                        report={'config': config, 'tce': tce, 'view': 'global_flux_view'},
                        tce_duration=tce['tce_duration']
                        )

        bin_counts[bin_counts == 0] = max(1, np.median(bin_counts))
        glob_flux_view_var /= np.sqrt(bin_counts)
        binned_timeseries['Global Flux'] = (binned_time, glob_flux_view, glob_flux_view_var)

        # create local flux view
        loc_flux_view, binned_time, loc_flux_view_var, inds_bin_nan['local_flux'], bin_counts = \
            local_view(time,
                       flux,
                       tce['tce_period'],
                       tce['tce_duration'],
                       tce=tce,
                       normalize=False,
                       centering=False,
                       num_durations=config['num_durations'],
                       num_bins=config['num_bins_loc'],
                       bin_width_factor=config['bin_width_factor_loc'],
                       report={'config': config, 'tce': tce, 'view': 'local_flux_view'}
                       )

        bin_counts[bin_counts == 0] = max(1, np.median(bin_counts))
        loc_flux_view_var /= np.sqrt(bin_counts)
        binned_timeseries['Local Flux'] = (binned_time, loc_flux_view, loc_flux_view_var)

        flux_views = {'global_flux_view': glob_flux_view,
                      'local_flux_view': loc_flux_view,
                      'local_flux_odd_view': odd_data['local_flux_view'],
                      'local_flux_even_view': even_data['local_flux_view'],
                      }

        flux_views_var = {'global_flux_view': glob_flux_view_var,
                          'local_flux_view': loc_flux_view_var,
                          'local_flux_odd_view': odd_data['local_flux_view_se'],
                          'local_flux_even_view': even_data['local_flux_view_se'],
                          }

        # create local view for the weak secondary flux
        loc_weak_secondary_view, binned_time, loc_weak_secondary_view_var, inds_bin_nan['local_wks'], bin_counts = \
            local_view(time_noprimary,
                       flux_noprimary,
                       tce['tce_period'],
                       tce['tce_duration'],
                       tce=tce,
                       normalize=False,
                       centering=False,
                       num_durations=config['num_durations'],
                       num_bins=config['num_bins_loc'],
                       bin_width_factor=config['bin_width_factor_loc'],
                       report={'config': config, 'tce': tce, 'view': 'local_wks_view'}
                       )

        bin_counts[bin_counts == 0] = max(1, np.median(bin_counts))
        loc_weak_secondary_view_var /= np.sqrt(bin_counts)
        binned_timeseries['Local Weak Secondary Flux'] = (binned_time,
                                                          loc_weak_secondary_view,
                                                          loc_weak_secondary_view_var
                                                          )

        # if plot_preprocessing_tce:
        #     utils_visualization.plot_wks(glob_view, glob_view_weak_secondary, tce, config,
        #                                  os.path.join(config['output_dir'], 'plots'), '8_wks_test')

        # get median and std statistics for global and local flux views
        flux_views_stats = {'median': {'global': np.median(flux_views['global_flux_view']),
                                       'local': np.median(flux_views['local_flux_view'])}}
        flux_views_stats['min'] = {'global': np.abs(np.min(flux_views['global_flux_view'] -
                                                           flux_views_stats['median']['global'])),
                                   'local': np.abs(np.min(flux_views['local_flux_view'] -
                                                          flux_views_stats['median']['local']))}

        # center by the flux view median and normalize by the flux view absolute minimum
        views_aux = {}
        for flux_view in flux_views:
            views_aux[f'{flux_view}_fluxnorm'] = \
                centering_and_normalization(flux_views[flux_view],
                                            flux_views_stats['median'][('local', 'global')['global' in flux_view]],
                                            flux_views_stats['min'][('local', 'global')['global' in flux_view]],
                                            report={'config': config, 'tce': tce, 'view': flux_view}
                                            )
        flux_views.update(views_aux)
        views_aux = {}
        for flux_view in flux_views_var:
            views_aux[f'{flux_view}_fluxnorm'] = \
                centering_and_normalization(flux_views_var[flux_view],
                                            0,
                                            flux_views_stats['min'][('local', 'global')['global' in flux_view]],
                                            report={'config': config, 'tce': tce, 'view': flux_view}
                                            )
        flux_views_var.update(views_aux)

        # create local flux view for detecting non-centered transits
        loc_flux_view_shift, _, _, _, _ = local_view(time,
                                                     flux,
                                                     tce['tce_period'],
                                                     tce['tce_duration'],
                                                     tce=tce,
                                                     normalize=False,
                                                     centering=False,
                                                     num_durations=5,
                                                     num_bins=config['num_bins_loc'],
                                                     bin_width_factor=config['bin_width_factor_loc'],
                                                     report={'config': config, 'tce': tce, 'view': 'local_flux_view'}
                                                     )
        flux_views['local_flux_view_shift'] = loc_flux_view_shift
        med_view = np.median(flux_views['local_flux_view_shift'])
        flux_views['local_flux_view_shift_fluxnorm'] = \
            centering_and_normalization(flux_views['local_flux_view_shift'],
                                        med_view,
                                        np.abs(np.min(flux_views['local_flux_view_shift'] - med_view)),
                                        report={'config': config, 'tce': tce, 'view': 'local_flux_view_shift'})

        # center by the weak secondary flux view median and normalize by the weak secondary flux view absolute minimum
        weak_secondary_flux_views = {
            'local_weak_secondary_view': loc_weak_secondary_view
        }
        weak_secondary_flux_views_var = {
            'local_weak_secondary_view': loc_weak_secondary_view_var
        }
        views_aux = {}
        for flux_view in weak_secondary_flux_views:
            weak_secondary_view_median = np.median(weak_secondary_flux_views[flux_view])
            norm_wks_factor_selfnorm = np.abs(np.min(weak_secondary_flux_views[flux_view] - weak_secondary_view_median))

            # normalize by self absolute minimum
            views_aux[f'{flux_view}_selfnorm'] = \
                centering_and_normalization(weak_secondary_flux_views[flux_view],
                                            weak_secondary_view_median,
                                            norm_wks_factor_selfnorm,
                                            report={'config': config, 'tce': tce, 'view': flux_view}
                                            )
            # normalize by flux absolute minimum
            norm_wks_factor_fluxnorm = flux_views_stats['min'][('local', 'global')['global' in flux_view]]
            views_aux[f'{flux_view}_fluxnorm'] = \
                centering_and_normalization(weak_secondary_flux_views[flux_view],
                                            flux_views_stats['median'][('local', 'global')['global' in flux_view]],
                                            norm_wks_factor_fluxnorm,
                                            report={'config': config, 'tce': tce, 'view': flux_view}
                                            )
            # normalize by max between flux absolute minimum and wks flux absolute minimum
            norm_wks_factor_fluxselfnorm = max(norm_wks_factor_fluxnorm, norm_wks_factor_selfnorm),
            views_aux[f'{flux_view}_max_flux-wks_norm'] = \
                centering_and_normalization(weak_secondary_flux_views[flux_view],
                                            flux_views_stats['median'][('local', 'global')['global' in flux_view]],
                                            norm_wks_factor_fluxselfnorm,
                                            report={'config': config, 'tce': tce, 'view': flux_view}
                                            )
        weak_secondary_flux_views.update(views_aux)

        views_aux = {}
        for flux_view in weak_secondary_flux_views_var:
            weak_secondary_view_median = np.median(weak_secondary_flux_views[flux_view])
            norm_wks_factor_selfnorm = np.abs(np.min(weak_secondary_flux_views[flux_view] -
                                                     weak_secondary_view_median))

            # normalize by self absolute minimum
            views_aux[f'{flux_view}_selfnorm'] = \
                centering_and_normalization(weak_secondary_flux_views_var[flux_view],
                                            0,
                                            norm_wks_factor_selfnorm,
                                            report={'config': config, 'tce': tce, 'view': flux_view}
                                            )
            # normalize by flux absolute minimum
            norm_wks_factor_fluxnorm = flux_views_stats['min'][('local', 'global')['global' in flux_view]]
            views_aux[f'{flux_view}_fluxnorm'] = \
                centering_and_normalization(weak_secondary_flux_views_var[flux_view],
                                            0,
                                            norm_wks_factor_fluxnorm,
                                            report={'config': config, 'tce': tce, 'view': flux_view}
                                            )
            # normalize by max between flux absolute minimum and wks flux absolute minimum
            norm_wks_factor_fluxselfnorm = max(norm_wks_factor_selfnorm, norm_wks_factor_fluxnorm)
            views_aux[f'{flux_view}_max_flux-wks_norm'] = \
                centering_and_normalization(weak_secondary_flux_views_var[flux_view],
                                            0,
                                            norm_wks_factor_fluxselfnorm,
                                            report={'config': config, 'tce': tce, 'view': flux_view}
                                            )
        weak_secondary_flux_views_var.update(views_aux)

        # get centroid views
        glob_centr_view, binned_time, glob_centr_view_var, inds_bin_nan['global_centroid'], bin_counts = \
            global_view(time_centroid_dist,
                        centroid_dist,
                        tce['tce_period'],
                        tce=tce,
                        centroid=True,
                        normalize=False,
                        centering=False,
                        num_bins=config['num_bins_glob'],
                        bin_width_factor=config['bin_width_factor_glob'],
                        report={'config': config, 'tce': tce,
                                'view': 'global_centr_view'},
                        tce_duration=tce['tce_duration']
                        )

        bin_counts[bin_counts == 0] = max(1, np.median(bin_counts))
        glob_centr_view_var /= np.sqrt(bin_counts)
        binned_timeseries['Global Centroid Offset Distance'] = (binned_time, glob_centr_view, glob_centr_view_var)

        loc_centr_view, binned_time, loc_centr_view_var, inds_bin_nan['local_centroid'], bin_counts = \
            local_view(time_centroid_dist,
                       centroid_dist,
                       tce['tce_period'],
                       tce['tce_duration'],
                       tce=tce,
                       centroid=True,
                       normalize=False,
                       centering=False,
                       num_durations=config['num_durations'],
                       num_bins=config['num_bins_loc'],
                       bin_width_factor=config['bin_width_factor_loc'],
                       report={'config': config, 'tce': tce,
                               'view': 'local_centr_view'}
                       )

        bin_counts[bin_counts == 0] = max(1, np.median(bin_counts))
        loc_centr_view_var /= np.sqrt(bin_counts)
        binned_timeseries['Local Centroid Offset Distance'] = (binned_time, loc_centr_view, loc_centr_view_var)

        # if plot_preprocessing_tce:
        #     utils_visualization.plot_centroids_views(glob_centr_view, loc_centr_view, tce, config,
        #                                              os.path.join(config['output_dir'], 'plots'),
        #                                              f'7_non-normalized_centroid_views_aug{tce["augmentation_idx"]}')

        # # get median statistics for global and local centroid views
        # centr_views_stats = {'median': {'global': np.median(glob_centr_view),
        #                                 'local': np.median(loc_centr_view)}
        #                      }
        #
        # # median centering, absolute and max normalization of the centroid views
        # # removes magnitude and direction of shift; describes shift relative to oot only
        # # defined in [0, 1]
        # glob_centr_view_medc_abs_maxn = centering_and_normalization(glob_centr_view,
        #                                                             centr_views_stats['median']['global'],
        #                                                             1,
        #                                                             # np.max(glob_centr_view - centr_views_stats['median']['global']),
        #                                                             report={'config': config, 'tce': tce,
        #                                                                     'view': 'global_centr_view_medcmaxn'}
        #                                                             )
        # glob_centr_view_medc_abs_maxn = np.abs(glob_centr_view_medc_abs_maxn)
        # normval = np.max(glob_centr_view_medc_abs_maxn)
        # if normval != 0:
        #     glob_centr_view_medc_abs_maxn /= normval
        # else:
        #     print('Dividing view by 0. Returning the non-normalized view global_centr_view_medcmaxn.')
        #     report_exclusion(config, tce,
        #                      f'Dividing view by 0. Returning the non-normalized view global_centr_view_medcmaxn.')
        #
        # loc_centr_view_medc_abs_maxn = centering_and_normalization(loc_centr_view,
        #                                                            centr_views_stats['median']['local'],
        #                                                            1,
        #                                                            # np.max(loc_centr_view - centr_views_stats['median']['local']),
        #                                                            report={'config': config, 'tce': tce,
        #                                                                    'view': 'local_centr_view_medcmaxn'}
        #                                                            )
        # loc_centr_view_medc_abs_maxn = np.abs(loc_centr_view_medc_abs_maxn)
        # normval = np.max(loc_centr_view_medc_abs_maxn)
        # if normval != 0:
        #     loc_centr_view_medc_abs_maxn /= normval
        # else:
        #     print('Dividing view by 0. Returning the non-normalized view local_centr_view_medcmaxn.')
        #     report_exclusion(config, tce,
        #                      f'Dividing view by 0. Returning the non-normalized view local_centr_view_medcmaxn.')
        #
        # # median centering, max absolute normalization of the centroid views
        # # removes magnitude of shift, but keeps direction of shift from oot relative to the target star
        # # closer to -1, closer to target star; closer to 0, closer to oot; closer to 1, further from target star
        # # defined in [-1, 1]
        # glob_centr_view_medc_maxabsn = centering_and_normalization(glob_centr_view,
        #                                                            centr_views_stats['median']['global'],
        #                                                            1,
        #                                                            # np.max(glob_centr_view - centr_views_stats['median']['global']),
        #                                                            report={'config': config, 'tce': tce,
        #                                                                    'view': 'global_centr_view_medcmaxn'}
        #                                                            )
        # normval = np.max(np.abs(glob_centr_view_medc_maxabsn))
        # if normval != 0:
        #     glob_centr_view_medc_maxabsn /= normval
        # else:
        #     print('Dividing view by 0. Returning the non-normalized view global_centr_view_medcmaxn_dir.')
        #     report_exclusion(config, tce,
        #                      f'Dividing view by 0. Returning the non-normalized view global_centr_view_medcmaxn_dir.')
        #
        # loc_centr_view_medc_maxabsn = centering_and_normalization(loc_centr_view,
        #                                                           centr_views_stats['median']['local'],
        #                                                           1,
        #                                                           # np.max(loc_centr_view - centr_views_stats['median']['local']),
        #                                                           report={'config': config, 'tce': tce,
        #                                                                   'view': 'local_centr_view_medcmaxn'}
        #                                                           )
        # normval = np.max(np.abs(loc_centr_view_medc_maxabsn))
        # if normval != 0:
        #     loc_centr_view_medc_maxabsn /= normval
        # else:
        #     print('Dividing view by 0. Returning the non-normalized view.')
        #     report_exclusion(config, tce,
        #                      f'Dividing view by 0. Returning the non-normalized view local_centr_view_medcmaxn_dir.')
        #
        # # median normalization for centroid views
        # # centroid is normalized by oot value, which is set to around 1
        # # the shift is scaled by oot, so bound is not defined [P/N, 1] or [1, P/N]
        # glob_centr_view_mn = centering_and_normalization(glob_centr_view, 0, centr_views_stats['median']['global'],
        #                                                  report={'config': config, 'tce': tce,
        #                                                          'view': 'global_centr_view_medn'}
        #                                                  )
        # loc_centr_view_mn = centering_and_normalization(loc_centr_view, 0, centr_views_stats['median']['local'],
        #                                                 report={'config': config, 'tce': tce,
        #                                                         'view': 'local_centr_view_medn'}
        #                                                 )

        # FDL centroid normalization
        # non-normalized views that need to be normalized using statistics from the training set
        glob_centr_fdl_view, _, _, _, _ = global_view(time_centroid_dist_fdl,
                                                      centroid_dist_fdl,
                                                      tce['tce_period'],
                                                      tce=tce,
                                                      centroid=True,
                                                      normalize=False,
                                                      centering=False,
                                                      num_bins=config['num_bins_glob'],
                                                      bin_width_factor=config['bin_width_factor_glob'],
                                                      report={'config': config, 'tce': tce,
                                                              'view': 'global_centr_fdl_view'},
                                                      tce_duration=tce['tce_duration']
                                                      )
        loc_centr_fdl_view, _, _, _, _ = local_view(time_centroid_dist_fdl,
                                                    centroid_dist_fdl,
                                                    tce['tce_period'],
                                                    tce['tce_duration'],
                                                    tce=tce,
                                                    centroid=True,
                                                    normalize=False,
                                                    centering=False,
                                                    num_durations=config['num_durations'],
                                                    num_bins=config['num_bins_loc'],
                                                    bin_width_factor=config['bin_width_factor_loc'],
                                                    report={'config': config, 'tce': tce,
                                                            'view': 'local_centr_fdl_view'}
                                                    )

        # if plot_preprocessing_tce:
        #     utils_visualization.plot_centroids_views(glob_centr_fdl_view, loc_centr_fdl_view, tce, config,
        #                                              os.path.join(config['output_dir'], 'plots'),
        #                                              '7_non-normalized_centroid_fdl_views_aug{}'.format(tce['augmentation_idx']))

        centr_views = {'global_centr_view': glob_centr_view,
                       'local_centr_view': loc_centr_view,
                       # 'global_centr_view_medcmaxn': glob_centr_view_medc_abs_maxn,
                       # 'local_centr_view_medcmaxn': loc_centr_view_medc_abs_maxn,
                       # 'global_centr_view_medn': glob_centr_view_mn,
                       # 'local_centr_view_medn': loc_centr_view_mn,
                       # 'global_centr_view_medcmaxn_dir': glob_centr_view_medc_maxabsn,
                       # 'local_centr_view_medcmaxn_dir': loc_centr_view_medc_maxabsn,
                       'global_centr_fdl_view': glob_centr_fdl_view,
                       'local_centr_fdl_view': loc_centr_fdl_view
                       }

        # adjust TESS centroids to Kepler by dividing by pixel scale factor
        if config['satellite'] == 'tess':
            centr_views['global_centr_view_adjscl'] = glob_centr_view / config['tess_to_kepler_px_scale_factor']
            centr_views['local_centr_view_adjscl'] = loc_centr_view / config['tess_to_kepler_px_scale_factor']

        centr_views_var = {'global_centr_view': glob_centr_view_var,
                           'local_centr_view': loc_centr_view_var,
                           }

        # initialize dictionary with the views that will be stored in the TFRecords example for this TCE
        views = {}
        views.update(flux_views)
        views.update(weak_secondary_flux_views)
        views.update(centr_views)

        views_var = {}
        views_var.update(flux_views_var)
        views_var.update(weak_secondary_flux_views_var)
        views_var.update(centr_views_var)

        if plot_preprocessing_tce:
            # CHANGE NUMBER OF VIEWS PLOTTED!!!
            views_to_plot = ['global_flux_view',
                             'local_flux_view',
                             'local_flux_odd_view',
                             'local_flux_even_view',
                             'global_flux_view_fluxnorm',
                             'local_flux_view_fluxnorm',
                             'local_flux_odd_view_fluxnorm',
                             'local_flux_even_view_fluxnorm',
                             'local_weak_secondary_view',
                             'local_weak_secondary_view_selfnorm',
                             'local_weak_secondary_view_max_flux-wks_norm',
                             'local_weak_secondary_view_fluxnorm',
                             'global_centr_view',
                             'local_centr_view',
                             'global_centr_fdl_view',
                             'local_centr_fdl_view'
                             ]
            views_plot = {view_name: view for view_name, view in views.items() if view_name in views_to_plot}
            utils_visualization.plot_all_views(views_plot, tce, config, (4, 4),
                                               os.path.join(config['output_dir'], 'plots'),
                                               f'8_final_views_aug{tce["augmentation_idx"]}',
                                               num_transits)

            utils_visualization.plot_phasefolded_and_binned(phasefolded_timeseries,
                                                            binned_timeseries,
                                                            tce,
                                                            config,
                                                            os.path.join(config['output_dir'], 'plots'),
                                                            f'8_phasefoldedbinned_timeseries_'
                                                            f'aug{tce["augmentation_idx"]}'
                                                            )

            views_var_plot = views_var
            utils_visualization.plot_all_views_var({view_name: view for view_name, view in views.items()
                                                    if view_name in views_var_plot},
                                                   views_var_plot,
                                                   tce, config, (4, 4),
                                                   os.path.join(config['output_dir'], 'plots'),
                                                   f'8_final_views_with_var_aug{tce["augmentation_idx"]}',
                                                   num_transits)
            # utils_visualization.plot_all_views(views_var_plot,
            #                                    tce, config, (4, 4),
            #                                    os.path.join(config['output_dir'], 'plots'),
            #                                    f'8_final_views_var_aug{tce["augmentation_idx"]}',
            #                                    num_transits)

    except Exception as e:
        report_exclusion(config, tce, 'Error when creating views', stderr=e)
        return None

    # check if time series have NaN values
    for view in views:
        if np.any(~np.isfinite(views[view])):  # at least one point is non-finite (infinite or NaN)
            report_exclusion(config, tce, f'View has at least one non-finite data point in view {view}.')
            return None

    for view in views_var:
        if np.any(~np.isfinite(views_var[view])):  # at least one point is non-finite (infinite or NaN)
            report_exclusion(config, tce, f'View has at least one non-finite data point in view {view}_var.')
            return None

    # set time series features in the example to be written to a TFRecord
    for view in views:
        example_util.set_float_feature(ex, view, views[view])
    for view in views_var:
        example_util.set_float_feature(ex, f'{view}_var', views[view])

    # set other features from the TCE table - diagnostic statistics, transit fits, stellar parameters...
    for name, value in tce.items():
        if name == 'Public Comment':  # can add special characters that are not serializable in the TFRecords
            continue
        try:
            example_util.set_feature(ex, name, [value])
        except:
            report_exclusion(config, tce, f'Could not set up this TCE table parameter: {name}.')
            print(name, value)

    # add number of transits per view
    for view in num_transits:
        example_util.set_int64_feature(ex, f'{view}_num_transits', [num_transits[view]])

    # add odd and even non timeseries features
    for field in odd_data:
        if field not in ['local_flux_view', 'local_flux_view_se', 'binned_time']:
            if 'se_oot' in field:  # normalize oot SE by normalization factor used for both odd and even flux views
                example_util.set_float_feature(ex, f'odd_{field}',
                                               [odd_data[field] / flux_views_stats['min']['local']])
                example_util.set_float_feature(ex, f'even_{field}',
                                               [even_data[field] / flux_views_stats['min']['local']])
            else:
                example_util.set_float_feature(ex, f'odd_{field}', [odd_data[field]])
                example_util.set_float_feature(ex, f'even_{field}', [even_data[field]])

    # TODO: add this feature to the TCE table or compute it here?
    # add ghost diagnostic statistic difference
    example_util.set_float_feature(ex, 'tce_cap_hap_stat_diff', [tce['tce_cap_stat'] - tce['tce_hap_stat']])

    # data for preprocessing table
    if config['satellite'] == 'tess':
        example_stats['sectors'] = tce['sectors']

    example_stats.update({f'num_transits_{view}': num_tr for view, num_tr in num_transits.items()})
    example_stats.update({f'odd_{key}': val for key, val in odd_data.items()
                          if key not in ['local_flux_view', 'local_flux_view_se', 'binned_time']})
    example_stats.update({f'even_{key}': val for key, val in even_data.items()
                          if key not in ['local_flux_view', 'local_flux_view_se', 'binned_time']})
    example_stats['odd_even_flag'] = odd_even_flag

    glob_centr_oot_bins = (binned_timeseries['Global Centroid Offset Distance'][0] < t_min_transit) | \
                          (binned_timeseries['Global Centroid Offset Distance'][0] > t_max_transit)
    loc_centr_it_bins = (binned_timeseries['Local Centroid Offset Distance'][0] >= t_min_transit) & \
                        (binned_timeseries['Local Centroid Offset Distance'][0] <= t_max_transit)
    example_stats['avg_oot_centroid_offset'] = np.median(glob_centr_view[glob_centr_oot_bins])
    example_stats['std_oot_centroid_offset'] = mad_std(glob_centr_view[glob_centr_oot_bins])
    example_stats['peak_centroid_offset'] = np.max(loc_centr_view[loc_centr_it_bins]) - \
                                            example_stats['avg_oot_centroid_offset']

    example_stats['mid_local_flux_shift'] = np.argmin(loc_flux_view) - int(config['num_bins_loc'] / 2)
    example_stats['mid_global_flux_shift'] = np.argmin(glob_flux_view) - int(config['num_bins_glob'] / 2)

    loc_flux_it_bins = (binned_timeseries['Local Flux'][0] >= t_min_transit) & \
                       (binned_timeseries['Local Flux'][0] <= t_max_transit)
    ind = np.argmin(views['local_flux_view'][loc_flux_it_bins])
    example_stats['transit_depth_hat'] = (1 - views['local_flux_view'][loc_flux_it_bins][ind]) * 1e6
    example_stats['transit_depth_hat_se'] = loc_flux_view_var[loc_flux_it_bins][ind] * 1e6

    loc_flux_odd_it_bins = (binned_timeseries['Local Odd Flux'][0] >= t_min_transit) & \
                           (binned_timeseries['Local Odd Flux'][0] <= t_max_transit)
    ind = np.argmin(views['local_flux_odd_view'][loc_flux_odd_it_bins])
    example_stats['transit_depth_odd_hat'] = (1 - views['local_flux_odd_view'][loc_flux_odd_it_bins][ind]) * 1e6
    example_stats['transit_depth_odd_hat_se'] = odd_data['local_flux_view_se'][loc_flux_odd_it_bins][ind] * 1e6

    loc_flux_even_it_bins = (binned_timeseries['Local Even Flux'][0] >= t_min_transit) & \
                           (binned_timeseries['Local Even Flux'][0] <= t_max_transit)
    ind = np.argmin(views['local_flux_even_view'][loc_flux_even_it_bins])
    example_stats['transit_depth_even_hat'] = (1 - views['local_flux_even_view'][loc_flux_even_it_bins][ind]) * 1e6
    example_stats['transit_depth_even_hat_se'] = even_data['local_flux_view_se'][loc_flux_even_it_bins][ind] * 1e6

    loc_flux_wks_it_bins = (binned_timeseries['Local Weak Secondary Flux'][0] >= t_min_transit) & \
                           (binned_timeseries['Local Weak Secondary Flux'][0] <= t_max_transit)
    ind = np.argmin(views['local_weak_secondary_view'][loc_flux_wks_it_bins])
    example_stats['wks_transit_depth_hat'] = (1 - views['local_weak_secondary_view'][loc_flux_wks_it_bins][ind]) * 1e6
    example_stats['wks_transit_depth_hat_se'] = loc_weak_secondary_view_var[loc_flux_wks_it_bins][ind] * 1e6

    # add number of missing bins for each view (excluding odd and even local views)
    for key in inds_bin_nan:
        example_stats[f'{key}_num_bins_it_nan'] = inds_bin_nan[key]['it'].sum()
        example_stats[f'{key}_num_bins_oot_nan'] = inds_bin_nan[key]['oot'].sum()

    # for param in ['avg_oot_centroid_offset', 'std_oot_centroid_offset', 'peak_centroid_offset', 'transit_depth_hat',
    #               'transit_depth_hat_std', 'transit_depth_odd_hat', 'transit_depth_odd_hat_se',
    #               'transit_depth_even_hat', 'transit_depth_even_hat_std', 'wks_transit_depth_hat',
    #               'wks_transit_depth_hat_std']:
    #     example_util.set_float_feature(ex, param, [example_stats[param]])
    #
    # for param in ['mid_local_flux_shift', 'mid_global_flux_shift']:
    #     example_util.set_int64_feature(ex, param, [example_stats[param]])

    return ex, example_stats
