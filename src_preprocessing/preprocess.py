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
import os
import numpy as np
import tensorflow as tf
from astropy.stats import mad_std
import lightkurve as lk

# local
from src_preprocessing import utils_visualization, kepler_io, tess_io
from src_preprocessing.light_curve import util
from src_preprocessing.tf_util import example_util
from src_preprocessing.third_party.kepler_spline import kepler_spline
from src_preprocessing.utils_centroid_preprocessing import (kepler_transform_pxcoordinates_mod13,
                                                            compute_centroid_distance,
                                                            correct_centroid_using_transit_depth)
from src_preprocessing.utils_ephemeris import create_binary_time_series, find_first_epoch_after_this_time
from src_preprocessing.utils_gapping import gap_other_tces
from src_preprocessing.utils_imputing import imputing_gaps
from src_preprocessing.utils_odd_even import create_odd_even_views, phase_fold_and_sort_light_curve_odd_even
from src_preprocessing.utils_preprocessing import (count_transits, remove_non_finite_values,
                                                   check_inputs_generate_example, remove_positive_outliers,
                                                   lininterp_transits, check_inputs)
from src_preprocessing.utils_preprocessing_io import report_exclusion
from src_preprocessing.detrend_timeseries import detrend_flux_using_spline, detrend_flux_using_sg_filter
from src_preprocessing.phase_fold_and_binning import (global_view, local_view, phase_fold_and_sort_light_curve,
                                                      phase_split_light_curve, centering_and_normalization,
                                                      generate_view_momentum_dump)

DEGREETOARCSEC = 3600


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
                report_exclusion(config, tce, f'No available lightcurve FITS files in {config["lc_data_dir"]} for '
                                              f'KIC {tce.target_id}')
                return None

        data, fits_files_not_read = kepler_io.read_kepler_light_curve(
            file_names,
            centroid_radec=not config['px_coordinates'],
            prefer_psfcentr=config['prefer_psfcentr'],
            light_curve_extension=config['light_curve_extension'],
            scramble_type=config['scramble_type'],
            cadence_no_quarters_tbl_fp=config['cadence_no_quarters_tbl_fp'],
            invert=config['invert'],
            dq_values_filter=config['dq_values_filter'],
            get_momentum_dump=config['get_momentum_dump'],
        )

        if len(fits_files_not_read) > 0:
            report_exclusion(config, tce, 'FITS files not read correctly')
            if len(fits_files_not_read) == len(file_names):
                return None

        return data

    else:  # TESS

        # get sectors for the run
        if '-' in tce['sector_run']:
            s_sector, e_sector = [int(sector) for sector in tce['sector_run'].split('-')]
        else:
            s_sector, e_sector = [int(tce['sector_run'])] * 2
        sectors = range(s_sector, e_sector + 1)
        # sectors = [int(sect) for sect in tce['sectors'].split(' ')]

        # get lc FITS files for the respective target star if it was observed for that modality in the given sectors
        if config['ffi_data']:
            file_names = tess_io.tess_ffi_filenames(config['lc_data_dir'], tce.target_id, sectors)
        else:
            file_names = tess_io.tess_filenames(config['lc_data_dir'], tce.target_id, sectors)

        if not file_names:
            if not config['omit_missing']:
                raise IOError(f'Failed to find .fits files in {config["lc_data_dir"]} for TESS ID {tce.target_id}')
            else:
                report_exclusion(config, tce, f'No available lightcurve FITS files in {config["lc_data_dir"]} for '
                                              f'TIC {tce.target_id}')
                return None

        fits_data, fits_files_not_read = tess_io.read_tess_light_curve(file_names,
                                                                       centroid_radec=not config['px_coordinates'],
                                                                       prefer_psfcentr=config['prefer_psfcentr'],
                                                                       light_curve_extension=config[
                                                                           'light_curve_extension'],
                                                                       get_momentum_dump=config['get_momentum_dump'],
                                                                       dq_values_filter=config['dq_values_filter'],
                                                                       )

        if len(fits_files_not_read) > 0:
            err_str = ''
            for el in fits_files_not_read:
                err_str += f'FITS file was not read correctly: {el}'
                report_exclusion(config, tce, err_str)
            if len(fits_files_not_read) == len(file_names):
                report_exclusion(config, tce, f'No FITS files were read in {config["lc_data_dir"]}.')
                return None

        return fits_data


def process_tce(tce, table, config):
    """ Preprocesses the timeseries and scalar features for the TCE and returns an Example proto.

    :param tce: row of the input TCE table.
    :param table: pandas DataFrame, ephemeris information on all TCEs (used when gapping)
    :param config: dict, holds preprocessing parameters

    return:
        a tensorflow.train.Example proto containing TCE features
    """

    # setting primary gap duration
    if 'tce_maxmesd' in tce:
        config['duration_gapped_primary'] = min((1 + 2 * config['gap_padding']) * tce['tce_duration'],
                                                np.abs(tce['tce_maxmesd']), tce['tce_period'])
        # setting secondary gap duration
        config['duration_gapped_secondary'] = min(max(0, tce['tce_period'] - config['duration_gapped_primary'] -
                                                      config['primary_buffer_time']),
                                                  config['gap_padding'] * tce['tce_duration'])
    else:
        config['duration_gapped_primary'] = min((1 + 2 * config['gap_padding']) * tce['tce_duration'],
                                                tce['tce_period'])
        config['duration_gapped_secondary'] = 0

    # set Savitzky-Golay window
    config['sg_win_len'] = int(config['sg_n_durations_win'] * tce['tce_duration'] * 24 *
                               config['sampling_rate_h'][f'{config["satellite"]}'])
    config['sg_win_len'] = config['sg_win_len'] if config['sg_win_len'] % 2 != 0 else config['sg_win_len'] + 1

    # check if preprocessing pipeline figures are saved for the TCE
    plot_preprocessing_tce = False  # False
    if np.random.random() < config['plot_prob']:
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

    # get cadence, flux and centroid data for the tce
    data = read_light_curve(tce, config)
    if data is None:
        report_exclusion(config, tce, 'Issue when reading data from the FITS file(s).')
        return None

    # update target position in FITS file with that from the TCE table
    if ~np.isnan(tce['ra']) and ~np.isnan(tce['dec']):
        data['target_position'] = [tce['ra'], tce['dec']]
    config['delta_dec'] = np.cos(data['target_position'][1] * np.pi / 180)

    data['errors'] = check_inputs(data)

    if config['satellite'] == 'kepler':
        add_info = {'quarter': data['quarter'], 'module': data['module']}
    else:
        add_info = {'sectors': data['sectors']}

    # # TODO: do the same thing for TESS
    # if config['get_denoised_centroids']:
    #     all_centroids, idxs_nan_centroids = get_denoised_centroids_kepler(tce.target_id, config['denoised_centroids_dir'])

    # if plot_preprocessing_tce:
    #     utils_visualization.plot_centroids(data['all_time'],
    #                                        data['all_centroids'],
    #                                        None,
    #                                        tce,
    #                                        config,
    #                                        os.path.join(config['output_dir'], 'plots'),
    #                                        f'1_raw_aug{tce["augmentation_idx"]}',
    #                                        add_info=add_info,
    #                                        target_position=None
    #                                        )
    #
    #     if not config['px_coordinates']:
    #         utils_visualization.plot_centroids(data['all_time'],
    #                                            data['all_centroids'],
    #                                            None,
    #                                            tce,
    #                                            config,
    #                                            os.path.join(config['output_dir'], 'plots'),
    #                                            f'1_raw_target_aug{tce["augmentation_idx"]}',
    #                                            add_info=add_info,
    #                                            target_position=data['target_position']
    #                                            )

    # utils_visualization.plot_centroids(data['all_time'], data['all_centroids_px'], None, tce, config,
    #                                    os.path.join(config['output_dir'], 'plots'),
    #                                    '1_raw_fdl_aug{}'.format(tce['augmentation_idx']),
    #                                    add_info=add_info,
    #                                    pxcoordinates=True)

    # # gap TCE to get weak secondary test time-series
    # # get gapped indices
    # data['gap_time_noprimary'] = None
    # # initialize arrays for gapped time series
    # timeseries_gapped = ['all_time_noprimary', 'all_flux_noprimary']
    # data['all_time_noprimary'] = [np.array(arr) for arr in data['all_time']]
    # data['all_flux_noprimary'] = [np.array(arr) for arr in data['all_flux']]
    # gapped_idxs_noprimary = gap_this_tce(data['all_time_noprimary'], tce, gap_pad=config['gap_padding_primary'])
    # if config['gap_imputed']:
    #     data['gap_time_noprimary'] = []
    # # set to NaN gapped cadences
    # for arr_i in range(len(gapped_idxs_noprimary)):
    #     if len(gapped_idxs_noprimary[arr_i]) == 0:
    #         continue
    #     for timeseries in timeseries_gapped:
    #         data[timeseries][arr_i][gapped_idxs_noprimary[arr_i]] = np.nan
    #     if config['gap_imputed'] > 0:
    #         data['gap_time_noprimary'].append(data['all_time'][arr_i][gapped_idxs_noprimary[arr_i]])

    # # non-gapped time array required for FDL centroid time-series
    # data['all_time_nongapped'] = [np.array(el) for el in data['all_time']]

    # gap cadences belonging to the transits of other TCEs in the same target star
    # FIXME: what if removes a whole quarter? need to adjust all_additional_info to it
    timeseries_gapped = ['all_time', 'all_flux', 'all_centroids']  # timeseries to be gapped
    data['gap_time'] = None
    if config['gapped']:
        gapped_idxs = gap_other_tces(data['all_time'],
                                     add_info,
                                     tce,
                                     table,
                                     config,
                                     gap_pad=config['gap_padding'],
                                     keep_overlap=config['gap_keep_overlap'])
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
    example_tce = generate_example_for_tce(data_processed, tce, config, plot_preprocessing_tce)

    return example_tce


def flux_preprocessing(all_time, all_flux, gap_time, tce, config, plot_preprocessing_tce):
    """ Preprocess the flux time series.

    :param all_time: list of NumPy arrays, timestamps
    :param all_flux: list of NumPy arrays, flux time series
    :param gap_time: list of NumPy arrays, gapped timestamps
    :param tce: pandas Series, TCE parameters
    :param config: dict, preprocessing parameters
    :param plot_preprocessing_tce: bool, set to True to plot figures related to different preprocessing steps
    :return:
        time: NumPy array, timestamps for preprocessed flux time series
        detrended_flux: NumPy array, preprocessed flux timeseries
    """

    time_arrs, flux_arrs = [np.array(el) for el in all_time], [np.array(el) for el in all_flux]

    # remove non-finite values
    time_arrs, flux_arrs = remove_non_finite_values([time_arrs, flux_arrs])

    # # add gap after and before transit based on transit duration
    # if 'tce_maxmesd' in tce:
    #     duration_gapped = min((1 + 2 * config['gap_padding']) * tce['tce_duration'], np.abs(tce['tce_maxmesd']),
    #                           tce['tce_period'])
    # else:
    #     duration_gapped = min((1 + 2 * config['gap_padding']) * tce['tce_duration'], tce['tce_period'])

    # get epoch of first transit for each time array
    first_transit_time_all = [find_first_epoch_after_this_time(tce['tce_time0bk'], tce['tce_period'], time[0])
                              for time in time_arrs]

    # create binary time series for each time array in which in-transit points are labeled as 1's, otherwise as 0's
    binary_time_all = [create_binary_time_series(time, first_transit_time, config['duration_gapped_primary'],
                                                 tce['tce_period'])
                       for first_transit_time, time in zip(first_transit_time_all, time_arrs)]

    if plot_preprocessing_tce:
        utils_visualization.plot_intransit_binary_timeseries(time_arrs, flux_arrs, binary_time_all, tce,
                                                             config['plot_dir'],
                                                             f'2_intransit_flux_binary_timeseries_aug{tce["augmentation_idx"]}')

    # gap secondary transits
    # duration_gapped_secondary = duration_gapped - tce['tce_period'] - 30 / config['sampling_rate_h'][config['satellite']] / 24
    # get epoch of first transit for each time array
    first_transit_time_secondary_all = [find_first_epoch_after_this_time(tce['tce_time0bk'] + tce['tce_maxmesd'],
                                                                         tce['tce_period'], time[0])
                                        for time in time_arrs]
    # create binary time series for each time array in which in-transit points are labeled as 1's, otherwise as 0's
    binary_time_secondary_all = [create_binary_time_series(time, first_transit_time_secondary,
                                                           config['duration_gapped_secondary'],
                                                           tce['tce_period'])
                                 for first_transit_time_secondary, time in
                                 zip(first_transit_time_secondary_all, time_arrs)]
    # set secondary in-transit cadences to np.nan
    flux_arrs = [np.where(binary_time_secondary, np.nan, flux_arr)
                 for flux_arr, binary_time_secondary in zip(flux_arrs, binary_time_secondary_all)]

    # remove non-finite values
    time_arrs, flux_arrs = remove_non_finite_values([time_arrs, flux_arrs])

    flux = np.concatenate(flux_arrs)

    # detrend flux
    if config['detrending_method'] == 'spline':

        time, detrended_flux, trend, res_flux, flux_lininterp = detrend_flux_using_spline(flux_arrs, time_arrs,
                                                                                          binary_time_all, config)

    elif config['detrending_method'] == 'savitzky-golay':
        # convert data to lightcurve object
        time, flux = np.concatenate(time_arrs), np.concatenate(flux_arrs)
        lc = lk.LightCurve(data={'time': np.array(time), 'flux': np.array(flux)})
        # mask in-transit cadences
        mask_in_transit = lc.create_transit_mask(tce['tce_period'], tce['tce_time0bk'],
                                                 config['duration_gapped_primary'])

        time, detrended_flux, trend, res_flux = detrend_flux_using_sg_filter(lc, mask_in_transit,
                                                                             config['sg_win_len'],
                                                                             config['sg_sigma'],
                                                                             config['sg_max_poly_order'],
                                                                             config['sg_penalty_weight'],
                                                                             config['sg_break_tolerance'])
        flux_lininterp = None
    else:
        raise ValueError(f'Detrending method not recognized: {config["detrending_method"]}')

    if plot_preprocessing_tce:
        utils_visualization.plot_flux_detrend(time,
                                              flux,
                                              trend,
                                              detrended_flux,
                                              tce,
                                              config['plot_dir'],
                                              f'3_detrendedflux_aug{tce["augmentation_idx"]}',
                                              flux_interp=flux_lininterp)

        utils_visualization.plot_residual(time,
                                          res_flux,
                                          tce,
                                          config['plot_dir'],
                                          f'3_residual_flux_aug{tce["augmentation_idx"]}',
                                          )

    # # impute the time series with Gaussian noise based on global estimates of median and std
    # if config['gap_imputed']:
    #     time_arrs, flux_arrs = imputing_gaps(time_arrs, flux_arrs, gap_time)

    return time, detrended_flux


def weak_secondary_flux_preprocessing(all_time, all_flux_noprimary, gap_time, tce, config, plot_preprocessing_tce):
    """ Preprocess the weak secondary flux timeseries.

    :param all_time: list of NumPy arrays, timestamps
    :param all_flux_noprimary: list of NumPy arrays, weak secondary flux time series
    :param gap_time: list of NumPy arrays, gapped timestamps
    :param tce: Pandas Series, TCE parameters
    :param config: dict, preprocessing parameters
    :param plot_preprocessing_tce: bool, set to True to plot figures related to different preprocessing steps
    :return:
        time: NumPy array, timestamps for preprocessed flux time series
        detrended_flux: NumPy array, preprocessed weak secondary flux timeseries
    """

    time_arrs, flux_arrs = [np.array(el) for el in all_time], [np.array(el) for el in all_flux_noprimary]

    # remove non-finite values
    time_arrs, flux_arrs = remove_non_finite_values([time_arrs, flux_arrs])

    # # add gap after and before transit based on transit duration
    # duration_gapped = min((1 + 2 * config['gap_padding']) * tce['tce_duration'], np.abs(tce['tce_maxmesd']),
    #                       tce['tce_period'])

    first_transit_time_all = [find_first_epoch_after_this_time(tce['tce_time0bk'] + tce['tce_maxmesd'],
                                                               tce['tce_period'], time[0])
                              for time in time_arrs]
    binary_time_all = [create_binary_time_series(time, first_transit_time, config['duration_gapped_primary'],
                                                 tce['tce_period'])
                       for first_transit_time, time in zip(first_transit_time_all, time_arrs)]

    if plot_preprocessing_tce:
        utils_visualization.plot_intransit_binary_timeseries(time_arrs, flux_arrs, binary_time_all, tce,
                                                             config['plot_dir'],
                                                             '2_intransit_wksflux_binary_timeseries')
    # gap primary transits
    # get epoch of first transit for each time array
    first_transit_time_primary_all = [find_first_epoch_after_this_time(tce['tce_time0bk'], tce['tce_period'], time[0])
                                      for time in time_arrs]
    # create binary time series for each time array in which in-transit points are labeled as 1's, otherwise as 0's
    binary_time_primary_all = [create_binary_time_series(time, first_transit_time_secondary,
                                                         config['duration_gapped_secondary'],
                                                         tce['tce_period'])
                               for first_transit_time_secondary, time in zip(first_transit_time_primary_all, time_arrs)]
    # set primary in-transit cadences to np.nan
    flux_arrs = [np.where(binary_time_primary, np.nan, flux_arr)
                 for flux_arr, binary_time_primary in zip(flux_arrs, binary_time_primary_all)]

    # remove non-finite values
    time_arrs, flux_arrs = remove_non_finite_values([time_arrs, flux_arrs])

    flux = np.concatenate(flux_arrs)

    # detrend flux
    if config['detrending_method'] == 'spline':
        time, detrended_flux, trend, res_flux, flux_lininterp = detrend_flux_using_spline(flux_arrs, time_arrs,
                                                                                          binary_time_all, config)

    elif config['detrending_method'] == 'savitzky-golay':
        # convert data to lightcurve object
        time, flux = np.concatenate(time_arrs), np.concatenate(flux_arrs)
        lc = lk.LightCurve(data={'time': np.array(time), 'flux': np.array(flux)})
        # mask in-transit cadences for secondary
        mask_in_transit = lc.create_transit_mask(tce['tce_period'], tce['tce_time0bk'] + tce['tce_maxmesd'],
                                                 config['duration_gapped_primary'])

        time, detrended_flux, trend, _ = detrend_flux_using_sg_filter(lc, mask_in_transit,
                                                                             config['sg_win_len'],
                                                                             config['sg_sigma'],
                                                                             config['sg_max_poly_order'],
                                                                             config['sg_penalty_weight'],
                                                                             config['sg_break_tolerance'])
        flux_lininterp = None
    else:
        raise ValueError(f'Detrending method not recognized: {config["detrending_method"]}')

    if plot_preprocessing_tce:
        utils_visualization.plot_flux_detrend(time,
                                              flux,
                                              trend,
                                              detrended_flux,
                                              tce,
                                              config['plot_dir'],
                                              f'3_detrendedwksflux_aug{tce["augmentation_idx"]}',
                                              flux_interp=flux_lininterp)

    # # impute the time series with Gaussian noise based on global estimates of median and std
    # if config['gap_imputed']:
    #     time_arrs, flux_arrs = imputing_gaps(time_arrs, flux_arrs, gap_time)

    return time, detrended_flux


def centroid_preprocessing(all_time, all_centroids, target_position, add_info, gap_time, tce, config,
                           plot_preprocessing_tce):
    """ Preprocess the centroid timeseries.

    :param all_time: list of NumPy arrays, timestamps
    :param all_centroids: dictionary for the two centroid coordinates coded as 'x' and 'y'. Each key maps to a list of
    NumPy arrays for the respective centroid coordinate time series
    # :param avg_centroid_oot: dictionary for the two centroid coordinates coded as 'x' and 'y'. Each key maps to the
    # estimate of the average out-of-transit centroid
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

    time_arrs, centroid_dict = [np.array(el) for el in all_time], \
        {coord: [np.array(el) for el in centroid_arrs] for coord, centroid_arrs in all_centroids.items()}

    time_arrs, centroid_dict['x'], centroid_dict['y'] = remove_non_finite_values([time_arrs,
                                                                                  centroid_dict['x'],
                                                                                  centroid_dict['y']])

    # pixel coordinate transformation for targets on module 13 for Kepler
    if config['px_coordinates'] and config['satellite'] == 'kepler':
        if add_info['module'][0] == 13:
            centroid_dict = kepler_transform_pxcoordinates_mod13(centroid_dict, add_info)

    # # add gap after and before transit based on transit duration
    # if 'tce_maxmesd' in tce:
    #     duration_gapped = min((1 + 2 * config['gap_padding']) * tce['tce_duration'], np.abs(tce['tce_maxmesd']),
    #                           tce['tce_period'])
    # else:
    #     duration_gapped = min((1 + 2 * config['gap_padding']) * tce['tce_duration'], tce['tce_period'])

    # gap primary in-transit cadences
    # get epoch of first transit for each time array
    first_transit_time_all = [find_first_epoch_after_this_time(tce['tce_time0bk'], tce['tce_period'], time[0])
                              for time in time_arrs]
    # create in-transit binary time series for each time array
    binary_time_all = [create_binary_time_series(time, first_transit_time, config['duration_gapped_primary'],
                                                 tce['tce_period'])
                       for first_transit_time, time in zip(first_transit_time_all, time_arrs)]

    # gap secondary in-transit cadences
    # get epoch of first transit for each time array
    first_transit_time_all = [find_first_epoch_after_this_time(tce['tce_time0bk'] + tce['tce_maxmesd'],
                                                               tce['tce_period'], time[0])
                              for time in time_arrs]
    # create in-transit binary time series for each time array
    binary_time_secondary_all = [create_binary_time_series(time, first_transit_time,
                                                           config['duration_gapped_secondary'], tce['tce_period'])
                                 for first_transit_time, time in zip(first_transit_time_all, time_arrs)]

    # merge both in-transit binary timeseries for primary and secondary in-transit cadences
    binary_time_all = [np.logical_or(binary_time_arr_primary, binary_time_arr_secondary)
                       for binary_time_arr_primary, binary_time_arr_secondary
                       in zip(binary_time_all, binary_time_secondary_all)]

    # get out-of-transit indices for the centroid time series
    centroid_oot = {coord: [centroids[~binary_time] for binary_time, centroids
                            in zip(binary_time_all, centroid_dict[coord])] for coord in centroid_dict}
    # estimate average out-of-transit centroid as the median
    avg_centroid_oot = {coord: np.nanmedian(np.concatenate(centroid_oot[coord])) for coord in centroid_oot}
    # if there is no valid out-of-transit cadences, use median of the whole centroid array
    avg_centroid_oot = {coord: np.nanmedian(np.concatenate(centroid_dict[coord])) if np.isnan(avg_centroid_oot_coord)
                        else avg_centroid_oot_coord for coord, avg_centroid_oot_coord in avg_centroid_oot.items()}

    if plot_preprocessing_tce:
        utils_visualization.plot_intransit_binary_timeseries(time_arrs,
                                                             centroid_dict,
                                                             binary_time_all,
                                                             tce,
                                                             config['plot_dir'],
                                                             f'2_intransit_centroids_binary_timeseries_aug{tce["augmentation_idx"]}',
                                                             centroid=True)

    time_arrs, centroid_dict['x'], centroid_dict['y'] = remove_non_finite_values([time_arrs,
                                                                                  centroid_dict['x'],
                                                                                  centroid_dict['y']])

    detrended_centroid_dict = {centroid_coord: {'detrended': None, 'trend': None}
                               for centroid_coord in centroid_dict.keys()}
    for centroid_coord, centroid_coord_data in centroid_dict.items():

        # detrend centroid
        if config['detrending_method'] == 'spline':
            time, detrended_centroid, trend, res_centroid, centroid_lininterp = (
                detrend_flux_using_spline(centroid_coord_data, time_arrs, binary_time_all, config))
            detrended_centroid_dict[centroid_coord]['linear_interp'] = centroid_lininterp

        elif config['detrending_method'] == 'savitzky-golay':
            # convert data to lightcurve object
            time, centroid_arr = np.concatenate(time_arrs), np.concatenate(centroid_coord_data)
            lc = lk.LightCurve(data={'time': np.array(time), 'flux': np.array(centroid_arr)})
            # mask in-transit cadences
            mask_in_transit = lc.create_transit_mask(tce['tce_period'], tce['tce_time0bk'],
                                                     config['duration_gapped_primary'])

            _, detrended_centroid, trend, res_centroid = detrend_flux_using_sg_filter(lc,
                                                                                      mask_in_transit,
                                                                                      config['sg_win_len'],
                                                                                      config['sg_sigma'],
                                                                                      config['sg_max_poly_order'],
                                                                                      config['sg_penalty_weight'],
                                                                                      config['sg_break_tolerance'],
                                                                                      )
        else:
            raise ValueError(f'Detrending method not recognized: {config["detrending_method"]}')

        detrended_centroid_dict[centroid_coord]['detrended'] = detrended_centroid
        detrended_centroid_dict[centroid_coord]['trend'] = trend
        detrended_centroid_dict[centroid_coord]['residual'] = res_centroid

    time_centroid, centroid_dict_concat = (np.concatenate(time_arrs),
                                           {coord: np.concatenate(centroid_coord) for coord, centroid_coord
                                            in centroid_dict.items()})

    # recover the original central tendency of the centroid by multiplying the detrended centroid by the average
    # centroid
    for centroid_coord in detrended_centroid_dict:
        detrended_centroid_dict[centroid_coord]['detrended'] *= avg_centroid_oot[centroid_coord]

    if plot_preprocessing_tce:
        utils_visualization.plot_centroids(time_centroid,
                                           centroid_dict_concat,
                                           detrended_centroid_dict,
                                           tce,
                                           config,
                                           config['plot_dir'],
                                           f'3_detrendedcentroids_aug{tce["augmentation_idx"]}',
                                           config['px_coordinates'],
                                           target_position,
                                           config['delta_dec'],
                                           )

        for coord, centroid_coord_data in detrended_centroid_dict.items():
            utils_visualization.plot_residual(time_centroid,
                                              centroid_coord_data['residual'],
                                              tce,
                                              config['plot_dir'],
                                              f'3_residual_centroid{coord}_aug{tce["augmentation_idx"]}',
                                              )

    # # set outliers to zero using Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    # q25_75 = {'x': {'q25': np.percentile(np.concatenate(centroid_dict['x']), 25),
    #                 'q75': np.percentile(np.concatenate(centroid_dict['x']), 75)},
    #           'y': {'q25': np.percentile(np.concatenate(centroid_dict['y']), 25),
    #                 'q75': np.percentile(np.concatenate(centroid_dict['y']), 75)}
    #           }
    # iqr = {'x': q25_75['x']['q75'] - q25_75['x']['q25'],
    #        'y': q25_75['y']['q75'] - q25_75['y']['q25']}
    # outlier_thr = 1.5
    # for coord in centroid_dict:
    #     for i in range(len(centroid_dict[coord])):
    #         centroid_dict[coord][i][np.where(centroid_dict[coord][i] > q25_75[coord]['q75'] + outlier_thr * iqr[coord])] = avg_centroid_oot[coord]
    #         centroid_dict[coord][i][np.where(centroid_dict[coord][i] < q25_75[coord]['q25'] - outlier_thr * iqr[coord])] = avg_centroid_oot[coord]

    # avg_centroid_oot = {coord: np.median(np.concatenate(centroid_dict[coord])) for coord in centroid_dict}

    # compute the new average oot after the spline fitting and normalization
    # TODO: how to compute the average oot? mean, median, other...
    # if plot_preprocessing_tce:
    #     utils_visualization.plot_centroids_it_oot(time_arrs,
    #                                               binary_time_all,
    #                                               centroid_dict,
    #                                               avg_centroid_oot,
    #                                               target_position,
    #                                               tce, config,
    #                                               os.path.join(config['output_dir'], 'plots'),
    #                                               f'4_centroidtimeseries_it-ot-target_aug{tce["augmentation_idx"]}',
    #                                               target_center=True)

    transit_depth = tce['tce_depth'] + 1  # avoid zero transit depth
    corrected_centroids = correct_centroid_using_transit_depth(detrended_centroid_dict['x']['detrended'],
                                                               detrended_centroid_dict['y']['detrended'],
                                                               transit_depth,
                                                               avg_centroid_oot)
    # corrected_centroids = {'x': detrended_centroid_dict['x']['detrended'],
    #                        'y': detrended_centroid_dict['y']['detrended']}

    if plot_preprocessing_tce:
        utils_visualization.plot_corrected_centroids(time_centroid,
                                                     corrected_centroids,
                                                     avg_centroid_oot,
                                                     tce,
                                                     config,
                                                     config['plot_dir'],
                                                     f'4_correctedcentroids_aug{tce["augmentation_idx"]}',
                                                     config['px_coordinates'],
                                                     target_position=target_position,
                                                     delta_dec=config['delta_dec']
                                                     )

    # compute distance of centroid to target position
    centroid_dist = compute_centroid_distance(corrected_centroids, target_position, config['delta_dec'])

    # convert from degree to arcsec
    if not config['px_coordinates']:
        centroid_dist *= DEGREETOARCSEC

    if plot_preprocessing_tce:
        utils_visualization.plot_dist_centroids(time_centroid,
                                                centroid_dist,
                                                tce,
                                                config,
                                                config['plot_dir'],
                                                f'5_distcentr_aug{tce["augmentation_idx"]}')

    # # impute the time series with Gaussian noise based on global estimates of median and std
    # if config['gap_imputed']:
    #     time, all_centroid_dist = imputing_gaps(time_arrs, all_centroid_dist, gap_time)

    return time_centroid, centroid_dist


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

    time_arrs, centroid_dict = [np.array(el) for el in all_time], \
        {coord: [np.array(el) for el in centroid_arrs] for coord, centroid_arrs in all_centroids.items()}

    time_arrs, centroid_dict['x'], centroid_dict['y'] = remove_non_finite_values([time_arrs,
                                                                                  centroid_dict['x'],
                                                                                  centroid_dict['y']])

    time_arrs, centroid_dict, add_info = util.split(time_arrs, centroid_dict,
                                                    gap_width=config['gapWidth'],
                                                    centroid=True,
                                                    add_info=add_info)

    if config['satellite'] == 'kepler' and add_info['quarter'][0] == 13:
        centroid_dict = kepler_transform_pxcoordinates_mod13(centroid_dict, add_info)

    first_transit_time_all = [find_first_epoch_after_this_time(tce['tce_time0bk'], tce['tce_period'], time[0])
                              for time in time_arrs]

    if 'tce_maxmesd' in tce:
        duration_gapped = min((1 + 2 * config['gap_padding']) * tce['tce_duration'], np.abs(tce['tce_maxmesd']),
                              tce['tce_period'])
    else:
        duration_gapped = min((1 + 2 * config['gap_padding']) * tce['tce_duration'], tce['tce_period'])

    binary_time_all = [create_binary_time_series(time, first_transit_time, duration_gapped, tce['tce_period'])
                       for first_transit_time, time in zip(first_transit_time_all, time_arrs)]

    all_centroids_lininterp = lininterp_transits(centroid_dict, binary_time_all, centroid=True)

    spline_centroid = {coord: kepler_spline.fit_kepler_spline(time_arrs,
                                                              all_centroids_lininterp[coord],
                                                              verbose=False)[0] for coord in all_centroids_lininterp}

    finite_i_centroid = [np.logical_and(np.isfinite(spline_centroid['x'][i]), np.isfinite(spline_centroid['y'][i]))
                         for i in range(len(spline_centroid['x']))]

    # all_centroids = {coord: [all_centroids[coord][i][finite_i_centroid[i]] /
    #                          spline_centroid[coord][i][finite_i_centroid[i]]
    #                          for i in range(len(spline_centroid[coord])) if len(finite_i_centroid[i] > 0)]
    #                  for coord in all_centroids}
    centroid_dict = {coord: [centroid_dict[coord][i][finite_i_centroid[i]] -
                             spline_centroid[coord][i][finite_i_centroid[i]]
                             for i in range(len(spline_centroid[coord])) if len(finite_i_centroid[i] > 0)]
                     for coord in centroid_dict}

    time_arrs = [time_arrs[i][finite_i_centroid[i]] for i in range(len(time_arrs)) if len(finite_i_centroid[i]) > 0]

    all_centroid_dist = [np.sqrt(np.square(centroid_dict['x'][i]) + np.square(centroid_dict['y'][i]))
                         for i in range(len(centroid_dict['x']))]

    if plot_preprocessing_tce:
        utils_visualization.plot_dist_centroids(time_arrs,
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
        time_arrs, all_centroid_dist = imputing_gaps(time_arrs, all_centroid_dist, gap_time)

    time = np.concatenate(time_arrs)
    centroid_dist = np.concatenate(all_centroid_dist)

    return time, centroid_dist


def process_light_curve(data, config, tce, plot_preprocessing_tce=False):
    """ Preprocesses different timeseries (e.g., detrending) such as flux and centroid motion.

    Args:
      data: dictionary containing data related to the time-series to be preprocessed before generating views
      config: dict, holds preprocessing parameters
      tce: Pandas Series, row of the TCE table for the TCE that is being processed; it contains data such as the
      ephemerides
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
    except Exception as e:
        report_exclusion(config, tce, 'Error when preprocessing flux.', stderr=e)
        time, flux = None, None

    # preprocess weak secondary flux time series
    try:
        time_wksecondaryflux, wksecondaryflux = weak_secondary_flux_preprocessing(data['all_time'],
                                                                                  data['all_flux'],
                                                                                  [],  # data['gap_time_noprimary'],
                                                                                  tce,
                                                                                  config,
                                                                                  plot_preprocessing_tce)
    except Exception as e:
        report_exclusion(config, tce, 'Error when preprocessing secondary flux.', stderr=e)
        time_wksecondaryflux, wksecondaryflux = None, None

    # preprocess centroid time series
    try:
        time_centroid, centroid_dist = centroid_preprocessing(data['all_time'],
                                                              data['all_centroids'],
                                                              data['target_position'],
                                                              add_info_centr,
                                                              data['gap_time'],
                                                              tce,
                                                              config,
                                                              plot_preprocessing_tce)
    except Exception as e:
        report_exclusion(config, tce, 'Error when preprocessing centroids.', stderr=e)
        time_centroid, centroid_dist = None, None

    # # preprocess FDL centroid time series
    # try:
    #     time_centroidFDL, centroid_distFDL = centroidFDL_preprocessing(data['all_time_nongapped'],
    #                                                                    data['all_centroids_px'],
    #                                                                    add_info_centr,
    #                                                                    [],
    #                                                                    tce,
    #                                                                    config,
    #                                                                    plot_preprocessing_tce)
    # except:
    #     time_centroidFDL, centroid_distFDL = None, None

    if config['get_momentum_dump']:
        time_momentum_dump, momentum_dump = remove_non_finite_values([data['time_momentum_dump'],
                                                                      data['momentum_dump']])
        momentum_dump, time_momentum_dump = np.concatenate(momentum_dump), np.concatenate(time_momentum_dump)

        if plot_preprocessing_tce:
            utils_visualization.plot_momentum_dump_timeseries(time_momentum_dump, momentum_dump, tce,
                                                              os.path.join(config['output_dir'], 'plots'),
                                                              '2_momentum_dump_timeseries')

    # dictionary with preprocessed time series used to generate views
    data_views = {}
    data_views['time'] = time
    data_views['flux'] = flux
    data_views['time_centroid_dist'] = time_centroid
    data_views['centroid_dist'] = centroid_dist
    data_views['time_wksecondaryflux'] = time_wksecondaryflux
    data_views['wksecondaryflux'] = wksecondaryflux
    # data_views['time_centroid_distFDL'] = time_centroidFDL
    # data_views['centroid_distFDL'] = centroid_distFDL
    data_views['errors'] = data['errors']
    if config['satellite'] == 'kepler':
        data_views['quarter_timestamps'] = data['quarter_timestamps']
    if config['get_momentum_dump']:
        data_views['momentum_dump'] = momentum_dump
        data_views['time_momentum_dump'] = time_momentum_dump

    return data_views


def create_example_stats(binned_timeseries, views, views_var, odd_data, even_data, odd_even_flag, num_transits,
                         inds_bin_nan, t_min_transit, t_max_transit, config):
    """ Computes several summary statistics based on the preprocessed data for a given TCE.

    Args:
        binned_timeseries: dict, binned timeseries
        views: dict, views
        views_var: dict, variability views
        odd_data: dict, odd data
        even_data: dict, even data
        odd_even_flag: str, odd and even flag
        num_transits: dict, number of transits per timeseries
        inds_bin_nan: dict, indices of bins with missing values per views
        t_min_transit: float, left edge in-transit phase time
        t_max_transit: float, right edge in-transit phase time
        config: dict, preprocessing parameters for the run

    Returns: dict, with summary statistics for the example

    """

    # initialize dictionary with data for the preprocessing table
    example_stats = {}

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
    example_stats['avg_oot_centroid_offset'] = np.median(views['global_centr_view'][glob_centr_oot_bins])
    example_stats['std_oot_centroid_offset'] = mad_std(views['global_centr_view'][glob_centr_oot_bins])
    example_stats['peak_centroid_offset'] = np.max(views['local_centr_view'][loc_centr_it_bins]) - \
                                            example_stats['avg_oot_centroid_offset']

    example_stats['mid_local_flux_shift'] = np.argmin(views['local_centr_view']) - int(config['num_bins_loc'] / 2)
    example_stats['mid_global_flux_shift'] = np.argmin(['global_centr_view']) - int(config['num_bins_glob'] / 2)

    loc_flux_it_bins = (binned_timeseries['Local Flux'][0] >= t_min_transit) & \
                       (binned_timeseries['Local Flux'][0] <= t_max_transit)
    ind = np.argmin(views['local_flux_view'][loc_flux_it_bins])
    example_stats['transit_depth_hat'] = (1 - views['local_flux_view'][loc_flux_it_bins][ind]) * 1e6
    example_stats['transit_depth_hat_se'] = views_var['local_centr_view'][loc_flux_it_bins][ind] * 1e6

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
    example_stats['wks_transit_depth_hat'] = (1 - views['local_weak_secondary_view'][loc_flux_wks_it_bins][
        ind]) * 1e6
    example_stats['wks_transit_depth_hat_se'] = views_var['local_weak_secondary_view'][loc_flux_wks_it_bins][ind] * 1e6

    # add number of missing bins for each view (excluding odd and even local views)
    for key in inds_bin_nan:
        example_stats[f'{key}_num_bins_it_nan'] = inds_bin_nan[key]['it'].sum()
        example_stats[f'{key}_num_bins_oot_nan'] = inds_bin_nan[key]['oot'].sum()

    return example_stats


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

    data = check_inputs_generate_example(data, tce, config)

    # check if data for all views is valid after preprocessing the raw time series
    for key, val in data.items():
        if val is None:
            report_exclusion(config, tce, f'No data for {key} before creating the views.')
            return None

    # initialize number of transits per time series dictionary
    num_transits = {}
    # initialize dictionary with number of empty bins per view
    inds_bin_nan = {}

    # time interval for the transit
    t_min_transit, t_max_transit = max(-tce['tce_period'] / 2, -tce['tce_duration'] / 2), \
        min(tce['tce_period'] / 2, tce['tce_duration'] / 2)

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

    # phase folding for flux time series to generate phases separately
    time_split, flux_split, n_phases_split, _ = phase_split_light_curve(
        data['time'],
        data['flux'],
        tce['tce_period'],
        tce['tce_time0bk'],
        tce['tce_duration'],
        config['n_max_phases'],
        config['keep_odd_even_order'],
        config['frac_it_cadences_thr'],
        config['sampling_rate_h'][config['satellite']],
        config['phase_extend_method'],
        quarter_timestamps=data['quarter_timestamps'] if config['satellite'] == 'kepler' and config['quarter_sampling']
        else None
    )

    if n_phases_split < config['min_n_phases']:
        report_exclusion(config, tce, f'Only found {n_phases_split} phase, need at least '
                                      f'{config["min_n_phases"]} to create example.')
        return None

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

    # # same for FDL centroid time-series
    # # phase folding for flux and centroid time series
    # time_centroid_dist_fdl, centroid_dist_fdl, \
    # num_transits['centroid_fdl'] = phase_fold_and_sort_light_curve(data['time_centroid_distFDL'],
    #                                                                data['centroid_distFDL'],
    #                                                                tce['tce_period'],
    #                                                                tce['tce_time0bk'],
    #                                                                augmentation=False)
    # num_transits['centroid_fdl'] = count_transits(data['time_centroid_distFDL'],
    #                                               tce['tce_period'],
    #                                               tce['tce_time0bk'],
    #                                               tce['tce_duration'])

    # phase folding for flux time series
    if config['get_momentum_dump']:
        time_momentum_dump, momentum_dump, _ = phase_fold_and_sort_light_curve(data['time_momentum_dump'],
                                                                               data['momentum_dump'],
                                                                               tce['tce_period'],
                                                                               tce['tce_time0bk'],
                                                                               augmentation=False)

    phasefolded_timeseries = {'Flux': (time, flux),
                              'Odd Flux': (odd_time, odd_flux),
                              'Even Flux': (even_time, even_flux),
                              'Weak Secondary Flux': (time_noprimary, flux_noprimary),
                              'Centroid Offset Distance': (time_centroid_dist, centroid_dist),
                              # 'Centroid Offset Distance FDL': (time_centroid_dist_fdl, centroid_dist_fdl)
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
                                                           (3, 2),
                                                           os.path.join(config['output_dir'], 'plots'),
                                                           f'7_phasefolded_timeseries_'
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
        # bins with zero cadences are set to max(1, median of bin cadences)
        bin_counts[bin_counts == 0] = max(1, np.median(bin_counts))
        glob_flux_view_var /= np.sqrt(bin_counts)  # divide number of cadences in each bin by SE of the mean
        binned_timeseries['Global Flux'] = (binned_time, glob_flux_view, glob_flux_view_var)

        # create unfolded global flux views
        unfolded_glob_flux_view = []
        unfolded_glob_flux_view_var = []
        for phase in range(n_phases_split):
            unfolded_glob_flux_view_phase, binned_time, unfolded_glob_flux_view_var_phase, _, bin_counts = \
                global_view(time_split[phase],
                            flux_split[phase],
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
            unfolded_glob_flux_view_var_phase /= np.sqrt(bin_counts)

            unfolded_glob_flux_view.append(unfolded_glob_flux_view_phase)
            unfolded_glob_flux_view_var.append(unfolded_glob_flux_view_var_phase)

        unfolded_glob_flux_view = np.array(unfolded_glob_flux_view)
        unfolded_glob_flux_view_var = np.array(unfolded_glob_flux_view_var)

        if plot_preprocessing_tce and len(unfolded_glob_flux_view) > 0:
            utils_visualization.plot_riverplot(unfolded_glob_flux_view,
                                               config['num_bins_glob'],
                                               tce,
                                               os.path.join(config['output_dir'], 'plots'),
                                               f'8_riverplot_aug{tce["augmentation_idx"]}')

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

        # create unfolded local flux views
        unfolded_loc_flux_view = []
        unfolded_loc_flux_view_var = []
        for phase in range(n_phases_split):
            unfolded_loc_flux_view_phase, binned_time, unfolded_loc_flux_view_var_phase, _, bin_counts = \
                local_view(time_split[phase],
                           flux_split[phase],
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
            unfolded_loc_flux_view_var_phase /= np.sqrt(bin_counts)
            # binned_timeseries[f'Unfolded Local Flux Phase #{phase}'] = (binned_time, unfolded_loc_flux_view_phase,
            #                                                             unfolded_loc_flux_view_var_phase)

            unfolded_loc_flux_view.append(unfolded_loc_flux_view_phase)
            unfolded_loc_flux_view_var.append(unfolded_loc_flux_view_var_phase)

        unfolded_loc_flux_view = np.array(unfolded_loc_flux_view)
        unfolded_loc_flux_view_var = np.array(unfolded_loc_flux_view_var)

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
        # set local wks view to Gaussian noise using statistics from global weak secondary view when there are no data
        # to create the local view
        if np.all(np.isnan(loc_weak_secondary_view)):
            report_exclusion(config, tce,
                             f'No data available for local weak secondary view. Setting it to Gaussian noise using '
                             f'statistics from the global wks view.')
            mu, sigma = np.nanmedian(flux_noprimary), mad_std(flux_noprimary, ignore_nan=True)
            rng = np.random.default_rng()
            loc_weak_secondary_view = rng.normal(mu, sigma, len(loc_weak_secondary_view))
            loc_weak_secondary_view_var = sigma * np.ones(len(loc_flux_view_var))
            _, _, _, _, bin_counts = \
                global_view(time_noprimary,
                            flux_noprimary,
                            tce['tce_period'],
                            tce=tce,
                            normalize=False,
                            centering=False,
                            num_bins=config['num_bins_glob'],
                            bin_width_factor=config['bin_width_factor_glob'],
                            report={'config': config, 'tce': tce, 'view': 'global_flux_view'},
                            tce_duration=tce['tce_duration']
                            )
            mu, sigma = np.nanmedian(bin_counts), mad_std(bin_counts, ignore_nan=True)
            bin_counts = rng.normal(mu, sigma, len(loc_weak_secondary_view))
            bin_counts[bin_counts < 0] = 1
        bin_counts[bin_counts == 0] = max(1, np.median(bin_counts))
        loc_weak_secondary_view_var /= np.sqrt(bin_counts)
        binned_timeseries['Local Weak Secondary Flux'] = (binned_time,
                                                          loc_weak_secondary_view,
                                                          loc_weak_secondary_view_var
                                                          )
        # if plot_preprocessing_tce:
        #     utils_visualization.plot_wks(glob_view, glob_view_weak_secondary, tce, config,
        #                                  os.path.join(config['output_dir'], 'plots'), '8_wks_test')

        flux_views = {'global_flux_view': glob_flux_view,
                      'local_flux_view': loc_flux_view,
                      'local_flux_odd_view': odd_data['local_flux_view'],
                      'local_flux_even_view': even_data['local_flux_view'],
                      'unfolded_global_flux_view': unfolded_glob_flux_view,
                      'unfolded_local_flux_view': unfolded_loc_flux_view,
                      }

        flux_views_var = {'global_flux_view': glob_flux_view_var,
                          'local_flux_view': loc_flux_view_var,
                          'local_flux_odd_view': odd_data['local_flux_view_se'],
                          'local_flux_even_view': even_data['local_flux_view_se'],
                          'unfolded_global_flux_view': unfolded_glob_flux_view_var,
                          'unfolded_local_flux_view': unfolded_loc_flux_view_var,
                          }

        # get normalization statistics for global and local flux views
        flux_views_stats = {'median': {'global': np.median(flux_views['global_flux_view']),
                                       'local': np.median(flux_views['local_flux_view'])}}
        flux_views_stats['min'] = {'global': np.abs(np.min(flux_views['global_flux_view'] -
                                                           flux_views_stats['median']['global'])),
                                   'local': np.abs(np.min(flux_views['local_flux_view'] -
                                                          flux_views_stats['median']['local']))}

        # center by the flux view median and normalize by the flux view absolute minimum
        views_aux = {}
        for flux_view in flux_views:

            if 'unfolded' in flux_view:
                views_aux[f'{flux_view}_fluxnorm'] = []
                for phase in flux_views[flux_view]:
                    new_phase = \
                        centering_and_normalization(phase,
                                                    flux_views_stats['median'][('local', 'global')['global' in flux_view]],
                                                    flux_views_stats['min'][('local', 'global')['global' in flux_view]],
                                                    report={'config': config, 'tce': tce, 'view': flux_view}
                                                    )
                    views_aux[f'{flux_view}_fluxnorm'].append(new_phase)
                views_aux[f'{flux_view}_fluxnorm'] = np.array(views_aux[f'{flux_view}_fluxnorm'])

            views_aux[f'{flux_view}_fluxnorm'] = \
                centering_and_normalization(flux_views[flux_view],
                                            flux_views_stats['median'][('local', 'global')['global' in flux_view]],
                                            flux_views_stats['min'][('local', 'global')['global' in flux_view]],
                                            report={'config': config, 'tce': tce, 'view': flux_view}
                                            )
        flux_views.update(views_aux)
        views_aux = {}
        for flux_view in flux_views_var:
            if 'unfolded' in flux_view:
                views_aux[f'{flux_view}_fluxnorm'] = []
                for phase in flux_views_var[flux_view]:
                    new_phase = \
                        centering_and_normalization(phase,
                                                    0,
                                                    flux_views_stats['min'][('local', 'global')['global' in flux_view]],
                                                    report={'config': config, 'tce': tce, 'view': flux_view}
                                                    )
                    views_aux[f'{flux_view}_fluxnorm'].append(new_phase)
                views_aux[f'{flux_view}_fluxnorm'] = np.array(views_aux[f'{flux_view}_fluxnorm'])

            views_aux[f'{flux_view}_fluxnorm'] = \
                centering_and_normalization(flux_views_var[flux_view],
                                            0,
                                            flux_views_stats['min'][('local', 'global')['global' in flux_view]],
                                            report={'config': config, 'tce': tce, 'view': flux_view}
                                            )
        flux_views_var.update(views_aux)

        # normalize odd-even uncertainty by transit depth
        odd_data['se_oot'] = odd_data['se_oot'] / flux_views_stats['min']['local']
        odd_data['std_oot_bin'] = odd_data['std_oot_bin'] / flux_views_stats['min']['local']
        even_data['se_oot'] = even_data['se_oot'] / flux_views_stats['min']['local']
        even_data['std_oot_bin'] = even_data['std_oot_bin'] / flux_views_stats['min']['local']

        # # create local flux view for detecting non-centered transits
        # loc_flux_view_shift, _, _, _, _ = local_view(time,
        #                                              flux,
        #                                              tce['tce_period'],
        #                                              tce['tce_duration'],
        #                                              tce=tce,
        #                                              normalize=False,
        #                                              centering=False,
        #                                              num_durations=5,
        #                                              num_bins=config['num_bins_loc'],
        #                                              bin_width_factor=config['bin_width_factor_loc'],
        #                                              report={'config': config, 'tce': tce, 'view': 'local_flux_view'}
        #                                              )
        # flux_views['local_flux_view_shift'] = loc_flux_view_shift
        # med_view = np.median(flux_views['local_flux_view_shift'])
        # flux_views['local_flux_view_shift_fluxnorm'] = \
        #     centering_and_normalization(flux_views['local_flux_view_shift'],
        #                                 med_view,
        #                                 np.abs(np.min(flux_views['local_flux_view_shift'] - med_view)),
        #                                 report={'config': config, 'tce': tce, 'view': 'local_flux_view_shift'})

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
        # glob_centr_fdl_view, _, _, _, _ = global_view(time_centroid_dist_fdl,
        #                                               centroid_dist_fdl,
        #                                               tce['tce_period'],
        #                                               tce=tce,
        #                                               centroid=True,
        #                                               normalize=False,
        #                                               centering=False,
        #                                               num_bins=config['num_bins_glob'],
        #                                               bin_width_factor=config['bin_width_factor_glob'],
        #                                               report={'config': config, 'tce': tce,
        #                                                       'view': 'global_centr_fdl_view'},
        #                                               tce_duration=tce['tce_duration']
        #                                               )
        # loc_centr_fdl_view, _, _, _, _ = local_view(time_centroid_dist_fdl,
        #                                             centroid_dist_fdl,
        #                                             tce['tce_period'],
        #                                             tce['tce_duration'],
        #                                             tce=tce,
        #                                             centroid=True,
        #                                             normalize=False,
        #                                             centering=False,
        #                                             num_durations=config['num_durations'],
        #                                             num_bins=config['num_bins_loc'],
        #                                             bin_width_factor=config['bin_width_factor_loc'],
        #                                             report={'config': config, 'tce': tce,
        #                                                     'view': 'local_centr_fdl_view'}
        #                                             )

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
                       # 'global_centr_fdl_view': glob_centr_fdl_view,
                       # 'local_centr_fdl_view': loc_centr_fdl_view
                       }

        # # adjust TESS centroids to Kepler by dividing by pixel scale factor
        # if config['satellite'] == 'tess':
        #     centr_views['global_centr_view_adjscl'] = glob_centr_view / config['tess_to_kepler_px_scale_factor']
        #     centr_views['local_centr_view_adjscl'] = loc_centr_view / config['tess_to_kepler_px_scale_factor']

        centr_views_var = {'global_centr_view': glob_centr_view_var,
                           'local_centr_view': loc_centr_view_var,
                           }

        # create local view for momentum dump
        if config['get_momentum_dump']:

            loc_mom_dump_view, loc_mom_dump_view_var, binned_time = generate_view_momentum_dump(
                time_momentum_dump,
                momentum_dump,
                config['num_bins_loc'],
                config['bin_width_factor_loc'] * tce['tce_duration'],
                max(-tce['tce_period'] / 2, -tce['tce_duration'] * config['num_durations']),
                min(tce['tce_period'] / 2, tce['tce_duration'] * config['num_durations']),
            )

            if plot_preprocessing_tce:
                utils_visualization.plot_momentum_dump(loc_mom_dump_view, loc_mom_dump_view_var,
                                                       binned_time, momentum_dump,
                                                       time_momentum_dump, tce,
                                                       os.path.join(config['output_dir'], 'plots'),
                                                       f'9_momentum_dump_phase_and_binned')

        # initialize dictionary with the views that will be stored in the TFRecords example for this TCE
        views = {}
        views.update(flux_views)
        views.update(weak_secondary_flux_views)
        views.update(centr_views)
        if config['get_momentum_dump']:
            views['local_momentum_dump_view'] = loc_mom_dump_view

        views_var = {}
        views_var.update(flux_views_var)
        views_var.update(weak_secondary_flux_views_var)
        views_var.update(centr_views_var)
        if config['get_momentum_dump']:
            views_var['local_momentum_dump_view'] = loc_mom_dump_view_var

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
                             # 'global_centr_fdl_view',
                             # 'local_centr_fdl_view'
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

            utils_visualization.plot_odd_even(phasefolded_timeseries,
                                              binned_timeseries,
                                              tce,
                                              config,
                                              os.path.join(config['output_dir'], 'plots'),
                                              f'8_oddeven_transitdepth_phasefoldedbinned_timeseries_'
                                              f'aug{tce["augmentation_idx"]}'
                                              )

            views_var_plot = {view_name: view for view_name, view in views_var.items() if view_name in views_to_plot}
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
        if 'unfolded' in view:
            example_util.set_tensor_feature(ex, view, views[view])
        else:
            example_util.set_float_feature(ex, view, views[view])
    for view in views_var:
        if 'unfolded' in view:
            example_util.set_tensor_feature(ex, f'{view}_var', views[view])
        else:
            example_util.set_float_feature(ex, f'{view}_var', views[view])

    # set other features from the TCE table - diagnostic statistics, transit fits, stellar parameters...
    for name, value in tce.items():

        if name == 'Public Comment':  # can add special characters that are not serializable in the TFRecords
            continue
        try:
            if isinstance(value, str):
                example_util.set_bytes_feature(ex, name, [value])
            else:
                example_util.set_feature(ex, name, [value])
        except Exception as e:
            report_exclusion(config, tce, f'Could not set up this TCE table parameter: {name}.')
            print(name, value)

    # add number of transits per view
    for view in num_transits:
        example_util.set_int64_feature(ex, f'{view}_num_transits', [num_transits[view]])

    # add odd and even scalar features
    for field in ['se_oot', 'std_oot_bin']:
        example_util.set_float_feature(ex, f'odd_{field}', [odd_data[field]])
        example_util.set_float_feature(ex, f'even_{field}', [even_data[field]])

        # provide adjusted odd-even SE due to TESS having higher sampling rate
        if config['satellite'] == 'tess' and field == 'se_oot':
            example_util.set_float_feature(ex, f'odd_{field}_adjsampl',
                                           [odd_data[field] * config['tess_to_kepler_sampling_ratio']])
            example_util.set_float_feature(ex, f'even_{field}_adjsampl',
                                           [even_data[field] * config['tess_to_kepler_sampling_ratio']])

    example_stats = create_example_stats(binned_timeseries, views, views_var, odd_data, even_data, odd_even_flag,
                                         num_transits, inds_bin_nan, t_min_transit, t_max_transit, config)

    return ex, example_stats
