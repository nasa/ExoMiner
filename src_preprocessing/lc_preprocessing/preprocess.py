"""
Light curve data preprocessing module.
"""

# 3rd party
import numpy as np
import tensorflow as tf
from astropy.stats import mad_std
import lightkurve as lk
import logging

# local
from src_preprocessing.lc_preprocessing import utils_visualization
from src_preprocessing.tf_util import example_util
from src_preprocessing.lc_preprocessing.utils_centroid_preprocessing import (kepler_transform_pxcoordinates_mod13,
                                                                             preprocess_centroid_motion_for_tce)
from src_preprocessing.lc_preprocessing.utils_ephemeris import create_binary_time_series, \
    find_first_epoch_after_this_time
from src_preprocessing.lc_preprocessing.utils_imputing import imputing_gaps
from src_preprocessing.lc_preprocessing.utils_odd_even import create_odd_even_views, \
    phase_fold_and_sort_light_curve_odd_even
from src_preprocessing.lc_preprocessing.utils_preprocessing import (remove_non_finite_values,
                                                                    check_inputs_generate_example,
                                                                    remove_outliers,
                                                                    check_inputs)
from src_preprocessing.lc_preprocessing.utils_preprocessing_io import report_exclusion, read_light_curve
from src_preprocessing.lc_preprocessing.detrend_timeseries import detrend_flux_using_spline, \
    detrend_flux_using_sg_filter
from src_preprocessing.lc_preprocessing.phase_fold_and_binning import (global_view, local_view,
                                                                       phase_fold_and_sort_light_curve,
                                                                       phase_split_light_curve,
                                                                       centering_and_normalization,
                                                                       generate_view_momentum_dump)
from src_preprocessing.lc_preprocessing.lc_periodogram import lc_periodogram_pipeline


NORM_SIGMA_EPS = 1e-12
PHASEFOLD_GRID_PLOTS = (3, 3)
MAP_VIEWS_TO_OLD_NAMES = {

    'flux_global': 'global_flux_view',
    'flux_local': 'local_flux_view',
    'flux_global_norm': 'global_flux_view_fluxnorm',
    'flux_local_norm': 'local_flux_view_fluxnorm',

    'flux_global_unfolded': 'unfolded_global_flux_view',
    'flux_global_unfolded_norm': 'unfolded_global_flux_view_fluxnorm',
    'flux_local_unfolded': 'unfolded_local_flux_view',
    'flux_local_unfolded_norm': 'unfolded_local_flux_view_fluxnorm',

    'flux_odd_local': 'local_flux_odd_view',
    'flux_even_local': 'local_flux_even_view',
    'flux_odd_local_norm': 'local_flux_odd_view_fluxnorm',
    'flux_even_local_norm': 'local_flux_even_view_fluxnorm',

    'flux_weak_secondary_local': 'local_weak_secondary_view',
    'flux_weak_secondary_local_norm': 'local_weak_secondary_view_selfnorm',
    # 'local_weak_secondary_view_max_flux-wks_norm',
    # 'local_weak_secondary_view_fluxnorm',

    'centroid_offset_distance_to_target_global': 'global_centr_view',
    'centroid_offset_distance_to_target_local': 'local_centr_view',

    'flux_trend_global': 'flux_trend_global',
    'flux_trend_local': 'flux_trend_local',
    'flux_trend_global_norm': 'flux_trend_global_norm',
    'flux_trend_local_norm': 'flux_trend_local_norm',

    'flux_trend_global_unfolded': 'flux_trend_global_unfolded',
    'flux_trend_global_unfolded_norm': 'flux_trend_global_unfolded_norm',
    'flux_trend_local_unfolded': 'flux_trend_local_unfolded',
    'flux_trend_local_unfolded_norm': 'flux_trend_local_unfolded_norm',

    'momentum_dump_local': 'local_momentum_dump_view'
}
BINNED_TIMESERIES_JOINT_PLOT_GRID = (4, 4)
BINNED_TIMESERIES_JOINT_PLOT = [
    # flux ###

    'flux_global',
    'flux_local',
    'flux_global_norm',
    'flux_local_norm',

    # flux_global_unfolded -> has a plot
    # flux_global_unfolded_norm

    # 'flux_local_unfolded',
    # 'flux_local_unfolded_norm',

    # trend ###

    'flux_trend_global',
    'flux_trend_local',
    'flux_trend_global_norm',
    # 'flux_trend_local_norm',

    # flux_trend_global_unfolded
    # flux_trend_global_unfolded_norm

    # flux_trend_local_unfolded
    # flux_trend_local_unfolded_norm

    # wks ###
    'flux_weak_secondary_local',
    'flux_weak_secondary_local_norm',

    # odd/even ###
    'flux_odd_local',  # -> has a plot
    'flux_even_local',  # -> has a plot

    'flux_odd_local_norm',
    'flux_even_local_norm',

    # centroid ###
    'centroid_offset_distance_to_target_global',
    'centroid_offset_distance_to_target_local',

    # momentum ###
    'momentum_dump_local',  # -> has a plot
]

logger = logging.getLogger(__name__)


def sample_ephemerides(tce, config):
    """ Sample ephemerides using ephemerides uncertainty.

    Args:
        tce: pandas Series, TCE data
        config: dict, preprocessing parameters

    Returns: tce, pandas Series with sampled ephemerides

    """

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

    return tce


def find_intransit_cadences(tces_in_target, time_arrs, transit_duration_factor):
    """ Find in-transit cadences (primary and secondary transits) for TCEs in target.

    Args:
        tces_in_target: pandas DataFrame, target TCEs
        time_arrs: list of NumPy arrays, timestamp values

    Returns:
        target_intransit_cadences_arr, list of boolean NumPy arrays (n_tces_in_target, n_cadences); each row is for a
            given TCE in the target star; True for in-transit cadences, False otherwise
    """

    n_tces_in_target = len(tces_in_target)

    target_intransit_cadences_arr = [np.zeros((n_tces_in_target, len(time_arr)), dtype='bool')
                                     for time_arr in time_arrs]
    for tce_in_target_i, tce_in_target in tces_in_target.iterrows():  # iterate on target's TCEs

        duration_primary, duration_secondary = set_duration_gap_for_tce(
            tce_in_target['tce_duration'], 
            tce_in_target['tce_period'], 
            transit_duration_factor, 
            tce_in_target['tce_maxmesd']
            )
        
        # get epoch of first transit for each time array
        first_transit_time = [find_first_epoch_after_this_time(tce_in_target['tce_time0bk'],
                                                               tce_in_target['tce_period'],
                                                               time[0])
                              for time in time_arrs]

        # create binary time series for each time array in which in-transit points are labeled as True, otherwise False
        tce_intransit_cadences_arr = [create_binary_time_series(time,
                                                                first_transit_time,
                                                                duration_primary,
                                                                tce_in_target['tce_period'])
                                      for first_transit_time, time in zip(first_transit_time, time_arrs)]

        if 'tce_maxmesd' in tce_in_target and duration_secondary is not None:  # do the same for the detected secondary transits

            first_transit_time_secondary = [find_first_epoch_after_this_time(tce_in_target['tce_time0bk'] +
                                                                             tce_in_target['tce_maxmesd'],
                                                                             tce_in_target['tce_period'],
                                                                             time[0])
                                            for time in time_arrs]

            tce_intransit_cadences_arr_secondary = [create_binary_time_series(time,
                                                                              first_transit_time,
                                                                              duration_secondary,
                                                                              tce_in_target['tce_period'])
                                                    for first_transit_time, time in
                                                    zip(first_transit_time_secondary, time_arrs)]

            # update TCE array for primary with secondary transits
            tce_intransit_cadences_arr = [np.logical_or(primary_arr, secondary_arr)
                                          for primary_arr, secondary_arr
                                          in zip(tce_intransit_cadences_arr, tce_intransit_cadences_arr_secondary)]

        # update array for target with tce transits
        for arr_i, intransit_cadences_arr in enumerate(tce_intransit_cadences_arr):
            target_intransit_cadences_arr[arr_i][tce_in_target_i] = intransit_cadences_arr

    return target_intransit_cadences_arr


def gap_intransit_cadences_other_tces(target_intransit_cadences_arr, idx_tce, data, gap_keep_overlap=True,
                                      impute_gaps=False, seed=None, tce=None, config=None):
    """ Find in-transit cadences (primary and secondary transits) for other detected TCEs in the light curve time series
    and gaps them (i.e., sets cadences to NaN) or imputes them if `impute_gaps` is set to True.

        Args:
            target_intransit_cadences_arr: list of boolean NumPy arrays (n_tces_in_target, n_cadences); each row is for
                a given TCE in the target star; True for in-transit cadences, False otherwise
            idx_tce: index of TCE of interest in `target_intransit_cadences_arr`
            data: dict, lightcurve raw data from the FITS files (e.g., flux, centroid motion time series)
            gap_keep_overlap: bool, if True it does not gap cadences belonging to the TCE of interest that overlap with
                other TCEs
            impute_gaps: bool, if True it imputes data in the gapped transits instead of setting them to NaN.
            seed: rng seed
            tce: pandas Series, TCE parameters
            config: dict, preprocessing parameters
        Returns:
            data, dict with light curve raw data from the FITS files, but now with cadences for the transits of other
            detected TCEs set to NaN or imputed
    """

    other_tces_idxs = np.ones(len(target_intransit_cadences_arr[0]), dtype='bool')
    other_tces_idxs[idx_tce] = False

    it_cadences_other_tces = [np.sum(target_intransit_cadences[other_tces_idxs, :], axis=0) != 0
                              for target_intransit_cadences in target_intransit_cadences_arr]
    if gap_keep_overlap:
        for arr_i, it_cadences_other_tces_arr in enumerate(it_cadences_other_tces):
            it_cadences_other_tces_arr[target_intransit_cadences_arr[arr_i][idx_tce]] = False

    for arr_i, it_cadences_other_tces_arr in enumerate(it_cadences_other_tces):
        for timeseries in data:
            if impute_gaps:  # impute gapped transit cadences
                if 'centroids' in timeseries:
                    data[timeseries]['x'][arr_i] = (
                        imputing_gaps(data[timeseries]['x'][arr_i], it_cadences_other_tces_arr, seed=seed, tce=tce,
                                      config=config))
                    data[timeseries]['y'][arr_i] = imputing_gaps(data[timeseries]['y'][arr_i],
                                                                 it_cadences_other_tces_arr, seed=seed, tce=tce,
                                                                 config=config)
                else:
                    data[timeseries][arr_i] = (
                        imputing_gaps(data[timeseries][arr_i], it_cadences_other_tces_arr, seed=seed, tce=tce,
                                      config=config))
            else:  # set gapped transit cadences to NaN
                if 'centroids' in timeseries:
                    data[timeseries]['x'][arr_i][it_cadences_other_tces_arr] = np.nan
                    data[timeseries]['y'][arr_i][it_cadences_other_tces_arr] = np.nan
                else:
                    data[timeseries][arr_i][it_cadences_other_tces_arr] = np.nan

    return data


def phase_fold_timeseries(data, config, tce, plot_preprocessing_tce):
    """ Compute the phase folded versions of the detrended time series.

    Args:
        data: dict, detrended time series
        config: dict, preprocessing parameters
        tce: pandas Series, parameters and DV diagnostics for TCE of interest
        plot_preprocessing_tce: bool, if True it generates plots

    Returns: phase-folded_timeseries, dict that maps time series to a tuple with the phase, phase-folded values, and
    number of transits (partial transits included) found

    """

    data = check_inputs_generate_example(data, tce, config)

    # check if data for all views is valid after preprocessing the raw time series
    for key, val in data.items():
        if val is None:
            raise KeyError(f'No detrended data for {key} time series before creating the phase folded time series.')

    phasefolded_timeseries = {field: 3 * (None,)
                              for field in ['flux', 'flux_unfolded', 'flux_trend', 'flux_trend_unfolded', 'flux_odd',
                                            'flux_even', 'flux_weak_secondary', 'centroid_offset_distance_to_target',
                                            'momentum_dump']}

    # phase folding for flux time series
    time, flux, n_transits_flux = phase_fold_and_sort_light_curve(data['flux_time'],
                                                                  data['flux'],
                                                                  tce['tce_period'],
                                                                  tce['tce_time0bk'],
                                                                  augmentation=False)

    phasefolded_timeseries['flux'] = (time, flux, n_transits_flux)

    # phase folding for flux trend time series
    time_trend, flux_trend, n_transits_flux_trend = phase_fold_and_sort_light_curve(data['flux_time'],
                                                                                    data['flux_trend'],
                                                                                    tce['tce_period'],
                                                                                    tce['tce_time0bk'],
                                                                                    augmentation=False)

    phasefolded_timeseries['flux_trend'] = (time_trend, flux_trend, n_transits_flux_trend)

    # phase folding for the weak secondary flux time series
    time_wksecondary, flux_wksecondary, n_transits_wksecondary = (
        phase_fold_and_sort_light_curve(data['flux_time'],
                                        data['flux'],
                                        tce['tce_period'],
                                        tce['tce_time0bk'] + tce['tce_maxmesd'],
                                        augmentation=False))
    phasefolded_timeseries['flux_weak_secondary'] = (time_wksecondary, flux_wksecondary, n_transits_wksecondary)

    # phase folding for odd and even time series
    (odd_time, odd_flux, n_transits_odd, odd_time_nophased), \
        (even_time, even_flux, n_transits_even, even_time_nophased) = (
        phase_fold_and_sort_light_curve_odd_even(data['flux_time'],
                                                 data['flux'],
                                                 tce['tce_period'],
                                                 tce['tce_time0bk'],
                                                 augmentation=False))
    phasefolded_timeseries['flux_odd'] = (odd_time, odd_flux, n_transits_odd)
    phasefolded_timeseries['flux_even'] = (even_time, even_flux, n_transits_even)

    # phase folding for centroid time series
    time_centroid_dist, centroid_dist, n_transits_centroid = (
        phase_fold_and_sort_light_curve(data['centroid_dist_time'],
                                        data['centroid_dist'],
                                        tce['tce_period'],
                                        tce['tce_time0bk'],
                                        augmentation=False))

    phasefolded_timeseries['centroid_offset_distance_to_target'] = \
        (time_centroid_dist, centroid_dist, n_transits_centroid)

    # phase folding for momentum time series
    if config['get_momentum_dump']:
        time_momentum_dump, momentum_dump, n_transits_momentum_dump = (
            phase_fold_and_sort_light_curve(data['momentum_dump_time'],
                                            data['momentum_dump'],
                                            tce['tce_period'],
                                            tce['tce_time0bk'],
                                            augmentation=False))
        phasefolded_timeseries['momentum_dump'] = (time_momentum_dump, momentum_dump, n_transits_momentum_dump)

    # phase folding for flux time series to generate phases separately
    time_split, flux_split, n_phases_split, _ = phase_split_light_curve(
        data['flux_time'],
        data['flux'],
        tce['tce_period'],
        tce['tce_time0bk'],
        tce['tce_duration'],
        config['n_max_phases'],
        config['keep_odd_even_order'],
        config['frac_it_cadences_thr'],
        config['sampling_rate_h'],  # [config['satellite']],
        config['phase_extend_method'],
        quarter_timestamps=data['quarter_timestamps'] if config['satellite'] == 'kepler' and config[
            'quarter_sampling']
        else None,
        remove_outliers_params={'sigma': config['outlier_removal_sigma'],
                                'fill': config['outlier_removal_fill'],
                                'outlier_type': 'upper',
                                } if config['outlier_removal'] else None
    )
    phasefolded_timeseries['flux_unfolded'] = (time_split, flux_split, n_phases_split)

    if n_phases_split < config['min_n_phases']:
        raise ValueError(f'Only found {n_phases_split} phase(s) for flux, need at least {config["min_n_phases"]} to '
                         f'create example.')

    if n_phases_split == 0:  # replace by phase from phase-folding
        phasefolded_timeseries['flux_unfolded'] = (
            np.tile(phasefolded_timeseries['flux'][0], (config['n_max_phases'], 1)),
            np.tile(phasefolded_timeseries['flux'][1], (config['n_max_phases'], 1)),
            config['n_max_phases']
        )

    # phase folding for flux trend time series to generate phases separately
    time_trend_split, flux_trend_split, n_phases_trend_split, _ = phase_split_light_curve(
        data['flux_time'],
        data['flux_trend'],
        tce['tce_period'],
        tce['tce_time0bk'],
        tce['tce_duration'],
        config['n_max_phases'],
        config['keep_odd_even_order'],
        0,
        config['sampling_rate_h'],  # [config['satellite']],
        config['phase_extend_method'],
        quarter_timestamps=data['quarter_timestamps'] if config['satellite'] == 'kepler' and config['quarter_sampling']
        else None,
        remove_outliers_params={'sigma': config['outlier_removal_sigma'],
                                'fill': config['outlier_removal_fill'],
                                'outlier_type': 'upper',
                                } if config['outlier_removal'] else None
    )
    phasefolded_timeseries['flux_trend_unfolded'] = (time_trend_split, flux_trend_split, n_phases_trend_split)

    if n_phases_trend_split < config['min_n_phases']:
        raise ValueError(f'Only found {n_phases_trend_split} phase(s) for flux trend, need at least '
                         f'{config["min_n_phases"]} to create example.')

    if plot_preprocessing_tce:
        # CHANGE NUMBER OF PLOTS AS FUNCTION OF THE TIME SERIES PREPROCESSED
        utils_visualization.plot_all_phasefoldedtimeseries({ts_name: ts_data
                                                            for ts_name, ts_data in phasefolded_timeseries.items()
                                                            if 'unfolded' not in ts_name},
                                                           tce,
                                                           PHASEFOLD_GRID_PLOTS,
                                                           config['plot_dir'] /
                                                           f'{tce["uid"]}_{tce["label"]}_'
                                                           f'5_phasefolded_timeseries.png',
                                                           None
                                                           )

    # remove outliers from the phase folded flux time series
    if config['outlier_removal']:

        for ts_name, (ts_time, ts_values, n_transits_ts) in phasefolded_timeseries.items():

            if ts_name in ['momentum_dump', 'centroid_offset_distance_to_target']:
                continue

            if 'unfolded' in ts_name:  # outlier removal was already performed
                continue

            if 'trend' in ts_name:

                ts_values, _ = remove_outliers(
                    ts_values,
                    sigma=config['outlier_removal_sigma'],
                    fill=config['outlier_removal_fill'],
                    outlier_type='both',
                )

            else:  # any flux time series
                ts_values, _ = remove_outliers(
                    ts_values,
                    sigma=config['outlier_removal_sigma'],
                    fill=config['outlier_removal_fill'],
                    outlier_type='upper',
                )

    if plot_preprocessing_tce:
        # CHANGE NUMBER OF PLOTS AS FUNCTION OF THE TIME SERIES PREPROCESSED
        utils_visualization.plot_all_phasefoldedtimeseries({ts_name: ts_data
                                                            for ts_name, ts_data in phasefolded_timeseries.items()
                                                            if 'unfolded' not in ts_name},
                                                           tce,
                                                           PHASEFOLD_GRID_PLOTS,
                                                           config['plot_dir'] /
                                                           f'{tce["uid"]}_{tce["label"]}_'
                                                           f'5_phasefolded_timeseries_outlierrem.png',
                                                           None
                                                           )

    return phasefolded_timeseries


def set_duration_gap_for_tce(tce_duration, tce_period, transit_duration_factor, tce_secondary_offset=None):
    
    # setting primary and secondary transit gap duration
    if tce_secondary_offset is not None:
        duration_gapped_primary = max(
            min(transit_duration_factor * tce_duration,
                2 * np.abs(tce_secondary_offset) - tce_duration,
                tce_period),
            tce_duration
        )
        # setting secondary gap duration
        duration_gapped_secondary = duration_gapped_primary
        # config['duration_gapped_secondary'] = (
        #     min(
        #         max(
        #             0,
        #             2 * np.abs(tce['tce_maxmesd']) - config['duration_gapped_primary'] - 2 * config['primary_buffer_time']),
        #         config['gap_padding'] * tce['tce_duration']))
    else:
        # config['tr_dur_f']
        duration_gapped_primary = min(transit_duration_factor * tce_duration, tce_period)
        duration_gapped_secondary = None
    
    return duration_gapped_primary, duration_gapped_secondary


def process_single_tce(tce, detrended_data, data, target_position, config, plot_preprocessing_tce=False):
    """ Preprocesses the detrended light curve time series data and scalar features for a given SPOC TCE `tce`, and 
        returns a dictionary of TCEs' Example proto to be writtend to TFRecord files.

    :param tce: pandas Series, SPOC TCE parameters and statistics
    :param detrended_data: dict, detrended flux and centroid motion time series for the target
    :param data: dict, raw flux and centroid motion time series for the target
    :param target_position: list of RA and Dec coordinates of the target star
    :param config: dict, holds preprocessing parameters
    :param plot_preprocessing_tce: bool, if True it generates preprocesssing plots for the TCE

    return: a tensorflow.train.Example proto containing TCE features
    """
    
    # # sample TCE ephemeris using uncertainty interval
    # tce = sample_ephemerides(tce, config)
    
    # get boolean array with in-transit cadences for the TCE of interest
    tce_intransit_cadences_arr = [target_intransit_cadences[config['idx_tce']]
                                  for target_intransit_cadences in config['target_intransit_cadences_arr']]
    config['intransit_cadences_tce'] = tce_intransit_cadences_arr
    
    if plot_preprocessing_tce:
        utils_visualization.plot_intransit_binary_timeseries(data['all_time'],
                                                             data['all_flux'],
                                                             config['intransit_cadences_target'],
                                                             config['intransit_cadences_tce'],
                                                             tce,
                                                             config['plot_dir'] /
                                                             f'{tce["uid"]}_{tce["label"]}_'
                                                             f'1_intransit_cadences.png')

    # FIX data being updated
    # gap cadences belonging to the transits of other TCEs in the same target star
    if config['gapped']:
        logger.info(f'[{tce["uid"]}] Gap in-transit cadences from other TCEs in target {tce["target_id"]}.')
        data_to_be_gapped = {ts_name: ts_data for ts_name, ts_data in data.items()
                             if ts_name in ['all_flux', 'all_centroids']}
        # if config['get_momentum_dump']:
        #     data_to_be_gapped['momentum_dump'] = data['momentum_dump']

        data_gapped = gap_intransit_cadences_other_tces(
            config['target_intransit_cadences_arr'], 
            config['idx_tce'], 
            data_to_be_gapped,
            config['gap_keep_overlap'], 
            config['gap_imputed'],
            seed=config['random_seed'], 
            tce=tce, 
            config=config
            )
        data.update(data_gapped)

    # preprocess centroid motion distance into distance from target coordinates
    centroid_dist_time, centroid_dist = preprocess_centroid_motion_for_tce(
        tce,
        detrended_data['centroid_time'], 
        detrended_data['centroid'], 
        detrended_data['avg_centroid_oot'], 
        target_position, 
        config, 
        plot_preprocessing_tce
        )
    
    detrended_data['centroid_dist_time'] = centroid_dist_time
    detrended_data['centroid_dist'] = centroid_dist
    
    # compute lc periodogram
    logger.info(f'[{tce["uid"]}] Computing periodogram data for TCE...')
    pgram_data = lc_periodogram_pipeline(
        config['p_min_tce'], config['k_harmonics'], config['p_max_obs'], config['downsampling_f'],
        config['smooth_filter_type'], config['smooth_filter_w_f'],
        config['tr_dur_f'],
        tce, data['all_time'], data['all_flux'], data['all_flux_err'],
        save_fp=config['plot_dir'] / f'{tce["uid"]}_{tce["label"]}_2_lc_periodogram.png' if config['plot_figures'] else None,
        plot_preprocessing_tce=plot_preprocessing_tce)

    # phase fold detrended data using detected orbital period
    logger.info(f'[{tce["uid"]}] Phase-folding time series using TCE\'s orbital period found in TCE table...')
    phase_folded_data = phase_fold_timeseries(detrended_data, config, tce, plot_preprocessing_tce)

    # bin phase folded data and generate TCE example based on the preprocessed data
    logger.info(f'[{tce["uid"]}] Generating features for TCE based on preprocessed light curve data...')
    example_tce = generate_example_for_tce(phase_folded_data, pgram_data, tce, config, plot_preprocessing_tce)
    
    return example_tce
    
    
def process_target_tces(target_uid, target_tces_tbl, config):
    """ Preprocesses the light curve time series data and scalar features for the SPOC TCEs detected for the KIC/TIC ID 
     (for a specific sector run, in the case of TESS), and returns a dictionary of TCEs' Example proto to be writtend to TFRecord files.

    :param target_uid: str, target unique id (TIC ID-Sector run/KIC ID)
    :param target_tces_tbl: pandas Series, target TCEs parameters and statistics
    :param config: dict, holds preprocessing parameters

    return: a dictionary mapping TCEs unique IDs to a tensorflow.train.Example proto containing TCE features
    """

    # check if preprocessing pipeline figures are saved for the target and its TCEs
    plot_preprocessing_tce = False
    if np.random.random() < config['plot_prob']:
        plot_preprocessing_tce = config['plot_figures']

    target_dict = {k: v for k, v in target_tces_tbl.iloc[0].items() if k in ['target_id', 'sector_run', 'sectors_observed']}
    # get cadence, flux and centroid data for the tce
    logger.info(f'[{target_uid}] Reading light curve data for target {target_uid}.')
    data = read_light_curve(target_dict, config)
    if data is None:
        raise IOError(f'Issue when reading data from the FITS file(s) for target {target_uid}.')

    # update target position in FITS file with that from the TCE table
    tce_table_target_ra, tce_table_target_dec = target_tces_tbl[['ra', 'dec']].values[0, :]
    if ~np.isnan(tce_table_target_ra) and ~np.isnan(tce_table_target_dec):
        logger.info(f'[{target_uid}] Using targets\' RA and Dec coordinates from the TCE table.')
        data['target_position'] = [tce_table_target_ra, tce_table_target_dec]
    config['delta_dec'] = np.cos(data['target_position'][1] * np.pi / 180)

    # config['primary_buffer_time'] = (config['primary_buffer_nsamples'] /
    #                                  config['sampling_rate_h'][config['satellite']] / 24)
    data['errors'] = check_inputs(data)

    # get number of samples per hour
    config['sampling_rate_h'] = 1 / (np.median(np.diff(np.concatenate(data['all_time']))) * 24)

    # find transits for all detected TCEs in the target
    logger.info(f'[{target_uid}] Finding transits for all detected TCEs in target.')
    target_intransit_cadences_arr = find_intransit_cadences(
        target_tces_tbl,
        data['all_time'],
        config['tr_dur_f'],
        )
    config['target_intransit_cadences_arr'] = target_intransit_cadences_arr
    # build boolean array with in-transit cadences based on all detected TCEs in the target
    target_intransit_cadences_bool_arr = [np.sum(target_intransit_cadences, axis=0) != 0
                                          for target_intransit_cadences in target_intransit_cadences_arr]
    config['intransit_cadences_target'] = target_intransit_cadences_bool_arr
    # # get boolean array with out-of-transit cadences based on all detected TCEs in the target
    # target_outoftransit_cadences_bool_arr = [~target_intransit_cadences_bool
    #                                          for target_intransit_cadences_bool in target_intransit_cadences_bool_arr]
    # config['outoftransit_cadences_target'] = target_outoftransit_cadences_bool_arr  # NOTE Is this needed?

    # detrend the flux and centroid time series
    if config['detrending_method'] == 'savitzky-golay':  # set Savitzky-Golay window
        # config['sg_win_len'] = int(config['sg_n_durations_win'] * tce['tce_duration'] * 24 *
        #                            config['sampling_rate_h'][f'{config["satellite"]}'])
        # win_dur_h = config['sg_n_durations_win'] * tce['tce_duration'] * 24
        win_dur_h = 1.2 * 24
        config['sg_win_len'] = int(win_dur_h * config['sampling_rate_h'])  # [f'{config["satellite"]}'])
        config['sg_win_len'] = config['sg_win_len'] if config['sg_win_len'] % 2 != 0 else config['sg_win_len'] + 1

    logger.info(f'[{target_uid}] Preprocessing light curve data for target {target_uid}...')
    detrended_data = process_light_curve(data, config, target_uid, plot_preprocessing_tce)
    logger.info(f'[{target_uid}] Finished preprocessing light curve data for target {target_uid}.\n Started preprocessing {len(target_tces_tbl)} TCEs...')

    examples_tces_dict = {tce_uid: {field: None for field in ['data', 'processed', 'error']} for tce_uid in target_tces_tbl['uid']}
    for tce_i, tce in target_tces_tbl.iterrows():  # preprocess data for each TCE
        
        logger.info(f'Preprocessing TCE {tce["uid"]} for target {target_uid}...')
        
        config['idx_tce'] = tce_i  # set TCE index
        
        try:
            example_tce_data = process_single_tce(tce, detrended_data, data, data['target_position'], config, plot_preprocessing_tce)
            examples_tces_dict[tce['uid']]['data'] = example_tce_data
            examples_tces_dict[tce['uid']]['processed'] = True
        except Exception as tce_preprocessing_error:
            examples_tces_dict[tce['uid']]['error'] = tce_preprocessing_error
            examples_tces_dict[tce['uid']]['processed'] = False
        
        logger.info(f'Finished preprocessed TCE {tce["uid"]} for target {target_uid}. Status: {examples_tces_dict[tce["uid"]]["processed"]}')
                
    return examples_tces_dict


def flux_preprocessing(all_time, all_flux, target_uid, config, plot_preprocessing_tce):
    """ Preprocess the flux time series.

    :param all_time: list of NumPy arrays, timestamps
    :param all_flux: list of NumPy arrays, flux time series
    :param target_uid: str, target unique id (TIC ID-Sector run/KIC ID)
    :param config: dict, preprocessing parameters
    :param plot_preprocessing_tce: bool, set to True to plot figures related to different preprocessing steps
    :return:
        time: NumPy array, timestamps for preprocessed flux time series
        detrended_flux: NumPy array, detrended flux timeseries
        trend: NumPy array, trend obtained when fitting detrending method to the flux timeseries
    """

    # copy arrays
    time_arrs, flux_arrs, intransit_cadences_target = \
        ([np.array(el) for el in all_time],
         [np.array(el) for el in all_flux],
         [np.array(el) for el in config['intransit_cadences_target']],
        )
    # remove non-finite values
    time_arrs, flux_arrs, intransit_cadences_target = (
        remove_non_finite_values([
            time_arrs,
            flux_arrs,
            intransit_cadences_target,
        ]))

    flux = np.concatenate(flux_arrs)

    # detrend flux
    if config['detrending_method'] == 'spline':

        time, detrended_flux, trend, flux_lininterp = (
            detrend_flux_using_spline(flux_arrs, time_arrs, intransit_cadences_target, config))

    elif config['detrending_method'] == 'savitzky-golay':
        # convert data to lightcurve object
        time, flux, intransit_cadences_target = (np.concatenate(time_arrs), np.concatenate(flux_arrs),
                                                 np.concatenate(intransit_cadences_target))
        lc = lk.LightCurve(data={'time': np.array(time), 'flux': np.array(flux)})

        time, detrended_flux, trend, models_info_df = detrend_flux_using_sg_filter(lc,
                                                                                   intransit_cadences_target,
                                                                                   config['sg_win_len'],
                                                                                   config['sg_sigma'],
                                                                                   config['sg_max_poly_order'],
                                                                                   config['sg_penalty_weight'],
                                                                                   config['sg_break_tolerance']
                                                                   )

        logger.info(f'[{target_uid}] SG detrending model flux data info. Chosen polynomial order: '
                    f'{models_info_df.index[0]}')

        flux_lininterp = None

    elif config['detrending_method'] is None:
        time, flux = np.concatenate(time_arrs), np.concatenate(flux_arrs)
        trend = np.nanmedian(flux) * np.ones(len(flux))
        detrended_flux = flux / trend
        flux_lininterp = None
    else:
        raise ValueError(f'Detrending method not recognized: {config["detrending_method"]}')

    if plot_preprocessing_tce:
        utils_visualization.plot_flux_detrend(time,
                                              flux,
                                              trend,
                                              detrended_flux,
                                              target_uid,
                                              config['plot_dir'],
                                              f'2_detrendedflux',
                                              flux_interp=flux_lininterp)

    return time, detrended_flux, trend


def centroid_preprocessing(all_time, all_centroids, target_position, add_info, target_uid, config, plot_preprocessing_tce):
    """ Preprocess the centroid timeseries.

    :param all_time: list of NumPy arrays, timestamps
    :param all_centroids: dictionary for the two centroid coordinates coded as 'x' and 'y'. Each key maps to a list of
        NumPy arrays for the respective centroid coordinate time series
    # :param avg_centroid_oot: dictionary for the two centroid coordinates coded as 'x' and 'y'. Each key maps to the
    # estimate of the average out-of-transit centroid
    :param target_position: list, target star position in 'x' and 'y'
    :param add_info: dictionary, additional information such as quarters and modules
    :param target_uid: str, target unique id (TIC ID-Sector run/KIC ID)
    :param config: dict, preprocessing parameters
    :param plot_preprocessing_tce: bool, set to True to plot figures related to different preprocessing steps
    
    :return:
        NumPy array, centroid timestamps
        dict, containing detrended centroid time series for 'x' and 'y' coordinates and timestamps 'time'
        dict, average out-of-transit centroid position 'x', and 'y' coordinates
    """

    # copy arrays
    time_arrs, centroid_dict, intransit_cadences_target = (
        [np.array(el) for el in all_time],
        {coord: [np.array(el) for el in centroid_arrs] for coord, centroid_arrs in all_centroids.items()},
        [np.array(el) for el in config['intransit_cadences_target']],
    )

    if np.isnan(np.concatenate(centroid_dict['x'])).all():  # when there's no centroid data
        time_centroid = np.concatenate(time_arrs)
        centroid_dist = np.zeros(len(time_centroid), dtype='float')
        avg_centroid_oot = {coord: np.zeros(len(time_centroid), dtype='float') for coord in centroid_dict}

        report_exclusion(f'No available flux-weighted centroid data for target {target_uid}. '
                         f'Setting transit offset distance from target to zero.',
                         config['exclusion_logs_dir'] / f'exclusions_{target_uid}.txt')

        return time_centroid, centroid_dist, avg_centroid_oot

    # remove missing NaN values from the time series
    time_arrs, centroid_dict['x'], centroid_dict['y'], intransit_cadences_target = (
        remove_non_finite_values([
            time_arrs,
            centroid_dict['x'],
            centroid_dict['y'],
            intransit_cadences_target,
        ]
        ))

    # pixel coordinate transformation for targets on module 13 for Kepler
    if config['px_coordinates'] and config['satellite'] == 'kepler':
        if add_info['module'][0] == 13:
            centroid_dict = kepler_transform_pxcoordinates_mod13(centroid_dict, add_info)

    # get out-of-transit indices for the centroid time series
    centroid_oot = {coord: [centroids[~intransit_cadences_target_arr] for intransit_cadences_target_arr, centroids
                            in zip(intransit_cadences_target, centroid_dict[coord])] for coord in centroid_dict}
    # estimate average out-of-transit centroid as the median
    avg_centroid_oot = {coord: np.nanmedian(np.concatenate(centroid_oot[coord])) for coord in centroid_oot}
    # if there is no valid out-of-transit cadences, use median of the whole centroid array
    avg_centroid_oot = {coord: np.nanmedian(np.concatenate(centroid_dict[coord])) if np.isnan(avg_centroid_oot_coord)
    else avg_centroid_oot_coord for coord, avg_centroid_oot_coord in avg_centroid_oot.items()}

    # detrend centroid motion time series
    detrended_centroid_dict = {centroid_coord: {'detrended': None, 'trend': None}
                               for centroid_coord in centroid_dict.keys()}
    for centroid_coord, centroid_coord_data in centroid_dict.items():

        if config['detrending_method'] == 'spline':
            time, detrended_centroid, trend, centroid_lininterp = (
                detrend_flux_using_spline(centroid_coord_data, time_arrs, intransit_cadences_target, config))
            detrended_centroid_dict[centroid_coord]['linear_interp'] = centroid_lininterp

        elif config['detrending_method'] == 'savitzky-golay':
            # convert data to lightcurve object
            time, centroid_arr = (np.concatenate(time_arrs), np.concatenate(centroid_coord_data))
            if isinstance(intransit_cadences_target, list):
                intransit_cadences_target = np.concatenate(intransit_cadences_target)

            lc = lk.LightCurve(data={'time': np.array(time), 'flux': np.array(centroid_arr)})
            # # mask in-transit cadences
            # mask_in_transit = lc.create_transit_mask(tce['tce_period'], tce['tce_time0bk'],
            #                                          config['duration_gapped_primary'])

            _, detrended_centroid, trend, models_info_df = detrend_flux_using_sg_filter(lc,
                                                                                        intransit_cadences_target,
                                                                                        config['sg_win_len'],
                                                                                        config['sg_sigma'],
                                                                                        config['sg_max_poly_order'],
                                                                                        config['sg_penalty_weight'],
                                                                                        config['sg_break_tolerance'],
                                                                                        )

            logger.info(f'[{target_uid}] SG detrending model centroid {centroid_coord} data info. '
                        f'Chosen polynomial order: {models_info_df.index[0]}')

        elif config['detrending_method'] is None:
            time, centroid_arr = (np.concatenate(time_arrs), np.concatenate(centroid_coord_data))
            trend = np.nanmedian(centroid_arr) * np.ones(len(centroid_arr))
            detrended_centroid = centroid_arr / trend

        else:
            raise ValueError(f'Detrending method not recognized: {config["detrending_method"]}')

        detrended_centroid_dict[centroid_coord]['detrended'] = detrended_centroid
        detrended_centroid_dict[centroid_coord]['trend'] = trend

    time_centroid, centroid_dict_concat = (np.concatenate(time_arrs),
                                           {coord: np.concatenate(centroid_coord) for coord, centroid_coord
                                            in centroid_dict.items()})

    # recover the original central tendency of the centroid by multiplying the detrended centroid by the average
    # centroid
    for centroid_coord in detrended_centroid_dict:
        detrended_centroid_dict[centroid_coord]['detrended'] *= avg_centroid_oot[centroid_coord]
        detrended_centroid_dict[centroid_coord]['trend'] *= avg_centroid_oot[centroid_coord]

    if plot_preprocessing_tce:
        utils_visualization.plot_centroids(time_centroid,
                                           centroid_dict_concat,
                                           detrended_centroid_dict,
                                           target_uid,
                                           config,
                                           config['plot_dir'] /
                                           f'{target_uid}_'
                                           f'3_1_detrendedcentroids.png',
                                           config['px_coordinates'],
                                           target_position,
                                           config['delta_dec'],
                                           )

        # for coord, centroid_coord_data in detrended_centroid_dict.items():
        #     utils_visualization.plot_residual(time_centroid,
        #                                       centroid_coord_data['residual'],
        #                                       tce,
        #                                       config['plot_dir'],
        #                                       f'3_residual_centroid{coord}_aug{tce["augmentation_idx"]}',
        #                                       )

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

    
    return time_centroid, detrended_centroid_dict, avg_centroid_oot 


def process_light_curve(data, config, target_uid, plot_preprocessing_tce=False):
    """ Detrends different time series (e.g., detrending) such as flux and centroid motion.

    Args:
      data: dictionary containing raw lc data
      config: dict, holds preprocessing parameters
      target_uid: str, target unique id (TIC ID-Sector run/KIC ID)
      plot_preprocessing_tce: bool, if True plots figures for several steps in the preprocessing pipeline

    Returns:
      detrended_data: dict, containing detrended data for different time series
    """

    # FIXME: what is this for
    if config['satellite'] == 'kepler':
        add_info_centr = {'quarter': data['quarter'], 'module': data['module']}
    else:
        add_info_centr = None

    flux_time, flux, flux_trend = flux_preprocessing(
        data['all_time'],
        data['all_flux'],
        target_uid,
        config,
        plot_preprocessing_tce
        )

    centroid_time, detrended_centroid_dict, avg_centroid_oot = centroid_preprocessing(
        data['all_time'],
        data['all_centroids'],
        data['target_position'],
        add_info_centr,
        target_uid,
        config,
        plot_preprocessing_tce
        )

    if config['get_momentum_dump']:
        momentum_dump_time, momentum_dump = remove_non_finite_values([data['time_momentum_dump'],
                                                                      data['momentum_dump']])
        momentum_dump_time, momentum_dump = np.concatenate(momentum_dump_time), np.concatenate(momentum_dump)

        if plot_preprocessing_tce:
            utils_visualization.plot_momentum_dump_timeseries(momentum_dump_time,
                                                              momentum_dump,
                                                              config['plot_dir'] /
                                                              f'{target_uid}_'
                                                              f'4_momentum_dump_timeseries.png')

    # dictionary with detrended time series
    detrended_data = {
        'flux_time': flux_time,
        'flux': flux,
        'flux_trend': flux_trend,
        'centroid_time': centroid_time,
        'centroid': {centroid_coord: centroid_data['detrended'] 
                     for centroid_coord, centroid_data in detrended_centroid_dict.items()},
        'centroid_trend': {centroid_coord: centroid_data['trend'] 
                     for centroid_coord, centroid_data in detrended_centroid_dict.items()},
        'avg_centroid_oot': avg_centroid_oot,
    }

    if config['get_momentum_dump']:
        detrended_data['momentum_dump'] = momentum_dump
        detrended_data['momentum_dump_time'] = momentum_dump_time

    detrended_data['errors'] = data['errors']
    if config['satellite'] == 'kepler':
        detrended_data['quarter_timestamps'] = data['quarter_timestamps']

    return detrended_data


def create_example_stats(binned_timeseries, odd_even_flag, t_min_transit, t_max_transit, config):
    """ Computes several summary statistics based on the preprocessed data for a given TCE.

    Args:
        binned_timeseries: dict, binned timeseries
        # odd_data: dict, odd data
        # even_data: dict, even data
        odd_even_flag: str, odd and even flag
        # num_transits: dict, number of transits per timeseries
        # inds_bin_nan: dict, indices of bins with missing values per views
        t_min_transit: float, left edge in-transit phase time
        t_max_transit: float, right edge in-transit phase time
        config: dict, preprocessing parameters for the run

    Returns: dict, with summary statistics for the example

    """

    # initialize dictionary with data for the preprocessing table
    example_stats = {}

    # number of transits
    example_stats.update({f'num_transits_{ts_name}': ts_data[3] for ts_name, ts_data in binned_timeseries.items()
                          if 'stat' not in ts_name})
    example_stats.update({stat_name: stat_val for stat_name, stat_val in binned_timeseries.items()
                          if 'stat' in stat_name})

    # centroid
    glob_centr_oot_bins = (binned_timeseries['centroid_offset_distance_to_target_global'][0] < t_min_transit) | \
                          (binned_timeseries['centroid_offset_distance_to_target_global'][0] > t_max_transit)
    loc_centr_it_bins = (binned_timeseries['centroid_offset_distance_to_target_local'][0] >= t_min_transit) & \
                        (binned_timeseries['centroid_offset_distance_to_target_local'][0] <= t_max_transit)
    example_stats['avg_oot_centroid_offset'] = np.median(
        binned_timeseries['centroid_offset_distance_to_target_global'][1][glob_centr_oot_bins])
    example_stats['std_oot_centroid_offset'] = mad_std(
        binned_timeseries['centroid_offset_distance_to_target_global'][1][glob_centr_oot_bins])
    example_stats['peak_centroid_offset'] = np.max(
        binned_timeseries['centroid_offset_distance_to_target_local'][1][loc_centr_it_bins]) - \
                                            example_stats['avg_oot_centroid_offset']

    example_stats['mid_local_flux_shift'] = np.argmin(
        binned_timeseries['centroid_offset_distance_to_target_local'][1]) - int(config['num_bins_loc'] / 2)
    example_stats['mid_global_flux_shift'] = np.argmin(
        binned_timeseries['centroid_offset_distance_to_target_global'][1]) - int(config['num_bins_glob'] / 2)

    # flux
    loc_flux_it_bins = (binned_timeseries['flux_local'][0] >= t_min_transit) & \
                       (binned_timeseries['flux_local'][0] <= t_max_transit)
    ind = np.argmin(binned_timeseries['flux_local'][1][loc_flux_it_bins])
    example_stats['transit_depth_hat'] = (1 - binned_timeseries['flux_local'][1][loc_flux_it_bins][ind]) * 1e6
    example_stats['transit_depth_hat_se'] = binned_timeseries['flux_local'][1][loc_flux_it_bins][ind] * 1e6

    # odd/even flux
    # example_stats.update({f'odd_{key}': val for key, val in odd_data.items()
    #                       if key not in ['local_flux_view', 'local_flux_view_se', 'binned_time']})
    # example_stats.update({f'even_{key}': val for key, val in even_data.items()
    #                       if key not in ['local_flux_view', 'local_flux_view_se', 'binned_time']})
    example_stats['odd_even_flag'] = odd_even_flag

    loc_flux_odd_it_bins = (binned_timeseries['flux_odd_local'][0] >= t_min_transit) & \
                           (binned_timeseries['flux_odd_local'][0] <= t_max_transit)
    ind = np.argmin(binned_timeseries['flux_odd_local'][1][loc_flux_odd_it_bins])
    example_stats['transit_depth_odd_hat'] = (1 - binned_timeseries['flux_odd_local'][1][loc_flux_odd_it_bins][
        ind]) * 1e6
    example_stats['transit_depth_odd_hat_se'] = binned_timeseries['flux_odd_local'][2][loc_flux_odd_it_bins][ind] * 1e6

    loc_flux_even_it_bins = (binned_timeseries['flux_even_local'][0] >= t_min_transit) & \
                            (binned_timeseries['flux_even_local'][0] <= t_max_transit)
    ind = np.argmin(binned_timeseries['flux_even_local'][1][loc_flux_even_it_bins])
    example_stats['transit_depth_even_hat'] = (1 - binned_timeseries['flux_even_local'][1][loc_flux_even_it_bins][
        ind]) * 1e6
    example_stats['transit_depth_even_hat_se'] = binned_timeseries['flux_even_local'][2][loc_flux_even_it_bins][
                                                     ind] * 1e6

    # weak secondary flux
    loc_flux_wks_it_bins = (binned_timeseries['flux_weak_secondary_local'][0] >= t_min_transit) & \
                           (binned_timeseries['flux_weak_secondary_local'][0] <= t_max_transit)
    ind = np.argmin(binned_timeseries['flux_weak_secondary_local'][1][loc_flux_wks_it_bins])
    example_stats['wks_transit_depth_hat'] = (1 -
                                              binned_timeseries['flux_weak_secondary_local'][1][loc_flux_wks_it_bins][
                                                  ind]) * 1e6
    example_stats['wks_transit_depth_hat_se'] = binned_timeseries['flux_weak_secondary_local'][2][loc_flux_wks_it_bins][
                                                    ind] * 1e6

    return example_stats


def generate_odd_even_binned_views(data, tce, config, norm_stats, plot_preprocessing_tce):
    """ Generates odd and even flux local binned time series from phase folded data.

         Args:
           data: dictionary, containing preprocessed (detrended and phase-folded) time series
           tce: pandas Series, contains TCE parameters and DV statistics. Additional items are included as features in the
           example output.
           config: dict; preprocessing parameters.
           norm_stats: dict, normalization statistics
           plot_preprocessing_tce: bool, if True, generates figure plots

         Returns:
           A dict with keys 'flux_odd_local' and 'flux_even_local' plus their normalization versions ({}_norm). Each key
           maps to a tuple of the binned time, binned time series, the companion variability binned time series, and the
           number of phase folded cycles used when binning. Additionally, there are two keys that map to the absolute
           minimum of the odd and even local fluxes, plus some statistics that include the standard deviation and
           standard error of the out-of-transit flux values in the phase folded time series.
           A str flag describing the preprocessing of the odd and even phase folded data into binned time series
     """
    odd_data, even_data, odd_even_flag = create_odd_even_views(data['flux_odd'][0],
                                                               data['flux_odd'][1],
                                                               data['flux_even'][0],
                                                               data['flux_even'][1],
                                                               data['flux_odd'][2],
                                                               data['flux_even'][2],
                                                               tce,
                                                               config)

    # normalize odd and even binned data
    odd_data['se_oot'] = odd_data['se_oot'] / norm_stats['sigma']
    odd_data['std_oot_bin'] = odd_data['std_oot_bin'] / norm_stats['sigma']
    even_data['se_oot'] = even_data['se_oot'] / norm_stats['sigma']
    even_data['std_oot_bin'] = even_data['std_oot_bin'] / norm_stats['sigma']

    odd_local_abs_min = np.abs(np.min(odd_data['local_flux_view'] - np.nanmedian(odd_data['local_flux_view'])))
    flux_odd_local_norm = (odd_data['local_flux_view'] - norm_stats['mu']) / norm_stats['sigma']
    flux_odd_local_var_norm = odd_data['local_flux_view_se'] / norm_stats['sigma']

    even_local_abs_min = np.abs(np.min(even_data['local_flux_view'] - np.nanmedian(even_data['local_flux_view'])))
    flux_even_local_norm = (even_data['local_flux_view'] - norm_stats['mu']) / norm_stats['sigma']
    flux_even_local_var_norm = even_data['local_flux_view_se'] / norm_stats['sigma']

    binned_timeseries = {
        'flux_odd_local': (odd_data['binned_time'], odd_data['local_flux_view'], odd_data['local_flux_view_se'],
                           data['flux_odd'][2]),
        'flux_odd_local_norm': (odd_data['binned_time'], flux_odd_local_norm, flux_odd_local_var_norm,
                                data['flux_odd'][2]),
        'flux_even_local': (even_data['binned_time'], even_data['local_flux_view'], even_data['local_flux_view_se'],
                            data['flux_even'][2]),
        'flux_even_local_norm': (even_data['binned_time'], flux_even_local_norm, flux_even_local_var_norm,
                                 data['flux_even'][2]),
    }

    if plot_preprocessing_tce:
        utils_visualization.plot_odd_even(binned_timeseries,
                                          {ts_name: data[ts_name] for ts_name in ['flux_odd', 'flux_even']},
                                          tce,
                                          config,
                                          config['plot_dir'] /
                                          f'{tce["uid"]}_{tce["label"]}_8_1_oddeven_transitdepth_phasefoldedbinned_'
                                          f'timeseries.png'
                                          )

    # add statistics
    binned_timeseries.update(
        {
            'flux_odd_local_stat_se_oot': odd_data['se_oot'],
            'flux_odd_local_stat_std_oot_bin': odd_data['std_oot_bin'],
            'flux_even_local_stat_se_oot': even_data['se_oot'],
            'flux_even_local_stat_std_oot_bin': even_data['std_oot_bin'],

            'flux_odd_local_stat_abs_min': odd_local_abs_min,
            'flux_even_local_stat_abs_min': even_local_abs_min,
        }
    )

    return binned_timeseries, odd_even_flag


def generate_flux_binned_views(data, tce, config, plot_preprocessing_tce):
    """ Generates flux binned time series from phase folded data.

         Args:
           data: dictionary, containing preprocessed (detrended and phase-folded) time series
           tce: pandas Series, contains TCE parameters and DV statistics. Additional items are included as features in the
           example output.
           config: dict; preprocessing parameters.
           plot_preprocessing_tce: bool, if True, generates figure plots

         Returns:
           A dict with keys 'flux_global' and 'flux_local', their corresponding unfolded versions
           'flux_{global_local}_unfolded', plus their normalization versions ({}_norm). Each key maps to a tuple of the
           binned time, binned time series, the companion variability binned time series, and the number of phase
           folded cycles used when binning. Additionally, there are two keys that map to the absolute minimum of the
           global and local flux, respectively.
           A dict with two keys map to the mu and sigma normalization statistics of the global flux.
           A dict with two keys map to the mu and sigma normalization statistics of the local flux.

     """

    # create global flux view
    glob_flux_view, glob_binned_time, glob_flux_view_var, _, bin_counts = \
        global_view(data['flux'][0],
                    data['flux'][1],
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

    # create unfolded global flux views
    unfolded_glob_flux_view = []
    unfolded_glob_flux_view_var = []
    for phase_i in range(data['flux_unfolded'][2]):
        unfolded_glob_flux_view_phase, binned_time, unfolded_glob_flux_view_var_phase, _, bin_counts = \
            global_view(data['flux_unfolded'][0][phase_i],
                        data['flux_unfolded'][1][phase_i],
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
                                           config['plot_dir'] /
                                           f'{tce["uid"]}_{tce["label"]}_'
                                           f'6_riverplot_flux_aug.png')

    # create local flux view
    loc_flux_view, loc_binned_time, loc_flux_view_var, _, bin_counts = \
        local_view(data['flux'][0],
                   data['flux'][1],
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

    # set local flux view to Gaussian noise using statistics from flux phase folded time series when there are no data
    # to create the local view
    if np.all(np.isnan(loc_flux_view)):
        report_exclusion(
            f'No data available for local flux view. Setting it to median and var view to mad std using '
            f'statistics from the flux phase folded time series.',
            config['exclusion_logs_dir'] / f'exclusions-{tce["uid"]}.txt')
        mu, sigma = np.nanmedian(data['flux'][1]), mad_std(data['flux'][1], ignore_nan=True)

        if np.isfinite(mu) and np.isfinite(sigma):
            loc_flux_view = mu * np.ones(config['num_bins_loc'])
            loc_flux_view_var = sigma * np.ones(config['num_bins_loc'])
        else:
            loc_flux_view = np.zeros(config['num_bins_loc'])
            loc_flux_view_var = np.zeros(config['num_bins_loc'])

    bin_counts[bin_counts == 0] = max(1, np.median(bin_counts))
    loc_flux_view_var /= np.sqrt(bin_counts)

    # create unfolded local flux views
    unfolded_loc_flux_view = []
    unfolded_loc_flux_view_var = []
    for phase_i in range(data['flux_unfolded'][2]):
        unfolded_loc_flux_view_phase, binned_time, unfolded_loc_flux_view_var_phase, _, bin_counts = \
            local_view(data['flux_unfolded'][0][phase_i],
                       data['flux_unfolded'][1][phase_i],
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

        unfolded_loc_flux_view.append(unfolded_loc_flux_view_phase)
        unfolded_loc_flux_view_var.append(unfolded_loc_flux_view_var_phase)

    unfolded_loc_flux_view = np.array(unfolded_loc_flux_view)
    unfolded_loc_flux_view_var = np.array(unfolded_loc_flux_view_var)

    binned_timeseries = {
        'flux_global': (glob_binned_time, glob_flux_view, glob_flux_view_var, data['flux'][2]),
        'flux_local': (loc_binned_time, loc_flux_view, loc_flux_view_var, data['flux'][2]),
        'flux_global_unfolded': (None, unfolded_glob_flux_view, unfolded_glob_flux_view_var,
                                 data['flux_unfolded'][2]),
        'flux_local_unfolded': (None, unfolded_loc_flux_view, unfolded_loc_flux_view_var,
                                data['flux_unfolded'][2])
    }

    # normalize data
    norm_flux_global_stats = {
        'mu': np.median(glob_flux_view),
    }
    norm_flux_global_stats['sigma'] = np.abs(np.min(glob_flux_view - norm_flux_global_stats['mu'])) + NORM_SIGMA_EPS

    norm_flux_local_stats = {
        'mu': np.median(loc_flux_view),
    }
    norm_flux_local_stats['sigma'] = np.abs(np.min(loc_flux_view - norm_flux_local_stats['mu'])) + NORM_SIGMA_EPS

    # global normalization
    binned_timeseries_norm = {f'{ts_name}_norm': (binned_time,
                                                  (binned_values - norm_flux_global_stats['mu']) /
                                                  norm_flux_global_stats['sigma'],
                                                  binned_values_var / norm_flux_global_stats['sigma'],
                                                  n_transits)
                              for ts_name, (binned_time, binned_values, binned_values_var, n_transits) in
                              binned_timeseries.items() if 'global' in ts_name}
    # local normalization
    binned_timeseries_norm.update(
        {f'{ts_name}_norm': (binned_time,
                             (binned_values - norm_flux_local_stats['mu']) /
                             norm_flux_local_stats['sigma'],
                             binned_values_var / norm_flux_local_stats['sigma'],
                             n_transits)
         for ts_name, (binned_time, binned_values, binned_values_var, n_transits) in
         binned_timeseries.items() if 'local' in ts_name}
    )

    binned_timeseries.update(binned_timeseries_norm)

    # add absolute minimum statistics for global and local fluxes
    binned_timeseries.update(
        {
            'flux_global_stat_abs_min': norm_flux_global_stats['sigma'],
            'flux_local_stat_abs_min': norm_flux_local_stats['sigma'],
        }
    )

    return binned_timeseries, norm_flux_global_stats, norm_flux_local_stats


def generate_flux_trend_binned_views(data, tce, config, plot_preprocessing_tce):
    """ Generates flux trend binned time series from phase folded data.

            Args:
              data: dictionary, containing preprocessed (detrended and phase-folded) time series
              tce: pandas Series, contains TCE parameters and DV statistics. Additional items are included as features
              in the example output.
              config: dict; preprocessing parameters.
              plot_preprocessing_tce: bool, if True, generates figure plots

            Returns:
              A dict with keys 'flux_trend_global' and 'flux_trend_local', their corresponding unfolded versions
              'flux_trend_{global_local}_unfolded', plus their normalization versions ({}_norm). Each key maps to a
              tuple of the binned time, binned time series, the companion variability binned time series, and the number
               of phase folded cycles used when binning. Additionally, there are two keys that map to the max and min
               statistics of the global trend flux.
    """

    # create global flux view
    glob_flux_view, glob_binned_time, glob_flux_view_var, _, bin_counts = \
        global_view(data['flux_trend'][0],
                    data['flux_trend'][1],
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

    # create unfolded global flux views
    unfolded_glob_flux_view = []
    unfolded_glob_flux_view_var = []
    for phase_i in range(data['flux_trend_unfolded'][2]):
        unfolded_glob_flux_view_phase, binned_time, unfolded_glob_flux_view_var_phase, _, bin_counts = \
            global_view(data['flux_trend_unfolded'][0][phase_i],
                        data['flux_trend_unfolded'][1][phase_i],
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
                                           config['plot_dir'] /
                                           f'{tce["uid"]}_{tce["label"]}_'
                                           f'7_1_riverplot_flux_trend.png')

    # create local flux view
    loc_flux_view, loc_binned_time, loc_flux_view_var, _, bin_counts = \
        local_view(data['flux_trend'][0],
                   data['flux_trend'][1],
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

    # set local flux view to Gaussian noise using statistics from flux phase folded time series when there are no data
    # to create the local view
    if np.all(np.isnan(loc_flux_view)):
        report_exclusion(
            f'No data available for local flux trend view. Setting it to median and var view to mad std using '
            f'statistics from the flux trend phase folded time series.',
            config['exclusion_logs_dir'] / f'exclusions-{tce["uid"]}.txt')
        mu, sigma = np.nanmedian(data['flux_trend'][1]), mad_std(data['flux_trend'][1], ignore_nan=True)

        if np.isfinite(mu) and np.isfinite(sigma):
            loc_flux_view = mu * np.ones(config['num_bins_loc'])
            loc_flux_view_var = sigma * np.ones(config['num_bins_loc'])
        else:
            loc_flux_view = np.zeros(config['num_bins_loc'])
            loc_flux_view_var = np.zeros(config['num_bins_loc'])

    bin_counts[bin_counts == 0] = max(1, np.median(bin_counts))
    loc_flux_view_var /= np.sqrt(bin_counts)

    unfolded_loc_flux_view = []
    unfolded_loc_flux_view_var = []
    for phase_i in range(data['flux_trend_unfolded'][2]):
        unfolded_loc_flux_view_phase, binned_time, unfolded_loc_flux_view_var_phase, _, bin_counts = \
            local_view(data['flux_trend_unfolded'][0][phase_i],
                       data['flux_trend_unfolded'][1][phase_i],
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

        if np.all(~np.isfinite(unfolded_loc_flux_view_phase)):
            unfolded_loc_flux_view_phase = np.median(glob_flux_view) * np.ones(config['num_bins_loc'])
            unfolded_loc_flux_view_var_phase = (
                    mad_std(glob_flux_view, ignore_nan=False) * np.ones(config['num_bins_loc']))

        bin_counts[bin_counts == 0] = max(1, np.median(bin_counts))
        unfolded_loc_flux_view_var_phase /= np.sqrt(bin_counts)

        unfolded_loc_flux_view.append(unfolded_loc_flux_view_phase)
        unfolded_loc_flux_view_var.append(unfolded_loc_flux_view_var_phase)

    unfolded_loc_flux_view = np.array(unfolded_loc_flux_view)
    unfolded_loc_flux_view_var = np.array(unfolded_loc_flux_view_var)

    # normalize data
    # norm_flux_median = np.median(glob_flux_view)
    # norm_flux_factor = np.abs(np.min(glob_flux_view - norm_flux_median)) + NORM_SIGMA_EPS
    norm_flux_max = np.max(glob_flux_view)
    norm_flux_min = np.min(glob_flux_view)
    delta_flux_max_min = norm_flux_max - norm_flux_min + NORM_SIGMA_EPS

    binned_timeseries = {
        'flux_trend_global': (glob_binned_time, glob_flux_view, glob_flux_view_var, data['flux'][2]),
        'flux_trend_local': (loc_binned_time, loc_flux_view, loc_flux_view_var, data['flux'][2]),
        'flux_trend_global_unfolded': (None, unfolded_glob_flux_view, unfolded_glob_flux_view_var,
                                       data['flux_unfolded'][2]),
        'flux_trend_local_unfolded': (None, unfolded_loc_flux_view, unfolded_loc_flux_view_var,
                                      data['flux_unfolded'][2]),
    }

    binned_timeseries_norm = {f'{ts_name}_norm': (binned_time,
                                                  (binned_values - norm_flux_min) / delta_flux_max_min,
                                                  binned_values_var / delta_flux_max_min,
                                                  n_transits)
                              for ts_name, (binned_time, binned_values, binned_values_var, n_transits) in
                              binned_timeseries.items()}

    binned_timeseries.update(binned_timeseries_norm)

    # add max and min statistics
    binned_timeseries.update(
        {
            'flux_trend_global_stat_max': norm_flux_max,
            'flux_trend_global_stat_min': norm_flux_min
        }
    )

    if plot_preprocessing_tce:
        utils_visualization.plot_phasefolded_and_binned_trend(
            data, binned_timeseries,
            tce,
            config['plot_dir'] /
            f'{tce["uid"]}_{tce["label"]}_'
            f'7_2_phasefoldedbinned_trend.png')

    return binned_timeseries


def generate_weak_secondary_binned_views(data, tce, config, norm_stats=None, plot_preprocessing_tce=False):
    """ Generates weak secondary flux local binned time series from phase folded data.

        Args:
          data: dictionary, containing preprocessed (detrended and phase-folded) time series
          tce: pandas Series, contains TCE parameters and DV statistics. Additional items are included as features in the
          example output.
          config: dict; preprocessing parameters.
          norm_stats: dict, containing normalization statistics
          plot_preprocessing_tce: boolean, if True plots figure

        Returns:
          A dict with keys 'flux_weak_secondary_local' and 'flux_weak_secondary_local_norm'
          (before/after normalization), each containing a tuple of the local binned time, local binned time series, the
          companion variability binned time series, and the number of phase folded cycles used when binning.
          Additionally, there is one key that maps to the absolute minimum of the local weak secondary flux.
    """

    # gap primary in-transit cadences in the phase folded secondary time series by imputing them with Gaussian noise
    # estimated from the full phase folded time series
    midpoint_primary_transit_time = -tce['tce_maxmesd']
    tr_dur_f_primary_gap = 1

    primary_gap_indices = np.logical_and(data['flux_weak_secondary'][0] > midpoint_primary_transit_time -
                                         tr_dur_f_primary_gap * tce['tce_duration'],
                                         data['flux_weak_secondary'][0] < midpoint_primary_transit_time +
                                         tr_dur_f_primary_gap * tce['tce_duration'],
                                         )
    mu = np.nanmedian(data['flux_weak_secondary'][1])
    sigma = mad_std(data['flux_weak_secondary'][1], ignore_nan=True)
    rng = np.random.default_rng(seed=config['random_seed'])
    data['flux_weak_secondary'][1][primary_gap_indices] = rng.normal(mu, sigma, primary_gap_indices.sum())

    # create local view for the weak secondary flux
    loc_weak_secondary_view, binned_time, loc_weak_secondary_view_var, _, bin_counts = \
        local_view(data['flux_weak_secondary'][0],
                   data['flux_weak_secondary'][1],
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

    # set local weak secondary flux view to Gaussian noise using statistics from the weak secondary flux phase folded
    # time series when there are no data to create the local view
    if np.all(np.isnan(loc_weak_secondary_view)):
        report_exclusion(
            f'No data available for local weak secondary view. Setting it to median and var view to mad std using '
            f'statistics from the weak secondary phase folded time series.',
            config['exclusion_logs_dir'] / f'exclusions-{tce["uid"]}.txt')
        mu, sigma = np.nanmedian(data['flux_weak_secondary'][1]), mad_std(data['flux_weak_secondary'][1],
                                                                          ignore_nan=True)

        if np.isfinite(mu) and np.isfinite(sigma):
            loc_weak_secondary_view = mu * np.ones(config['num_bins_loc'])
            loc_weak_secondary_view_var = sigma * np.ones(config['num_bins_loc'])
        else:
            loc_weak_secondary_view = np.zeros(config['num_bins_loc'])
            loc_weak_secondary_view_var = np.zeros(config['num_bins_loc'])

        # _, _, _, _, bin_counts = \
        #     global_view(data['flux_weak_secondary'][0],
        #                 data['flux_weak_secondary'][1],
        #                 tce['tce_period'],
        #                 tce=tce,
        #                 normalize=False,
        #                 centering=False,
        #                 num_bins=config['num_bins_glob'],
        #                 bin_width_factor=config['bin_width_factor_glob'],
        #                 report={'config': config, 'tce': tce, 'view': 'global_flux_view'},
        #                 tce_duration=tce['tce_duration']
        #                 )
        # mu, sigma = np.nanmedian(bin_counts), mad_std(bin_counts, ignore_nan=True)
        # bin_counts = mu * np.ones(config['num_bins_loc'])

    bin_counts[bin_counts == 0] = max(1, np.median(bin_counts))
    loc_weak_secondary_view_var /= np.sqrt(bin_counts)

    # normalize binned time series
    if norm_stats is None:  # normalize by self absolute minimum
        norm_stats = {'mu': np.median(loc_weak_secondary_view)}

        norm_stats['sigma'] = np.abs(np.min(loc_weak_secondary_view - norm_stats['mu'])) + NORM_SIGMA_EPS

    loc_weak_secondary_view_norm = \
        centering_and_normalization(loc_weak_secondary_view,
                                    norm_stats['mu'],
                                    norm_stats['sigma'],
                                    report={'config': config, 'tce': tce, 'view': 'weak secondary flux'}
                                    )
    loc_weak_secondary_view_var_norm = loc_weak_secondary_view_var / norm_stats['sigma']

    # # normalize by flux absolute minimum
    # norm_wks_factor_fluxnorm = flux_views_stats['min'][('local', 'global')['global' in flux_view]]
    # views_aux[f'{flux_view}_fluxnorm'] = \
    #     centering_and_normalization(weak_secondary_flux_views[flux_view],
    #                                 flux_views_stats['median'][('local', 'global')['global' in flux_view]],
    #                                 norm_wks_factor_fluxnorm,
    #                                 report={'config': config, 'tce': tce, 'view': flux_view}
    #                                 )
    # # normalize by max between flux absolute minimum and wks flux absolute minimum
    # norm_wks_factor_fluxselfnorm = max(norm_wks_factor_fluxnorm, norm_wks_factor_selfnorm),
    # views_aux[f'{flux_view}_max_flux-wks_norm'] = \
    #     centering_and_normalization(weak_secondary_flux_views[flux_view],
    #                                 flux_views_stats['median'][('local', 'global')['global' in flux_view]],
    #                                 norm_wks_factor_fluxselfnorm,
    #                                 report={'config': config, 'tce': tce, 'view': flux_view}
    #                                 )

    binned_timeseries = {
        'flux_weak_secondary_local': (binned_time,
                                      loc_weak_secondary_view,
                                      loc_weak_secondary_view_var,
                                      data['flux_weak_secondary'][2]),
        'flux_weak_secondary_local_norm': (binned_time,
                                           loc_weak_secondary_view_norm,
                                           loc_weak_secondary_view_var_norm,
                                           data['flux_weak_secondary'][2]),

        'flux_weak_secondary_local_stat_abs_min': norm_stats['sigma'],
    }

    if plot_preprocessing_tce:
        utils_visualization.plot_phasefolded_and_binned_weak_secondary_flux(
            data, binned_timeseries,
            tce,
            config['plot_dir'] /
            f'{tce["uid"]}_{tce["label"]}_'
            f'8_3_flux_weak_secondary.png')

    return binned_timeseries


def generate_centroid_binned_views(data, tce, config):
    """ Generates centroid binned time series from phase folded data.

    Args:
      data: dictionary, containing preprocessed (detrended and phase-folded) time series
      tce: pandas Series, contains TCE parameters and DV statistics. Additional items are included as features in the
      example output.
      config: dict; preprocessing parameters.

    Returns:
      A dict with keys 'centroid_offset_distance_to_target_global' and 'centroid_offset_distance_to_target_global', each
      containing a tuple of the global/local binned time, global/local binned time series, the companion variability
      binned time series, and the number of phase folded cycles used when binning. Additionally, there are two keys that
      map to the maximum local and global centroid phase folded and binned time series.
    """

    glob_centr_view, glob_binned_time, glob_centr_view_var, _, bin_counts = \
        global_view(data['centroid_offset_distance_to_target'][0],
                    data['centroid_offset_distance_to_target'][1],
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

    loc_centr_view, loc_binned_time, loc_centr_view_var, _, bin_counts = \
        local_view(data['centroid_offset_distance_to_target'][0],
                   data['centroid_offset_distance_to_target'][1],
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

    if np.isnan(loc_centr_view).all():
        report_exclusion(
            f'No data available for local centroid view. Setting it to median and var view to mad std using '
            f'statistics from the phase folded time series.',
            config['exclusion_logs_dir'] / f'exclusions-{tce["uid"]}.txt')
        t_min = max(-tce['tce_period'] / 2, -tce['tce_duration'] * config['num_durations'])
        t_max = min(tce['tce_period'] / 2, tce['tce_duration'] * config['num_durations'])
        loc_binned_time = np.linspace(t_min, t_max, config['num_bins_loc'], endpoint=True)
        mu = np.nanmedian(data['centroid_offset_distance_to_target'][1])
        sigma = mad_std(data['centroid_offset_distance_to_target'][1], ignore_nan=True)
        if np.isfinite(mu) and np.isfinite(sigma):
            loc_centr_view = mu * np.ones(config['num_bins_loc'], dtype='float')
            loc_centr_view_var = sigma * np.ones(config['num_bins_loc'], dtype='float')
        else:
            loc_centr_view = np.zeros(config['num_bins_loc'])
            loc_centr_view_var = np.zeros(config['num_bins_loc'])

    bin_counts[bin_counts == 0] = max(1, np.median(bin_counts))
    loc_centr_view_var /= np.sqrt(bin_counts)

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

    # centr_views = {'global_centr_view': glob_centr_view,
    #                'local_centr_view': loc_centr_view,
    #                # 'global_centr_view_medcmaxn': glob_centr_view_medc_abs_maxn,
    #                # 'local_centr_view_medcmaxn': loc_centr_view_medc_abs_maxn,
    #                # 'global_centr_view_medn': glob_centr_view_mn,
    #                # 'local_centr_view_medn': loc_centr_view_mn,
    #                # 'global_centr_view_medcmaxn_dir': glob_centr_view_medc_maxabsn,
    #                # 'local_centr_view_medcmaxn_dir': loc_centr_view_medc_maxabsn,
    #                # 'global_centr_fdl_view': glob_centr_fdl_view,
    #                # 'local_centr_fdl_view': loc_centr_fdl_view
    #                }
    #
    # # # adjust TESS centroids to Kepler by dividing by pixel scale factor
    # # if config['satellite'] == 'tess':
    # #     centr_views['global_centr_view_adjscl'] = glob_centr_view / config['tess_to_kepler_px_scale_factor']
    # #     centr_views['local_centr_view_adjscl'] = loc_centr_view / config['tess_to_kepler_px_scale_factor']
    #
    # centr_views_var = {'global_centr_view': glob_centr_view_var,
    #                    'local_centr_view': loc_centr_view_var,
    #                    }

    binned_timeseries = {
        'centroid_offset_distance_to_target_global': (glob_binned_time, glob_centr_view, glob_centr_view_var,
                                                      data['centroid_offset_distance_to_target'][2]),
        'centroid_offset_distance_to_target_local': (loc_binned_time, loc_centr_view, loc_centr_view_var,
                                                     data['centroid_offset_distance_to_target'][2]),

        'centroid_offset_distance_to_target_global_stat_abs_max': np.nanmax(glob_centr_view),
        'centroid_offset_distance_to_target_local_stat_abs_max': np.nanmax(loc_centr_view),

    }

    return binned_timeseries


def generate_momentum_dump_views(data, tce, config, plot_preprocessing_tce):
    """ Generates momentum dump binned time series from phase folded data.

    Args:
      data: dictionary, containing preprocessed (detrended and phase-folded) time series
      tce: pandas Series, contains TCE parameters and DV statistics. Additional items are included as features in the
      example output.
      config: dict; preprocessing parameters.
      plot_preprocessing_tce: bool, if True plots figures for some steps while generating the inputs.

    Returns:
      A dict with key 'momentum_dump_local' containing a tuple of the local binned time, local binned time
      series, the companion variability binned time series, and the number of phase folded cycles used when binning
      (set to -1 means unused)
    """

    loc_mom_dump_view, loc_mom_dump_view_var, binned_time = generate_view_momentum_dump(
        data['momentum_dump'][0],
        data['momentum_dump'][1],
        config['num_bins_loc'],
        config['bin_width_factor_loc'] * tce['tce_duration'],
        max(-tce['tce_period'] / 2, -tce['tce_duration'] * config['num_durations']),
        min(tce['tce_period'] / 2, tce['tce_duration'] * config['num_durations']),
    )

    if plot_preprocessing_tce:
        utils_visualization.plot_momentum_dump(loc_mom_dump_view,
                                               loc_mom_dump_view_var,
                                               binned_time, data['momentum_dump'][1],
                                               data['momentum_dump'][0],
                                               tce,
                                               config['plot_dir'] /
                                               f'{tce["uid"]}_{tce["label"]}_8_2_momentum_dump_phase_and_binned.png')

    binned_timeseries = {
        'momentum_dump_local': (binned_time, loc_mom_dump_view, loc_mom_dump_view_var, -1)
    }

    return binned_timeseries


def generate_example_for_tce(phase_folded_data, pgram_data, tce, config, plot_preprocessing_tce=False):
    """ Generates a tf.train.Example representing an input TCE.

    Args:
      phase_folded_data: dictionary, containing preprocessed (detrended and phase-folded) time series used to generate
      the binned time series (aka views)
      pgram_data: dict that maps to different computed periodograms (e.g., whether for the lc or transit pulse model
      data, whether smoothed, and whether normalized)
      tce: pandas Series, contains TCE parameters and DV statistics. Additional items are included as features in the
      example output.
      config: dict; preprocessing parameters.
      plot_preprocessing_tce: bool, if True plots figures for some steps while generating the inputs.

    Returns:
      A tf.train.Example containing features. These features can be time series, stellar parameters, statistical
      quantities, .... it returns None if some exception while creating these features occurs
    """

    # time interval for the transit
    t_min_transit, t_max_transit = max(-tce['tce_period'] / 2, -tce['tce_duration'] / 2), \
        min(tce['tce_period'] / 2, tce['tce_duration'] / 2)

    # try:

    # make output proto
    ex = tf.train.Example()

    # initialize binned time series
    binned_timeseries = {}
    # inds_bin_nan = {}

    # create binned time series for flux
    binned_timeseries_flux, binned_flux_global_norm_stats, binned_flux_local_norm_stats = (
        generate_flux_binned_views(phase_folded_data, tce, config, plot_preprocessing_tce))
    binned_timeseries.update(binned_timeseries_flux)

    # create binned time series for flux trend
    binned_timeseries_flux_trend = generate_flux_trend_binned_views(phase_folded_data,
                                                                    tce, config, plot_preprocessing_tce)
    binned_timeseries.update(binned_timeseries_flux_trend)

    # create binned time series for odd/even flux
    binned_timeseries_odd_even_flux, odd_even_flag = generate_odd_even_binned_views(phase_folded_data,
                                                                                    tce,
                                                                                    config,
                                                                                    binned_flux_local_norm_stats,
                                                                                    plot_preprocessing_tce)
    binned_timeseries.update(binned_timeseries_odd_even_flux)

    # create binned time series for weak secondary flux
    binned_timeseries_wksecondary = generate_weak_secondary_binned_views(phase_folded_data,
                                                                         tce,
                                                                         config,
                                                                         norm_stats=None,
                                                                         plot_preprocessing_tce=plot_preprocessing_tce)
    binned_timeseries.update(binned_timeseries_wksecondary)

    # create binned time series for centroid motion
    binned_timeseries_centroid = generate_centroid_binned_views(phase_folded_data, tce, config)
    binned_timeseries.update(binned_timeseries_centroid)

    # create binned time series for momentum dump
    if config['get_momentum_dump']:
        binned_timeseries_momentum_dump = generate_momentum_dump_views(phase_folded_data,
                                                                       tce, config, plot_preprocessing_tce)
        binned_timeseries.update(binned_timeseries_momentum_dump)

    if plot_preprocessing_tce:
        utils_visualization.plot_phasefolded_and_binned(phase_folded_data,
                                                        binned_timeseries,
                                                        tce,
                                                        config,
                                                        config['plot_dir'] /
                                                        f'{tce["uid"]}_{tce["label"]}_'
                                                        f'9_1_phasefoldedbinned_timeseries.png'
                                                        )

        utils_visualization.plot_all_views({ts_name: ts_data for ts_name, ts_data in binned_timeseries.items()
                                            if ts_name in BINNED_TIMESERIES_JOINT_PLOT and
                                            ts_name in binned_timeseries},
                                           tce,
                                           config,
                                           BINNED_TIMESERIES_JOINT_PLOT_GRID,
                                           config['plot_dir'] /
                                           f'{tce["uid"]}_{tce["label"]}_'
                                           f'9_2_binned_timeseries.png',
                                           plot_var=True
                                           )

    # initialize dictionary with the binned time series and statistics that will be stored in the TFRecords example for
    # this TCE
    for binned_ts_name, binned_ts_data in binned_timeseries.items():

        if 'stat' in binned_ts_name:  # for statistics
            if not np.isfinite(binned_ts_data):  # check if statistic value is non-finite
                raise ValueError(f'Statistic {binned_ts_name} is non-finite.')

            example_util.set_float_feature(ex, binned_ts_name, [binned_ts_data])

        else:

            # check if time series have NaN values
            if np.any(~np.isfinite(binned_ts_data[1])):  # at least one point is non-finite (infinite or NaN)
                raise ValueError(f'Binned time series {binned_ts_name} has at least one non-finite data point.')

            if np.any(~np.isfinite(binned_ts_data[2])):  # at least one point is non-finite (infinite or NaN)
                raise ValueError(f'Binned var time series {binned_ts_name} has at least one non-finite data point.')

            # add binned time series to the example
            if 'unfolded' in binned_ts_name:  # unfolded phase binned time series
                example_util.set_tensor_feature(ex, MAP_VIEWS_TO_OLD_NAMES[binned_ts_name], binned_ts_data[1])
                example_util.set_tensor_feature(ex, f'{MAP_VIEWS_TO_OLD_NAMES[binned_ts_name]}_var',
                                                binned_ts_data[2])
            else:  # phase folded time series
                example_util.set_float_feature(ex, MAP_VIEWS_TO_OLD_NAMES[binned_ts_name], binned_ts_data[1])
                example_util.set_float_feature(ex, f'{MAP_VIEWS_TO_OLD_NAMES[binned_ts_name]}_var',
                                               binned_ts_data[2])

            # add number of transits per binned time series
            example_util.set_int64_feature(ex, f'{MAP_VIEWS_TO_OLD_NAMES[binned_ts_name]}_num_transits',
                                           [binned_ts_data[3]])

    # add periodogram data
    for pgram_name, pgram in pgram_data.items():

        if 'power' in pgram_name:
            pgram_arr = pgram
        else:
            pgram_arr = pgram.power.value

        # check if periodogram have NaN values
        if np.any(~np.isfinite(pgram_arr)):  # at least one point is non-finite (infinite or NaN)
            raise ValueError(f'Periodogram {pgram_name} has at least one non-finite data point.')

        example_util.set_float_feature(ex, pgram_name, pgram_arr)

    # set other features from the TCE table - diagnostic statistics, transit fits, stellar parameters...
    for name, value in tce.items():

        if name == 'Public Comment':  # can add special characters that are not serializable in the TFRecords
            continue
        try:
            if isinstance(value, str):
                example_util.set_bytes_feature(ex, name, [value])
            else:
                example_util.set_feature(ex, name, [value])
        except Exception:
            raise TypeError(f'Could not set up this TCE table parameter: {name}.')

    example_stats = create_example_stats(binned_timeseries, odd_even_flag, t_min_transit, t_max_transit, config)

    return ex, example_stats
