"""
Light curve data preprocessing module.
"""

# 3rd party
import os
import numpy as np
import tensorflow as tf
from astropy.stats import mad_std
import lightkurve as lk
import logging

# local
from src_preprocessing.lc_preprocessing import utils_visualization, kepler_io, tess_io
from src_preprocessing.light_curve import util
from src_preprocessing.tf_util import example_util
from src_preprocessing.third_party.kepler_spline import kepler_spline
from src_preprocessing.lc_preprocessing.utils_centroid_preprocessing import (kepler_transform_pxcoordinates_mod13,
                                                                             compute_centroid_distance,
                                                                             correct_centroid_using_transit_depth)
from src_preprocessing.lc_preprocessing.utils_ephemeris import create_binary_time_series, \
    find_first_epoch_after_this_time
from src_preprocessing.lc_preprocessing.utils_gapping import gap_other_tces
from src_preprocessing.lc_preprocessing.utils_imputing import imputing_gaps
from src_preprocessing.lc_preprocessing.utils_odd_even import create_odd_even_views, \
    phase_fold_and_sort_light_curve_odd_even
from src_preprocessing.lc_preprocessing.utils_preprocessing import (count_transits, remove_non_finite_values,
                                                                    check_inputs_generate_example,
                                                                    remove_positive_outliers,
                                                                    lininterp_transits, check_inputs)
from src_preprocessing.lc_preprocessing.utils_preprocessing_io import report_exclusion
from src_preprocessing.lc_preprocessing.detrend_timeseries import detrend_flux_using_spline, \
    detrend_flux_using_sg_filter
from src_preprocessing.lc_preprocessing.phase_fold_and_binning import (global_view, local_view,
                                                                       phase_fold_and_sort_light_curve,
                                                                       phase_split_light_curve,
                                                                       centering_and_normalization,
                                                                       generate_view_momentum_dump)

DEGREETOARCSEC = 3600
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
            raise FileNotFoundError(f'No available lightcurve FITS files in {config["lc_data_dir"]} for '
                                    f'KIC {tce.target_id}')

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
            raise IOError(f'FITS files not read correctly for KIC {tce.target_id}: {fits_files_not_read}')

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
            # if not config['omit_missing']:
            #     raise IOError(f'Failed to find .fits files in {config["lc_data_dir"]} for TESS ID {tce.target_id}')
            # else:
            #     report_exclusion(config, tce, f'No available lightcurve FITS files in {config["lc_data_dir"]} for '
            #                                   f'TIC {tce.target_id}')
            #     return None
            raise FileNotFoundError(f'No available lightcurve FITS files in {config["lc_data_dir"]} for '
                                    f'TIC {tce.target_id}')

        fits_data, fits_files_not_read = tess_io.read_tess_light_curve(file_names,
                                                                       centroid_radec=not config['px_coordinates'],
                                                                       prefer_psfcentr=config['prefer_psfcentr'],
                                                                       light_curve_extension=config[
                                                                           'light_curve_extension'],
                                                                       get_momentum_dump=config['get_momentum_dump'],
                                                                       dq_values_filter=config['dq_values_filter'],
                                                                       )

        # if len(fits_files_not_read) > 0:
        #     err_str = ''
        #     for el in fits_files_not_read:
        #         err_str += f'FITS file was not read correctly: {el}'
        #         report_exclusion(config, tce, err_str)
        #     if len(fits_files_not_read) == len(file_names):
        #         report_exclusion(config, tce, f'No FITS files were read in {config["lc_data_dir"]}.')
        #         return None
        if len(fits_files_not_read) > 0:
            raise IOError(f'FITS files not read correctly for TIC {tce.target_id}: {fits_files_not_read}')

        return fits_data


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


def find_intransit_cadences(tce, table, data, config):
    """ Find in-transit cadences (primary and secondary transits) for TCE of interest and also other detected TCEs for
    the target star found in `table`.

    Args:
        tce: pandas Series, TCE of interest
        table: pandas Dataframe, TCE dataset
        data: dict, lightcurve raw data from the FITS files (e.g., timestamps, flux, centroid motion time series)
        config: dict, preprocessing parameters

    Returns:
        target_intransit_cadences_arr, list of boolean NumPy arrays (n_tces_in_target, n_cadences); each row is for a
        given TCE in the target star; True for in-transit cadences, False otherwise
        idx_tce, index of TCE of interest in `target_intransit_cadences_arr`
    """

    # get all TCEs detected for the target stars' TCE of interest
    tces_in_target = table.loc[table['target_id'] == tce['target_id']].reset_index(inplace=False, drop=True)
    # bookkeeping: get index of TCE of interest in the table of detected TCEs for the target star
    idx_tce = tces_in_target.loc[tces_in_target['uid'] == tce['uid']].index[0]
    n_tces_in_target = len(tces_in_target)
    logger.info(f'[{tce["uid"]}] Found {n_tces_in_target} TCEs in target {tce["target_id"]}.')

    target_intransit_cadences_arr = [np.zeros((n_tces_in_target, len(time_arr)), dtype='bool')
                                     for time_arr in data['all_time']]
    for tce_in_target_i, tce_in_target in tces_in_target.iterrows():  # iterate on target's TCEs

        # get epoch of first transit for each time array
        first_transit_time = [find_first_epoch_after_this_time(tce_in_target['tce_time0bk'],
                                                               tce_in_target['tce_period'],
                                                               time[0])
                              for time in data["all_time"]]

        # create binary time series for each time array in which in-transit points are labeled as True, otherwise False
        tce_intransit_cadences_arr = [create_binary_time_series(time,
                                                                first_transit_time,
                                                                config['duration_gapped_primary'],
                                                                tce_in_target['tce_period'])
                                      for first_transit_time, time in zip(first_transit_time, data["all_time"])]

        if 'tce_maxmesd' in tce_in_target:  # do the same for the detected secondary transits

            first_transit_time_secondary = [find_first_epoch_after_this_time(tce_in_target['tce_time0bk'] +
                                                                             tce_in_target['tce_maxmesd'],
                                                                             tce_in_target['tce_period'],
                                                                             time[0])
                                            for time in data["all_time"]]

            tce_intransit_cadences_arr_secondary = [create_binary_time_series(time,
                                                                              first_transit_time,
                                                                              config['duration_gapped_secondary'],
                                                                              tce_in_target['tce_period'])
                                                    for first_transit_time, time in
                                                    zip(first_transit_time_secondary, data["all_time"])]

            # update TCE array for primary with secondary transits
            tce_intransit_cadences_arr = [np.logical_or(primary_arr, secondary_arr)
                                          for primary_arr, secondary_arr
                                          in zip(tce_intransit_cadences_arr, tce_intransit_cadences_arr_secondary)]

        # update array for target with tce transits
        for arr_i, intransit_cadences_arr in enumerate(tce_intransit_cadences_arr):
            target_intransit_cadences_arr[arr_i][tce_in_target_i] = intransit_cadences_arr

    return target_intransit_cadences_arr, idx_tce


def gap_intransit_cadences_other_tces(tce, table, data, config):
    # FIXME: what if removes a whole quarter? need to adjust all_additional_info to it
    timeseries_gapped = ['all_time', 'all_flux', 'all_centroids']  # timeseries to be gapped
    data['gap_time'] = None

    # get cadence indices to be gapped/imputed
    gapped_idxs = gap_other_tces(data['all_time'],
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
            # report_exclusion(config, tce, f'No data for {key} before creating the views.')
            # return None

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
        config['sampling_rate_h'][config['satellite']],
        config['phase_extend_method'],
        quarter_timestamps=data['quarter_timestamps'] if config['satellite'] == 'kepler' and config[
            'quarter_sampling']
        else None
    )
    phasefolded_timeseries['flux_unfolded'] = (time_split, flux_split, n_phases_split)

    if n_phases_split < config['min_n_phases']:
        raise ValueError(f'Only found {n_phases_split} phase for flux, need at least {config["min_n_phases"]} to '
                         f'create example.')

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
        config['sampling_rate_h'][config['satellite']],
        config['phase_extend_method'],
        quarter_timestamps=data['quarter_timestamps'] if config['satellite'] == 'kepler' and config['quarter_sampling']
        else None
    )
    phasefolded_timeseries['flux_trend_unfolded'] = (time_trend_split, flux_trend_split, n_phases_trend_split)

    if n_phases_trend_split < config['min_n_phases']:
        raise ValueError(f'Only found {n_phases_split} phase for flux trend, need at least {config["min_n_phases"]} to '
                         f'create example.')

    # remove positive outliers from the phase folded flux time series
    if config['pos_outlier_removal']:
        phasefolded_timeseries_posout = {ts_name: (np.array(ts_arr[0]), np.array(ts_arr[1]))
                                         for ts_name, ts_arr in phasefolded_timeseries.items()
                                         if 'flux' in ts_name and
                                         ('trend' not in ts_name and 'unfolded' not in ts_name)}
        for ts_name, (ts_time, ts_values, n_transits_ts) in phasefolded_timeseries.items():
            if 'flux' in ts_name and ('trend' not in ts_name and 'unfolded' not in ts_name):
                ts_time, ts_arr, idxs_out = remove_positive_outliers(
                    ts_time,
                    ts_values,
                    sigma=config['pos_outlier_removal_sigma'],
                    fill=config['pos_outlier_removal_fill']
                )
                phasefolded_timeseries[ts_name] = (ts_time, ts_arr, n_transits_ts)
                phasefolded_timeseries_posout[ts_name] = (ts_time[idxs_out],
                                                          phasefolded_timeseries_posout[ts_name][1][idxs_out])

    else:
        phasefolded_timeseries_posout = None

    if plot_preprocessing_tce:
        # CHANGE NUMBER OF PLOTS AS FUNCTION OF THE TIME SERIES PREPROCESSED
        utils_visualization.plot_all_phasefoldedtimeseries({ts_name: ts_data
                                                            for ts_name, ts_data in phasefolded_timeseries.items()
                                                            if 'unfolded' not in ts_name},
                                                           tce,
                                                           PHASEFOLD_GRID_PLOTS,
                                                           config['plot_dir'] /
                                                           f'{tce["uid"]}_{tce["label"]}_'
                                                           f'4_phasefolded_timeseries_aug{tce["augmentation_idx"]}.png',
                                                           phasefolded_timeseries_posout
                                                           )

    return phasefolded_timeseries


def process_tce(tce, table, config):
    """ Preprocesses the light curve time series data and scalar features for the TCE and returns an Example proto.

    :param tce: pandas Series, TCE parameters and statistics
    :param table: pandas DataFrame, TCE table
    :param config: dict, holds preprocessing parameters

    return: a tensorflow.train.Example proto containing TCE features
    """

    # setting primary and secondary transit gap duration
    if 'tce_maxmesd' in tce:
        config['duration_gapped_primary'] = min(config['gap_padding'] * tce['tce_duration'],
                                                2 * np.abs(tce['tce_maxmesd']) - tce['tce_duration'],
                                                tce['tce_period'])
        # setting secondary gap duration
        config['duration_gapped_secondary'] = config['duration_gapped_primary']
        # config['duration_gapped_secondary'] = (
        #     min(
        #         max(
        #             0,
        #             2 * np.abs(tce['tce_maxmesd']) - config['duration_gapped_primary'] - 2 * config['primary_buffer_time']),
        #         config['gap_padding'] * tce['tce_duration']))
    else:
        config['duration_gapped_primary'] = min(config['gap_padding'] * tce['tce_duration'], tce['tce_period'])
        config['duration_gapped_secondary'] = 0

    # set Savitzky-Golay window
    if config['detrending_method'] == 'savitzky-golay':
        # config['sg_win_len'] = int(config['sg_n_durations_win'] * tce['tce_duration'] * 24 *
        #                            config['sampling_rate_h'][f'{config["satellite"]}'])
        win_dur_h = 1.2 * 24
        config['sg_win_len'] = int(win_dur_h * config['sampling_rate_h'][f'{config["satellite"]}'])
        config['sg_win_len'] = config['sg_win_len'] if config['sg_win_len'] % 2 != 0 else config['sg_win_len'] + 1

    # check if preprocessing pipeline figures are saved for the TCE
    plot_preprocessing_tce = False  # False
    if np.random.random() < config['plot_prob']:
        plot_preprocessing_tce = config['plot_figures']

    # sample TCE ephemeris using uncertainty interval
    tce = sample_ephemerides(tce, config)

    # get cadence, flux and centroid data for the tce
    data = read_light_curve(tce, config)
    if data is None:
        raise IOError(f'Issue when reading data from the FITS file(s) for target {tce["target_id"]}.')
        # report_exclusion(config, tce, 'Issue when reading data from the FITS file(s).')
        # return None

    # update target position in FITS file with that from the TCE table
    if ~np.isnan(tce['ra']) and ~np.isnan(tce['dec']):
        data['target_position'] = [tce['ra'], tce['dec']]
    config['delta_dec'] = np.cos(data['target_position'][1] * np.pi / 180)

    data['errors'] = check_inputs(data)

    # TODO: use this?
    if config['satellite'] == 'kepler':
        add_info = {'quarter': data['quarter'], 'module': data['module']}
    else:
        add_info = {'sectors': data['sectors']}

    # find transits for all detected TCEs in the target
    target_intransit_cadences_arr, idx_tce = find_intransit_cadences(tce, table, data, config)

    # build boolean array with in-transit cadences based on all detected TCEs in the target
    target_intransit_cadences_bool_arr = [np.sum(target_intransit_cadences, axis=0) != 0
                                          for target_intransit_cadences in target_intransit_cadences_arr]
    config['intransit_cadences_target'] = target_intransit_cadences_bool_arr

    # get boolean array with out-of-transit cadences based on all detected TCEs in the target
    target_outoftransit_cadences_bool_arr = [~target_intransit_cadences_bool
                                             for target_intransit_cadences_bool in target_intransit_cadences_bool_arr]
    config['outoftransit_cadences_target'] = target_outoftransit_cadences_bool_arr

    # get boolean array with in-transit cadences for the TCE of interest
    tce_intransit_cadences_arr = [target_intransit_cadences[idx_tce].astype('bool') for target_intransit_cadences
                                  in target_intransit_cadences_arr]
    config['intransit_cadences_tce'] = tce_intransit_cadences_arr

    if plot_preprocessing_tce:
        utils_visualization.plot_intransit_binary_timeseries(data['all_time'],
                                                             data['all_flux'],
                                                             config['intransit_cadences_target'],
                                                             config['intransit_cadences_tce'],
                                                             tce,
                                                             config['plot_dir'],
                                                             f'1_intransit_cadences_aug{tce["augmentation_idx"]}')

    # gap cadences belonging to the transits of other TCEs in the same target star
    if config['gapped']:
        # TODO: fix code for gapping transits
        data = gap_intransit_cadences_other_tces(tce, table, data, config)

    # detrend the flux and centroid time series
    data_detrended = process_light_curve(data, config, tce, plot_preprocessing_tce)

    # phase fold detrended data using detected orbital period
    phase_folded_data = phase_fold_timeseries(data_detrended, config, tce, plot_preprocessing_tce)

    # bin phase folded data and generate TCE example based on the preprocessed data
    example_tce = generate_example_for_tce(phase_folded_data, tce, config, plot_preprocessing_tce)

    return example_tce


def flux_preprocessing(all_time, all_flux, tce, config, plot_preprocessing_tce):
    """ Preprocess the flux time series.

    :param all_time: list of NumPy arrays, timestamps
    :param all_flux: list of NumPy arrays, flux time series
    :param tce: pandas Series, TCE parameters
    :param config: dict, preprocessing parameters
    :param plot_preprocessing_tce: bool, set to True to plot figures related to different preprocessing steps
    :return:
        time: NumPy array, timestamps for preprocessed flux time series
        detrended_flux: NumPy array, detrended flux timeseries
        trend: NumPy array, trend obtained when fitting detrending method to the flux timeseries
    """

    time_arrs, flux_arrs, intransit_cadences_target, intransit_cadences_tce = \
        ([np.array(el) for el in all_time],
         [np.array(el) for el in all_flux],
         [np.array(el) for el in config['intransit_cadences_target']],
         [np.array(el) for el in config['intransit_cadences_tce']])

    # remove non-finite values
    time_arrs, flux_arrs, intransit_cadences_target, intransit_cadences_tce = (
        remove_non_finite_values([
            time_arrs,
            flux_arrs,
            intransit_cadences_target,
            intransit_cadences_tce
        ]))

    # # add gap after and before transit based on transit duration
    # if 'tce_maxmesd' in tce:
    #     duration_gapped = min((1 + 2 * config['gap_padding']) * tce['tce_duration'], np.abs(tce['tce_maxmesd']),
    #                           tce['tce_period'])
    # else:
    #     duration_gapped = min((1 + 2 * config['gap_padding']) * tce['tce_duration'], tce['tce_period'])

    # # get epoch of first transit for each time array
    # first_transit_time_all = [find_first_epoch_after_this_time(tce['tce_time0bk'], tce['tce_period'], time[0])
    #                           for time in time_arrs]
    #
    # # create binary time series for each time array in which in-transit points are labeled as 1's, otherwise as 0's
    # binary_time_all = [create_binary_time_series(time, first_transit_time, config['duration_gapped_primary'],
    #                                              tce['tce_period'])
    #                    for first_transit_time, time in zip(first_transit_time_all, time_arrs)]

    # if plot_preprocessing_tce:
    #     utils_visualization.plot_intransit_binary_timeseries(time_arrs, flux_arrs, binary_time_all, tce,
    #                                                          config['plot_dir'],
    #                                                          f'2_intransit_flux_binary_timeseries_aug{tce["augmentation_idx"]}')

    # # gap secondary transits
    # # get epoch of first transit for each time array
    # first_transit_time_secondary_all = [find_first_epoch_after_this_time(tce['tce_time0bk'] + tce['tce_maxmesd'],
    #                                                                      tce['tce_period'], time[0])
    #                                     for time in time_arrs]
    # # create binary time series for each time array in which in-transit points are labeled as 1's, otherwise as 0's
    # binary_time_secondary_all = [create_binary_time_series(time, first_transit_time_secondary,
    #                                                        config['duration_gapped_secondary'],
    #                                                        tce['tce_period'])
    #                              for first_transit_time_secondary, time in
    #                              zip(first_transit_time_secondary_all, time_arrs)]
    # # set secondary in-transit cadences to np.nan
    # flux_arrs = [np.where(binary_time_secondary, np.nan, flux_arr)
    #              for flux_arr, binary_time_secondary in zip(flux_arrs, binary_time_secondary_all)]
    #
    # # remove non-finite values
    # time_arrs, flux_arrs = remove_non_finite_values([time_arrs, flux_arrs])

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
        # # mask in-transit cadences
        # mask_in_transit = lc.create_transit_mask(tce['tce_period'], tce['tce_time0bk'],
        #                                          config['duration_gapped_primary'])

        time, detrended_flux, trend = detrend_flux_using_sg_filter(lc,
                                                                   intransit_cadences_target,
                                                                   config['sg_win_len'],
                                                                   config['sg_sigma'],
                                                                   config['sg_max_poly_order'],
                                                                   config['sg_penalty_weight'],
                                                                   config['sg_break_tolerance']
                                                                   )
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
                                              f'2_detrendedflux_aug{tce["augmentation_idx"]}',
                                              flux_interp=flux_lininterp)

        # utils_visualization.plot_residual(time,
        #                                   res_flux,
        #                                   tce,
        #                                   config['plot_dir'],
        #                                   f'3_residual_flux_aug{tce["augmentation_idx"]}',
        #                                   )

    return time, detrended_flux, trend


def weak_secondary_flux_preprocessing(all_time, all_flux_noprimary, tce, config, plot_preprocessing_tce):
    """ Preprocess the weak secondary flux timeseries.

    :param all_time: list of NumPy arrays, timestamps
    :param all_flux_noprimary: list of NumPy arrays, weak secondary flux time series
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
    # # gap primary transits
    # # get epoch of first transit for each time array
    # first_transit_time_primary_all = [find_first_epoch_after_this_time(tce['tce_time0bk'], tce['tce_period'], time[0])
    #                                   for time in time_arrs]
    # # create binary time series for each time array in which in-transit points are labeled as 1's, otherwise as 0's
    # binary_time_primary_all = [create_binary_time_series(time, first_transit_time_secondary,
    #                                                      config['duration_gapped_secondary'],
    #                                                      tce['tce_period'])
    #                            for first_transit_time_secondary, time in zip(first_transit_time_primary_all, time_arrs)]
    # # set primary in-transit cadences to np.nan
    # flux_arrs = [np.where(binary_time_primary, np.nan, flux_arr)
    #              for flux_arr, binary_time_primary in zip(flux_arrs, binary_time_primary_all)]

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

    return time, detrended_flux


def centroid_preprocessing(all_time, all_centroids, target_position, add_info, tce, config, plot_preprocessing_tce):
    """ Preprocess the centroid timeseries.

    :param all_time: list of NumPy arrays, timestamps
    :param all_centroids: dictionary for the two centroid coordinates coded as 'x' and 'y'. Each key maps to a list of
    NumPy arrays for the respective centroid coordinate time series
    # :param avg_centroid_oot: dictionary for the two centroid coordinates coded as 'x' and 'y'. Each key maps to the
    # estimate of the average out-of-transit centroid
    :param target_position: list, target star position in 'x' and 'y'
    :param add_info: dictionary, additional information such as quarters and modules
    :param tce: Pandas Series, TCE parameters
    :param config: dict, preprocessing parameters
    :param plot_preprocessing_tce: bool, set to True to plot figures related to different preprocessing steps
    :return:
        time: NumPy array, timestamps for preprocessed centroid time series
        centroid_dist: NumPy array, preprocessed centroid time series which is an estimate of the distance of the
        transit to the target
    """

    time_arrs, centroid_dict, intransit_cadences_target, intransit_cadences_tce = (
        [np.array(el) for el in all_time],
        {coord: [np.array(el) for el in centroid_arrs] for coord, centroid_arrs in all_centroids.items()},
        [np.array(el) for el in config['intransit_cadences_target']],
        [np.array(el) for el in config['intransit_cadences_tce']]
    )

    if np.isnan(np.concatenate(centroid_dict['x'])).all():  # when there's no centroid data
        time_centroid = np.concatenate(time_arrs)
        centroid_dist = np.zeros(len(time_centroid), dtype='float')

        report_exclusion(f'No available flux-weighted centroid data for target {tce.target_id}. '
                         f'Setting transit offset distance from target to zero.',
                         config['exclusion_logs_dir'] / f'exclusions_{tce["uid"]}.txt')

        return time_centroid, centroid_dist

    time_arrs, centroid_dict['x'], centroid_dict['y'], intransit_cadences_target, intransit_cadences_tce = (
        remove_non_finite_values([
            time_arrs,
            centroid_dict['x'],
            centroid_dict['y'],
            intransit_cadences_target,
            intransit_cadences_tce,
        ]
        ))

    # pixel coordinate transformation for targets on module 13 for Kepler
    if config['px_coordinates'] and config['satellite'] == 'kepler':
        if add_info['module'][0] == 13:
            centroid_dict = kepler_transform_pxcoordinates_mod13(centroid_dict, add_info)

    # # gap primary in-transit cadences
    # # get epoch of first transit for each time array
    # first_transit_time_all = [find_first_epoch_after_this_time(tce['tce_time0bk'], tce['tce_period'], time[0])
    #                           for time in time_arrs]
    # # create in-transit binary time series for each time array
    # binary_time_all = [create_binary_time_series(time, first_transit_time, config['duration_gapped_primary'],
    #                                              tce['tce_period'])
    #                    for first_transit_time, time in zip(first_transit_time_all, time_arrs)]
    #
    # # gap secondary in-transit cadences
    # # get epoch of first transit for each time array
    # first_transit_time_all = [find_first_epoch_after_this_time(tce['tce_time0bk'] + tce['tce_maxmesd'],
    #                                                            tce['tce_period'], time[0])
    #                           for time in time_arrs]
    # # create in-transit binary time series for each time array
    # binary_time_secondary_all = [create_binary_time_series(time, first_transit_time,
    #                                                        config['duration_gapped_secondary'], tce['tce_period'])
    #                              for first_transit_time, time in zip(first_transit_time_all, time_arrs)]
    #
    # # merge both in-transit binary timeseries for primary and secondary in-transit cadences
    # binary_time_all = [np.logical_or(binary_time_arr_primary, binary_time_arr_secondary)
    #                    for binary_time_arr_primary, binary_time_arr_secondary
    #                    in zip(binary_time_all, binary_time_secondary_all)]

    # get out-of-transit indices for the centroid time series
    centroid_oot = {coord: [centroids[~intransit_cadences_target_arr] for intransit_cadences_target_arr, centroids
                            in zip(intransit_cadences_target, centroid_dict[coord])] for coord in centroid_dict}
    # estimate average out-of-transit centroid as the median
    avg_centroid_oot = {coord: np.nanmedian(np.concatenate(centroid_oot[coord])) for coord in centroid_oot}
    # if there is no valid out-of-transit cadences, use median of the whole centroid array
    avg_centroid_oot = {coord: np.nanmedian(np.concatenate(centroid_dict[coord])) if np.isnan(avg_centroid_oot_coord)
    else avg_centroid_oot_coord for coord, avg_centroid_oot_coord in avg_centroid_oot.items()}

    # if plot_preprocessing_tce:
    #     utils_visualization.plot_intransit_binary_timeseries(time_arrs,
    #                                                          centroid_dict,
    #                                                          binary_time_all,
    #                                                          tce,
    #                                                          config['plot_dir'],
    #                                                          f'2_intransit_centroids_binary_timeseries_aug{tce["augmentation_idx"]}',
    #                                                          centroid=True)

    time_arrs, centroid_dict['x'], centroid_dict['y'] = remove_non_finite_values([time_arrs,
                                                                                  centroid_dict['x'],
                                                                                  centroid_dict['y']])

    detrended_centroid_dict = {centroid_coord: {'detrended': None, 'trend': None}
                               for centroid_coord in centroid_dict.keys()}
    for centroid_coord, centroid_coord_data in centroid_dict.items():

        # detrend centroid
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

            _, detrended_centroid, trend = detrend_flux_using_sg_filter(lc,
                                                                        intransit_cadences_target,
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
                                           f'3_1_detrendedcentroids_aug{tce["augmentation_idx"]}',
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
                                                     f'3_2_correctedcentroids_aug{tce["augmentation_idx"]}',
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
                                                f'3_3_distcentr_aug{tce["augmentation_idx"]}')

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
    """ Detrends different time series (e.g., detrending) such as flux and centroid motion.

    Args:
      data: dictionary containing raw lc data
      config: dict, holds preprocessing parameters
      tce: Pandas Series, row of the TCE table for the TCE that is being processed; it contains data such as the
      ephemerides
      plot_preprocessing_tce: bool, if True plots figures for several steps in the preprocessing pipeline

    Returns:
      detrended_data: dict, containing detrended data for different time series
    """

    # FIXME: what is this for
    if config['satellite'] == 'kepler':
        add_info_centr = {'quarter': data['quarter'], 'module': data['module']}
    else:
        add_info_centr = None

    # detrend flux time series
    # try:
    flux_time, flux, flux_trend = flux_preprocessing(data['all_time'],
                                                     data['all_flux'],
                                                     tce,
                                                     config,
                                                     plot_preprocessing_tce)
    # except Exception as e:
    #     report_exclusion(config, tce, 'Error when preprocessing flux.', stderr=e)
    #     time, flux, flux_trend = None, None, None

    # detrend centroid time series
    # try:
    centroid_dist_time, centroid_dist = centroid_preprocessing(data['all_time'],
                                                               data['all_centroids'],
                                                               data['target_position'],
                                                               add_info_centr,
                                                               tce,
                                                               config,
                                                               plot_preprocessing_tce)
    # except Exception as e:
    #     report_exclusion(config, tce, 'Error when preprocessing centroids.', stderr=e)
    #     centroid_dist_time, centroid_dist = None, None

    if config['get_momentum_dump']:
        momentum_dump_time, momentum_dump = remove_non_finite_values([data['time_momentum_dump'],
                                                                      data['momentum_dump']])
        momentum_dump_time, momentum_dump = np.concatenate(momentum_dump_time), np.concatenate(momentum_dump)

        if plot_preprocessing_tce:
            utils_visualization.plot_momentum_dump_timeseries(momentum_dump_time,
                                                              momentum_dump,
                                                              tce,
                                                              config['plot_dir'] /
                                                              f'{tce["uid"]}_{tce["label"]}_'
                                                              f'2_momentum_dump_timeseries.png')

    # dictionary with detrended time series
    detrended_data = {
        'flux_time': flux_time,
        'flux': flux,
        'flux_trend': flux_trend,
        'centroid_dist_time': centroid_dist_time,
        'centroid_dist': centroid_dist,
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
        num_transits: dict, number of transits per timeseries
        # inds_bin_nan: dict, indices of bins with missing values per views
        t_min_transit: float, left edge in-transit phase time
        t_max_transit: float, right edge in-transit phase time
        config: dict, preprocessing parameters for the run

    Returns: dict, with summary statistics for the example

    """

    # initialize dictionary with data for the preprocessing table
    example_stats = {}

    # number of transits
    example_stats.update({f'num_transits_{ts_name}': ts_data[3] for ts_name, ts_data in binned_timeseries.items()})

    # centroid
    glob_centr_oot_bins = (binned_timeseries['centroid_offset_distance_to_target_global'][0] < t_min_transit) | \
                          (binned_timeseries['centroid_offset_distance_to_target_global'][0] > t_max_transit)
    loc_centr_it_bins = (binned_timeseries['centroid_offset_distance_to_target_local'][0] >= t_min_transit) & \
                        (binned_timeseries['centroid_offset_distance_to_target_local'][0] <= t_max_transit)
    example_stats['avg_oot_centroid_offset'] = np.median(binned_timeseries['centroid_offset_distance_to_target_global'][1][glob_centr_oot_bins])
    example_stats['std_oot_centroid_offset'] = mad_std(binned_timeseries['centroid_offset_distance_to_target_global'][1][glob_centr_oot_bins])
    example_stats['peak_centroid_offset'] = np.max(binned_timeseries['centroid_offset_distance_to_target_local'][1][loc_centr_it_bins]) - \
                                            example_stats['avg_oot_centroid_offset']

    example_stats['mid_local_flux_shift'] = np.argmin(binned_timeseries['centroid_offset_distance_to_target_local'][1]) - int(config['num_bins_loc'] / 2)
    example_stats['mid_global_flux_shift'] = np.argmin(binned_timeseries['centroid_offset_distance_to_target_global'][1]) - int(config['num_bins_glob'] / 2)

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
    example_stats['transit_depth_odd_hat'] = (1 - binned_timeseries['flux_odd_local'][1][loc_flux_odd_it_bins][ind]) * 1e6
    example_stats['transit_depth_odd_hat_se'] = binned_timeseries['flux_odd_local'][2][loc_flux_odd_it_bins][ind] * 1e6

    loc_flux_even_it_bins = (binned_timeseries['flux_even_local'][0] >= t_min_transit) & \
                            (binned_timeseries['flux_even_local'][0] <= t_max_transit)
    ind = np.argmin(binned_timeseries['flux_even_local'][1][loc_flux_even_it_bins])
    example_stats['transit_depth_even_hat'] = (1 - binned_timeseries['flux_even_local'][1][loc_flux_even_it_bins][ind]) * 1e6
    example_stats['transit_depth_even_hat_se'] = binned_timeseries['flux_even_local'][2][loc_flux_even_it_bins][ind] * 1e6

    # weak secondary flux
    loc_flux_wks_it_bins = (binned_timeseries['flux_weak_secondary_local'][0] >= t_min_transit) & \
                           (binned_timeseries['flux_weak_secondary_local'][0] <= t_max_transit)
    ind = np.argmin(binned_timeseries['flux_weak_secondary_local'][1][loc_flux_wks_it_bins])
    example_stats['wks_transit_depth_hat'] = (1 - binned_timeseries['flux_weak_secondary_local'][1][loc_flux_wks_it_bins][
        ind]) * 1e6
    example_stats['wks_transit_depth_hat_se'] = binned_timeseries['flux_weak_secondary_local'][2][loc_flux_wks_it_bins][ind] * 1e6

    # # add number of missing bins for each view (excluding odd and even local views)
    # for key in inds_bin_nan:
    #     example_stats[f'{key}_num_bins_it_nan'] = inds_bin_nan[key]['it'].sum()
    #     example_stats[f'{key}_num_bins_oot_nan'] = inds_bin_nan[key]['oot'].sum()

    return example_stats


def generate_odd_even_binned_views(data, tce, config, norm_stats, plot_preprocessing_tce):

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

    flux_odd_local_norm = (odd_data['local_flux_view'] - norm_stats['mu']) / norm_stats['sigma']
    flux_odd_local_var_norm = odd_data['local_flux_view_se'] / norm_stats['sigma']

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
                                          f'{tce["uid"]}_{tce["label"]}_8_oddeven_transitdepth_phasefoldedbinned_'
                                          f'timeseries_aug{tce["augmentation_idx"]}.png'
                                          )

    return binned_timeseries, odd_even_flag


def generate_flux_binned_views(data, tce, config, plot_preprocessing_tce):

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
                                           f'{tce["uid"]}_{tce["label"]}_8_riverplot_aug{tce["augmentation_idx"]}.png')

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
    bin_counts[bin_counts == 0] = max(1, np.median(bin_counts))
    loc_flux_view_var /= np.sqrt(bin_counts)

    # create unfolded local flux views
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

    return binned_timeseries, norm_flux_global_stats, norm_flux_local_stats


def generate_flux_trend_binned_views(data, tce, config, plot_preprocessing_tce):

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
                                           f'8_riverplot_trend_aug{tce["augmentation_idx"]}')

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
    bin_counts[bin_counts == 0] = max(1, np.median(bin_counts))
    loc_flux_view_var /= np.sqrt(bin_counts)

    # create unfolded local flux views
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

        bin_counts[bin_counts == 0] = max(1, np.median(bin_counts))
        unfolded_loc_flux_view_var_phase /= np.sqrt(bin_counts)

        unfolded_loc_flux_view.append(unfolded_loc_flux_view_phase)
        unfolded_loc_flux_view_var.append(unfolded_loc_flux_view_var_phase)

    unfolded_loc_flux_view = np.array(unfolded_loc_flux_view)
    unfolded_loc_flux_view_var = np.array(unfolded_loc_flux_view_var)

    # normalize data
    norm_flux_median = np.median(glob_flux_view)
    norm_flux_factor = np.abs(np.min(glob_flux_view - norm_flux_median)) + NORM_SIGMA_EPS

    binned_timeseries = {
        'flux_trend_global': (glob_binned_time, glob_flux_view, glob_flux_view_var, data['flux'][2]),
        'flux_trend_local': (loc_binned_time, loc_flux_view, loc_flux_view_var, data['flux'][2]),
        'flux_trend_global_unfolded': (None, unfolded_glob_flux_view, unfolded_glob_flux_view_var,
                                       data['flux_unfolded'][2]),
        'flux_trend_local_unfolded': (None, unfolded_loc_flux_view, unfolded_loc_flux_view_var,
                                      data['flux_unfolded'][2])
    }

    binned_timeseries_norm = {f'{ts_name}_norm': (binned_time,
                                                  (binned_values - norm_flux_median) / norm_flux_factor,
                                                  binned_values_var / norm_flux_factor,
                                                  n_transits)
                              for ts_name, (binned_time, binned_values, binned_values_var, n_transits) in
                              binned_timeseries.items()}

    binned_timeseries.update(binned_timeseries_norm)

    return binned_timeseries


def generate_weak_secondary_binned_views(data, tce, config, norm_stats=None):

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

    # set local wks view to Gaussian noise using statistics from global weak secondary view when there are no data
    # to create the local view
    if np.all(np.isnan(loc_weak_secondary_view)):
        report_exclusion(
                         f'No data available for local weak secondary view. Setting it to Gaussian noise using '
                         f'statistics from the global wks view.',
                         config['exclusion_logs_dir'] / f'exclusions-{tce["uid"]}.txt')
        mu, sigma = np.nanmedian(data['flux_weak_secondary'][1]), mad_std(data['flux_weak_secondary'][1],
                                                                          ignore_nan=True)
        rng = np.random.default_rng()
        loc_weak_secondary_view = rng.normal(mu, sigma, config['num_bins_loc'])
        loc_weak_secondary_view_var = sigma * np.ones(config['num_bins_loc'])

        _, _, _, _, bin_counts = \
            global_view(data['flux_weak_secondary'][0],
                        data['flux_weak_secondary'][1],
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

    # normalize binned time series
    if norm_stats is None:  # normalize by self absolute minimum
        norm_stats = {'mu': np.median(loc_weak_secondary_view)}
        norm_stats['sigma'] = np.abs(np.min(loc_weak_secondary_view - norm_stats['mu']))

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
    }

    return binned_timeseries


def generate_centroid_binned_views(data, tce, config):

    # get centroid views
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

    binned_timeseries = {
        'centroid_offset_distance_to_target_global': (glob_binned_time, glob_centr_view, glob_centr_view_var,
                                                      data['centroid_offset_distance_to_target'][2]),
        'centroid_offset_distance_to_target_local': (loc_binned_time, loc_centr_view, loc_centr_view_var,
                                                     data['centroid_offset_distance_to_target'][2]),
    }

    return binned_timeseries


def generate_momentum_dump_views(data, tce, config, plot_preprocessing_tce):

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
                                               f'{tce["uid"]}_{tce["label"]}_9_momentum_dump_phase_and_binned.png')

    binned_timeseries = {
        'momentum_dump_local': (binned_time, loc_mom_dump_view, loc_mom_dump_view_var, -1)
    }

    return binned_timeseries


def generate_example_for_tce(data, tce, config, plot_preprocessing_tce=False):
    """ Generates a tf.train.Example representing an input TCE.

    Args:
      data: dictionary, containing preprocessed (detrended and phase-folded) time series used to generate the views
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

    binned_timeseries_flux, binned_flux_global_norm_stats, binned_flux_local_norm_stats = (
        generate_flux_binned_views(data, tce, config, plot_preprocessing_tce))
    binned_timeseries.update(binned_timeseries_flux)

    binned_timeseries_flux_trend = generate_flux_trend_binned_views(data, tce, config, plot_preprocessing_tce)
    binned_timeseries.update(binned_timeseries_flux_trend)

    binned_timeseries_odd_even_flux, odd_even_flag = generate_odd_even_binned_views(data,
                                                                                    tce,
                                                                                    config,
                                                                                    binned_flux_local_norm_stats,
                                                                                    plot_preprocessing_tce)
    binned_timeseries.update(binned_timeseries_odd_even_flux)

    binned_timeseries_wksecondary = generate_weak_secondary_binned_views(data,
                                                                         tce,
                                                                         config,
                                                                         norm_stats=binned_flux_local_norm_stats)
    binned_timeseries.update(binned_timeseries_wksecondary)

    binned_timeseries_centroid = generate_centroid_binned_views(data, tce, config)
    binned_timeseries.update(binned_timeseries_centroid)

    # if plot_preprocessing_tce:
    #     utils_visualization.plot_wks(glob_view, glob_view_weak_secondary, tce, config,
    #                                  os.path.join(config['output_dir'], 'plots'), '8_wks_test')

    # flux_views = {'global_flux_view': glob_flux_view,
    #               'local_flux_view': loc_flux_view,
    #               'local_flux_odd_view': odd_data['local_flux_view'],
    #               'local_flux_even_view': even_data['local_flux_view'],
    #               'unfolded_global_flux_view': unfolded_glob_flux_view,
    #               'unfolded_local_flux_view': unfolded_loc_flux_view,
    #               }
    #
    # flux_views_var = {'global_flux_view': glob_flux_view_var,
    #                   'local_flux_view': loc_flux_view_var,
    #                   'local_flux_odd_view': odd_data['local_flux_view_se'],
    #                   'local_flux_even_view': even_data['local_flux_view_se'],
    #                   'unfolded_global_flux_view': unfolded_glob_flux_view_var,
    #                   'unfolded_local_flux_view': unfolded_loc_flux_view_var,
    #                   }

    # get normalization statistics for global and local flux views
    # flux_views_stats = {'median': {'global': np.median(binned_timeseries['flux_global'][1]),
    #                                'local': np.median(binned_timeseries['flux_global'][1])}}
    # flux_views_stats['min'] = {'global': np.abs(np.min(binned_timeseries['flux_global'][1] -
    #                                                    flux_views_stats['median']['global'])),
    #                            'local': np.abs(np.min(binned_timeseries['flux_local'][1] -
    #                                                   flux_views_stats['median']['local']))}
    #
    # # center by the flux view median and normalize by the flux view absolute minimum
    # binned_timeseries_norm = {timeseries_name: 4 * (None,) for timeseries_name in binned_timeseries}
    # for timeseries_name, (binned_time, binned_values, binned_values_var, n_transits) in binned_timeseries.items():
    #     if 'unfolded' in timeseries_name:
    #         binned_values_norm = np.nan * np.zeros_like(binned_values)
    #         for binned_phase_i, binned_phase in enumerate(binned_values):
    #             binned_values_norm[binned_phase_i] = (
    #                 centering_and_normalization(binned_phase,
    #                                             flux_views_stats['median'][
    #                                                 ('local', 'global')['global' in timeseries_name]],
    #                                             flux_views_stats['min'][
    #                                                 ('local', 'global')['global' in timeseries_name]],
    #                                             report={'config': config, 'tce': tce, 'view': timeseries_name}
    #                                             ))
    #         binned_timeseries_norm[timeseries_name][1] = binned_values_norm
    # views_aux = {}
    # for flux_view in flux_views:
    #
    #     if 'unfolded' in flux_view:
    #         views_aux[f'{flux_view}_fluxnorm'] = []
    #         for phase in flux_views[flux_view]:
    #             new_phase = \
    #                 centering_and_normalization(phase,
    #                                             flux_views_stats['median'][
    #                                                 ('local', 'global')['global' in flux_view]],
    #                                             flux_views_stats['min'][('local', 'global')['global' in flux_view]],
    #                                             report={'config': config, 'tce': tce, 'view': flux_view}
    #                                             )
    #             views_aux[f'{flux_view}_fluxnorm'].append(new_phase)
    #         views_aux[f'{flux_view}_fluxnorm'] = np.array(views_aux[f'{flux_view}_fluxnorm'])
    #
    #     views_aux[f'{flux_view}_fluxnorm'] = \
    #         centering_and_normalization(flux_views[flux_view],
    #                                     flux_views_stats['median'][('local', 'global')['global' in flux_view]],
    #                                     flux_views_stats['min'][('local', 'global')['global' in flux_view]],
    #                                     report={'config': config, 'tce': tce, 'view': flux_view}
    #                                     )
    # flux_views.update(views_aux)
    # views_aux = {}
    # for flux_view in flux_views_var:
    #     if 'unfolded' in flux_view:
    #         views_aux[f'{flux_view}_fluxnorm'] = []
    #         for phase in flux_views_var[flux_view]:
    #             new_phase = \
    #                 centering_and_normalization(phase,
    #                                             0,
    #                                             flux_views_stats['min'][('local', 'global')['global' in flux_view]],
    #                                             report={'config': config, 'tce': tce, 'view': flux_view}
    #                                             )
    #             views_aux[f'{flux_view}_fluxnorm'].append(new_phase)
    #         views_aux[f'{flux_view}_fluxnorm'] = np.array(views_aux[f'{flux_view}_fluxnorm'])
    #
    #     views_aux[f'{flux_view}_fluxnorm'] = \
    #         centering_and_normalization(flux_views_var[flux_view],
    #                                     0,
    #                                     flux_views_stats['min'][('local', 'global')['global' in flux_view]],
    #                                     report={'config': config, 'tce': tce, 'view': flux_view}
    #                                     )
    # flux_views_var.update(views_aux)

    # normalize odd-even uncertainty by transit depth

    # # center by the weak secondary flux view median and normalize by the weak secondary flux view absolute minimum
    # weak_secondary_flux_views = {
    #     'local_weak_secondary_view': loc_weak_secondary_view
    # }
    # weak_secondary_flux_views_var = {
    #     'local_weak_secondary_view': loc_weak_secondary_view_var
    # }
    # views_aux = {}
    # for flux_view in weak_secondary_flux_views:
    #     weak_secondary_view_median = np.median(weak_secondary_flux_views[flux_view])
    #     norm_wks_factor_selfnorm = np.abs(np.min(weak_secondary_flux_views[flux_view] - weak_secondary_view_median))
    #
    #     # normalize by self absolute minimum
    #     views_aux[f'{flux_view}_selfnorm'] = \
    #         centering_and_normalization(weak_secondary_flux_views[flux_view],
    #                                     weak_secondary_view_median,
    #                                     norm_wks_factor_selfnorm,
    #                                     report={'config': config, 'tce': tce, 'view': flux_view}
    #                                     )
    #     # normalize by flux absolute minimum
    #     norm_wks_factor_fluxnorm = flux_views_stats['min'][('local', 'global')['global' in flux_view]]
    #     views_aux[f'{flux_view}_fluxnorm'] = \
    #         centering_and_normalization(weak_secondary_flux_views[flux_view],
    #                                     flux_views_stats['median'][('local', 'global')['global' in flux_view]],
    #                                     norm_wks_factor_fluxnorm,
    #                                     report={'config': config, 'tce': tce, 'view': flux_view}
    #                                     )
    #     # normalize by max between flux absolute minimum and wks flux absolute minimum
    #     norm_wks_factor_fluxselfnorm = max(norm_wks_factor_fluxnorm, norm_wks_factor_selfnorm),
    #     views_aux[f'{flux_view}_max_flux-wks_norm'] = \
    #         centering_and_normalization(weak_secondary_flux_views[flux_view],
    #                                     flux_views_stats['median'][('local', 'global')['global' in flux_view]],
    #                                     norm_wks_factor_fluxselfnorm,
    #                                     report={'config': config, 'tce': tce, 'view': flux_view}
    #                                     )
    # weak_secondary_flux_views.update(views_aux)
    #
    # views_aux = {}
    # for flux_view in weak_secondary_flux_views_var:
    #     weak_secondary_view_median = np.median(weak_secondary_flux_views[flux_view])
    #     norm_wks_factor_selfnorm = np.abs(np.min(weak_secondary_flux_views[flux_view] -
    #                                              weak_secondary_view_median))
    #
    #     # normalize by self absolute minimum
    #     views_aux[f'{flux_view}_selfnorm'] = \
    #         centering_and_normalization(weak_secondary_flux_views_var[flux_view],
    #                                     0,
    #                                     norm_wks_factor_selfnorm,
    #                                     report={'config': config, 'tce': tce, 'view': flux_view}
    #                                     )
    #     # normalize by flux absolute minimum
    #     norm_wks_factor_fluxnorm = flux_views_stats['min'][('local', 'global')['global' in flux_view]]
    #     views_aux[f'{flux_view}_fluxnorm'] = \
    #         centering_and_normalization(weak_secondary_flux_views_var[flux_view],
    #                                     0,
    #                                     norm_wks_factor_fluxnorm,
    #                                     report={'config': config, 'tce': tce, 'view': flux_view}
    #                                     )
    #     # normalize by max between flux absolute minimum and wks flux absolute minimum
    #     norm_wks_factor_fluxselfnorm = max(norm_wks_factor_selfnorm, norm_wks_factor_fluxnorm)
    #     views_aux[f'{flux_view}_max_flux-wks_norm'] = \
    #         centering_and_normalization(weak_secondary_flux_views_var[flux_view],
    #                                     0,
    #                                     norm_wks_factor_fluxselfnorm,
    #                                     report={'config': config, 'tce': tce, 'view': flux_view}
    #                                     )
    # weak_secondary_flux_views_var.update(views_aux)
    #
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

    # create local view for momentum dump
    if config['get_momentum_dump']:
        binned_timeseries_momentum_dump = generate_momentum_dump_views(data, tce, config, plot_preprocessing_tce)
        binned_timeseries.update(binned_timeseries_momentum_dump)

    if plot_preprocessing_tce:
        utils_visualization.plot_phasefolded_and_binned(data,
                                                        binned_timeseries,
                                                        tce,
                                                        config,
                                                        config['plot_dir'] /
                                                        f'{tce["uid"]}_{tce["label"]}_'
                                                        f'8_phasefoldedbinned_timeseries_'
                                                        f'aug{tce["augmentation_idx"]}.png'
                                                        )

        utils_visualization.plot_all_views({ts_name: ts_data for ts_name, ts_data in binned_timeseries.items()
                                            if ts_name in BINNED_TIMESERIES_JOINT_PLOT and
                                            ts_name in binned_timeseries},
                                           tce,
                                           config,
                                           BINNED_TIMESERIES_JOINT_PLOT_GRID,
                                           config['plot_dir'] /
                                           f'{tce["uid"]}_{tce["label"]}_'
                                           f'8_binned_timeseries_aug{tce["augmentation_idx"]}.png',
                                           plot_var=True
                                           )

    # initialize dictionary with the views that will be stored in the TFRecords example for this TCE
    for binned_ts_name, binned_ts_data in binned_timeseries.items():

        # check if time series have NaN values
        if np.any(~np.isfinite(binned_ts_data[1])):  # at least one point is non-finite (infinite or NaN)
            raise ValueError(f'Binned time series {binned_ts_name} has at least one non-finite data point.')
            # report_exclusion(config, tce, f'View has at least one non-finite data point in binned time series '
            #                               f'{binned_ts_name}.')
            # return None

        if np.any(~np.isfinite(binned_ts_data[2])):  # at least one point is non-finite (infinite or NaN)
            raise ValueError(f'Binned var time series {binned_ts_name} has at least one non-finite data point.')
            # report_exclusion(config, tce,
            #                  f'View has at least one non-finite data point in binned time series var '
            #                  f'{binned_ts_name}.')
            # return None

        # add binned time series to the example
        if 'unfolded' in binned_ts_name:
            example_util.set_tensor_feature(ex, MAP_VIEWS_TO_OLD_NAMES[binned_ts_name], binned_ts_data[1])
            example_util.set_tensor_feature(ex, f'{MAP_VIEWS_TO_OLD_NAMES[binned_ts_name]}_var',
                                            binned_ts_data[2])
        else:
            example_util.set_float_feature(ex, MAP_VIEWS_TO_OLD_NAMES[binned_ts_name], binned_ts_data[1])
            example_util.set_float_feature(ex, f'{MAP_VIEWS_TO_OLD_NAMES[binned_ts_name]}_var',
                                           binned_ts_data[2])

        # add number of transits per view
        example_util.set_int64_feature(ex, f'{MAP_VIEWS_TO_OLD_NAMES[binned_ts_name]}_num_transits',
                                       [binned_ts_data[3]])

        # views[MAP_VIEWS_TO_OLD_NAMES[binned_ts_name]] = binned_ts_data[1]
        # views_var[f'{MAP_VIEWS_TO_OLD_NAMES[binned_ts_name]}_var'] = binned_ts_data[2]
        # view_n_transits[] = binned_ts_data[3]

    # views = {}
    # views.update(flux_views)
    # views.update(weak_secondary_flux_views)
    # views.update(centr_views)
    # if config['get_momentum_dump']:
    #     views['local_momentum_dump_view'] = loc_mom_dump_view
    #
    # views_var = {}
    # views_var.update(flux_views_var)
    # views_var.update(weak_secondary_flux_views_var)
    # views_var.update(centr_views_var)
    # if config['get_momentum_dump']:
    #     views_var['local_momentum_dump_view'] = loc_mom_dump_view_var

    # except Exception as e:
    #     report_exclusion(config, tce, 'Error when creating views', stderr=e)
    #     return None

    # check if time series have NaN values
    # for view in views:
    #     if np.any(~np.isfinite(views[view])):  # at least one point is non-finite (infinite or NaN)
    #         report_exclusion(config, tce, f'View has at least one non-finite data point in view {view}.')
    #         return None
    #
    # for view in views_var:
    #     if np.any(~np.isfinite(views_var[view])):  # at least one point is non-finite (infinite or NaN)
    #         report_exclusion(config, tce, f'View has at least one non-finite data point in view {view}_var.')
    #         return None
    #
    # # set time series features in the example to be written to a TFRecord
    # for view in views:
    #     if 'unfolded' in view:
    #         example_util.set_tensor_feature(ex, view, views[view])
    #     else:
    #         example_util.set_float_feature(ex, view, views[view])
    # for view in views_var:
    #     if 'unfolded' in view:
    #         example_util.set_tensor_feature(ex, f'{view}_var', views[view])
    #     else:
    #         example_util.set_float_feature(ex, f'{view}_var', views[view])

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
            report_exclusion(f'Could not set up this TCE table parameter: {name}.',
                             config['exclusion_logs_dir'] / f'exclusions-{tce["uid"]}.txt',
                             e
                            )

    # # add number of transits per view
    # for view in num_transits:
    #     example_util.set_int64_feature(ex, f'{view}_num_transits', [num_transits[view]])

    # # add odd and even scalar features
    # for field in ['se_oot', 'std_oot_bin']:
    #     example_util.set_float_feature(ex, f'odd_{field}', [odd_data[field]])
    #     example_util.set_float_feature(ex, f'even_{field}', [even_data[field]])

    #     # provide adjusted odd-even SE due to TESS having higher sampling rate
    #     if config['satellite'] == 'tess' and field == 'se_oot':
    #         example_util.set_float_feature(ex, f'odd_{field}_adjsampl',
    #                                        [odd_data[field] * config['tess_to_kepler_sampling_ratio']])
    #         example_util.set_float_feature(ex, f'even_{field}_adjsampl',
    #                                        [even_data[field] * config['tess_to_kepler_sampling_ratio']])

    example_stats = create_example_stats(binned_timeseries, odd_even_flag, t_min_transit, t_max_transit, config)

    try:
        a = []
        a[0]
    except Exception as e:
        raise ValueError('aaaa', e.__traceback__).with_traceback(e.__traceback__) from None
        # raise ValueError('aa')

    return ex, example_stats
