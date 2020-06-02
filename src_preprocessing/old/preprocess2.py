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

# The average oot is computed for each array instead of across all quarters
# FIXME: why do we need the flux values for the imputed intervals?

"""

# 3rd party
import numpy as np
import tensorflow as tf
import paths
# if 'home6' in paths.path_hpoconfigs:
import matplotlib; matplotlib.use('agg')
import os
import socket

# local
from src_preprocessing.light_curve import kepler_io
from src_preprocessing.light_curve import median_filter
from src_preprocessing.light_curve import util
from src_preprocessing.tf_util import example_util
from src_preprocessing.third_party.kepler_spline import kepler_spline
from src_preprocessing import utils_visualization
from src_preprocessing.utils_centroid_preprocessing import kepler_transform_pxcoordinates_mod13
from src_preprocessing.utils_ephemeris import create_binary_time_series, find_first_epoch_after_this_time


def is_pfe():
    """
    Returns boolean which indicates whether this script is being run on Pleiades or local computer.
    """

    nodename = os.uname().nodename

    if nodename[:3] == 'pfe':
        return True

    if nodename[0] == 'r':
        try:
            int(nodename[-1])
            return True
        except ValueError:
            return False

    return False


def report_exclusion(config, tce, id_str, stderr=None):
    """ Error log is saved into a txt file with the reasons why a given TCE was not preprocessed.

    :param config: Config object with parameters for the preprocessing. Check the Config class
    :param tce: pandas Series, row of the input TCE table Pandas DataFrame.
    :param id_str: str, contains info on the cause of exclusion
    :param stderr: str, error
    :return:
    """

    # create path to exclusion logs directory
    savedir = os.path.join(config.output_dir, 'exclusion_logs')

    # create exclusion logs directory if it does not exist
    os.makedirs(savedir, exist_ok=True)

    # TODO: what if TESS changes to multi-sector analysis; sector becomes irrelevant...
    if config.satellite == 'kepler':
        main_str = 'KepID {} TCE {}'.format(tce.kepid, tce.tce_plnt_num)
    else:  # 'tess'
        main_str = 'TICID {} TCE {} Sector {}'.format(tce.ticid, tce.tce_plnt_num, tce.sector)

    if is_pfe():

        # get node id
        node_id = socket.gethostbyname(socket.gethostname()).split('.')[-1]

        # write to exclusion log pertaining to this process and node
        with open(os.path.join(savedir, 'exclusions_{}_{}-{}.txt'.format(config.process_i, node_id,
                                                                         main_str.replace(" ", ""))),
                  "a") as myfile:
            myfile.write('{}, {}\n{}'.format(main_str, id_str, (stderr, '')[stderr is None]))
    else:
        # write to exclusion log locally
        with open(os.path.join(savedir, 'exclusions-{}.txt'.format(main_str.replace(" ", ""))), "a") as myfile:
            myfile.write('{}, {}\n{}'.format(main_str, id_str, (stderr, '')[stderr is None]))


def read_light_curve(tce, config):
    """ Reads the FITS files pertaining to a Kepler target.

    Args:
        tce: row of DataFrame, information on the TCE (ID, ephemeris, ...).
        config: Config object, preprocessing parameters

    Returns:
        all_time: A list of numpy arrays; the time values of the raw light curve
        all_flux: A list of numpy arrays corresponding to the PDC flux time series
        all_centroid: A list of numpy arrays corresponding to the raw centroid time series
        add_info: A dict with additional data extracted from the FITS files; 'quarter' is a list of quarters for each numpy
        array of the light curve; 'module' is the same but for the module in which the target is in every quarter;
        'target position' is a list of two elements which correspond to the target star position, either in world (RA, Dec)
        or local CCD (x, y) pixel coordinates

    Raises:
      IOError: If the light curve files for this Kepler ID cannot be found.
    """

    # gets all filepaths for the FITS files for this target Kepler ID
    file_names = kepler_io.kepler_filenames(config.lc_data_dir, tce.kepid)

    if not file_names:
        if not config.omit_missing:
            raise IOError("Failed to find .fits files in {} for Kepler ID {}".format(config.lc_data_dir, tce.kepid))
        else:
            report_exclusion(config, tce, 'No available lightcurve .fits files')
            return None, None, None, None

    return kepler_io.read_kepler_light_curve(file_names, centroid_radec=not config.px_coordinates)


def get_gap_indices(flux, checkfuncs=None):
    """ Finds gaps in time series data (where time series is one of [0, nan, inf, -inf])

    :param flux:  flux time-series
    :return:
        id_dict: dict with start and end indices of each gap in flux time series
    """

    # maybe only the checks requested should be performed
    checkfuncdict = {0: flux == 0,
                     'nan': np.isnan(flux),
                     'inf': np.isinf(flux),
                     '-inf': np.isinf(-flux)}

    # set which checks to do
    checkfuncs = checkfuncdict if not checkfuncs else {i: checkfuncdict[i] for i in checkfuncs}

    id_dict = {}
    for checkstr, checkfunc in checkfuncs.items():
        arr = np.where(checkfunc)[0]

    #     id_dict_type = {}
    #     count = 0
    #     for i in range(len(arr)):
    #         if count not in id_dict_type:
    #             id_dict_type[count] = [arr[i], -1]
    #         if i + 1 <= len(arr) - 1 and arr[i + 1] - arr[i] > 1:
    #             id_dict_type[count] = [id_dict_type[count][0], arr[i]]
    #             count += 1
    #         else:
    #             if arr[i] - arr[i - 1] > 1:
    #                 id_dict_type[count] = [arr[i], arr[i]]
    #             else:
    #                 id_dict_type[count] = [id_dict_type[count][0], arr[i]]
    #     id_dict[checkstr] = id_dict_type

        ####
        # CASE NO GAPS
        if len(arr) == 0:
            id_dict[checkstr] = []  # {}  # EMPTY DICT, NONE?
        # CASE ONLY ONE SINGLE IDX GAP
        elif len(arr) == 1:
            id_dict[checkstr] = [[arr[0], arr[0] + 1]]  # {0: [arr[0], arr[0] + 1]}
        # CASE TWO OR MORE IDXS GAP OR MORE THAN ONE GAP
        # id_dict_type = {}  # WHY IS IT A DICT??? IT COULD BE A SIMPLE LIST!!!!!!!!!
        id_dict_type = []
        arr_diff = np.diff(arr)
        jump_idxs = np.where(arr_diff > 1)[0]
        jump_idxs += 1
        jump_idxs = np.insert(jump_idxs, [0, len(jump_idxs)], [0, len(arr)])
        for start, end in zip(jump_idxs[:-1], jump_idxs[1:]):
            # id_dict_type[len(id_dict_type)] = [start, end]
            id_dict_type.append([start, end])
        id_dict[checkstr] = id_dict_type

    return id_dict


def synchronize_centroids_with_flux(all_time, centroid_time, all_centroids, thres=0.005):
    """
    Synchronizes centroid cadences with flux cadences by comparing flux time vector with centroid time vector.
    Operations:
        - Fills centroid cadences with nan's where flux cadence is present and centroid cadence is not present
        - Removes centroid cadences where flux cadences not present

    :param all_time: flux time vector
    :param centroid_time: centroid time vector
    :param all_centroids: centroid vector
    :param thres: float, time cadence match threshold: 0.005 days = ~7 minutes
    :return:
        dict, synchronized all_centroids
    """

    # remove nans in centroid time vector. Nans yield invalid subtractions further on
    finite_id_centr = np.isfinite(centroid_time)
    centroid_time = centroid_time[finite_id_centr]
    all_centroids['x'] = all_centroids['x'][finite_id_centr]
    all_centroids['y'] = all_centroids['y'][finite_id_centr]

    # vertically stack flux time and centroid time, same for centroids and nans. 2nd column id's
    # centroid and flux cadences
    a = np.array([
        np.concatenate((all_time, centroid_time)),
        np.concatenate((np.full(len(all_time), 1), np.full(len(all_centroids['x']), 0))),
        np.concatenate((np.full(len(all_time), np.nan), all_centroids['x'])),
        np.concatenate((np.full(len(all_time), np.nan), all_centroids['y']))
    ]).transpose()

    # first sort by identifier, then by time cadence values. now every first of time cadences which occur in both
    # flux and centroid is of a centroid cadence
    a = a[np.lexsort((a[:, 1], a[:, 0]))]

    # collect rows to delete; keep only centroid cadences for cadences which flux also has and
    # delete centroid time cadences if flux does not have that time cadence
    rows_del = []
    for i in range(len(a) - 1):
        if abs(a[i][0] - a[i + 1][0]) < thres:
            rows_del.append(i + 1)
        elif a[i][1] == 0:
            rows_del.append(i)

    a = np.delete(a, rows_del, axis=0)

    return {'x': a[:, 2], 'y': a[:, 3]}


def _process_tce(tce, table, all_flux, all_time, config, conf_dict, gap_ids=None):
    """ Processes the time-series features for the TCE and returns an Example proto.

    Args:
        tce: row of the input TCE table.
        table: pandas DataFrame, ephemeris information on other TCEs used when gapping
        all_flux: list of numpy arrays, flux time series for TCEs; None when not using whitened data
        all_time: list of numpy arrays, cadences time series for TCEs; None when not using whitened data
        config: Config object, holds preprocessing parameters
        conf_dict:
        gap_ids:

    Returns:
        A tensorflow.train.Example proto containing TCE features.
    """

    if tce['av_training_set'] == 'NTP' and tce['kepid'] == 2447832 and tce['tce_plnt_num'] == 1:
        print(tce[['kepid', 'tce_plnt_num', 'ra', 'dec', 'av_training_set']])
    else:
        return None

    # check if preprocessing pipeline figures are saved for the TCE
    plot_preprocessing_tce = True
    if np.random.random() < 0.01:
        plot_preprocessing_tce = config.plot_figures

    centroid_time = None  # pre-fill

    if all_flux is None:  # all_flux = None if satellite = kepler and non-whitened light curves

        # get cadence, flux and centroid data for the tce
        all_time, all_flux, all_centroids, add_info = read_light_curve(tce, config)

        if plot_preprocessing_tce:
            utils_visualization.plot_centroids(all_time, all_centroids, None, tce, config.px_coordinates,
                                               os.path.join(config.output_dir, 'plots/'), '1_raw',
                                               add_info=add_info)

    else:  # only keep centroids as whitened source does not have centroid information
        centroid_time, _, all_centroids, add_info = read_light_curve(tce, config)

    if all_time is None:
        report_exclusion(config, tce, 'Empty arrays')
        return None

    # for the whitened data series we need to do some extra data manipulation and processing
    if config.whitened:

        # FIXME does it give the expected structure in terms of centroids?
        all_centroids['x'] = np.concatenate(all_centroids['x'])
        all_centroids['y'] = np.concatenate(all_centroids['y'])
        centroid_time = np.concatenate(centroid_time)

        all_flux = np.array(all_flux)
        all_time = np.array(all_time)

        if np.max(all_flux) == 0:
            report_exclusion(config, tce, 'All whitened flux import values are 0')
            return None

        # remove patches of 0's (whitened time series are filled with 0's instead of nan's)
        zero_gaps = get_gap_indices(all_flux, checkfuncs=[0])[0]
        # if removing different valued gaps, we would need an additional step to order the gaps
        # starting from the last gap to the first one in order to not mess with the indexing
        for gap_n in range(len(zero_gaps) - 1, -1, -1):
            gap = zero_gaps[gap_n]
            gap_length = gap[-1] - gap[0] + 1
            if gap_length > 2:  # avoiding gaps smaller than 3 cadences
                all_flux = np.concatenate((all_flux[:gap[0]], all_flux[gap[-1]:]))
                all_time = np.concatenate((all_time[:gap[0]], all_time[gap[-1]:]))

        # align cadences of centroid and time
        all_centroids = synchronize_centroids_with_flux(all_time, centroid_time, all_centroids)
        if len(all_centroids['x']) != len(all_flux):
            report_exclusion(config, tce, 'after sychronizing the centroid with the flux time series, the two time '
                                          'series do not have the same length.')

        # if not config.gapped:
        all_time, all_flux, all_centroids['x'], all_centroids['y'] = \
            [all_time], [all_flux], [all_centroids['x']], [all_centroids['y']]

    # oot_rms = get_centr_oot_rms(all_centroids, all_time, tce, table, config)  # centroid Out Of transit (oot) RMS

    # FIXME: what if removes a whole quarter? need to adjust all_additional_info to it
    # gap other TCEs in the light curve
    if config.gapped:
        all_time, all_flux, all_centroids, gap_ids = \
            kepler_io.gap_other_tces(all_time, all_flux, all_centroids, tce, table, config, conf_dict)

    # Remove timestamps with NaN or infinite time, flux or centroid values in each quarter
    # At least some of these NaNs come from gapping the time series
    # Other NaNs can come from missing time values from the fits files
    for i, time_series_set in enumerate(zip(all_flux, all_centroids['x'], all_centroids['y'])):
        finite_id = np.isfinite(all_time[i])
        for time_series in time_series_set:
            finite_id = np.logical_and(finite_id, np.isfinite(time_series))
        all_time[i] = all_time[i][finite_id]
        all_flux[i] = time_series_set[0][finite_id]
        all_centroids['x'][i] = time_series_set[1][finite_id]
        all_centroids['y'][i] = time_series_set[2][finite_id]

    if plot_preprocessing_tce:
        utils_visualization.plot_centroids(all_time, all_centroids, None, tce, config.px_coordinates,
                                           os.path.join(config.output_dir, 'plots/'), '2_rawwithoutnans',
                                           add_info=add_info)

    # preprocess the flux and centroid time series
    time, flux, centroids = process_light_curve(all_time, all_flux, all_centroids, gap_ids, config, add_info,
                                                tce, plot_preprocessing_tce)

    # create the different channels
    return generate_example_for_tce(time, flux, centroids, tce, config, plot_preprocessing_tce)


def process_light_curve(all_time, all_flux, all_centroids, gap_ids, config, add_info, tce,
                        plot_preprocessing_tce=False):
    """ Removes low-frequency variability from the flux and centroid time-series.

    Args:
      all_time: A list of numpy arrays; the cadences of the raw light curve in each quarter
      all_flux: A list of numpy arrays corresponding to the PDC flux time series in each quarter
      all_centroids: A list of numpy arrays corresponding to the raw centroid time series in each quarter
      gap_ids: A list of numpy arrays; the cadences and flux for the imputed intervals
      config: Config object, holds preprocessing parameters
      add_info: A dict with additional data extracted from the FITS files; 'quarter' is a list of quarters for each
      numpy array of the light curve; 'module' is the same but for the module in which the target is in every quarter;
      'target position' is a list of two elements which correspond to the target star position, either in world
      (RA, Dec) or local CCD (x, y) pixel coordinates
      tce: pandas Series, row of the TCE table for the TCE that is being processed; it contains data such as the
      ephemeris
      plot_preprocessing_tce: bool, if True plots figures for several steps in the preprocessing pipeline

    Returns:
      time: 1D NumPy array; the time values of the light curve
      flux: 1D NumPy array; the normalized flux values of the light curve
      centroid_dist NumPy array; the distance of the corrected centroid time series to the target position
    """

    # FIXME: why do we use this if we concatenate the arrays right after? Check Shallue's code (why 0.75 gap?)
    # Split on gaps.
    all_time, all_flux, all_centroids, add_info = util.split_wcentroids(all_time, all_flux, all_centroids, add_info,
                                                                        gap_width=0.75)

    # pixel coordinate transformation for targets on module 13 for Kepler
    if config.px_coordinates and config.satellite == 'kepler':
        if add_info['module'][0] == 13:
            all_centroids = kepler_transform_pxcoordinates_mod13(all_centroids, add_info)

            if plot_preprocessing_tce:
                utils_visualization.plot_centroids(all_time, all_centroids, None, tce, config.px_coordinates,
                                                   os.path.join(config.output_dir, 'plots/'),
                                                   '2_rawaftertransformation', add_info=add_info)

    # get epoch of first transit for each time array
    first_transit_time_all = [find_first_epoch_after_this_time(tce["tce_time0bk"], tce["tce_period"], time[0])
                              for time in all_time]
    # create binary time series for each time array in which in-transit points are labeled as 1's, otherwise as 0's
    binary_time_all = [create_binary_time_series(time, first_transit_time, tce["tce_duration"], tce["tce_period"])
                       for first_transit_time, time in zip(first_transit_time_all, all_time)]

    if plot_preprocessing_tce:
        utils_visualization.plot_binseries_flux(all_time, all_flux, binary_time_all, tce,
                                                os.path.join(config.output_dir, 'plots/'), '3_binarytimeseriesandflux')

    # get centroid time series out-of-transit points
    centroid_oot = {coord: [centroids[np.where(binary_time == 0)] for binary_time, centroids in
                     zip(binary_time_all, all_centroids[coord])] for coord in all_centroids}

    # TODO: how to compute the average oot? mean, median, other...
    # average the centroid time series values for the oot points
    # avg_centroid_oot = {coord: np.median(np.concatenate(centroid_oot[coord])) for coord in centroid_oot}
    avg_centroid_oot = {coord: [np.median(centroid_oot[coord][i]) for i in range(len(centroid_oot[coord]))]
                        for coord in centroid_oot}

    # spline fitting and normalization - fit a piecewise-cubic spline with default arguments
    # FIXME: wouldn't it be better to fit a spline to oot values or a Savitzky Golay filter
    #  (as Jeff and Doug mentioned)?
    # FIXME: fit spline only to the oot values - linearly interpolate the transits; same thing for flux
    # all_centroids_lininterp = all_centroids
    # for i in range(len(all_centroids_lininterp['x'])):
    #     idxs_it = np.where(binary_time_all[i] == 1)[0]
    #     idxs_lim = np.where(np.diff(idxs_it) > 1)[0] + 1
    #     start_idxs = np.insert(idxs_lim, 0, 0)
    #     end_idxs = np.append(idxs_lim, len(idxs_it))
    #     for start_idx, end_idx in zip(start_idxs, end_idxs):
    #
    #         idxs_interp = np.array([start_idx - 1, end_idx])
    #         all_centroids_lininterp['x'][i][start_idx:end_idx] = np.interp(np.arange(start_idx, end_idx),
    #                                                                        idxs_interp,
    #                                                                        all_centroids_lininterp['x'][i][
    #                                                                            np.array([start_idx - 1, end_idx])])
    #         all_centroids_lininterp['y'][i][start_idx:end_idx] = np.interp(np.arange(start_idx, end_idx),
    #                                                                        idxs_interp,
    #                                                                        all_centroids_lininterp['y'][i][
    #                                                                            np.array([start_idx - 1, end_idx])])
    #
    # spline_centroid = {coord: kepler_spline.fit_kepler_spline(all_time, all_centroids_lininterp[coord],
    #                                                           verbose=False)[0] for coord in all_centroids_lininterp}
    spline_centroid = {coord: kepler_spline.fit_kepler_spline(all_time, all_centroids[coord], verbose=False)[0]
                       for coord in all_centroids}

    if not config.whitened:
        # Fit a piecewise-cubic spline with default arguments.
        spline_flux = kepler_spline.fit_kepler_spline(all_time, all_flux, verbose=False)[0]

    if plot_preprocessing_tce:
        if not config.whitened:
            utils_visualization.plot_flux_fit_spline(all_time, all_flux, spline_flux, tce,
                                                     os.path.join(config.output_dir, 'plots/'),
                                                     '4_smoothingandnormalizationflux')

        utils_visualization.plot_centroids(all_time, all_centroids, spline_centroid, tce, config.px_coordinates,
                                           os.path.join(config.output_dir, 'plots/'),
                                           '4_smoothingandnormalizationcentroid')

    # In rare cases the piecewise spline contains NaNs in places the spline could not be fit. We can't normalize those
    # points if the spline isn't defined there. Instead we just remove them.
    finite_i_centroid = [np.logical_and(np.isfinite(spline_centroid['x'][i]), np.isfinite(spline_centroid['y'][i]))
                         for i in range(len(spline_centroid['x']))]
    if not config.whitened:
        finite_i_flux = [np.isfinite(spline_flux[i]) for i in range(len(spline_flux))]
        finite_i = [np.logical_and(finite_i_centroid[i], finite_i_flux[i]) for i in range(len(finite_i_centroid))]
    else:
        finite_i = finite_i_centroid

    # remove cadences for which the spline did not fit
    all_time = [all_time[i][finite_i[i]] for i in range(len(all_time))]
    binary_time_all = [binary_time_all[i][finite_i[i]] for i in range(len(binary_time_all))]

    # normalize by the spline
    # all_centroids = {coord: [all_centroids[coord][i][finite_i[i]] / spline_centroid[coord][i][finite_i[i]] *
    #                          avg_centroid_oot[coord] for i in range(len(spline_centroid[coord]))]
    #                  for coord in all_centroids}
    all_centroids = {coord: [all_centroids[coord][i][finite_i[i]] / spline_centroid[coord][i][finite_i[i]] *
                             avg_centroid_oot[coord][i] for i in range(len(spline_centroid[coord]))]
                     for coord in all_centroids}

    if not config.whitened:
        all_flux = [all_flux[i][finite_i[i]] / spline_flux[i][finite_i[i]] for i in range(len(spline_flux))]

    # get the position of the target star
    target_coords = add_info['target position']

    # compute the new average oot after the spline fitting and normalization
    # TODO: how to compute the average oot? mean, median, other...
    centroid_oot = {coord: [centroids[np.where(binary_time == 0)] for binary_time, centroids in
                            zip(binary_time_all, all_centroids[coord])] for coord in all_centroids}
    # avg_centroid_oot = {coord: np.median(np.concatenate(centroid_oot[coord])) for coord in all_centroids}
    avg_centroid_oot = {coord: [np.median(centroid_oot[coord][i]) for i in range(len(all_centroids[coord]))]
                        for coord in all_centroids}

    if plot_preprocessing_tce:
        utils_visualization.plot_centroids_it_oot(all_time, binary_time_all, all_centroids, centroid_oot,
                                                  avg_centroid_oot, target_coords, tce, config.px_coordinates,
                                                  os.path.join(config.output_dir, 'plots/'),
                                                  '5_centroidtimeseries_it-ot-target')

    # compute the corrected centroid time-series normalized by the transit depth fraction and centered on the avg oot
    # centroid position
    # all_centroids = {coord: [-((all_centroids[coord][i] - avg_centroid_oot[coord]) / (tce['tce_depth'] * 1e-6)) +
    #                          avg_centroid_oot[coord] for i in range(len(all_centroids[coord]))]
    #                  for coord in all_centroids}
    all_centroids = {coord: [-((all_centroids[coord][i] - avg_centroid_oot[coord][i]) / (tce['tce_depth'] * 1e-6)) +
                             avg_centroid_oot[coord][i] for i in range(len(all_centroids[coord]))]
                     for coord in all_centroids}

    if plot_preprocessing_tce:
        utils_visualization.plot_corrected_centroids(all_time, all_centroids, avg_centroid_oot, target_coords, tce,
                                                     config.px_coordinates, os.path.join(config.output_dir, 'plots/'),
                                                     '6_correctedcentroidtimeseries')

    if config.px_coordinates:
        # compute the euclidean distance of the corrected centroid time series to the target star position
        centroid_dist = [np.sqrt(np.square(all_centroids['x'][i] - target_coords[0]) +
                          np.square(all_centroids['y'][i] - target_coords[1]))
                          for i in range(len(all_centroids['x']))]
    else:

        # compute the angular distance of the corrected centroid time series to the target star position
        centroid_dist = [np.sqrt(np.square((all_centroids['x'][i] - target_coords[0]) *
                                           np.cos(all_centroids['y'][i] - target_coords[1])) +
                                 np.square(all_centroids['y'][i] - target_coords[1]))
                         for i in range(len(all_centroids['x']))]

    # # get the average oot distance
    # centroid_dist_oot = [centroid_dist_block[np.where(binary_time == 0)] for binary_time, centroid_dist_block
    #                      in zip(binary_time_all, centroid_dist)]
    # avg_centroid_dist_oot = np.median(np.concatenate(centroid_dist_oot))

    # convert from degree to arcsec
    if not config.px_coordinates:
        centroid_dist = [centroid_dist_arr * 3600 for centroid_dist_arr in centroid_dist]

    if plot_preprocessing_tce:
        utils_visualization.plot_dist_centroids(all_time, centroid_dist, None, None, tce, config.px_coordinates,
                                                os.path.join(config.output_dir, 'plots/'), '7_distcentr')

    # # spline fitting and normalization - fit a piecewise-cubic spline with default arguments
    # # FIXME: wouldn't it be better to fit a spline to oot values or a Savitzky Golay filter
    # #  (as Jeff and Doug mentioned)?
    # # FIXME: fit spline only to the oot values - linearly interpolate the transits; same thing for flux
    # spline_centroid = kepler_spline.fit_kepler_spline(all_time, centroid_dist, verbose=False)[0]
    # if not config.whitened:
    #     # Fit a piecewise-cubic spline with default arguments.
    #     spline_flux = kepler_spline.fit_kepler_spline(all_time, all_flux, verbose=False)[0]
    #
    # if plot_preprocessing_tce:
    #     utils_visualization.plot_flux_fit_spline(all_time, all_flux, spline_flux, tce,
    #                                              os.path.join(config.output_dir, 'plots/'),
    #                                              '7_smoothingandnormalizationflux')
    #     utils_visualization.plot_dist_centroids(all_time, centroid_dist, spline_centroid, avg_centroid_dist_oot,
    #                                             tce, config.px_coordinates, os.path.join(config.output_dir, 'plots/'),
    #                                             '7_smoothingandnormalizationcentroid')
    #
    # # In rare cases the piecewise spline contains NaNs in places the spline could not be fit. We can't normalize those
    # # points if the spline isn't defined there. Instead we just remove them.
    # finite_i_centroid = [np.isfinite(spline_centroid[i]) for i in range(len(spline_centroid))]
    # if not config.whitened:
    #     finite_i_flux = [np.isfinite(spline_flux[i]) for i in range(len(spline_flux))]
    #     finite_i = [np.logical_or(finite_i_centroid[i], finite_i_flux[i]) for i in range(len(finite_i_centroid))]
    # else:
    #     finite_i = finite_i_centroid
    #
    # # remove cadences for which the spline did not fit
    # all_time = [all_time[i][finite_i[i]] for i in range(len(all_time))]
    #
    # # normalize by the spline
    # centroid_dist = [centroid_dist[i][finite_i[i]] / spline_centroid[i][finite_i[i]] * avg_centroid_dist_oot
    #                  for i in range(len(spline_centroid))]
    #
    # if not config.whitened:
    #     all_flux = [all_flux[i][finite_i[i]] / spline_flux[i][finite_i[i]] for i in range(len(spline_flux))]

    centroid_dist = np.concatenate(centroid_dist)
    flux = np.concatenate(all_flux)
    time = np.concatenate(all_time)

    # impute the time series with noise base on global estimates of median and std
    if gap_ids:
        med = {'flux': np.median(flux), 'centroid': np.median(centroid_dist)}
        # robust std estimator of the time series
        std_rob_estm = {'flux': np.median(np.abs(flux - med['flux'])) * 1.4826,
                        'centroid': np.median(np.abs(centroid_dist - med['centroid'])) * 1.4826}
        for [time_slice, flux_slice, centroid_slice] in gap_ids:

            imputed_flux = med['flux'] + np.random.normal(0, std_rob_estm['flux'], flux_slice.shape)
            flux = np.append(flux, imputed_flux.astype(flux.dtype))

            imputed_centr = med['centroid'] + np.random.normal(0, std_rob_estm['centroid'], centroid_slice.shape)
            centroid_dist = np.append(centroid_dist, imputed_centr.astype(centroid_dist.dtype))

            time = np.append(time, time_slice.astype(time.dtype))

    return time, flux, centroid_dist


def min_max_normalization(arr, max_val, min_val):
    """ Min-max normalization.

    :param arr: array
    :param max_val: float, max val
    :param min_val: float, min val
    :return:
        normalized array
    """

    return (arr - min_val) / (max_val - min_val)


def phase_fold_and_sort_light_curve(time, flux, centroids, period, t0):
    """Phase folds a light curve and sorts by ascending time.

    Args:
      time: 1D NumPy array of time values.
      flux: 1D NumPy array of flux values.
      centroids: 1D NumPy array of centroid values
      period: A positive real scalar; the period to fold over.
      t0: The center of the resulting folded vector; this value is mapped to 0.

    Returns:
      folded_time: 1D NumPy array of phase folded time values in
          [-period / 2, period / 2), where 0 corresponds to t0 in the original
          time array. Values are sorted in ascending order.
      folded_flux: 1D NumPy array. Values are the same as the original input
          array, but sorted by folded_time.
      folded_centroids: 1D NumPy array. Values are the same as the original input array, but sorted by folded_time.
    """

    # Phase fold time.
    time = util.phase_fold_time(time, period, t0)

    # Sort by ascending time.
    sorted_i = np.argsort(time)
    time = time[sorted_i]
    flux = flux[sorted_i]
    centroids = centroids[sorted_i]

    return time, flux, centroids


def phase_fold_and_sort_light_curve_odd_even(time, flux, centroids, period, t0):
    """Creates separate phase-folded time vectors for odd and even periods.

    :param time: 1D NumPy array of time values
    :param flux: 1D NumPy array of flux values
    :param centroids: 1D NumPy array of centroid values
    :param period: A positive real scalar; the period to fold over.
    :param t0: The center of the resulting folded vector; this value is mapped to 0.
    :return:
        Two tuples of size three, each of which contains odd or even time values, flux values, and centroid values.
        time: 1D NumPy array of phase folded time values in [-period / 2, period / 2), where 0 corresponds to t0
            in the original time array. Values are sorted in ascending order.
        folded_flux: 1D NumPy array. Values are the same as the original input array, but sorted by folded_time.
        folded_centroids: 1D NumPy array. Values are the same as the original input array, but sorted by folded_time.
    """

    half_period = period / 2
    odd_indices = np.array([], dtype='int64')
    even_indices = np.array([], dtype='int64')

    switcher = 1
    i = t0

    # FIXME: using switcher here just makes sure that the period centered at t0 is odd - why is this relevant?
    # is the number of periods between t0 and tmin even (switcher=1) or odd (switcher=-1)?
    # starts counting in the period centered in t0 (assumed as odd period)
    # this is done so that we get the first valid period
    # if np.min(time) < i - half_period:
    if np.min(time) < i + half_period:
        # is the right side of the current period interval after minimum time?
        # while i - half_period >= np.min(time):
        # while i + half_period >= np.min(time):
        while i + half_period > np.min(time):
            # iterate to the previous period
            i -= period
            switcher *= -1

    # start in the first valid period interval
    i += period

    # starting from the first valid period interval, is the left side of the current period interval before the maximum
    # time?
    while i - half_period <= np.amax(time):
        if switcher == 1:
            # add odd indices
            odd_indices = np.concatenate(
                (odd_indices, np.where(
                    np.logical_and(
                        time < i + half_period,
                        time >= i - half_period
                    ))),
                axis=None
            )
        else:
            # add even indices
            even_indices = np.concatenate(
                (even_indices, np.where(
                    np.logical_and(
                        time < i + half_period,
                        time >= i - half_period
                    ))),
                axis=None
            )

        # iterate to the next period
        i += period
        # alternate between odd and even periods
        switcher *= -1

    # get odd and even values
    odd_time = np.take(time, odd_indices)
    even_time = np.take(time, even_indices)
    odd_flux = np.take(flux, odd_indices)
    even_flux = np.take(flux, even_indices)
    odd_centroids = np.take(centroids, odd_indices)
    even_centroids = np.take(centroids, even_indices)

    # Phase fold time.
    odd_time = util.phase_fold_time(odd_time, period, t0)
    even_time = util.phase_fold_time(even_time, period, t0)

    # Sort by ascending time.
    sorted_i_odd = np.argsort(odd_time)
    sorted_i_even = np.argsort(even_time)

    odd_result = odd_time[sorted_i_odd]
    even_result = even_time[sorted_i_even]
    odd_flux = odd_flux[sorted_i_odd]
    even_flux = even_flux[sorted_i_even]
    odd_centroids = odd_centroids[sorted_i_odd]
    even_centroids = even_centroids[sorted_i_even]

    assert(len(odd_time) + len(even_time) - len(time) == 0 and len(set(odd_time) & set(even_time)) == 0)

    return (odd_result, odd_flux, odd_centroids), (even_result, even_flux, even_centroids)


def generate_view(time, flux, num_bins, bin_width, t_min, t_max,
                  centering=True, normalize=True, centroid=False):
    """Generates a view of a phase-folded light curve using a median filter.

    Args:
      time: 1D array of time values, sorted in ascending order.
      flux: 1D array of flux/centroid values.
      num_bins: The number of intervals to divide the time axis into.
      bin_width: The width of each bin on the time axis.
      t_min: The inclusive leftmost value to consider on the time axis.
      t_max: The exclusive rightmost value to consider on the time axis.
      centering: bool, whether to center the view by subtracting the median
      normalize: Whether to perform normalization
      centroid: bool, if True considers these view a centroid time series

    Returns:
      1D NumPy array of size num_bins containing the median flux values of
      uniformly spaced bins on the phase-folded time axis.
    """

    # # TODO: do we need this check?
    # if centroid:
    #     finite_idxs = np.isfinite(flux)
    #     flux_allfinite = flux[finite_idxs]
    #     time_allfinite = time[finite_idxs]
    # else:
    #     flux_allfinite = flux
    #     time_allfinite = time

    # binning using median
    # view = median_filter.median_filter(time_allfinite, flux_allfinite, num_bins, bin_width, t_min, t_max)
    view = median_filter.median_filter(time, flux, num_bins, bin_width, t_min, t_max)

    # global median centering
    if centering:
        view -= np.median(view)

    # normalization
    if normalize:
        view = normalize_view(view, val=None, centroid=centroid)

    return view


def normalize_view(view, val=None, centroid=False):
    """ Normalize the phase-folded time series.

    :param view: array, phase-folded time series
    :param val: float, value used to normalize the time series
    :param centroid: bool, True for centroid time series
    :return:
        array, normalized phase-folded time series
    """

    # # median centering
    # view -= np.median(view)

    # for the centroid time series
    # range [new_min_val, 1], assuming max is positive, which should be since we are removing the median from a
    # non-negative time series
    # for the flux time series
    # range [-1, new_max_val], assuming min is negative, if not [1, new_max_val]
    if val is None:
        val = np.abs(np.max(view)) if centroid else np.abs(np.min(view))

    # TODO: divide view by an infinitesimal instead of returning it? The normalization value is provided to the function
    #  so I dont think that makes sense
    if val == 0:
        print('Dividing view by 0. Returning the non-normalized view.')
        return view
    else:
        view = view / val
        # view /= val

    return view


def global_view(time, flux, period, num_bins=2001, bin_width_factor=1 / 2001, centroid=False, normalize=True,
                centering=True):
    """Generates a 'global view' of a phase folded light curve.

    See Section 3.3 of Shallue & Vanderburg, 2018, The Astronomical Journal.
    http://iopscience.iop.org/article/10.3847/1538-3881/aa9e09/meta

    Args:
      time: 1D array of time values, sorted in ascending order.
      flux: 1D array of flux values.
      period: The period of the event (in days).
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
        bin_width=period * bin_width_factor,
        t_min=-period / 2,
        t_max=period / 2,
        centroid=centroid,
        normalize=normalize,
        centering=centering)


def local_view(time,
               flux,
               period,
               duration,
               num_bins=201,
               bin_width_factor=0.16,
               num_durations=4,
               centroid=False,
               normalize=True,
               centering=True):
    """Generates a 'local view' of a phase folded light curve.

    See Section 3.3 of Shallue & Vanderburg, 2018, The Astronomical Journal.
    http://iopscience.iop.org/article/10.3847/1538-3881/aa9e09/meta

    Args:
      time: 1D array of time values, sorted in ascending order.
      flux: 1D array of flux values.
      period: The period of the event (in days).
      duration: The duration of the event (in days).
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
        num_bins=num_bins,
        bin_width=duration * bin_width_factor,
        t_min=max(-period / 2, -duration * num_durations),
        t_max=min(period / 2, duration * num_durations),
        centroid=centroid,
        normalize=normalize,
        centering=centering)


# def generate_example_for_tce(time, flux, centroids, tce, config, transit_times):
def generate_example_for_tce(time, flux, centroids, tce, config, plot_preprocessing_tce=False):
    """ Generates a tf.train.Example representing an input TCE.

    Args:
      time: 1D NumPy array; the time values of the light curve.
      flux: 1D NumPy array; the flux values of the light curve.
      centroids: 1D NumPy array; the flux values of the light curve.
      tce: Dict-like object containing at least 'tce_period', 'tce_duration', and
        'tce_time0bk'. Additional items are included as features in the output.
      config: Config object; preprocessing parameters.
      plot_preprocessing_tce: bool, if True plots figures for some steps while generating the inputs.

    Returns:
      A tf.train.Example containing features. These features can be time series, stellar parameters, statistical
      quantities, .... it returns None if some exception while creating these features occurs
    """

    # get ephemeris
    period = tce["tce_period"]
    duration = tce["tce_duration"]
    t0 = tce["tce_time0bk"]

    # # phase folding for odd and even time series
    # (odd_time, odd_flux, odd_centroid), \
    #     (even_time, even_flux, even_centroid) = phase_fold_and_sort_light_curve_odd_even(time, flux, centroids, period,
    #                                                                                      t0)

    # phase folding for flux and centroid time series
    time, flux, centroid = phase_fold_and_sort_light_curve(time, flux, centroids, period, t0)

    # Make output proto
    ex = tf.train.Example()

    # Set time series features
    try:
        # get flux views
        glob_view = global_view(time, flux, period, normalize=True, centering=True,
                                num_bins=config.num_bins_glob, bin_width_factor=config.bin_width_factor_glob)
        loc_view = local_view(time, flux, period, duration, normalize=True, centering=True,
                              num_bins=config.num_bins_loc, bin_width_factor=config.bin_width_factor_loc)

        # get centroid views
        glob_view_centr = global_view(time, centroid, period, centroid=True, normalize=False, centering=False,
                                      num_bins=config.num_bins_glob, bin_width_factor=config.bin_width_factor_glob)
        loc_view_centr = local_view(time, centroid, period, duration, centroid=True, normalize=False, centering=False,
                                    num_bins=config.num_bins_loc, bin_width_factor=config.bin_width_factor_loc)

        views = {'global_view': glob_view, 'local_view': loc_view, 'global_view_centr': glob_view_centr,
                 'local_view_centr': loc_view_centr}

        if plot_preprocessing_tce:
            utils_visualization.plot_all_views(views, tce, (2, 2), os.path.join(config.output_dir, 'plots/'),
                                               '8_views_nonnormalizedcentroids')

        # # get odd views (flux and centroid)
        # glob_view_odd = global_view(odd_time, odd_flux, period, normalize=False, centering=True,
        #                             num_bins=config.num_bins_glob, bin_width_factor=config.bin_width_factor_glob)
        # loc_view_odd = local_view(odd_time, odd_flux, period, duration, normalize=False, centering=True,
        #                           num_bins=config.num_bins_loc, bin_width_factor=config.bin_width_factor_loc)

        # glob_view_odd_centr = global_view(odd_time, odd_centroid, period, centroid=True, normalize=False,
        #                                   num_bins=config.num_bins_glob, bin_width_factor=config.bin_width_factor_glob)
        # loc_view_odd_centr = local_view(odd_time, odd_centroid, period, duration, centroid=True, normalize=False,
        #                                 num_bins=config.num_bins_loc, bin_width_factor=config.bin_width_factor_loc)

        # # get even views (flux and centroid)
        # glob_view_even = global_view(even_time, even_flux, period, normalize=False, centering=True,
        #                              num_bins=config.num_bins_glob, bin_width_factor=config.bin_width_factor_glob)
        # loc_view_even = local_view(even_time, even_flux, period, duration, normalize=False, centering=True,
        #                            num_bins=config.num_bins_loc, bin_width_factor=config.bin_width_factor_loc)

        # glob_view_even_centr = global_view(even_time, even_centroid, period, centroid=True, normalize=False,
        #                                    num_bins=config.num_bins_glob, bin_width_factor=config.bin_width_factor_glob)
        # loc_view_even_centr = local_view(even_time, even_centroid, period, duration, centroid=True, normalize=False,
        #                                  num_bins=config.num_bins_loc, bin_width_factor=config.bin_width_factor_loc)

        # # normalize odd and even views by their joint minimum (flux)/maximum (centroid)
        # val_norm = np.abs(min(np.min(glob_view_even), np.min(glob_view_odd)))
        # glob_view_even = normalize_view(glob_view_even, val=val_norm)
        # glob_view_odd = normalize_view(glob_view_odd, val=val_norm)
        #
        # val_norm = np.abs(min(np.min(loc_view_even), np.min(loc_view_odd)))
        # loc_view_even = normalize_view(loc_view_even, val=val_norm)
        # loc_view_odd = normalize_view(loc_view_odd, val=val_norm)

        # # TODO: should we normalize the odd and even centroid jointly or use the normalization factor used for the main
        # #   centroid time series?
        # # val_norm = np.abs(max(np.max(glob_view_even_centr), np.max(glob_view_odd_centr)))
        # # glob_view_even_centr = normalize_view(glob_view_even_centr, val=val_norm, centroid=True)
        # # glob_view_odd_centr = normalize_view(glob_view_odd_centr, val=val_norm, centroid=True)
        # #
        # # val_norm = np.abs(max(np.max(loc_view_even_centr), np.max(loc_view_odd_centr)))
        # # loc_view_odd_centr = normalize_view(loc_view_odd_centr, val=val_norm, centroid=True)
        # # loc_view_even_centr = normalize_view(loc_view_even_centr, val=val_norm, centroid=True)
        #
        # # # normalize the global and local views centroids by the maximum value in the training set
        # # max_centr = np.load('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Kepler_planet_finder/src_preprocessing/'
        # #                     'max_centr_trainingset.npy').item()
        # # glob_view_centr = normalize_view(glob_view_centr, val=max_centr['global_view'], centroid=True)
        # # loc_view_centr = normalize_view(loc_view_centr, val=max_centr['local_view'], centroid=True)

        # # if plot_preprocessing_tce:
        # #     views = {'global_view': glob_view, 'local_view': loc_view, 'global_view_odd': glob_view_odd,
        # #              'local_view_odd': loc_view_odd, 'global_view_even': glob_view_even,
        # #              'local_view_even': loc_view_even, 'global_view_centr': glob_view_centr,
        # #              'local_view_centr': loc_view_centr}
        # #
        # #     plot_all_views(views, tce, '5nonnormalized_centroids', (2, 4))
        #

        # load training set statistics for normalization of the centroid time-series views
        # stats_trainset = np.load(config.stats_preproc_filepath).item()

        # if plot_preprocessing_tce:
        #     views = {'global_view': glob_view, 'local_view': loc_view,
        #              'global_view_centr': glob_view_centr, 'local_view_centr': loc_view_centr}
        #     # views = {'global_view': glob_view, 'local_view': loc_view, 'global_view_odd': glob_view_odd,
        #     #          'local_view_odd': loc_view_odd, 'global_view_even': glob_view_even,
        #     #          'local_view_even': loc_view_even, 'global_view_centr': glob_view_centr,
        #     #          'local_view_centr': loc_view_centr}
        #
        #     utils_visualization.plot_all_views(views, tce, (2, 2), os.path.join(config.output_dir, 'plots/'),
        #                                        '8_views_nonnormalizedcentroids')

        # # FDL centroid normalization
        # # median centering using median computed based on the training set
        # glob_view_centr1 = glob_view_centr - stats_trainset['global_view_centr']['median_oot']
        # loc_view_centr1 = loc_view_centr - stats_trainset['local_view_centr']['median_oot']
        # # normalization using std of the centroid and flux views computed based on the training set
        # glob_view_centr1 = normalize_view(glob_view_centr1, val=stats_trainset['global_view_centr']['std_oot'] /
        #                                                        np.std(glob_view), centroid=True)
        # loc_view_centr1 = normalize_view(loc_view_centr1, val=stats_trainset['local_view_centr']['std_oot'] /
        #                                                      np.std(loc_view), centroid=True)

        # min-max normalization of the centroid views computed based on the training set
        # glob_view_centr2 = min_max_normalization(glob_view_centr,
        #                                          stats_trainset['global_view_centr']['max'],
        #                                          stats_trainset['global_view_centr']['min'])
        # loc_view_centr2 = min_max_normalization(loc_view_centr,
        #                                         stats_trainset['local_view_centr']['max'],
        #                                         stats_trainset['local_view_centr']['min'])
        # glob_view_centr2 = min_max_normalization(glob_view_centr,
        #                                          glob_view_centr.max(),
        #                                          glob_view_centr.min())
        # loc_view_centr2 = min_max_normalization(loc_view_centr,
        #                                         loc_view_centr.max(),
        #                                         loc_view_centr.min())

        # # individual median centering
        # glob_view_centr2 = glob_view_centr2 - np.median(glob_view_centr2)
        # loc_view_centr2 = loc_view_centr2 - np.median(loc_view_centr2)

        # views = {'global_view': glob_view, 'local_view': loc_view, 'global_view_centr_fdl': glob_view_centr1,
        #          'local_view_centr_fdl': loc_view_centr1, 'global_view_centr': glob_view_centr2,
        #          'local_view_centr': loc_view_centr2, 'global_view_odd': glob_view_odd,
        #          'local_view_odd': loc_view_odd, 'global_view_even': glob_view_even,
        #          'local_view_even': loc_view_even}
        # views = {'global_view': glob_view, 'local_view': loc_view, 'global_view_centr': glob_view_centr2,
        #          'local_view_centr': loc_view_centr2, 'global_view_odd': glob_view_odd,
        #          'local_view_odd': loc_view_odd, 'global_view_even': glob_view_even,
        #          'local_view_even': loc_view_even}

        # if plot_preprocessing_tce:
        #     utils_visualization.plot_all_views(views, tce, (2, 2), os.path.join(config.output_dir, 'plots/'),
        #                                        '8_views_normalizedcentroids')

    except Exception as e:
        report_exclusion(config, tce, 'Error when creating views', stderr=e)
        return None

    # set time series features in the tfrecord
    for view in views:
        example_util.set_float_feature(ex, view, views[view])

    # TODO: exclude TCEs that have at least one view with at least a NaN value?
    for view in views:
        if not np.any(np.isfinite(views[view])):
            report_exclusion(config, tce, 'Views are NaNs.')
            return None

    # # save statistics about the non-normalized views in numpy files, one per TCE
    # extrema_centr = {'glob_view_centr': {'max': np.nanmax(glob_view_centr), 'min': np.nanmin(glob_view_centr)},
    #                  'loc_view_centr': {'max': np.nanmax(loc_view_centr), 'min': np.nanmin(loc_view_centr)}}
    # np.save(os.path.join(config.output_dir, 'stats',
    #                      'extrema_centroid-{}_{}'.format(tce['kepid'], tce['tce_plnt_num'])), extrema_centr)

    # set other features from the TCE table - TCE and stellar parameters
    for name, value in tce.items():
        example_util.set_feature(ex, name, [value])

    return ex
