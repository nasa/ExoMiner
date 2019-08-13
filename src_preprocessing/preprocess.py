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

Functions for reading and preprocessing light curves

Authors:
- Laurent Wilkens
- Nikash Walia

"""

# 3rd party
import numpy as np
import tensorflow as tf
import os
import socket

# local
from src_preprocessing.light_curve import kepler_io
from src_preprocessing.light_curve import median_filter
from src_preprocessing.light_curve import util
from src_preprocessing.tf_util import example_util
from src_preprocessing.third_party.kepler_spline import kepler_spline


def is_pfe():
    """
    Returns boolean which indicates whether this script is being run on pleiades or local computer
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
    """

    :param config: Config object with parameters for the preprocessing. Check the Config class
    :param tce: row of the input TCE table Pandas DataFrame.
    :param id_str: str, contains info on the cause of exclusion
    :param stderr: str, error
    :return:
    """

    if is_pfe():

        # create path to exclusion logs directory
        savedir = os.path.join(config.w_dir, 'exclusion_logs')
        # create exclusion logs directory if it does not exist
        os.makedirs(savedir, exist_ok=True)
        # get node id
        node_id = socket.gethostbyname(socket.gethostname()).split('.')[-1]

        # write to exclusion log pertaining to this process and node
        with open(os.path.join(savedir, 'exclusions_%d_%s.txt' % (config.process_i, node_id)), "a") as myfile:
            myfile.write('kepid: {}, tce_n: {}, {}{}'.format(tce.kepid, tce.tce_plnt_num, id_str,
                                                             ('\n' + stderr, '\n')[stderr is None]))

        # if stderr is None:
        #     with open(os.path.join(savedir, 'exclusions_%d_%s.txt' % (config.process_i, node_id)), "a") as myfile:
        #         myfile.write('kepid: %d, tce_n: %d, %s' % (tce.kepid, tce.tce_plnt_num, id_str))
        # else:
        #     with open(os.path.join(savedir, 'stderrs_%d_%s.txt' % (config.process_i, node_id)), "a") as myfile:
        #         myfile.write('\nkepid: %d, tce_n: %d, %s:\n%s' % (tce.kepid, tce.tce_plnt_num, id_str, stderr))


def get_gap_indices(flux, checkfuncs=None):
    """
    Finds gaps in time series data (where time series is one of [0, nan, inf, -inf])
    TODO: test the new code and check how other functions are expecting it
          change the variable id_dict_type to a list and change where the function is called

    :param flux: 1D numpy array, flux time-series
    :param checkfuncs: list, defines which gap values to look for in the flux time series
    :return: id_dict: dict with start and end indices of each gap in flux time series
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

        # id_dict_type = {}
        # count = 0  # number of gaps intervals
        # for i in range(len(arr)):
        #     if count not in id_dict_type:
        #         id_dict_type[count] = [arr[i], -1]
        #     if i + 1 <= len(arr) - 1 and arr[i + 1] - arr[i] > 1:
        #         id_dict_type[count] = [id_dict_type[count][0], arr[i]]  # gaps with only one index have [gap_idx, next_gap_start_idx] but gaps with more than one index have [start_gap_idx, end_gap_idx]
        #         count += 1
        #     else:
        #         if arr[i] - arr[i - 1] > 1:
        #             id_dict_type[count] = [arr[i], arr[i]]
        #         else:
        #             id_dict_type[count] = [id_dict_type[count][0], arr[i]]
        # id_dict[checkstr] = id_dict_type

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


def synchronize_centroids_with_flux(all_time, centroid_time, all_centroids):
    """
    Synchronizes centroid cadences with flux cadences by comparing flux time vector with centroid time vector.
    Operations:
        - Fills centroid cadences with nan's where flux cadence is present and centroid cadence is not present
        - Removes centroid cadences where flux cadences not present

    :param all_time: 1D float numpy array, flux time vector
    :param centroid_time: 1D float numpy array, centroid time vector
    :param all_centroids: 1D float numpy array, centroid vector
    :return:
        dict with centroid values whose centroid cadences were matched with the flux cadences or with NaNs for those
        flux cadences that were not matched
    """

    thres = 0.005  # time cadence match threshold: 0.005 days = ~7 minutes

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

    # first sort by cadence values, then by identifier
    # for the matched cadences, the centroid cadence will come first (almost always, since due to float precision the
    # values may not be exactly the same)
    a = a[np.lexsort((a[:, 1], a[:, 0]))]

    # collect rows to delete
    # delete centroid time cadences if flux does not have that time cadence
    rows_del = []
    for i in range(len(a) - 1):
        # if the current and next cadence are a match, add the flux cadence to the deletion list
        if abs(a[i][0] - a[i + 1][0]) < thres:
            if a[i][1] == 1:  # if flux cadence comes first
                rows_del.append(i)
            else:  # if flux cadence comes second
                rows_del.append(i + 1)
        # if it is not a match and the current cadence is a centroid cadence, add it to the deletion list
        elif a[i][1] == 0:
            rows_del.append(i)

    a = np.delete(a, rows_del, axis=0)

    return {'x': a[:, 2], 'y': a[:, 3]}


def _process_tce(tce, table, all_flux, all_time, config, conf_dict, gap_ids=None):
    """Processes the light curve and returns an Example proto.

    Args:
    tce: Row of the input TCE table.
    table: pandas DataFrame, ephemeris table
    all_flux:
    all_time:
    config: Config object
    config_dict:
    gap_ids: None if not gapping

    Returns:
    A tensorflow.train.Example proto containing TCE features.
    """

    centroid_time = None  # pre-fill
    if all_flux is None:  # all_flux = None if satellite = kepler and non-whitened light curves
        all_time, all_flux, all_centroids = read_light_curve(tce, config)

        if all_flux is None and config.omit_missing:
            report_exclusion(config, tce, 'all flux import values are NaNs')
            return None
    else:  # only keep centroids as whitened source does not have centroid information
        centroid_time, __, all_centroids = read_light_curve(tce, config)

    if config.whitened:
        all_centroids['x'] = np.concatenate(all_centroids['x'])
        all_centroids['y'] = np.concatenate(all_centroids['y'])
        centroid_time = np.concatenate(centroid_time)

        all_flux = np.array(all_flux)
        all_time = np.array(all_time)

        if np.max(all_flux) == 0:
            report_exclusion(config, tce, 'all whitened flux import values are 0')
            return None

        # remove patches of 0's (whitened time series are filled with 0's instead of nan's)
        zero_gaps = get_gap_indices(all_flux, checkfuncs=[0])[0]
        # if removing different valued gaps, we would need an additional step to order the gaps
        # starting from the last gap to the first one in order to not mess with the indexing
        # for gap_n in reversed(list(range(len(zero_gaps.items())))):
        # for gap_n in range(len(zero_gaps.items()) - 1, -1, -1):
        for gap_n in range(len(zero_gaps) - 1, -1, -1):
            gap = zero_gaps[gap_n]
            gap_length = gap[-1] - gap[0] + 1
            if gap_length > 2:  # avoiding gaps smaller than 3 idxs
                all_flux = np.concatenate((all_flux[:gap[0]], all_flux[gap[-1]:]))
                all_time = np.concatenate((all_time[:gap[0]], all_time[gap[-1]:]))

        # align cadences of centroid and time
        all_centroids = synchronize_centroids_with_flux(all_time, centroid_time, all_centroids)
        # assert len(all_centroids['x']) == len(all_flux)
        if len(all_centroids['x']) != len(all_flux):
            report_exclusion(config, tce, 'after sychronizing the centroid with the flux time series, the two time '
                                          'series do not have the same length.')
            return None

        # FIXME: BUT INSIDE THE GAPPING FUNCTION WAS DOINT THIS, SO WE MIGHT AS WELL DO IT ALWAYS HERE
        #       if not config.gapped:  # WHY ONLY IN THIS CASE? SINCE THE SPLINE IS ALSO TESTED FOR GAPPING AND IT
        #       ALWAYS HAS THIS STRCTURE (LIST OF NUMPY ARRAYS)
        all_time, all_flux, all_centroids['x'], all_centroids['y'] = \
                [all_time], [all_flux], [all_centroids['x']], [all_centroids['y']]

    # oot_rms = get_centr_oot_rms(all_centroids, all_time, tce, table, config)  # centroid Out Of transit (oot) RMS

    if config.gapped:
        all_time, all_flux, all_centroids, gap_ids = \
            kepler_io.gap_other_tces(all_time, all_flux, all_centroids, tce, table, config, conf_dict)

    # Remove cadences with NaN time, flux or centroid values.
    for i, (time, flux, centr_x, centr_y) in enumerate(zip(all_time, all_flux, all_centroids['x'], all_centroids['y'])):
        finite_id_flux = np.logical_and(np.isfinite(flux), np.isfinite(time))
        finite_id_centr = np.logical_and(np.isfinite(centr_x), np.isfinite(centr_y))
        finite_id = np.logical_and(finite_id_flux, finite_id_centr)
        all_time[i] = time[finite_id]
        all_flux[i] = flux[finite_id]
        all_centroids['x'][i] = centr_x[finite_id]
        all_centroids['y'][i] = centr_y[finite_id]

    # centroid squared euclidean distance to median
    # spline
    # imputing
    time, flux, centroids = process_light_curve(all_time, all_flux, all_centroids, gap_ids, config.whitened)

    # phase-folding using median filtering and binning
    # generate views
    # add ephemeris information
    return generate_example_for_tce(time, flux, centroids, tce, config)


# if config.gapped:
#     all_time1, all_flux1, all_centroids1, gap_ids = \
#         kepler_io.gap_other_tces(all_time, all_flux, all_centroids, tce, table, config, conf_dict)
# all_time, all_flux = all_time[0], all_flux[0]
# Remove timestamps with NaN time, flux or centroid values.
# for i, (time, flux, centr_x, centr_y) in enumerate(zip(all_time1, all_flux1, all_centroids1['x'], all_centroids1['y'])):
#     finite_id_flux = np.logical_and(np.isfinite(flux), np.isfinite(time))
#     finite_id_centr = np.logical_and(np.isfinite(centr_x), np.isfinite(centr_y))
#     finite_id = np.logical_and(finite_id_flux, finite_id_centr)
#     all_time1[i] = time[finite_id]
#     all_flux1[i] = flux[finite_id]
#     all_centroids1['x'][i] = centr_x[finite_id]
#     all_centroids1['y'][i] = centr_y[finite_id]


def read_light_curve(tce, config):
    """Reads a Kepler light curve.

    Args:
      kepid: Kepler id of the target star.
      kepler_data_dir: Base directory containing Kepler data. See
        kepler_io.kepler_filenames().

    Returns:
      all_time: A list of numpy arrays; the time values of the raw light curve.
      all_flux: A list of numpy arrays corresponding to the time arrays in
          all_time.

    Raises:
      IOError: If the light curve files for this Kepler ID cannot be found.
    """
    kepler_data_dir = config.lc_data_dir

    # Read the Kepler light curve.
    file_names = kepler_io.kepler_filenames(kepler_data_dir, tce.kepid)

    if not file_names:
        if not config.omit_missing:
            raise IOError("Failed to find .fits files in {} for Kepler ID {}".format(kepler_data_dir, tce.kepid))
        else:
            report_exclusion(config, tce, 'No available lightcurve .fits files')
            return None, None, None  # why?

    return kepler_io.read_kepler_light_curve(file_names)


def patch_centroid_curve(all_centroids):
    """Connects the separated quarters of a centroid shift time series.

    :param all_centroids: x and y centroid shift time series.
    :return: corrected x and y centroid shift time series.
    """
    if isinstance(all_centroids, np.ndarray) and all_centroids.ndim == 1:
        return all_centroids
    end = len(all_centroids['x'])
    i = 1
    while i < end:
        init_x = all_centroids['x'][i][np.where(~np.isnan(all_centroids['x'][i]))[0][0]]
        end_x = all_centroids['x'][i-1][np.where(~np.isnan(all_centroids['x'][i-1]))[0][-1]]
        # med_prev_x = all_centroids['x'][i-1][np.nanmedian(all_centroids['x'][i-1])]
        init_y = all_centroids['y'][i][np.where(~np.isnan(all_centroids['y'][i]))[0][0]]
        end_y = all_centroids['y'][i-1][np.where(~np.isnan(all_centroids['y'][i-1]))[0][-1]]
        # med_prev_y = all_centroids['y'][i-1][np.nanmedian(all_centroids['y'][i-1])]
        all_centroids['x'][i] -= (init_x - end_x)
        all_centroids['y'][i] -= (init_y - end_y)
        i += 1
    return all_centroids


def process_light_curve(all_time, all_flux, all_centroids, gap_ids, whitened):
    """Removes low-frequency variability from a light curve.

    Args:
      all_time: A list of numpy arrays; the time values of the raw light curve.
      all_flux: A list of numpy arrays corresponding to the time arrays in
        all_time.

    Returns:
      time: 1D NumPy array; the time values of the light curve.
      flux: 1D NumPy array; the normalized flux values of the light curve.
    """

    # FIXME: remove this unnecessary function
    # Split on gaps.
    all_time, all_flux, all_centroids = util.split_wcentroids(all_time, all_flux, all_centroids, gap_width=0.75)

    # Concatenate the piecewise light curve and spline.
    # try:
    time = np.concatenate(all_time)
    # except ValueError:
    #     print(all_time)

    # try:
    flux = np.concatenate(all_flux)
    # except ValueError:
    #     print(all_flux)

    # Median window filtering
    bin_size = int(np.ceil(len(all_centroids['x']) / 2000))
    limit = len(all_centroids['x'])
    i = 0
    while i < limit:
        loc_median_x = np.nanmedian(all_centroids['x'][i:min(i + bin_size, limit)])
        loc_median_y = np.nanmedian(all_centroids['y'][i:min(i + bin_size, limit)])
        all_centroids['x'][i:i + bin_size] /= loc_median_x
        all_centroids['y'][i:i + bin_size] /= loc_median_y
        i += bin_size

    for dim, array in all_centroids.items():
        all_centroids[dim] = np.concatenate(array)

    # compute the median centroid across all quarters and compute the squared Euclidean distance to this value
    median_x = np.nanmedian(all_centroids['x'])
    median_y = np.nanmedian(all_centroids['y'])
    centroid_quadr = np.sqrt(np.square(all_centroids['x'] - median_x) + np.square(all_centroids['y'] - median_y))

    # spline preprocessing
    if not whitened:

        # Fit a piecewise-cubic spline with default arguments.
        spline = kepler_spline.fit_kepler_spline(all_time, all_flux, verbose=False)[0]
        spline = np.concatenate(spline)

        # In rare cases the piecewise spline contains NaNs in places the spline could
        # not be fit. We can't normalize those points if the spline isn't defined
        # there. Instead we just remove them.
        finite_i = np.isfinite(spline)
        if not np.all(finite_i):
            time = time[finite_i]
            flux = flux[finite_i]
            spline = spline[finite_i]

        # "Flatten" the light curve (remove low-frequency variability) by dividing by
        # the spline.
        flux /= spline

    # if type(flux) is not np.ndarray:  # flux and time type = Maskedcolumn if imported from .tbl files, which are whitened
    #   flux = np.array(flux)
    #   time = np.array(time)

    # add noise to the gapped TCE in place of the removed TCEs
    if gap_ids:  # if gaps need not be imputed, gap_ids = None
        # estimate noise
        flux_med = np.median(flux)
        std_rob_estm = np.median(np.abs(flux - flux_med)) * 1.4826
        # add noise
        for [time_slice, flux_slice] in gap_ids:
            imputed_flux = flux_med + np.random.normal(0, std_rob_estm, flux_slice.shape)
            time = np.append(time, time_slice.astype(time.dtype))
            flux = np.append(flux, imputed_flux.astype(flux.dtype))

    return time, flux, centroid_quadr


def phase_fold_and_sort_light_curve(time, flux, centroids, period, t0):
    """Phase folds a light curve and sorts by ascending time.

    Args:
      time: 1D NumPy array of time values.
      flux: 1D NumPy array of flux values.
      period: A positive real scalar; the period to fold over.
      t0: The center of the resulting folded vector; this value is mapped to 0.

    Returns:
      folded_time: 1D NumPy array of phase folded time values in
          [-period / 2, period / 2), where 0 corresponds to t0 in the original
          time array. Values are sorted in ascending order.
      folded_flux: 1D NumPy array. Values are the same as the original input
          array, but sorted by folded_time.
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

    :param centroids: 1D NumPy array of centroid values
    :param flux: 1D NumPy array of flux values
    :param time: 1D NumPy array of time values
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

    if np.min(time) < i - half_period:
        while i - half_period >= np.min(time):
            i -= period
            switcher *= -1

    while i - half_period <= np.amax(time):
        if switcher == 1:
            odd_indices = np.concatenate(
                (odd_indices, np.where(
                    np.logical_and(
                        time < i + half_period,
                        time >= i - half_period
                    ))),
                axis=None
            )
        else:
            even_indices = np.concatenate(
                (even_indices, np.where(
                    np.logical_and(
                        time < i + half_period,
                        time >= i - half_period
                    ))),
                axis=None
            )
        i += period
        switcher *= -1

    odd_time = np.take(time, odd_indices)
    even_time = np.take(time, even_indices)
    odd_flux = np.take(flux, odd_indices)
    even_flux = np.take(flux, even_indices)
    odd_centroids = np.take(centroids, odd_indices)
    even_centroids = np.take(centroids, even_indices)

    odd_result = np.mod(odd_time + (half_period - t0), period)
    odd_result -= half_period
    even_result = np.mod(even_time + (half_period - t0), period)
    even_result -= half_period

    sorted_i_odd = np.argsort(odd_result)
    sorted_i_even = np.argsort(even_result)
    odd_result = odd_result[sorted_i_odd]
    even_result = even_result[sorted_i_even]
    odd_flux = odd_flux[sorted_i_odd]
    even_flux = even_flux[sorted_i_even]
    odd_centroids = odd_centroids[sorted_i_odd]
    even_centroids = even_centroids[sorted_i_even]

    assert(len(odd_time) + len(even_time) - len(time) == 0 and len(set(odd_time) & set(even_time)) == 0)

    return (odd_result, odd_flux, odd_centroids), (even_result, even_flux, even_centroids)


def generate_view(time, flux, num_bins, bin_width, t_min, t_max,
                  normalize=True, centroid=False):
    """Generates a view of a phase-folded light curve using a median filter.

    Args:
      time: 1D array of time values, sorted in ascending order.
      flux: 1D array of flux values.
      num_bins: The number of intervals to divide the time axis into.
      bin_width: The width of each bin on the time axis.
      t_min: The inclusive leftmost value to consider on the time axis.
      t_max: The exclusive rightmost value to consider on the time axis.
      normalize: Whether to center the median at 0 and minimum value at -1.

    Returns:
      1D NumPy array of size num_bins containing the median flux values of
      uniformly spaced bins on the phase-folded time axis.
    """

    # get centroid time series points that are not NaN
    if centroid:
        finite_idxs = np.isfinite(flux)
        flux_allfinite = flux[finite_idxs]
        time_allfinite = time[finite_idxs]
    else:
        flux_allfinite = flux
        time_allfinite = time

    # apply median filter to the sorted time series
    view = median_filter.median_filter(time_allfinite, flux_allfinite, num_bins, bin_width, t_min, t_max)

    # normalize the phase-folded time series
    if normalize:
        # # median centering
        # view -= np.median(view)
        # # for the centroid time series
        # if centroid:  # range [new_min_val, 1], assuming max is positive, which should be since we are removing the median from a non-negative time series
        #     if np.abs(np.max(view)) == 0:
        #         return view
        #     view /= np.abs(np.max(view))
        # # for the flux time series
        # else:  # range [-1, new_max_val], assuming min is negative, if not [1, new_max_val]
        #     if np.abs(np.min(view)) == 0:
        #         return view
        #     view /= np.abs(np.min(view))
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

    # median centering
    view -= np.median(view)

    # normalization
    # for the centroid time series
    # range [new_min_val, 1], assuming max is positive, which should be since we are removing the median from a
    # non-negative time series
    if centroid:
        if np.abs(np.max(view)) == 0:
            return view
        if val is None:
            view /= np.abs(np.max(view))
        else:
            view /= val
    # for the flux time series
    # range [-1, new_max_val], assuming min is negative, if not [1, new_max_val]
    else:
        if np.abs(np.min(view)) == 0:
            return view
        if val is None:
            view /= np.abs(np.min(view))
        else:
            view /= val

    return view


def global_view(time,
                flux,
                period,
                num_bins=2001,
                bin_width_factor=1 / 2001,
                centroid=False,
                normalize=True):
    """Generates a 'global view' of a phase folded light curve.

    See Section 3.3 of Shallue & Vanderburg, 2018, The Astronomical Journal.
    http://iopscience.iop.org/article/10.3847/1538-3881/aa9e09/meta

    Args:
      time: 1D array of time values, sorted in ascending order.
      flux: 1D array of flux values.
      period: The period of the event (in days).
      num_bins: The number of intervals to divide the time axis into.
      bin_width_factor: Width of the bins, as a fraction of period.

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
        normalize=normalize)


def local_view(time,
               flux,
               period,
               duration,
               num_bins=201,
               bin_width_factor=0.16,
               num_durations=4,
               centroid=False,
               normalize=True):
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
      num_durations: The number of durations to consider on either side of 0 (the
        event is assumed to be centered at 0).

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
        normalize=normalize)


def generate_example_for_tce(time, flux, centroids, tce, config):
    """Generates a tf.train.Example representing an input TCE.

    Args:
      time: 1D NumPy array; the time values of the light curve.
      flux: 1D NumPy array; the normalized flux values of the light curve.
      tce: Dict-like object containing at least 'tce_period', 'tce_duration', and
        'tce_time0bk'. Additional items are included as features in the output.

    Returns:
      A tf.train.Example containing features 'global_view', 'local_view', and all
      values present in `tce`.
    """

    # ephemeris
    period = tce["tce_period"]  # period of the TCE
    duration = tce["tce_duration"]  # transit duration (avg?)
    t0 = tce["tce_time0bk"]  # epoch, instant of the first transit of the TCE

    # phase folding of the flux and centroid time series
    time, flux, centroid = phase_fold_and_sort_light_curve(time, flux, centroids, period, t0)
    # odd and even time series
    (odd_time, odd_flux, odd_centroid), \
        (even_time, even_flux, even_centroid) = phase_fold_and_sort_light_curve_odd_even(
                                                            time, flux, centroids, period, t0
                                                            )

    # Make output proto.
    ex = tf.train.Example()

    # Create time series features.
    try:
        glob_view = global_view(time, flux, period,
                                num_bins=config.num_bins_glob, bin_width_factor=config.bin_width_factor_glob)

        glob_view_odd = global_view(odd_time, odd_flux, period,
                                    num_bins=config.num_bins_glob, bin_width_factor=config.bin_width_factor_glob,
                                    normalize=False)

        glob_view_even = global_view(even_time, even_flux, period,
                                     num_bins=config.num_bins_glob, bin_width_factor=config.bin_width_factor_glob,
                                     normalize=False)

        val_norm = np.abs(min(np.min(glob_view_even), np.min(glob_view_odd)))

        glob_view_odd = normalize_view(glob_view_odd, val=val_norm)

        glob_view_even = normalize_view(glob_view_even, val=val_norm)

    except ValueError as e:
        report_exclusion(config, tce, 'Global view returned error', stderr=e)
        return None

    try:
        loc_view = local_view(time, flux, period, duration,
                              num_bins=config.num_bins_loc, bin_width_factor=config.bin_width_factor_loc)

        loc_view_odd = local_view(odd_time, odd_flux, period, duration,
                                    num_bins=config.num_bins_loc, bin_width_factor=config.bin_width_factor_loc,
                                    normalize=False)

        loc_view_even = local_view(even_time, even_flux, period, duration,
                                     num_bins=config.num_bins_loc, bin_width_factor=config.bin_width_factor_loc,
                                     normalize=False)

        val_norm = np.abs(min(np.min(loc_view_even), np.min(loc_view_odd)))

        loc_view_odd = normalize_view(loc_view_odd, val=val_norm)

        loc_view_even = normalize_view(loc_view_even, val=val_norm)

    except ValueError as e:
        report_exclusion(config, tce, 'Local view returned error', stderr=e)
        return None

    try:
        glob_view_centr = global_view(time, centroid, period, centroid=True,
                                      num_bins=config.num_bins_glob, bin_width_factor=config.bin_width_factor_glob)

        glob_view_odd_centr = global_view(odd_time, odd_centroid, period,
                                    num_bins=config.num_bins_glob, bin_width_factor=config.bin_width_factor_glob,
                                    normalize=False)

        glob_view_even_centr = global_view(even_time, even_centroid, period,
                                     num_bins=config.num_bins_glob, bin_width_factor=config.bin_width_factor_glob,
                                     normalize=False)

        val_norm = np.abs(max(np.max(glob_view_even_centr), np.max(glob_view_odd_centr)))

        glob_view_odd_centr = normalize_view(glob_view_odd_centr, val=val_norm, centroid=True)

        glob_view_even_centr = normalize_view(glob_view_even_centr, val=val_norm, centroid=True)

    except ValueError as e:
        report_exclusion(config, tce, 'Global centroid view returned error', stderr=e)
        return None

    try:
        loc_view_centr = local_view(time, centroid, period, duration, centroid=True,
                                    num_bins=config.num_bins_loc, bin_width_factor=config.bin_width_factor_loc)

        loc_view_odd_centr = local_view(odd_time, odd_centroid, period, duration,
                                   num_bins=config.num_bins_loc, bin_width_factor=config.bin_width_factor_loc,
                                   normalize=False)

        loc_view_even_centr = local_view(even_time, even_centroid, period, duration,
                                    num_bins=config.num_bins_loc, bin_width_factor=config.bin_width_factor_loc,
                                    normalize=False)

        val_norm = np.abs(min(np.min(loc_view_even_centr), np.min(loc_view_odd_centr)))

        loc_view_odd_centr = normalize_view(loc_view_odd_centr, val=val_norm, centroid=True)

        loc_view_even_centr = normalize_view(loc_view_even_centr, val=val_norm, centroid=True)

    except ValueError as e:
        report_exclusion(config, tce, 'Local centroid view returned error', stderr=e)
        return None

    # check if time series are not all NaNs
    if True not in np.isfinite(glob_view):
        report_exclusion(config, tce, 'Global view is all NaNs')
        return None

    if True not in np.isfinite(loc_view):
        report_exclusion(config, tce, 'Local view is all NaNs')
        return None

    if True not in np.isfinite(glob_view_centr):
        report_exclusion(config, tce, 'Global centroid view is all NaNs')
        return None

    if True not in np.isfinite(loc_view_centr):
        report_exclusion(config, tce, 'Local centroid view is all NaNs')
        return None

    # add features to the new example
    example_util.set_float_feature(ex, "global_view", glob_view)
    example_util.set_float_feature(ex, "local_view", loc_view)
    example_util.set_float_feature(ex, "global_view_centr", glob_view_centr)
    example_util.set_float_feature(ex, "local_view_centr", loc_view_centr)

    example_util.set_float_feature(ex, "global_view_odd", glob_view_odd)
    example_util.set_float_feature(ex, "local_view_odd", loc_view_odd)
    example_util.set_float_feature(ex, "global_view_odd_centr", glob_view_odd_centr)
    example_util.set_float_feature(ex, "local_view_odd_centr", loc_view_odd_centr)

    example_util.set_float_feature(ex, "global_view_even", glob_view_even)
    example_util.set_float_feature(ex, "local_view_even", loc_view_even)
    example_util.set_float_feature(ex, "global_view_even_centr", glob_view_even_centr)
    example_util.set_float_feature(ex, "local_view_even_centr", loc_view_even_centr)

    # Set other features in `tce`.
    for name, value in tce.items():
        example_util.set_feature(ex, name, [value])

    return ex
