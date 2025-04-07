""" General utility functions for preprocessing. """

# 3rd party
import numpy as np
from astropy.stats import mad_std

# local
from src_preprocessing.lc_preprocessing.utils_preprocessing_io import report_exclusion


def get_out_of_transit_idxs_loc(num_bins_loc, num_transit_durations):
    """ Get indices of out-of-transit cadences for the local views.

    :param num_bins_loc: int, number of bins for the local views
    :param num_transit_durations: int, number of transit durations in the local views
    :return:
        idxs_nontransitcadences_loc: list with out-of-transit indices for the local views
    """

    # get out-of-transit indices for local views
    transit_duration_bins_loc = num_bins_loc / num_transit_durations  # number of bins per transit duration
    nontransitcadences_loc = np.array([True] * num_bins_loc)
    # set to False the indices outside of the transit which is in the center of the view
    nontransitcadences_loc[np.arange(int(np.floor((num_bins_loc - transit_duration_bins_loc) / 2)),
                                     int(np.ceil((num_bins_loc + transit_duration_bins_loc) / 2)))] = False
    idxs_nontransitcadences_loc = np.where(nontransitcadences_loc)

    return idxs_nontransitcadences_loc


def get_out_of_transit_idxs_glob(num_bins_glob, transit_duration, orbital_period):
    """ Get indices of out-of-transit cadences for the global views.

    :param num_bins_glob: int, number of bins for the global views
    :param transit_duration: float, transit duration
    :param orbital_period: float, orbital period
    :return:
        idxs_nontransitcadences_glob: list with out-of-transit indices for the global views
    """

    # get out-of-transit indices for global views
    frac_durper = transit_duration / orbital_period  # ratio transit duration to orbital period
    nontransitcadences_glob = np.array([True] * num_bins_glob)
    # set To False the indices outside of the transit which is in the center of the view
    nontransitcadences_glob[np.arange(int(num_bins_glob / 2 * (1 - frac_durper)),
                                      int(num_bins_glob / 2 * (1 + frac_durper)))] = False
    idxs_nontransitcadences_glob = np.where(nontransitcadences_glob)

    return idxs_nontransitcadences_glob


def _count_num_bins(x, x_max, x_min, bin_width, num_bins):
    """ Count number of points per bin.

    x: 1D array of x-coordinates sorted in ascending order. Must have at least 2
      elements, and all elements cannot be the same value.
    bin_width: The width of each bin on the x-axis. Must be positive, and less
      than x_max - x_min. Defaults to (x_max - x_min) / num_bins.
    x_min: The inclusive leftmost value to consider on the x-axis. Must be less
      than or equal to the largest value of x. Defaults to min(x).
    x_max: The exclusive rightmost value to consider on the x-axis. Must be
      greater than x_min. Defaults to max(x).
    num_bins: The number of intervals to divide the x-axis into. Must be at
      least 2.
    :return:
        Numpy array, number of points per each bin
    """

    pts_per_bin = np.zeros(num_bins, dtype='uint64')
    # bin_time_idx = []

    if num_bins < 2:
        raise ValueError("num_bins must be at least 2. Got: {}".format(num_bins))

    # Validate the lengths of x.
    x_len = len(x)
    if x_len < 2:
        # raise ValueError("len(x) must be at least 2. Got: {}".format(x_len))
        pts_per_bin[0] = 1
        # bin_time_idx.append(np.arange(x[0], x[0]))
        return pts_per_bin  # , bin_time_idx

    # Validate x_min and x_max.
    x_min = x_min if x_min is not None else x[0]
    x_max = x_max if x_max is not None else x[-1]
    if x_min >= x_max:
        raise ValueError("x_min (got: {}) must be less than x_max (got: {})".format(
            x_min, x_max))
    if x_min > x[-1]:
        # raise ValueError(
        #     "x_min (got: {}) must be less than or equal to the largest value of x "
        #     "(got: {})".format(x_min, x[-1]))
        return pts_per_bin  # , bin_time_idx

    # Validate bin_width.
    bin_width = bin_width if bin_width is not None else (x_max - x_min) / num_bins
    if bin_width <= 0:
        raise ValueError("bin_width must be positive. Got: {}".format(bin_width))
    if bin_width >= x_max - x_min:
        raise ValueError(
            "bin_width (got: {}) must be less than x_max - x_min (got: {})".format(
                bin_width, x_max - x_min))

    bin_spacing = (x_max - x_min - bin_width) / (num_bins - 1)

    # Find the first element of x >= x_min. This loop is guaranteed to produce
    # a valid index because we know that x_min <= x[-1].
    x_start = 0
    while x[x_start] < x_min:
        x_start += 1

    # The bin at index i is the median of all elements y[j] such that
    # bin_min <= x[j] < bin_max, where bin_min and bin_max are the endpoints of
    # bin i.
    bin_min = x_min  # Left endpoint of the current bin.
    bin_max = x_min + bin_width  # Right endpoint of the current bin.
    j_start = x_start  # Inclusive left index of the current bin.
    j_end = x_start  # Exclusive end index of the current bin.

    for i in range(num_bins):
        # Move j_start to the first index of x >= bin_min.
        while j_start < x_len and x[j_start] < bin_min:
            j_start += 1

        # Move j_end to the first index of x >= bin_max (exclusive end index).
        while j_end < x_len and x[j_end] < bin_max:
            j_end += 1

        pts_per_bin[i] = j_end - j_start
        # bin_time_idx.append(np.arange(j_start, j_end))

        # Advance the bin.
        bin_min += bin_spacing
        bin_max += bin_spacing

    return pts_per_bin  # , bin_time_idx


def count_transits(time, period, epoch, duration, num_cadences_min=1):
    """ Count number of transits. A transit is considered valid if at least num_cadences_min exist.

    :param time: NumPy array, timestamp array
    :param period: float, orbital period
    :param epoch: float, epoch
    :param duration:  float, transit duration
    :param num_cadences_min: int, minimum number of candences to consider the transit valid
    :return:
        int, number of valid transits
    """

    if len(time) == 0:
        return 0

    # find first transit midpoint
    mid_transit_time = epoch + period * np.ceil((time[0] - epoch) / period)
    mid_transit_times = [mid_transit_time]

    while True:
        mid_transit_time += period
        mid_transit_times.append(mid_transit_time)
        if mid_transit_time > time[-1]:
            break

    return np.sum([len(np.where((time >= mid_transit_time - duration / 2) &
                                (time <= mid_transit_time + duration / 2))[0]) >= num_cadences_min
                   for mid_transit_time in mid_transit_times])


def min_max_normalization(arr, max_val, min_val):
    """ Min-max normalization.

    :param arr: array
    :param max_val: float, max val
    :param min_val: float, min val
    :return:
        normalized array
    """

    return (arr - min_val) / (max_val - min_val)


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


def remove_non_finite_values(arrs):
    """ Remove cadences with non-finite values (NaN or infinite) for timestamps or timeseries in each array. Some these
    values come from: gapping the timeseries, missing time values from the FITS files.

    :param arrs: list, arrays to be checked for non-finite values.
    :return:
        list, arrays without non-finite values.
    """

    num_arrs = len(arrs)
    num_sub_arrs = len(arrs[0])

    for sub_arr_i in range(num_sub_arrs):  # assuming same size for all arrays
        finite_idxs = []

        for arr in arrs:
            finite_idxs.append(np.isfinite(arr[sub_arr_i]))

        finite_idxs = np.logical_and.reduce(finite_idxs, dtype='bool')

        for arr_i in range(num_arrs):
            arrs[arr_i][sub_arr_i] = arrs[arr_i][sub_arr_i][finite_idxs]

    return arrs


def remove_outliers(ts, sigma, fill=False, outlier_type='upper', seed=None):
    """ Remove outliers (option for positive, negative or both types of outliers) in the flux time series using a
    threshold in the standard deviation. Option to fill outlier cadences using Gaussian noise with global statistics
    of the time series. If not, those will be filled out with NaNs.

    :param ts: NumPy array or list of NumPy arrays, time series
    :param sigma: float, sigma factor
    :param fill: bool, if True fills outliers
    :param outlier_type: str, type of outlier to be removed. `upper` for positive outliers, `lower` for negative
    outliers, `both` for both types
    :param seed: rng seed
    :return:
        time: NumPy array, time stamps for the new time series
        ts: NumPy array or list of NumPy array is `ts` is a list of NumPy arrays, time series without outliers
        idxs_out: NumPy array or list of NumPy array is `ts` is a list of NumPy arrays, outlier indices are set to True
    """

    if outlier_type not in ['upper', 'lower', 'both']:
        raise TypeError(f'Outlier type {outlier_type} not recognized. Choose between `upper`, `lower`, or `both`.')

    ts_arr_list = False
    if isinstance(ts, list):  # time series is a list of NumPy arrays with different dimensions
        flattened_arr = [x for xs in ts for x in xs]
        rob_std = mad_std(flattened_arr, ignore_nan=True)
        mu_val = np.nanmedian(flattened_arr)
        ts_arr_list = True

    else:  # time series is a NumPy array
        rob_std = mad_std(ts, ignore_nan=True)
        mu_val = np.nanmedian(ts)

        ts = [ts]

    idxs_out = []
    for np_ts_arr in ts:

        if outlier_type == 'upper':
            idxs_out_arr = np_ts_arr > mu_val + sigma * rob_std
        elif outlier_type == 'lower':
            idxs_out_arr = np_ts_arr < mu_val - sigma * rob_std
        elif outlier_type == 'both':
            idxs_out_arr = np.abs(np_ts_arr) > mu_val + sigma * rob_std

        if fill:
            # fill with Gaussian noise with global time series statistics
            rng = np.random.default_rng(seed)
            np_ts_arr[idxs_out_arr] = rng.normal(mu_val, rob_std, idxs_out_arr.sum())
        else:  # set to NaN
            np_ts_arr[idxs_out_arr] = np.nan

        idxs_out.append(idxs_out_arr)

    if not ts_arr_list:
        ts = ts[0]
        idxs_out = idxs_out[0]

    return ts, idxs_out


def check_inputs_generate_example(data, tce, config):
    """ Check inputs to the function that phase folds and bins the preprocessed timeseries to create the input views.

    :param data: dict, containing the preprocessed timeseries.
    :param tce: pandas Series, TCE ephemeris, parameters and diagnostics from DV.
    :param config: dict, preprocessing parameters
    :return:
        data, dict containing the preprocessed timeseries after checking inputs and adjusting them accordingly.
    """

    if data['centroid_dist'] is None:  # set it to empty list
        data['centroid_dist'] = []

    # set centroid distance to zero (meaning on-target)
    if len(data['centroid_dist']) == 0 or np.isnan(data['centroid_dist']).all():
        report_exclusion(
            f'No data points available. Setting centroid offset time series to zero (target star position).',
            config['exclusion_logs_dir'] / f'exclusions-{tce["uid"]}.txt'
        )

        data['centroid_dist_time'] = np.array(data['flux_time'])
        data['centroid_dist'] = np.zeros(len(data['flux_time']))

    # fill with Gaussian noise using time series statistics
    elif 'Less than 0.5% cadences are valid for centroid data.' in data['errors']:
        report_exclusion(
            f'Less than 0.5% cadences are valid for centroid data. Setting centroid '
            f'offset time series to Gaussian noise using statistics from this time series.',
            config['exclusion_logs_dir'] / f'exclusions-{tce["uid"]}.txt',
                         )

        rob_std = mad_std(data['centroid_dist'], ignore_nan=True)
        med = np.nanmedian(data['centroid_dist'])
        rng = np.random.default_rng(seed=config['random_seed'])
        data['centroid_dist'] = rng.normal(med, rob_std, data['flux'].shape)
        data['centroid_dist_time'] = np.array(data['flux_time'])

    return data


def check_inputs(data):
    """ Check data coming from the lc FITS files.

    :param data: dict, data read from the lc FITS files
    :return:
        err, list with errors for the data read
    """

    errs = []

    # no centroid data available from the FITS files
    # assuming that if there is not 'x' component, there is also not 'y' component
    # assuming there is other data; otherwise, it wouldn't get here
    # setting centroid to target star position if the number of valid cadences is less than 100
    # 0.5% observation time in one TESS sector
    # setting centroid pixel to zero
    n_valid_cad = np.isfinite(np.concatenate(data['all_centroids']['x'])).sum()
    n_cad = len(np.concatenate(data['all_centroids']['x']))
    if n_valid_cad / n_cad < 0.005:
        errs.append('Less than 0.5% cadences are valid for centroid data.')

    return errs
