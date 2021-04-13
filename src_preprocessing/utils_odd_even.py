""" Utility functions used to preprocess odd and even data. """

# 3rd party
import numpy as np
from astropy import stats

# local
from src_preprocessing.utils_preprocessing import _count_num_bins
from src_preprocessing.light_curve import util
from src_preprocessing.utils_imputing import impute_binned_ts


def phase_fold_and_sort_light_curve_odd_even(time, timeseries, period, t0, augmentation=False):
    """ Creates separate phase-folded time vectors for odd and even periods.

    :param time: 1D NumPy array of time values
    :param timeseries: 1D NumPy array of time series values
    :param period: A positive real scalar; the period to fold over.
    :param t0: The center of the resulting folded vector; this value is mapped to 0.
    :return:
        Two tuples of size three, each of which contains odd or even time values, flux values, and centroid values.
        time: 1D NumPy array of phase folded time values in [-period / 2, period / 2), where 0 corresponds to t0
            in the original time array. Values are sorted in ascending order.
        folded_timeseries: 1D NumPy array. Values are the same as the original input array, but sorted by folded_time.
        num_transits_even: int, number of even transits in the time series.
        num_transits_odd: int, number of odd transits in the time series.
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
    odd_timeseries = np.take(timeseries, odd_indices)
    even_timeseries = np.take(timeseries, even_indices)

    # Phase fold time.
    if not augmentation:
        odd_time = util.phase_fold_time(odd_time, period, t0)
        even_time = util.phase_fold_time(even_time, period, t0)
    else:
        odd_time, odd_sampled_idxs, _ = util.phase_fold_time_aug(odd_time, period, t0)
        even_time, even_sampled_idxs, _ = util.phase_fold_time_aug(even_time, period, t0)
        odd_timeseries = odd_timeseries[odd_sampled_idxs]
        even_timeseries = even_timeseries[even_sampled_idxs]

    # count number of transits in the phase domain
    if len(odd_time) == 0:
        num_transits_odd = 0
    else:
        num_transits_odd = np.sum(np.diff(odd_time) < 0) + 1
    if len(even_time) == 0:
        num_transits_even = 0
    else:
        num_transits_even = np.sum(np.diff(even_time) < 0) + 1

    # Sort by ascending time.
    sorted_i_odd = np.argsort(odd_time)
    sorted_i_even = np.argsort(even_time)

    odd_result = odd_time[sorted_i_odd]
    even_result = even_time[sorted_i_even]
    odd_timeseries = odd_timeseries[sorted_i_odd]
    even_timeseries = even_timeseries[sorted_i_even]

    # assert(len(odd_time) + len(even_time) - len(time) == 0 and len(set(odd_time) & set(even_time)) == 0)

    return (odd_result, odd_timeseries, num_transits_odd), (even_result, even_timeseries, num_transits_even)


def compute_oot_it_var_oddeven(odd_time, odd_flux, even_time, even_flux, t_min_local, t_max_local):
    """ Computes the variability out-of-transit and in-transit for the odd and even phase folded time series.

    :param odd_time: NumPy array, odd flux time series phase
    :param odd_flux: NumPy array, phase folded odd flux time series
    :param even_time: NumPy array, even flux time series phase
    :param even_flux: NumPy array, phase folded even flux time series
    :param t_min_local: float, minimum in-transit phase
    :param t_max_local: float, maximum in-transit phase
    :return:
        sigma_oot_odd: float, out-of-transit standard deviation of the odd phase folded time series
        sigma_it_odd: float, in-transit standard deviation of the odd phase folded time series
        sigma_oot_even: float, out-of-transit standard deviation of the even phase folded time series
        sigma_it_even: float, in-transit standard deviation of the even phase folded time series
    """

    sigma_oot_odd = stats.mad_std(odd_flux[(odd_time < t_min_local) | (odd_time > t_max_local)], ignore_nan=True)
    sigma_it_odd = stats.mad_std(odd_flux[(odd_time >= t_min_local) & (odd_time <= t_max_local)], ignore_nan=True)

    sigma_oot_even = stats.mad_std(even_flux[(even_time < t_min_local) | (even_time > t_max_local)], ignore_nan=True)
    sigma_it_even = stats.mad_std(even_flux[(even_time >= t_min_local) & (even_time <= t_max_local)], ignore_nan=True)

    return sigma_oot_odd, sigma_it_odd, sigma_oot_even, sigma_it_even


def check_odd_even_views(odd_view, even_view, odd_view_var=None, even_view_var=None):
    """  Check if odd and even views are flat. If one is flat, but the other one is not, replace the former by the
    latter. A view is assumed flat if it is all a vector of ones (due to previous preprocessing, that is where the
    phase folded time series are centered on.

    :param odd_view: Numpy array, phase folded and binned odd flux view
    :param even_view: Numpy array, phase folded and binned even flux view
    :param odd_view_var: Numpy array, phase folded and binned odd flux view SE of the mean
    :param even_view_var: Numpy array, phase folded and binned odd flux view SE of the mean
    :return:
        same views, with possible replacement for odd/even.
    """

    odd_even_flag = 'ok'

    if np.all(odd_view == 1) and np.any(even_view != 1):
        odd_view = np.array(even_view)
        if odd_view_var is not None and even_view_var is not None:
            odd_view_var = np.array(even_view_var)
        odd_even_flag = 'odd_flat'

    elif np.any(odd_view != 1) and np.all(even_view == 1):
        even_view = np.array(odd_view)
        if odd_view_var is not None and even_view_var is not None:
            even_view_var = np.array(odd_view_var)
        odd_even_flag = 'even_flat'

    if odd_view_var is not None and even_view_var is not None:
        return odd_view, even_view, odd_view_var, even_view_var, odd_even_flag
    else:
        return odd_view, even_view, odd_even_flag


def check_odd_even_phasefolded(odd_time, odd_flux, num_transits_odd, even_time, even_flux, num_transits_even):
    """ Check number of transits in the odd and even phase-folded time series. If there are no transits for one of them,
    but there is for the other one, replace the former by the latter

    :param odd_time: NumPy array, phase for the odd flux time series
    :param odd_flux: NumPy array, phase folded odd flux time series
    :param num_transits_odd: int, number of odd transits
    :param even_time: NumPy array, phase for the even flux time series
    :param even_flux: NumPy array, phase folded even flux time series
    :param num_transits_even: int, number of even transits
    :return:
        same variables as arguments, with possible replacement for odd/even
    """

    odd_even_flag = 'ok'

    if num_transits_odd == 0:
        odd_time, odd_flux, num_transits_odd = np.array(even_time), np.array(even_flux), np.array(num_transits_even)
        odd_even_flag = 'no_odd_time'

    if num_transits_odd == 0:
        even_time, even_flux, num_transits_even = np.array(odd_time), np.array(odd_flux), np.array(num_transits_odd)
        odd_even_flag = 'no_even_time'

    return (odd_time, odd_flux, num_transits_odd), (even_time, even_flux, num_transits_even), odd_even_flag


def check_odd_even_phasefolded_local(odd_time, odd_flux, num_transits_odd, even_time, even_flux, num_transits_even,
                                     num_bins_local, bin_width_local, t_min_local, t_max_local):
    """ Check odd and even number of valid in-transit bins in the phase folded time series.

    :param odd_time: NumPy array, odd flux time series phase
    :param odd_flux: NumPy array, phase folded odd flux time series
    :param num_transits_odd: int, number of odd transits
    :param even_time: NumPy array, even flux time series phase
    :param even_flux: NumPy array, phase folded even flux time series
    :param num_transits_even: int, number of even transits
    :param num_bins_local: int, number of in-transit bins
    :param bin_width_local: float, bin width
    :param t_min_local: float, minimum in-transit phase
    :param t_max_local: float, maximum in-transit phase
    :return:
        odd_time: NumPy array, odd flux time series phase
        odd_flux: NumPy array, phase folded odd flux time series
        num_transits_odd: int, number of odd transits
        even_time: NumPy array, even flux time series phase
        even_flux: NumPy array, phase folded even flux time series
        num_transits_even: int, number of even transits
        odd_even_flag: bool, odd and even flag
    """

    odd_even_flag, odd_flag, even_flag = 'ok', 'ok', 'ok'

    pts_per_bin_odd, _ = _count_num_bins(odd_time, t_max_local, t_min_local, bin_width_local, num_bins_local)
    num_bins_odd = len(pts_per_bin_odd[pts_per_bin_odd > 0])
    if len(pts_per_bin_odd) - num_bins_odd > 1 or pts_per_bin_odd[len(pts_per_bin_odd) // 2] == 0:
        odd_flag = 'replace'

    pts_per_bin_even, _ = _count_num_bins(even_time, t_max_local, t_min_local, bin_width_local, num_bins_local)
    num_bins_even = len(pts_per_bin_even[pts_per_bin_even > 0])
    if len(pts_per_bin_even) - num_bins_even > 1 or pts_per_bin_even[len(pts_per_bin_even) // 2] == 0:
        even_flag = 'replace'

    if odd_flag == 'replace' and even_flag == 'replace':
        odd_even_flag = f'odd(even): {num_bins_odd}({num_bins_even}) bins'
        if num_bins_odd >= num_bins_even:
            even_time, even_flux, num_transits_even = np.array(odd_time), np.array(odd_flux), num_transits_odd
        else:
            odd_time, odd_flux, num_transits_odd = np.array(even_time), np.array(even_flux), num_transits_even

    elif odd_flag == 'replace' and even_flag == 'ok':
        odd_time, odd_flux, num_transits_odd = np.array(even_time), np.array(even_flux), num_transits_even
        odd_even_flag = f'odd: {num_bins_odd} bins'
    elif odd_flag == 'replace' and even_flag == 'ok':
        even_time, even_flux, num_transits_even = np.array(odd_time), np.array(odd_flux), num_transits_odd
        odd_even_flag = f'even: {num_bins_even} bins'
    elif num_bins_even < num_bins_local and num_bins_odd < num_bins_local:
        if num_bins_odd >= num_bins_even:
            even_time, even_flux, num_transits_even = np.array(odd_time), np.array(odd_flux), num_transits_odd
        else:
            odd_time, odd_flux, num_transits_odd = np.array(even_time), np.array(even_flux), num_transits_even
        odd_even_flag = f'odd(even): {num_bins_odd}({num_bins_even}) bins'

    return odd_time, odd_flux, num_transits_odd, even_time, even_flux, num_transits_even, odd_even_flag


def impute_odd_even_views(loc_flux_odd_view, loc_flux_odd_view_var, bin_counts_odd, loc_flux_even_view,
                          loc_flux_even_view_var, bin_counts_even, inds_oot):
    """ Replace empty bins in odd/even view with the corresponding bin in the other view. If they are both missing,
    replace using the out-of-transit median value of the view that has less empty bins.

    :param loc_flux_odd_view:
    :param loc_flux_odd_view_var:
    :param bin_counts_odd:
    :param loc_flux_even_view:
    :param loc_flux_even_view_var:
    :param bin_counts_even:
    :param inds_oot:
    :return:
    """

    # get encoding for which view has the missing bin
    bin_counts_odd_nonzero = bin_counts_odd > 0
    bin_counts_even_nonzero = bin_counts_even > 0
    bin_count_zero_enc = 2 * bin_counts_odd_nonzero.astype('int') - bin_counts_even_nonzero.astype('int')

    odd_to_even_bins = bin_count_zero_enc == -1
    odd_to_even_bins[inds_oot] = False  # mask out-of-transit bins
    even_to_odd_bins = bin_count_zero_enc == 2
    even_to_odd_bins[inds_oot] = False
    odd_even_miss_bins = bin_count_zero_enc == 0
    odd_even_miss_bins[inds_oot] = False

    bins_repl = {'odd_to_even': odd_to_even_bins,
                 'even_to_odd': even_to_odd_bins,
                 'odd_even_miss': odd_even_miss_bins}

    # replace empty bins in odd view by those in even view
    loc_flux_odd_view[odd_to_even_bins] = loc_flux_even_view[odd_to_even_bins]
    loc_flux_odd_view_var[odd_to_even_bins] = loc_flux_even_view_var[odd_to_even_bins]

    # replace empty bins in even view by those in odd view
    loc_flux_even_view[even_to_odd_bins] = loc_flux_odd_view[even_to_odd_bins]
    loc_flux_even_view_var[even_to_odd_bins] = loc_flux_odd_view_var[even_to_odd_bins]

    # # replace empty bins in both views by median value of view with less empty bins
    # if np.sum(bin_counts_even_nonzero) >= np.sum(bin_counts_odd_nonzero):
    #     fill_value_view, fill_value_view_var = np.median(loc_flux_even_view[inds_oot]), \
    #                                            np.median(loc_flux_even_view_var[inds_oot])
    # else:
    #     fill_value_view, fill_value_view_var = np.median(loc_flux_odd_view[inds_oot]), \
    #                                            np.median(loc_flux_odd_view_var[inds_oot])
    #
    # loc_flux_odd_view[odd_even_miss_bins] = fill_value_view
    # loc_flux_odd_view_var[odd_even_miss_bins] = fill_value_view_var
    # loc_flux_even_view[odd_even_miss_bins] = fill_value_view
    # loc_flux_even_view_var[odd_even_miss_bins] = fill_value_view_var

    # bins_repl['odd_even_miss_fill_value'] = fill_value_view

    odd_even_flag = f'replaced bins {np.sum(odd_to_even_bins)} (odd) {np.sum(even_to_odd_bins)} (even) '

    return loc_flux_odd_view, loc_flux_odd_view_var, loc_flux_even_view, loc_flux_even_view_var, odd_even_flag, bins_repl


def odd_even_binning(x, y, num_bins, bin_width=None, x_min=None, x_max=None):
    """ Binning for odd and even phase folded time series.

  The interval [x_min, x_max) is divided into num_bins uniformly spaced
  intervals of width bin_width. The value computed for each bin is the median
  of all y-values whose corresponding x-value is in the interval.

  NOTE: x must be sorted in ascending order or the results will be incorrect.

  Args:
    x: 1D array of x-coordinates sorted in ascending order. Must have at least 2
      elements, and all elements cannot be the same value.
    y: 1D array of y-coordinates with the same size as x.
    num_bins: The number of intervals to divide the x-axis into. Must be at
      least 2.
    bin_width: The width of each bin on the x-axis. Must be positive, and less
      than x_max - x_min. Defaults to (x_max - x_min) / num_bins.
    x_min: The inclusive leftmost value to consider on the x-axis. Must be less
      than or equal to the largest value of x. Defaults to min(x).
    x_max: The exclusive rightmost value to consider on the x-axis. Must be
      greater than x_min. Defaults to max(x).

  Returns:
    1D NumPy array of size num_bins containing the median y-values of uniformly
    spaced bins on the x-axis.
    1D Numpy array of size num_bins containing the the timestamp of the bin.
    1D NumPy array of size num_bins containing the MAD std y-values of uniformly
    spaced bins on the x-axis.
    1D NumPy array of size num_bins containing the number of y-values in each one of the uniformly
    spaced bins on the x-axis.

  Raises:
    ValueError: If an argument has an inappropriate value.
  """

    if num_bins < 2:
        raise ValueError("num_bins must be at least 2. Got: {}".format(num_bins))

    # Validate the lengths of x and y.
    x_len = len(x)
    if x_len < 2:
        raise ValueError("len(x) must be at least 2. Got: {}".format(x_len))
    if x_len != len(y):
        raise ValueError("len(x) (got: {}) must equal len(y) (got: {})".format(
            x_len, len(y)))

    # Validate x_min and x_max.
    x_min = x_min if x_min is not None else x[0]
    x_max = x_max if x_max is not None else x[-1]
    if x_min >= x_max:
        raise ValueError("x_min (got: {}) must be less than x_max (got: {})".format(
            x_min, x_max))
    if x_min > x[-1]:
        raise ValueError(
            "x_min (got: {}) must be less than or equal to the largest value of x "
            "(got: {})".format(x_min, x[-1]))

    # Validate bin_width.
    bin_width = bin_width if bin_width is not None else (x_max - x_min) / num_bins
    if bin_width <= 0:
        raise ValueError("bin_width must be positive. Got: {}".format(bin_width))
    if bin_width >= x_max - x_min:
        raise ValueError(
            "bin_width (got: {}) must be less than x_max - x_min (got: {})".format(
                bin_width, x_max - x_min))

    bin_spacing = (x_max - x_min - bin_width) / (num_bins - 1)

    # Bins with no y-values will fall back to the global median.
    # result = np.repeat(np.median(y), num_bins)
    result = np.nan * np.ones(num_bins, dtype='float')
    result_var = np.zeros(num_bins)
    result_time = np.zeros(num_bins)
    bin_counts = np.zeros(num_bins, dtype='uint64')
    bin_values = [np.zeros(int(np.ceil(bin_width)))] * num_bins

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

        if j_end > j_start:
            # Compute and insert the median bin value.
            result[i] = np.median(y[j_start:j_end])
            # result_var[i] = np.std(y[j_start:j_end], ddof=1)
            result_var[i] = stats.mad_std(y[j_start:j_end])
            # _, result[i], result_var[i] = stats.sigma_clipped_stats(y[j_start:j_end], sigma=2, maxiters=10)

        bin_counts[i] = j_end - j_start
        bin_values[i] = y[j_start:j_end]

        result_time[i] = bin_min + bin_width / 2

        # Advance the bin.
        bin_min += bin_spacing
        bin_max += bin_spacing

    return result, result_time, result_var, bin_counts, bin_values


def generate_view(time, flux, num_bins, bin_width, t_min, t_max):
    """ Generates a view of a phase-folded light curve using a median filter.

    Args:
      time: 1D array of time values, sorted in ascending order.
      flux: 1D array of flux/centroid values.
      num_bins: The number of intervals to divide the time axis into.
      bin_width: The width of each bin on the time axis.
      t_min: The inclusive leftmost value to consider on the time axis.
      t_max: The exclusive rightmost value to consider on the time axis.

    Returns:
      1D NumPy array of size num_bins containing the median flux values of
      uniformly spaced bins on the phase-folded time axis.
    """

    # binning using median
    view, time_bins, view_var, bin_counts, bin_values = odd_even_binning(time, flux, num_bins, bin_width, t_min, t_max)

    return view, time_bins, view_var, bin_counts, bin_values


def compute_oot_it_var_oddeven_repl(bin_values_odd, bin_values_even, bins_repl, inds_oot):
    """ Computes the variability out-of-transit and in-transit for the odd and even phase folded time series taking
    into account bin replacement.

    :param bin_values_odd:
    :param bin_values_even:
    :param bins_repl:
    :param inds_oot:
    :return:
        sigma_oot_odd: float, out-of-transit standard deviation of the odd phase folded time series
        sigma_it_odd: float, in-transit standard deviation of the odd phase folded time series
        sigma_oot_even: float, out-of-transit standard deviation of the even phase folded time series
        sigma_it_even: float, in-transit standard deviation of the even phase folded time series
    """

    # replace bins for odd
    for bin_i in np.where(bins_repl['odd_to_even'])[0]:
        bin_values_odd[bin_i] = bin_values_even[bin_i]

    for bin_i in np.where(bins_repl['even_to_odd'])[0]:
        bin_values_even[bin_i] = bin_values_odd[bin_i]

    # for bin_i in np.where(bins_repl['odd_even_miss'])[0]:
    #     bin_values_odd[bin_i] = [bins_repl['odd_even_miss_fill_value']]
    #     bin_values_even[bin_i] = [bins_repl['odd_even_miss_fill_value']]

    # get in-transit bin indices
    inds_it = np.setdiff1d(np.arange(len(bin_values_odd)), inds_oot)

    sigma_oot_odd = stats.mad_std(np.concatenate([bin_values_odd[ind_oot] for ind_oot in inds_oot]), ignore_nan=True)
    sigma_it_odd = stats.mad_std(np.concatenate([bin_values_odd[ind_it] for ind_it in inds_it]), ignore_nan=True)

    sigma_oot_even = stats.mad_std(np.concatenate([bin_values_even[ind_oot] for ind_oot in inds_oot]), ignore_nan=True)
    sigma_it_even = stats.mad_std(np.concatenate([bin_values_even[ind_it] for ind_it in inds_it]), ignore_nan=True)

    return sigma_oot_odd, sigma_it_odd, sigma_oot_even, sigma_it_even


def create_odd_even_views(odd_time, odd_flux, even_time, even_flux, num_tr_odd, num_tr_even, tce, config):
    """ Create odd and even views and related data.

    :param odd_time:
    :param odd_flux:
    :param even_time:
    :param even_flux:
    :param tce:
    :param config:
    :return:
    """

    odd_even_flag = 'ok'

    # if len(odd_time) < 2 and len(even_time) >= 2:
    #     odd_time, odd_flux = np.array(even_time), np.array(even_flux)
    #     odd_even_flag = 'no time series (odd)'
    # elif len(even_time) < 2 and len(odd_time) >= 2:
    #     even_time, even_flux = np.array(odd_time), np.array(odd_flux)
    #     odd_even_flag = 'no time series (even)'

    # check to see if there is at least one point in the time interval for the local view
    tmin_local = max(-tce['tce_period'] / 2, -tce['tce_duration'] * config['num_durations'])
    tmax_local = min(tce['tce_period'] / 2, tce['tce_duration'] * config['num_durations'])
    num_pts_local_odd = len(odd_time[(odd_time >= tmin_local) & (odd_time <= tmax_local)])
    num_pts_local_even = len(even_time[(even_time >= tmin_local) & (even_time <= tmax_local)])

    if num_pts_local_odd == 0 and num_pts_local_even > 0:
        odd_time, odd_flux, num_tr_odd = np.array(even_time), np.array(even_flux), num_tr_even
        odd_even_flag = 'no local time series (odd)'
    elif num_pts_local_even == 0 and num_pts_local_odd > 0:
        even_time, even_flux, num_tr_even = np.array(odd_time), np.array(odd_flux), num_tr_odd
        odd_even_flag = 'no local time series (even)'

    # time interval for the transit
    t_min, t_max = max(-tce['tce_period'] / 2, -tce['tce_duration'] / 2), \
                   min(tce['tce_period'] / 2, tce['tce_duration'] / 2)

    if 'no local time series' not in odd_even_flag:

        # create local odd flux view
        loc_flux_odd_view, binned_time_odd, loc_flux_odd_view_var, bin_counts_odd, bin_values_odd = \
            generate_view(odd_time,
                          odd_flux,
                          config['num_bins_loc'],
                          tce['tce_duration'] * config['bin_width_factor_loc'],
                          tmin_local,
                          tmax_local
                          )

        # create local even flux view
        loc_flux_even_view, binned_time_even, loc_flux_even_view_var, bin_counts_even, bin_values_even = \
            generate_view(even_time,
                          even_flux,
                          config['num_bins_loc'],
                          tce['tce_duration'] * config['bin_width_factor_loc'],
                          tmin_local,
                          tmax_local
                          )

        # get indices of both views that are out-of-transit
        inds_even_oot = np.nonzero(~((binned_time_even >= t_min) & (binned_time_even <= t_max)))
        inds_odd_oot = np.nonzero(~((binned_time_odd >= t_min) & (binned_time_odd <= t_max)))
        inds_oot = np.intersect1d(inds_even_oot, inds_odd_oot)

        # replace bin values for both views
        loc_flux_odd_view, loc_flux_odd_view_var, loc_flux_even_view, loc_flux_even_view_var, odd_even_flag, \
        bins_repl = \
            impute_odd_even_views(loc_flux_odd_view,
                                  loc_flux_odd_view_var,
                                  bin_counts_odd,
                                  loc_flux_even_view,
                                  loc_flux_even_view_var,
                                  bin_counts_even,
                                  inds_oot)

        # fill missing bin values
        loc_flux_odd_view, inds_nan_odd = impute_binned_ts(binned_time_odd, loc_flux_odd_view, odd_time, odd_flux,
                                                           tce['tce_period'], tce['tce_duration'])
        # fill remaining missing bins for even with odd view values
        inds_nan_even = np.isnan(loc_flux_even_view)
        loc_flux_even_view[inds_nan_even] = loc_flux_odd_view[inds_nan_even]
        _, inds_nan_even = impute_binned_ts(binned_time_even, loc_flux_even_view, even_time, even_flux,
                                            tce['tce_period'], tce['tce_duration'])

        # variability of bins without any values is set to the std of the phase folded time series
        inds_nan_var = np.isnan(loc_flux_odd_view_var)
        loc_flux_odd_view_var[inds_nan_var] = stats.mad_std(odd_flux, ignore_nan=True)
        inds_nan_var = np.isnan(loc_flux_even_view_var)
        loc_flux_even_view_var[inds_nan_var] = stats.mad_std(even_flux, ignore_nan=True)

        odd_even_flag += f'{np.sum(inds_nan_var)} (both)'

        # compute SE of the mean for in-transit and out-of-transit points for odd and even phase folded time series
        # after replacing bins
        sigma_oot_odd, sigma_it_odd, sigma_oot_even, sigma_it_even = compute_oot_it_var_oddeven_repl(bin_values_odd,
                                                                                                     bin_values_even,
                                                                                                     bins_repl,
                                                                                                     inds_oot)
    else:

        # create local odd flux view
        loc_flux_odd_view, binned_time_odd, loc_flux_odd_view_var, bin_counts_odd, bin_values_odd = \
            generate_view(odd_time,
                          odd_flux,
                          config['num_bins_loc'],
                          tce['tce_duration'] * config['bin_width_factor_loc'],
                          tmin_local,
                          tmax_local
                          )

        sigma_oot_odd, sigma_it_odd, sigma_oot_even, sigma_it_even = compute_oot_it_var_oddeven(odd_time,
                                                                                                odd_flux,
                                                                                                even_time,
                                                                                                even_flux,
                                                                                                t_min,
                                                                                                t_max)

        # fill missing bin values
        loc_flux_odd_view, inds_nan_odd = impute_binned_ts(binned_time_odd, loc_flux_odd_view, odd_time, odd_flux,
                                                           tce['tce_period'], tce['tce_duration'])
        inds_nan_var = np.isnan(loc_flux_odd_view_var)
        loc_flux_odd_view_var[inds_nan_var] = stats.mad_std(odd_flux, ignore_nan=True)

        # given that odd and even are a the same
        loc_flux_even_view, loc_flux_even_view_var, binned_time_even, inds_nan_even = \
            np.array(loc_flux_odd_view), np.array(loc_flux_odd_view_var), np.array(binned_time_odd), \
            {key: np.array(val) for key, val in inds_nan_odd.items()}

    odd_even_stats = {
        'sigma_oot_odd': sigma_oot_odd / max(1, num_tr_odd),
        'sigma_it_odd': sigma_it_odd / max(1, num_tr_odd),
        'sigma_oot_even': sigma_oot_even / max(1, num_tr_even),
        'sigma_it_even': sigma_it_even / max(1, num_tr_even)
    }

    return loc_flux_odd_view, binned_time_odd, loc_flux_odd_view_var, loc_flux_even_view, binned_time_even, \
           loc_flux_even_view_var, odd_even_flag, odd_even_stats, inds_nan_odd, inds_nan_even
