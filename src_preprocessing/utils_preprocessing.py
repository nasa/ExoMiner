""" General utility functions for preprocessing. """

# 3rd party
import numpy as np


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
