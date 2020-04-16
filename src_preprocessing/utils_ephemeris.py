"""
Auxiliary functions that use ephemeris information.
"""

# 3rd party
import numpy as np


def find_first_epoch_after_this_time(epoch, period, reference_time):
    """ Finds the first epoch after a certain reference time.

    :param epoch: float initial epoch value (days)
    :param period: float, period (days)
    :param reference_time: float, start reference time (days)
    :return:
        tepoch: float, new epoch value

    # Code is ported from Jeff Smith's Matlab ephemeris matching code
    """

    if epoch < reference_time:
        tepoch = epoch + period * np.ceil((reference_time - epoch) / period)
    else:
        tepoch = epoch - period * np.floor((epoch - reference_time) / period)

    return tepoch


def create_binary_time_series(time, epoch, duration, period):
    """ Creates a binary time series based on the the ephemeris.

    :param time: numpy array, time in days
    :param epoch: float, epoch in days
    :param duration: float, transit duration in days
    :param period: float, orbital period in days
    :return:
        binary_time_series: binary array with 1's for in-transit points and 0's otherwise

    # Code is ported from Jeff Smith's Matlab ephemeris matching code
    """

    # initialize binary time series - points with 1's belong to the transit
    binary_time_series = np.zeros(len(time), dtype='uint8')

    # mid-transit timestamps between epoch and reference end time
    midTransitTimes = np.arange(epoch, time[-1], period)

    # get transits whose mid-point is before and after the reference start and end times
    if len(midTransitTimes) != 0:  # epoch before reference end time
        midTransitTimeBefore = midTransitTimes[0] - period
        midTransitTimeAfter = midTransitTimes[-1] + period
    else:  # epoch after reference end time
        midTransitTimeBefore = epoch - period
        midTransitTimeAfter = epoch

    # concatenate the mid-transit timestamps before, inside and after the time array
    extendedMidTransitTimes = np.concatenate([[midTransitTimeBefore], midTransitTimes, [midTransitTimeAfter]])

    # get beginning and end timestamps of the transits
    startTransitTimes = extendedMidTransitTimes - 0.5 * duration
    endTransitTimes = extendedMidTransitTimes + 0.5 * duration

    # set to 1 the in-transit timestamps
    for sTransit, eTransit in zip(startTransitTimes, endTransitTimes):

        # warning value when the array has NaN values; they are considered False, so not in the interval defined
        transit_idxs = np.where(np.logical_and(time >= sTransit, time < eTransit))[0]
        # transit_idxs = np.where(time >= sTransit)[0]
        # transit_idxs = np.intersect1d(transit_idxs, np.where(time < eTransit)[0])
        binary_time_series[transit_idxs] = 1

    return binary_time_series


def lininterp_transits(timeseries, transit_pulse_train, idxs_it, start_idxs, end_idxs):
    """ Linearly interpolate the timeseries across the in-transit cadences.

    :param timeseries: list of numpy arrays, time-series; if centroid is True, then it is a dictionary with a list of
    numpy arrays for each coordinate ('x' and 'y')
    :param transit_pulse_train: list of numpy arrays, binary arrays that are 1's for in-transit cadences and 0 otherwise
    :param idxs_it: list of numpy arrays indicating the indexes of in-transit cadences in transit_pulse_train
    :param start_idxs: list of numpy arrays indicating the starting index for each transit in idxs_it
    :param end_idxs: list of numpy arrays indicating the last index for each transit in idxs_it
    :return:
        timeseries_interp: list of numpy arrays, time-series linearly interpolated at the transits
    """

    # initialize variables
    timeseries_interp = []

    for i in range(len(timeseries)):

        timeseries_interp.append(np.array(timeseries[i]))

        if len(idxs_it[i]) == 0:  # no transits in the array
            continue

        for start_idx, end_idx in zip(start_idxs[i], end_idxs[i]):

            # boundary issue - do nothing, since the whole array is a transit; does this happen?
            if idxs_it[i][start_idx] == 0 and idxs_it[i][end_idx - 1] == len(transit_pulse_train[i]) - 1:
                continue

            if idxs_it[i][start_idx] == 0:  # boundary issue start - give constant value at the end
                timeseries_interp[i][idxs_it[i][start_idx]:idxs_it[i][end_idx - 1] + 1] = \
                    timeseries[i][idxs_it[i][end_idx - 1] + 1]

            elif idxs_it[i][end_idx - 1] == len(transit_pulse_train[i]) - 1:  # boundary issue end - constant value start
                timeseries_interp[i][idxs_it[i][start_idx]:idxs_it[i][end_idx - 1] + 1] = \
                    timeseries[i][idxs_it[i][start_idx] - 1]

            else:  # linear interpolation
                idxs_interp = np.array([idxs_it[i][start_idx] - 1, idxs_it[i][end_idx - 1] + 1])
                idxs_to_interp = np.arange(idxs_it[i][start_idx], idxs_it[i][end_idx - 1] + 1)

                timeseries_interp[i][idxs_to_interp] = np.interp(idxs_to_interp,
                                                                 idxs_interp,
                                                                 timeseries[i][idxs_interp])

    return timeseries_interp


def get_startend_idxs_inter(transit_pulse_train):
    """ Get in-transit indexes and the respective start and end indexes based on a transit pulse train of 0's for the
    out-of-transit cadences and 1's for the in-transit cadences.

    :param transit_pulse_train: list of numpy arrays, binary arrays that are 1's for in-transit cadences and 0 otherwise
    :return:
        idxs_it: list of numpy arrays indicating the indexes of in-transit cadences in transit_pulse_train
        start_idxs: list of numpy arrays indicating the starting index for each transit
        end_idxs: list of numpy arrays indicating the last index for each transit
    """

    idxs_it, start_idxs, end_idxs = [], [], []
    for i in range(len(transit_pulse_train)):

        # get in-transit indexes
        idxs_it.append(np.where(transit_pulse_train[i] == 1)[0])

        if len(idxs_it[-1]) == 0:  # no transits in the array
            start_idxs.append([])
            end_idxs.append([])
        else:
            # get indexes in idxs_it that correspond to the edges of the transits
            idxs_lim = np.where(np.diff(idxs_it[-1]) > 1)[0] + 1

            # add first index
            start_idxs.append(np.insert(idxs_lim, 0, 0))
            # add last index
            end_idxs.append(np.append(idxs_lim, len(idxs_it[-1])))

    return idxs_it, start_idxs, end_idxs
