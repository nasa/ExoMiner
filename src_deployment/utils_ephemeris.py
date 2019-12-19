"""
Auxiliary functions that use ephemeris information
"""

# 3rd party
import numpy as np


def find_first_epoch_after_this_time(epoch, period, reference_time):
    """ Finds the first epoch after a certain reference time.

    :param epoch: float initial epoch value (TJD/KJD)
    :param period: float, period (d)
    :param reference_time: float, start reference time
    :return:
        tepoch: float, new epoch value

    # Code is ported from Jeff's Matlab ephemeris matching code
    """

    if epoch < reference_time:
        tepoch = epoch + period * np.ceil((reference_time - epoch) / period)
    else:
        tepoch = epoch - period * np.floor((epoch - reference_time) / period)

    return tepoch


def create_binary_time_series(time, epoch, duration, period):
    """ Creates a binary time series based on the the ephemeris.

    :param time: list, time in days
    :param epoch: float, epoch in days
    :param duration: float, transit duration in days
    :param period: float, orbital period in days
    :return:
        binary_time_series: binary array with 1's for in-transit points and 0's otherwise

    # Code is ported from Jeff's Matlab ephemeris matching code
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

        transit_idxs = np.where(time >= sTransit)[0]
        transit_idxs = np.intersect1d(transit_idxs, np.where(time < eTransit)[0])
        binary_time_series[transit_idxs] = 1

    return binary_time_series
