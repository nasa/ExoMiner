""" Utility functions used for matching between transit signals using ephemerides information. """

# 3rd party
import numpy as np


def find_nearest_epoch_to_this_time(epoch, period, reference_time):
    """ Finds the nearest epoch to a certain reference time.

    :param epoch: float initial epoch value (TJD/KJD)
    :param period: float, period (d)
    :param reference_time: float, start reference time
    :return:
        tepoch: float, new epoch value

    # Code is ported from Jeff's Matlab ephemeris matching code
    """

    tepoch = epoch + period * np.round((reference_time - epoch) / period)

    return tepoch


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


def create_binary_time_series(epoch, duration, period, tStart, tEnd, samplingInterval):
    """ Creates a binary time series based on the the ephemeris.

    :param epoch: float, epoch in days
    :param duration: float, transit duration in days
    :param period: float, orbital period in days
    :param tStart: float, reference start time
    :param tEnd: float, reference end time
    :param samplingInterval: float, sampling interval in days
    :return:
        binary_time_series: binary array with 1's for in-transit points and 0's otherwise

    # Code is ported from Jeff's Matlab ephemeris matching code
    """

    # sampleTimes = np.linspace(tStart / samplingInterval, tEnd / samplingInterval, (tEnd - tStart) / samplingInterval,
    #
    #                           endpoint=True)
    sampleTimes = np.linspace(tStart / samplingInterval, tEnd / samplingInterval,
                              int((tEnd - tStart) / samplingInterval),
                              endpoint=True)

    # mid-transit timestamps between epoch and reference end time
    midTransitTimes = np.arange(epoch, tEnd, period)

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
    # convert to units of sampling interval
    startTransitTimes = (extendedMidTransitTimes - 0.5 * duration) / samplingInterval
    endTransitTimes = (extendedMidTransitTimes + 0.5 * duration) / samplingInterval

    # initialize binary time series - points with 1's belong to the transit
    binary_time_series = np.zeros(len(sampleTimes), dtype='uint8')
    # binary_time_series = np.zeros(len(time), dtype='bool')

    # set to 1 the in-transit timestamps
    for sTransit, eTransit in zip(startTransitTimes, endTransitTimes):

        transit_idxs = np.where(sampleTimes >= sTransit)[0]
        transit_idxs = np.intersect1d(transit_idxs, np.where(sampleTimes < eTransit)[0])
        binary_time_series[transit_idxs] = 1
        # binary_time_series[transit_idxs] = True

    return binary_time_series
