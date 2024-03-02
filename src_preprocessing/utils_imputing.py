""" Utility function to impute missing values in time series. """

# 3rd party
import numpy as np
from astropy.stats import mad_std


def impute_binned_ts(binned_phase, binned_ts, period, duration, binned_ts_var=None, bin_fn=np.nanmedian,
                     bin_var_fn=mad_std):
    """ Impute binned time series:
        - for out-of-transit bins, Gaussian noise based on statistics computed using the out-of-transit points in the
        binned time series is used to fill missing values; if no valid out-of-transit bins are available, uses all valid
        bins to compute these statistics for imputing the bins with missing values.
        - for in-transit bins, linear interpolation is performed (unless there are no valid in-transit bins available;
        in this case it uses the same strategy as for out-of-transit bins).

    :param binned_phase: NumPy array, phase for the binned time series
    :param binned_ts: NumPy array, binned timeseries
    :param period: float, TCE orbital period
    :param duration: float, TCE transit duration
    :param binned_ts_var: NumPy array, variability of binned time series
    :param bin_fn: function used to aggregate bin values
    :param bin_var_fn: function used to estimate uncertainty per bin value
    :return:
        binned_ts: NumPy array, imputed binned timeseries
        inds_nan: dict, boolean indices for the in-transit and out-of-transit imputed bins
        binned_ts_var: NumPy array, variability of binned time series after imputing it
    """

    # get time interval for in-transit cadences
    tmin_it, tmax_it = max(-period / 2, -1.5 * duration / 2), min(period/2, 1.5 * duration / 2)

    # get empty bins indices (nan)
    inds_nan_bins = np.isnan(binned_ts)

    # get empty bin indices for in-transit bins
    inds_it_binned_ts = (binned_phase >= tmin_it) & (binned_phase <= tmax_it)
    # get it missing bin values
    inds_it_nan = np.logical_and(inds_nan_bins, inds_it_binned_ts)
    inds_it_valid = np.logical_and(~inds_nan_bins, inds_it_binned_ts)

    # get empty bin indices for out-of-transit bins
    inds_oot_binned_ts = ~inds_it_binned_ts
    # get oot missing bin values
    inds_oot_nan = np.logical_and(inds_nan_bins, inds_oot_binned_ts)
    # get oot not-missing bin values
    inds_oot_valid = np.logical_and(~inds_nan_bins, inds_oot_binned_ts)

    inds_nan = {'oot': inds_oot_nan, 'it': inds_it_nan}

    if np.any(inds_oot_nan):  # there are oot indices with missing bin values that need to be imputed
        if len(inds_oot_valid) > 0:  # there is at least one not missing oot bin value
            # set Gaussian noise based on out-of-transit phase folded time series statistics
            mu = bin_fn(binned_ts[inds_oot_valid])
            if bin_var_fn.__name__ == 'mad_std':  # astropy
                sigma = bin_var_fn(binned_ts[inds_oot_valid], ignore_nan=True)
            else:
                sigma = bin_var_fn(binned_ts[inds_oot_valid])
        else:  # set Gaussian noise based on entire phase folded time series statistics
            mu = bin_fn(binned_ts)
            if bin_var_fn.__name__ == 'mad_std':  # astropy
                sigma = bin_var_fn(binned_ts, ignore_nan=True)
            else:
                sigma = bin_var_fn(binned_ts)

        # fill missing oot bin values with central tendency and variability measures
        binned_ts = np.where(inds_oot_nan, mu, binned_ts)
        if binned_ts_var is not None:
            binned_ts_var = np.where(inds_oot_nan, sigma, binned_ts_var)

    # fill missing in-transit bin values by linear interpolation
    if np.any(inds_it_valid):  # there's at least one valid in-transit bin value

        if bin_var_fn.__name__ == 'mad_std':  # astropy
            sigma = bin_var_fn(binned_ts, ignore_nan=True)
        else:
            sigma = bin_var_fn(binned_ts)

        binned_ts[inds_it_nan] = np.interp(binned_phase[inds_it_nan], binned_phase[inds_it_valid],
                                           binned_ts[inds_it_valid])
        if binned_ts_var is not None:
            binned_ts_var[inds_it_nan] = sigma
    else:  # if there are no valid in-transit bins, use same approach as for out-of-transit bins
        mu = bin_fn(binned_ts)
        if bin_var_fn.__name__ == 'mad_std':  # astropy
            sigma = bin_var_fn(binned_ts, ignore_nan=True)
        else:
            sigma = bin_var_fn(binned_ts)

        binned_ts = np.where(inds_it_nan, mu, binned_ts)
        if binned_ts_var is not None:
            binned_ts_var = np.where(inds_it_nan, sigma, binned_ts_var)

    return binned_ts, binned_ts_var, inds_nan


def imputing_gaps(time, timeseries, all_gap_time):
    """ Imputing missing time with Gaussian noise computed taking into account global statistics of the data.

    :param time: list of Numpy arrays, contains the non-gapped timestamps
    :param timeseries: list of Numpy arrays, 1D time-series
    :param all_gap_time: list of Numpy arrays, gapped timestamps
    :return:
        time: list of Numpy arrays, now with added gapped timestamps
        timeseries: list of Numpy arrays, now with imputed data

    TODO: compute statistics using only datapoints from cadences close to the gapped time intervals and also in o.o.t
          values.
    """

    med = np.median(timeseries)
    # robust std estimator of the time series
    std_rob_estm = mad_std(timeseries)

    for gap_time in all_gap_time:
        imputed_timeseries = med + np.random.normal(0, std_rob_estm, time.shape)
        timeseries.append(imputed_timeseries.astype(timeseries[0].dtype))
        time.append(gap_time, time.astype(time[0].dtype))

    return time, timeseries


def impute_transits(timeseries, transit_pulse_train, centroid=False):
    """ Impute transits by linearly interpolating them using as boundary values the out-of-transit median in windows
    around the transits. These windows have the same width as transit.

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

            numcadences_win = idxs_it[end_idx - 1] - idxs_it[start_idx] + 1

            # boundary issue - do nothing, since the whole array is a transit; does this happen?
            if idxs_it[start_idx] == 0 and idxs_it[end_idx - 1] == len(transit_pulse_train[i]) - 1:
                continue

            if idxs_it[start_idx] == 0:  # boundary issue start - constant value end

                if centroid:
                    val_interp = {'x': np.median(timeseries['x'][i][
                                                 idxs_it[end_idx - 1] + 1:min(len(timeseries['x'][i]), idxs_it[
                                                     end_idx - 1] + 1 + numcadences_win)]),
                                  'y': np.median(timeseries['y'][i][
                                                 idxs_it[end_idx - 1] + 1:min(len(timeseries['y'][i]), idxs_it[
                                                     end_idx - 1] + 1 + numcadences_win)])}
                    timeseries_interp['x'][i][idxs_it[start_idx]:idxs_it[end_idx - 1] + 1] = val_interp['x']
                    timeseries_interp['y'][i][idxs_it[start_idx]:idxs_it[end_idx - 1] + 1] = val_interp['y']

                else:
                    val_interp = np.median(timeseries[i][idxs_it[end_idx - 1] + 1:min(len(timeseries[i]), idxs_it[
                        end_idx - 1] + 1 + numcadences_win)])
                    timeseries_interp[i][idxs_it[start_idx]:idxs_it[end_idx - 1] + 1] = val_interp

            elif idxs_it[end_idx - 1] == len(transit_pulse_train[i]) - 1:  # boundary issue end - constant value start

                if centroid:
                    val_interp = {'x': np.median(
                        timeseries['x'][i][max(0, idxs_it[start_idx] - numcadences_win):idxs_it[start_idx] + 1]),
                        'y': np.median(timeseries['y'][i][
                                       max(0, idxs_it[start_idx] - numcadences_win):idxs_it[start_idx] + 1])}

                    timeseries_interp['x'][i][idxs_it[start_idx]:idxs_it[end_idx - 1] + 1] = val_interp['x']
                    timeseries_interp['y'][i][idxs_it[start_idx]:idxs_it[end_idx - 1] + 1] = val_interp['y']
                else:
                    val_interp = np.median(
                        timeseries[i][max(0, idxs_it[start_idx] - numcadences_win):idxs_it[start_idx] + 1])
                    timeseries_interp[i][idxs_it[start_idx]:idxs_it[end_idx - 1] + 1] = val_interp

            else:  # linear interpolation
                idxs_interp = np.array([idxs_it[start_idx] - 1, idxs_it[end_idx - 1] + 1])
                idxs_to_interp = np.arange(idxs_it[start_idx], idxs_it[end_idx - 1] + 1)

                if centroid:
                    val_interp = {'x': np.array(
                        [np.median(timeseries['x'][i][max(0, idxs_interp[0] - numcadences_win):idxs_interp[0] + 1]),
                         np.median(timeseries['x'][i][idxs_interp[1]:min(len(timeseries['x'][i]),
                                                                         idxs_interp[1] + numcadences_win) + 1])]),
                        'y': np.array([np.median(
                            timeseries['y'][i][max(0, idxs_interp[0] - numcadences_win):idxs_interp[0] + 1]),
                            np.median(timeseries['y'][i][
                                      idxs_interp[1]:min(len(timeseries['y'][i]),
                                                         idxs_interp[1] + numcadences_win) + 1])])}
                    timeseries_interp['x'][i][idxs_it[start_idx]:idxs_it[end_idx - 1] + 1] = \
                        np.interp(idxs_to_interp, idxs_interp, val_interp['x'])
                    timeseries_interp['y'][i][idxs_it[start_idx]:idxs_it[end_idx - 1] + 1] = \
                        np.interp(idxs_to_interp, idxs_interp, val_interp['y'])

                else:
                    val_interp = np.array(
                        [np.median(timeseries[i][max(0, idxs_interp[0] - numcadences_win):idxs_interp[0] + 1]),
                         np.median(timeseries[i][
                                   idxs_interp[1]:min(len(timeseries[i]), idxs_interp[1] + numcadences_win) + 1])])

                    timeseries_interp[i][idxs_it[start_idx]:idxs_it[end_idx - 1] + 1] = \
                        np.interp(idxs_to_interp, idxs_interp, val_interp)

    return timeseries_interp
