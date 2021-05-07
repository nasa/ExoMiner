""" Utility function to impute missing values in time series. """

# 3rd party
import numpy as np
from astropy.stats import mad_std


def impute_binned_ts(binned_phase, binned_ts, phase, phasefolded_ts, period, duration):
    """ Impute binned time series:
        - for out-of-transit bins, Gaussian noise based on statistics computed using the out-of-transit points in the
        binned time series is used to fill missing values
        - for in-transit bins, linear interpolation is performed

    :param binned_phase: NumPy array, phase for the binned time series
    :param binned_ts: NumPy array, binned timeseries
    :param phase: NumPy array, phase for the phase folded time series
    :param phasefolded_ts: NumPy array, phase folded time series
    :param period: float, TCE orbital period
    :param duration: float, TCE transit duration
    :return:
        binned_ts: NumPy array, imputed binned timeseries
        inds_nan: dict, boolean indices for the in-transit and out-of-transit imputed bins
    """

    # get time interval for in-transit cadences
    tmin_it, tmax_it = max(-period / 2, -1.5 * duration / 2), min(period/2, 1.5 * duration / 2)

    # get empty bins indices (nan)
    inds_nan_bins = np.isnan(binned_ts)

    # get indices for in-transit bins
    inds_it_binned_ts = (binned_phase >= tmin_it) & (binned_phase <= tmax_it)

    # get out-of-transit cadences in the phase folded time series
    inds_oot_phasefolded_ts = ~((phase >= tmin_it) & (phase <= tmax_it))
    if np.any(inds_oot_phasefolded_ts):
        # set Gaussian noise based on out-of-transit phase folded time series statistics
        mu, sigma = np.nanmedian(phasefolded_ts[inds_oot_phasefolded_ts]), \
                    mad_std(binned_ts[~inds_it_binned_ts], ignore_nan=True)
    else:
        # set Gaussian noise based on entire phase folded time series statistics
        mu, sigma = np.nanmedian(phasefolded_ts), mad_std(binned_ts, ignore_nan=True)

    rng = np.random.default_rng()

    # get oot missing bin values
    inds_oot_nan = np.logical_and(inds_nan_bins, ~inds_it_binned_ts)

    # fill missing bin values with Gaussian noise using statistics computed from the out-of-transit phase folded time
    # series
    binned_ts = np.where(inds_oot_nan, rng.normal(mu, sigma, len(inds_oot_nan)), binned_ts)

    # fill missing in-transit bin values by linear interpolation
    inds_it_nan = np.logical_and(inds_nan_bins, inds_it_binned_ts)
    inds_it_valid = np.logical_and(~inds_nan_bins, inds_it_binned_ts)
    if np.any(inds_it_valid):
        binned_ts[inds_it_nan] = np.interp(binned_phase[inds_it_nan], binned_phase[inds_it_valid], binned_ts[inds_it_valid])
    else:  # if there are no valid in-transit bins, use same approach as for out-of-transit bins
        binned_ts = np.where(inds_it_nan, rng.normal(mu, sigma, len(inds_it_nan)), binned_ts)

    inds_nan = {'oot': inds_oot_nan, 'it': inds_it_nan}

    return binned_ts, inds_nan
