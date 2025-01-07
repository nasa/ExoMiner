"""
Utility functions used to preprocess flux time series data.
"""

# 3rd party
import numpy as np
from astropy.stats import mad_std
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import warnings

# local
from src_preprocessing.third_party.kepler_spline import kepler_spline
from src_preprocessing.lc_preprocessing.utils_ephemeris import find_first_epoch_after_this_time


def split_timeseries_on_time_gap(time, flux, gap_width):
    """ Split timestamps array and flux time series based on `gap_width` days time interval between continuous
    timestamps.

    Args:
        time: NumPy array, timestamps
        flux: NumPy array, flux time series
        gap_width: float, gap width in days. If two contiguous timestamps have a time gap width larger than the gap
        width, the two arrays are split on that index.

    Returns:
        time_arr, list of NumPy arrays of timestamps after splitting based on `gap_width`
        flux_arr, list of NumPy arrays of flux time series after splitting based on `gap_width`

    """

    time_diff = np.diff(time)
    idxs_split = np.where(time_diff >= gap_width)[0] + 1
    time_arr, flux_arr = np.split(time, idxs_split), np.split(flux, idxs_split)

    return time_arr, flux_arr


def detrend_flux_using_spline(time_arr, flux_arr, rng):
    """ Detrend timeseries by fitting a spline to a version of the timeseries with linear interpolation performed across
    the transits.

    Args:
        time_arr: NumPy array, timestamps
        flux_arr: NumPy array, time series
        rng: NumPy rng generator

    Returns:
        - time_arr, NumPy array with timestamps
        - detrended_flux_arr, NumPy array with detrended timeseries (i.e., detrended_flux = flux / spline_flux)
        - spline_flux_arr, NumPy array with fitted spline (i.e., the trend)
        - res_flux_arr, NumPy array with the residual timeseries (i.e., res_flux = flux - spline_flux)
    """

    # fit a spline to the flux time-series
    spline_flux_arr = kepler_spline.fit_kepler_spline([time_arr], [flux_arr],
                                                      bkspace_min=0.5,
                                                      bkspace_max=20,
                                                      bkspace_num=20,
                                                      maxiter=5,
                                                      penalty_coeff=1.0,
                                                      outlier_cut=3,
                                                      verbose=False)[0][0]

    # get indices for which the spline has finite values
    finite_idxs_arr = np.isfinite(spline_flux_arr)

    if finite_idxs_arr.any():  # interpolate spline using linear interpolation
        spline_flux_arr = np.interp(time_arr, time_arr[finite_idxs_arr], spline_flux_arr[finite_idxs_arr],
                                    left=np.nan, right=np.nan)
    else:
        spline_flux_arr = np.nanmedian(flux_arr) * np.ones(len(flux_arr))  # spline is set to median flux

    # check for non-finite indices that were not interpolated/extrapolated
    finite_idxs_arr = np.isfinite(spline_flux_arr)
    # fill non-finite points in the spline using median and mad std Gaussian statistics
    mu, std = np.nanmedian(spline_flux_arr), mad_std(spline_flux_arr, ignore_nan=True)
    spline_flux_arr[~finite_idxs_arr] = rng.normal(mu, std, (~finite_idxs_arr).sum())

    # compute residual
    res_flux_arr = flux_arr - spline_flux_arr

    # detrend flux by normalize flux time series with the fitted spline
    detrended_flux_arr = flux_arr / spline_flux_arr

    return time_arr, detrended_flux_arr, spline_flux_arr, res_flux_arr


def resample_timeseries(time, flux, num_resampled_points):
    """

    Args:
        time: NumPy array, timestamps
        flux: NumPy array, flux values
        num_resampled_points: int, number of points to resample

    Returns:
        resampled_time: NumPy array, resampled timestamps
        resampled_flux: NumPy array, resampled flux values
    """

    f = interp1d(time, flux, kind='nearest', bounds_error=False, fill_value="extrapolate")

    resampled_time = np.linspace(time[0], time[-1], num=num_resampled_points)

    resampled_flux = f(resampled_time)

    return resampled_time, resampled_flux


def extract_flux_windows_for_tce(time, flux, transit_mask, tce_time0bk, period_days, tce_duration, n_durations_window, 
                                 gap_width, buffer_time, frac_valid_cadences_in_window_thr, frac_valid_cadences_it_thr,
                                 resampled_num_points, tce_uid, rng, plot_dir=None, logger=None):
    """ Extract flux windows for a TCE based on its orbital period, epoch, and transit duration. These windows are set
    to `n_durations_window` * `tce_duration` and are resampled to `resampled_num_points`. Transit windows are first 
    built centered around the midtransit points available in the `time` timestamps array. Out-of-transit windows are 
    chosen in equal number to the transit windows by randomly choosing a set of out-of-transit cadences that do not 
    overlap with in-transit regions (which are defined as midtransit +- tce_duration).

    Args:
        time: NumPy array, timestamps
        flux: NumPy array, flux values
        transit_mask: NumPy array with boolean elements for in-transit (True) and out-of-transit (False) timestamps.
        tce_time0bk: float,
        period_days: float, orbital period in days
        tce_duration: float, transit duration in hours
        n_durations_window: int, number of transit durations in the extracted window.
        gap_width: float, gap width in days between consecutive cadences to split arrays into sub-arrays
        buffer_time: int, buffer in minutes between out-of-transit and in-transit cadences
        frac_valid_cadences_in_window_thr: float, fraction of valid cadences in window
        frac_valid_cadences_it_thr: float, fraction of valid in-transit cadences
        resampled_num_points: int, number of points to resample
        tce_uid: str, TCE uid
        rng: NumPy rng generator
        plot_dir: Path, plot directory; set to None to disable plotting

    Returns:
        resampled_flux_it_windows_arr, NumPy array with resampled flux for transit windows
        resampled_flux_oot_windows_arr, NumPy array with resampled flux for out-of-transit windows
        midtransit_points_windows_arr, list with midtransit points for transit windows
        midoot_points_windows_arr, list with midtransit points for out-of-transit windows
    """

    buffer_time /= 1440  # convert from minutes to days
    tce_duration /= 24  # convert from hours to days

    # TODO: deal with NaNs; need to interpolate time?; impute cadences with missing values using what strategy?
    # remove NaNs
    valid_idxs_data = np.logical_and(np.isfinite(time), np.isfinite(flux))
    time, flux, transit_mask = time[valid_idxs_data], flux[valid_idxs_data], transit_mask[valid_idxs_data]

    # split time series on gaps
    time_arrs, flux_arrs = split_timeseries_on_time_gap(time, flux, gap_width)
    if logger:
        logger.info(f'Time series split into {len(time_arrs)} arrays due to gap(s) in time.')

    # find first midtransit point in the time array
    first_transit_time = find_first_epoch_after_this_time(tce_time0bk, period_days, time[0])
    # compute all midtransit points in the time array
    midtransit_points_arr = np.array([first_transit_time + phase_k * period_days
                                      for phase_k in range(int(np.ceil((time[-1] - time[0]) / period_days)))])
    if logger:
        logger.info(f'Found {len(midtransit_points_arr)} midtransit points.')

    # get start and end timestamps for the windows based on the available midtransit points
    start_time_windows, end_time_windows = (midtransit_points_arr - n_durations_window * tce_duration / 2,
                                            midtransit_points_arr + n_durations_window * tce_duration / 2)

    # choose only those midtransit points whose windows fit completely inside the time array
    valid_midtransit_points_arr = midtransit_points_arr[np.logical_and(start_time_windows >= time[0],
                                                                      end_time_windows <= time[-1])]

    if logger:
        logger.info(f'Found {len(valid_midtransit_points_arr)} midtransit points whose windows fit completely inside the time '
          f'array.')

    valid_idxs_data = np.logical_and(np.isfinite(time), np.isfinite(flux))

    # extract transit windows
    time_it_windows_arr, flux_it_windows_arr, midtransit_points_windows_arr = [], [], []
    for start_time_window, end_time_window, midtransit_point_window in zip(start_time_windows, end_time_windows,
                                                                           valid_midtransit_points_arr):

        # find indices in window
        idxs_window = np.logical_and(time >= start_time_window, time <= end_time_window)
        if idxs_window.sum() == 0:
            if logger:
                logger.info(f'No valid indices in window [{start_time_window}, {end_time_window}.')
            continue

        # find in-transit windows
        idxs_it_window = np.logical_and(time >= midtransit_point_window - tce_duration / 2,
                                        time <= midtransit_point_window + tce_duration / 2)

        if idxs_it_window.sum() == 0:
            if logger:
                logger.info(f'No valid in-transit indices in window [{start_time_window}, {end_time_window}.')
            continue

        # check for valid window and in-transit region
        valid_window_flag = (((idxs_window & valid_idxs_data).sum() / idxs_window.sum()) >
                             frac_valid_cadences_in_window_thr)
        valid_it_window_flag = (((idxs_it_window & valid_idxs_data).sum() / idxs_it_window.sum()) >
                                frac_valid_cadences_it_thr)

        if valid_window_flag and valid_it_window_flag:
            time_window = time[idxs_window]
            flux_window = flux[idxs_window]

            time_it_windows_arr.append(time_window)
            flux_it_windows_arr.append(flux_window)
            midtransit_points_windows_arr.append(midtransit_point_window)

    n_it_windows = len(time_it_windows_arr)
    if logger:
        logger.info(f'Extracted {n_it_windows} transit windows.')

    if n_it_windows == 0:
        raise ValueError(f'No valid transit windows detected for tce: {tce_uid}')

    # get start and end timestamps for the windows that should be excluded from out-of-transit windows
    start_time_windows, end_time_windows = (midtransit_points_arr - (n_durations_window + 1) * tce_duration / 2 -
                                            buffer_time,
                                            midtransit_points_arr + (n_durations_window + 1) * tce_duration / 2 +
                                            buffer_time)



    # get oot candidates for oot windows, that do not fall on other transit events
    it_window_mask = np.zeros(len(time), dtype=bool)
    for start_time, end_time in zip(start_time_windows, end_time_windows):
        it_window_mask |= np.logical_and(time >= start_time, time <= end_time)
    
    oot_points_arr = time[np.logical_and(~it_window_mask, ~transit_mask)]

    if logger:
        logger.info(f'Found {len(oot_points_arr)} out-of-transit points.')

    rng.shuffle(oot_points_arr)

    # extract oot windows
    time_oot_windows_arr, flux_oot_windows_arr, midoot_points_windows_arr = [], [], []
    for midoot_point in oot_points_arr:

        # find indices in window
        start_time_window, end_time_window = (midoot_point - n_durations_window * tce_duration / 2,
                                              midoot_point + n_durations_window * tce_duration / 2)
        idxs_window = np.logical_and(time >= start_time_window, time <= end_time_window)

        if idxs_window.sum() == 0:
            if logger:
                logger.info(f'No valid indices in window [{start_time_window}, {end_time_window}.')
            continue

        # check for valid window and in-transit region
        valid_window_flag = (((idxs_window & valid_idxs_data).sum() / idxs_window.sum()) >
                             frac_valid_cadences_in_window_thr)

        if valid_window_flag:
            time_window = time[idxs_window]
            flux_window = flux[idxs_window]

            time_oot_windows_arr.append(time_window)
            flux_oot_windows_arr.append(flux_window)
            midoot_points_windows_arr.append(midoot_point)

        if len(midoot_points_windows_arr) == n_it_windows:
            break

    n_oot_windows = len(time_oot_windows_arr)
    if logger:
        logger.info(f'Extracted {len(time_oot_windows_arr)} out-of-transit windows.')

    if n_oot_windows < n_it_windows:
        if logger:
            logger.warn(f'Number of out-of-transit windows ({n_oot_windows}) is less than number of transit windows '
                      f'({n_it_windows}).')
        warnings.warn(f'Number of out-of-transit windows ({n_oot_windows}) is less than number of transit windows '
                      f'({n_it_windows}).')

    # plot transit and oot windows
    if plot_dir:
        f, ax = plt.subplots(figsize=(12, 6))
        for transit_i in range(n_it_windows):
            ax.scatter(time_it_windows_arr[transit_i], flux_it_windows_arr[transit_i], c='g', s=8,
                       label='Transit Windows' if transit_i == 0 else None)
        for transit_i in range(n_oot_windows):
            ax.scatter(time_oot_windows_arr[transit_i], flux_oot_windows_arr[transit_i], c='b', s=8,
                       label='Out-of-transit Windows' if transit_i == 0 else None)
        ax.set_xlabel('Time - 2457000 [BTJD days]')
        ax.set_ylabel('Detrended Normalized Flux')
        ax.legend()
        f.tight_layout()
        f.savefig(plot_dir / f'{tce_uid}_it_oot_windows_flux.png')
        plt.close()

    # resample transit windows
    resampled_time_it_windows_arr, resampled_flux_it_windows_arr = [], []
    for time_window, flux_window in zip(time_it_windows_arr, flux_it_windows_arr):
        resampled_time_window, resampled_flux_window = resample_timeseries(time_window, flux_window,
                                                                           resampled_num_points)

        resampled_time_it_windows_arr.append(resampled_time_window)
        resampled_flux_it_windows_arr.append(resampled_flux_window)

    # resample out-of-transit windows
    resampled_time_oot_windows_arr, resampled_flux_oot_windows_arr = [], []
    for time_window, flux_window in zip(time_oot_windows_arr, flux_oot_windows_arr):
        resampled_time_window, resampled_flux_window = resample_timeseries(time_window, flux_window,
                                                                           resampled_num_points)

        resampled_time_oot_windows_arr.append(resampled_time_window)
        resampled_flux_oot_windows_arr.append(resampled_flux_window)

    if plot_dir:
        f, ax = plt.subplots(figsize=(12, 6))
        for transit_i in range(n_it_windows):
            ax.scatter(resampled_time_it_windows_arr[transit_i], resampled_flux_it_windows_arr[transit_i], c='g', s=8,
                       label='Transit Windows' if transit_i == 0 else None)
        for transit_i in range(n_oot_windows):
            ax.scatter(resampled_time_oot_windows_arr[transit_i], resampled_flux_oot_windows_arr[transit_i], c='b', s=8,
                       label='Out-of-transit Windows' if transit_i == 0 else None)
        ax.set_xlabel('Time - 2457000 [BTJD days]')
        ax.set_ylabel('Detrended Normalized Flux')
        ax.legend()
        f.suptitle('Resampled Windows')
        f.tight_layout()
        f.savefig(plot_dir / f'{tce_uid}_it_oot_windows_flux_resampled.png')
        plt.close()

    return (resampled_flux_it_windows_arr, resampled_flux_oot_windows_arr, midtransit_points_windows_arr,
            midoot_points_windows_arr)

def build_transit_mask_for_lightcurve(time, tce_list):
    """
    Generates in transit mask for a given light curve time array based on a given list of TCEs. A mask the same size of the 
    time value array is created for which given elements are set to 'True' for in-transit timestamps and 'False' for out of 
    transit ones, based on the orbital period, epoch, and transit duration of the TCEs provided.

    Args:
        time: NumPy array, timestamps
        tce_list: List containing dict of ephemerides data for all tces used.
                            **Epoch should be the midpoint time for a transit

    Returns:
        in_transit_mask, NumPy array with boolean elements for in-transit (True) and out-of-transit (False) timestamps.
    """
    
    in_transit_mask = np.zeros(len(time), dtype=bool)
    for tce in tce_list:
        epoch = tce['tce_time0bk']
        period = tce['tce_period']
        duration = tce['tce_duration']
    
        duration /= 24 #convert hours to days
        
        in_transit_mask |= np.abs((time - epoch) % period ) < duration #one transit duration to left and right

    return in_transit_mask

def plot_detrended_flux_time_series_sg(time, flux, detrend_time, detrend_flux, trend, sector, plot_dir=None):
    """
    Builds plot for a given time series that was detrended using savitzky golay. Builds three plots over time:
    1. Raw Flux
    2. Detrended Normalized Flux
    3. Trend against Detrended Normalized Flux

    Args:
        time: NumPy array, timestamps
        flux: NumPy array, flux time series
        detrend_time: NumPy array, timestamps associated with detrended flux
        detrend_flux: NumPy array, flux time series detrended with savitzky golay
        trend: NumPy array, trend returned by savitzky golay detrending.
        sector: String, of an integer, for the sector corresponding to the time series
        plot_dir: Path, directory to create plot

    Returns:
        None
    """

    if plot_dir:
        f, ax = plt.subplots(3, 1, figsize=(12, 8))
        ax[0].scatter(time, flux, s=8, label='PDCSAP Flux')
        ax[0].set_ylabel('PDCSAP Flux [e-/cadence]')
        ax[1].scatter(detrend_time, detrend_flux, s=8)
        ax[1].set_ylabel('Detrended Normalized Flux')
        ax[2].scatter(detrend_time, detrend_flux, s=8, label= 'Detrended')
        ax[2].scatter(detrend_time, trend, s=8)
        ax[2].set_ylabel('Flux Trend [e-/cadence]')
        ax[2].set_xlabel('Time - 2457000 [BTJD days]')
        f.savefig(plot_dir / f'sector_{sector}_detrended_flux.png')
        plt.close()