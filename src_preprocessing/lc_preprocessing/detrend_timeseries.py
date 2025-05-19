"""
Methods used for detrending timeseries such as flux and flux-weighted centroid motion.
"""

# 3rd party
import numpy as np
from astropy.stats import mad_std
import pandas as pd

# local
from src_preprocessing.third_party.kepler_spline import kepler_spline
from src_preprocessing.light_curve import util


def detrend_flux_using_spline(flux_arrs, time_arrs, intransit_cadences, config):
    """ Detrend timeseries by fitting a spline to a version of the timeseries with linear interpolation performed across
    the transits.

    Args:
        flux_arrs: list of NumPy arrays, timeseries
        time_arrs: list of NumPy arrays, timestamps
        intransit_cadences: list of NumPy arrays, binary arrays with flags for in-transit cadences
        config: dict, preprocessing parameters

    Returns:
        - time, NumPy array with timestamps
        - detrended_flux, NumPy array with detrended timeseries (i.e., detrended_flux = flux / spline_flux)
        - spline_flux, NumPy array with fitted spline (i.e., the trend)
        - flux_lininterp, NumPy array with the timeseries linearly interpolated across transits (i.e, they are "masked")
    """

    # split on gaps
    time_arrs, (flux_arrs, intransit_cadences), _ = util.split(time_arrs,
                                                            [flux_arrs, intransit_cadences],
                                                            gap_width=config['gapWidth'])

    time, flux = np.concatenate(time_arrs), np.concatenate(flux_arrs)

    # linearly interpolate across TCE transits
    flux_arrs_lininterp = []
    for time_arr, flux_arr, intransit_cadences_arr in zip(time_arrs, flux_arrs, intransit_cadences):
        if intransit_cadences_arr.all():
            flux_lininterp_arr = np.nanmedian(flux_arr) * np.ones(len(time_arr), dtype='float')
        else:
            flux_lininterp_arr = np.interp(time_arr, time_arr[~intransit_cadences_arr],
                                           flux_arr[~intransit_cadences_arr],
                                           left=np.nan, right=np.nan)

        flux_arrs_lininterp.append(flux_lininterp_arr)
 
    # fill in missing values in linearly interpolated array
    rng = np.random.default_rng(seed=config['random_seed'])
    for arr_i in range(len(flux_arrs_lininterp)):
        finite_idxs = np.isfinite(flux_arrs_lininterp[arr_i])
        mu, std = (np.nanmedian(flux_arrs_lininterp[arr_i][finite_idxs]),
                   mad_std(flux_arrs_lininterp[arr_i][finite_idxs], ignore_nan=True))
        flux_arrs_lininterp[arr_i][~finite_idxs] = rng.normal(mu, std, (~finite_idxs).sum())

    # time_arrs, flux_arrs_lininterp = remove_non_finite_values([time_arrs, flux_arrs_lininterp])

    # flux_arrs_lininterp = lininterp_transits(flux_arrs, binary_time_all, centroid=False)
    flux_lininterp = np.concatenate(flux_arrs_lininterp)

    # fit a spline to the flux time-series
    spline_flux_arrs = kepler_spline.fit_kepler_spline(time_arrs, flux_arrs_lininterp,
                                                       bkspace_min=config['spline_bkspace_min'],
                                                       bkspace_max=config['spline_bkspace_max'],
                                                       bkspace_num=config['spline_bkspace_num'],
                                                       maxiter=config['spline_maxiter'],
                                                       penalty_coeff=config['spline_penalty_coeff'],
                                                       outlier_cut=config['spline_outlier_cut'],
                                                       verbose=False)[0]
    spline_flux = np.concatenate(spline_flux_arrs)

    # get indices for which the spline has finite values
    finite_idxs = np.isfinite(spline_flux)

    if finite_idxs.any():  # interpolate spline using linear interpolation
        spline_flux = np.interp(time, time[finite_idxs], spline_flux[finite_idxs], left=np.nan, right=np.nan)
    else:
        spline_flux = np.nanmedian(flux) * np.ones(len(flux))  # spline is set to median flux

    # # check for non-finite indices that were not interpolated/extrapolated
    # finite_idxs = np.isfinite(spline_flux)
    # # fill non-finite points in the spline using median and mad std Gaussian statistics
    # mu, std = np.nanmedian(spline_flux), mad_std(spline_flux, ignore_nan=True)
    # rng = np.random.default_rng(seed=config['random_seed'])
    # spline_flux[~finite_idxs] = rng.normal(mu, std, (~finite_idxs).sum())

    # detrend flux by normalize flux time series with the fitted spline
    detrended_flux = flux / spline_flux

    # remove median from trend (by time splits)
    dt = np.diff(time)

    cut = np.where(dt > 5 * np.nanmedian(dt))[0] + 1
    low = np.append([0], cut)
    high = np.append(cut, len(time))
    for low_i, high_i in zip(low, high):
        spline_flux[low_i: high_i] /= np.nanmedian(spline_flux[low_i: high_i])

    return time, detrended_flux, spline_flux, flux_lininterp


def compute_bic(k, n, x, x_fit, penalty_weight=1):
    """ Compute BIC score.

    Args:
        k: int, polynomial order
        n: int, number of cadences in time series
        x: NumPy array, flux time series (raw PDC-SAP flux) [1xn]
        x_fit: NumPy array, flux trend [1xn]
        penalty_weight: float, penalty weight for model complexity (polynomial order)

    Returns: float, BIC score

    """

    sse = np.nansum((x - x_fit) ** 2)
    # bic_score = (k + 1) * np.log(n) + n * np.log(sse / n)

    scaled_diffs = np.diff(x) / np.sqrt(2)
    sigma = np.nanmedian(np.abs(scaled_diffs)) * 1.48
    likelihood_term = k * np.log(2 * np.pi * sigma ** 2) + sse / sigma ** 2
    penalty_term = k * np.log(n)
    bic_score = likelihood_term + penalty_weight * penalty_term

    return bic_score


def detrend_flux_using_sg_filter(lc, mask_in_transit, win_len, sigma, max_poly_order, penalty_weight, break_tolerance):
    """ Detrend timeseries by applying a Savitzky-Golay filter to a version of the timeseries where in-transit cadences
    are masked (i.e., not taken into account).

    Args:
        lc: lightcurve object, with raw timestamps and timeseries
        mask_in_transit: NumPy array, in-transit cadences are flagged as True
        win_len: int, number of points considered in the window
        sigma: outlier factor used to mask data points before detrending (i.e., x - med(flux) >= sigma * std(flux))
        max_poly_order: int, maximum polynomial order tested
        penalty_weight: float: weight given to model's complexity when using BIC for model selection
        break_tolerance: float, factor used to split the timeseries into smaller segments based on time interval between
            cadences (i.e., there's a split if diff(t_x) > break_tolerance * med(diff(T)))

    Returns:
        - time, NumPy array with timestamps
        - detrended_flux, NumPy array with detrended timeseries (i.e., detrended_flux = flux / trend)
        - trend, NumPy array with trend
        - models_info_df, pandas DataFrame with BIC scores for the fitted models
    """

    if mask_in_transit.all():
        time = lc.time.value
        trend = np.nanmedian(lc.flux.value) * np.ones_like(lc.flux.value)
        detrended_flux = lc.flux.value / trend
        trend = np.ones_like(lc.flux.value)
        models_info_df = pd.DataFrame({'poly_order': [0], 'bic_score': [np.nan]})

        return time, detrended_flux, trend, models_info_df

    n_samples = len(lc.flux.value)
    poly_order_arr = np.arange(0, max_poly_order + 1)
    best_bic, best_model = np.inf, 0
    lc_detrended = {poly_order: {'flux': None, 'trend': None, 'bic': np.nan} for poly_order in poly_order_arr}
    for poly_order in poly_order_arr:

        lc_sector_flatten, lc_sector_trend = lc.flatten(window_length=win_len, polyorder=poly_order, return_trend=True,
                                                        sigma=sigma, mask=mask_in_transit,
                                                        break_tolerance=break_tolerance)
        lc_detrended[poly_order]['flux'] = lc_sector_flatten.copy()
        lc_detrended[poly_order]['trend'] = lc_sector_trend.copy()

        bic_k = compute_bic(poly_order, n_samples, lc.flux.value, lc_sector_trend.flux.value,
                            penalty_weight=penalty_weight)
        lc_detrended[poly_order]['bic'] = bic_k
        if bic_k < best_bic:
            best_bic, best_model = bic_k, poly_order

    bic_scores = [lc_detrended[poly_order]['bic'] for poly_order in lc_detrended]
    models_info_df = pd.DataFrame({'poly_order': poly_order_arr, 'bic_score': bic_scores})
    models_info_df.sort_values(by='bic_score', ascending=True, inplace=True)
    models_info_df.set_index('poly_order', inplace=True)

    time, detrended_flux = lc_detrended[best_model]['flux'].time.value, lc_detrended[best_model]['flux'].flux.value
    trend = lc_detrended[best_model]['trend'].flux.value

    # remove median from trend (by time splits)
    dt = np.diff(time)
    cut = np.where(dt > break_tolerance * np.nanmedian(dt))[0] + 1
    low = np.append([0], cut)
    high = np.append(cut, len(time))
    for low_i, high_i in zip(low, high):
        trend[low_i: high_i] /= np.nanmedian(trend[low_i: high_i])

    return time, detrended_flux, trend, models_info_df
