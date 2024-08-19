"""
Set of functions designed to compute and manipulate periodograms built from light curve time series.
"""

# 3rd party
import numpy as np
import lightkurve as lk
from lightkurve import periodogram as lk_periodogram
import matplotlib.pyplot as plt
from astropy import units as u
import logging

# local
from src_preprocessing.lc_preprocessing.phase_fold_and_binning import local_view
from src_preprocessing.lc_preprocessing.utils_ephemeris import (create_binary_time_series,
                                                                find_first_epoch_after_this_time)


logger = logging.getLogger(__name__)


def compute_frequency_range(p_min_tce, k_harmonics, p_max_obs, downsampling_f):
    """ Sets the frequency values for which to compute the periodogram.

    Args:
        p_min_tce: float, shortest period
        k_harmonics: int, number of harmonics. This determines the highest frequency for which to estimate the
            periodogram (f_max = k_harmonics / p_min_tce).
        p_max_obs: float, longest period for which to compute the periodogram.
        downsampling_f: float, downsampling factor for the frequency range

    Returns: NumPy array, frequency values for which to compute the periodogram

    """

    # compute fmax
    f_max = (1 / p_min_tce) * k_harmonics  # 1/days
    # compute fmin
    f_min = 1 / p_max_obs  # 1/days
    # number of points in frequency range
    n_freq_pts = int((f_max - f_min) / (downsampling_f * f_min))
    # get range between fmin and fmax
    f_arr = np.linspace(f_min, f_max, n_freq_pts, endpoint=True)

    return f_arr


def get_lc_data_into_lk_curve(time_arrs, flux_arrs, flux_err_arrs):
    """ Creates lightkurve lightcurve object from time series and flux arrays. Stitches the multiple flux time series
    and removes NaN flux values.

    Args:
        time_arrs: list of NumPy arrays, timestamps
        flux_arrs: list of NumPy arrays, flux values
        flux_err_arrs: list of NumPy arrays, flux error values

    Returns: lightkurve lightcurve object

    """

    # put all light curve arrays into a collection to then stitch them together
    lc_collection = lk.LightCurveCollection([lk.LightCurve(data={'time': np.array(time_arr),
                                                                 'flux': flux_arr,
                                                                 'flux_err': flux_err_arr})
                                             for time_arr, flux_arr, flux_err_arr
                                             in zip(time_arrs, flux_arrs, flux_err_arrs)])

    lcf = lc_collection.stitch()

    # remove NaN cadences
    lcf = lcf.remove_nans()

    return lcf


def compute_lc_transit_pulse_model(lc_data, tce_data, n_durations):
    """ Computes transit pulse model based on TCE ephemerides and transit depth.

    Args:
        lc_data: lightkurve lightcurve object with raw light curve data
        tce_data: pandas Series, TCE parameters
        n_durations: number of TCE transit durations to use when building the transit pulse model

    Returns: lightkurve lightcurve object for the transit pulse model
    """

    flux_med = np.array(np.nanmedian(lc_data.flux.value))  # median oot flux

    # create lightcurve transit pulse model
    ts, te = lc_data.time.value[0], lc_data.time.value[-1]
    first_transit_time_tpm = find_first_epoch_after_this_time(tce_data['tce_time0bk'], tce_data['tce_period'], ts)
    midtransit_points_arr_tpm = np.array([first_transit_time_tpm + phase_k * tce_data['tce_period']
                                          for phase_k in range(int(np.ceil((te - ts) / tce_data['tce_period'])))])
    tce_intransit_cadences_arr = create_binary_time_series(lc_data.time.value,
                                                           midtransit_points_arr_tpm[0],
                                                           tce_data['tce_duration'] / 24 * n_durations,
                                                           tce_data['tce_period'])
    tce_intransit_cadences_sec_arr = create_binary_time_series(lc_data.time.value,
                                                               midtransit_points_arr_tpm[0] + tce_data['tce_maxmesd'],
                                                               tce_data['tce_duration'] / 24 * n_durations,
                                                               tce_data['tce_period'])
    tce_transit_pulse = np.array(flux_med * np.ones_like(lc_data.time.value))
    tce_transit_pulse_err = np.zeros_like(lc_data.time.value)

    # estimate transit depth from folded and binned time series when value is not available from DV
    if tce_data['tce_depth'] == 0 or tce_data['tce_depth_err'] == -1:
        logger.info(f'Setting transit depth based on data for TCE TIC {tce_data["tce_uid"]}')

        lcf_folded = lc_data.fold(period=tce_data['tce_period'], epoch_time=tce_data['tce_time0bk'])

        loc_flux_view, _, loc_flux_view_var, _, bin_counts = \
            local_view(np.array(lcf_folded.time.value),
                       np.array(lcf_folded.flux.value),
                       tce_data['tce_period'],
                       tce_data['tce_duration'],
                       tce=tce_data,
                       normalize=False,
                       centering=False,
                       num_durations=n_durations / 2,
                       num_bins=31,
                       bin_width_factor=0.16,
                       report={'config': {'exclusion_logs_dir': None}, 'tce': tce_data, 'view': 'local_flux_view'}
                       )
        bin_counts[bin_counts == 0] = max(1, np.median(bin_counts))
        loc_flux_view_var /= np.sqrt(bin_counts)

        min_idx = loc_flux_view.argmin()
        tce_depth, tce_depth_err = flux_med - loc_flux_view[min_idx], loc_flux_view_var[min_idx]

    else:
        tce_depth = tce_data['tce_depth'] * 1e-6
        tce_depth_err = tce_data['tce_depth_err'] * 1e-6

    # do the same for the secondary
    if tce_data['wst_depth'] == 0 or tce_data['wst_depth_err'] == -1:
        logger.info(f'Setting secondary transit depth based on data for TCE TIC {tce_data["tce_uid"]}')

        lcf_folded = lc_data.fold(period=tce_data['tce_period'],
                                  epoch_time=tce_data['tce_time0bk'] + tce_data['tce_maxmesd'])

        loc_flux_view, _, loc_flux_view_var, _, bin_counts = \
            local_view(np.array(lcf_folded.time.value),
                       np.array(lcf_folded.flux.value),
                       tce_data['tce_period'],
                       tce_data['tce_duration'],
                       tce=tce_data,
                       normalize=False,
                       centering=False,
                       num_durations=n_durations / 2,
                       num_bins=31,
                       bin_width_factor=0.16,
                       report={'config': {'exclusion_logs_dir': None}, 'tce': tce_data, 'view': 'local_flux_view'}
                       )
        bin_counts[bin_counts == 0] = max(1, np.median(bin_counts))
        loc_flux_view_var /= np.sqrt(bin_counts)

        min_idx = loc_flux_view.argmin()
        tce_sec_depth, tce_sec_depth_err = flux_med - loc_flux_view[min_idx], loc_flux_view_var[min_idx]

    else:
        tce_sec_depth = tce_data['wst_depth'] * 1e-6
        tce_sec_depth_err = tce_data['wst_depth_err'] * 1e-6

    tce_transit_pulse[tce_intransit_cadences_arr] = tce_transit_pulse[tce_intransit_cadences_arr] - tce_depth
    tce_transit_pulse[tce_intransit_cadences_sec_arr] = tce_transit_pulse[
                                                            tce_intransit_cadences_sec_arr] - tce_sec_depth

    tce_transit_pulse_err[tce_intransit_cadences_arr] = tce_depth_err
    tce_transit_pulse_err[tce_intransit_cadences_sec_arr] = tce_sec_depth_err

    lc_tpm = lk.LightCurve(data={'time': np.array(lc_data.time.value),
                                 'flux': tce_transit_pulse,
                                 'flux_err': tce_transit_pulse_err})

    return lc_tpm


def plot_periodogram(tce_data, save_fp, lc_data, lc_tpm_data, pgram_res, n_harmonics=5):
    """ Creates figure with plots of 1) the raw flux time series and transit pulse model, 2) the periodograms for both
    time series (not-smoothed and smoothed versions), and 3) the corresponding normalized periodograms.

    Args:
        tce_data: pandas Series, TCE parameters
        save_fp: Path, save filepath
        lc_data: lightkurve LightCurve object with raw light curve data
        lc_tpm_data: lightkurve LightCurve object with transit pulse model data
        pgram_res: dict that maps to different computed periodograms (e.g., whether for the lc or transit pulse model
        data, whether smoothed, and whether normalized)
        n_harmonics: int, number of harmonics to display

    Returns:
    """

    f0_tce = 1 / (tce_data['tce_period'] * 24 * 3600)  # in 1/s

    f, ax = plt.subplots(3, 1, figsize=(14, 10))

    ax[0].scatter(lc_data.time.value, lc_data.flux.value, s=6, color='b')

    ax[0].plot(lc_tpm_data.time.value, lc_tpm_data.flux.value, linestyle='--', label='Transit Pulse Model (TPM)',
               color='tab:orange')

    ax[0].legend()
    ax[0].set_xlabel('Time [day]')
    ax[0].set_ylabel(r'PDCSAP Flux [$e^-s^{-1}$]')
    ax[0].set_xlim(lc_tpm_data.time.value[[0, -1]])

    # plot periodogram of the data
    ax[1].plot(pgram_res['pgram'].frequency.to(1 / u.s), pgram_res['pgram'].power, zorder=3, color='b', label='Data')
    ax[1].plot(pgram_res['pgram_smooth'].frequency.to(1 / u.s), pgram_res['pgram_smooth'].power, zorder=3,
               linestyle='--', color='c', alpha=0.5, label='Data Smoothed')
    # add vertical line for frequency at maximum power
    # ax[1].axvline(x=pgram_res['pgram'].frequency_at_max_power.to(1 / u.s).value, c='m', linestyle='-',
    #               label=None, zorder=2)

    # add vertical lines for f0 and harmonics of the TCE
    # ax[1].axvline(x=f0_tce, c='k', linestyle='-.',
    #               label=fr'$f_0={f0_tce:.3e} /s$', zorder=2, linewidth=2)
    for harmonic_i in range(1, n_harmonics + 1):
        ax[1].axvline(x=f0_tce * harmonic_i, c='r', linestyle='--',
                      label='Harmonics' if harmonic_i == n_harmonics + 1 else None, zorder=2, linewidth=2, alpha=0.5)

    # plot periodograms of TPM model
    ax[1].plot(pgram_res['pgram_tpm'].frequency.to(1 / u.s), pgram_res['pgram_tpm'].power, linestyle='-',
               zorder=3, alpha=1, label='TPM', color='tab:orange')
    ax[1].plot(pgram_res['pgram_tpm_smooth'].frequency.to(1 / u.s), pgram_res['pgram_tpm_smooth'].power, linestyle='--',
               zorder=3, alpha=0.5, label='TPM Smoothed', color='k')
    # # add vertical line for frequency at maximum power for TPM model
    # ax[1].axvline(x=pgram_res['pgram_tpm_smooth'].frequency_at_max_power.to(1 / u.s).value, c='k', linestyle='-',
    #               label=None,
    #               zorder=2)

    ax[1].legend()
    ax[1].set_ylabel('Amplitude')
    ax[1].set_xlabel('Frequency [Hz]')
    ax[1].set_xlim(pgram_res['pgram'].frequency.to(1 / u.s).value[[0, -1]])
    ax[1].set_yscale('log')
    ax[1].set_xscale('log')

    ax[1].set_title(fr'Peak Amplitude @ '  
                    fr'$f_{{Data, max}}={pgram_res["pgram"].frequency_at_max_power.to(1 / u.s):.3e}$:' 
                    fr'$={pgram_res["pgram"].max_power:.3e}$ | TPM Peak Amplitude @ ' 
                    fr'$f_{{TPM, max}}={pgram_res["pgram_tpm"].frequency_at_max_power.to(1 / u.s):.3e}$:' 
                    fr'${pgram_res["pgram_tpm"].max_power:.3e}$')

    ax[2].plot(pgram_res['pgram_norm'].frequency.to(1 / u.s), pgram_res['pgram_norm'].power, zorder=2, linestyle='-',
               color='b', label='Data')
    ax[2].plot(pgram_res['pgram_smooth_norm'].frequency.to(1 / u.s), pgram_res['pgram_smooth_norm'].power, zorder=3,
               linestyle='--', color='c',
               alpha=0.5, label='Data Smoothed')

    ax[2].plot(pgram_res['pgram_tpm_norm'].frequency.to(1 / u.s), pgram_res['pgram_tpm_norm'].power, 'tab:orange',
               label='TPM', alpha=0.5,
               linestyle='-', zorder=2)
    ax[2].plot(pgram_res['pgram_tpm_smooth_norm'].frequency.to(1 / u.s), pgram_res['pgram_tpm_smooth_norm'].power,
               'k', label='TPM Smoothed', alpha=0.5, linestyle='dashed', zorder=3)

    ax[2].legend()
    ax[2].set_ylabel('Normalized Amplitude')
    ax[2].set_xlabel('Frequency [Hz]')
    ax[2].set_xlim(pgram_res['pgram_norm'].frequency.to(1 / u.s).value[[0, -1]])
    ax[2].set_yscale('log')
    ax[2].set_xscale('log')

    f.suptitle(fr'{tce_data["uid"]} {tce_data["label"]}' 
               fr'Period: {tce_data["tce_period"]:.3f} day | $f_0={f0_tce:.3e} /s$')
    f.tight_layout()
    f.savefig(save_fp)
    plt.close()


def compute_lc_periodogram(f_arr, smooth_filter_type, smooth_filter_w_f, lc_data, lc_tpm_data):
    """ Computes the periodogram for the lc data.

    Args:
        f_arr: NumPy array, frequency values for which to estimate the power spectra
        smooth_filter_type: str, smoothing kernel type. Either 'boxkernel' or 'logmedian'
        smooth_filter_w_f: int, smoothing kernel width factor
        lc_data: lightkurve LightCurve object with raw light curve data
        lc_tpm_data: lightkurve LightCurve object with transit pulse model data

    Returns:
        pgram_res: dict that maps to different computed periodograms (e.g., whether for the lc or transit pulse model
        data, whether smoothed, and whether normalized)
    """

    if smooth_filter_type == 'boxkernel':
        smooth_filter_w = np.diff(f_arr)[0] * smooth_filter_w_f
    elif smooth_filter_type == 'logmedian':
        smooth_filter_w = np.diff(np.log10(f_arr))[0] * smooth_filter_w_f
    else:
        raise NotImplementedError(f"Smooth filter type {smooth_filter_type} not implemented.")

    # compute periodogram from data
    pgram = lk_periodogram.LombScarglePeriodogram.from_lightcurve(lc_data,
                                                                  frequency=f_arr,
                                                                  normalization='amplitude',
                                                                  ls_method='fast')
    # compute periodogram from transit model
    pgram_tpm = lk_periodogram.LombScarglePeriodogram.from_lightcurve(lc_tpm_data,
                                                                      frequency=f_arr,
                                                                      normalization='amplitude',
                                                                      ls_method='fast')

    # smooth periodogram
    pgram_smooth = pgram.smooth(method=smooth_filter_type, filter_width=smooth_filter_w)
    pgram_tpm_smooth = pgram_tpm.smooth(method=smooth_filter_type, filter_width=smooth_filter_w)

    # normalize periodogram
    pgram_norm = pgram / pgram.max_power
    pgram_tpm_norm = pgram_tpm / pgram_tpm.max_power

    pgram_smooth_norm = pgram_smooth / pgram_smooth.max_power
    pgram_tpm_smooth_norm = pgram_tpm_smooth / pgram_tpm_smooth.max_power

    pgram_res = {
        'pgram': pgram,
        'pgram_norm': pgram_norm,

        'pgram_smooth': pgram_smooth,
        'pgram_smooth_norm': pgram_smooth_norm,

        'pgram_tpm': pgram_tpm,
        'pgram_tpm_norm': pgram_tpm_norm,

        'pgram_tpm_smooth': pgram_tpm_smooth,
        'pgram_tpm_smooth_norm': pgram_tpm_smooth_norm,

    }

    return pgram_res


def lc_periodogram_pipeline(p_min_tce, k_harmonics, p_max_obs, downsampling_f, smooth_filter_type, smooth_filter_w_f,
                            n_durations, tce_data, time_arrs, flux_arrs, flux_err_arrs, save_fp=None,
                            plot_preprocessing_tce=False):
    """ Preprocessing pipeline for computing lc periodogram data.

    Args:
        p_min_tce: float, shortest period
        k_harmonics: int, number of harmonics. This determines the highest frequency for which to estimate the
            periodogram (f_max = k_harmonics / p_min_tce).
        p_max_obs: float, longest period for which to compute the periodogram.
        downsampling_f: float, downsampling factor for the frequency range
        smooth_filter_type: str, smoothing kernel type. Either 'boxkernel' or 'logmedian'
        smooth_filter_w_f: int, smoothing kernel width factor
        n_durations: number of TCE transit durations to use when building the transit pulse model
        tce_data: pandas Series, TCE parameters
        time_arrs: list of NumPy arrays, timestamps
        flux_arrs: list of NumPy arrays, flux values
        flux_err_arrs: list of NumPy arrays, flux error values
        save_fp: Path, save filepath
        plot_preprocessing_tce: bool, if True it saves the figure

    Returns:
        pgram_res: dict that maps to different computed periodograms (e.g., whether for the lc or transit pulse model
        data, whether smoothed, and whether normalized)
    """

    lc_data = get_lc_data_into_lk_curve(time_arrs, flux_arrs, flux_err_arrs)

    lc_tpm_data = compute_lc_transit_pulse_model(lc_data, tce_data, n_durations)

    f_arr = compute_frequency_range(p_min_tce, k_harmonics, p_max_obs, downsampling_f)

    pgram_res = compute_lc_periodogram(f_arr, smooth_filter_type, smooth_filter_w_f, lc_data, lc_tpm_data)

    if plot_preprocessing_tce:
        plot_periodogram(tce_data, save_fp, lc_data, lc_tpm_data, pgram_res, n_harmonics=5)

    # compute periodogram with a downsampled frequency range
    f_arr = compute_frequency_range(p_min_tce, k_harmonics, p_max_obs, downsampling_f / 2)
    pgram_res_downsampled = compute_lc_periodogram(f_arr, smooth_filter_type, smooth_filter_w_f, lc_data, lc_tpm_data)
    pgram_res.update({f'{pgram_name}_downsampled': pgram_data
                      for pgram_name, pgram_data in pgram_res_downsampled.items()})

    return pgram_res
