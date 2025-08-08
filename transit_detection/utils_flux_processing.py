"""
Utility functions used to preprocess flux time series data.
"""

# 3rd party
import numpy as np
from astropy.stats import mad_std
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import warnings
from scipy.ndimage import gaussian_filter1d

# local
from src_preprocessing.third_party.kepler_spline import kepler_spline
from src_preprocessing.lc_preprocessing.utils_ephemeris import (
    find_first_epoch_after_this_time,
)


def split_timeseries_on_time_gap(time, flux, gap_width):
    """Split timestamps array and flux time series based on `gap_width` days time interval between continuous
    timestamps.

    Args:
        time: NumPy array, timestamps
        flux: NumPy array, flux time series
        gap_width: float, gap width in days. If two contiguous timestamps have a time gap width larger than the gap

    Returns:
        time_arr, list of NumPy arrays of timestamps after splitting based on `gap_width`
        flux_arr, list of NumPy arrays of flux time series after splitting based on `gap_width`

    """

    time_diff = np.diff(time)
    idxs_split = np.where(time_diff >= gap_width)[0] + 1
    time_arr, flux_arr = np.split(time, idxs_split), np.split(flux, idxs_split)

    return time_arr, flux_arr


def extend_transit_mask_edges_by_half_window(
    time, transit_mask, n_durations_window, tce_duration
):
    """Extrend transit mask rising and falling edge values by a half window duration for selecting mid oot window points.

    Args:
        time: NumPy array, timestamps
        transit_mask: NumPy array, a mask with True at idxs corresponding to timestamps that
                        should not be overlapped by a window of n_durations.
    Returns:
        valid_midoot_point_mask: NumPy array, mask with True at idxs corresponding to timestamps for points which
                            serve as valid mid oot points for a window of n_durations

    """

    extended_mask = np.copy(transit_mask).astype(bool)

    eps = 2 / 1440  # Prevent float rounding error by additional cadence

    transit_mask_diff = np.diff(transit_mask)
    transit_edge_idxs = np.where(transit_mask_diff)[0] + 1
    transit_edge_times = time[transit_edge_idxs]

    for edge_time in transit_edge_times:
        extended_mask |= (
            time >= edge_time - (n_durations_window * tce_duration / 2) - eps
        ) & (time <= edge_time + (n_durations_window * tce_duration / 2) + eps)

    return extended_mask


def detrend_flux_using_spline(time_arr, flux_arr, rng):
    """Detrend timeseries by fitting a spline to a version of the timeseries with linear interpolation performed across
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
    spline_flux_arr = kepler_spline.fit_kepler_spline(
        [time_arr],
        [flux_arr],
        bkspace_min=0.5,
        bkspace_max=20,
        bkspace_num=20,
        maxiter=5,
        penalty_coeff=1.0,
        outlier_cut=3,
        verbose=False,
    )[0][0]

    # get indices for which the spline has finite values
    finite_idxs_arr = np.isfinite(spline_flux_arr)

    if finite_idxs_arr.any():  # interpolate spline using linear interpolation
        spline_flux_arr = np.interp(
            time_arr,
            time_arr[finite_idxs_arr],
            spline_flux_arr[finite_idxs_arr],
            left=np.nan,
            right=np.nan,
        )
    else:
        spline_flux_arr = np.nanmedian(flux_arr) * np.ones(
            len(flux_arr)
        )  # spline is set to median flux

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


def interpolate_missing_flux(time, flux):
    """
    Uses linear interpolation to fill missing flux values for cadences in between known values,
    and extrapolation for edge values. Builds cadence and quality mask based on flux with missing cadences.

    Args:
        time: NumPy array, timestamps
        flux: NumPy array, flux values

    Returns:
        filled_flux: NumPy array, flux with missing cadences populated
        cadence_mask, NumPy boolean array corresponding to flux cadences that were interpolated.
        quality_mask, NumPy array of floats (0 <= f <= 1) representing the quality of each cadence.
                    cadences that were not interpolated receive a value of 1, and interpolated values
                    receive a gaussian score using a sigma of 1.0.
    """
    valid_flux_mask = np.isfinite(flux)

    f = interp1d(
        time[valid_flux_mask],
        flux[valid_flux_mask],
        kind="linear",
        bounds_error=False,
        fill_value=(
            # "extrapolate"
            flux[valid_flux_mask][0],
            flux[valid_flux_mask][-1],
        ),  # assume_sorted=True
    )

    flux = f(time)

    cadence_mask = ~valid_flux_mask.astype(bool)

    quality_mask = np.zeros_like(flux).astype(float)
    quality_mask[valid_flux_mask] = 1.0

    quality_mask[~valid_flux_mask] = gaussian_filter1d(
        valid_flux_mask.astype(float), sigma=1.0, mode="reflect"
    )[~valid_flux_mask]

    quality_mask = np.clip(quality_mask, 0.0, 1.0)  # clip [0, 1]

    return flux, cadence_mask, quality_mask


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

    f = interp1d(
        time,
        flux,
        kind="nearest",
        bounds_error=False,
        fill_value="extrapolate",  # assume_sorted=True
    )

    resampled_time = np.linspace(time[0], time[-1], num=num_resampled_points)

    resampled_flux = f(resampled_time)

    return resampled_time, resampled_flux


def resample_timeseries_and_masks(time, flux, masks, num_resampled_points):
    """

    Args:
        time: NumPy array, timestamps
        flux: NumPy array, flux values
        masks: dict, with NumPy arr values, masks corresponding to timestamps
        num_resampled_points: int, number of points to resample

    Returns:
        resampled_time: NumPy array, resampled timestamps
        resampled_flux: NumPy array, resampled flux values
        resampled_masks: dict, with NumPy arr values, resampled masks
    """

    f_flux = interp1d(
        time,
        flux,
        kind="nearest",
        bounds_error=False,
        fill_value="extrapolate",  # assume_sorted=True
    )

    resampled_time = np.linspace(time[0], time[-1], num=num_resampled_points)

    resampled_flux = f_flux(resampled_time)

    resampled_masks = {}
    for k, mask in masks.items():
        f_mask = interp1d(
            time,
            mask,
            kind="nearest",
            bounds_error=False,
            fill_value="extrapolate",  # assume_sorted=True
        )
        resampled_masks[k] = f_mask(resampled_time)

    return resampled_time, resampled_flux, resampled_masks


def resample_time_flux_quality_and_mask(
    time, flux, flux_quality, mask, num_resampled_points
):
    """

    Args:
        time: NumPy array, timestamps
        flux: NumPy array, flux values
        flux_quality: NumPy array, flux quality values
        mask: NumPy arr values, of mask corresponding to timestamps
        num_resampled_points: int, number of points to resample

    Returns:
        resampled_time: NumPy array, resampled timestamps
        resampled_flux: NumPy array, resampled flux values
        resampled_flux_quality: NumPy array, resampled flux_quality values
        resampled_mask: NumPy array, resampled mask
    """

    assert (
        len(time) == len(flux) == len(flux_quality) == len(mask)
    ), f"ERROR: Cannot resample, length mismatch for: time: {time.shape}, flux: {flux.shape}, flux_quality: {flux_quality.shape}, mask: {mask.shape}"

    resampled_time = np.linspace(time[0], time[-1], num=num_resampled_points)

    f_flux = interp1d(
        time,
        flux,
        kind="nearest",
        bounds_error=False,
        fill_value="extrapolate",  # assume_sorted=True
    )

    resampled_flux = f_flux(resampled_time)

    f_flux_quality = interp1d(
        time,
        flux_quality,
        kind="nearest",
        bounds_error=False,
        fill_value="extrapolate",  # assume_sorted=True
    )
    resampled_flux_quality = f_flux_quality(resampled_time)

    f_mask = interp1d(
        time,
        mask,
        kind="nearest",
        bounds_error=False,
        fill_value="extrapolate",  # assume_sorted=True
    )
    resampled_mask = f_mask(resampled_time)

    return resampled_time, resampled_flux, resampled_flux_quality, resampled_mask


def extract_flux_windows_for_tce(
    time,
    flux,
    flux_quality,
    true_it_mask,
    pos_disp_it_mask,
    all_disp_it_mask,
    cadence_mask,
    tce_uid,
    tce_time0bk,
    tce_duration,
    period_days,
    disposition,
    n_durations_window,
    frac_valid_cadences_in_window_thr,
    frac_valid_cadences_it_thr,
    resampled_num_points,
    rng,
    plot_dir=None,
    logger=None,
):
    """Extract flux windows for a TCE based on its orbital period, epoch, and transit duration. These windows are set
    to `n_durations_window` * `tce_duration` and are resampled to `resampled_num_points`. Transit windows are first
    built centered around the midtransit points available in the `time` timestamps array. Out-of-transit windows are
    chosen in equal number to the transit windows by randomly choosing a set of out-of-transit cadences that do not
    overlap with in-transit regions (which are defined as midtransit +- tce_duration).

    Args:
        time: NumPy array, timestamps
        flux: NumPy array, flux values
        flux_quality: NumPy array, quality of flux values
        true_it_mask: NumPy array, informational mask used to flag cadences from TRUE transit events (EB/CP/KP/NPC/NEB)
        pos_disp_it_mask: NumPy array, mask for excluding in_transit_windows for (NTP/NEB/NPC) that overlap with marked cadences from positive dispositions.
        all_disp_it_mask: NumPy array, mask for excluding oot_windows that overlap with marked cadences from any disposition.
        cadence_mask: NumPy array, mask for values which were interpolated. Used for selecting windows based on frac_valid_cadences thresholds.
        tce_uid: str, TCE uid
        tce_time0bk: float,
        tce_duration: float, transit duration in hours
        period_days: float, orbital period in days
        disposition: str, tce disposition
        n_durations_window: int, number of transit durations in the extracted window.
        frac_valid_cadences_in_window_thr: float, fraction of valid cadences in window
        frac_valid_cadences_it_thr: float, fraction of valid in-transit cadences
        resampled_num_points: int, number of points to resample
        rng: NumPy rng generator
        plot_dir: Path, plot directory; set to None to disable plotting

    Returns:
        resampled_flux_it_windows_arr, NumPy array with resampled flux for transit windows
        resampled_flux_oot_windows_arr, NumPy array with resampled flux for out-of-transit windows
        resampled_flux_quality_it_windows_arr, NumPy array with resampled flux quality for transit windows
        resampled_flux_quality_oot_windows_arr, NumPy array with resampled flux quality for out-of-transit windows
        resampled_true_it_mask_it_windows_arr, NumPy array with resampled true transit cadences marked for transit windows
        resampled_true_it_mask_oot_windows_arr, NumPy array with resampled true transit cadences marked for out-of-transit windows
        midtransit_points_windows_arr, list with midtransit points for transit windows
        midoot_points_windows_arr, list with midtransit points for out-of-transit windows
    """

    tce_duration /= 24  # convert from hours to days

    # invalid indices defined by cadence_mask rather than missing values, to keep track of interpolated
    # idxs that may have real values, but we consider them invalid/low quality
    valid_idxs_data = ~cadence_mask

    n_it_windows = 0
    n_oot_windows = 0

    (
        time_it_windows_arr,
        flux_it_windows_arr,
        flux_quality_it_windows_arr,
        true_it_mask_it_windows_arr,
        midtransit_points_windows_arr,
    ) = ([], [], [], [], [])

    (
        time_oot_windows_arr,
        flux_oot_windows_arr,
        flux_quality_oot_windows_arr,
        true_it_mask_oot_windows_arr,
        midoot_points_windows_arr,
    ) = ([], [], [], [], [])

    # extend transit masks to only allow valid window center points
    pos_disp_it_window_mask = extend_transit_mask_edges_by_half_window(
        time, pos_disp_it_mask, n_durations_window, tce_duration
    )
    all_disp_it_window_mask = extend_transit_mask_edges_by_half_window(
        time, all_disp_it_mask, n_durations_window, tce_duration
    )

    # 1) Extract in-transit windows

    # find first midtransit point in the time array
    first_transit_time = find_first_epoch_after_this_time(
        tce_time0bk, period_days, time[0]
    )

    # compute all midtransit points in the time array, (non-deterministic, ie exact value not in arr)
    midtransit_points_arr = np.array(
        [
            first_transit_time + phase_k * period_days
            for phase_k in range(int(np.ceil((time[-1] - time[0]) / period_days)))
        ]
    )

    # delete mtps for neg disps that significantly overlap w/ potential EB/CP/KP in_transit windows
    if disposition in ["NTP", "NEB", "NPC"]:
        exclude_idxs_mtp_arr = []
        for mtp_i, mtp_time in enumerate(midtransit_points_arr):
            closest_time_idx = np.argmin(np.abs(time - mtp_time))
            if pos_disp_it_window_mask[closest_time_idx]:
                exclude_idxs_mtp_arr.append(mtp_i)
        if logger and (len(exclude_idxs_mtp_arr) > 0):
            logger.info(
                f"Excluding {len(exclude_idxs_mtp_arr)} / {len(midtransit_points_arr)} midtransit points for neg disp ({disposition}) tce, based on window overlap with pos disp tce for: {tce_uid}"
            )
        midtransit_points_arr = np.delete(
            np.array(midtransit_points_arr), exclude_idxs_mtp_arr, axis=0
        )

    if logger:
        logger.info(f"Found {len(midtransit_points_arr)} midtransit points.")

    # get start and end timestamps for the windows based on the available midtransit points
    start_time_windows, end_time_windows = (
        midtransit_points_arr - n_durations_window * tce_duration / 2,
        midtransit_points_arr + n_durations_window * tce_duration / 2,
    )

    # choose only those midtransit points whose windows fit completely inside the time array
    valid_window_idxs = np.logical_and(
        start_time_windows >= time[0], end_time_windows <= time[-1]
    )

    valid_midtransit_points = midtransit_points_arr[valid_window_idxs]
    valid_start_time_windows = start_time_windows[valid_window_idxs]
    valid_end_time_windows = end_time_windows[valid_window_idxs]

    if logger:
        logger.info(
            f"Found {len(valid_midtransit_points)} midtransit points whose windows fit completely inside the time array."
        )

    # # TODO: can pad time/flux/mask with nans in case we want to allow for frac valid idxs
    # valid_idxs_data = np.logical_and(np.isfinite(time), np.isfinite(flux))

    # extract transit windows
    for start_time_window, end_time_window, midtransit_point_window in zip(
        valid_start_time_windows, valid_end_time_windows, valid_midtransit_points
    ):

        # find indices in window
        idxs_window = np.logical_and(time >= start_time_window, time <= end_time_window)

        if idxs_window.sum() == 0:
            if logger:
                logger.info(
                    f"No valid indices in window [{start_time_window}, {end_time_window}."
                )
            continue

        # find in-transit indices of window
        idxs_it_window = np.logical_and(
            time >= midtransit_point_window - tce_duration / 2,
            time <= midtransit_point_window + tce_duration / 2,
        )

        if idxs_it_window.sum() == 0:
            if logger:
                logger.info(
                    f"No valid in-transit indices in window [{start_time_window}, {end_time_window}."
                )
            continue

        # check for valid window and in-transit region (+- 0.5td)
        valid_window_flag = (
            (idxs_window & valid_idxs_data).sum() / idxs_window.sum()
        ) > frac_valid_cadences_in_window_thr

        valid_it_window_flag = (
            (idxs_it_window & valid_idxs_data).sum() / idxs_it_window.sum()
        ) > frac_valid_cadences_it_thr

        # NOTE: at this stage, nonfinite values have not been removed
        if valid_window_flag and valid_it_window_flag:
            time_window = time[idxs_window]  # & valid_idxs_data)]
            flux_window = flux[idxs_window]  # & valid_idxs_data)]
            flux_quality_window = flux_quality[idxs_window]
            true_it_mask_window = true_it_mask[idxs_window]
            # masks_window = {k: mask[idxs_window] for k, mask in masks.items()}

            time_it_windows_arr.append(time_window)
            flux_it_windows_arr.append(flux_window)
            flux_quality_it_windows_arr.append(flux_quality_window)
            true_it_mask_it_windows_arr.append(true_it_mask_window)

            midtransit_points_windows_arr.append(midtransit_point_window)

    n_it_windows = len(time_it_windows_arr)

    if logger:
        logger.info(f"Extracted {n_it_windows} transit windows.")

    if n_it_windows == 0:
        raise ValueError(f"No valid transit windows detected for tce: {tce_uid}")

    # get oot candidates for oot windows, that do not fall on other transit events
    midoot_points_arr = time[~all_disp_it_window_mask]
    rng.shuffle(midoot_points_arr)

    if logger:
        logger.info(f"Found {len(midoot_points_arr)} out-of-transit points.")

    start_time_windows, end_time_windows = (
        midoot_points_arr - n_durations_window * tce_duration / 2,
        midoot_points_arr + n_durations_window * tce_duration / 2,
    )

    # choose only those midtransit points whose windows fit completely inside the time array
    valid_window_idxs = np.logical_and(
        start_time_windows >= time[0], end_time_windows <= time[-1]
    )

    valid_midoot_points = midoot_points_arr[valid_window_idxs]
    valid_start_time_windows = start_time_windows[valid_window_idxs]
    valid_end_time_windows = end_time_windows[valid_window_idxs]

    # extract out of transit windows
    for start_time_window, end_time_window, midoot_point_window in zip(
        valid_start_time_windows, valid_end_time_windows, valid_midoot_points
    ):

        idxs_window = np.logical_and(time >= start_time_window, time <= end_time_window)

        if idxs_window.sum() == 0:
            if logger:
                logger.info(
                    f"No valid indices in window [{start_time_window}, {end_time_window}."
                )
            continue

        # check for valid window
        valid_window_flag = (
            (idxs_window & valid_idxs_data).sum() / idxs_window.sum()
        ) > frac_valid_cadences_in_window_thr

        if valid_window_flag:
            time_window = time[idxs_window]  # & valid_idxs_data)]
            flux_window = flux[idxs_window]  # & valid_idxs_data)]
            # masks_window = {k: mask[idxs_window] for k, mask in masks.items()}
            flux_quality_window = flux_quality[idxs_window]
            true_it_mask_window = true_it_mask[idxs_window]

            time_oot_windows_arr.append(time_window)
            flux_oot_windows_arr.append(flux_window)
            flux_quality_oot_windows_arr.append(flux_quality_window)
            true_it_mask_oot_windows_arr.append(true_it_mask_window)

            midoot_points_windows_arr.append(midoot_point_window)

        if len(midoot_points_windows_arr) == n_it_windows:
            break

    n_oot_windows = len(time_oot_windows_arr)

    if logger:
        logger.info(f"Extracted {len(time_oot_windows_arr)} out-of-transit windows.")

    if n_oot_windows < n_it_windows:
        if logger:
            logger.warning(
                f"Number of out-of-transit windows: ({n_oot_windows}) is less than number of transit windows: ({n_it_windows})."
            )
        # warnings.warn(
        #     f"Number of out-of-transit windows ({n_oot_windows}) is less than number of transit windows "
        #     f"({n_it_windows})."
        # )

    # plot transit and oot windows
    if plot_dir:
        f, ax = plt.subplots(figsize=(12, 6))
        for transit_i in range(n_it_windows):
            ax.scatter(
                time_it_windows_arr[transit_i],
                flux_it_windows_arr[transit_i],
                c="g",
                s=8,
                label="Transit Windows" if transit_i == 0 else None,
            )
        for transit_i in range(n_oot_windows):
            ax.scatter(
                time_oot_windows_arr[transit_i],
                flux_oot_windows_arr[transit_i],
                c="b",
                s=8,
                label="Out-of-transit Windows" if transit_i == 0 else None,
            )

        ax.set_xlabel("Time - 2457000 [BTJD days]")
        ax.set_ylabel("Detrended Normalized Flux")
        ax.legend()
        f.tight_layout()
        f.savefig(plot_dir / f"{tce_uid}_it_oot_windows_flux.png")
        plt.close()

    # resample transit windows
    (
        resampled_time_it_windows_arr,
        resampled_flux_it_windows_arr,
        resampled_flux_quality_it_windows_arr,
        resampled_true_it_mask_it_windows_arr,
    ) = ([], [], [], [])

    for (
        time_window,
        flux_window,
        flux_quality_window,
        true_it_mask_window,
        midtransit_point,
    ) in zip(
        time_it_windows_arr,
        flux_it_windows_arr,
        flux_quality_it_windows_arr,
        true_it_mask_it_windows_arr,
        midtransit_points_windows_arr,
    ):
        (
            resampled_time_window,
            resampled_flux_window,
            resampled_flux_quality_window,
            resampled_true_it_mask_window,
        ) = resample_time_flux_quality_and_mask(
            time_window,
            flux_window,
            flux_quality_window,
            true_it_mask_window,
            resampled_num_points,
        )

        # logger.info(
        #     f"IT {tce_uid}_t_{midtransit_point} has max unsampled flux_window {max(flux_window)} and resampled flux_window of {max(resampled_flux_window)}"
        # )

        resampled_time_it_windows_arr.append(resampled_time_window)
        resampled_flux_it_windows_arr.append(resampled_flux_window)
        resampled_flux_quality_it_windows_arr.append(resampled_flux_quality_window)
        resampled_true_it_mask_it_windows_arr.append(resampled_true_it_mask_window)

    # resample out-of-transit windows
    (
        resampled_time_oot_windows_arr,
        resampled_flux_oot_windows_arr,
        resampled_flux_quality_oot_windows_arr,
        resampled_true_it_mask_oot_windows_arr,
    ) = ([], [], [], [])

    for (
        time_window,
        flux_window,
        flux_quality_window,
        true_it_mask_window,
        midoot_point,
    ) in zip(
        time_oot_windows_arr,
        flux_oot_windows_arr,
        flux_quality_oot_windows_arr,
        true_it_mask_oot_windows_arr,
        midoot_points_windows_arr,
    ):
        (
            resampled_time_window,
            resampled_flux_window,
            resampled_flux_quality_window,
            resampled_true_it_mask_window,
        ) = resample_time_flux_quality_and_mask(
            time_window,
            flux_window,
            flux_quality_window,
            true_it_mask_window,
            resampled_num_points,
        )
        # logger.info(
        #     f"OOT {tce_uid}_t_{midoot_point} has max unsampled flux_window {max(flux_window)} and resampled flux_window of {max(resampled_flux_window)}"
        # )

        resampled_time_oot_windows_arr.append(resampled_time_window)
        resampled_flux_oot_windows_arr.append(resampled_flux_window)
        resampled_flux_quality_oot_windows_arr.append(resampled_flux_quality_window)
        resampled_true_it_mask_oot_windows_arr.append(resampled_true_it_mask_window)

    if plot_dir:
        f, ax = plt.subplots(figsize=(12, 6))
        for transit_i in range(n_it_windows):
            ax.scatter(
                resampled_time_it_windows_arr[transit_i],
                resampled_flux_it_windows_arr[transit_i],
                c="g",
                s=8,
                label="Transit Windows" if transit_i == 0 else None,
            )
        for transit_i in range(n_oot_windows):
            ax.scatter(
                resampled_time_oot_windows_arr[transit_i],
                resampled_flux_oot_windows_arr[transit_i],
                c="b",
                s=8,
                label="Out-of-transit Windows" if transit_i == 0 else None,
            )
        ax.set_xlabel("Time - 2457000 [BTJD days]")
        ax.set_ylabel("Detrended Normalized Flux")
        ax.legend()
        f.suptitle("Resampled Windows")
        f.tight_layout()
        f.savefig(plot_dir / f"{tce_uid}_it_oot_windows_flux_resampled.png")
        plt.close()

    return (
        resampled_flux_it_windows_arr,
        resampled_flux_oot_windows_arr,
        resampled_flux_quality_it_windows_arr,
        resampled_flux_quality_oot_windows_arr,
        resampled_true_it_mask_it_windows_arr,
        resampled_true_it_mask_oot_windows_arr,
        midtransit_points_windows_arr,
        midoot_points_windows_arr,
    )


# def plot_detrended_flux_time_series_sg(
#     time, flux, detrend_time, detrend_flux, trend, sector, plot_dir=None
# ):
#     """
#     Builds plot for a given time series that was detrended using savitzky golay. Builds three plots over time:
#     1. Raw Flux
#     2. Detrended Normalized Flux
#     3. Trend against Detrended Normalized Flux

#     Args:
#         time: NumPy array, timestamps
#         flux: NumPy array, flux time series
#         detrend_time: NumPy array, timestamps associated with detrended flux
#         detrend_flux: NumPy array, flux time series detrended with savitzky golay
#         trend: NumPy array, trend returned by savitzky golay detrending.
#         sector: String, of an integer, for the sector corresponding to the time series
#         plot_dir: Path, directory to create plot

#     Returns:
#         None
#     """

#     if plot_dir:
#         f, ax = plt.subplots(3, 1, figsize=(12, 8))
#         ax[0].scatter(time, flux, s=8, label="PDCSAP Flux")
#         ax[0].set_ylabel("PDCSAP Flux [e-/cadence]")
#         ax[1].scatter(detrend_time, detrend_flux, s=8)
#         ax[1].set_ylabel("Detrended Normalized Flux")
#         ax[2].scatter(detrend_time, detrend_flux, s=8, label="Detrended")
#         ax[2].scatter(detrend_time, trend, s=8)
#         ax[2].set_ylabel("Flux Trend [e-/cadence]")
#         ax[2].set_xlabel("Time - 2457000 [BTJD days]")
#         f.savefig(plot_dir / f"sector_{sector}_detrended_flux.png")
#         plt.close()


def build_transit_mask_for_lightcurve(
    time,
    tce_list,
    n_durations_window: int = 3,
    maxmes_threshold: float = 7.1,
):
    """
    Generates in transit mask mask, using both primary and potential weak secondary transit information, for a given light
    curve time array based on a given list of TCEs. A mask the same size of the time value array is created for which given
    elements are set to 'True' for in-transit timestamps and 'False' for out of transit ones, based on the orbital period,
    epoch, and transit duration of the TCEs provided.

    Args:
        time: NumPy array, timestamps
        tce_list: List, containing dict of ephemerides data for all tces used.
        n_durations_window: int, number of td to use for window size mask per tce
        maxmes_threshold: float, minimum maxmes score of tce to use weak secondary transits cadences in mask

    Returns:
        in_transit_mask, NumPy array with boolean elements for in-transit (True) and out-of-transit (False) timestamps.
    """
    in_transit_mask = np.zeros(len(time), dtype=bool)

    for tce in tce_list:
        epoch = tce["tce_time0bk"]
        period = tce["tce_period"]
        duration = tce["tce_duration"]
        maxmes = tce["tce_maxmes"]
        maxmesd = tce["tce_maxmesd"]

        duration /= 24  # convert hours to days

        # distance to nearest transit center
        transit_proximity_days = ((time - epoch + (period / 2)) % period) - (period / 2)
        in_transit_mask |= np.abs(transit_proximity_days) <= (
            ((n_durations_window * duration) / 2)
        )  # mask for primary transit

        if maxmes > maxmes_threshold:
            sec_epoch = epoch + maxmesd
            # distance to nearest potential secondary transit center
            sec_transit_proximity_days = (
                (time - sec_epoch + (period / 2)) % period
            ) - (period / 2)
            in_transit_mask |= np.abs(sec_transit_proximity_days) <= (
                ((n_durations_window * duration) / 2)
            )  # mask for potential secondary transit

    return in_transit_mask


def build_primary_transit_mask_for_lightcurve(
    time,
    tce_list,
    n_durations_window: int = 3,
):
    """
    Generates primary in transit mask, for a given light curve time array based on a given list of TCEs. A mask the same size of the
    time value array is created for which given elements are set to 'True' for in-transit timestamps and 'False' for out of
    transit ones, based on the orbital period, epoch, and transit duration of the TCEs provided.

    Args:
        time: NumPy array, timestamps
        tce_list: List, containing dict of ephemerides data for all tces used.
        n_durations_window: int, number of td to use for window size mask per tce

    Returns:
        in_transit_mask, NumPy array with boolean elements for in-transit (True) and out-of-transit (False) timestamps.
    """
    in_transit_mask = np.zeros(len(time), dtype=bool)

    for tce in tce_list:
        epoch = tce["tce_time0bk"]
        period = tce["tce_period"]
        duration = tce["tce_duration"]

        duration /= 24  # convert hours to days

        # distance to nearest transit center (days)
        transit_proximity_days = ((time - epoch + (period / 2)) % period) - (period / 2)
        in_transit_mask |= np.abs(transit_proximity_days) <= (
            ((n_durations_window * duration) / 2)
        )

    return in_transit_mask


def build_secondary_transit_mask_for_lightcurve(
    time,
    tce_list,
    n_durations_window: int = 3,
    maxmes_threshold: float = 7.1,
):
    """
    Generates weak secondary in transit mask for a given light curve time array based on a given list of TCEs, whose
    tce_maxmes > provided maxmes_threshold. A mask the same size of the time value array is created for which given
    elements are set to 'True' for potential in-transit timestamps and 'False' for out of transit ones, based on the
    orbital period, epoch, transit duration, and tce_maxmesd of the TCEs provided.

    Args:
        time: NumPy array, timestamps
        tce_list: List, containing dict of ephemerides data for all tces used.
                            **Epoch should be the midpoint time for a transit
        n_durations_window: int, number of td to use for window size per tce
        maxmes_threshold: float, minimum maxmes score of tce to consider weak secondary transits in mask

    Returns:
        in_transit_mask, NumPy array with boolean elements for in-transit (True) and out-of-transit (False) timestamps.
    """

    sec_in_transit_mask = np.zeros(len(time), dtype=bool)

    for tce in tce_list:
        epoch = tce["tce_time0bk"]
        period = tce["tce_period"]
        duration = tce["tce_duration"]
        maxmes = tce["tce_maxmes"]
        maxmesd = tce["tce_maxmesd"]

        duration /= 24  # convert hours to days

        if maxmes > maxmes_threshold:
            sec_epoch = epoch + maxmesd
            sec_transit_proximity_days = (
                (time - sec_epoch + (period / 2)) % period
            ) - (period / 2)

            sec_in_transit_mask |= np.abs(sec_transit_proximity_days) <= (
                ((n_durations_window * duration) / 2)
            )

    return sec_in_transit_mask


def plot_interpolated_flux_mask(
    time,
    flux,
    cadence_mask,
    flux_quality,
    sector=None,
    plot_dir=None,
    figsize=(10, 4),
):
    """
    Show the interpolated flux with shaded regions for both
    interpolation gaps and low-quality intervals using fill_between.

    Single-panel plot:
      • Solid blue line: filled flux
      • Green fill: interpolation mask
      • Red fill: low-quality cadences (where quality<1)
    """
    plt.style.use("default")

    mask_max_alpha = 0.4
    palette = {
        "flux": "#0a3553",  # blue
        "interp_mask": "#a5f369",  # green
        "quality": "#e24e4e",  # red
    }

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.set_facecolor("white")

    # 1) Shade interpolation gaps under everything
    ymin, ymax = np.nanmin(flux), np.nanmax(flux)
    ax.fill_between(
        time,
        ymin,
        ymax,
        where=cadence_mask,
        color=palette["interp_mask"],
        alpha=0.3,
        label="Interp Mask",
        zorder=1,
    )

    # 2) Shade low-quality cadences under the line
    lowq = flux_quality < 1.0
    if np.any(lowq):
        ax.fill_between(
            time,
            ymin,
            ymax,
            where=lowq,
            color=palette["quality"],
            alpha=mask_max_alpha,
            label="Low Quality",
            zorder=2,
        )

    # 3) Plot the flux on top
    ax.plot(time, flux, color=palette["flux"], lw=1.5, label="Flux", zorder=3)

    # 4) Construct a clear legend
    handles = [
        Line2D([0], [0], color=palette["flux"], lw=1.5, label="Flux"),
        Patch(facecolor=palette["interp_mask"], alpha=0.3, label="Interp Mask"),
        Patch(facecolor=palette["quality"], alpha=mask_max_alpha, label="Low Quality"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize="small", frameon=False)

    ax.set_xlabel("Time [BTJD days]")
    ax.set_ylabel("Flux [e⁻/cadence]")
    ax.set_title("Interpolated Flux & Quality")

    if sector is not None:
        fig.suptitle(f"Sector {sector}", y=1.02)

    fig.tight_layout()
    if plot_dir is not None:
        plot_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(plot_dir / f"sector_{sector}_interpolated_mask.png", dpi=150)
        plt.close(fig)
    else:
        plt.show()


def plot_flux_diagnostics_sg(
    time,
    flux,
    detrend_time,
    detrended_flux,
    trend,
    detrend_mask,
    quality_mask,
    sector=None,
    plot_dir=None,
    figsize=(16, 12),
):
    """
    Eight-panel diagnostics of raw vs detrended light curves using fill_between:

    Panels (4×2):
      0) Raw + low-quality + detrend-mask
      1) Raw + detrend-mask only
      2) Detrended only
      3) Detrended + trend line
      4) Detrended + detrend-mask only
      5) Detrended + low-quality only
      6) Raw summary: both masks
      7) Detrended summary: both masks
    """
    plt.style.use("default")
    mask_max_alpha = 0.4
    palette = {
        "flux": "#0a3553",  # blue
        "trend": "#ff963a",  # orange
        "detrend_mask": "#8edd52",  # green
        "quality": "#e24e4e",  # red
    }

    fig, axes = plt.subplots(4, 2, figsize=figsize)
    axes = axes.flatten()
    ymin_raw, ymax_raw = np.min(flux), np.max(flux)
    ymin_det, ymax_det = np.min(detrended_flux), np.max(detrended_flux)

    def add_masks(ax, x, y0, y1, mask, color, alpha, label):
        """Fill under mask with given color/alpha."""
        ax.fill_between(
            x, y0, y1, where=mask, color=color, alpha=alpha, label=label, zorder=1
        )

    # Panel 0: Raw + quality + detrend-mask
    ax = axes[0]
    add_masks(
        ax,
        time,
        ymin_raw,
        ymax_raw,
        detrend_mask,
        palette["detrend_mask"],
        0.3,
        "Detrend Mask",
    )
    add_masks(
        ax,
        time,
        ymin_raw,
        ymax_raw,
        quality_mask < 1.0,
        palette["quality"],
        mask_max_alpha,
        "Low Quality",
    )
    ax.plot(time, flux, color=palette["flux"], lw=1.5, zorder=3)
    ax.set_title("Raw: Quality & Detrend Mask")
    ax.set_ylabel("Flux")

    # Panel 1: Raw + detrend-mask only
    ax = axes[1]
    add_masks(
        ax,
        time,
        ymin_raw,
        ymax_raw,
        detrend_mask,
        palette["detrend_mask"],
        0.3,
        "Detrend Mask",
    )
    ax.plot(time, flux, color=palette["flux"], lw=1.5, zorder=3)
    ax.set_title("Raw: Detrend Mask")

    # Panel 2: Detrended only
    ax = axes[2]
    ax.plot(detrend_time, detrended_flux, color=palette["flux"], lw=1.5, zorder=3)
    ax.set_title("Detrended Flux")
    ax.set_ylabel("Flux")

    # Panel 3: Detrended + trend line
    ax = axes[3]
    ax.plot(detrend_time, detrended_flux, color=palette["flux"], lw=1.5, zorder=3)
    ax.plot(detrend_time, trend, color=palette["trend"], lw=1.5, zorder=4)
    ax.set_title("Detrended & Trend")

    # Panel 4: Detrended + detrend-mask only
    ax = axes[4]
    add_masks(
        ax,
        detrend_time,
        ymin_det,
        ymax_det,
        detrend_mask,
        palette["detrend_mask"],
        0.3,
        "Detrend Mask",
    )
    ax.plot(detrend_time, detrended_flux, color=palette["flux"], lw=1.5, zorder=3)
    ax.set_title("Detrended: Detrend Mask")
    ax.set_ylabel("Flux")

    # Panel 5: Detrended + low-quality only
    ax = axes[5]
    add_masks(
        ax,
        detrend_time,
        ymin_det,
        ymax_det,
        quality_mask < 1.0,
        palette["quality"],
        mask_max_alpha,
        "Low Quality",
    )
    ax.plot(detrend_time, detrended_flux, color=palette["flux"], lw=1.5, zorder=3)
    ax.set_title("Detrended: Quality Mask")

    # Panel 6: Raw summary both masks
    ax = axes[6]
    add_masks(
        ax,
        time,
        ymin_raw,
        ymax_raw,
        detrend_mask,
        palette["detrend_mask"],
        0.3,
        "Detrend Mask",
    )
    add_masks(
        ax,
        time,
        ymin_raw,
        ymax_raw,
        quality_mask < 1.0,
        palette["quality"],
        mask_max_alpha,
        "Low Quality",
    )
    ax.plot(time, flux, color=palette["flux"], lw=1.5, zorder=3, label="Flux")
    ax.set_title("Raw Summary")
    ax.set_xlabel("Time [BTJD days]")
    ax.legend(loc="upper right", fontsize="small", frameon=False)

    # Panel 7: Detrended summary both masks
    ax = axes[7]
    add_masks(
        ax,
        detrend_time,
        ymin_det,
        ymax_det,
        detrend_mask,
        palette["detrend_mask"],
        0.3,
        "Detrend Mask",
    )
    add_masks(
        ax,
        detrend_time,
        ymin_det,
        ymax_det,
        quality_mask < 1.0,
        palette["quality"],
        mask_max_alpha,
        "Low Quality",
    )
    ax.plot(
        detrend_time,
        detrended_flux,
        color=palette["flux"],
        lw=1.5,
        zorder=3,
        label="Flux",
    )
    ax.set_title("Detrended Summary")
    ax.set_xlabel("Time [BTJD days]")
    ax.legend(loc="upper right", fontsize="small", frameon=False)

    if sector is not None:
        fig.suptitle(f"Sector {sector}", y=0.98)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    if plot_dir is not None:
        plot_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(plot_dir / f"sector_{sector}_flux_diagnostics.png", dpi=150)
        plt.close(fig)
    else:
        plt.show()
