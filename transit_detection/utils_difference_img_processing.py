"""
Utility functions to get difference image data for the selected time window.
"""

# 3rd party
from astropy.io import fits
import numpy as np
from astropy.wcs import WCS
import matplotlib.pyplot as plt
import warnings

# local
from src_preprocessing.lc_preprocessing.utils_ephemeris import find_first_epoch_after_this_time
from src_preprocessing.lc_preprocessing.utils_ephemeris import create_binary_time_series
from src_preprocessing.diff_img.extracting.utils_diff_img import plot_diff_img_data as plot_diff_img_data_extracted
from src_preprocessing.diff_img.preprocessing.preprocess_diff_img import (preprocess_single_diff_img_data_for_example,
                                                                          plot_diff_img_data)

ELECTRONS_PER_S_TO_CADENCE = 96


def get_diff_img_data_for_window_with_ephemerides(fits_file_path, t_window_edges, dur_hours, period_days, tce_time0bk,
                                                  transit_depth_ppm, buffer=30, thr_transit_depth=0.75, plot_dir=None):
    """ Computes difference, in-transit, out-of-transit, and SNR images from flux data and based on ephemerides
    information (epoch, period, and transit duration) and for a time window specfied by `t_window_edges`.

    Args:
        fits_file_path: str, path to target pixel fits file
        dur_hours: float, transit duration in hours
        t_window_edges: list, start and end timestamps of the window
        period_days: float, orbital period in days
        tce_time0bk: float, epoch in BTJD (day)
        transit_depth_ppm: float, transit depth in ppm
        buffer: float, buffer size in minutes between in-transit and out-transit windows
        thr_transit_depth: float, transit depth in ppm to threshold in-transit cadences
        plot_dir: Path, path to directory to save plots

    Returns:
        diff_img, NumPy array difference image
        mean_out_of_transit_image, NumPy array mean out-of-transit image
        mean_in_transit_image, NumPy array mean in-transit image
        snr_img, NumPy array SNR image
    """

    dur_days = dur_hours / 24.0  # convert duration from hours to days
    transit_depth = transit_depth_ppm / 1e6  # convert parts per million to fraction, not being used
    threshold_flux_factor = 1 - thr_transit_depth * transit_depth  # calculate 75% threshold of transit depth
    buffer_days = buffer / 1440

    with fits.open(fits_file_path) as hdul:  # get data from the tpf
        data = hdul[1].data
        time = data['TIME']
        flux = data['FLUX'] * ELECTRONS_PER_S_TO_CADENCE  # convert flux from electrons/s to electrons/cadence
        flux_err = data['FLUX_ERR'] * ELECTRONS_PER_S_TO_CADENCE  # convert flux from electrons/s to electrons/cadence
        quality = data['QUALITY']
        aperture = hdul[2].data

    # exclude points outside of window
    idxs_window = (time >= t_window_edges[0]) & (time <= t_window_edges[-1])
    idxs_window = (time >= time[0]) & (time <= time[-1])

    time, flux, flux_err, quality = time[idxs_window], flux[idxs_window], flux_err[idxs_window], quality[idxs_window]

    # find pixels in optimal photometric aperture; 2nd bit is on
    aperture_bit_mask = np.unpackbits(aperture.astype('uint8')).reshape((
        aperture.shape[0], aperture.shape[1], 8))[:, :, -2].astype('bool')
    baseline_flux = np.nanmedian(np.nansum(flux[:, aperture_bit_mask], axis=1))
    threshold_flux = baseline_flux * threshold_flux_factor

    # handle missing time values - exclude those cadences
    valid_time_idxs = np.isfinite(time)
    time, flux, flux_err, quality = (time[valid_time_idxs], flux[valid_time_idxs], flux_err[valid_time_idxs],
                                     quality[valid_time_idxs])

    # # handle NaN values by replacing them with the median flux
    # flux[np.isnan(flux)] = median_flux

    # filter out bad data using quality flags
    good_data_indices = quality == 0
    time = time[good_data_indices]
    flux = flux[good_data_indices]
    flux_err = flux_err[good_data_indices]

    cadences = np.arange(len(time))  # cadence number array
    in_transit_cadences = []
    out_of_transit_cadences_left = []
    out_of_transit_cadences_right = []

    # find first midtransit point in the time array
    first_transit_time = find_first_epoch_after_this_time(tce_time0bk, period_days, time[0])
    # compute all midtransit points in the time array
    midtransit_points_arr = [first_transit_time + phase_k * period_days
                             for phase_k in range(int(np.ceil((time[-1] - time[0]) / period_days)))]
    for t in midtransit_points_arr:  # iterate over midtransit points to get indices for start and end of it/oot windows

        # get indices for in-transit window
        in_start = np.searchsorted(time, t - dur_days / 2)  # gives index of cadence right after t - dur / 2
        in_end = np.searchsorted(time, t + dur_days / 2)  # gives index of cadence right after t + dur / 2

        # get indices for out-of-transit window to the left of in-transit window
        out_start_left = np.searchsorted(time, t - dur_days / 2 - buffer_days - dur_days)
        out_end_left = np.searchsorted(time, t - dur_days / 2 - buffer_days)

        # get indices for out-of-transit window to the right of in-transit window
        out_start_right = np.searchsorted(time, t + dur_days / 2 + buffer_days)
        out_end_right = np.searchsorted(time, t + dur_days / 2 + buffer_days + dur_days)

        # TODO: this does not need a for loop
        for cadence in cadences[in_start:in_end]:
            if np.isnan(flux[cadence][aperture_bit_mask]).any():
                print(f'Cadence {cadence} has {np.isnan(flux[cadence][aperture_bit_mask]).sum()} NaN values.')
            # if np.any(flux[cadence] < threshold_flux):  # threshold depth check
            # if np.any(flux[cadence][a == 155] < threshold_flux):  # threshold depth check
            if np.sum(flux[cadence][aperture_bit_mask]) < threshold_flux:  # threshold depth check
                in_transit_cadences.append(cadence)
            # else:
            # print(f'cadence {cadence} for time {time[cadence]} is above the threshold.')

        out_of_transit_cadences_left.extend(cadences[out_start_left:out_end_left])
        out_of_transit_cadences_right.extend(cadences[out_start_right:out_end_right])

    out_of_transit_cadences = np.concatenate((out_of_transit_cadences_left, out_of_transit_cadences_right))

    if plot_dir:
        sap_flux = flux[:, aperture_bit_mask].sum(axis=1)
        f, ax = plt.subplots(figsize=(12, 8))
        ax.scatter(time, sap_flux, s=8)
        ax.set_xlabel('Time - 2457000 [BTJD days]')
        ax.set_ylabel('SAP Flux [e-/cadence]')
        f.tight_layout()
        f.savefig(plot_dir / f'sap_flux.png')
        # plt.show()
        plt.close()

    # TODO: check and remove cadences from in-transit that are also out-of-transit; or exclude from both?
    if len(in_transit_cadences) == 0 or len(out_of_transit_cadences) == 0:
        warnings.warn(f"No cadences selected for in-transit and/or out-of-transit: \n "
                      f"In-transit: {len(in_transit_cadences)}\n "
                      f"Out-of-transit: {len(out_of_transit_cadences)}.")

        # return None, None, None

    if plot_dir:
        # plt.switch_backend('TkAgg')
        first_transit_time = find_first_epoch_after_this_time(tce_time0bk, period_days, time[0])
        binary_time = create_binary_time_series(time, first_transit_time, dur_days, period_days)
        f, ax = plt.subplots()
        ax.plot(time, binary_time)
        ax.scatter(time[in_transit_cadences], np.ones(len(time[in_transit_cadences])), c='g', s=8)
        ax.scatter(time[out_of_transit_cadences], np.zeros(len(time[out_of_transit_cadences])), c='r', s=8)
        f.savefig(plot_dir / 'bin_cadences.png')
        # plt.show()
        plt.close()

    # compute mean in-transit and out-of-transit images
    mean_in_transit_image = np.mean(flux[in_transit_cadences], axis=0)
    mean_out_of_transit_image = np.mean(flux[out_of_transit_cadences], axis=0)
    # compute difference image
    diff_img = mean_out_of_transit_image - mean_in_transit_image

    # compute snr image
    mean_in_transit_image_sigma = np.sqrt(np.sum(flux_err[in_transit_cadences] ** 2, axis=0)) / len(in_transit_cadences)
    mean_oot_transit_image_sigma = (np.sqrt(np.sum(flux_err[out_of_transit_cadences] ** 2, axis=0)) /
                                    len(out_of_transit_cadences))
    mean_img_sigma = np.sqrt((mean_in_transit_image_sigma ** 2) + (mean_oot_transit_image_sigma ** 2))
    snr_img = diff_img / mean_img_sigma

    return diff_img, mean_out_of_transit_image, mean_in_transit_image, snr_img


def get_diff_img_data_for_window(fits_file_path, t_window, dur_hours, buffer=30, plot_dir=None):
    """ Computes difference, in-transit, out-of-transit, and SNR images from flux data and based on a timestamp
    `t_window` and transit duration `dur_hours`.

    Args:
        fits_file_path: str, path to target pixel fits file
        t_window: float, timestamp that is the center of the window
        dur_hours: float, transit duration in hours
        buffer: float, buffer size in minutes between in-transit and out-transit windows
        plot_dir: Path, path to directory to save plots

    Returns:
        diff_img, NumPy array difference image
        mean_out_of_transit_image, NumPy array mean out-of-transit image
        mean_in_transit_image, NumPy array mean in-transit image
        snr_img, NumPy array SNR image
    """

    dur_days = dur_hours / 24.0  # convert duration from hours to days
    buffer_days = buffer / 1440

    with fits.open(fits_file_path) as hdul:  # get data from the tpf
        data = hdul[1].data
        time = data['TIME']
        flux = data['FLUX'] * ELECTRONS_PER_S_TO_CADENCE  # convert flux from electrons/s to electrons/cadence
        flux_err = data['FLUX_ERR'] * ELECTRONS_PER_S_TO_CADENCE  # convert flux from electrons/s to electrons/cadence
        quality = data['QUALITY']
        aperture = hdul[2].data

    # find pixels in optimal photometric aperture; 2nd bit is on
    aperture_bit_mask = np.unpackbits(aperture.astype('uint8')).reshape((
        aperture.shape[0], aperture.shape[1], 8))[:, :, -2].astype('bool')
    # baseline_flux = np.nanmedian(np.nansum(flux[:, aperture_bit_mask], axis=1))

    # handle missing time values - exclude those cadences
    valid_time_idxs = np.isfinite(time)
    time, flux, flux_err, quality = (time[valid_time_idxs], flux[valid_time_idxs], flux_err[valid_time_idxs],
                                     quality[valid_time_idxs])

    # # handle NaN values by replacing them with the median flux
    # flux[np.isnan(flux)] = median_flux

    # filter out bad data using quality flags
    good_data_indices = quality == 0
    time = time[good_data_indices]
    flux = flux[good_data_indices]
    flux_err = flux_err[good_data_indices]

    cadences = np.arange(len(time))  # cadence number array
    in_start = np.searchsorted(time, t_window - dur_days / 2)  # gives index of cadence right after t - dur / 2
    in_end = np.searchsorted(time, t_window + dur_days / 2)
    in_transit_cadences = cadences[in_start:in_end]

    # TODO: this does not need a for loop
    for cadence in in_transit_cadences:
        if np.isnan(flux[cadence][aperture_bit_mask]).any():
            print(f'Cadence {cadence} has {np.isnan(flux[cadence][aperture_bit_mask]).sum()} NaN values.')

    # get indices for out-of-transit window to the left of in-transit window
    out_start_left = np.searchsorted(time, t_window - dur_days / 2 - buffer_days - dur_days)
    out_end_left = np.searchsorted(time, t_window - dur_days / 2 - buffer_days)
    # get indices for out-of-transit window to the right of in-transit window
    out_start_right = np.searchsorted(time, t_window + dur_days / 2 + buffer_days)
    out_end_right = np.searchsorted(time, t_window + dur_days / 2 + buffer_days + dur_days)
    out_of_transit_cadences = np.concatenate(
        [cadences[out_start_left:out_end_left], cadences[out_start_right:out_end_right]])

    if plot_dir:
        sap_flux = flux[:, aperture_bit_mask].sum(axis=1)
        f, ax = plt.subplots(figsize=(12, 8))
        ax.scatter(time[in_transit_cadences], sap_flux[in_transit_cadences], s=8, c='g', label='Transit Cadences')
        ax.scatter(time[out_of_transit_cadences], sap_flux[out_of_transit_cadences], s=8, c='r',
                   label='Out-of-Transit Cadences')
        ax.set_xlabel('Time - 2457000 [BTJD days]')
        ax.set_ylabel('SAP Flux [e-/cadence]')
        ax.legend()
        f.tight_layout()
        f.savefig(plot_dir / f'window_t_{t_window}_sap_flux.png')
        # plt.show()
        plt.close()

    # TODO: check and remove cadences from in-transit that are also out-of-transit; or exclude from both?
    if len(in_transit_cadences) == 0 or len(out_of_transit_cadences) == 0:
        # warnings.warn(f"No cadences selected for in-transit and/or out-of-transit: \n "
        #               f"In-transit: {len(in_transit_cadences)}\n "
        #               f"Out-of-transit: {len(out_of_transit_cadences)}.")

        raise ValueError(f"No cadences selected for in-transit and/or out-of-transit: \n "
                         f"In-transit: {len(in_transit_cadences)}\n "
                         f"Out-of-transit: {len(out_of_transit_cadences)}.")

        # return None, None, None, None

    if plot_dir:
        f, ax = plt.subplots()
        ax.scatter(time[in_transit_cadences], np.ones(len(time[in_transit_cadences])), c='g', s=8,
                   label='Transit Cadences')
        ax.scatter(time[out_of_transit_cadences], np.zeros(len(time[out_of_transit_cadences])), c='r', s=8,
                   label='Out-of-Transit Cadences')
        ax.legend()
        f.savefig(plot_dir / f'window_t_{t_window}_bin_cadences.png')
        # plt.show()
        plt.close()

    # compute mean in-transit and out-of-transit images
    mean_in_transit_image = np.mean(flux[in_transit_cadences], axis=0)
    mean_out_of_transit_image = np.mean(flux[out_of_transit_cadences], axis=0)
    # compute difference image
    diff_img = mean_out_of_transit_image - mean_in_transit_image

    # compute snr image
    mean_in_transit_image_sigma = np.sqrt(np.sum(flux_err[in_transit_cadences] ** 2, axis=0)) / len(in_transit_cadences)
    mean_oot_transit_image_sigma = (np.sqrt(np.sum(flux_err[out_of_transit_cadences] ** 2, axis=0)) /
                                    len(out_of_transit_cadences))
    mean_img_sigma = np.sqrt((mean_in_transit_image_sigma ** 2) + (mean_oot_transit_image_sigma ** 2))
    snr_img = diff_img / mean_img_sigma

    return diff_img, mean_out_of_transit_image, mean_in_transit_image, snr_img


def convert_target_position_from_ra_dec_to_pixel(fits_file_path, ra, dec):
    """ Convert RA and Dec to pixel coordinates on the CCD frame.

    Args:
        fits_file_path: string, path to the TESS FITS file.
        ra: float, Right Ascension of the target in degrees.
        dec: float, Declination of the target in degrees.

    Returns:
        target_pos_col: float, column coordinate of the target in the CCD frame.
        target_pos_row: float, row coordinate of the target in the CCD frame.
    """

    with fits.open(fits_file_path) as hdul:
        if 'APERTURE' in hdul:
            wcs_info = WCS(hdul['APERTURE'].header)
        elif 'PIXELS' in hdul:
            wcs_info = WCS(hdul['PIXELS'].header)
        else:
            raise ValueError("WCS information not found in the FITS file.")

        target_pos_col, target_pos_row = wcs_info.all_world2pix(ra, dec, 0)

    return target_pos_col, target_pos_row


def extract_diff_img_data_from_window(tpf_path, t_stamp, tce_duration, buffer_oot, target_ra, target_dec, size_img,
                                      f_size, sector_run, center_target, tce_uid, plot_dir=None):
    """ Extracts and preprocesses difference image data for a window of time centered on timestamp `t_stamp`.

    Args:
        tpf_path: Path, path to target pixel file
        t_stamp: float, timestamp of center of window
        tce_duration: float, transit duration in hours
        buffer_oot: float, buffer for out-of-transit cadences in minutes
        target_ra: float, target right ascension
        target_dec: float, target declination
        size_img: list, resize images to `size_img`
        f_size: list, factor used to enlarge the size of the image after resizing to `size_img`
        sector_run: int, sector run
        center_target: bool, if True image is centered on target
        tce_uid: string, TCE unique id
        plot_dir: Path, directory to save plots; if None, plotting is disabled

    Returns:
        diff_img_processed, NumPy array preprocessed difference image
        oot_img_processed, NumPy array preprocessed out-of-transit image
        snr_img_processed, NumPy array preprocessed SNR image
        target_img, NumPy array target image
        target_pos_col, float target column position in pixels
        target_pos_row, float target row position in pixels
    """

    # extract difference image data for window centered on t_window
    try:
        diff_img, oot_img, it_img, snr_img = get_diff_img_data_for_window(tpf_path,
                                                                          t_stamp,
                                                                          tce_duration,
                                                                          buffer=buffer_oot,
                                                                          plot_dir=plot_dir)
    except ValueError as error:
        raise error
    # if np.isnan(diff_img).all():
    #     warnings.warn(f'No difference image data computed for timestamp {t_stamp} for TCE {tce_uid}.')
    #
    #     return

    # get target position in ccd pixel frame
    target_pos_px = convert_target_position_from_ra_dec_to_pixel(tpf_path, target_ra, target_dec)
    target_pos_px = (target_pos_px[0].item(), target_pos_px[1].item())

    # plot difference image data
    diff_img_data = np.dstack((it_img, oot_img, diff_img, snr_img))
    diff_img_data = np.expand_dims(diff_img_data, axis=-1)
    if plot_dir:
        plot_diff_img_data_extracted(diff_img_data, target_pos_px[0], target_pos_px[1],
                                     plot_dir / f'{tce_uid}_window_t_{t_stamp:.3f}_extracted_diff_img_data.png',
                                     logscale=False)

    (diff_img_processed, oot_img_processed, snr_img_processed, target_img, (target_pos_col, target_pos_row), _, _) = (
        preprocess_single_diff_img_data_for_example(
            diff_img=diff_img,
            oot_img=oot_img,
            snr_img=snr_img,
            target_pos_col=target_pos_px[0],
            target_pos_row=target_pos_px[1],
            size_h=size_img[0],
            size_w=size_img[1],
            size_f_h=f_size[0],
            size_f_w=f_size[1],
            img_n=sector_run,
            tce_uid=tce_uid,
            prefix='s',
            center_target=center_target,
        ))

    if plot_dir:
        plot_diff_img_data(diff_img_processed, oot_img_processed, snr_img_processed, target_img,
                           {'x': target_pos_row, 'y': target_pos_col}, np.nan, sector_run, tce_uid,
                           plot_dir / f'{tce_uid}_t_{t_stamp:.3f}_preprocessed_diff_img_data.png', logscale=False)

    return diff_img_processed, oot_img_processed, snr_img_processed, target_img, target_pos_col, target_pos_row
