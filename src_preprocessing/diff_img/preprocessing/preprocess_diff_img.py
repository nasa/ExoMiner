"""
Preprocess extracted difference image data.

Sampling quarter/sector image data:
1 - Get number of valid quarters/sector_runs.
2 - If there are no valid images, set difference and oot images to NaN, px and subpx coordinates to zero, quality metric
to NaN.
3 - Sample quarters/sector_runs from valid quarters/sector runs set.

Preprocessing quarter/sector image data:
1 - Create discrete mapping of target location to 3x3 grid.
2 - Set negative oot pixels to NaN in oot and diff images.
3 - Fill out missing values by using nearest neighbors with same unit weight
(missing values add zero and have no weight).
4 - Pixels for which the padding window was all missing values are set to zero and min value
for diff and oot images, respectively.
5 - Pad images by extending edges.
6 - Resize images using nearest neighbor interpolation.
7 - [Optional] Center images on target star.
8 - Crop images.
9 - Create target pixel image.
"""

# 3rd party
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import os
import multiprocessing
import argparse
from PIL import Image
import yaml
from scipy.signal import convolve2d

# local
from src_preprocessing.diff_img.preprocessing.utils_diff_img import plot_diff_img_data


def check_for_missing_values_in_preproc_diff_data(data):
    """ Checks for missing values (NaNs) in the preprocessed data.

    Args:
        data: dict, preprocessed data

    Returns:
        bool, True if there's at least a missing value in any preprocessed data.

    """

    missing_value_found = False

    for k, v in data.items():
        if k in ['quality', 'images_numbers']:
            missing_value_found = np.isnan(v).sum() > 0
            if missing_value_found:
                return missing_value_found
        else:
            for k2, v2 in v.items():
                missing_value_found = np.isnan(v2).sum() > 0
                if missing_value_found:
                    return missing_value_found

    return missing_value_found


def initialize_data_example_with_missing_values(size_h, size_w, number_of_imgs_to_sample):
    """ Initializes data for a given example.

    Args:
        size_h: int, height
        size_w: int, width
        number_of_imgs_to_sample: int, number of images to create

    Returns:
        dict, with initialized data

    """

    initialized_data_dict = {
        'images': {
            'diff_imgs': [np.nan * np.ones((size_h, size_w), dtype='float')
                          for _ in range(number_of_imgs_to_sample)],
            'oot_imgs': [np.nan * np.ones((size_h, size_w), dtype='float')
                         for _ in range(number_of_imgs_to_sample)],
            'snr_imgs': [np.nan * np.ones((size_h, size_w), dtype='float')
                         for _ in range(number_of_imgs_to_sample)],
            'target_imgs': [np.nan * np.ones((size_h, size_w), dtype='float')
                            for _ in range(number_of_imgs_to_sample)],

            'diff_imgs_tc': [np.nan * np.ones((size_h, size_w), dtype='float')
                             for _ in range(number_of_imgs_to_sample)],
            'oot_imgs_tc': [np.nan * np.ones((size_h, size_w), dtype='float')
                            for _ in range(number_of_imgs_to_sample)],
            'snr_imgs_tc': [np.nan * np.ones((size_h, size_w), dtype='float')
                            for _ in range(number_of_imgs_to_sample)],
            'target_imgs_tc': [np.nan * np.ones((size_h, size_w), dtype='float')
                               for _ in range(number_of_imgs_to_sample)],

        },
        'target_position': {
            'pixel_x': [np.nan] * number_of_imgs_to_sample,
            'pixel_y': [np.nan] * number_of_imgs_to_sample,
            'subpixel_x': [np.nan] * number_of_imgs_to_sample,
            'subpixel_y': [np.nan] * number_of_imgs_to_sample,
            'pixel_x_tc': [np.nan] * number_of_imgs_to_sample,
            'pixel_y_tc': [np.nan] * number_of_imgs_to_sample,
            'subpixel_x_tc': [np.nan] * number_of_imgs_to_sample,
            'subpixel_y_tc': [np.nan] * number_of_imgs_to_sample
        },
        'quality': [np.nan] * number_of_imgs_to_sample,
        'images_numbers': [np.nan] * number_of_imgs_to_sample,
    }

    return initialized_data_dict


def set_data_example_to_placeholder_values(size_h, size_w, number_of_imgs_to_sample):
    """ Sets data for a given example with placeholder values.

    Args:
        size_h: int, height
        size_w: int, width
        number_of_imgs_to_sample: int, number of images to create

    Returns:
        dict, with placeholder data

    """

    half_height, half_width = size_h // 2, size_w // 2

    # update data using placeholder values
    missing_data_placeholder = {
        'images': {
            'diff_imgs': [np.zeros((size_h, size_w), dtype='float')
                          for _ in range(number_of_imgs_to_sample)],
            'oot_imgs': [np.zeros((size_h, size_w), dtype='float')
                         for _ in range(number_of_imgs_to_sample)],
            'snr_imgs': [np.zeros((size_h, size_w), dtype='float')
                         for _ in range(number_of_imgs_to_sample)],
            'target_imgs': [np.zeros((size_h, size_w), dtype='float')
                            for _ in range(number_of_imgs_to_sample)],

            'diff_imgs_tc': [np.zeros((size_h, size_w), dtype='float')
                             for _ in range(number_of_imgs_to_sample)],
            'oot_imgs_tc': [np.zeros((size_h, size_w), dtype='float')
                            for _ in range(number_of_imgs_to_sample)],
            'snr_imgs_tc': [np.zeros((size_h, size_w), dtype='float')
                            for _ in range(number_of_imgs_to_sample)],
            'target_imgs_tc': [np.zeros((size_h, size_w), dtype='float')
                               for _ in range(number_of_imgs_to_sample)],

        },
        'target_position': {
            'pixel_x': [half_height] * number_of_imgs_to_sample,
            'pixel_y': [half_width] * number_of_imgs_to_sample,
            'subpixel_x': [0] * number_of_imgs_to_sample,
            'subpixel_y': [0] * number_of_imgs_to_sample,
            'pixel_x_tc': [half_height] * number_of_imgs_to_sample,
            'pixel_y_tc': [half_width] * number_of_imgs_to_sample,
            'subpixel_x_tc': [0] * number_of_imgs_to_sample,
            'subpixel_y_tc': [0] * number_of_imgs_to_sample,
        },
        'quality': [0] * number_of_imgs_to_sample,
        'images_numbers': [np.nan] * number_of_imgs_to_sample,
    }

    for target_img_i in range(number_of_imgs_to_sample):
        missing_data_placeholder['images']['target_imgs'][target_img_i][half_height, half_width] = 1
        missing_data_placeholder['images']['target_imgs_tc'][target_img_i][half_height, half_width] = 1

    return missing_data_placeholder


def sample_image_data(n_valid_imgs, valid_images_idxs, number_of_imgs_to_sample):
    """ Samples randomly valid quarters/sectors of data.

    Args:
        n_valid_imgs: int, number of valid images
        valid_images_idxs: list, indices of valid quarters/sectors
        number_of_imgs_to_sample: int, number of images to sample

    Returns:
        NumPy array, sampled valid images indices

    """

    if n_valid_imgs < number_of_imgs_to_sample:
        # use all quarters/sectors available before random sampling
        k_n_valid_imgs = number_of_imgs_to_sample // n_valid_imgs
        random_sample_imgs_idxs = np.tile(valid_images_idxs, k_n_valid_imgs)
        # fill the remaining spots by sampling randomly without replacement
        random_sample_imgs_idxs = np.concatenate([random_sample_imgs_idxs,
                                                  np.random.choice(valid_images_idxs,
                                                                   number_of_imgs_to_sample % n_valid_imgs,
                                                                   replace=False)])
    else:  # no sampling with replacement since there are more valid images than the requested number to sample
        random_sample_imgs_idxs = np.random.choice(valid_images_idxs, number_of_imgs_to_sample, replace=False)

    return random_sample_imgs_idxs


def set_negative_value_oot_pixels(diff_img, oot_img, snr_img):
    """ Find pixels with negative out-of-transit values, and then sets them to NaNs (missing) for both difference and
    out-of-transit images.

    Args:
        diff_img: NumPy array, difference image
        oot_img: NumPy array, out-of-transit image
        snr_img: NumPy array, snr image

    Returns:
        diff_img: NumPy array, updated difference image
        oot_img: NumPy array, updated out-of-transit image
        snr_img: NumPy array, updated snr image
    """

    # find pixels with negative values
    curr_img_neg = oot_img < 0

    # set to nan
    diff_img[curr_img_neg] = np.nan
    oot_img[curr_img_neg] = np.nan
    snr_img[curr_img_neg] = np.nan

    return diff_img, oot_img, snr_img


def crop_images_to_valid_size(diff_img, oot_img, snr_img, target_pos_col, target_pos_row):
    """ Crops images to their valid size, i.e., the minimum height and width that include any non-missing pixels.
    Missing pixels need to be represented by NaNs.

    Args:
        diff_img: NumPy array, difference image
        oot_img: NumPy array, out-of-transit image
        snr_img: NumPy array, snr image
        target_pos_col: float, target position column coordinate
        target_pos_row: float, target position row coordinate

    Returns:
        diff_img_crop: NumPy array, updated difference image
        oot_img_crop: NumPy array, updated out-of-transit image
        snr_img_crop: NumPy array, updated snr image
        target_pos_col_crop: float, updated target position column coordinate
        target_pos_row_crop: float, updated target position row coordinate
    """

    # find missing pixels
    idxs_not_missing = np.where(~np.isnan(diff_img))
    min_row, max_row = idxs_not_missing[0].min(), idxs_not_missing[0].max()
    min_col, max_col = idxs_not_missing[1].min(), idxs_not_missing[1].max()

    # choose smallest size that includes all valid pixels
    diff_img_crop = diff_img[min_row:max_row + 1, min_col: max_col + 1]
    oot_img_crop = oot_img[min_row:max_row + 1, min_col: max_col + 1]
    snr_img_crop = snr_img[min_row:max_row + 1, min_col: max_col + 1]

    # update target pixel position
    target_pos_col_crop = target_pos_col - min_col
    target_pos_row_crop = target_pos_row - min_row

    return diff_img_crop, oot_img_crop, snr_img_crop, target_pos_col_crop, target_pos_row_crop


def fill_missing_values_nearest_neighbors(img, window):
    """ Fills missing values in an image through convolution of a kernel window. The pixels with missing value are set
    to zero during the convolution. The missing values need to be represented by NaNs. If all pixels in the window
    are missing, the output value for the respective pixel is also NaN.

    Args:
        img: NumPy array, image
        window: NumPy array, kernel

    Returns:
        img_fill, NumPy array of image with filled missing values
    """

    idxs_nan = np.isnan(img)  # find pixels with missing values

    # create 2d array to count valid pixels (i.e., non-missing pixels)
    valid_px_arr = np.ones(img.shape)

    valid_px_arr[idxs_nan] = 0  # missing pixels have no contribution to the imputation

    # count how many valid pixels exist for each convolution
    valid_px_arr_conv = convolve2d(valid_px_arr, window, mode='same', boundary='fill', fillvalue=0)

    # fill values for image
    img_fill = img.copy()
    img_fill[idxs_nan] = 0  # set missing values to zero, so we can compute the convolutions

    # run kxk convolution on the image
    img_conv = convolve2d(img_fill, window, mode='same', boundary='fill', fillvalue=0)

    # normalize convolution values by valid pixel count
    img_conv /= valid_px_arr_conv

    # fill missing values with convolution values
    img_fill[idxs_nan] = img_conv[idxs_nan]

    return img_fill


def pad_images_by_extending_edges(diff_img, oot_img, snr_img, target_pos_col, target_pos_row, add_pad_h, add_pad_w):
    """ Pads images by extending edges. The padding is such that 1) makes sure the target pixel will not end up
    outside the cropping area when not centering the images on the target pixel, and 2) both dimensions have the same
    size.

    Args:
        diff_img: NumPy array, difference image
        oot_img: NumPy array, out-of-transit image
        snr_img: NumPy array, snr image
        target_pos_col: float, target position column coordinate
        target_pos_row: float, target position row coordinate
        add_pad_h: int, additional padding for the height dimension
        add_pad_w: int, additional padding for the width dimension

    Returns:
        diff_img_pad: NumPy array, updated difference image
        oot_img_pad: NumPy array, updated out-of-transit image
        snr_img_pad: NumPy array, updated snr image
        target_pos_col_pad: float, updated target position column coordinate
        target_pos_row_pad: float, updated target position row coordinate
    """

    # initialize padding for each dimension
    pad_len = [diff_img.shape[0] // 2 + add_pad_h + 1, diff_img.shape[1] // 2 + add_pad_w + 1]

    # set paddings such that both dimensions have the same size after padding
    pad_len_diff = 2 * pad_len[0] + diff_img.shape[0] - (2 * pad_len[1] + diff_img.shape[1])  # size difference
    if pad_len_diff >= 0:  # case height is larger than width

        pad_len_row = [pad_len[0]] * 2

        # extra padding that should be added to each edge for the smaller dimension
        pad_len_col = [pad_len[1] + pad_len_diff // 2] * 2
        pad_len_col[1] += pad_len_diff % 2

    else:  # case width is larger than height
        pad_len_col = [pad_len[1]] * 2

        pad_len_diff *= -1
        pad_len_row = [pad_len[0] + pad_len_diff // 2] * 2
        pad_len_row[1] += pad_len_diff % 2

    diff_img_pad = np.pad(diff_img, (pad_len_row, pad_len_col), mode='edge')
    oot_img_pad = np.pad(oot_img, (pad_len_row, pad_len_col), mode='edge')
    snr_img_pad = np.pad(snr_img, (pad_len_row, pad_len_col), mode='edge')

    # update target pixel position
    target_pos_col_pad = target_pos_col + pad_len_col[0]
    target_pos_row_pad = target_pos_row + pad_len_row[0]

    return diff_img_pad, oot_img_pad, snr_img_pad, target_pos_col_pad, target_pos_row_pad


def center_images_to_target_pixel_location(diff_img, oot_img, snr_img, target_pos_col, target_pos_row):
    """ Centers images on the target pixel by padding through edge extension.

    Args:
        diff_img: NumPy array, difference image
        oot_img: NumPy array, out-of-transit image
        snr_img: NumPy array, snr image
        target_pos_col: float, target position column coordinate
        target_pos_row: float, target position row coordinate

    Returns:
        diff_img_tcenter: NumPy array, updated difference image
        oot_img_tcenter: NumPy array, updated out-of-transit image
        snr_img_tcenter: NumPy array, updated snr image
        target_pos_col_tcenter: float, updated target position column coordinate
        target_pos_row_tcenter: float, updated target position row coordinate
    """

    # center is shifted on both axis because it is padded by `pad_n_pxs`
    target_pos_pixel_col = int(target_pos_col)
    target_pos_pixel_row = int(target_pos_row)

    # find padding for each dimension
    center_row, center_col = diff_img.shape[0] // 2, diff_img.shape[1] // 2

    if target_pos_pixel_row >= center_row:  # target pixel location is after center row
        # set padding
        pad_len_h = [0, 2 * target_pos_pixel_row - diff_img.shape[0] + 1]
        # update target pixel position
        target_pos_row_tcenter = target_pos_row
    else:  # target pixel location is before center row
        pad_len_h = [diff_img.shape[0] - 2 * target_pos_pixel_row - 1, 0]
        target_pos_row_tcenter = target_pos_row + pad_len_h[0]

    if target_pos_pixel_col >= center_col:
        pad_len_w = [0, 2 * target_pos_pixel_col - diff_img.shape[1] + 1]
        target_pos_col_tcenter = target_pos_col
    else:
        pad_len_w = [diff_img.shape[1] - 2 * target_pos_pixel_col - 1, 0]
        target_pos_col_tcenter = target_pos_col + pad_len_w[1]

    # pad images by extending edges
    diff_img_tcenter = np.pad(diff_img, (pad_len_h, pad_len_w), mode='edge')
    oot_img_tcenter = np.pad(oot_img, (pad_len_h, pad_len_w), mode='edge')
    snr_img_tcenter = np.pad(snr_img, (pad_len_h, pad_len_w), mode='edge')

    return diff_img_tcenter, oot_img_tcenter, snr_img_tcenter, target_pos_col_tcenter, target_pos_row_tcenter


def crop_images_to_size(diff_img, oot_img, snr_img, target_pos_col, target_pos_row, size_h, size_w):
    """ Crops images to a given size defined by `size_h` and `size_w`. The cropping is done around the center of the
    image.

    Args:
        diff_img: NumPy array, difference image
        oot_img: NumPy array, out-of-transit image
        snr_img: NumPy array, snr image
        target_pos_col: float, target position column coordinate
        target_pos_row: float, target position row coordinate
        size_h: int, crop to this height
        size_w: int, crop to this width

    Returns:
        diff_img_crop: NumPy array, updated difference image
        oot_img_crop: NumPy array, updated out-of-transit image
        snr_img_crop: NumPy array, updated snr image
        target_pos_col_crop: float, updated target position column coordinate
        target_pos_row_crop: float, updated target position row coordinate
    """

    # crop images to target dimension if they are larger
    half_height, half_width = size_h // 2, size_w // 2

    # convert to PIL Image object
    diff_img = Image.fromarray(diff_img)
    oot_img = Image.fromarray(oot_img)
    snr_img = Image.fromarray(snr_img)

    center_col, center_row = diff_img.size[0] // 2, diff_img.size[1] // 2

    if diff_img.size[1] <= size_h:
        upper, lower = 0, diff_img.size[1]
    else:
        upper, lower = center_row - half_height, center_row + half_height + 1

    if diff_img.size[0] <= size_w:
        left, right = 0, diff_img.size[0]
    else:
        left, right = center_col - half_width, center_col + half_width + 1

    diff_img_crop = diff_img.crop((left, upper, right, lower))
    oot_img_crop = oot_img.crop((left, upper, right, lower))
    snr_img_crop = snr_img.crop((left, upper, right, lower))

    # update target pixel position
    target_pos_col -= left
    target_pos_row -= upper

    diff_img_crop = np.array(diff_img_crop)
    oot_img_crop = np.array(oot_img_crop)
    snr_img_crop = np.array(snr_img_crop)

    return diff_img_crop, oot_img_crop, snr_img_crop, target_pos_col, target_pos_row


def resize_images_by_resampling(diff_img, oot_img, snr_img, target_pos_col, target_pos_row, size_f_h, size_f_w):
    """ Resizes images using nearest neighbor interpolation and resampling factors `size_f_h` and `size_f_w` for height
    and width, respectively.

    Args:
        diff_img: NumPy array, difference image
        oot_img: NumPy array, out-of-transit image
        oot_img: NumPy array, snr image
        target_pos_col: float, target position column coordinate
        target_pos_row: float, target position row coordinate
        size_f_h: float, resampling factor for height
        size_f_w: float, resampling factor for width

    Returns:
        diff_img_resize: NumPy array, updated difference image
        oot_img_resize: NumPy array, updated out-of-transit image
        snr_img_resize: NumPy array, updated snr image
        target_pos_col_resize: float, updated target position column coordinate
        target_pos_row_resize: float, updated target position row coordinate
    """

    size_h, size_w = diff_img.shape

    # convert back to PIL Image object
    diff_img = Image.fromarray(diff_img)
    oot_img = Image.fromarray(oot_img)
    snr_img = Image.fromarray(snr_img)

    diff_img_resize = diff_img.resize(size=(size_h * size_f_h, size_w * size_f_w), resample=Image.Resampling.NEAREST)
    oot_img_resize = oot_img.resize(size=(size_h * size_f_h, size_w * size_f_w), resample=Image.Resampling.NEAREST)
    snr_img_resize = snr_img.resize(size=(size_h * size_f_h, size_w * size_f_w), resample=Image.Resampling.NEAREST)

    # convert back to NumPy array
    diff_img_resize = np.array(diff_img_resize)
    oot_img_resize = np.array(oot_img_resize)
    snr_img_resize = np.array(snr_img_resize)

    # map target position in resized image
    target_pos_col_up = target_pos_col * size_f_w + 1
    target_pos_row_up = target_pos_row * size_f_h + 1

    return diff_img_resize, oot_img_resize, snr_img_resize, target_pos_col_up, target_pos_row_up


def create_target_image(size_h, size_w, target_pos_pixel_col, target_pos_pixel_row):
    """ Creates target image.

    Args:
        size_h: int, height
        size_w: int, width
        target_pos_pixel_col: int, target pixel column coordinate
        target_pos_pixel_row: int, target pixel row coordinate

    Returns:
        target_img: NumPy array of target image

    """
    # create target image

    # initialize with all zeros
    target_img = np.zeros((size_h, size_w), dtype='float')

    # set target pixel in resized image with value of 1
    if (0 <= target_pos_pixel_row <= size_h - 1) and (0 <= target_pos_pixel_col <= size_w - 1):
        target_img[target_pos_pixel_row, target_pos_pixel_col] = 1

    return target_img


def map_target_subpixel_location_to_discrete_grid(target_pos_col, target_pos_row):
    """ Maps subpixel target coordinates to discrete range {-1, 0, 1}. The zero is the target pixel.

    Args:
        target_pos_col: float, target position column coordinate
        target_pos_row: float, target position row coordinate

    Returns:
        target_pos_col_disc: int, mapping for target position column coordinate
        target_pos_row_disc: int, mapping for target position row coordinate

    """

    target_pos_pixel_col = int(np.round(target_pos_col))
    target_pos_pixel_row = int(np.round(target_pos_row))

    target_col_diff = target_pos_pixel_col - target_pos_col
    target_row_diff = target_pos_pixel_row - target_pos_row

    target_col_disc = int(2 * target_col_diff)
    target_row_disc = int(2 * target_row_diff)

    return target_col_disc, target_row_disc


def preprocess_single_diff_img_data_for_example(diff_img, oot_img, snr_img, target_pos_col, target_pos_row, size_h,
                                                size_w, size_f_h, size_f_w, img_n, tce_uid, prefix, log=None,
                                                center_target=True, proc_id=-1):
    """ Preprocesses the difference image data for a single example.

    Args:
        diff_img: NumPy array, difference image
        oot_img: NumPy array, out-of-transit image
        snr_img: NumPy array, snr image
        target_pos_col: float, target location column coordinate
        target_pos_row: float, target location row coordinate
        size_h: int, output image height (before scaling by resampling factor `size_f_h`)
        size_w: int, output image width (before scaling by resampling factor `size_f_w`)
        size_f_h: int, height resampling factor
        size_f_w: int, width resampling factor
        img_n: int, quarter or sector for Kepler or TESS, respectively
        tce_uid: str, TCE unique id. For Kepler, it should be '{kic_id}-{tce_planet_number}';
        for TESS, '{tic_id}-{tce_planet_number}-S{sector_run}'
        prefix: str, 'q' or 's' for Kepler or TESS, respectively
        log: logger
        center_target: bool, if True the images are centered in the target pixel by padding through edge extension
        proc_id: int, process id

    Returns:
        diff_img, NumPy array for preprocessed difference image
        oot_img, NumPy array for preprocessed out-of-transit image
        snr_img, NumPy array for preprocessed snr image
        target_img, NumPy array for target location image
        target_pos_col, int for target column pixel
        target_pos_row, int for target row pixel
        target_col_disc, int in {-1, 0, 1} for target subpixel column value
        target_row_disc, int in {-1, 0, 1} for target subpixel row value
    """
    # map subpixel coordinates to discrete range {-1, 0, 1}; zero is target pixel
    target_col_disc, target_row_disc = map_target_subpixel_location_to_discrete_grid(target_pos_col, target_pos_row)

    half_height, half_width = size_h // 2, size_w // 2

    # replace pixels that have negative oot values with nan in both images
    diff_img, oot_img, snr_img = set_negative_value_oot_pixels(diff_img, oot_img, snr_img)

    if np.isnan(diff_img).all() or np.isnan(oot_img).all() or np.isnan(snr_img).all():
        if log:
            log.info(
                f'[{proc_id}] All pixels in the difference, out-of-transit, or SNR image were missing their values '
                f'after setting pixels with negative out-of-transit values to zero on both images '
                f'for example {tce_uid} in {prefix} {img_n}.')

    # get min and max indices on both dimensions that have at least one non-missing pixel
    diff_img, oot_img, snr_img, target_pos_col, target_pos_row = (
        crop_images_to_valid_size(diff_img, oot_img, snr_img, target_pos_col, target_pos_row))

    # fill out missing values by using nearest neighbors with same weight
    diff_img = fill_missing_values_nearest_neighbors(diff_img, np.ones((3, 3)))
    oot_img = fill_missing_values_nearest_neighbors(oot_img, np.ones((3, 3)))
    snr_img = fill_missing_values_nearest_neighbors(snr_img, np.ones((3, 3)))

    # deal with pixels for which the padding window was all missing values
    idxs_nan = np.isnan(diff_img)
    if idxs_nan.sum() != 0:
        if log:
            log.info(
                f'[{proc_id}] {idxs_nan.sum()} pixels were missing a value after nearest neighborhood padding'
                f' for example {tce_uid} in {prefix} {img_n}.')

        diff_img[idxs_nan] = 0
        snr_img[idxs_nan] = 0
        oot_img[idxs_nan] = np.nanmin(oot_img)

    # pad image by extending edges
    diff_img, oot_img, snr_img, target_pos_col, target_pos_row = (
        pad_images_by_extending_edges(diff_img, oot_img, snr_img,
                                      target_pos_col, target_pos_row,
                                      half_height, half_width))

    # resize image using nearest neighbor interpolation to three times the target dimension
    diff_img, oot_img, snr_img, target_pos_col, target_pos_row = resize_images_by_resampling(
        diff_img, oot_img, snr_img, target_pos_col, target_pos_row, size_f_h, size_f_w)

    if center_target:
        diff_img, oot_img, snr_img, target_pos_col, target_pos_row = (center_images_to_target_pixel_location(
            diff_img, oot_img, snr_img, target_pos_col, target_pos_row))

    # crop images to target dimension if they are larger
    diff_img, oot_img, snr_img, target_pos_col, target_pos_row = crop_images_to_size(diff_img, oot_img, snr_img,
                                                                                     target_pos_col,
                                                                                     target_pos_row,
                                                                                     size_h * size_f_h,
                                                                                     size_w * size_f_w)

    # check if target pixel location is inside the image
    if (target_pos_col < 0 or target_pos_row < 0 or target_pos_col >= diff_img.shape[0] or
            target_pos_row >= diff_img.shape[1]):
        if log:
            log.info(f'[{proc_id}] Target pixel is outside of image after cropping for example '
                     f'{tce_uid} in {prefix} {img_n}.')

        # target_pos_row, target_pos_col = np.unravel_index(np.argmax(diff_img, axis=None), diff_img.shape)

        # raise ValueError(f'[{proc_id}] Target pixel is outside of image after cropping for example '
        #                  f'{tce_uid} in {prefix} {imgs_t[q_i]}.')

    # create target image
    target_pos_pixel_col_up = int(np.round(target_pos_col))
    target_pos_pixel_row_up = int(np.round(target_pos_row))

    target_img = create_target_image(size_h * size_f_h, size_w * size_f_w,
                                     target_pos_pixel_col_up, target_pos_pixel_row_up)

    return diff_img, oot_img, snr_img, target_img, target_pos_col, target_pos_row, target_col_disc, target_row_disc


def preprocess_image_data_from_examples_in_saturated_targets(data, final_size, upscale_f, number_of_imgs_to_sample):
    """ Deal with image data from examples in saturated targets. Replace image data by placeholder data.

    Args:
        data: dict, image data for example
        final_size: dict, image size ('x', and 'y')
        upscale_f: dict, upscale factor for image ('x' and 'y')
        number_of_imgs_to_sample: int, number of images to create for each image type

    Returns:
        data, dict with preprocessed data
    """

    # update data using placeholder values
    missing_data_placeholder = set_data_example_to_placeholder_values(final_size['x'] * upscale_f['x'],
                                                                      final_size['y'] * upscale_f['y'],
                                                                      number_of_imgs_to_sample)

    data.update(missing_data_placeholder)

    return data


def preprocess_diff_img_tces(diff_img_data_dict, number_of_imgs_to_sample, upscale_f, final_size, quality_metrics_tbl,
                             saturated_tce_ids_lst, mission_name, save_dir, log=None, plot_prob=0):
    """ Preprocessing pipeline for difference image data for a set of TCEs.

    Args:
        diff_img_data_dict: dict, {tce_id: data_dict}, where `data_dict` is another dictionary of the
        type {'target_ref_centroid': [t_ref_1, t_ref_2, ...], 'image_data': [img_1, img_2, ...],
        'image_number': [img_n_1, img_n_2, ...]}. `t_ref_i` is a dictionary of the type
        {'col': {'value': col_val, 'uncertainty': col_un}, 'row': {'value': col_val, 'uncertainty': col_un}}.
        This dictionary contains the target star position in each image and the associated uncertainty. The coordinates
        are in pixels and are relative to pixel (0, 0) in the image. `image_data` is a list of NumPy arrays. Each NumPy
        array is the set of images for a given quarter/sector run, and has dimensions [height, width, 4, 2]. The height
        and width are specific to the given target, and quarter/sector run. The 3rd dimension is associated with the
        different types of images: index 0 is the in-transit image, index 1 is the out-of-transit image, index 2 the
        difference image, and index 3 the SNR. The 4th dimension contains the pixel values and the uncertainties.
        Index 0 gives the values and index 1 the uncertainties. `image_number` is a list of integers that are the
        quarter/sector numbers for the extracted data.
        number_of_imgs_to_sample: int, number of quarters/sectors to sample
        upscale_f: dict, resize factor for 'x' and 'y' dimensions. Final size of images is
        (final_size['x'] * upscale_f['x'], final_size['y'] * upscale_f['y'])
        final_size: dict, image size before resizing (final_size['x'], final_size['y'])
        quality_metrics_tbl: pandas DataFrame, quality metrics for TCEs
        saturated_tce_ids_lst: list, TCE IDs associated with saturated targets
        mission_name: str, mission from where the difference image data is from. Either `kepler` or `tess`
        save_dir: Path, destination directory for preprocessed data
        log: logger
        plot_prob: float, probability to plot preprocessing results

    Returns:
            preprocessing_dict, dict with preprocessed data
            tces_info_tbl, pandas DataFrame with information on the preprocessing run
    """

    proc_id = os.getpid()  # get process id

    if log is None:
        # set up logger for the process
        log = logging.getLogger(name=f'preprocess_{proc_id}')
        logger_handler = logging.FileHandler(filename=save_dir / f'preprocess_{proc_id}.log', mode='w')
        logger_formatter = logging.Formatter('%(asctime)s - %(message)s')
        log.setLevel(logging.INFO)
        logger_handler.setFormatter(logger_formatter)
        log.addHandler(logger_handler)
        log.info(f'[{proc_id}] Starting preprocessing...')

    if mission_name == 'kepler':
        prefix = 'q'
    elif mission_name == 'tess':
        prefix = 's'
    else:
        raise ValueError(f'Mission not recognized ({mission_name}). Set variable to `kepler` or `tess`.')

    # check the (potential) number of maximum images available using the quality metrics table
    imgs_t = [int(col[1:-6]) for col in quality_metrics_tbl.columns if '_value' in col]
    n_max_imgs_avail = imgs_t[-1] - imgs_t[0] + 1

    log.info(f'[{proc_id}] Checking examples in saturated targets...')
    tces_in_sat_tbl = [tce_id for tce_id in diff_img_data_dict if tce_id in saturated_tce_ids_lst]
    log.info(f'Found {len(tces_in_sat_tbl)} examples in saturated targets.')

    # initialized TCE table with information on the preprocessing
    tces_info_tbl = pd.DataFrame({'uid': [uid for uid in diff_img_data_dict]})
    # tces_info_tbl['oot_negative_values'] = ''
    for col in ['available', 'valid', 'sampled']:
        tces_info_tbl[f'{col}_{prefix}s'] = ''
    tces_info_tbl['saturated'] = False
    # initialize sampled quality metrics dictionary
    sampled_qmetrics = {'sampled_qmetrics': [np.nan * np.ones(number_of_imgs_to_sample)
                                             for _ in range(len(tces_info_tbl))]}

    # initialize data dictionaries
    preprocessing_dict = {}
    for tce_uid in diff_img_data_dict:  # iterate over TCEs

        # initialize dictionary for the preprocessing results
        preprocessing_dict[tce_uid] = initialize_data_example_with_missing_values(final_size['x'] * upscale_f['x'],
                                                                                  final_size['y'] * upscale_f['y'],
                                                                                  number_of_imgs_to_sample)

    # make a new dictionary for the preprocessed data
    for tce_i, tce_uid in enumerate(preprocessing_dict):

        if tce_i % 500 == 0:
            log.info(f'[{proc_id}] Preprocessed {tce_i + 1} examples out of {len(diff_img_data_dict)}.')

        # # checking if TCE is in a saturated target; do not preprocess data for these cases
        # if tce_uid in saturated_tce_ids_lst:
        #     # update data using placeholder values
        #     log.info(f'[{proc_id}] Example {tce_uid} is in a saturated target. Setting data to placeholder value.')
        #     preprocessing_dict[tce_uid] = (
        #         preprocess_image_data_from_examples_in_saturated_targets(preprocessing_dict[tce_uid], final_size,
        #                                                                  upscale_f, number_of_imgs_to_sample))
        #     continue

        # get the list of valid quarters/sectors
        curr_tce_df = quality_metrics_tbl[quality_metrics_tbl['uid'] == tce_uid]
        # get quarters/sectors with data (i.e., quality metrics is a defined value)
        available_imgs_idxs = [q_i for q_i in range(n_max_imgs_avail)
                               if ~np.isnan(curr_tce_df[f'{prefix}{imgs_t[q_i]}_value'].values[0])]
        tces_info_tbl.loc[tces_info_tbl['uid'] == tce_uid, [f'available_{prefix}s']] = (
            ''.join(['1' if q_i in available_imgs_idxs else '0' for q_i in range(n_max_imgs_avail)]))

        # get quarters/sectors with valid data
        valid_images_idxs = []
        for i, q_i in enumerate(available_imgs_idxs):
            # quarter/sector is valid and uncertainty if
            # 1) target position uncertainty is not -1, which means target location is not available
            # 2) quality metric table shows 'TRUE' in the 'valid' field
            # 3) diff and oot images are not all NaNs
            if ((diff_img_data_dict[tce_uid]['target_ref_centroid'][i]['col']['uncertainty'] != -1 and
                 (curr_tce_df[f'{prefix}{imgs_t[q_i]}_valid'].item())) and
                    ~np.isnan(diff_img_data_dict[tce_uid]['image_data'][i][:, :, 1, 0]).all()
                    and ~np.isnan(diff_img_data_dict[tce_uid]['image_data'][i][:, :, 2, 0]).all()):
                valid_images_idxs.append(q_i)
        tces_info_tbl.loc[tces_info_tbl['uid'] == tce_uid, [f'valid_{prefix}s']] = \
            (''.join(['1' if q_i in valid_images_idxs else '0' for q_i in range(n_max_imgs_avail)]))

        n_valid_imgs = len(valid_images_idxs)

        if n_valid_imgs == 0:  # if no valid quarters/sectors

            log.info(f'[{proc_id}] No valid images for {tce_uid}. Setting data to placeholder value.')

            tces_info_tbl.loc[tces_info_tbl['uid'] == tce_uid, [f'valid_{prefix}s', f'sampled_{prefix}s']] = (
                '-' * n_max_imgs_avail, '-' * n_max_imgs_avail)

            # update data using placeholder values
            missing_data_placeholder = set_data_example_to_placeholder_values(final_size['x'] * upscale_f['x'],
                                                                              final_size['y'] * upscale_f['y'],
                                                                              number_of_imgs_to_sample)
            preprocessing_dict[tce_uid].update(missing_data_placeholder)

            continue

        # randomly sample valid quarters/sectors
        random_sample_imgs_idxs = sample_image_data(n_valid_imgs, valid_images_idxs, number_of_imgs_to_sample)

        tces_info_tbl.loc[tces_info_tbl['uid'] == tce_uid, [f'valid_{prefix}s', f'sampled_{prefix}s']] = (
            ''.join(['1' if q_i in valid_images_idxs else '0' for q_i in range(n_max_imgs_avail)]),
            ''.join([f'{(random_sample_imgs_idxs == q_i).sum()}' for q_i in range(n_max_imgs_avail)]))

        # get quality metrics for sampled quarters/sector runs
        sampled_qmetrics['sampled_qmetrics'][tce_i] = [curr_tce_df[f'{prefix}{imgs_t[q_i]}_value'].item()
                                                       for q_i in random_sample_imgs_idxs]

        # preprocess images
        for i, q_i in enumerate(random_sample_imgs_idxs):

            # get array index for sampled quarter/sector run
            arr_q_i = diff_img_data_dict[tce_uid]['image_number'].index(imgs_t[q_i])

            for option in ['target_centered', 'target_not_centered']:
                # get images and target position in the pixel frame
                diff_img = diff_img_data_dict[tce_uid]['image_data'][arr_q_i][:, :, 2, 0].copy()
                oot_img = diff_img_data_dict[tce_uid]['image_data'][arr_q_i][:, :, 1, 0].copy()
                snr_img = diff_img_data_dict[tce_uid]['image_data'][arr_q_i][:, :, 3, 0].copy()
                target_pos_col = float(diff_img_data_dict[tce_uid]['target_ref_centroid'][arr_q_i]['col']['value'])
                target_pos_row = float(diff_img_data_dict[tce_uid]['target_ref_centroid'][arr_q_i]['row']['value'])

                (diff_img_preproc, oot_img_preproc, snr_img_preproc, target_img, target_pos_col, target_pos_row,
                 target_col_disc, target_row_disc) = (
                    preprocess_single_diff_img_data_for_example(diff_img,
                                                                oot_img,
                                                                snr_img,
                                                                target_pos_col,
                                                                target_pos_row,
                                                                final_size['x'],
                                                                final_size['y'],
                                                                upscale_f['x'],
                                                                upscale_f['y'],
                                                                imgs_t[q_i],
                                                                tce_uid,
                                                                prefix,
                                                                log,
                                                                center_target=option == 'target_centered',
                                                                proc_id=proc_id))

                # add to dictionary
                suffix_str = '_tc' if option == 'target_centered' else ''

                # add image data
                preprocessing_dict[tce_uid]['images'][f'diff_imgs{suffix_str}'][i] = diff_img_preproc
                preprocessing_dict[tce_uid]['images'][f'oot_imgs{suffix_str}'][i] = oot_img_preproc
                preprocessing_dict[tce_uid]['images'][f'snr_imgs{suffix_str}'][i] = snr_img_preproc
                preprocessing_dict[tce_uid]['images'][f'target_imgs{suffix_str}'][i] = target_img

                # add target pixel coordinates
                preprocessing_dict[tce_uid]['target_position'][f'pixel_x{suffix_str}'][i] = target_pos_row
                preprocessing_dict[tce_uid]['target_position'][f'pixel_y{suffix_str}'][i] = target_pos_col

                # add discrete coordinates of target position relative to target pixel when expanding it to a 3x3 grid
                preprocessing_dict[tce_uid]['target_position'][f'subpixel_x{suffix_str}'][i] = target_row_disc
                preprocessing_dict[tce_uid]['target_position'][f'subpixel_y{suffix_str}'][i] = target_col_disc

            # add quality metric
            preprocessing_dict[tce_uid]['quality'][i] = (curr_tce_df[f'{prefix}{imgs_t[q_i]}_value'].item())

            # add current quarter/sector number to dictionary
            preprocessing_dict[tce_uid]['images_numbers'][i] = imgs_t[arr_q_i]

            if np.random.uniform() <= plot_prob:  # plot final images
                for suffix_str in ['', '_tc']:
                    plot_diff_img_data(
                        preprocessing_dict[tce_uid]['images'][f'diff_imgs{suffix_str}'][i],
                        preprocessing_dict[tce_uid]['images'][f'oot_imgs{suffix_str}'][i],
                        preprocessing_dict[tce_uid]['images'][f'snr_imgs{suffix_str}'][i],
                        preprocessing_dict[tce_uid]['images'][f'target_imgs{suffix_str}'][i],
                        {'x': preprocessing_dict[tce_uid]['target_position'][f'pixel_x{suffix_str}'][i],
                         'y': preprocessing_dict[tce_uid]['target_position'][f'pixel_y{suffix_str}'][i]},
                        preprocessing_dict[tce_uid]['quality'][i],
                        f'{prefix}_{preprocessing_dict[tce_uid]["images_numbers"][i]}',
                        tce_uid,
                        save_dir / 'plot_examples' /
                        f'{tce_uid}_diff_img_{preprocessing_dict[tce_uid]["images_numbers"][i]}{suffix_str}.png',
                    True)

        # last check for missing values
        missing_value_found = check_for_missing_values_in_preproc_diff_data(preprocessing_dict[tce_uid])

        if missing_value_found:
            log.info(f'[{proc_id}] At least one data array contained missing values for {tce_uid}. Setting data to '
                     f'placeholder value.')

            tces_info_tbl.loc[tces_info_tbl['uid'] == tce_uid,
            [f'valid_{prefix}s', f'sampled_{prefix}s']] = ('-' * n_max_imgs_avail, '-' * n_max_imgs_avail)

            # update data using placeholder values
            missing_data_placeholder = set_data_example_to_placeholder_values(final_size['x'] * upscale_f['x'],
                                                                              final_size['y'] * upscale_f['y'],
                                                                              number_of_imgs_to_sample)
            preprocessing_dict[tce_uid].update(missing_data_placeholder)

    # add quality metrics data
    tces_info_df = pd.concat([tces_info_tbl, pd.DataFrame(sampled_qmetrics)], axis=1, ignore_index=False)

    log.info(f'[{proc_id}] Finished preprocessing difference image data for {len(diff_img_data_dict)} examples.')

    return preprocessing_dict, tces_info_df


if __name__ == '__main__':

    # used in job arrays
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_fp', type=str, help='Configuration file with processing parameters.',
                        default='/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/codebase/diff_img/preprocessing/config_preprocessing.yaml')
    args = parser.parse_args()

    # load yaml file with run setup
    with(open(args.config_fp, 'r')) as file:
        config = yaml.safe_load(file)

    # mission; either `tess` or `kepler`
    mission = config['mission']

    # destination file path to preprocessed data
    dest_dir = Path(config['dest_dir'])
    dest_dir.mkdir(exist_ok=True)
    # save run setup into a yaml file
    with open(dest_dir / 'run_params.yaml', 'w') as setup_file:
        yaml.dump(config, setup_file, sort_keys=False)
    if config['plot_prob'] > 0:
        (dest_dir / 'plot_examples').mkdir(exist_ok=True)

    # set up logger
    logger = logging.getLogger(name=f'preprocess')
    logger_handler = logging.FileHandler(filename=dest_dir / f'preprocess.log', mode='w')
    logger_formatter = logging.Formatter('%(asctime)s - %(message)s')
    logger.setLevel(logging.INFO)
    logger_handler.setFormatter(logger_formatter)
    logger.addHandler(logger_handler)
    logger.info(f'Starting preprocessing...')

    # load difference image data
    logger.info(f'Loading difference image data from {config["diff_img_data_fp"]}')
    diff_img_data = np.load(config['diff_img_data_fp'], allow_pickle=True).item()
    # diff_img_data = {k: v for k, v in diff_img_data.items() if k == '11709244-2'}

    logger.info(f'Number of TCEs to preprocess: {len(diff_img_data)}')
    # load table with quality metrics
    logger.info(f'Loading quality metrics data from {config["qual_metrics_tbl_fp"]}')
    quality_metrics = pd.read_csv(config['qual_metrics_tbl_fp'])

    # load table with information on saturated target stars
    # logger.info(f'Loading saturated target data from {config["sat_tbl_fp"]}')
    # saturated = pd.read_csv(config['sat_tbl_fp'])
    # get only the TCE ids for saturated targets
    # saturated_tce_ids = saturated.loc[((saturated['mag'] < config['sat_thr']) &
    #                                    (saturated['uid'].isin(list(diff_img_data.keys())))), 'uid'].to_list()
    saturated_tce_ids = []

    logger.info(f'Number of saturated TCEs found: {len(saturated_tce_ids)}')

    # parallelize work; split by TCEs
    n_processes = config['n_processes']
    n_jobs = config['n_jobs']
    tces_ids = np.array_split(np.array(list(diff_img_data.keys())), n_jobs)
    pool = multiprocessing.Pool(processes=n_processes)
    jobs = [({tce_id: tce_diff_data for tce_id, tce_diff_data in diff_img_data.items() if tce_id in tces_ids_job},
             config['num_sampled_imgs'], config['upscale_f'], config['final_size'], quality_metrics, saturated_tce_ids,
             mission, dest_dir, None, config['plot_prob']) for job_i, tces_ids_job in enumerate(tces_ids)]
    async_results = [pool.apply_async(preprocess_diff_img_tces, job) for job in jobs]
    pool.close()
    pool.join()

    diff_img_data = {}
    tces_info_df = []
    for async_result in async_results:
        job_res = async_result.get()
        tces_info_df.append(job_res[1])
        diff_img_data.update(job_res[0])
    tces_info_df = pd.concat(tces_info_df, axis=0, ignore_index=True)

    logger.info(f'Saving preprocessed data to {dest_dir / "diffimg_preprocess.npy"}...')
    tces_info_df.to_csv(dest_dir / 'info_tces.csv', index=False)
    np.save(dest_dir / "diffimg_preprocess.npy", diff_img_data)

    logger.info(f'Finished.')
