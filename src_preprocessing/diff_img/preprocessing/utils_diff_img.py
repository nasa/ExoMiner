""" Utility functions for processing difference imaging. """

# 3rd party
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from PIL import Image
from scipy.signal import convolve2d

plt.switch_backend('agg')

MIN_IMG_VALUE = 1e-12


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
                if k2 == 'neighbors_imgs' and v2[0] is None: # neighbors data is not used
                    continue

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
            'neighbors_imgs': [np.nan * np.ones((size_h, size_w), dtype='float')
                               for _ in range(number_of_imgs_to_sample)],

            'diff_imgs_tc': [np.nan * np.ones((size_h, size_w), dtype='float')
                             for _ in range(number_of_imgs_to_sample)],
            'oot_imgs_tc': [np.nan * np.ones((size_h, size_w), dtype='float')
                            for _ in range(number_of_imgs_to_sample)],
            'snr_imgs_tc': [np.nan * np.ones((size_h, size_w), dtype='float')
                            for _ in range(number_of_imgs_to_sample)],
            'target_imgs_tc': [np.nan * np.ones((size_h, size_w), dtype='float')
                               for _ in range(number_of_imgs_to_sample)],
            'neighbors_imgs_tc': [np.nan * np.ones((size_h, size_w), dtype='float')
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
    data_placeholder = {
        'images': {
            'diff_imgs': [np.zeros((size_h, size_w), dtype='float')
                          for _ in range(number_of_imgs_to_sample)],
            'oot_imgs': [np.zeros((size_h, size_w), dtype='float')
                         for _ in range(number_of_imgs_to_sample)],
            'snr_imgs': [np.zeros((size_h, size_w), dtype='float')
                         for _ in range(number_of_imgs_to_sample)],
            'target_imgs': [np.zeros((size_h, size_w), dtype='float')
                            for _ in range(number_of_imgs_to_sample)],
            'neighbors_imgs': [np.zeros((size_h, size_w), dtype='float')
                               for _ in range(number_of_imgs_to_sample)],

            'diff_imgs_tc': [np.zeros((size_h, size_w), dtype='float')
                             for _ in range(number_of_imgs_to_sample)],
            'oot_imgs_tc': [np.zeros((size_h, size_w), dtype='float')
                            for _ in range(number_of_imgs_to_sample)],
            'snr_imgs_tc': [np.zeros((size_h, size_w), dtype='float')
                            for _ in range(number_of_imgs_to_sample)],
            'target_imgs_tc': [np.zeros((size_h, size_w), dtype='float')
                               for _ in range(number_of_imgs_to_sample)],
            'neighbors_imgs_tc': [np.zeros((size_h, size_w), dtype='float')
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

    # set target position to center of image
    for target_img_i in range(number_of_imgs_to_sample):
        data_placeholder['images']['target_imgs'][target_img_i][half_height, half_width] = 1
        data_placeholder['images']['target_imgs_tc'][target_img_i][half_height, half_width] = 1

    return data_placeholder


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


def crop_images_to_valid_size(diff_img, oot_img, snr_img):
    """ Crops images to their valid size, i.e., the minimum height and width that include any non-missing pixels.
    Missing pixels need to be represented by NaNs.

    Args:
        diff_img: NumPy array, difference image
        oot_img: NumPy array, out-of-transit image
        snr_img: NumPy array, snr image

    Returns:
        diff_img_crop: NumPy array, updated difference image
        oot_img_crop: NumPy array, updated out-of-transit image
        snr_img_crop: NumPy array, updated snr image
        min_col: int, min column pixel after cropping
        min_row: int, min row pixel after cropping
    """

    # find missing pixels
    idxs_not_missing = np.where(~np.isnan(diff_img))
    min_row, max_row = idxs_not_missing[0].min(), idxs_not_missing[0].max()
    min_col, max_col = idxs_not_missing[1].min(), idxs_not_missing[1].max()

    # choose smallest size that includes all valid pixels
    diff_img_crop = diff_img[min_row:max_row + 1, min_col: max_col + 1]
    oot_img_crop = oot_img[min_row:max_row + 1, min_col: max_col + 1]
    snr_img_crop = snr_img[min_row:max_row + 1, min_col: max_col + 1]

    return diff_img_crop, oot_img_crop, snr_img_crop, min_col, min_row


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
    valid_idxs = valid_px_arr_conv != 0
    img_conv[valid_idxs] = img_conv[valid_idxs] / valid_px_arr_conv[valid_idxs]
    img_conv[~valid_idxs] = np.nan

    # fill missing values with convolution values
    img_fill[idxs_nan] = img_conv[idxs_nan]

    return img_fill


def pad_images_by_extending_edges(diff_img, oot_img, snr_img, add_pad_h, add_pad_w):
    """ Pads images by extending edges. The padding is such that 1) makes sure the target pixel will not end up
    outside the cropping area when not centering the images on the target pixel, and 2) both dimensions have the same
    size.

    Args:
        diff_img: NumPy array, difference image
        oot_img: NumPy array, out-of-transit image
        snr_img: NumPy array, snr image
        add_pad_h: int, additional padding for the height dimension
        add_pad_w: int, additional padding for the width dimension

    Returns:
        diff_img_pad: NumPy array, updated difference image
        oot_img_pad: NumPy array, updated out-of-transit image
        snr_img_pad: NumPy array, updated snr image
        int, number of padded pixels column-wise
        int: number of padded pixels row-wise
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

    return diff_img_pad, oot_img_pad, snr_img_pad, pad_len_col[0], pad_len_row[0]


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
        int, column offset after centering target
        int, row offset after centering target
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
        # target_pos_row_tcenter = target_pos_row
        row_offset = 0
    else:  # target pixel location is before center row
        pad_len_h = [diff_img.shape[0] - 2 * target_pos_pixel_row - 1, 0]
        row_offset = pad_len_h[0]
        # target_pos_row_tcenter = target_pos_row + pad_len_h[0]

    if target_pos_pixel_col >= center_col:
        pad_len_w = [0, 2 * target_pos_pixel_col - diff_img.shape[1] + 1]
        col_offset = 0
        # target_pos_col_tcenter = target_pos_col
    else:
        pad_len_w = [diff_img.shape[1] - 2 * target_pos_pixel_col - 1, 0]
        col_offset = pad_len_w[1]
        # target_pos_col_tcenter = target_pos_col + pad_len_w[1]

    # pad images by extending edges
    diff_img_tcenter = np.pad(diff_img, (pad_len_h, pad_len_w), mode='edge')
    oot_img_tcenter = np.pad(oot_img, (pad_len_h, pad_len_w), mode='edge')
    snr_img_tcenter = np.pad(snr_img, (pad_len_h, pad_len_w), mode='edge')

    return diff_img_tcenter, oot_img_tcenter, snr_img_tcenter, col_offset, row_offset


def crop_images_to_size(diff_img, oot_img, snr_img, size_h, size_w):
    """ Crops images to a given size defined by `size_h` and `size_w`. The cropping is done around the center of the
    image.

    Args:
        diff_img: NumPy array, difference image
        oot_img: NumPy array, out-of-transit image
        snr_img: NumPy array, snr image
        size_h: int, crop to this height
        size_w: int, crop to this width

    Returns:
        diff_img_crop: NumPy array, updated difference image
        oot_img_crop: NumPy array, updated out-of-transit image
        snr_img_crop: NumPy array, updated snr image
        int, column offset after cropping to size
        int, row offset after cropping to size
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

    diff_img_crop = np.array(diff_img_crop)
    oot_img_crop = np.array(oot_img_crop)
    snr_img_crop = np.array(snr_img_crop)

    return diff_img_crop, oot_img_crop, snr_img_crop, left, upper


def resize_images_by_resampling(diff_img, oot_img, snr_img, size_f_h, size_f_w):
    """ Resizes images using nearest neighbor interpolation and resampling factors `size_f_h` and `size_f_w` for height
    and width, respectively.

    Args:
        diff_img: NumPy array, difference image
        oot_img: NumPy array, out-of-transit image
        oot_img: NumPy array, snr image
        size_f_h: float, resampling factor for height
        size_f_w: float, resampling factor for width

    Returns:
        diff_img_resize: NumPy array, updated difference image
        oot_img_resize: NumPy array, updated out-of-transit image
        snr_img_resize: NumPy array, updated snr image
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

    return diff_img_resize, oot_img_resize, snr_img_resize


def create_target_image(size_h, size_w, target_pos_pixel_col, target_pos_pixel_row):
    """ Creates target image.

    Args:
        size_h: int, height
        size_w: int, width
        target_pos_pixel_col: float, target pixel column coordinate
        target_pos_pixel_row: float, target pixel row coordinate

    Returns:
        target_img: NumPy array of target image

    """

    target_pos_pixel_col_up = int(np.round(target_pos_pixel_col))
    target_pos_pixel_row_up = int(np.round(target_pos_pixel_row))

    # initialize with all zeros
    target_img = np.zeros((size_h, size_w), dtype='float')

    # set target pixel in resized image with value of 1
    if (0 <= target_pos_pixel_row_up <= size_h - 1) and (0 <= target_pos_pixel_col_up <= size_w - 1):
        target_img[target_pos_pixel_col_up, target_pos_pixel_row_up] = 1

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

    if np.isnan(target_pos_row):
        return 0, 0

    target_pos_pixel_col = int(np.round(target_pos_col))
    target_pos_pixel_row = int(np.round(target_pos_row))

    target_col_diff = target_pos_pixel_col - target_pos_col
    target_row_diff = target_pos_pixel_row - target_pos_row

    target_col_disc = int(2 * target_col_diff)
    target_row_disc = int(2 * target_row_diff)

    return target_col_disc, target_row_disc


def create_neighbors_img(neighbor_data, img_shape, target_mag, exclude_objs_outside=True):
    """ Creates neighbors image based on the location and magnitude of neighbor objects in `neighbor_data`. The neighbor
    image is set to zero for pixels with no neighbors. The pixels with neighbors are set to the relative magnitude of
    the brightest neighbor to the target star.

    Args:
        neighbor_data: dict, each key is the ID of a neighbor that maps to a dictionary with the keys 'col_px',
        'row_px' and 'Tmag' that map to the column pixel, row pixel, and magnitude, respectively, of the neighbor object
        img_shape: tuple, desired image shape
        target_mag: float, target magnitude
        exclude_objs_outside: bool, if True objects that fall outside the image are not considered

    Returns:
        neighbor_img: NumPy array, neighbor image [`img_shape`]
    """

    if exclude_objs_outside:
        neighbor_data = {neighbor_id: neighbor_id_data
                         for neighbor_id, neighbor_id_data in neighbor_data.items()
                         if 0 <= neighbor_id_data['col_px'] < img_shape[1]
                         and 0 <= neighbor_id_data['row_px'] < img_shape[0]}

    n_neighbors = len(neighbor_data)
    if n_neighbors == 0:
        return np.zeros(img_shape + (n_neighbors,), dtype='float')

    # initialize with all infinity values
    neighbor_img = np.inf * np.ones(img_shape + (n_neighbors,), dtype='float')

    # sort neighbors from brightest to dimmest
    neighbor_data = dict(sorted(neighbor_data.items(), key=lambda item: item[1]['Tmag']))

    # # get minimum and maximum magnitudes for all neighbors including target star
    # min_mag, max_mag = min(target_mag, next(iter(neighbor_data.items()))), max(target_mag, list(neighbor_data.items())[-1])
    # delta_mag = max_mag - min_mag

    # compute relative neighbor-to-target magnitude ratio for each neighbor
    for neighbor_i, neighbor_id in enumerate(neighbor_data):
        neighbor_row, neighbor_col = (int(neighbor_data[neighbor_id]['row_px']),
                                      int(neighbor_data[neighbor_id]['col_px']))
        neighbor_img[neighbor_row, neighbor_col, neighbor_i] = target_mag / neighbor_data[neighbor_id]['Tmag']

    # in each pixel, choose only the brightest target
    neighbor_img = np.min(neighbor_img, axis=-1)

    # set pixels with no neighbors to zero (TMag = 0)
    neighbor_img[~np.isfinite(neighbor_img)] = 0

    return neighbor_img


def plot_diff_img_data(diff_imgs, target_coords, save_fp, neighbors_img=None, logscale=True):
    """ Plot difference image data for TCE in a given quarter/sector.

    Args:
        diff_imgs: NumPy array, difference image data [cols, rows, oot|diff|snr]
        target_coords: dict, target location 'x' and 'y'
        save_fp: Path, file path to saved plot
        neighbors_img: if not None, plots neighbors image
        logscale: bool, if True images color is set to log scale

    Returns:

    """

    def _create_subplot(ax_img, img, img_title, target_coords, logscale=True, mask_invalid_pixels=False):

        if mask_invalid_pixels:
            img = np.ma.masked_less(img, 0)
            cmap_img = plt.cm.viridis
            cmap_img.set_bad(color='gray')  # Set the color for non-positive values
        else:
            cmap_img = plt.cm.viridis

        if logscale:  # handle zero-valued pixels
            img[img == 0] = MIN_IMG_VALUE

        # plot image data
        im = ax_img.imshow(img, cmap=cmap_img, norm=LogNorm() if logscale else None, origin='lower')
        # set target location and magnitude
        _ = ax_img.scatter(target_coords[0], target_coords[1], marker='x', c='r', label='Target', zorder=2)

        # set color bars
        cbar_im = plt.colorbar(im, ax=ax_img, orientation='vertical', fraction=0.046, pad=0.04)
        # Set colorbar labels
        if img_title == 'Neighbors Image':
            cbar_im.set_label(r'Target-to-Neighbor Magnitude Ratio')
        else:
            cbar_im.set_label(r'Flux [$e^-/cadence$]')

        cbar_im.ax.set_position([cbar_im.ax.get_position().x1 - 0.02,
                                 cbar_im.ax.get_position().y0,
                                 cbar_im.ax.get_position().width,
                                 cbar_im.ax.get_position().height])

        ax_img.set_ylabel('Row')
        ax_img.set_xlabel('Col', labelpad=10)

        ax_img.set_title(img_title, pad=50)

    f, ax = plt.subplots(2, 2, figsize=(14, 14))

    # diff img
    _create_subplot(ax[0, 0], diff_imgs[:, :, 1], 'Difference Flux', target_coords, logscale=False)

    # oot img
    _create_subplot(ax[0, 1], diff_imgs[:, :, 0], 'Out-of-transit Flux', target_coords, logscale=logscale,
                    mask_invalid_pixels=True)

    # neighbors img
    if neighbors_img is not None:
        _create_subplot(ax[1, 0], neighbors_img, 'Neighbors Image', target_coords, logscale=False)

    # snr img
    _create_subplot(ax[1, 1], diff_imgs[:, :, 2], 'SNR Flux', target_coords, logscale=logscale,
                    mask_invalid_pixels=True)

    f.subplots_adjust(hspace=0.4, wspace=0.4)
    f.savefig(save_fp)
    plt.close()
