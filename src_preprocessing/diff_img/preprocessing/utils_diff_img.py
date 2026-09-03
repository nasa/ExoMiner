""" Utility functions for processing difference imaging. """

# 3rd party
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from PIL import Image
from scipy.signal import convolve2d

# local
from src_preprocessing.diff_img.preprocessing.placeholders_diff_img import set_data_example_to_placeholder_values

plt.switch_backend('agg')


def compute_neighbors_feature_vector(neighbor_data_tce_sector: dict, target_pos: tuple, max_neighbor_distance: float = 8, neighbor_delta_tmag_bound: list = [-12, 12], 
                                     top_k: int =5, gaussian_att_sigma: float = 1, eps=1e-10) -> np.ndarray:
    """
    Compute a compact, target-centric feature representation for the top-k neighboring TIC objects.

    Each neighbor is encoded as a fixed-length feature vector:
    [flux_ratio_norm, explanation_ratio, dx_norm, dy_norm, valid_flag]

    where:
    - flux_ratio_norm is the neighbor-to-target flux ratio, log10-transformed, clipped to a global Tmag-based bound, and rescaled to [0, 1]
    - explanation_ratio_norm is the neighbor-to-target flux to measured transit depth ratio , log10-transformed, clipped to a global bound, and rescaled to [0, 1]
    - dx_norm and dy_norm are the column and row offsets relative to the target, normalized
        by `max_neighbor_distance` (typically the neighbor search radius)
    - valid_flag is 1 for real neighbors and 0 for padded entries

    Padded neighbors (when fewer than top-k exist) have all features set to zero.

    Neighbors are ranked using a physics-motivated score based on linear flux ratio
    attenuated by a Gaussian kernel of their distance from the target; normalization is
    applied only after ranking.

    Parameters
    ----------
    neighbor_data_tce_sector : dict
        Mapping of neighbor TIC ID → {'Tmag', 'col_px', 'row_px', 'flux_ratio', 'explanation_ratio'}
    target_pos : tuple
        Target pixel position (column, row)
    max_neighbor_distance : float, optional
        Distance used to normalize positional offsets; typically the neighbor search radius
    neighbor_delta_tmag_bound : list, optional
        Global (min, max) ΔTmag bounds used to clip and normalize flux ratios
    top_k : int, optional
        Number of neighbors to include in the fixed-size representation
    gaussian_att_sigma : float, optional
        Width (in pixels) of the Gaussian attenuation used for neighbor ranking

    Returns
    -------
    np.ndarray
        Array of shape (top_k, n_features) encoding normalized neighbor features
    """

    N_FEATURES = 5  # flux ratio, colum offset, row offset, valid
    n_neighbors = len(neighbor_data_tce_sector)
    
    # Handle no neighbors
    if n_neighbors == 0:
        return np.zeros((top_k, N_FEATURES), dtype=float)

    neighbors_feature_vector = np.zeros((n_neighbors, N_FEATURES), dtype='float')
    
    for neighbor_i, (_, neighbor_data) in enumerate(neighbor_data_tce_sector.items()):
        
        # # compute delta tmag
        # delta_mag_n = neighbor_data['Tmag'] - target_tmag
        # # compute flux ratio
        # flux_ratio = 10.0 ** (-0.4 * delta_mag_n)
        flux_ratio = neighbor_data['flux_ratio']
        
        col_offset =  neighbor_data['col_px'] - target_pos[0]
        row_offset = neighbor_data['row_px'] - target_pos[1]
        
        explanation_ratio = neighbor_data['explanation_ratio']
        
        neighbors_feature_vector[neighbor_i] = [flux_ratio, explanation_ratio, col_offset, row_offset, 1]

    # compute squared distance to target
    sq_distance = neighbors_feature_vector[:, 2] ** 2 + neighbors_feature_vector[:, 3] ** 2
    
    # compute flux ratio over squared distance
    
    # rank neighbors (Gaussian attenuation)
    rank_score = neighbors_feature_vector[:, 0] * np.exp(- sq_distance / (2 * gaussian_att_sigma ** 2))
    
    # choose top-k neighbors
    top_idxs = np.argsort(rank_score)[::-1][:top_k]    
    chosen_neighbors_feature_vector = neighbors_feature_vector[top_idxs]
    
    # in case there are fewer than top-k neighbors
    n_chosen_n = len(chosen_neighbors_feature_vector)
    neighbors_padded = np.zeros((top_k, N_FEATURES), dtype='float')
    neighbors_padded[:n_chosen_n] = chosen_neighbors_feature_vector
    
    # normalize offsets by max neighbor distance
    neighbors_padded[:, [2, 3]] /= max_neighbor_distance
    
    # normalize flux ratio: log-transform, then clip, then rescale to [0, 1]
    min_log_flux_ratio, max_log_flux_ratio = -0.4 * neighbor_delta_tmag_bound[1],  -0.4 * neighbor_delta_tmag_bound[0] 
    neighbors_padded[:, 0] = np.clip(np.log10(neighbors_padded[:, 0] + eps), min_log_flux_ratio, max_log_flux_ratio)
    neighbors_padded[:, 0] = (neighbors_padded[:, 0] - min_log_flux_ratio) / (max_log_flux_ratio - min_log_flux_ratio)
    
    # normalize explanation ratio: log-transform, then clip, then rescale to [0, 1]
    # log_tr_obs = [np.log10(1e-5), np.log10(1e-1)]
    neighbors_padded[:, 1] = np.clip(np.log10(neighbors_padded[:, 1] + eps), -4, 4)
    neighbors_padded[:, 1] = (neighbors_padded[:, 1] + 4) / 8
    
    return neighbors_padded
    

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


def check_for_missing_values_in_preproc_diff_data(data: dict, has_neighbor_data: bool) -> bool:
    """ Checks for missing values (NaNs) in the preprocessed data.

    Args:
        data: dict, preprocessed data
        has_neighbor_data: bool, flag for whether neighbor data should be considered

    Returns:
        bool, True if there's at least a missing value in any preprocessed data.

    """

    missing_value_found = False

    for k, v in data.items():
        # check 1-D list variables
        if k in ['quality', 'images_numbers', 'neighbors_feats']:
            if k == 'neighbors_feats' and not has_neighbor_data:
                continue
            
            missing_value_found = np.isnan(v).sum() > 0
            if missing_value_found:
                return missing_value_found
        else:  # check dictionaries with N-D list variables like 'images' and 'target_position'
            for k2, v2 in v.items():
                if k2 in ['neighbors_imgs', 'neighbors_imgs_tc'] and not has_neighbor_data: # neighbors data is not used
                    continue

                missing_value_found = np.isnan(v2).sum() > 0

                if missing_value_found:
                    return missing_value_found

    return missing_value_found


def initialize_data_example_with_missing_values(size_h, size_w, number_of_imgs_to_sample, top_k_neighbors, n_feats_neighbors=5):
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
            
            'validpxs_imgs': [np.nan * np.ones((size_h, size_w), dtype='float')
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
            
            'validpxs_imgs_tc': [np.nan * np.ones((size_h, size_w), dtype='float')
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
            'subpixel_y_tc': [np.nan] * number_of_imgs_to_sample,
            
            'target_positon_res': [[0, 0, 0]] * number_of_imgs_to_sample,
            'target_positon_res_tc': [[0, 0, 0]] * number_of_imgs_to_sample,
        },
        
        'neighbors_feats': np.zeros((number_of_imgs_to_sample, top_k_neighbors, n_feats_neighbors)),
        
        'quality': [np.nan] * number_of_imgs_to_sample,
        
        'images_numbers': [np.nan] * number_of_imgs_to_sample,
    }
    
    return initialized_data_dict


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


def crop_images_to_valid_size(diff_img, oot_img, snr_img, valid_pxs_img):
    """ Crops images to their valid size, i.e., the minimum height and width that include any non-missing pixels.
    Missing pixels need to be represented by NaNs.

    Args:
        diff_img: NumPy array, difference image
        oot_img: NumPy array, out-of-transit image
        snr_img: NumPy array, snr image
        valid_pxs_img: NumPy array, valid pixels image

    Returns:
        diff_img_crop: NumPy array, updated difference image
        oot_img_crop: NumPy array, updated out-of-transit image
        snr_img_crop: NumPy array, updated snr image
        valid_pxs_img_crop: NumPy array, updated valid pixels image
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
    valid_pxs_img_crop = valid_pxs_img[min_row:max_row + 1, min_col: max_col + 1]

    return diff_img_crop, oot_img_crop, snr_img_crop, valid_pxs_img_crop, min_col, min_row


def fill_missing_values_nearest_neighbors(img, window=[3, 3], max_missing_pxs_neighborhood=3):
    """ Fills missing values in an image through convolution of a kernel window. The pixels with missing value are set
    to zero during the convolution. The missing values need to be represented by NaNs. If all pixels in the window
    are missing, the output value for the respective pixel is also NaN.

    Args:
        img: NumPy array, image
        window: NumPy array, kernel
        max_missing_pxs_neighborhood: int, maximum number of missing pixels allowed in the `window`

    Returns:
        img_fill, NumPy array of image with filled missing values
        valid_idxs, the pixels that were succesfully imputed using their local region
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
    valid_idxs = valid_px_arr_conv >= max_missing_pxs_neighborhood
    img_conv[valid_idxs] = img_conv[valid_idxs] / valid_px_arr_conv[valid_idxs]
    img_conv[~valid_idxs] = np.nan

    # fill missing values with convolution values
    img_fill[idxs_nan] = img_conv[idxs_nan]
    
    filled_pxs_idxs = idxs_nan & ~np.isnan(img_fill)

    return img_fill, filled_pxs_idxs


def pad_images_by_extending_edges(diff_img, oot_img, snr_img, valid_pxs_img, add_pad_h, add_pad_w):
    """ Pads images by extending edges. The padding is such that 1) makes sure the target pixel will not end up
    outside the cropping area when not centering the images on the target pixel, and 2) both dimensions have the same
    size.

    Args:
        diff_img: NumPy array, difference image
        oot_img: NumPy array, out-of-transit image
        snr_img: NumPy array, snr image
        valid_pxs_img: NumPy array, valid pixels image
        add_pad_h: int, additional padding for the height dimension
        add_pad_w: int, additional padding for the width dimension

    Returns:
        diff_img_pad: NumPy array, updated difference image
        oot_img_pad: NumPy array, updated out-of-transit image
        snr_img_pad: NumPy array, updated snr image
        valid_pxs_img_pad: NumPy array, updated valid pixels image
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
    valid_pxs_img_pad = np.pad(valid_pxs_img, (pad_len_row, pad_len_col), mode='edge')

    return diff_img_pad, oot_img_pad, snr_img_pad, valid_pxs_img_pad, pad_len_col[0], pad_len_row[0]


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
    target_pos_pixel_col = int(round(target_pos_col))
    target_pos_pixel_row = int(round(target_pos_row))

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
        col_offset = pad_len_w[0]
        # target_pos_col_tcenter = target_pos_col + pad_len_w[1]

    # pad images by extending edges
    diff_img_tcenter = np.pad(diff_img, (pad_len_h, pad_len_w), mode='edge')
    oot_img_tcenter = np.pad(oot_img, (pad_len_h, pad_len_w), mode='edge')
    snr_img_tcenter = np.pad(snr_img, (pad_len_h, pad_len_w), mode='edge')

    return diff_img_tcenter, oot_img_tcenter, snr_img_tcenter, col_offset, row_offset


def create_target_offset_relative_to_image_center_feature(
    target_pos_col: float,
    target_pos_row: float,
    img_width: int,
    img_height: int,
    clip_range: tuple = (-2.0, 2.0),
    image_valid: int = 1,
) -> list:
    """
    Create a target-offset feature relative to the image center.

    Returns:
        [dx_norm_clipped, dy_norm_clipped, image_valid, target_outside_crop]
    """

    if np.isnan(target_pos_col) or np.isnan(target_pos_row):
        return [0, 0, 0, 0]
    
    # geometric center for even-sized crops
    center_col = (img_width - 1) / 2
    center_row = (img_height - 1) / 2

    # raw offsets
    dx = target_pos_col - center_col
    dy = target_pos_row - center_row

    # normalize
    dx_norm = dx / (img_width / 2)
    dy_norm = dy / (img_height / 2)

    # semantic outside-crop flag (NOT tied to clipping)
    target_outside_crop = int(abs(dx_norm) > 1.0 or abs(dy_norm) > 1.0)

    # clip only the continuous variables
    dx_clip = np.clip(dx_norm, clip_range[0], clip_range[1])
    dy_clip = np.clip(dy_norm, clip_range[0], clip_range[1])

    return [dx_clip, dy_clip, image_valid, target_outside_crop]


def center_images_to_target_centroid(diff_img:np.ndarray, oot_img:np.ndarray, snr_img: np.ndarray, valid_pxs_img: np.ndarray,
                                     target_pos_col: float, target_pos_row: float, out_h: int, out_w: int) -> tuple:
    """ Centers images so that the target location is as close as possible to the image center (sub-pixel optimal).

    Args:
        diff_img: NumPy array, difference image
        oot_img: NumPy array, out-of-transit image
        snr_img: NumPy array, snr image
        valid_pxs_img: NumPy array, valid pixels image
        target_pos_col: float, target position column coordinate
        target_pos_row: float, target position row coordinate
        out_h: int, output image height
        out_w: int, output image width

    Returns:
        NumPy array, centered difference image
        NumPy array, centered out-of-transit image
        NumPy array, centered snr image
        NumPy array, centered valid pixels image
        float, target's new column position
        float, target's new row position
        float, column offset due to centering
        float, row offset due to centering
        list (float, float, int {0|1}), column and row offset residuals of target position after centering target, 
            and valid residual offset (set to 1)
    """
    
    # assert out_h % 2 == 1 and out_w % 2 == 1, "Output size must be odd"

    H, W = diff_img.shape

    # nearest integer pixel to the target
    tgt_col_px = int(np.floor(target_pos_col + 0.5))
    tgt_row_px = int(np.floor(target_pos_row + 0.5))

    half_h = out_h // 2
    half_w = out_w // 2

    # desired bounds in input image coordinates
    row_min = tgt_row_px - half_h
    row_max = tgt_row_px + half_h + 1
    col_min = tgt_col_px - half_w
    col_max = tgt_col_px + half_w + 1

    # compute padding if window exceeds image bounds
    pad_top    = max(0, -row_min)
    pad_bottom = max(0, row_max - H)
    pad_left   = max(0, -col_min)
    pad_right  = max(0, col_max - W)

    pad_h = (pad_top, pad_bottom)
    pad_w = (pad_left, pad_right)

    def _pad(img):
        return np.pad(img, (pad_h, pad_w), mode="edge")

    diff_p = _pad(diff_img)
    oot_p  = _pad(oot_img)
    snr_p  = _pad(snr_img)
    valid_pxs_p  = np.pad(valid_pxs_img, (pad_h, pad_w), mode="constant", constant_values=1)

    # shift bounds into padded frame
    row_min += pad_top
    row_max += pad_top
    col_min += pad_left
    col_max += pad_left

    # extract centered window
    diff_c = diff_p[row_min:row_max, col_min:col_max]
    oot_c  = oot_p[row_min:row_max,  col_min:col_max]
    snr_c  = snr_p[row_min:row_max,  col_min:col_max]
    valid_pxs_p  = valid_pxs_p[row_min:row_max,  col_min:col_max]

    # offset from the original closest pixel to target position to center of output image
    dcol = tgt_col_px - half_w
    drow = tgt_row_px - half_h
    
    # target location in the output frame
    target_col_out = target_pos_col - dcol
    target_row_out = target_pos_row - drow

    # # residual sub-pixel offset relative to window center
    # dcol_res = target_col_out - half_w
    # drow_res = target_row_out - half_h
    
    offset_res = create_target_offset_relative_to_image_center_feature(target_col_out, target_row_out, out_w, out_h)

    return diff_c, oot_c, snr_c, valid_pxs_p, target_col_out, target_row_out, dcol, drow, offset_res # [dcol_res, drow_res, 1]


def crop_images_to_size(diff_img, oot_img, snr_img, valid_pxs_img, size_h, size_w):
    """ Crops images to a given size defined by `size_h` and `size_w`. The cropping is done around the center of the
    image.

    Args:
        diff_img: NumPy array, difference image
        oot_img: NumPy array, out-of-transit image
        snr_img: NumPy array, snr image
        snr_img: NumPy array, valid pixels image
        size_h: int, crop to this height
        size_w: int, crop to this width

    Returns:
        diff_img_crop: NumPy array, updated difference image
        oot_img_crop: NumPy array, updated out-of-transit image
        snr_img_crop: NumPy array, updated snr image
        valid_pxs_img_crop: NumPy array, updated valid pixels image
        int, column offset after cropping to size
        int, row offset after cropping to size
    """

    # convert to PIL Image object
    diff_img = Image.fromarray(diff_img)
    oot_img = Image.fromarray(oot_img)
    snr_img = Image.fromarray(snr_img)
    valid_pxs_img = Image.fromarray(valid_pxs_img.astype('float'))
    
    # crop images to target dimension if they are larger    
    W, H = diff_img.size            # PIL order: width, height
    half_h, half_w = size_h // 2, size_w // 2
    center_x, center_y = W // 2, H // 2

    left  = max(0, center_x - half_w)
    upper = max(0, center_y - half_h)
    right = min(W, left + size_w)   # ensure exact width
    lower = min(H, upper + size_h)  # ensure exact height
        
    diff_img_crop = np.array(diff_img.crop((left, upper, right, lower)))
    oot_img_crop  = np.array(oot_img.crop((left, upper, right, lower)))
    snr_img_crop  = np.array(snr_img.crop((left, upper, right, lower)))
    valid_pxs_img_crop  = np.array(valid_pxs_img.crop((left, upper, right, lower))).astype('int')

    return diff_img_crop, oot_img_crop, snr_img_crop, valid_pxs_img_crop, left, upper


def resize_images_by_resampling(diff_img, oot_img, snr_img, valid_pxs_img, size_f_h, size_f_w):
    """ Resizes images using nearest neighbor interpolation and resampling factors `size_f_h` and `size_f_w` for height
    and width, respectively.

    Args:
        diff_img: NumPy array, difference image
        oot_img: NumPy array, out-of-transit image
        snr_img: NumPy array, snr image
        valid_pxs_img: NumPy array, valid pixels image
        size_f_h: float, resampling factor for height
        size_f_w: float, resampling factor for width

    Returns:
        diff_img_resize: NumPy array, updated difference image
        oot_img_resize: NumPy array, updated out-of-transit image
        snr_img_resize: NumPy array, updated snr image
        valid_pxs_img_resize: NumPy array, updated valid pixels image
    """

    size_h, size_w = diff_img.shape

    # convert back to PIL Image object
    diff_img = Image.fromarray(diff_img)
    oot_img = Image.fromarray(oot_img)
    snr_img = Image.fromarray(snr_img)
    valid_pxs_img = Image.fromarray(valid_pxs_img.astype('float'))

    diff_img_resize = diff_img.resize(size=(size_w * size_f_w, size_h * size_f_h), resample=Image.Resampling.NEAREST)
    oot_img_resize = oot_img.resize(size=(size_w * size_f_w, size_h * size_f_h), resample=Image.Resampling.NEAREST)
    snr_img_resize = snr_img.resize(size=(size_w * size_f_w, size_h * size_f_h), resample=Image.Resampling.NEAREST)
    valid_pxs_img_resize = valid_pxs_img.resize(size=(size_w * size_f_w, size_h * size_f_h), resample=Image.Resampling.NEAREST)

    # convert back to NumPy array
    diff_img_resize = np.array(diff_img_resize)
    oot_img_resize = np.array(oot_img_resize)
    snr_img_resize = np.array(snr_img_resize)
    valid_pxs_img_resize = np.array(valid_pxs_img_resize)

    return diff_img_resize, oot_img_resize, snr_img_resize, valid_pxs_img_resize


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
        target_img[target_pos_pixel_row_up, target_pos_pixel_col_up] = 1

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


def create_neighbors_img(neighbor_data: dict, img_shape: tuple, target_mag: float,  tmag_diff_range: tuple =(-10, 15)) -> np.ndarray:
    """ Creates neighbors image based on the location and magnitude of neighbor objects in `neighbor_data`. 
    The pixels with neighbors are set to the magnitude difference betweent the target and the brightest neighbor in each pixel. 
    Pixels without neighbors are set to -np.inf. `tmag_diff_range` is used to clip results.

    Args:
        neighbor_data: dict, each key is the ID of a neighbor that maps to a dictionary with the keys 'col_px',
        'row_px' and 'Tmag' that map to the column pixel, row pixel, and magnitude, respectively, of the neighbor object
        img_shape: tuple, desired image shape
        target_mag: float, target magnitude
        tmag_diff_range: tuple, tmag difference range for neighbors images; values outside this range are clipped. Default (-10, 15)
    Returns:
        neighbor_img: NumPy array, neighbor image [`img_shape`]
    """

    n_neighbors = len(neighbor_data)
    
    # initialize with all infinity values
    neighbor_img = -1 * np.inf * np.ones(img_shape + (max(1, n_neighbors),), dtype='float')    
    
    if n_neighbors > 0:
        
        # sort neighbors from brightest to dimmest
        neighbor_data = dict(sorted(neighbor_data.items(), key=lambda item: item[1]['Tmag']))

        # compute tmag difference between target and neighbor
        for neighbor_i, neighbor_id in enumerate(neighbor_data):
            # round pixel locations to nearest integer
            neighbor_row, neighbor_col = (int(round(neighbor_data[neighbor_id]['row_px'])),
                                        int(round(neighbor_data[neighbor_id]['col_px'])))
            # check if neighbor pixel location is within image boundaries
            if neighbor_col >= 0 and neighbor_col < img_shape[1] and neighbor_row >=0 and neighbor_row < img_shape[0]:
                neighbor_img[neighbor_row, neighbor_col, neighbor_i] = target_mag - neighbor_data[neighbor_id]['Tmag']

    # in each pixel, choose only the brightest target (i.e., max tmag difference)
    neighbor_img = np.nanmax(neighbor_img, axis=-1)
    
    # clip tmag difference range
    neighbor_img = np.clip(neighbor_img, tmag_diff_range[0], tmag_diff_range[1])

    return neighbor_img


def plot_diff_img_data(diff_imgs, target_coords, save_fp, valid_pxs_img=None, neighbors_img=None, neighbors_coords=None, logscale=True, title_str=''):
    """ Plot difference image data for TCE in a given quarter/sector.

    Args:
        diff_imgs: NumPy array, difference image data [row, col, channels - oot|diff|snr]
        target_coords: dict, target location col 'x' and row 'y'
        save_fp: Path, file path to saved plot
        valid_pxs_img: NumPy array, valid pixels image
        neighbors_img: if not None, plots neighbors image
        neighbors_coords: if not None, dict with neighbors location col 'x' and row 'y'
        logscale: bool, if True images color is set to log scale
        title_str: str, title string with auxiliary information

    Returns:

    """

    def _create_subplot(ax_img, img, img_title, target_coords, logscale=True, mask_invalid_pixels=False, min_img_value=1e-12):
        """Creates a subplot for a given image.

        :param matplotlib ax subplot ax_img: subplot axis
        :param NumPy array img: image data
        :param str img_title: image title
        :param tuple target_coords: target coordinates (col, row)
        :param bool logscale: sets log scale for image plot, defaults to True
        :param bool mask_invalid_pixels: masks negative pixels and sets them to gray, defaults to False
        :param float min_img_value: zero-value pixels are set to this number to still show them when using log scale
        """

        if mask_invalid_pixels:
            img = np.ma.masked_less(img, 0)
            cmap_img = plt.cm.viridis
            cmap_img.set_bad(color='gray')  # Set the color for non-positive values
        else:
            cmap_img = plt.cm.viridis

        if logscale:  # handle zero-valued pixels
            img[img == 0] = min_img_value

        # plot image data
        if img_title == 'Valid Pixels Image':
            im = ax_img.imshow(img, cmap=cmap_img, origin='lower', vmin=0, vmax=1)
        else:
            im = ax_img.imshow(img, cmap=cmap_img, norm=LogNorm() if logscale else None, origin='lower')
        
        # set target location and magnitude
        _ = ax_img.scatter(target_coords[0], target_coords[1], marker='x', c='r', label='Target', zorder=2)
        
        # overlay neighbor positions on Neighbors Image panel
        if img_title == 'Neighbors Image' and neighbors_coords:
            # neighbors_coords is a list of (col, row) tuples
            cols, rows = zip(*neighbors_coords) if len(neighbors_coords) > 0 else ([], [])
            ax_img.scatter(cols, rows, marker='*', color='white', s=18, facecolors='none', edgecolors='white', linewidths=0.7, alpha=0.5, zorder=2, label='Neighbors')

            # set axis limits to avoid extra padding when neighbors are close to image edges
            H, W = img.shape
            extent = (-0.5, W - 0.5, -0.5, H - 0.5)
            ax_img.set_xlim(extent[0], extent[1])
            ax_img.set_ylim(extent[2], extent[3])
            ax_img.margins(0)              # no extra padding
            ax_img.set_autoscale_on(False) # disable autoscale after we set limits

        # set color bars
        # if img_title not in ['Valid Pixels Image']:
        cbar_im = plt.colorbar(im, ax=ax_img, orientation='vertical', fraction=0.046, pad=0.04)
        
        # set colorbar labels
        if img_title == 'Neighbors Image':
            cbar_im.set_label(r'$T_{Mag, Target}-T_{Mag, Neighbor}$')
        elif img_title not in ['SNR Flux', 'Valid Pixels Image']:
            cbar_im.set_label(r'Flux [$e^-/cadence$]')

        # # if img_title not in ['Valid Pixels Image']:
        # cbar_im.ax.set_position([cbar_im.ax.get_position().x1 - 0.02,
        #                         cbar_im.ax.get_position().y0,
        #                         cbar_im.ax.get_position().width,
        #                         cbar_im.ax.get_position().height])

        ax_img.set_ylabel('Row')
        ax_img.set_xlabel('Col', labelpad=10)

        ax_img.set_title(img_title, pad=15)
    
    f = plt.figure(figsize=(20, 14))
    
    gs = f.add_gridspec(ncols=3, nrows=2)

    # diff img; linear scale is preferred to better visualize both positive and negative fluxes
    ax = f.add_subplot(gs[0, 0])
    _create_subplot(ax, diff_imgs[:, :, 1], 'Difference Flux', target_coords, logscale=False)

    # oot img; by design, out-of-transit pixels cannot have non-positive values
    ax = f.add_subplot(gs[0, 1])
    _create_subplot(ax, diff_imgs[:, :, 0], 'Out-of-transit Flux', target_coords, logscale=logscale, mask_invalid_pixels=True)

    # snr img
    ax = f.add_subplot(gs[0, 2])
    _create_subplot(ax, diff_imgs[:, :, 2], 'SNR Flux', target_coords, logscale=False, mask_invalid_pixels=False)
    
    # valid pixels img
    ax = f.add_subplot(gs[1, 0])
    _create_subplot(ax, valid_pxs_img, 'Valid Pixels Image', target_coords, logscale=False)
    
    # neighbors img
    if neighbors_img is not None:
        ax = f.add_subplot(gs[1, 1])
        _create_subplot(ax, neighbors_img, 'Neighbors Image', target_coords, logscale=False)

    f.suptitle(title_str)
    
    # add a text box at the bottom center of the figure
    f.text(0.5, 0.01, 'x: target catalog position', 
           ha='center', va='bottom', fontsize=12,
           bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='gray', alpha=0.8))
    
    # ajust tight_layout to leave room at the bottom for the text box
    f.tight_layout(rect=[0, 0.05, 1, 0.93]) 
    
    f.savefig(save_fp)
    plt.close()
