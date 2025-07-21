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
import multiprocessing
import argparse
import yaml

# local
from src_preprocessing.diff_img.preprocessing.utils_diff_img import (
    plot_diff_img_data, create_neighbors_img, map_target_subpixel_location_to_discrete_grid,
    set_negative_value_oot_pixels, fill_missing_values_nearest_neighbors, center_images_to_target_pixel_location,
    crop_images_to_size, crop_images_to_valid_size, set_data_example_to_placeholder_values,
    check_for_missing_values_in_preproc_diff_data, sample_image_data, initialize_data_example_with_missing_values,
    create_target_image, resize_images_by_resampling, pad_images_by_extending_edges)

CENTER_OPTIONS = ['target_not_centered', 'target_centered']


def preprocess_single_diff_img_data_for_example(diff_img, oot_img, snr_img, target_pos_col, target_pos_row, size_h,
                                                size_w, size_f_h, size_f_w, img_n, tce_uid, prefix, center_target=True,
                                                neighbor_data=None, target_mag=None, exclude_neighbor_objs_outside=True,
                                                log=None, proc_id=-1):
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
        center_target: bool, if True the images are centered in the target pixel by padding through edge extension
        neighbor_data: list, for each sector, contains a dictionary where each key is the TIC ID of
            neighboring objects that maps to a dictionary with the column 'col_px' and row 'row_px' coordinates of these
            objects in the CCD pixel frame of the target star along with the corresponding magnitude 'TMag' and distance
            to the target in arcseconds 'dst_arcsec'.
        target_mag: float, target magnitude
        exclude_neighbor_objs_outside: bool, if True and `neighbor_data` is not None, neighboring objects that are
            outside the target mask are ignored when creating the neighbors image
        proc_id: str, process id
        log: logger

    Returns:
        diff_img, NumPy array for preprocessed difference image
        oot_img, NumPy array for preprocessed out-of-transit image
        snr_img, NumPy array for preprocessed snr image
        target_img, NumPy array for target location image
        target_pos, tuple for target column and row pixels
        target_pos, tuple int in {-1, 0, 1} for target subpixel column and row values
        neighbors_img, NumPy array for neighbors image
    """

    target_pos_col_input, target_pos_row_input = target_pos_col, target_pos_row

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
    diff_img, oot_img, snr_img, crop_min_col, crop_min_row = crop_images_to_valid_size(diff_img, oot_img, snr_img)

    # fill out missing values by using nearest neighbors with same weight
    fill_missing_filter = np.ones((3, 3))
    diff_img = fill_missing_values_nearest_neighbors(diff_img, fill_missing_filter)
    oot_img = fill_missing_values_nearest_neighbors(oot_img, fill_missing_filter)
    snr_img = fill_missing_values_nearest_neighbors(snr_img, fill_missing_filter)

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
    diff_img, oot_img, snr_img, pad_col, pad_row = pad_images_by_extending_edges(diff_img, oot_img, snr_img,
                                                                                 half_height, half_width)
    # update target position in padded image
    target_pos_col = target_pos_col - crop_min_col + pad_col
    target_pos_row = target_pos_row - crop_min_row + pad_row
    if neighbor_data is not None:
        for neighbor_id, neighbor_id_data in neighbor_data.items():
            neighbor_data[neighbor_id]['col_px'] = neighbor_id_data['col_px'] - crop_min_col + pad_col
            neighbor_data[neighbor_id]['row_px'] = neighbor_id_data['row_px'] - crop_min_row + pad_row

    # resize image using nearest neighbor interpolation to `size_f_h` * `size_f_w` times the target dimension
    diff_img, oot_img, snr_img = resize_images_by_resampling(diff_img, oot_img, snr_img, size_f_h, size_f_w)
    # update target position in resized image
    target_pos_col = target_pos_col * size_f_w + (size_f_w - 1) * 0.5
    target_pos_row = target_pos_row * size_f_h + (size_f_h - 1) * 0.5
    if neighbor_data is not None:
        for neighbor_id, neighbor_id_data in neighbor_data.items():
            neighbor_data[neighbor_id]['col_px'] = neighbor_id_data['col_px'] * size_f_w + (size_f_w - 1) * 0.5
            neighbor_data[neighbor_id]['row_px'] = neighbor_id_data['row_px'] * size_f_h + (size_f_h - 1) * 0.5

    if center_target:
        if np.isnan(target_pos_col_input):  # target location not available
            if log:
                log.info(
                    f'[{proc_id}] Target location from DV was not available ({target_pos_row_input}, '
                    f'{target_pos_col_input}). No centering performed for example {tce_uid} in {prefix} {img_n}.'
                    f' for example {tce_uid} in {prefix} {img_n}.')
        else:
            diff_img, oot_img, snr_img, center_col_offset, center_row_offset = (center_images_to_target_pixel_location(
                diff_img, oot_img, snr_img, target_pos_col, target_pos_row))
            # update target location after centering on target
            target_pos_col = target_pos_col + center_col_offset
            target_pos_row = target_pos_row + center_row_offset
            if neighbor_data is not None:
                for neighbor_id, neighbor_id_data in neighbor_data.items():
                    neighbor_data[neighbor_id]['col_px'] = neighbor_id_data['col_px'] + center_col_offset
                    neighbor_data[neighbor_id]['row_px'] = neighbor_id_data['row_px'] + center_row_offset

    # crop images to target dimension if they are larger
    diff_img, oot_img, snr_img, crop_size_col_offset, crop_size_row_offset = crop_images_to_size(diff_img,
                                                                                                 oot_img,
                                                                                                 snr_img,
                                                                                                 size_h * size_f_h,
                                                                                                 size_w * size_f_w)

    # update target pixel position after cropping image
    target_pos_col -= crop_size_col_offset
    target_pos_row -= crop_size_row_offset
    if neighbor_data is not None:
        for neighbor_id, neighbor_id_data in neighbor_data.items():
            neighbor_data[neighbor_id]['col_px'] = neighbor_id_data['col_px'] - crop_size_col_offset
            neighbor_data[neighbor_id]['row_px'] = neighbor_id_data['row_px'] - crop_size_row_offset

    # check if target pixel location is inside the image
    if (target_pos_col < 0 or target_pos_row < 0 or target_pos_col >= diff_img.shape[0] or
            target_pos_row >= diff_img.shape[1]):
        if log:
            log.info(f'[{proc_id}] Target pixel is outside of image after cropping for example '
                     f'{tce_uid} in {prefix} {img_n}.')

    # create target image
    target_img = create_target_image(size_h * size_f_h, size_w * size_f_w, target_pos_col, target_pos_row)

    if neighbor_data is not None:
        neighbors_img = create_neighbors_img(neighbor_data, diff_img.shape, target_mag, exclude_neighbor_objs_outside)
    else:
        neighbors_img = None

    return (diff_img, oot_img, snr_img, target_img, (target_pos_col, target_pos_row),
            (target_col_disc, target_row_disc), neighbors_img)


def preprocess_diff_img_tces(diff_img_data_fp, number_of_imgs_to_sample, upscale_f, final_size, mission_name,
                             save_dir, exclude_neighbor_objs_outside=True, log=None, plot_prob=0):
    """ Preprocessing pipeline for difference image data for a set of TCEs.

    Args:
        diff_img_data_fp: Path, to NumPy file with a dictionary. Each item is the difference image data for a given TCE.
            The TCE is identified by the string key. The value is a dictionary that contains six items:
            - 'target_ref_centroid' is a list of dictionaries that contain the value and uncertainty for the reference
            coordinates of the target star in the pixel domain in each observed sector;
            - 'image_data' is a list of NumPy array (n_rows, n_cols, n_imgs, 2) that contains the in-transit,
            out-of-transit, difference, and SNR flux images in this order (pixel values and uncertainties are addressed
            by the last dimension of the array, in this order) for each observe sector;
            - 'image_number' is a list that contains the integer sector number of the corresponding sequence of
            difference image data extracted for the TCE.
            - 'mag' is the target's magnitude.
            - 'neighbor_data' is a list that, for each sector, contains a dictionary where each key is the TIC ID of
            neighboring objects that maps to a dictionary with the column 'col_px' and row 'row_px' coordinates of these
            objects in the CCD pixel frame of the target star along with the corresponding magnitude 'TMag' and distance
            to the target in arcseconds 'dst_arcsec'.
        number_of_imgs_to_sample: int, number of quarters/sectors to sample
        upscale_f: dict, resize factor for 'x' and 'y' dimensions. Final size of images is
            (final_size['x'] * upscale_f['x'], final_size['y'] * upscale_f['y'])
        final_size: dict, image size before resizing (final_size['x'], final_size['y'])
        mission_name: str, mission from where the difference image data is from. Either `kepler` or `tess`
        save_dir: Path, destination  directory for preprocessed data
        exclude_neighbor_objs_outside: bool, if True and `neighbor_data` is not None, neighboring objects that are
            outside the target mask are ignored when creating the neighbors image
        log: logger
        plot_prob: float, probability to plot preprocessing results

    Returns:

    """

    save_dir.mkdir(exist_ok=True)

    if plot_prob > 0:
        (save_dir / 'plot_examples').mkdir(exist_ok=True)

    if log is None:
        # set up logger for the process
        log = logging.getLogger(name=f'preprocess_{diff_img_data_fp.stem}')
        logger_handler = logging.FileHandler(filename=save_dir / f'preprocess_{diff_img_data_fp.stem}.log', mode='a')
        logger_formatter = logging.Formatter('%(asctime)s - %(message)s')
        log.setLevel(logging.INFO)
        logger_handler.setFormatter(logger_formatter)
        log.addHandler(logger_handler)
        log.info(f'[{diff_img_data_fp.name}] Starting preprocessing...')

    # load difference image data
    log.info(f'Loading difference image data from {str(diff_img_data_fp)}')
    diff_img_data_dict = np.load(diff_img_data_fp, allow_pickle=True).item()

    log.info(f'Number of TCEs to preprocess: {len(diff_img_data_dict)}')

    if mission_name == 'kepler':
        prefix = 'quarter'
    elif mission_name == 'tess':
        prefix = 'sector'
    else:
        raise ValueError(f'Mission not recognized ({mission_name}). Set variable to `kepler` or `tess`.')

    # initialized TCE table with information on the preprocessing
    tces_info_dict = {
        'uid': [uid for uid in diff_img_data_dict],
        'saturated': [None] * len(diff_img_data_dict),
        f'num_available_{prefix}s': np.nan * np.ones(len(diff_img_data_dict)),
        f'num_valid_{prefix}s': np.nan * np.ones(len(diff_img_data_dict)),
        f'num_sampled_{prefix}s': np.nan * np.ones(len(diff_img_data_dict)),
        f'sampled_{prefix}s': [np.nan * np.ones(number_of_imgs_to_sample) for _ in diff_img_data_dict],
        'sampled_qmetrics': [np.nan * np.ones(number_of_imgs_to_sample) for _ in diff_img_data_dict],
        f'sampled_{prefix}s_missingvalues': ['no'] * len(diff_img_data_dict),
    }

    # initialize data dictionaries
    preprocessing_dict = {}
    for tce_i, tce_uid in enumerate(diff_img_data_dict):  # iterate over TCEs

        # initialize dictionary for the preprocessing results
        preprocessing_dict[tce_uid] = initialize_data_example_with_missing_values(final_size['x'] * upscale_f['x'],
                                                                                  final_size['y'] * upscale_f['y'],
                                                                                  number_of_imgs_to_sample)

        n_max_imgs_avail = len(diff_img_data_dict[tce_uid]['image_number'])

        if tce_i % 500 == 0:
            log.info(f'[{diff_img_data_fp.stem}] Preprocessed {tce_i + 1} example(s) out of {len(diff_img_data_dict)}.')

        # get quarters/sectors with data (i.e., quality metrics is a defined value)
        available_imgs_idxs = [img_idx for img_idx in range(n_max_imgs_avail)
                               if ~np.isnan(diff_img_data_dict[tce_uid]['quality_metric'][img_idx]['value'])]

        tces_info_dict[f'num_available_{prefix}s'][tce_i] = n_max_imgs_avail

        # get quarters/sectors with valid data
        valid_images_idxs = []
        for img_idx in available_imgs_idxs:
            # quarter/sector is valid and uncertainty if
            # 1) target position uncertainty is not -1, which means target location is not available
            # 2) quality metric table shows 'TRUE' in the 'valid' field
            # 3) diff and oot images are not all NaNs
            if ((diff_img_data_dict[tce_uid]['target_ref_centroid'][img_idx]['col']['uncertainty'] != -1 and
                 (diff_img_data_dict[tce_uid]['quality_metric'][img_idx]['valid'])) and
                    ~np.isnan(diff_img_data_dict[tce_uid]['image_data'][img_idx][:, :, 1, 0]).all()
                    and ~np.isnan(diff_img_data_dict[tce_uid]['image_data'][img_idx][:, :, 2, 0]).all()):
                valid_images_idxs.append(img_idx)

        n_valid_imgs = len(valid_images_idxs)

        tces_info_dict[f'num_valid_{prefix}s'][tce_i] = n_valid_imgs

        if n_valid_imgs == 0:  # if no valid quarters/sectors

            log.info(f'[{diff_img_data_fp.stem}] No valid images for {tce_uid}. Setting data to placeholder value.')

            # update data using placeholder values
            missing_data_placeholder = set_data_example_to_placeholder_values(
                final_size['x'] * upscale_f['x'],
                final_size['y'] * upscale_f['y'],
                number_of_imgs_to_sample,
                'neighbor_data' in diff_img_data_dict[tce_uid]
            )
            preprocessing_dict[tce_uid].update(missing_data_placeholder)

            continue

        # randomly sample valid quarters/sectors
        random_sample_imgs_idxs = sample_image_data(n_valid_imgs, valid_images_idxs, number_of_imgs_to_sample)
        tces_info_dict[f'num_sampled_{prefix}s'][tce_i] = len(random_sample_imgs_idxs)

        # get quality metrics for sampled quarters/sector runs
        tces_info_dict[f'sampled_{prefix}s'][tce_i] = (
            np.array([diff_img_data_dict[tce_uid]['image_number'][idx] for idx in random_sample_imgs_idxs]))
        tces_info_dict[f'sampled_qmetrics'][tce_i] = (
            np.array([diff_img_data_dict[tce_uid]['quality_metric'][idx]['value'] for idx in random_sample_imgs_idxs]))

        # preprocess images
        for sampled_img_idx in np.unique(random_sample_imgs_idxs):

            img_idxs = np.where(random_sample_imgs_idxs == sampled_img_idx)[0]

            for option in CENTER_OPTIONS:

                # get images and target position in the pixel frame
                diff_img = diff_img_data_dict[tce_uid]['image_data'][sampled_img_idx][:, :, 2, 0].copy()
                oot_img = diff_img_data_dict[tce_uid]['image_data'][sampled_img_idx][:, :, 1, 0].copy()
                # snr_img = diff_img_data_dict[tce_uid]['image_data'][sampled_img_idx][:, :, 3, 0].copy()
                snr_img = diff_img / diff_img_data_dict[tce_uid]['image_data'][sampled_img_idx][:, :, 2, 1]
                # check if target location is missing
                if float(diff_img_data_dict[tce_uid]['target_ref_centroid'][sampled_img_idx]['col']['value']) == -1:
                    target_pos_col, target_pos_row = np.isnan, np.isnan
                else:
                    target_pos_col = (
                        float(diff_img_data_dict[tce_uid]['target_ref_centroid'][sampled_img_idx]['col']['value']))
                    target_pos_row = (
                        float(diff_img_data_dict[tce_uid]['target_ref_centroid'][sampled_img_idx]['row']['value']))
                image_number = diff_img_data_dict[tce_uid]['image_number'][sampled_img_idx]
                target_mag = diff_img_data_dict[tce_uid]['mag']
                if 'neighbor_data' in diff_img_data_dict[tce_uid]:
                    neighbors_data = {neighbor_id: dict(neighbor_data) for neighbor_id, neighbor_data
                                      in diff_img_data_dict[tce_uid]['neighbor_data'][sampled_img_idx].items()}
                else:
                    neighbors_data = None

                (diff_img_preproc, oot_img_preproc, snr_img_preproc, target_img, target_pos, target_pos_disc,
                 neighbors_img) = (
                    preprocess_single_diff_img_data_for_example(
                        diff_img,
                        oot_img,
                        snr_img,
                        target_pos_col,
                        target_pos_row,
                        final_size['x'],
                        final_size['y'],
                        upscale_f['x'],
                        upscale_f['y'],
                        image_number,
                        tce_uid,
                        prefix,
                        center_target=option == 'target_centered',
                        neighbor_data=neighbors_data,
                        target_mag=target_mag,
                        exclude_neighbor_objs_outside=exclude_neighbor_objs_outside,
                        log=log,
                        proc_id=diff_img_data_fp.stem
                    )
                )

                # add to dictionary
                suffix_str = '_tc' if option == 'target_centered' else ''

                for img_idx in img_idxs:

                    # add image data
                    preprocessing_dict[tce_uid]['images'][f'diff_imgs{suffix_str}'][img_idx] = diff_img_preproc
                    preprocessing_dict[tce_uid]['images'][f'oot_imgs{suffix_str}'][img_idx] = oot_img_preproc
                    preprocessing_dict[tce_uid]['images'][f'snr_imgs{suffix_str}'][img_idx] = snr_img_preproc
                    preprocessing_dict[tce_uid]['images'][f'target_imgs{suffix_str}'][img_idx] = target_img

                    # add target pixel coordinates
                    preprocessing_dict[tce_uid]['target_position'][f'pixel_x{suffix_str}'][img_idx] = target_pos[1]
                    preprocessing_dict[tce_uid]['target_position'][f'pixel_y{suffix_str}'][img_idx] = target_pos[0]

                    # add discrete coordinates of target position relative to target pixel when expanding it to a nxn
                    # grid
                    preprocessing_dict[tce_uid]['target_position'][f'subpixel_x{suffix_str}'][img_idx] = (
                        target_pos_disc)[1]
                    preprocessing_dict[tce_uid]['target_position'][f'subpixel_y{suffix_str}'][img_idx] = (
                        target_pos_disc)[0]

                    # add neighbors image
                    if 'neighbor_data' in diff_img_data_dict[tce_uid]:
                        preprocessing_dict[tce_uid]['images'][f'neighbors_imgs{suffix_str}'][img_idx] = neighbors_img

            for img_idx in img_idxs:  # data that does not require centering the target star

                # add quality metric
                preprocessing_dict[tce_uid]['quality'][img_idx] = (
                    diff_img_data_dict)[tce_uid]['quality_metric'][sampled_img_idx]['value']

                # add current quarter/sector number to dictionary
                preprocessing_dict[tce_uid]['images_numbers'][img_idx] = (
                    diff_img_data_dict)[tce_uid]['image_number'][sampled_img_idx]

            if np.random.uniform() <= plot_prob:  # plot final images
                for option in CENTER_OPTIONS:

                    suffix_str = '_tc' if option == 'target_centered' else ''

                    diff_imgs_arr_aux = np.concatenate([
                        np.expand_dims(preprocessing_dict[tce_uid]['images'][f'oot_imgs{suffix_str}'][img_idxs[0]],
                                       axis=2),
                        np.expand_dims(preprocessing_dict[tce_uid]['images'][f'diff_imgs{suffix_str}'][img_idxs[0]],
                                       axis=2),
                        np.expand_dims(preprocessing_dict[tce_uid]['images'][f'snr_imgs{suffix_str}'][img_idxs[0]],
                                       axis=2)],
                        axis=2)

                    plot_diff_img_data(
                        diff_imgs_arr_aux,
                        (preprocessing_dict[tce_uid]['target_position'][f'pixel_y{suffix_str}'][img_idxs[0]],
                         preprocessing_dict[tce_uid]['target_position'][f'pixel_x{suffix_str}'][img_idxs[0]]),
                        save_dir / 'plot_examples' /
                        f'{tce_uid}_diff_img_'
                        f'{preprocessing_dict[tce_uid]["images_numbers"][img_idxs[0]]}{suffix_str}.png',
                        neighbors_img=preprocessing_dict[tce_uid]['images'][f'neighbors_imgs{suffix_str}'][img_idxs[0]]
                        if 'neighbor_data' in diff_img_data_dict[tce_uid] else None,
                        logscale=True
                    )

        # last check for missing values
        missing_value_found = check_for_missing_values_in_preproc_diff_data(preprocessing_dict[tce_uid])

        if missing_value_found:
            log.info(f'[{diff_img_data_fp.stem}] At least one data array contained missing values for {tce_uid}. '
                     f'Setting data to placeholder value.')

            tces_info_dict[f'sampled_{prefix}s_missingvalues'][tce_i] = 'yes'

            # update data using placeholder values
            missing_data_placeholder = set_data_example_to_placeholder_values(final_size['x'] * upscale_f['x'],
                                                                              final_size['y'] * upscale_f['y'],
                                                                              number_of_imgs_to_sample)
            preprocessing_dict[tce_uid].update(missing_data_placeholder)

    tces_info_df = pd.DataFrame(tces_info_dict)

    log.info(f'Saving preprocessed data to {save_dir / "diffimg_preprocess.npy"}...')
    tces_info_df.to_csv(save_dir / 'info_tces.csv', index=False)
    np.save(save_dir / "diffimg_preprocess.npy", preprocessing_dict)

    log.info(f'[{diff_img_data_fp.stem}] Finished preprocessing difference image data for {len(preprocessing_dict)} '
             f'examples.')


def preprocess_diff_img_tces_main(config_fp, save_dir=None, diff_img_dir=None):
    """ Wrapper to `preprocess_diff_img_tces()`.

    Args:
        config_fp: str, path to config file
        save_dir: str, path to directory where preprocessed difference image data will be saved
        diff_img_dir: str, path to directory with extracted difference image data to be preprocessed

    Returns:

    """

    # load yaml file with run setup
    with open(config_fp, 'r') as file:
        config = yaml.safe_load(file)

    if save_dir is not None:
        config['dest_root_dir'] = save_dir
    if diff_img_dir is not None:
        config['diff_img_data_dir'] = diff_img_dir

    diff_img_data_dir = Path(config['diff_img_data_dir'])

    dest_root_dir = Path(config['dest_root_dir'])

    # list of file paths to DV NumPy files for the sector runs to be preprocessed
    diff_img_data_fps = list(diff_img_data_dir.glob('*.npy'))

    dest_root_dir.mkdir(exist_ok=True)

    # set up logger
    logger = logging.getLogger(name=f'preprocess_main')
    logger_handler = logging.FileHandler(filename=dest_root_dir / f'preprocess_main.log', mode='a')
    logger_formatter = logging.Formatter('%(asctime)s - %(message)s')
    logger.setLevel(logging.INFO)
    logger_handler.setFormatter(logger_formatter)
    logger.addHandler(logger_handler)
    logger.info(f'Starting preprocessing...')

    logger.info(f'Found {len(diff_img_data_fps)} NumPy files with extracted difference image.')

    # parallelize work; split by NumPy files
    config['n_jobs'] = len(diff_img_data_fps)
    config['n_processes'] = min(config['n_processes'], config['n_jobs'])

    # save run setup into a yaml file
    with open(dest_root_dir / 'run_params.yaml', 'w') as setup_file:
        yaml.dump(config, setup_file, sort_keys=False)

    jobs = [(diff_img_data_fp, config['num_sampled_imgs'], config['upscale_f'], config['final_size'],
             config['mission'], dest_root_dir / diff_img_data_fp.stem, config['exclude_neighbor_objs_outside'], None,
             config['plot_prob'])
            for diff_img_data_fp in diff_img_data_fps]

    if config['n_processes'] > 1:
        # parallel
        pool = multiprocessing.Pool(processes=config['n_processes'])
        _ = [pool.apply_async(preprocess_diff_img_tces, job) for job in jobs]
        pool.close()
        pool.join()
    else:
        # sequential
        for job in jobs:
            preprocess_diff_img_tces(*job)

    logger.info('Finished preprocessing difference image data from NumPy files with extracted data from DV xml files.')


if __name__ == '__main__':

    # used in job arrays
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_fp', type=str, help='Configuration file with processing parameters.',
                        default='/path/to/codebase/src_preprocessing/diff_img/preprocessing/config_preprocessing.yaml')
    args = parser.parse_args()

    preprocess_diff_img_tces_main(args.config_fp)