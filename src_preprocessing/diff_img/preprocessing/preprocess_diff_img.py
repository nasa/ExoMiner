"""
Preprocess extracted difference image data.

Extracted difference data is assumed to be stored as NumPy files "*.npy" stored under `diff_img_data_dir`, each with a dictionary with the 
following structure:
    - tce_uid '12345678-1-S1' (e.g.)
        - image_number list of sectors/quarters with available data
        - quality_metric: list of dicts for quality metric in each sector/quarter, each dict with keys 'value', 'valid', and 'attempted'
        - mag: target TMag
        - image_data: list of NumPy arrays [height, width, img_type, value/uncertainty], where img_type channel is in-transit, out-of-transit, difference image, and "SNR"
        - target_ref_centroid: list of dicts that contain the value and uncertainty for the reference coordinates of the target star in the pixel domain in each observed sector/quarter
            - {row: {value: X, uncertainty: Y}, col: {value: X, uncertainty: Y}}
            - ...
        - neighbor_data: dict of dicts for each sector/quarter with neighbors data; each sector/quarter dict maps the TIC ID of a neighbor to a dictionary with the column 'col_px' and row 'row_px' coordinates of these
            objects in the CCD pixel frame of the target star along with the corresponding magnitude 'TMag' and distance to the target in arcseconds 'dst_arcsec'.
            - sectorX
                - neighbor_1
                    - ra
                    - dec
                    - target_id
                    - col_px
                    - row_px
                    - Tmag
                    - flux/transit_depth ratio (flux ratio between neighbor and target / transit depth): flux_n / flux_t / tce_depth
                - ...
            - ...
            
--- Output Structure ---

preprocessing dictionary

    - tce_uid '12345678-1-S1' (e.g.)
        - images
            - diff_imgs_<center_suffix>: list of NumPy arrays [n_sectors, [height, width]] for preprocessed difference flux image (centered or not on target)
            - oot_imgs_<center_suffix>: list of NumPy arrays [n_sectors, [height, width]] for preprocessed out-of-transit flux image (centered or not on target)
            - snr_imgs_<center_suffix>: list of NumPy arrays [n_sectors, [height, width]] for preprocessed SNR difference flux image (centered or not on target)
            - validpxs_imgs_<center_suffix>: list of NumPy arrays [n_sectors, [height, width]] for valid pixels mask (centered or not on target)
            - target_imgs_<center_suffix>: list of NumPy arrays [n_sectors, [height, width]] for target pixel location image (centered or not on target)
            - neighbors_imgs_<center_suffix> (optional; None): list of NumPy arrays [n_sectors, [height, width]] for neighbors' image (centered or not on target)
        - target_position
            - pixel_x_<center_suffix>: list of target column coordinates [n_sectors] (float) after centering or not on target
            - pixel_y_<center_suffix>: list of target row coordinates [n_sectors] (float) after centering or not on target
            - subpixel_x_<center_suffix>: list of discrete grid mapping value for target column subpixel coordinate (int) after centering or not on target
            - subpixel_y_<center_suffix>: list of discrete grid mapping value for target row subpixel coordinate (int) after centering or not on target
            - target_position_res_<center_suffix>: list of tuples [n_sectors, (col_res, row_res, valid_flag)] with target offset from center of image after centering or not on target
        - neighbors_feats: list of top-k neighbors features [n_sectors, (top_k_neighbors, n_neighbors_features)]
        - quality: list of DV quality correlation metrics [n_sectors]
        - image_numbers: list of sectors with sampled images
        
Sampling quarter/sector image data:
1 - Get number of valid quarters/sector_runs.
2 - If there are no valid images, set difference and oot images to NaN, px and subpx coordinates to zero, quality metric to NaN.
3 - Sample quarters/sector_runs from valid quarters/sector runs set.

Preprocessing quarter/sector image data:
1 - Create discrete mapping of target location to 3x3 grid.
2 - Fill out missing values by using nearest neighbors with same unit weight (missing values add zero and have no weight).
3 - Pixels for which the padding window was all missing values are set to zero for diff and snr images, and median value for oot images.
4 - Pad images by extending edges.
5 - [Optional] Center images on target star.
6 - [Optional] Resize images using nearest neighbor interpolation.
7 - Crop images.
8 - Create target pixel image.
9 - [Optional] Create neighbors image.
"""

# 3rd party
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import multiprocessing
import argparse
import yaml
from tqdm import tqdm
import copy

# local
from src_preprocessing.diff_img.preprocessing.utils_diff_img import (
    plot_diff_img_data, create_neighbors_img, map_target_subpixel_location_to_discrete_grid,
    fill_missing_values_nearest_neighbors, center_images_to_target_centroid,
    crop_images_to_size, crop_images_to_valid_size, set_data_example_to_placeholder_values,
    check_for_missing_values_in_preproc_diff_data, sample_image_data, initialize_data_example_with_missing_values,
    create_target_image, resize_images_by_resampling, pad_images_by_extending_edges, compute_neighbors_feature_vector, 
    create_target_offset_relative_to_image_center_feature)
from src_preprocessing.diff_img.preprocessing.placeholders_diff_img import placeholder_image, placeholder_target_position_center

CENTER_OPTIONS = ['target_not_centered', 'target_centered']
BACKGROUND_VAL = 0  # use to impute out-of-transit image when all pixels are missing 
N_FEATURES_NEIGHBORS = 5


def _debug_rel_geom(tag, tgt_col, tgt_row, neighbors_data_dict, neighbor_id_to_check):
    """ Log neighbor offsets relative to target and the angle.
    
    param str tag: tag to identify the log message
    param float tgt_col: target column pixel coordinate
    param float tgt_row: target row pixel coordinate
    param dict neighbors_data_dict: dictionary where each key is the neighbor ID that maps to a dictionary
        with the column 'col_px' and row 'row_px' coordinates of these objects in the CCD pixel frame of the target star
    param str neighbor_id_to_check: neighbor ID to check
    """
    
    print(f'[{tag}] target=({tgt_col:.3f},{tgt_row:.3f})')
    
    if neighbors_data_dict is None:
        print(f'[{tag}] neighbors: None')
        return
    
    for nb_id, nb in neighbors_data_dict.items():
        if nb_id == neighbor_id_to_check:
            dx = float(nb['col_px']) - float(tgt_col)
            dy = float(nb['row_px']) - float(tgt_row)
            ang = np.degrees(np.arctan2(dy, dx))
            print(f'[{tag}] nb={nb_id} dx={dx:.3f} dy={dy:.3f} ang={ang:.2f}°')


def impute_all_missing_pixels_img(size_h, size_w, size_f_h, size_f_w, neighbor_data=None, tmag_diff_range=[-12, 12], top_k=5):
    """Impute cases where all pixels are missing in the image.

    :param int size_h: image height
    :param int size_w: image width
    :param int size_f_h: resizing height factor
    :param int size_f_w: resizing width factor
    :param dict neighbor_data: neighbors coordinates and TMag, defaults to None
    :param list tmag_diff_range: allowed TMag difference range between target and neighbors, defaults to None
    :param int top_k: top-k neighbors to consider, defaults to 5
    :return tuple: imputed images and other preprocessing features
    """
    
    out_h, out_w = size_h * size_f_h, size_w * size_f_w
    
    diff_out   = placeholder_image(out_h, out_w, 0.0)
    oot_out    = placeholder_image(out_h, out_w, BACKGROUND_VAL)
    snr_out    = placeholder_image(out_h, out_w, 0.0)
    
    target_out = placeholder_image(out_h, out_w, 0.0)
    target_out[out_h // 2, out_w // 2] = 1.0  # optional center marker

    target_pos_placeholder = placeholder_target_position_center(out_h, out_w, 1)[0]
    target_pos = (target_pos_placeholder['pixel_x'], target_pos_placeholder['pixel_y'])
    target_pos_disc = (target_pos_placeholder['subpixel_x'], target_pos_placeholder['subpixel_y'])
    
    valid_pxs_img = placeholder_image(out_h, out_w, 0.0).as_type('int')
    
    if neighbor_data is not None:
        neighbors_out = placeholder_image(out_h, out_w, tmag_diff_range[0])
        neighbors_coords = []
        neighbors_feature_vectors = np.zeros((top_k, N_FEATURES_NEIGHBORS), dtype=float)
    else:
        neighbors_out = None
        neighbors_coords = None
        neighbors_feature_vectors = None

    return diff_out, oot_out, snr_out, valid_pxs_img, target_out, target_pos, target_pos_disc, neighbors_out, neighbors_coords, neighbors_feature_vectors
    

def preprocess_single_diff_img_data_for_example(diff_img, oot_img, snr_img, target_pos_col, target_pos_row, size_h,
                                                size_w, size_f_h, size_f_w, img_n, tce_uid, prefix, center_target=True,
                                                neighbor_data=None, target_mag=None, tmag_diff_range=(-10, 15),
                                                log=None, proc_id=-1, 
                                                resize_images=False):
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
        neighbor_data: dict, for each sector, contains a dictionary where each key is the TIC ID of
            neighboring objects that maps to a dictionary with the column 'col_px' and row 'row_px' coordinates of these
            objects in the CCD pixel frame of the target star along with the corresponding magnitude 'TMag' and distance
            to the target in arcseconds 'dst_arcsec'.
        target_mag: float, target magnitude
        tmag_diff_range: tuple, (min, max) difference in TMag between target and neighboring objects to be included
        log: logger
        proc_id: str, process id
        resize_images: bool, if True images are resized using nearest neighbor interpolation

    Returns:
        NumPy array for preprocessed difference image
        NumPy array for preprocessed out-of-transit image
        NumPy array for preprocessed snr image
        NumPy array for target location image
        tuple for target column and row pixels
        tuple int in {-1, 0, 1} for target subpixel column and row values
        list with column and row offset residuals for the target location and valid index (0/1 for failure/success)
        NumPy array for neighbors image
        list of tuples with neighbor coordinates in the neighbors image (only the neighbors used to create the image)        
    """
    
    if neighbor_data is not None:
        neighbor_data_process = copy.deepcopy(neighbor_data)
    else:
        neighbor_data_process = {}
    
    # create valid pixels image; starts with all valid pixels as set to one
    valid_pxs_img = np.ones(diff_img.shape, dtype='int')

    # map subpixel coordinates to discrete range {-1, 0, 1}; zero is target pixel
    target_col_disc, target_row_disc = map_target_subpixel_location_to_discrete_grid(target_pos_col, target_pos_row)

    half_height, half_width = size_h // 2, size_w // 2
    
    # compute target pixel offset from image center
    offset_res = create_target_offset_relative_to_image_center_feature(target_pos_col, target_pos_row, size_w, size_h)

    # set pixels with negative oot flux to invalid
    neg_oot_pxs = oot_img < 0
    valid_pxs_img[neg_oot_pxs] = 0

    all_nan = np.isnan(diff_img).all() or np.isnan(oot_img).all() or np.isnan(snr_img).all()
    if all_nan:
            if log:
                log.info(f'[{proc_id}] All pixels are NaN; using placeholders for {tce_uid} in {prefix} {img_n}.')

            return impute_all_missing_pixels_img(size_h, size_w, size_f_h, size_f_w, neighbor_data=neighbor_data_process, tmag_diff_range=tmag_diff_range, top_k=5)

    # _debug_rel_geom('EXTRACT', target_pos_col, target_pos_row, neighbor_data)

    # get min and max indices on both dimensions that have at least one non-missing pixel
    diff_img, oot_img, snr_img, valid_pxs_img, crop_min_col, crop_min_row = crop_images_to_valid_size(diff_img, oot_img, snr_img, valid_pxs_img)
    
    # _debug_rel_geom('AFTER_CROP_VALID', target_pos_col, target_pos_row, neighbor_data)

    # use local context to fill out missing values by using nearest neighbors with same weight
    fill_missing_filter = np.ones((3, 3))
    diff_img, filled_pxs_diff_img = fill_missing_values_nearest_neighbors(diff_img, fill_missing_filter)
    oot_img, filled_pxs_oot_img = fill_missing_values_nearest_neighbors(oot_img, fill_missing_filter)
    snr_img, filled_pxs_snr_img = fill_missing_values_nearest_neighbors(snr_img, fill_missing_filter)
    # set pixels to invalid if any of the images have those pixels missing
    idxs_filled_pxs = filled_pxs_diff_img & filled_pxs_oot_img & filled_pxs_snr_img
    valid_pxs_img[idxs_filled_pxs] = 0

    # deal with pixels for which the padding window was all missing values
    idxs_nan = np.isnan(diff_img) | np.isnan(oot_img) | np.isnan(snr_img)
    if idxs_nan.sum() != 0:
        if log:
            log.info(
                f'[{proc_id}] {idxs_nan.sum()} pixels were missing a value after nearest neighborhood padding'
                f' for example {tce_uid} in {prefix} {img_n}.')

        diff_img[idxs_nan] = 0  # set to zero; encodes expected no flux change
        snr_img[idxs_nan] = 0  # follows from diff image
        # set to background level using median as estimator; assumes most pixels are background and background is fairly uniform
        oot_img[idxs_nan] = np.nanmedian(oot_img)  

    # pad image by extending edges
    diff_img, oot_img, snr_img, valid_pxs_img, pad_col, pad_row = pad_images_by_extending_edges(diff_img, oot_img, snr_img, valid_pxs_img, half_height, half_width)
    # update target position in padded image
    target_pos_col = target_pos_col - crop_min_col + pad_col
    target_pos_row = target_pos_row - crop_min_row + pad_row
    if neighbor_data is not None:
        for neighbor_id, neighbor_id_data in neighbor_data_process.items():
            neighbor_data_process[neighbor_id]['col_px'] = neighbor_id_data['col_px'] - crop_min_col + pad_col
            neighbor_data_process[neighbor_id]['row_px'] = neighbor_id_data['row_px'] - crop_min_row + pad_row
    
    # _debug_rel_geom('AFTER_PAD', target_pos_col, target_pos_row, neighbor_data)

    if center_target:
        if np.isnan(target_pos_col):  # target location not available
            if log:
                log.info(
                    f'[{proc_id}] Target location from DV was not available ({target_pos_row}, '
                    f'{target_pos_col}). No centering performed for example {tce_uid} in {prefix} {img_n}.'
                    f' for example {tce_uid} in {prefix} {img_n}.')
        else:
            diff_img, oot_img, snr_img, valid_pxs_img, target_pos_col, target_pos_row, center_col_offset, center_row_offset, offset_res = \
            (center_images_to_target_centroid(diff_img, oot_img, snr_img, valid_pxs_img, target_pos_col, target_pos_row, size_h, size_w))
            
            # update neighbbors location after centering on target
            if neighbor_data is not None:
                for neighbor_id, neighbor_id_data in neighbor_data_process.items():
                    neighbor_data_process[neighbor_id]['col_px'] = neighbor_id_data['col_px'] - center_col_offset
                    neighbor_data_process[neighbor_id]['row_px'] = neighbor_id_data['row_px'] - center_row_offset

        # _debug_rel_geom('AFTER_CENTER', target_pos_col, target_pos_row, neighbor_data)
        
    # resize image using nearest neighbor interpolation to `size_f_h` * `size_f_w` times the target dimension
    if resize_images:
        diff_img, oot_img, snr_img, valid_pxs_img = resize_images_by_resampling(diff_img, oot_img, snr_img, valid_pxs_img, size_f_h, size_f_w)

        # update target position in resized image
        target_pos_col = (target_pos_col + 0.5) * size_f_w - 0.5 
        target_pos_row = (target_pos_row + 0.5) * size_f_h - 0.5
         
        if neighbor_data is not None:
            for neighbor_id, neighbor_id_data in neighbor_data_process.items():
                neighbor_data_process[neighbor_id]['col_px'] = (neighbor_id_data['col_px'] + 0.5) * size_f_w - 0.5 
                neighbor_data_process[neighbor_id]['row_px'] = (neighbor_id_data['row_px'] + 0.5) * size_f_h - 0.5  
    else:
        # force resizing factors to one
        size_f_h, size_f_w = 1, 1

        # _debug_rel_geom('AFTER_RESIZE', target_pos_col, target_pos_row, neighbor_data)

    # crop images to target dimension if they are larger
    diff_img, oot_img, snr_img, valid_pxs_img, crop_size_col_offset, crop_size_row_offset = \
        crop_images_to_size(diff_img,
                            oot_img,
                            snr_img,
                            valid_pxs_img,
                            size_h * size_f_h,
                            size_w * size_f_w)

    # update target pixel position after cropping image
    target_pos_col -= crop_size_col_offset
    target_pos_row -= crop_size_row_offset
    if neighbor_data is not None:
        for neighbor_id, neighbor_id_data in neighbor_data_process.items():
            neighbor_data_process[neighbor_id]['col_px'] = neighbor_id_data['col_px'] - crop_size_col_offset
            neighbor_data_process[neighbor_id]['row_px'] = neighbor_id_data['row_px'] - crop_size_row_offset
    
    # _debug_rel_geom('AFTER_CROP_FINAL', target_pos_col, target_pos_row, neighbor_data)

    # check if target pixel location is inside the image
    if (target_pos_col < 0 or target_pos_row < 0 or target_pos_col >= diff_img.shape[1] or
            target_pos_row >= diff_img.shape[0]):
        if log:
            log.info(f'[{proc_id}] Target pixel is outside of image after cropping for example '
                     f'{tce_uid} in {prefix} {img_n}.')

    # create target image
    target_img = create_target_image(size_h * size_f_h, size_w * size_f_w, target_pos_col, target_pos_row)

    if neighbor_data is not None:
        
        neighbors_img = create_neighbors_img(neighbor_data_process, diff_img.shape, target_mag, tmag_diff_range)
        
        neighbors_coords = []
        for _, nb in neighbor_data_process.items():
            col = float(nb['col_px'])
            row = float(nb['row_px'])
            if 0 <= col < diff_img.shape[1] and 0 <= row < diff_img.shape[0]:
                neighbors_coords.append((col, row))

    else:
        neighbors_img = None
        neighbors_coords = None
        # neighbors_feature_vectors = None

    return diff_img, oot_img, snr_img, valid_pxs_img, target_img, (target_pos_col, target_pos_row), (target_col_disc, target_row_disc), offset_res, neighbors_img, neighbors_coords


def create_aux_table(tces_info_dict, tbl_fp):
    """Create auxiliary table.

    :param dict tces_info_dict: Contains information about the preprocessing run of difference image data
    :param Path tbl_fp: Filepath used to save table
    """
    
    tces_info_df = pd.DataFrame(tces_info_dict)
    
    tces_info_df.attrs['Description'] = 'This table contains auxliary information regarding the preprocessing of extracted difference image.'
    tces_info_df.attrs['Preprocessing Directory'] = str(tbl_fp.parent)
    tces_info_df.attrs['Table Filepath'] = str(tbl_fp)
    tces_info_df.attrs['Creation Date'] = pd.Timestamp.now().isoformat()
    tces_info_df.attrs['Created By'] = 'src_preprocessing/diff_img/preprocessing/preprocess_diff_img.py'

    with open(tbl_fp, "w") as f:
        for key, value in tces_info_df.attrs.items():
            f.write(f"# {key}: {value}\n")
        tces_info_df.to_csv(f, index=False)
    
    
def check_valid_img(img_data, centroid, qual_metric):
    """Checks if image from sector/quarter is valid. Checks for target location, quality metric, and NaN values in difference image data.

    :param NumPy array img_data: difference image data for sector/quarter
    :param dict centroid: target location in pixel frame
    :param dict qual_metric: quality metric
    :return bool: returns True if it passes all checks; plus dictionary of success/failure
    """
    
    is_valid = False
            
    # quarter/sector is valid and uncertainty if
    # 1) target position uncertainty is not -1, which means target location is not available
    col_ok = centroid['col']['uncertainty'] != -1
    row_ok = centroid['row']['uncertainty'] != -1
    # 2) quality metric table shows 'TRUE' in the 'valid' field
    quality_ok = bool(qual_metric['valid'])
    # 3) diff, oot, and diff unc images are not all NaNs
    diff_ok = ~np.isnan(img_data[:, :, 2, 0]).all()
    oot_ok  = ~np.isnan(img_data[:, :, 1, 0]).all()
    snr_ok  = ~np.isnan(img_data[:, :, 2, 1]).all()

    if (col_ok and row_ok) and quality_ok and diff_ok and oot_ok and snr_ok:
        is_valid = True
    
    error_log = {
        'target_col_ok': col_ok,
        'target_row_ok': row_ok,    
        'quality_ok': quality_ok,
        'diff_ok': diff_ok,
        'oot_ok': oot_ok,
        'snr_ok': snr_ok,
    }
    
    return is_valid, error_log

                
def preprocess_diff_img_tces(diff_img_data_fp, number_of_imgs_to_sample, 
                             upscale_f, final_size, resize_images,
                             mission_name,
                             save_dir, 
                             tmag_diff_range=(-10, 15), max_neighbor_distance=8, top_k=5, gaussian_att_sigma=1, 
                             log=None, check_exist=False,
                             plot_prob=0, plot_dir=None):
    """ Preprocessing pipeline for difference image data for a set of TCEs.

    Args:
        diff_img_data_fp: Path, to NumPy file with a dictionary. Each item is the difference image data for a given TCE.
            The TCE is identified by the string key. The value is a dictionary that contains six items:
            - 'target_ref_centroid' is a list of dictionaries that contain the value and uncertainty for the reference
            coordinates of the target star in the pixel domain in each observed sector;
            - 'image_data' is a list of NumPy array (n_rows, n_cols, n_imgs, 2) that contains the in-transit,
            out-of-transit, difference, and "SNR" flux images in this order (pixel values and uncertainties are addressed
            by the last dimension of the array, in this order) for each observe sector;
            - 'image_number' is a list that contains the integer sector number of the corresponding sequence of
            difference image data extracted for the TCE.
            - 'mag' is the target's magnitude.
            - 'neighbor_data' is a list that, for each sector, contains a dictionary where each key is the TIC ID of
            neighboring objects that maps to a dictionary with the column 'col_px' and row 'row_px' coordinates of these
            objects in the CCD pixel frame of the target star along with the corresponding magnitude 'TMag' and distance
            to the target in arcseconds 'dst_arcsec'.
        number_of_imgs_to_sample: int, number of quarters/sectors to sample
        resize_images: bool, whether to resize images using upscale factors `upscale_f`
        upscale_f: dict, resize factor for 'x' and 'y' dimensions. Final size of images is
            (final_size['x'] * upscale_f['x'], final_size['y'] * upscale_f['y'])
        final_size: dict, image size before resizing (final_size['x'], final_size['y'])
        mission_name: str, mission from where the difference image data is from. Either `kepler` or `tess`
        save_dir: Path, destination  directory for preprocessed data
        tmag_diff_range: tuple, (min, max) difference in TMag between target and neighboring objects to be included
        log: logger
        check_exist: bool, if True if checks whether there is already a NumPy file with preprocessed difference image data for the 
            sector run. Defaults to False.
        plot_prob: float, probability to plot preprocessing results
        plot_dir: Path, plot directory

    Returns:

    """

    save_dir.mkdir(exist_ok=True)
    
    save_fp = save_dir / "diffimg_preprocess.npy"

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
    
    if check_exist:  # check if file already exists
        if save_fp.exists():
            log.info(f'Found file {str(save_fp)}. Skipping data in {str(diff_img_data_fp)}.')
            return

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

    # log.info(f'[{proc_id}] Checking examples in saturated targets...')
    # log.info(f'Found {len(tces_in_sat_tbl)} examples in saturated targets.')

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
    # tces_info_tbl['oot_negative_values'] = ''

    # initialize data dictionaries
    preprocessing_dict = {}
    for tce_i, tce_uid in enumerate(diff_img_data_dict):  # iterate over TCEs
        
        # if tce_uid != '276743869-1-S75':
        #     continue
        
        neighbors_coords_tce = {}
        for centering_option in CENTER_OPTIONS:
            suffix_str = '_tc' if centering_option == 'target_centered' else ''
            neighbors_coords_tce[f'neighbors_coords{suffix_str}'] = [None] * number_of_imgs_to_sample

        # initialize dictionary for the preprocessing results
        preprocessing_dict[tce_uid] = initialize_data_example_with_missing_values(final_size['x'] * upscale_f['x'],
                                                                                  final_size['y'] * upscale_f['y'],
                                                                                  number_of_imgs_to_sample,
                                                                                  top_k_neighbors=top_k,
                                                                                  )

        n_max_imgs_avail = len(diff_img_data_dict[tce_uid]['image_number'])

        if tce_i % 500 == 0:
            log.info(f'[{diff_img_data_fp.stem}] Preprocessed {tce_i + 1} example(s) out of {len(diff_img_data_dict)}.')

        # # checking if TCE is in a saturated target; do not preprocess data for these cases
        #     continue

        # get quarters/sectors with data (i.e., quality metrics is a defined value)
        available_imgs_idxs = [img_idx for img_idx in range(n_max_imgs_avail)
                               if ~np.isnan(diff_img_data_dict[tce_uid]['quality_metric'][img_idx]['value'])]

        tces_info_dict[f'num_available_{prefix}s'][tce_i] = n_max_imgs_avail

        # get quarters/sectors with valid data
        valid_images_idxs = []
        test_errors_dict = {}
        for img_idx in available_imgs_idxs:
    
            centroid = diff_img_data_dict[tce_uid]['target_ref_centroid'][img_idx]
            img_data = diff_img_data_dict[tce_uid]['image_data'][img_idx]
            qual_metric = diff_img_data_dict[tce_uid]['quality_metric'][img_idx]
            
            is_valid, test_errors = check_valid_img(img_data, centroid, qual_metric)

            if is_valid:
                valid_images_idxs.append(img_idx)
            else:
                test_errors_dict[img_idx] = test_errors

        n_valid_imgs = len(valid_images_idxs)

        tces_info_dict[f'num_valid_{prefix}s'][tce_i] = n_valid_imgs

        if n_valid_imgs == 0:  # if no valid quarters/sectors

            log.info(f'[{diff_img_data_fp.stem}] No valid images for {tce_uid}: {test_errors_dict}\nSetting data to placeholder value.')

            # update data using placeholder values
            missing_data_placeholder = set_data_example_to_placeholder_values(
                final_size['x'] * upscale_f['x'],
                final_size['y'] * upscale_f['y'],
                number_of_imgs_to_sample,
                'neighbor_data' in diff_img_data_dict[tce_uid],
                top_k_neighbors=top_k,
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
            
            img_num = diff_img_data_dict[tce_uid]['image_number'][sampled_img_idx]  # image number (i.e., sector/quarter)
            qmetric = diff_img_data_dict[tce_uid]['quality_metric'][sampled_img_idx]['value']  # quality metric
            target_mag = diff_img_data_dict[tce_uid]['mag']  # target TMag
            
            # get images and target position in the pixel frame
            diff_img = diff_img_data_dict[tce_uid]['image_data'][sampled_img_idx][:, :, 2, 0].copy()
            oot_img = diff_img_data_dict[tce_uid]['image_data'][sampled_img_idx][:, :, 1, 0].copy()
            snr_img = diff_img / diff_img_data_dict[tce_uid]['image_data'][sampled_img_idx][:, :, 2, 1]
            
            target_pos_col = diff_img_data_dict[tce_uid]['target_ref_centroid'][sampled_img_idx]['col']['value']
            target_pos_row = diff_img_data_dict[tce_uid]['target_ref_centroid'][sampled_img_idx]['row']['value']

            if 'neighbor_data' in diff_img_data_dict[tce_uid]:
                if len(diff_img_data_dict[tce_uid]['neighbor_data']) > 0 and img_num in diff_img_data_dict[tce_uid]['neighbor_data']:
                    neighbor_data_tce_sector = diff_img_data_dict[tce_uid]['neighbor_data'][img_num]
                    neighbors_data = copy.deepcopy(neighbor_data_tce_sector)
                else:  # no neighbor data for this TCE in this image
                    neighbors_data = {}
            else:  # the extracted difference image does not contain neighbor data at all
                neighbors_data = None
                # raise ValueError(f'Neighbor data not found for TCE {tce_uid}.')

            for option in CENTER_OPTIONS:
                
                (diff_img_preproc, oot_img_preproc, snr_img_preproc, valid_pxs_img, target_img, target_pos, 
                target_pos_disc, offset_res, neighbors_img, neighbors_coords) = (
                    preprocess_single_diff_img_data_for_example(
                        diff_img.copy(),
                        oot_img.copy(),
                        snr_img.copy(),
                        target_pos_col,
                        target_pos_row,
                        final_size['x'],
                        final_size['y'],
                        upscale_f['x'],
                        upscale_f['y'],
                        img_num,
                        tce_uid,
                        prefix,
                        center_target=option == 'target_centered',
                        neighbor_data=neighbors_data,
                        target_mag=target_mag,
                        tmag_diff_range=tmag_diff_range,
                        log=log,
                        proc_id=diff_img_data_fp.stem,
                        resize_images=resize_images,
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
                    preprocessing_dict[tce_uid]['images'][f'validpxs_imgs{suffix_str}'][img_idx] = valid_pxs_img

                    # add target pixel coordinates
                    preprocessing_dict[tce_uid]['target_position'][f'pixel_x{suffix_str}'][img_idx] = target_pos[0]
                    preprocessing_dict[tce_uid]['target_position'][f'pixel_y{suffix_str}'][img_idx] = target_pos[1]

                    # add discrete coordinates of target position relative to target pixel when expanding it to a nxn grid
                    preprocessing_dict[tce_uid]['target_position'][f'subpixel_x{suffix_str}'][img_idx] = (target_pos_disc)[0]
                    preprocessing_dict[tce_uid]['target_position'][f'subpixel_y{suffix_str}'][img_idx] = (target_pos_disc)[1]
                    
                    # add target offset relative to image center
                    preprocessing_dict[tce_uid]['target_position'][f'target_positon_res{suffix_str}'][img_idx] = offset_res

                    # add neighbors image
                    if 'neighbor_data' in diff_img_data_dict[tce_uid]:
                        preprocessing_dict[tce_uid]['images'][f'neighbors_imgs{suffix_str}'][img_idx] = neighbors_img
                        # coordinates needed to display neighbors in auxiliary plots
                        neighbors_coords_tce[f'neighbors_coords{suffix_str}'][img_idx] = neighbors_coords
                        
                                                                
            for img_idx in img_idxs:  # data that does not require centering the target star

                # add quality metric
                preprocessing_dict[tce_uid]['quality'][img_idx] = qmetric

                # add current quarter/sector number to dictionary
                preprocessing_dict[tce_uid]['images_numbers'][img_idx] = img_num
                
                if 'neighbor_data' in diff_img_data_dict[tce_uid]:
                    neighbors_feature_vectors = compute_neighbors_feature_vector(
                        neighbor_data_tce_sector=neighbors_data if neighbors_data is not None else {}, 
                        target_pos=(target_pos_col, target_pos_row), 
                        neighbor_delta_tmag_bound=tmag_diff_range,
                        max_neighbor_distance=max_neighbor_distance,
                        top_k=top_k,
                        gaussian_att_sigma=gaussian_att_sigma,
                        )
                    
                    preprocessing_dict[tce_uid][f'neighbors_feats'][img_idx] = neighbors_feature_vectors

            if np.random.uniform() <= plot_prob:  # plot final images
                for option in CENTER_OPTIONS:

                    suffix_str = '_tc' if option == 'target_centered' else ''

                    diff_imgs_arr_aux = np.concatenate([
                        np.expand_dims(preprocessing_dict[tce_uid]['images'][f'oot_imgs{suffix_str}'][img_idxs[0]], axis=2),
                        np.expand_dims(preprocessing_dict[tce_uid]['images'][f'diff_imgs{suffix_str}'][img_idxs[0]], axis=2),
                        np.expand_dims(preprocessing_dict[tce_uid]['images'][f'snr_imgs{suffix_str}'][img_idxs[0]], axis=2)],
                        axis=2)

                    plot_info_str = f"Difference Image\nSector {img_num}\nQMetric={preprocessing_dict[tce_uid]['quality'][img_idxs[0]]:.2f} | Target TMag={target_mag:.2f}"
                    # f"Target Offset Center={','.join(preprocessing_dict[tce_uid]['target_position'][f'target_positon_res{suffix_str}'][img_idxs[0]])}"
                    
                    if plot_dir:
                        plot_fp = plot_dir / f'{tce_uid}_diff_img_{img_num}{suffix_str}.png'
                    else:
                        plot_fp = save_dir / 'plot_examples' / f'{tce_uid}_diff_img_{img_num}{suffix_str}.png'

                    plot_diff_img_data(
                        diff_imgs_arr_aux,
                        (preprocessing_dict[tce_uid]['target_position'][f'pixel_x{suffix_str}'][img_idxs[0]],
                         preprocessing_dict[tce_uid]['target_position'][f'pixel_y{suffix_str}'][img_idxs[0]]),
                        plot_fp,
                        valid_pxs_img=preprocessing_dict[tce_uid]['images'][f'validpxs_imgs{suffix_str}'][img_idxs[0]],
                        neighbors_img=preprocessing_dict[tce_uid]['images'][f'neighbors_imgs{suffix_str}'][img_idxs[0]]
                            if 'neighbor_data' in diff_img_data_dict[tce_uid] else None,                        
                        neighbors_coords=neighbors_coords_tce[f'neighbors_coords{suffix_str}'][img_idxs[0]]
                            if 'neighbor_data' in diff_img_data_dict[tce_uid] else None,
                        logscale=True,
                        title_str=plot_info_str,
                    )

        # last check for missing values
        missing_value_found = check_for_missing_values_in_preproc_diff_data(preprocessing_dict[tce_uid], has_neighbor_data='neighbor_data' in diff_img_data_dict[tce_uid])

        if missing_value_found:
            log.info(f'[{diff_img_data_fp.stem}] At least one data array contained missing values for {tce_uid}. '
                     f'Setting data to placeholder value.')

            tces_info_dict[f'sampled_{prefix}s_missingvalues'][tce_i] = 'yes'

            # update data using placeholder values
            missing_data_placeholder = set_data_example_to_placeholder_values(final_size['x'] * upscale_f['x'],
                                                                              final_size['y'] * upscale_f['y'],
                                                                              number_of_imgs_to_sample,
                                                                              top_k_neighbors=top_k,
                                                                              )
            preprocessing_dict[tce_uid].update(missing_data_placeholder)

    create_aux_table(tces_info_dict, save_dir / 'info_tces.csv')

    log.info(f'Saving preprocessed data to {save_dir / "diffimg_preprocess.npy"}...')
    np.save(save_fp, preprocessing_dict)

    log.info(f'[{diff_img_data_fp.stem}] Finished preprocessing difference image data for {len(preprocessing_dict)}/{len(diff_img_data_dict)} '
             f'TCEs.')


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
        config = yaml.unsafe_load(file)

    if save_dir is not None:
        config['dest_root_dir'] = save_dir
    if diff_img_dir is not None:
        config['diff_img_data_dir'] = diff_img_dir

    plot_dir = config.get('plot_dir')        

    diff_img_data_dir = Path(config['diff_img_data_dir'])

    dest_root_dir = Path(config['dest_root_dir'])

    # list of file paths to DV NumPy files for the sector runs to be preprocessed
    diff_img_data_fps = list(diff_img_data_dir.glob('*.npy'))
    # diff_img_data_fps = list(diff_img_data_dir.glob('*tess_diffimg_s0075.npy'))

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

    jobs = [(diff_img_data_fp, config['num_sampled_imgs'], config['upscale_f'], config['final_size'], config['resize_images'],
             config['mission'], dest_root_dir / diff_img_data_fp.stem, tuple(config['tmag_diff_range']), 
             config['max_neighbor_distance'], config['top_k'], config['gaussian_att_sigma'],
             None, config['check_exist'], config['plot_prob'], plot_dir)
            for diff_img_data_fp in diff_img_data_fps]

    if config['n_processes'] > 1:
        # parallel
        with multiprocessing.Pool(processes=config['n_processes']) as pool:
            
            with tqdm(total=len(jobs), desc='Sector Run Job', unit='job') as pbar:
                def _update_progress(_):
                    # increment 1 per finished job
                    pbar.update(1)

                async_results = [
                    pool.apply_async(preprocess_diff_img_tces, job, callback=_update_progress)
                    for job in jobs
                ]
                # wait for all jobs to finish (propagate exceptions)
                for res_i, res in enumerate(async_results):
                    res.get()
                    logger.info(f'Finished job {res_i + 1} out of {config["n_jobs"]}.')

    else:
        # sequential
        for job in tqdm(jobs, desc='Sector Run Job', total=len(jobs), unit='job'):
            preprocess_diff_img_tces(*job)

    logger.info('Finished preprocessing difference image data from NumPy files with extracted data from DV xml files.')


if __name__ == '__main__':

    # used in job arrays
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_fp', type=str, help='Configuration file with processing parameters.',
                        default='/Users/msaragoc/Projects/exoplanet_transit_classification/exoplanet_dl/src_preprocessing/diff_img/preprocessing/config_preprocessing.yaml')
    args = parser.parse_args()

    preprocess_diff_img_tces_main(args.config_fp)