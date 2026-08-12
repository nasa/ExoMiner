""" Script with placeholder functions for difference image preprocessing.
"""

# 3rd party
import numpy as np


def placeholder_image(size_h: int, size_w: int, fill_value: float = 0.0) -> np.ndarray:
    """Return a single image placeholder of shape [size_h, size_w]."""
    return np.full((size_h, size_w), fill_value, dtype='float')

def placeholder_image_list(size_h: int, size_w: int, n: int, fill_value: float = 0.0) -> list:
    """Return a list of n image placeholders."""
    return [placeholder_image(size_h, size_w, fill_value) for _ in range(n)]

def placeholder_target_position_center(size_h: int, size_w: int, n: int) -> dict:
    """Center target position: pixel_x=col=center_w, pixel_y=row=center_h; subpixel = 0."""
    
    center_row = size_h // 2
    center_col = size_w // 2
    
    return {
        'pixel_x':    [center_col] * n,   # column
        'pixel_y':    [center_row] * n,   # row
        
        'subpixel_x': [0] * n,
        'subpixel_y': [0] * n,
        
        'target_positon_res': [[0, 0, 0, 0]] * n,
    }

def placeholder_target_position_nan(n: int) -> dict:
    """Unknown target position: NaN everywhere; subpixel=0."""
    
    return {
        'pixel_x':    [np.nan] * n,
        'pixel_y':    [np.nan] * n,
        
        'subpixel_x': [0] * n,
        'subpixel_y': [0] * n,
        
        'target_positon_res': [[np.nan, np.nan, 0, 0]] * n,
    }

def placeholder_quality(n: int, fill_value: float = 0.0) -> list:
    return [fill_value] * n

def placeholder_images_numbers(n: int) -> list:
    return [np.nan] * n

def placeholder_neighbors_features(n: int, top_k_neighbors: int =5, n_feats_neighbors: int = 5) -> list:
    
    return [np.zeros((top_k_neighbors, n_feats_neighbors), dtype='float') for _ in range(n)]

def set_data_example_to_placeholder_values(
    size_h: int, size_w: int, number_of_imgs_to_sample: int,
    placeholder_neighbor_val: float = -10, top_k_neighbors: int = 5, n_feats_neighbors: int = 5
) -> dict:
    """Sets data for an example with consistent placeholder values.
    
    - All images are set to zero arrays, except target images which have a "1" at the center pixel, and neighbor images if requested.
    The neighbor images are set to a constant value (default -10) to indicate 'no neighbor' data.
    - Target positions are centered in the images.
    - Quality metric values are set to 0.0.
    - Image numbers are set to NaN.
    
    param size_h: int, height images
    param size_w: int, width images
    param number_of_imgs_to_sample: int, number of sampled images
    param placeholder_neighbor_val: float, placeholder value for neighbors images. Default is -10
    param top_k_neighbors: int, top-k neighbors encoded in the neighbors features
    param n_feats_neighbors: int, number of features representing each neighbor
    """

    center_row = size_h // 2
    center_col = size_w // 2

    data_placeholder = {
        'images': {
            'diff_imgs':       placeholder_image_list(size_h, size_w, number_of_imgs_to_sample, 0.0),
            'oot_imgs':        placeholder_image_list(size_h, size_w, number_of_imgs_to_sample, 0.0),
            'snr_imgs':        placeholder_image_list(size_h, size_w, number_of_imgs_to_sample, 0.0),
            
            'target_imgs':     placeholder_image_list(size_h, size_w, number_of_imgs_to_sample, 0.0),
            'validpxs_imgs':  placeholder_image_list(size_h, size_w, number_of_imgs_to_sample, 0.0),
            
            'diff_imgs_tc':    placeholder_image_list(size_h, size_w, number_of_imgs_to_sample, 0.0),
            'oot_imgs_tc':     placeholder_image_list(size_h, size_w, number_of_imgs_to_sample, 0.0),
            'snr_imgs_tc':     placeholder_image_list(size_h, size_w, number_of_imgs_to_sample, 0.0),
            
            'target_imgs_tc':  placeholder_image_list(size_h, size_w, number_of_imgs_to_sample, 0.0),
            
            'validpxs_imgs_tc':  placeholder_image_list(size_h, size_w, number_of_imgs_to_sample, 0.0),
            
            # add neighbors images placeholders
            'neighbors_imgs' : placeholder_image_list(size_h, size_w, number_of_imgs_to_sample, placeholder_neighbor_val),
            'neighbors_imgs_tc' : placeholder_image_list(size_h, size_w, number_of_imgs_to_sample, placeholder_neighbor_val),
            
        },
        
        'target_position': placeholder_target_position_center(size_h, size_w, number_of_imgs_to_sample),
        
        'quality':         placeholder_quality(number_of_imgs_to_sample, 0.0),
        
        'images_numbers':  placeholder_images_numbers(number_of_imgs_to_sample),
        
        'neighbors_feats': placeholder_neighbors_features(number_of_imgs_to_sample, top_k_neighbors, n_feats_neighbors),
    }
    
    target_pos_tc = {f'{k}_tc': v for k, v in placeholder_target_position_center(size_h, size_w, number_of_imgs_to_sample).items()}
    data_placeholder['target_position'].update(target_pos_tc)

    # Put a "1" at the centered target pixel for target images
    for i in range(number_of_imgs_to_sample):
        data_placeholder['images']['target_imgs'][i][center_row, center_col] = 1.0
        data_placeholder['images']['target_imgs_tc'][i][center_row, center_col] = 1.0

    return data_placeholder
