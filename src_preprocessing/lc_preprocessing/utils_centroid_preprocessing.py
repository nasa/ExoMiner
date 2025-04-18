"""
Auxiliary functions used preprocess the centroid time-series
"""

# 3rd party
import os
import numpy as np

CCD_SIZE_ROW = 1044  # number of physical pixels
CCD_SIZE_COL = 1100  # number of physical pixels
COL_OFFSET = 12
ROW_HALFWIDTH_GAP = 40


def kepler_transform_pxcoordinates_mod13(all_centroids, add_info):
    """ Transform coordinates when target is on module 13 (central module in the Kepler's CCD array).

    :param all_centroids: dict of list of numpy arrays, x and y centroid shift time series
    :param add_info: dict, 'quarter' has as value a list of the quarters and 'module' has as value a list of the
    respective modules
    :return: transformed x and y centroid shift time series
    """

    if add_info['quarter'][0] == 0:
        add_info['quarter'][0] = 1

    q0 = add_info['quarter'][0]

    for i in range(len(all_centroids['x'])):
        delta_q = (np.abs(add_info['quarter'][i] - q0)) % 4

        if delta_q == 1 or delta_q == 3:  # 90 or 270 degrees

            col_prev = all_centroids['x'][i]
            row_prev = all_centroids['y'][i]
            # col
            all_centroids['x'][i] = (COL_OFFSET + CCD_SIZE_COL - (CCD_SIZE_ROW - row_prev + ROW_HALFWIDTH_GAP))
            # row
            all_centroids['y'][i] = ROW_HALFWIDTH_GAP + CCD_SIZE_ROW - (CCD_SIZE_COL - (col_prev - COL_OFFSET))

    return all_centroids


def get_denoised_centroids_kepler(keplerid, denoised_centroids_dir):
    """ Get denoised centroid time-series for Kepler data.

    :param keplerid: int, Kepler ID of the target star
    :param denoised_centroids_dir: str, denoised centroids directory
    :return:
        denoised_centroids: dict, key-value pair is a coordinate ('x', 'y') that maps to a list of numpy arrays, which
        are the denoised centroid time-series for each quarter for the given target star
        idxs_nan_centroids: list, indexes of NaN values in the original centroid time-series; each subarray (numpy
        array) is for a given quarter
    """

    # initialize variables
    denoised_centroids = {'x': [], 'y': []}
    idxs_nan_centroids = []

    # get filepaths for the kepids dictionaries that map from Kepler ID to quarters in a given channel
    kepids_ch_fp = [os.path.join(denoised_centroids_dir, filename) for filename in os.listdir(denoised_centroids_dir)
                    if 'kepids_ch' in filename]

    # initialize quarters' list
    quarters = []

    # iterate through the channel kepids
    for filepath in kepids_ch_fp:

        # load kepids for a channel
        kepids_ch = np.load(filepath, allow_pickle=True).item()

        # check if Kepler ID is in this channel
        if keplerid in kepids_ch:

            # get the channel
            channel = filepath.split('_')[-1]

            # load the denoised centroids for that channel
            denoised_centroids_ch = np.load(os.path.join(denoised_centroids_dir,
                                                         'denoisedcentroidtimeseries_{}.npy'.format(channel)),
                                            allow_pickle=True).item()
            # load NaN indexes for that channel
            idxs_nan_ch = np.load(os.path.join(denoised_centroids_dir, 'idxsnotnan_{}.npy'.format(channel)),
                                  allow_pickle=True).item()

            # iterate through the quarters in this channel for these Kepler ID
            for quarter in denoised_centroids_ch[keplerid]:
                quarters.append(quarter)
                idxs_nan_centroids.append(idxs_nan_ch[quarter])
                denoised_centroids['x'].append(denoised_centroids_ch[keplerid][quarter]['x'])
                denoised_centroids['y'].append(denoised_centroids_ch[keplerid][quarter]['y'])

    # sort arrays by quarters
    quarters_idxs = np.argsort(quarters)
    idxs_nan_centroids = [idxs_nan_centroids[idx] for idx in quarters_idxs]
    denoised_centroids['x'] = [denoised_centroids['x'][idx] for idx in quarters_idxs]
    denoised_centroids['y'] = [denoised_centroids['y'][idx] for idx in quarters_idxs]

    return denoised_centroids, idxs_nan_centroids


def correct_centroid_using_transit_depth(centroid_x, centroid_y, transit_depth, avg_centroid_oot):
    """ Compute transit source location from centroid offset.

    Args:
        centroid_x: NumPy array, centroid timeseries 'x' (right ascension)
        centroid_y: NumPy array, centroid timeseries 'y' (declination)
        transit_depth: transit depth, in ppm
        avg_centroid_oot: dict, average out-of-transit centroid; keys 'x' and 'y'

    Returns: NumPy array, corrected centroid timeseries

    """

    transitdepth_term = (1e6 - transit_depth) / transit_depth

    corrected_centroid = {
        'x': avg_centroid_oot['x'] - transitdepth_term * (centroid_x - avg_centroid_oot['x']) /
             np.cos(centroid_y * np.pi / 180),
        'y': avg_centroid_oot['y'] - transitdepth_term * (centroid_y - avg_centroid_oot['y'])
    }

    return corrected_centroid


def compute_centroid_distance(centroid_dict, target_position, delta_dec=None):
    """ Compute distance between transit source and target.

    Args:
        centroid_dict: dict, centroid timeseries 'x' and 'y'
        target_position: list, target position 'x' and 'y'
        delta_dec: float, declination correction

    Returns: NumPy array, distance between the transit source and target

    """

    if delta_dec is None:  # in pixel coordinates
        all_centroid_dist = np.sqrt(np.square(centroid_dict['x'] - target_position[0])
                                    + np.square(centroid_dict['y'] - target_position[1]))
    else:
        all_centroid_dist = np.sqrt(np.square((centroid_dict['x'] - target_position[0]) * delta_dec)
                                    + np.square(centroid_dict['y'] - target_position[1]))

    return all_centroid_dist
