"""
Auxiliary functions used preprocess the centroid time-series
"""

# 3rd party
import os
import numpy as np


def kepler_transform_pxcoordinates_mod13(all_centroids, add_info):
    """ Transform coordinates when target is on module 13 (central module in the Kepler's CCD array).

    :param all_centroids: dict of list of numpy arrays, x and y centroid shift time series
    :param add_info: dict, 'quarter' has as value a list of the quarters and 'module' has as value a list of the
    respective modules
    :return: transformed x and y centroid shift time series
    """

    CCD_SIZE_ROW = 1044  # number of physical pixels
    CCD_SIZE_COL = 1100  # number of physical pixels

    COL_OFFSET = 12

    ROW_HALFWIDTH_GAP = 40

    if add_info['quarter'][0] == 0:
        add_info['quarter'][0] = 1

    q0 = add_info['quarter'][0]

    for i in range(len(all_centroids['x'])):
        delta_q = (np.abs(add_info['quarter'][i] - q0)) % 4

        if delta_q == 1 or delta_q == 3:  # 90 or 270 degrees

            col_prev = all_centroids['x'][i]
            row_prev = all_centroids['y'][i]
            # col

            all_centroids['x'][i] = COL_OFFSET + CCD_SIZE_COL - \
                                    (CCD_SIZE_ROW - row_prev + ROW_HALFWIDTH_GAP)
            # row
            all_centroids['y'][i] = ROW_HALFWIDTH_GAP + CCD_SIZE_ROW - \
                                    (CCD_SIZE_COL - (col_prev - COL_OFFSET))

    return all_centroids


def patch_centroid_curve(all_centroids):
    """Connects the separated quarters of a centroid shift time series.

    :param all_centroids: dict of list of numpy arrays, x and y centroid shift time series
    :return: corrected x and y centroid shift time series
    """

    # check if all_centroids is only on numpy array instead of a list of numpy arrays
    # just one quarter
    if isinstance(all_centroids, np.ndarray) and all_centroids.ndim == 1:
        return all_centroids

    # patch separate quarters
    end = len(all_centroids['x'])
    i = 1  # start in the second quarter
    while i < end:

        # value of the first index that is not nan in the current quarter
        init_x = all_centroids['x'][i][np.where(~np.isnan(all_centroids['x'][i]))[0][0]]
        # value of the last index that is not nan in the previous quarter
        end_x = all_centroids['x'][i-1][np.where(~np.isnan(all_centroids['x'][i-1]))[0][-1]]

        # med_prev_x = all_centroids['x'][i-1][np.nanmedian(all_centroids['x'][i-1])]

        # same for the y coordinate
        init_y = all_centroids['y'][i][np.where(~np.isnan(all_centroids['y'][i]))[0][0]]
        end_y = all_centroids['y'][i-1][np.where(~np.isnan(all_centroids['y'][i-1]))[0][-1]]

        # med_prev_y = all_centroids['y'][i-1][np.nanmedian(all_centroids['y'][i-1])]

        # subtract the difference between consecutive quarters
        # all_centroids['x'][i] -= (init_x - end_x)
        # all_centroids['y'][i] -= (init_y - end_y)
        all_centroids['x'][i] = all_centroids['x'][i] + end_x - init_x
        all_centroids['y'][i] = all_centroids['y'][i] + end_y - init_y

        i += 1

    return all_centroids


def local_normalization_centroid(all_centroids, win_len=2000):
    """ Compute the local median in windows of total_length/win_len and divide each window by the respective median.

    :param all_centroids: list of numpy arrays, centroid time series
    :param win_len: int, window size
    :return:
        all_centroids: list of numpy arrays, locally normalized centroid time series
    """

    for j in range(len(all_centroids['x'])):

        limit = len(all_centroids['x'][j])

        # bin_size = int(np.ceil(len(all_centroids['x'][j]) / win_len))
        bin_size = min(win_len, limit)

        i = 0
        while i < limit:
            loc_median_x = np.nanmedian(all_centroids['x'][j][i:min(i + bin_size, limit)])
            loc_median_y = np.nanmedian(all_centroids['y'][j][i:min(i + bin_size, limit)])
            all_centroids['x'][j][i:i + bin_size] /= loc_median_x
            all_centroids['y'][j][i:i + bin_size] /= loc_median_y
            i += bin_size

    return all_centroids


def synchronize_centroids_with_flux(all_time, centroid_time, all_centroids, thres=0.005):
    """
    Synchronizes centroid cadences with flux cadences by comparing flux time vector with centroid time vector.
    Operations:
        - Fills centroid cadences with nan's where flux cadence is present and centroid cadence is not present
        - Removes centroid cadences where flux cadences not present

    :param all_time: flux time vector
    :param centroid_time: centroid time vector
    :param all_centroids: centroid vector
    :param thres: float, time cadence match threshold: 0.005 days = ~7 minutes
    :return:
        dict, synchronized all_centroids
    """

    # remove nans in centroid time vector. Nans yield invalid subtractions further on
    finite_id_centr = np.isfinite(centroid_time)
    centroid_time = centroid_time[finite_id_centr]
    all_centroids['x'] = all_centroids['x'][finite_id_centr]
    all_centroids['y'] = all_centroids['y'][finite_id_centr]

    # vertically stack flux time and centroid time, same for centroids and nans. 2nd column id's
    # centroid and flux cadences
    a = np.array([
        np.concatenate((all_time, centroid_time)),
        np.concatenate((np.full(len(all_time), 1), np.full(len(all_centroids['x']), 0))),
        np.concatenate((np.full(len(all_time), np.nan), all_centroids['x'])),
        np.concatenate((np.full(len(all_time), np.nan), all_centroids['y']))
    ]).transpose()

    # first sort by identifier, then by time cadence values. now every first of time cadences which occur in both
    # flux and centroid is of a centroid cadence
    a = a[np.lexsort((a[:, 1], a[:, 0]))]

    # collect rows to delete; keep only centroid cadences for cadences which flux also has and
    # delete centroid time cadences if flux does not have that time cadence
    rows_del = []
    for i in range(len(a) - 1):
        if abs(a[i][0] - a[i + 1][0]) < thres:
            rows_del.append(i + 1)
        elif a[i][1] == 0:
            rows_del.append(i)

    a = np.delete(a, rows_del, axis=0)

    return {'x': a[:, 2], 'y': a[:, 3]}


def convertpxtoradec_centr(centroid_x, centroid_y, cd_transform_matrix, ref_px_apert, ref_angcoord, satellite):
    """ Convert the centroid time series from pixel coordinates to world coordinates right ascension (RA) and
    declination (Dec).

    :param centroid_x: list [num cadences], column centroid position time series [pixel] in the CCD frame
    :param centroid_y: list, row centroid position time series [pixel] in the CCD frame
    :param cd_transform_matrix: numpy array [2x2], coordinates transformation matrix from col, row aperture frame
    to world coordinates RA and Dec
    # :param ref_px:  numpy array [2x1], reference pixel [pixel] coordinates in the aperture frame of the target star
    # frame
    :param ref_px_apert: numpy array [2x1], reference pixel [pixel] coordinates of the origin of the aperture frame in
    the CCD frame
    :param ref_angcoord: numpy array [2x1], RA and Dec at reference pixel [RA, Dec]
    :param satellite: str, either 'kepler' or 'tess'
    :return:
        ra: numpy array [num cadences], right ascension coordinate centroid time series
        dec: numpy array [num cadences], declination coordinate centroid time series
    """

    px_coords = np.reshape(np.concatenate((centroid_x, centroid_y)), (2, len(centroid_x)))

    if satellite == 'kepler':
        # offset in the aperture of [1, 1]
        ra, dec = np.matmul(cd_transform_matrix, px_coords - ref_px_apert + np.array([[1], [1]])) + ref_angcoord
    else:
        ra, dec = np.matmul(cd_transform_matrix, px_coords - ref_px_apert) + ref_angcoord

    return ra, dec


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
