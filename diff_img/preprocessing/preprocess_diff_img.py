"""
Preprocess extracted difference image data.

1 - Set negative pixels in OOT images to NaN.
2 - Get number of valid quarters.
3 - Sample quarters from valid quarters set. If no valid quarters, images are set to NaN and px positions are set to
zero. Valid quarters are defined as those whose target reference pixel uncertainty is not -1 AND quality metric is
valid (i.e., qX_valid is set to True).
4 - Pad with NaNs and crop images to desired size.
5 - Set images to NaN and px positions to zero if TCE is from a saturated target star.
"""

# 3rd party
import pandas as pd
import numpy as np
from pathlib import Path
import math
from datetime import datetime
import logging
import os
import multiprocessing
import argparse


def preprocess_diff_img_tces(diff_img_data, number_of_imgs_to_sample, pad_n_pxs,
                             final_dim, quality_metrics, saturated_tce_ids, mission, dest_dir, logger=None):
    """ Preprocessing pipeline for difference image data for a set of TCEs.

    Args:
        diff_img_data: dict, {tce_id: diff_img_data}, where `diff_img_data` is another dictionary of the type
        {'target_ref_centroid': [t_ref_1, t_ref_2, ...], 'image_data': [img_1, img_2, ...]}. `t_ref_i` is a dictionary
        of the type {'col': {'value': col_val, 'uncertainty': col_un},
        'row': {'value': col_val, 'uncertainty': col_un}}. This dictionary contains the target star position in each
        image and the associated uncertainty. The coordinates are in pixels and are relative to pixel (0, 0) in the
        image. `image_data` is a list of NumPy arrays. Each NumPy array is the set of images for a given
        quarter/sector run, and has dimensions [height, width, 4, 2]. The height and width are specific to the given
        target, and quarter/sector run. The 3rd dimension is associated with the different types of images: index 0 is
        the in-transit image, index 1 is the out-of-transit image, index 2 the difference image, and index 3 the SNR.
        The 4th dimension contains the pixel values and the uncertainties. Index 0 gives the values and index 1 the
        uncertainties.
        # n_max_imgs_avail: int, maximum number of quarters/sectors that are available for a given run. For Kepler that
        # should be always 17. For TESS, it depends on how many sectors are in a given sector run. Single-sector runs
        # contain only one image at the most.
        number_of_imgs_to_sample: int, number of quarters/sectors to sample
        pad_n_pxs: int, number of pixels to pad images in each dimension and side
        final_dim: int, final image size (final_dim, final_dim)
        quality_metrics: pandas DataFrame, quality metrics for TCEs
        saturated_tce_ids: list, TCE IDs associated with saturated targets
        mission: str, mission from where the difference image data is from. Either `kepler` or `tess`
        dest_dir: Path, destination directory for preprocessed data
        logger: logger, log

    Returns:
            diff_img_data, dict with added preprocessed data
            tces_info_df, pandas DataFrame with information on the preprocessing run
    """

    if mission == 'kepler':
        # if n_max_imgs_avail != 17:
        #     raise ValueError(f'Number of max images available `n_max_imgs_avail` was not set to 17 '
        #                      f'({n_max_imgs_avail}).')
        prefix = 'q'
    elif mission == 'tess':
        prefix = 's'
    else:
        raise ValueError(f'Mission not recognized ({mission}). Set variable to `kepler` or `tess`.')

    imgs_t = [int(col[1:-6]) for col in quality_metrics.columns if '_value' in col]
    n_max_imgs_avail = imgs_t[-1] - imgs_t[0] + 1

    proc_id = os.getpid()

    if logger is None:
        # set up logger for the process (when parallelizing)
        logger = logging.getLogger(name=f'preprocess_{proc_id}')
        logger_handler = logging.FileHandler(filename=dest_dir / f'preprocess_{proc_id}.log', mode='w')
        logger_formatter = logging.Formatter('%(asctime)s - %(message)s')
        logger.setLevel(logging.INFO)
        logger_handler.setFormatter(logger_formatter)
        logger.addHandler(logger_handler)
        logger.info(f'[{proc_id}] Starting preprocessing...')

    # initialized TCE table with information on the preprocessing
    tces_info_df = pd.DataFrame({'uid': [uid for uid in diff_img_data]})
    tces_info_df['oot_negative_values'] = ''
    for col in ['available', 'valid', 'sampled']:
        tces_info_df[f'{col}_{prefix}s'] = ''
    tces_info_df['saturated'] = False
    sampled_qmetrics = {'sampled_qmetrics': [np.nan * np.ones(number_of_imgs_to_sample)
                                             for _ in range(len(tces_info_df))]}

    # replace all negative values with nan in the oot images
    logger.info(f'[{proc_id}] Setting negative values in OOT images to NaNs...')
    for tce_uid in diff_img_data:  # iterate over TCEs

        neg_oot_str = ['0'] * len(diff_img_data[tce_uid]['image_data'])

        for i in range(len(diff_img_data[tce_uid]['image_data'])):  # iterate over each image
            curr_img = diff_img_data[tce_uid]['image_data'][i][:, :, 1, 0]  # oot check negatives
            if (curr_img < 0).sum() > 0:
                neg_oot_str[i] = '1'
            curr_img[curr_img < 0] = np.nan  # set to nan
            diff_img_data[tce_uid]['image_data'][i][:, :, 1, 0] = curr_img

        tces_info_df.loc[tces_info_df['uid'] == tce_uid, 'oot_negative_values'] = ''.join(neg_oot_str)

    # make a new dictionary key for cropped images
    for tce_i, tce_uid in enumerate(diff_img_data):  # tce_uid is id, tce_data is data

        if tce_i % 500 == 0:
            logger.info(f'[{proc_id}] Preprocessed {tce_i + 1} TCEs out of {len(diff_img_data)}.')

        # initialize fields for final difference and oot images features
        diff_img_data[tce_uid].update(
            {'cropped_imgs': {key: [] for key
                              in ['diff_imgs', 'oot_imgs', 'x', 'y', 'sub_x', 'sub_y', 'quality', 'imgs_numbers']}})

        # get the list of valid quarters/sectors
        curr_tce_df = quality_metrics[quality_metrics['uid'] == tce_uid]
        # get quarters/sectors with data
        available_imgs_idxs = [q_i for q_i in range(1, n_max_imgs_avail)
                               if ~np.isnan(curr_tce_df[f'{prefix}{imgs_t[q_i]}_value'].values[0])]
        tces_info_df.loc[tces_info_df['uid'] == tce_uid, [f'available_{prefix}s']] = (
            ''.join(['1' if q_i in available_imgs_idxs else '0' for q_i in range(n_max_imgs_avail)]))

        # get quarters/sectors with valid data
        valid_images_idxs = []
        for i, q_i in enumerate(available_imgs_idxs):
            # quarter/sector is valid and uncertainty in target position is not -1
            if (diff_img_data[tce_uid]['target_ref_centroid'][i]['col']['uncertainty'] != -1 and
                    (curr_tce_df[f'{prefix}{imgs_t[q_i]}_valid'].item())):
                valid_images_idxs.append(q_i)

        n_valid_imgs = len(valid_images_idxs)

        if n_valid_imgs == 0:  # if no valid quarters/sectors

            tces_info_df.loc[tces_info_df['uid'] == tce_uid, [f'valid_{prefix}s', f'sampled_{prefix}s']] = (
                '-' * n_max_imgs_avail, '-' * n_max_imgs_avail)

            # add to dictionary
            diff_img_data[tce_uid]['cropped_imgs']['diff_imgs'] = [np.nan * np.ones((final_dim, final_dim),
                                                                                    dtype='float')
                                                                   for _ in range(number_of_imgs_to_sample)]
            diff_img_data[tce_uid]['cropped_imgs']['oot_imgs'] = [np.nan * np.ones((final_dim, final_dim),
                                                                                   dtype='float')
                                                                  for _ in range(number_of_imgs_to_sample)]

            diff_img_data[tce_uid]['cropped_imgs']['x'] = [0] * number_of_imgs_to_sample
            diff_img_data[tce_uid]['cropped_imgs']['y'] = [0] * number_of_imgs_to_sample

            # subpixel coordinates
            diff_img_data[tce_uid]['cropped_imgs']['sub_x'] = [0] * number_of_imgs_to_sample
            diff_img_data[tce_uid]['cropped_imgs']['sub_y'] = [0] * number_of_imgs_to_sample

            # quality metric
            diff_img_data[tce_uid]['cropped_imgs']['quality'] = [np.nan] * number_of_imgs_to_sample

            # quarter/sector number to dictionary
            diff_img_data[tce_uid]['cropped_imgs']['imgs_numbers'] = [np.nan] * number_of_imgs_to_sample

            continue

        # randomly sample valid quarters/sectors
        if n_valid_imgs < number_of_imgs_to_sample:
            # use all quarters/sectors available before random sampling
            k_n_valid_imgs = number_of_imgs_to_sample // n_valid_imgs
            random_sample_imgs_idxs = np.tile(valid_images_idxs, k_n_valid_imgs)
            # fill the remaining spots by sampling randomly without replacement
            random_sample_imgs_idxs = np.concatenate([random_sample_imgs_idxs,
                                                      np.random.choice(valid_images_idxs,
                                                                       number_of_imgs_to_sample % n_valid_imgs,
                                                                       replace=False)])
        else:
            random_sample_imgs_idxs = np.random.choice(valid_images_idxs, number_of_imgs_to_sample, replace=False)

        tces_info_df.loc[tces_info_df['uid'] == tce_uid, [f'valid_{prefix}s', f'sampled_{prefix}s']] = (
            ''.join(['1' if q_i in valid_images_idxs else '0' for q_i in range(n_max_imgs_avail)]),
            ''.join([f'{(random_sample_imgs_idxs == q_i).sum()}' for q_i in range(n_max_imgs_avail)]))

        sampled_qmetrics['sampled_qmetrics'][tce_i] = [curr_tce_df[f'{prefix}{imgs_t[q_i]}_value'].item()
                                                       for q_i in random_sample_imgs_idxs]

        # pad and crop images
        for q_i in random_sample_imgs_idxs:

            i = valid_images_idxs.index(q_i)

            # pad with nans
            padded_diff_img = np.pad(diff_img_data[tce_uid]['image_data'][i][:, :, 2, 0],
                                     pad_n_pxs,
                                     mode='constant', constant_values=np.nan)

            padded_oot_img = np.pad(diff_img_data[tce_uid]['image_data'][i][:, :, 1, 0],
                                    pad_n_pxs,
                                    mode='constant', constant_values=np.nan)

            x = diff_img_data[tce_uid]['target_ref_centroid'][i]['col']['value']
            y = diff_img_data[tce_uid]['target_ref_centroid'][i]['row']['value']

            target_x_rounded = math.floor(x)
            target_y_rounded = math.floor(y)

            # center is shifted on both axis because it's padded by `pad_n_pxs`
            padded_pos_x = target_x_rounded + pad_n_pxs
            padded_pos_y = target_y_rounded + pad_n_pxs

            # create a new image matrix
            # crop image to be centered around target pixel
            side_pixel_cnt = final_dim // 2
            cropped_diff_img = padded_diff_img[padded_pos_y - side_pixel_cnt:padded_pos_y + side_pixel_cnt + 1,
                               padded_pos_x - side_pixel_cnt:padded_pos_x + side_pixel_cnt + 1]
            cropped_oot_img = padded_oot_img[padded_pos_y - side_pixel_cnt:padded_pos_y + side_pixel_cnt + 1,
                              padded_pos_x - side_pixel_cnt:padded_pos_x + side_pixel_cnt + 1]

            # # we know the target should be in center pixel (5,5)
            # # this uses the subpixel coordinates of the target in the target pixel
            # # to locate the correct subpixel position in the new center (5,5)
            # final_x = side_pixel_cnt - (target_x_rounded - x)
            # final_y = side_pixel_cnt - (target_y_rounded - y)

            # add to dictionary
            diff_img_data[tce_uid]['cropped_imgs']['diff_imgs'].append(cropped_diff_img)
            diff_img_data[tce_uid]['cropped_imgs']['oot_imgs'].append(cropped_oot_img)

            # add target pixel coordinates; center of the final image
            diff_img_data[tce_uid]['cropped_imgs']['x'].append(side_pixel_cnt)
            diff_img_data[tce_uid]['cropped_imgs']['y'].append(side_pixel_cnt)

            # add target subpixel coordinates
            diff_img_data[tce_uid]['cropped_imgs']['sub_x'].append((x - target_x_rounded))
            diff_img_data[tce_uid]['cropped_imgs']['sub_y'].append((y - target_y_rounded))

            # add quality metric
            diff_img_data[tce_uid]['cropped_imgs']['quality'].append(curr_tce_df[f'{prefix}{imgs_t[q_i]}_value'].item())

            # add current quarter/sector number to dictionary
            diff_img_data[tce_uid]['cropped_imgs']['imgs_numbers'].append(imgs_t[q_i])

    # adjust results for TCEs belonging to saturated targets
    logger.info(f'[{proc_id}] Adjusting data for TCEs in saturated targets...')
    tces_in_sat_tbl = [tce_id for tce_id in diff_img_data if tce_id in saturated_tce_ids]
    logger.info(f'[{proc_id}] Found {len(tces_in_sat_tbl)} TCEs in saturated targets')
    for tce_in_sat_tbl in tces_in_sat_tbl:

        tces_info_df.loc[tces_info_df['uid'] == tce_in_sat_tbl, 'saturated'] = True

        # set images to NaNs
        diff_img_data[tce_in_sat_tbl]['cropped_imgs']['diff_imgs'] = [np.nan * np.ones((final_dim,
                                                                                          final_dim),
                                                                                         dtype='float')
                                                                        for _ in range(number_of_imgs_to_sample)]

        diff_img_data[tce_in_sat_tbl]['cropped_imgs']['oot_imgs'] = [np.nan * np.ones((final_dim,
                                                                                         final_dim),
                                                                                        dtype='float')
                                                                       for _ in range(number_of_imgs_to_sample)]

        # set pixel target location to zero
        diff_img_data[tce_in_sat_tbl]['cropped_imgs']['x'] = [0] * number_of_imgs_to_sample
        diff_img_data[tce_in_sat_tbl]['cropped_imgs']['y'] = [0] * number_of_imgs_to_sample

        # set subpixel target location to zero
        diff_img_data[tce_in_sat_tbl]['cropped_imgs']['sub_x'] = [0] * number_of_imgs_to_sample
        diff_img_data[tce_in_sat_tbl]['cropped_imgs']['sub_y'] = [0] * number_of_imgs_to_sample

    tces_info_df = pd.concat([tces_info_df, pd.DataFrame(sampled_qmetrics)], axis=1, ignore_index=False)

    logger.info(f'[{proc_id}] Finished preprocessing difference image data for {len(diff_img_data)} TCEs.')

    return diff_img_data, tces_info_df


if __name__ == '__main__':

    # used in job arrays
    parser = argparse.ArgumentParser()
    parser.add_argument('--sat_thr', type=int, help='Saturation threshold.', default=12)
    # parser.add_argument('--n_max_imgs_avail', type=int, help='Maximum number of images available.', default=5)
    parser.add_argument('--num_sampled_imgs', type=int, help='Number of images to sample.', default=5)
    parser.add_argument('--pad_n_pxs', type=int, help='Number of pixels to pad images in each dimension and side.', default=20)
    parser.add_argument('--final_dim', type=int, help='Final image size (final_dim, final_dim).', default=11)
    parser.add_argument('--mission', type=str, help='Mission. Either `kepler` or `tess`.', default='kepler')
    parser.add_argument('--sat_tbl_fp', type=str, help='File path to table with magnitude values for the target star associated with each TCE.', default='/Users/msaragoc/Projects/exoplanet_transit_classification/data/ephemeris_tables/kepler/q1-q17_dr25/11-17-2021_1243/q1_q17_dr25_tce_3-6-2023_1734.csv')
    parser.add_argument('--dest_dir', type=str, help='Path to directory with the preprocessed results is going to be created.', default=f'/Users/msaragoc/Projects/exoplanet_transit_classification/data/fits_files/tess/2min_cadence_data/dv/preprocessing_step2/{datetime.now().strftime("%m-%d-%Y_%H%M")}')
    parser.add_argument('--diff_img_tbl_fp', type=str, help='File path to NumPy array with the extracted difference image data for a set of TCEs.', default='/Users/msaragoc/Projects/exoplanet_transit_classification/data/fits_files/kepler/q1_q17_dr25/dv/preprocessing/8-17-2022_1205/keplerq1q17_dr25_diffimg.npy')
    parser.add_argument('--qual_metrics_tbl_fp', type=str, help='File path to table with quality metrics for each image in each TCE.', default='/Users/msaragoc/Projects/exoplanet_transit_classification/data/fits_files/kepler/q1_q17_dr25/dv/diff_img_quality_metric.csv')
    args = parser.parse_args()

    # mission; either `tess` or `kepler`
    mission = args.mission  # 'tess'

    # destination file path to preprocessed data
    dest_dir = Path(args.dest_dir)  # Path('/Users/msaragoc/Projects/exoplanet_transit_classification/data/fits_files/tess/2min_cadence_data/dv/preprocessing_step2') / f'{datetime.now().strftime("%m-%d-%Y_%H%M")}'
    dest_dir.mkdir(exist_ok=True)

    # file path for file with difference image data
    diff_img_data_fp = Path(args.diff_img_tbl_fp)  # Path('/Users/msaragoc/Projects/exoplanet_transit_classification/data/fits_files/tess/2min_cadence_data/dv/preprocessing/3-1-2023_1422/data')
    # file path to quality metrics table
    qual_metrics_tbl_fp = Path(args.qual_metrics_tbl_fp)  # Path('/Users/msaragoc/Projects/exoplanet_transit_classification/data/fits_files/kepler/q1_q17_dr25/dv/diff_img_quality_metric.csv')
    # file path to table with information on saturated stars
    saturated_tbl_fp = Path(args.sat_tbl_fp)  # Path('/Users/msaragoc/Projects/exoplanet_transit_classification/data/ephemeris_tables/kepler/q1-q17_dr25/11-17-2021_1243/q1_q17_dr25_tce_3-6-2023_1734.csv')
    sat_thr = args.sat_thr  # 12  # saturated target threshold
    # n_max_imgs_avail = args.n_max_imgs_avail  # 17  # maximum number of images available
    num_sampled_imgs = args.num_sampled_imgs  # 5  # number of quarters/sectors to get
    pad_n_pxs = args.pad_n_pxs  # 20  # padding original images with this number of pixels
    final_dim = args.final_dim  # 11  # dimension of final image

    # set up logger
    logger = logging.getLogger(name=f'preprocess')
    logger_handler = logging.FileHandler(filename=dest_dir / f'preprocess.log', mode='w')
    logger_formatter = logging.Formatter('%(asctime)s - %(message)s')
    logger.setLevel(logging.INFO)
    logger_handler.setFormatter(logger_formatter)
    logger.addHandler(logger_handler)
    logger.info(f'Starting preprocessing...')

    # load difference image data
    logger.info(f'Loading difference image data from {diff_img_data_fp}')
    diff_img_data = np.load(diff_img_data_fp, allow_pickle=True).item()
    # diff_img_data = {k: v for k, v in diff_img_data.items() if k == '757099-1'}

    logger.info(f'Number of TCEs to preprocess: {len(diff_img_data)}')
    # load table with quality metrics
    logger.info(f'Loading quality metrics data from {qual_metrics_tbl_fp}')
    quality_metrics = pd.read_csv(qual_metrics_tbl_fp)

    # load table with information on saturated target stars
    logger.info(f'Loading saturated target data from {saturated_tbl_fp}')
    saturated = pd.read_csv(saturated_tbl_fp)
    # get only the TCE ids for saturated targets
    saturated_tce_ids = saturated.loc[((saturated['mag'] < sat_thr) &
                                       (saturated['uid'].isin(list(diff_img_data.keys())))), 'uid'].to_list()

    logger.info(f'Number of saturated TCEs found: {len(saturated_tce_ids)}')

    # # sequential
    # diff_img_data, tces_info_df = preprocess_diff_img_tces(diff_img_data, num_sampled_imgs,
    #                                                        pad_n_pxs, final_dim, quality_metrics,
    #                                                        saturated_tce_ids, mission, dest_dir, logger)

    # parallelize work
    n_processes = 4
    n_jobs = 4
    tces_ids = np.array_split(np.array(list(diff_img_data.keys())), n_jobs)
    pool = multiprocessing.Pool(processes=n_processes)
    jobs = [({tce_id: tce_diff_data for tce_id, tce_diff_data in diff_img_data.items() if tce_id in tces_ids_job},
             num_sampled_imgs, pad_n_pxs, final_dim, quality_metrics,
             saturated_tce_ids, mission, dest_dir)
            for job_i, tces_ids_job in enumerate(tces_ids)]
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

    # adding additional TCE information to the preprocessing table
    tces_info_df = tces_info_df.merge(
        saturated[['uid', 'target_id', 'label',
                   # 'kepoi_name',
                   'matched_toi_our',
                   'tce_period',
                   # 'transit_depth',
                   'tce_depth',
                   'tce_max_mult_ev', 'mag', 'tce_dikco_msky', 'tce_dikco_msky_err',
                   'tce_dicco_msky', 'tce_dicco_msky_err',
                   # 'tce_fwm_stat',
                   'ruwe',
                   # 'tce_datalink_dvs', 'tce_datalink_dvr'
                   ]],
        how='left', on='uid', validate='one_to_one')

    logger.info(f'Saving preprocessed data to {dest_dir / "diffimg_preprocess.npy"}...')
    tces_info_df.to_csv(dest_dir / 'info_tces.csv', index=False)
    np.save(dest_dir / "diffimg_preprocess.npy", diff_img_data)

    logger.info(f'Finished.')
