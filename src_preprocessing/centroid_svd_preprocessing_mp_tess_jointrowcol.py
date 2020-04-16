# 3rd party
import pandas as pd
import numpy as np
from astropy.io import fits
import multiprocessing
import scipy.optimize as optimize
from sklearn.linear_model import Ridge, HuberRegressor, Lasso, BayesianRidge, ElasticNet, LassoLarsIC
import datetime
import os
import logging
import itertools

# local
from src_preprocessing import tess_io
from src_preprocessing.utils_centroid_svd_preprocessing import minimizer, l2_reg, l1_reg, report_exclusion, _has_finite
from src_preprocessing.utils_ephemeris import create_binary_time_series, find_first_epoch_after_this_time, \
    lininterp_transits, get_startend_idxs_inter


def preprocess_centroidtimeseries(sectcamccd, ticids, eph_tbl, lc_data_dir, save_dir, logger=None):
    """ TESS centroid time-series preprocessing for denoising performed per sector-camera-CCD, joint design matrices for
     row and col coordinates (px).

    :param sectcamccd: tuple, sector-camera-CCD to be processed
    :param eph_tbl: pandas DataFrame, TCE ephemeris table
    :param ticids:  list, Kepler IDs to be processed
    :param lc_data_dir: str, root directory for the FITS files
    :param save_dir: str, root directory for saving the data generated
    :return:
    """

    # initialize variables
    data_mat = []
    time_arr = {}
    # centr_tend = []
    # idxs_nnan = []
    ticids_aux = {}
    raw_centroid_data = {}

    # iterate through the target stars to get the data for this channel
    for ticid_i, ticid in enumerate(ticids):

        print('TIC ID {} {}/{} | Sector {} | {} % read'.format(ticid, ticid_i, len(ticids), sectcamccd[0],
                                                               ticid_i / len(ticids) * 100))

        # get FITS filepaths for the TIC ID and this sector
        ticid_fits_filenames, _ = tess_io.tess_filenames(lc_data_dir, ticid, sectors=[sectcamccd[0]],
                                                         check_existence=True)

        print('Found {} files for TIC ID {}'.format(len(ticid_fits_filenames), ticid))
        # if len(ticid_fits_filenames) == 0:
        #     logger.info('No FITS files found for TIC ID {} in sector {}'.format(ticid, sector))

        # iterate through the FITS files
        for filename in ticid_fits_filenames:

            # get header of the fits file
            fits_header = fits.getheader(filename)

            # skip other camera-CCD pairs
            if fits_header['CAMERA'] != sectcamccd[1] or fits_header['CCD'] != sectcamccd[2]:
                continue

            # get data from the FITS file
            with fits.open(filename, mode="readonly", memmap=False) as hdu_list:

                # TODO: only allow centroid time-series with a fraction of NaN values below a given threshold
                # check if there is at least one finite value in both row and col centroid time-series
                # # (prefer PSF to MOM if available)
                # if _has_finite(hdu_list['LIGHTCURVE'].data.PSF_CENTR1) and \
                #         _has_finite(hdu_list['LIGHTCURVE'].data.PSF_CENTR2):
                #     centroid_x, centroid_y = hdu_list['LIGHTCURVE'].data.PSF_CENTR1, \
                #                              hdu_list['LIGHTCURVE'].data.PSF_CENTR2
                # else:
                if np.any(np.isfinite(hdu_list['LIGHTCURVE'].data.MOM_CENTR1)) and \
                        np.any(np.isfinite(hdu_list['LIGHTCURVE'].data.MOM_CENTR2)):

                    centroid_x, centroid_y = hdu_list['LIGHTCURVE'].data.MOM_CENTR1 - \
                                             hdu_list['LIGHTCURVE'].data.POS_CORR1, \
                                             hdu_list['LIGHTCURVE'].data.MOM_CENTR2 - \
                                             hdu_list['LIGHTCURVE'].data.POS_CORR2
                else:
                    print('Centroid data for {} is all NaNs'.format(filename))
                    report_exclusion({'sector': sectcamccd[0], 'camera': sectcamccd[1], 'ccd': sectcamccd[2],
                                      'ticid': ticid},
                                     filename,
                                     id_str='Centroid data is all NaNs',
                                     savedir=os.path.join(save_dir, 'exclusion_logs'))
                    continue  # no data

                # add centroid time-series to the design matrix
                data_mat.append(centroid_x)
                data_mat.append(centroid_y)

                # add time array for the given target
                time_arr[ticid] = hdu_list['LIGHTCURVE'].data.TIME

                ticids_aux[ticid] = len(data_mat) - 2

    # convert to numpy array and transpose the data matrix so that each column is one observation (centroid time-series)
    data_mat = np.array(data_mat, dtype='float').T
    logger.info('# Before dealing with NaNs')
    logger.info('Matrix shape: {}'.format(data_mat.shape))
    # time_mat = {q: np.array(time_mat[q], dtype='float').T for q in time_mat}
    # for q in time_mat:
    #     if not np.all([np.array_equal(time_mat[q][:, 0], time_mat[q][:, i]) for i in range(1, time_mat[q].shape[1])]):
    #         raise ValueError('the time arrays are not equal in quarter {} channel {}'.format(q, ch))

    # deal with NaN values in the centroid time-series
    # TODO: replace NaN values instead of removing them
    print('Dealing with NaN values for Sector {} | Camera {} | CCD {}...'.format(*sectcamccd))
    logger.info('# After dealing with NaNs')
    if len(data_mat) == 0:
        print('Empty raw data matrix for Sector {} Camera {} CCD {}'.format(*sectcamccd))
        report_exclusion({'sector': sectcamccd[0], 'camera': sectcamccd[1], 'ccd': sectcamccd[2]}, 'No filename',
                         id_str='Empty raw data matrix',
                         savedir=os.path.join(save_dir, 'exclusion_logs'))
        return None

    # # option 1 - assume centroid time series noise is gaussian, use a robust estimator of the std, and
    # # impute it into the time series
    # # TODO: compute the median in a small window around the NaN values
    # med_centroidx = np.median(centroid_x)
    # med_centroidy = np.median(centroid_y)
    # std_rob_estmx = np.median(np.abs(centroid_x - med_centroidx)) * 1.4826
    # std_rob_estmy = np.median(np.abs(centroid_y - med_centroidy)) * 1.4826
    # centroid_xnan = np.isnan(centroid_x)
    # centroid_ynan = np.isnan(centroid_y)
    # centroid_x[centroid_xnan] = med_centroidx - \
    #                             np.random.normal(0, std_rob_estmx, np.nonzero(centroid_xnan)[0].shape)
    # centroid_y[centroid_ynan] = med_centroidy - \
    #                             np.random.normal(0, std_rob_estmy, np.nonzero(centroid_ynan)[0].shape)
    #
    # data_mat[camera_ccd].append(centroid_x)
    # data_mat[camera_ccd].append(centroid_y)

    # option 2 - remove indices for all target stars if at least one target shows a nan value for the centroid or
    # cadence time-series
    idxs_nnan = np.nonzero(np.all(np.isfinite(data_mat), axis=1))
    data_mat = data_mat[idxs_nnan]
    # removing indexes from the time arrays
    for ticid in ticids_aux:
        time_arr[ticid] = time_arr[ticid][idxs_nnan]

    logger.info('Sector {} | Camera {} | CCD {}: {}'.format(sectcamccd[0], sectcamccd[1], sectcamccd[2],
                                                            data_mat.shape))

    # removing mean value from each column in the data matrix
    # TODO: use a robust estimator of the mean
    print('Removing central tendency for sector {}...'.format(sectcamccd[0]))
    # get central tendency
    # median is more robust to outliers than mean
    centr_tend = np.nanmedian(data_mat, axis=0)

    # remove the central tendency
    data_mat -= centr_tend

    # get the raw centroid time series for each target star
    for ticid in ticids_aux:
        raw_centroid_data[ticid] = {'x': data_mat[:, ticids_aux[ticid]],
                                    'y': data_mat[:, ticids_aux[ticid] + 1]}

    # linearly interpolate across transits
    print('Linearly interpolating across the transits for sector {}...'.format(sectcamccd[0]))
    # instantiate the gapped data matrix
    data_mat_gapped = np.copy(data_mat)

    for ticid in ticids_aux:

        # instantiate binary time-series for the Kepler ID
        binary_time_series = np.zeros(len(time_arr[ticid]), dtype='uint8')

        # get ephemeris of TCEs detected in the target star
        # eph_tces = eph_tbl.loc[eph_tbl['target_id'] == ticid][['tce_period', 'tce_time0bk', 'tce_duration']]

        # no sector column in the TCE list
        # eph_tces = eph_tbl.loc[eph_tbl['target_id'] == ticid][['tce_period', 'tce_time0bk', 'tce_duration']]

        # sector column in the TCE list - gap only TCEs that were observed in this sector
        eph_tces = eph_tbl.loc[eph_tbl['target_id'] == ticid][['tce_period',
                                                               'tce_time0bk',
                                                               'tce_duration',
                                                               'sector']]
        # get TCEs whose observed sectors overlap with the observed sectors for the current TCE
        candidatesRemoved = []
        gapSectors = {}
        for i, candidate in eph_tces.iterrows():
            candidateSectors = candidate.sector  # candidate.sector.split(' ')
            # get only overlapping sectors
            sectorsIntersection = np.intersect1d(sectcamccd[0], candidateSectors)
            if len(sectorsIntersection) > 0:
                gapSectors[len(gapSectors)] = sectorsIntersection
            else:
                candidatesRemoved.append(i)
        # remove candidates that do not have any overlapping sectors
        eph_tces.drop(eph_tces.index[candidatesRemoved], inplace=True)

        if len(eph_tces) == 0:  # no TCEs to be gapped
            continue

        # create binary time-series with 1's for in-transit cadences for all TCEs detected in the target star
        for tce_i, tce in eph_tces.iterrows():
            # get first transit time for the TCE
            first_transit_time = find_first_epoch_after_this_time(tce.tce_time0bk,
                                                                  tce.tce_period,
                                                                  time_arr[ticid][0])

            # create binary time-series for the TCE
            binary_time_series_tce = create_binary_time_series(time_arr[ticid],
                                                               first_transit_time,
                                                               tce.tce_duration / 24,
                                                               tce.tce_period)

            # flag in-transit cadences for the TCE
            binary_time_series[np.where(binary_time_series_tce == 1)] = 1

        # perform gapping only if there are any transits for the target star
        if len(np.where(binary_time_series == 1)[0]) > 0:

            # get indexes for linear interpolation across the in-transit cadences
            idxs_it, start_idxs, end_idxs = get_startend_idxs_inter([binary_time_series])

            # linear interpolation across the in-transit cadences for row and col time-series
            # 'x'-coordinate
            data_mat_gapped[:, ticids_aux[ticid]] = lininterp_transits([data_mat[:, ticids_aux[ticid]]],
                                                                       [binary_time_series],
                                                                       idxs_it,
                                                                       start_idxs,
                                                                       end_idxs)[0]
            # 'y'-coordinate
            data_mat_gapped[:, ticids_aux[ticid] + 1] = lininterp_transits([data_mat[:, ticids_aux[ticid] + 1]],
                                                                           [binary_time_series],
                                                                           idxs_it,
                                                                           start_idxs,
                                                                           end_idxs)[0]

    # saving data
    print('Saving pre-denoising data for sector {} camera {} CCD {}...'.format(*sectcamccd))
    np.save(os.path.join(save_dir, 'ppdata', 'datamatrices_s{}_cam{}_ccd{}.npy'.format(*sectcamccd)), data_mat)
    np.save(os.path.join(save_dir, 'ppdata', 'gappeddatamatrices_s{}_cam{}_ccd{}.npy'.format(*sectcamccd)),
            data_mat_gapped)
    # np.save('{}ppdata/time_ch{}.npy'.format(save_dir, ch), time_mat)
    np.save(os.path.join(save_dir, 'ppdata', 'centraltendency_s{}_cam{}_ccd{}.npy'.format(*sectcamccd)), centr_tend)
    np.save(os.path.join(save_dir, 'ppdata', 'centroidtimeseries_s{}_cam{}_ccd{}.npy'.format(*sectcamccd)),
            raw_centroid_data)
    np.save(os.path.join(save_dir, 'ppdata', 'idxsnotnan_s{}_cam{}_ccd{}.npy'.format(*sectcamccd)), idxs_nnan)
    np.save(os.path.join(save_dir, 'ppdata', 'ticids_s{}_cam{}_ccd{}.npy'.format(*sectcamccd)), ticids_aux)


def pca_denoising(sectcamccd, num_singularvalues, save_dir, logger=None):
    """ Performs denoising on the centroid time-series for a given sector. The camera-CCD design matrices are assumed to
    be mxn, with N centroid time-series each with M points; Each centroid time-series was previously centered.

    :param sectcamccd: tuple, sector-camera-CCD to be processed
    :param num_singularvalues: int, number of singular values used when truncating the SVD matrices
    :param save_dir: str, root directory for saving the data generated
    :return:
    """

    # instantiate variables
    # singular_values = {}  # {q: [] for q in range(1, NUM_QUARTERS + 1)}
    centroid_data = {}  # {q: [] for q in range(1, NUM_QUARTERS + 1)}

    # load data
    data_mat = np.load(os.path.join(save_dir, 'ppdata', 'datamatrices_s{}_cam{}_ccd{}.npy'.format(*sectcamccd)),
                       allow_pickle=True)
    data_mat_gapped = np.load(os.path.join(save_dir, 'ppdata',
                                           'gappeddatamatrices_s{}_cam{}_ccd{}.npy'.format(*sectcamccd)),
                              allow_pickle=True)
    centr_tend = np.load(os.path.join(save_dir, 'ppdata', 'centraltendency_s{}_cam{}_ccd{}.npy'.format(*sectcamccd)),
                         allow_pickle=True)
    ticids_aux = np.load(os.path.join(save_dir, 'ppdata', 'ticids_s{}_cam{}_ccd{}.npy'.format(*sectcamccd)),
                         allow_pickle=True).item()

    print('Performing denoising for sector {} camera {} CCD {}...'.format(*sectcamccd))
    # preprocess the raw data
    if len(data_mat) == 0:
        print('Empty data matrix before SVD')
        report_exclusion({'channel': sectcamccd[0], 'camera': sectcamccd[1], 'ccd': sectcamccd[2]}, 'No filename',
                         id_str='Empty raw data matrix before SVD',
                         savedir=os.path.join(save_dir, 'exclusion_logs'))
        return None

    # compute SVD for the gapped design matrix

    # Full SVD: A [mxn] = U [mxm] * S [mxn] * V^T [nxn]
    svd_comps = np.linalg.svd(data_mat_gapped)

    # get the singular values
    singular_values = svd_comps[1]

    # TODO: implement criterion to choose number of components

    # print('Finding minimization coefficients for sector {} camera {} CCD {}...'.format(*sectcamccd))

    # # robust LS
    # beta = np.zeros((num_singularvalues, data_mat.shape[1]), dtype='float')
    # for col_i in range(data_mat].shape[1]):
    #
    #     print('Robust LS for sector {} camera {} CCD {} - coefs {}/{}'.format(*sectcamccd, col_i, data_mat.shape[1]))
    #
    #     # TODO: choose a better initialization?
    #     x0 = np.random.rand(num_singularvalues)
    #
    #     loss = 'linear'
    #     lambda_reg = 0
    #     reg_func = l2_reg
    #
    #     result = optimize.least_squares(minimizer,
    #                                     x0,
    #                                     args=(data_mat][:, col_i],
    #                                           svd_comps[0][:, :num_singularvalues],
    #                                           lambda_reg,
    #                                           reg_func),
    #                                     loss=loss,
    #                                     method='trf')
    #
    #     beta[:, col_i] = result.x

    # OLS with L2 regularizer (Ridge)
    # print('Performing Ridge regularization for sector {} camera {} CCD {}...'.format(*sectcamccd))
    # alpha = 10

    # clf_ridge = Ridge(alpha=alpha, solver='auto', fit_intercept=True)
    # # clf_ridge = Lasso(alpha=alpha, fit_intercept=True)
    # clf_ridge.fit(svd_comps[0][:, :num_singularvalues], data_mat_gapped)
    # beta = clf_ridge.coef_.T

    # clf_ridge = HuberRegressor(alpha=alpha, fit_intercept=True, epsilon=1.35)
    # beta = np.zeros((num_singularvalues, data_mat.shape[1]), dtype='float')
    # for col_i in range(data_mat.shape[1]):
    #     clf_ridge.fit(svd_comps[0][:, :num_singularvalues], data_mat[:, col_i])
    #     beta[:, col_i] = clf_ridge.coef_.T

    # print('Denoising design matrix for sector {} camera {} CCD {}...'.format(*sectcamccd))

    # # Vanilla Truncated SVD: remove the components associated with the largest singular values
    # # A_tr [mxn] = U [mxk] * S [kxk] * V^T [kxn]
    # # A_new [mxn] = A [mxn] - A_tr [mxn]
    data_mat -= np.dot(svd_comps[0][:, :num_singularvalues] *
                       svd_comps[1][:num_singularvalues],
                       svd_comps[2][:num_singularvalues, :])

    # # Optimized coefficients for SVD
    # # A_tr [mxn] = U [mxk] * beta [kxn]
    # # A_new [mxn] = A [mxn] - A_tr [mxn]
    # data_mat[camera_ccd] -= np.dot(svd_comps[0][:, :num_singularvalues], beta)

    # add back the central tendency
    data_mat += centr_tend

    # get the preprocessed centroid time series for each target star
    for ticid in ticids_aux:
            centroid_data[ticid] = {'x': data_mat[:, ticids_aux[ticid]],
                                    'y': data_mat[:, ticids_aux[ticid] + 1]}

    # save denoised data
    print('Saving preprocessed data for sector {} camera {} CCD {}...'.format(*sectcamccd))
    np.save(os.path.join(save_dir, 'pcadenoising', 'singularvalues_s{}_cam{}_ccd{}.npy'.format(*sectcamccd)),
            singular_values)
    np.save(os.path.join(save_dir, 'pcadenoising', 'denoiseddatamatrices_s{}_cam{}_ccd{}.npy'.format(*sectcamccd)),
            data_mat)
    np.save(os.path.join(save_dir, 'pcadenoising',
                         'denoisedcentroidtimeseries_s{}_cam{}_ccd{}.npy'.format(*sectcamccd)),
            centroid_data)


def pca_denoising_channels(sectcamccds, num_singularvalues, ticids, eph_tbl, lc_data_dir, save_dir, logger=None):
    """ Perform PCA denoising on a set of channels sequentially.

    :param sectcamccds: list, sector-camera-CCD sets
    :param num_singularvalues: int, number of singular values used when truncating the SVD matrices
    :param ticids: list, target list
    :param eph_tbl: pandas DataFrame, TCE ephemeris table
    :param lc_data_dir: str, root directory for the FITS files
    :param save_dir: str, root directory for saving the data generated
    :return:
    """

    for sectcamccd in sectcamccds:
        logger_channel = logging.getLogger(name='log_sector{}_camera{}_ccd{}'.format(*sectcamccd))
        logger_handler = logging.FileHandler(filename=os.path.join(save_dir,
                                                               'sector{}_camera{}_ccd{}.log'.format(
                                                                   *sectcamccd)),
                                         mode='a')
        logger_formatter = logging.Formatter('%(asctime)s - %(message)s')

        logger_channel.setLevel(logging.INFO)
        logger_handler.setFormatter(logger_formatter)
        logger_channel.addHandler(logger_handler)

        logger_channel.info('--- Sector-Camera-CCD {} ---'.format(sectcamccd))

        # prepare data for denoising
        preprocess_centroidtimeseries(sectcamccd, ticids, eph_tbl, lc_data_dir, save_dir, logger_channel)
        # denoising the centroid time-series
        pca_denoising(sectcamccd, num_singularvalues, save_dir, logger_channel)


if __name__ == '__main__':

    # NUM_SECTORS = 1-...
    # 4 cameras on a sequential grid
    # NUM_CAMERAS = 4
    # 4 CCDs in each camera on a square grid [[3, 4], [2, 1]]; each row has the same orientation
    # NUM_CCDS = 4

    # root save directory
    root_save_dir = '/home/msaragoc/Projects/Kepler-TESS_exoplanet/Data/centroid_pca_denoising/TESS'
    # list of TIC IDs
    ticids = np.load('/data5/tess_project/Data/TESS_TOI_fits(MAST)/final_target_list_s1-s19.npy',
                     allow_pickle=True)
    # FITS files root directory
    lc_data_dir = '/data5/tess_project/Data/TESS_TOI_fits(MAST)'

    # load TCE ephemeris table - needed to get the list of target stars and to gap the transits from the centroid
    # time-series
    tce_tbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/TESS/'
                          'toi_list_ssectors_dvephemeris_ephmatchnoepochthr0,25.csv')
    tce_tbl.rename(columns={'tic': 'target_id',
                            'transitDurationHours': 'tce_duration',
                            'orbitalPeriodDays': 'tce_period',
                            'transitEpochBtjd': 'tce_time0bk'}, inplace=True)

    # define save directory for the current study
    study = 'test'  # + datetime.datetime.now()
    # save_dir = os.path.join(root_save_dir, '{}_{}'.format(study, datetime.datetime.now()))
    save_dir = os.path.join(root_save_dir, study)

    # create folders for data
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'ppdata'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'pcadenoising'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'exclusion_logs'), exist_ok=True)

    num_singularvalues = 6  # number of top singular components to remove

    # sectors to be processed
    sectors = [5]
    cameras = [1]
    ccds = np.arange(1, 4 + 1)
    sectcamccd = list(itertools.product(sectors, cameras, ccds))

    n_procs = 1  # number of processes to span
    jobs = []

    print('Number of total targets = {}'.format(len(ticids)))
    print('Number of sector-camera-CCD (per process) = {} (~{})'.format(len(sectcamccd),
                                                                        int(len(sectcamccd) / n_procs)))
    print('Number of processes = {}'.format(n_procs))

    logger_main = logging.getLogger(name='log_main')
    logger_handler = logging.FileHandler(filename=os.path.join(save_dir, 'main.log'),
                                         mode='a')
    logger_formatter = logging.Formatter('%(asctime)s - %(message)s')

    logger_main.setLevel(logging.INFO)
    logger_handler.setFormatter(logger_formatter)
    logger_main.addHandler(logger_handler)

    logger_main.info('Total number of target stars = {}'.format(len(ticids)))
    logger_main.info('Total number of sector-camera-CCD = {}'.format(len(sectcamccd)))
    logger_main.info('Number of processes = {}'.format(n_procs))
    logger_main.info('Number of singular values removed = {}'.format(num_singularvalues))
    logger_main.info('Sector-camera-CCD sets processed:\n{}'.format(sectcamccd))

    # distribute channels across the channels
    boundaries = [int(i) for i in np.linspace(0, len(sectors), n_procs + 1)]

    # each process handles a subset of the channels
    for proc_i in range(n_procs):
        indices = [(boundaries[i], boundaries[i + 1]) for i in range(n_procs)][proc_i]
        sectcamccd_proc = sectcamccd[indices[0]:indices[1]]
        p = multiprocessing.Process(target=pca_denoising_channels, args=(sectcamccd_proc,
                                                                         num_singularvalues,
                                                                         ticids,
                                                                         tce_tbl,
                                                                         lc_data_dir,
                                                                         save_dir))
        jobs.append(p)
        p.start()

    map(lambda p: p.join(), jobs)
