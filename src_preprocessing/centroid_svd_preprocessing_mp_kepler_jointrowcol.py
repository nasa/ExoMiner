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

# local
from src_preprocessing.light_curve import kepler_io
from src_preprocessing.utils_centroid_svd_preprocessing import minimizer, l2_reg, l1_reg, report_exclusion, _has_finite
from src_preprocessing.utils_ephemeris import create_binary_time_series, find_first_epoch_after_this_time, \
    lininterp_transits, get_startend_idxs_inter


def preprocess_centroidtimeseries(channel, kepids, eph_tbl, lc_data_dir, save_dir, logger=None):
    """ Kepler centroid preprocessing for denoising performed per channel-quarter, joint design matrices for row and col
     coordinates (px).

    :param channel: int, channel to be processed
    :param eph_tbl: pandas DataFrame, TCE ephemeris table
    :param kepids:  list, Kepler IDs to be processed
    :param lc_data_dir: str, root directory for the FITS files
    :param save_dir: str, root directory for saving the data generated
    :return:
    """

    # initialize variables
    data_mat = {}  # {q: [] for q in range(1, NUM_QUARTERS + 1)}
    # time_mat = {q: [] for q in range(1, NUM_QUARTERS + 1)}
    time_arr = {}  # {kepid: {} for kepid in kepids}
    centr_tend = {}  # {q: [] for q in range(1, NUM_QUARTERS + 1)}
    idxs_nnan = {}  # {q: [] for q in range(1, NUM_QUARTERS + 1)}
    # kepids_aux = {q: [] for q in range(1, NUM_QUARTERS + 1)}
    kepids_aux = {}  # {kepid: {'quarters': []} for kepid in kepids}
    # raw_centroid_data = {q: [] for q in range(1, NUM_QUARTERS + 1)}
    raw_centroid_data = {}

    # iterate through the target stars to get the data for this channel
    for kepid_i, kepid in enumerate(kepids):

        print('Kepler ID {} {}/{} | Channel {} | {} % read'.format(kepid, kepid_i, len(kepids), channel,
                                                                   kepid_i / len(kepids) * 100))

        # get fits filepaths for the Kepler ID
        kepid_fits_filenames = kepler_io.kepler_filenames(lc_data_dir, kepid)

        print('Found {} files for Kepler ID {}'.format(len(kepid_fits_filenames), kepid))
        for filename in kepid_fits_filenames:  # kepids_fits_filenames[kepid]:

            # get header of the fits file
            fits_header = fits.getheader(filename)

            # skip other channels and quarter zero
            if fits_header['CHANNEL'] != channel or fits_header['QUARTER'] == 0:
                continue

            # with fits.open(gfile.Open(filename, "rb")) as hdu_list:
            with fits.open(filename, mode="readonly") as hdu_list:

                # channel = hdu_list["PRIMARY"].header["CHANNEL"]
                quarter = hdu_list["PRIMARY"].header["QUARTER"]

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
                    report_exclusion({'channel': channel, 'quarter': quarter, 'kepid': kepid}, filename,
                                     id_str='Centroid data is all NaNs',
                                     savedir=os.path.join(save_dir, 'exclusion_logs'))
                    continue  # no data

                # add centroid time-series to the design matrix
                if quarter in data_mat:
                    data_mat[quarter].append(centroid_x)
                    data_mat[quarter].append(centroid_y)
                else:  # first centroid time-series being added to the data matrix for this quarter
                    data_mat[quarter] = [centroid_x, centroid_y]

                # add time array for this quarter for the given target
                if kepid in time_arr:
                    time_arr[kepid][quarter] = hdu_list['LIGHTCURVE'].data.TIME
                else:  # first time array being added for this quarter
                    time_arr[kepid] = {quarter: hdu_list['LIGHTCURVE'].data.TIME}

                if kepid not in kepids_aux:
                    # kepids_aux[kepid] = {'quarters': [quarter], 'indexes': [len(data_mat[quarter]) - 2]}
                    kepids_aux[kepid] = {quarter: len(data_mat[quarter]) - 2}
                else:
                    # kepids_aux[kepid]['quarters'].append(quarter)
                    # kepids_aux[kepid]['indexes'].append(len(data_mat[quarter]) - 2)
                    kepids_aux[kepid][quarter] = len(data_mat[quarter]) - 2

    # convert to numpy array and transpose the data matrix so that each column is one observation (centroid time-series)
    data_mat = {q: np.array(data_mat[q], dtype='float').T for q in data_mat}
    logger.info('# Before dealing with NaNs')
    for q in data_mat:
        logger.info('Channel {} | Quarter {}: {}'.format(channel, q, data_mat[q].shape))
    # time_mat = {q: np.array(time_mat[q], dtype='float').T for q in time_mat}
    # for q in time_mat:
    #     if not np.all([np.array_equal(time_mat[q][:, 0], time_mat[q][:, i]) for i in range(1, time_mat[q].shape[1])]):
    #         raise ValueError('the time arrays are not equal in quarter {} channel {}'.format(q, ch))

    # deal with NaN values in the centroid time-series
    # TODO: replace NaN values instead of removing them
    print('Dealing with NaN values for channel {}...'.format(channel))
    logger.info('# After dealing with NaNs')
    for q in data_mat:

        if len(data_mat[q]) == 0:
            print('Empty raw data matrix for Channel {} quarter {}'.format(channel, q))
            report_exclusion({'channel': channel, 'quarter': q}, 'No filename',
                             id_str='Empty raw data matrix',
                             savedir=os.path.join(save_dir, 'exclusion_logs'))
            continue

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
        # data_mat[quarter].append(centroid_x)
        # data_mat[quarter].append(centroid_y)

        # option 2 - remove indices for all target stars if at least one target shows a nan value for the centroid or
        # cadence time-series
        idxs_nnan[q] = np.nonzero(np.all(np.isfinite(data_mat[q]), axis=1))
        data_mat[q] = data_mat[q][idxs_nnan[q]]
        # removing indexes from the time arrays
        for kepid in kepids_aux:
            if q in kepids_aux[kepid]:  # check if there is any data for the Kepler ID for this quarter
                if kepid in time_arr:
                    time_arr[kepid][q] = time_arr[kepid][q][idxs_nnan[q]]
                else:
                    time_arr[kepid] = {q: time_arr[kepid][q][idxs_nnan[q]]}

        logger.info('Channel {} | Quarter {}: {}'.format(channel, q, data_mat[q].shape))

    # removing mean value from each column in the data matrix
    # TODO: use a robust estimator of the mean
    print('Removing central tendency for channel {}...'.format(channel))
    for q in data_mat:

        # get central tendency
        # median is more robust to outliers than mean
        centr_tend[q] = np.nanmedian(data_mat[q], axis=0)

        # remove the central tendency
        data_mat[q] -= centr_tend[q]

    # get the raw centroid time series for each target star
    for q in data_mat:
        for kepid in kepids_aux:
            # check if there is any data for the Kepler ID for this quarter
            if q in kepids_aux[kepid]:
                if kepid not in raw_centroid_data:
                    raw_centroid_data[kepid] = {q: {'x': data_mat[q][:, kepids_aux[kepid][q]],
                                                    'y': data_mat[q][:, kepids_aux[kepid][q] + 1]}}
                else:
                    raw_centroid_data[kepid][q] = {'x': data_mat[q][:, kepids_aux[kepid][q]],
                                                   'y': data_mat[q][:, kepids_aux[kepid][q] + 1]}

    # linearly interpolate across transits
    print('Linearly interpolating across the transits for channel {}...'.format(channel))
    # instantiate the gapped data matrix
    data_mat_gapped = {q: np.copy(data_mat[q]) for q in data_mat}
    for q in data_mat:

        for kepid in kepids_aux:

            # check if there is any data for the Kepler ID for this quarter
            if q not in kepids_aux[kepid]:
                continue

            # instantiate binary time-series for the Kepler ID
            binary_time_series = np.zeros(len(time_arr[kepid][q]), dtype='uint8')

            # get ephemeris of TCEs detected in the target star
            eph_tces = eph_tbl.loc[eph_tbl['target_id'] == kepid][['tce_period', 'tce_time0bk', 'tce_duration']]

            if len(eph_tces) == 0:  # no TCEs to be gapped
                continue

            # create binary time-series with 1's for in-transit cadences for all TCEs detected in the target star
            for tce_i, tce in eph_tces.iterrows():
                # get first transit time for the TCE
                first_transit_time = find_first_epoch_after_this_time(tce.tce_time0bk,
                                                                      tce.tce_period,
                                                                      time_arr[kepid][q][0])

                # create binary time-series for the TCE
                binary_time_series_tce = create_binary_time_series(time_arr[kepid][q],
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
                data_mat_gapped[q][:, kepids_aux[kepid][q]] = lininterp_transits([data_mat[q][:, kepids_aux[kepid][q]]],
                                                                                 [binary_time_series],
                                                                                 idxs_it,
                                                                                 start_idxs,
                                                                                 end_idxs)[0]
                # 'y'-coordinate
                data_mat_gapped[q][:, kepids_aux[kepid][q] + 1] = lininterp_transits([data_mat[q][:,
                                                                                      kepids_aux[kepid][q] + 1]],
                                                                                     [binary_time_series],
                                                                                     idxs_it,
                                                                                     start_idxs,
                                                                                     end_idxs)[0]

    # saving data
    print('Saving pre-denoising data for channel {}...'.format(channel))
    np.save(os.path.join(save_dir, 'ppdata', 'datamatrices_ch{}.npy'.format(channel)), data_mat)
    np.save(os.path.join(save_dir, 'ppdata', 'gappeddatamatrices_ch{}.npy'.format(channel)), data_mat_gapped)
    # np.save('{}ppdata/time_ch{}.npy'.format(save_dir, ch), time_mat)
    np.save(os.path.join(save_dir, 'ppdata', 'centraltendency_ch{}.npy'.format(channel)), centr_tend)
    np.save(os.path.join(save_dir, 'ppdata', 'centroidtimeseries_ch{}.npy'.format(channel)), raw_centroid_data)
    np.save(os.path.join(save_dir, 'ppdata', 'idxsnotnan_ch{}.npy'.format(channel)), idxs_nnan)
    np.save(os.path.join(save_dir, 'ppdata', 'kepids_ch{}.npy'.format(channel)), kepids_aux)


def pca_denoising(channel, num_singularvalues, save_dir, logger=None):
    """ Performs denoising on the centroid time-series for a given channel. The quarter design matrices are assumed to
    be mxn, with N centroid time-series each with M points; Each centroid time-series was previously centered.

    :param channel: int, channel
    :param num_singularvalues: int, number of singular values used when truncating the SVD matrices
    :param save_dir: str, root directory for saving the data generated
    :return:
    """

    # instantiate variables
    singular_values = {}  # {q: [] for q in range(1, NUM_QUARTERS + 1)}
    centroid_data = {}  # {q: [] for q in range(1, NUM_QUARTERS + 1)}

    # load data
    data_mat = np.load(os.path.join(save_dir, 'ppdata', 'datamatrices_ch{}.npy'.format(channel)),
                       allow_pickle=True).item()
    data_mat_gapped = np.load(os.path.join(save_dir, 'ppdata', 'gappeddatamatrices_ch{}.npy'.format(channel)),
                              allow_pickle=True).item()
    centr_tend = np.load(os.path.join(save_dir, 'ppdata', 'centraltendency_ch{}.npy'.format(channel)),
                         allow_pickle=True).item()
    kepids_aux = np.load(os.path.join(save_dir, 'ppdata', 'kepids_ch{}.npy'.format(channel)),
                         allow_pickle=True).item()

    print('Performing denoising for channel {}...'.format(channel))
    # preprocess the raw data
    for q in data_mat:

        if len(data_mat[q]) == 0:
            print('Empty data matrix before SVD')
            report_exclusion({'channel': channel, 'quarter': q}, 'No filename',
                             id_str='Empty raw data matrix before SVD',
                             savedir=os.path.join(save_dir, 'exclusion_logs'))
            continue

        # compute SVD for the gapped design matrix

        # Full SVD: A [mxn] = U [mxm] * S [mxn] * V^T [nxn]
        svd_comps = np.linalg.svd(data_mat_gapped[q])

        # get the singular values
        singular_values[q] = svd_comps[1]

        # TODO: implement criterion to choose number of components

        # print('Finding minimization coefficients for channel {} quarter {}...'.format(channel, q))

        # # robust LS
        # beta = np.zeros((num_singularvalues, data_mat[q].shape[1]), dtype='float')
        # for col_i in range(data_mat[q].shape[1]):
        #
        #     print('Robust LS for channel {} quarter - coefs {}/{}'.format(ch, q, col_i, data_mat[q].shape[1]))
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
        #                                     args=(data_mat[q][:, col_i],
        #                                           svd_comps[0][:, :num_singularvalues],
        #                                           lambda_reg,
        #                                           reg_func),
        #                                     loss=loss,
        #                                     method='trf')
        #
        #     beta[:, col_i] = result.x

        # OLS with L2 regularizer (Ridge)
        # print('Performing Ridge regularization for channel {} quarter {}...'.format(channel, q))
        # alpha = 10

        # clf_ridge = Ridge(alpha=alpha, solver='auto', fit_intercept=True)
        # # clf_ridge = Lasso(alpha=alpha, fit_intercept=True)
        # clf_ridge.fit(svd_comps[0][:, :num_singularvalues], data_mat_gapped[q])
        # beta = clf_ridge.coef_.T

        # clf_ridge = HuberRegressor(alpha=alpha, fit_intercept=True, epsilon=1.35)
        # beta = np.zeros((num_singularvalues, data_mat[q].shape[1]), dtype='float')
        # for col_i in range(data_mat[q].shape[1]):
        #     clf_ridge.fit(svd_comps[0][:, :num_singularvalues], data_mat[q][:, col_i])
        #     beta[:, col_i] = clf_ridge.coef_.T

        # print('Denoising design matrix for channel {} quarter {}...'.format(channel, q))

        # # Vanilla Truncated SVD: remove the components associated with the largest singular values
        # # A_tr [mxn] = U [mxk] * S [kxk] * V^T [kxn]
        # # A_new [mxn] = A [mxn] - A_tr [mxn]
        data_mat[q] -= np.dot(svd_comps[0][:, :num_singularvalues] *
                              svd_comps[1][:num_singularvalues],
                              svd_comps[2][:num_singularvalues, :])

        # # Optimized coefficients for SVD
        # # A_tr [mxn] = U [mxk] * beta [kxn]
        # # A_new [mxn] = A [mxn] - A_tr [mxn]
        # data_mat[q] -= np.dot(svd_comps[0][:, :num_singularvalues], beta)

        # add back the central tendency
        data_mat[q] += centr_tend[q]

        # get the preprocessed centroid time series for each target star
        for kepid in kepids_aux:

            if q in kepids_aux[kepid]:
                if kepid not in centroid_data:
                    centroid_data[kepid] = {q: {'x': data_mat[q][:, kepids_aux[kepid][q]],
                                                'y': data_mat[q][:, kepids_aux[kepid][q] + 1]}}
                else:
                    centroid_data[kepid][q] = {'x': data_mat[q][:, kepids_aux[kepid][q]],
                                               'y': data_mat[q][:, kepids_aux[kepid][q] + 1]}

    # save denoised data
    print('Saving preprocessed data for channel {}...'.format(channel))
    np.save(os.path.join(save_dir, 'pcadenoising', 'singularvalues_ch{}.npy'.format(channel)), singular_values)
    np.save(os.path.join(save_dir, 'pcadenoising', 'denoiseddatamatrices_ch{}.npy'.format(channel)), data_mat)
    np.save(os.path.join(save_dir, 'pcadenoising', 'denoisedcentroidtimeseries_ch{}.npy'.format(channel)),
            centroid_data)


def pca_denoising_channels(channels, num_singularvalues, kepids, eph_tbl, lc_data_dir, save_dir, logger=None):
    """ Perform PCA denoising on a set of channels sequentially.

    :param channels: list, channels
    :param num_singularvalues: int, number of singular values used when truncating the SVD matrices
    :param kepids: list, target list
    :param eph_tbl: pandas DataFrame, TCE ephemeris table
    :param lc_data_dir: str, root directory for the FITS files
    :param save_dir: str, root directory for saving the data generated
    :return:
    """

    logger_channel = logging.getLogger(name='log_channels{}-{}'.format(channels[0], channels[-1]))
    logger_handler = logging.FileHandler(filename=os.path.join(save_dir,
                                                               'channels{}-{}.log'.format(channels[0], channels[-1])),
                                         mode='a')
    logger_formatter = logging.Formatter('%(asctime)s - %(message)s')

    logger_channel.setLevel(logging.INFO)
    logger_handler.setFormatter(logger_formatter)
    logger_channel.addHandler(logger_handler)

    for channel in channels:
        logger_channel.info('--- Channel {} ---'.format(channel))
        # prepare data for denoising
        preprocess_centroidtimeseries(channel, kepids, eph_tbl, lc_data_dir, save_dir, logger_channel)
        # denoising the centroid time-series
        # pca_denoising(channel, num_singularvalues, save_dir, logger_channel)


if __name__ == '__main__':

    # NUM_QUARTERS = 17
    # 21 science modules on a grid [3,5,5,5,3]; each module has 4 CCDs (channels); total number of channels: 21*4 = 84
    # each CCD in a module has a different orientation, so if one wants to combine different channels one needs to
    # perform coordinate transformation so that they share the same frame of reference
    # NUM_CHANNELS = 84

    # root save directory
    root_save_dir = '/home/msaragoc/Projects/Kepler-TESS_exoplanet/Data/centroid_pca_denoising/Kepler/'
    # FITS files root directory
    lc_data_dir = '/data5/tess_project/Data/Kepler-Q1-Q17-DR25/pdc-tce-time-series-fits'

    # load TCE ephemeris table - needed to get the list of target stars and to gap the transits from the centroid
    # time-series
    tce_tbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17 DR25/'
                          'q1_q17_dr25_tce_2020.04.15_23.19.10_cumkoi_2020.02.21_shuffled_norm.csv')

    # get list of unique targets
    kepids = tce_tbl['target_id'].unique()
    # kepids = kepids[:]

    # define save directory for the current study
    study = 'test_allchannels'  # + datetime.datetime.now()
    # save_dir = os.path.join(root_save_dir, '{}_{}'.format(study, datetime.datetime.now()))
    save_dir = os.path.join(root_save_dir, study)

    # create folders for data
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'ppdata'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'pcadenoising'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'exclusion_logs'), exist_ok=True)

    num_singularvalues = 6  # number of top singular components to remove

    # channels to be processed
    channels = np.arange(1, 85)
    # channels = [5]

    n_procs = 15  # number of processes to span
    jobs = []

    print('Number of total targets = {}'.format(len(kepids)))
    print('Number of channels (per process) = {} (~{})'.format(len(channels), int(len(channels) / n_procs)))
    print('Number of processes = {}'.format(n_procs))

    logger_main = logging.getLogger(name='log_main')
    logger_handler = logging.FileHandler(filename=os.path.join(save_dir, 'main.log'),
                                         mode='a')
    logger_formatter = logging.Formatter('%(asctime)s - %(message)s')

    logger_main.setLevel(logging.INFO)
    logger_handler.setFormatter(logger_formatter)
    logger_main.addHandler(logger_handler)

    logger_main.info('Total number of target stars = {}'.format(len(kepids)))
    logger_main.info('Total number of channels = {}'.format(len(channels)))
    logger_main.info('Number of processes = {}'.format(n_procs))
    logger_main.info('Number of singular values removed = {}'.format(num_singularvalues))

    # distribute channels across the channels
    boundaries = [int(i) for i in np.linspace(0, len(channels), n_procs + 1)]

    # each process handles a subset of the channels
    for proc_i in range(n_procs):
        indices = [(boundaries[i], boundaries[i + 1]) for i in range(n_procs)][proc_i]
        channels_proc = channels[indices[0]:indices[1]]
        p = multiprocessing.Process(target=pca_denoising_channels, args=(channels_proc, num_singularvalues, kepids,
                                                                         tce_tbl, lc_data_dir, save_dir))
        jobs.append(p)
        p.start()

    map(lambda p: p.join(), jobs)
