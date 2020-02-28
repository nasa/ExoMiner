# 3rd party
import pandas as pd
import numpy as np
from astropy.io import fits
from tensorflow import gfile
import multiprocessing
import scipy.optimize as optimize
from sklearn.linear_model import Ridge, HuberRegressor, Lasso, BayesianRidge, ElasticNet, LassoLarsIC
import datetime
import os

# local
from src_preprocessing.light_curve import kepler_io
from src_preprocessing.utils_centroid_svd_preprocessing import minimizer, l2_reg, l1_reg, report_exclusion, _has_finite
from src_preprocessing.utils_ephemeris import create_binary_time_series, find_first_epoch_after_this_time, \
    lininterp_transits, get_startend_idxs_inter


def preprocess_centroidtimeseries(ch, eph_tbl, kepids, lc_data_dir, save_dir):
    """ Kepler SVD performed per CCD, separate design matrices for row and col coordinates (px).

    :param ch: list, channels to be processed
    :param eph_tbl: pandas DataFrame, TCE ephemeris table
    :param kepids:  list, Kepler IDs to be processed
    :param lc_data_dir: str, root directory for the FITS files
    :param save_dir: str, root directory for saving the data generated
    :return:
    """

    # initialize variables
    data_mat = {q: [] for q in range(1, NUM_QUARTERS + 1)}
    time_mat = {q: [] for q in range(1, NUM_QUARTERS + 1)}
    centr_tend = {q: [] for q in range(1, NUM_QUARTERS + 1)}
    idxs_nnan = {q: [] for q in range(1, NUM_QUARTERS + 1)}
    kepids_aux = {q: [] for q in range(1, NUM_QUARTERS + 1)}
    raw_centroid_data = {q: [] for q in range(1, NUM_QUARTERS + 1)}

    for kepid_i, kepid in enumerate(kepids):

        # print('Channel {}/{} | Kepler ID {} {}/{}'.format(ch, channels[-1], kepid, kepid_i, len(kepids)))

        # get fits filenames for the Kepler ID
        kepid_fits_filenames = kepler_io.kepler_filenames(lc_data_dir, kepid)

        for filename in kepid_fits_filenames:  # kepids_fits_filenames[kepid]:

            # get header of the fits file
            fits_header = fits.getheader(filename)

            if fits_header['CHANNEL'] != ch or fits_header['QUARTER'] == 0:
                continue

            with fits.open(gfile.Open(filename, "rb")) as hdu_list:

                # channel = hdu_list["PRIMARY"].header["CHANNEL"]
                quarter = hdu_list["PRIMARY"].header["QUARTER"]

                print('Kepler ID {} {}/{} | Channel {} | Quarter {}/{} - {} % read'.format(kepid, kepid_i,
                                                                                           len(kepids),
                                                                                           ch, quarter, 17,
                                                                                           kepid_i / len(kepids)
                                                                                           * 100))
                # print('Channel {} Quarter {}'.format(ch, quarter))

                # check if there is at least one finite value in both row and col centroid time-series
                # (prefer PSF to MOM if available)
                if _has_finite(hdu_list['LIGHTCURVE'].data.PSF_CENTR1) and \
                        _has_finite(hdu_list['LIGHTCURVE'].data.PSF_CENTR2):
                    centroid_x, centroid_y = hdu_list['LIGHTCURVE'].data.PSF_CENTR1, \
                                             hdu_list['LIGHTCURVE'].data.PSF_CENTR2
                else:
                    if _has_finite(hdu_list['LIGHTCURVE'].data.MOM_CENTR1) and \
                            _has_finite(hdu_list['LIGHTCURVE'].data.MOM_CENTR2):
                        centroid_x, centroid_y = hdu_list['LIGHTCURVE'].data.MOM_CENTR1, \
                                                 hdu_list['LIGHTCURVE'].data.MOM_CENTR2
                    else:
                        print('Centroid data for {} is all NaNs'.format(filename))
                        report_exclusion({'channel': ch, 'quarter': quarter, 'kepid': kepid}, filename,
                                         id_str='Centroid data is all NaNs',
                                         savedir=save_dir + 'exclusion_logs/')
                        continue  # no data

                data_mat[quarter].append(centroid_x)
                data_mat[quarter].append(centroid_y)
                time_mat[quarter].append(hdu_list['LIGHTCURVE'].data.time)

                kepids_aux[quarter].append(kepid)

    # convert to numpy array
    data_mat = {q: np.array(data_mat[q], dtype='float').T for q in data_mat}
    time_mat = {q: np.array(time_mat[q], dtype='float').T for q in time_mat}
    for q in time_mat:
        if not np.all([np.array_equal(time_mat[q][:, 0], time_mat[q][:, i]) for i in range(1, time_mat[q].shape[1])]):
            raise ValueError('the time arrays are not equal in quarter {} channel {}'.format(q, ch))

    # get the raw data into data matrices and prepare them to SVD
    print('Dealing with NaN values...')
    for q in data_mat:

        if len(data_mat[q]) == 0:
            print('Empty raw data matrix')
            report_exclusion({'channel': ch, 'quarter': q}, 'No filename',
                             id_str='Empty raw data matrix',
                             savedir=save_dir + 'exclusion_logs/')
            continue

        # print('Quarter {} ({})'.format(q, list(data_mat.keys())))
        #
        # data_mat[q] = np.array(data_mat[q], dtype='float').T
        print('Matrix shape: {}'.format(data_mat[q].shape))

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
        idxs_nnan[q] = np.logical_and(np.nonzero(np.all(np.isfinite(data_mat[q]), axis=1)),
                                      np.nonzero(np.all(np.isfinite(time_mat[q]), axis=1)))
        data_mat[q] = data_mat[q][idxs_nnan[q]]
        time_mat[q] = time_mat[q][idxs_nnan[q]]
        print('Matrix shape after removing nans: {}'.format(data_mat[q].shape))

    print('Removing central tendency...')
    for q in data_mat:

        # get central tendency - median is more robust to outliers than mean
        # TODO: use a robust estimator of the mean
        centr_tend[q] = np.nanmedian(data_mat[q], axis=0)

        # remove the central tendency
        data_mat[q] = data_mat[q] - centr_tend[q]

        # get the raw centroid time series for each target star
        for kepid_i, kepid in enumerate(kepids_aux[q]):
            # raw_centroid_data[q][kepid] = data_mat[q][:, kepid_i]
            raw_centroid_data[q][kepid] = data_mat[q][:, [2 * kepid_i, 2 * kepid_i + 1]]  # get both row and col

    print('Linearly interpolating across the transits...')
    data_mat_gapped = data_mat
    for q in data_mat:

        for kepid_i, kepid in enumerate(kepids_aux):

            # instantiate binary time-series for the Kepler ID
            binary_time_series = np.zeros(len(time_mat[q]), dtype='uint8')

            # get ephemeris of TCEs detected in the target star
            eph_tces = eph_tbl.loc[eph_tbl['target_id'] == kepid][['tce_period', 'tce_time0bk', 'tce_duration']]

            # create binary time-series with 1's for in-transit cadences for all TCEs detected in the target star
            for tce_i, tce in eph_tces.iterrows():
                # get first transit time for the TCE
                first_transit_time = find_first_epoch_after_this_time(tce.time0bk, tce.tce_period, time_mat[q][0])

                # create binary time-series for the TCE
                binary_time_series_tce = create_binary_time_series(time_mat[q], first_transit_time, tce.tce_duration,
                                                                   tce.tce_period)

                # flag in-transit cadences for the TCE
                binary_time_series[np.where(binary_time_series_tce == 1)] = 1

            # get indexes for linear interpolation across the in-transit cadences
            idxs_it, start_idxs, end_idxs = get_startend_idxs_inter(binary_time_series)

            # linear interpolation across the in-transit cadences for row and col time-series
            data_mat_gapped[q][:, 2 * kepid_i] = lininterp_transits(data_mat_gapped[q][:, 2 * kepid_i],
                                                                    binary_time_series,
                                                                    idxs_it,
                                                                    start_idxs,
                                                                    end_idxs)
            data_mat_gapped[q][:, 2 * kepid_i + 1] = lininterp_transits(data_mat_gapped[q][:, 2 * kepid_i],
                                                                        binary_time_series,
                                                                        idxs_it,
                                                                        start_idxs,
                                                                        end_idxs)

    # saving raw data
    print('Saving raw data for channel {}...'.format(ch))
    np.save('{}ppdata/ppdata_ch{}.npy'.format(save_dir, ch), data_mat)
    np.save('{}ppdata/gappeddata_ch{}.npy'.format(save_dir, ch), data_mat_gapped)
    np.save('{}ppdata/time_ch{}.npy'.format(save_dir, ch), time_mat)
    np.save('{}ppdata/ppdata_ch{}_centraltendency.npy'.format(save_dir, ch), centr_tend)
    np.save('{}ppdata/ppcentroidtimeseries_ch{}.npy'.format(save_dir, ch), raw_centroid_data)
    np.save('{}ppdata/ppdata_ch{}_idxsnan.npy'.format(save_dir, ch), idxs_nnan)
    np.save('{}ppdata/kepids_ch{}.npy'.format(save_dir, ch), kepids_aux)


def pca_denoising(ch, num_singularvalues, save_dir):
    """ Performs denoising on the centroid time-series. The design matrix is assumed to be mxn,
    with N centroid time-series each with M points; Each centroid time-series was previously centered.

    :param ch: int, channel
    :param num_singularvalues: int, number of singular values used when truncating the SVD matrices
    :param save_dir: str, root directory for saving the data generated
    :return:
    """

    singular_values = {q: [] for q in range(1, NUM_QUARTERS + 1)}
    centroid_data = {q: [] for q in range(1, NUM_QUARTERS + 1)}

    # load raw data
    data_mat = np.load('{}ppdata/ppdata_ch{}.npy'.format(save_dir, ch)).item()
    data_mat_gapped = np.load('{}ppdata/gappeddata_ch{}.npy'.format(save_dir, ch)).item()
    centr_tend = np.load('{}ppdata/ppdata_ch{}_centraltendency.npy'.format(save_dir, ch)).item()
    kepids_aux = np.load('{}ppdata/kepids_ch{}.npy'.format(save_dir, ch)).item()

    # preprocess the raw data
    for q in data_mat:

        if len(data_mat[q]) == 0:
            print('Empty data matrix before SVD')
            report_exclusion({'channel': ch, 'quarter': q}, 'No filename',
                             id_str='Empty raw data matrix before SVD',
                             savedir=save_dir + 'exclusion_logs/')
            continue

        print('Matrix shape: {}'.format(data_mat[q].shape))

        # compute SVD for the gapped desing matrix
        print('SVD for channel {} quarter {}...'.format(ch, q))
        # Full SVD: A [mxn] = U [mxm] * S [mxn] * V^T [nxn]
        svd_comps = np.linalg.svd(data_mat_gapped[q])

        # get the singular values
        singular_values[q] = svd_comps[1]

        # # TODO: implement criterion to choose number of components
        #
        # print('Finding minimization coefficients for channel {} quarter {}...'.format(ch, q))
        #
        # # # robust LS
        # # beta = np.zeros((num_singularvalues, data_mat[q].shape[1]), dtype='float')
        # # for col_i in range(data_mat[q].shape[1]):
        # #
        # #     print('Robust LS for channel {} quarter - coefs {}/{}'.format(ch, q, col_i, data_mat[q].shape[1]))
        # #
        # #     # TODO: choose a better initialization?
        # #     x0 = np.random.rand(num_singularvalues)
        # #
        # #     loss = 'linear'
        # #     lambda_reg = 0
        # #     reg_func = l2_reg
        # #
        # #     result = optimize.least_squares(minimizer,
        # #                                     x0,
        # #                                     args=(data_mat[q][:, col_i],
        # #                                           svd_comps[0][:, :num_singularvalues],
        # #                                           lambda_reg,
        # #                                           reg_func),
        # #                                     loss=loss,
        # #                                     method='trf')
        # #
        # #     beta[:, col_i] = result.x
        #
        # # OLS with L2 regularizer (Ridge)
        # print('Performing Ridge regularization for channel {} quarter {}...'.format(ch, q))
        # alpha = 10
        #
        # clf_ridge = Ridge(alpha=alpha, solver='auto', fit_intercept=True)
        # # clf_ridge = Lasso(alpha=alpha, fit_intercept=True)
        # clf_ridge.fit(svd_comps[0][:, :num_singularvalues], data_mat_gapped[q])
        # beta = clf_ridge.coef_.T
        #
        # # clf_ridge = HuberRegressor(alpha=alpha, fit_intercept=True, epsilon=1.35)
        # # beta = np.zeros((num_singularvalues, data_mat[q].shape[1]), dtype='float')
        # # for col_i in range(data_mat[q].shape[1]):
        # #     clf_ridge.fit(svd_comps[0][:, :num_singularvalues], data_mat[q][:, col_i])
        # #     beta[:, col_i] = clf_ridge.coef_.T
        #
        # print('Denoising design matrix for channel {} quarter {}...'.format(ch, q))
        #
        # # # Vanilla Truncated SVD: remove the components associated with the largest singular values
        # # # A_tr [mxn] = U [mxk] * S [kxk] * V^T [kxn]
        # # # A_new [mxn] = A [mxn] - A_tr [mxn]
        # # data_mat[q] = {coord: data_mat[q][coord] -
        # #                       np.dot(svd_comps[coord][0][:, :num_singularvalues] *
        # #                              svd_comps[coord][1][:num_singularvalues],
        # #                              svd_comps[coord][2][:num_singularvalues, :])
        # #                for coord in ['x', 'y']}
        #
        # # Optimized coefficients for SVD
        # # A_tr [mxn] = U [mxk] * beta [kxn]
        # # A_new [mxn] = A [mxn] - A_tr [mxn]
        # data_mat[q] = data_mat[q] - np.dot(svd_comps[0][:, :num_singularvalues], beta)
        #
        # # add back the central tendency
        # data_mat[q] = data_mat[q] + centr_tend[q]
        #
        # # get the preprocessed centroid time series for each target star
        # for kepid_i, kepid in enumerate(kepids_aux[q]):
        #     centroid_data[q][kepid] = data_mat[q][:, kepid_i]

    # save preprocessed data
    print('Saving preprocessed data for channel {}...'.format(ch))
    np.save('{}pcadenoising/singularvalues_ch{}.npy'.format(save_dir, ch), singular_values)
    # np.save('{}pcadenoising/denoiseddata_ch{}.npy'.format(save_dir, ch), data_mat)
    # np.save('{}pcadenoising/denoisedcentroidtimeseries_ch{}.npy'.format(save_dir, ch), centroid_data)


def pca_denoising_channels(channels, num_singularvalues, kepids, eph_tbl, lc_data_dir, save_dir):
    """ Perform PCA denoising on a set of channels sequentially.

    :param channels: list, channels
    :param num_singularvalues: int, number of singular values used when truncating the SVD matrices
    :param kepids: list, target list
    :param eph_tbl: pandas DataFrame, TCE ephemeris table
    :param lc_data_dir: str, root directory for the FITS files
    :param save_dir: str, root directory for saving the data generated
    :return:
    """

    for ch in channels:

        preprocess_centroidtimeseries(ch, kepids, eph_tbl, lc_data_dir, save_dir)
        pca_denoising(ch, num_singularvalues, save_dir)


if __name__ == '__main__':

    NUM_QUARTERS = 17

    save_dir = '/home/msaragoc/Projects/Kepler-TESS_exoplanet/Data/centroid_pca_denoising/Kepler/'
    lc_data_dir = '/data5/tess_project/Data/Kepler-Q1-Q17-DR25/pdc-tce-time-series-fits'

    # get list of unique targets
    kepid_tbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/final_tce_tables/'
                            'q1_q17_dr25_tce_2019.03.12_updt_tcert_extendedtceparams_updt_normstellarparamswitherrors_processed.csv')['target_id']

    kepids = kepid_tbl.unique()
    # kepids = kepids[:]

    save_dir = save_dir + '{}-'.format(datetime.datetime.now())

    os.makedirs(save_dir)
    os.makedirs(os.path.join(save_dir, 'ppdata'))
    os.makedirs(os.path.join(save_dir, 'pcadenoising'))

    num_singularvalues = 6

    # channels = np.arange(1, 85)
    channels = [5, 6, 7]

    n_procs = 1
    jobs = []

    print('Number of total targets = {}'.format(len(kepids)))
    print('Number of channels (per process) = {} (~{})'.format(len(channels), int(len(channels) / n_procs)))
    print('Number of processes = {}'.format(n_procs))

    boundaries = [int(i) for i in np.linspace(0, len(channels), n_procs + 1)]

    for proc_i in range(n_procs):
        indices = [(boundaries[i], boundaries[i + 1]) for i in range(n_procs)][proc_i]
        channels_proc = channels[indices[0]:indices[1]]
        p = multiprocessing.Process(target=pca_denoising_channels, args=(channels_proc, num_singularvalues, kepids,
                                                                         lc_data_dir, save_dir))
        jobs.append(p)
        p.start()

    map(lambda p: p.join(), jobs)
