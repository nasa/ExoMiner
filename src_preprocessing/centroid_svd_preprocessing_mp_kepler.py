# 3rd party
import pandas as pd
import numpy as np
from astropy.io import fits
from tensorflow import gfile
import multiprocessing
import scipy.optimize as optimize
from sklearn.linear_model import Ridge, HuberRegressor, Lasso, BayesianRidge, ElasticNet, LassoLarsIC

# local
from src_preprocessing.light_curve import kepler_io
from src_preprocessing.utils_centroid_svd_preprocessing import minimizer, l2_reg, l1_reg, report_exclusion


def kepler_svdcentroidprocessing(channels, lc_data_dir, kepids, save_dir, num_singularvalues=6):
    """ Kepler SVD performed per CCD, separate design matrices for row and col coordinates (px).

    :param channels: list, channels to be processed
    :param lc_data_dir: str, root directory for the FITS files
    :param kepids:  list, Kepler IDs to be processed
    :param save_dir: str, root directory for saving the data generated
    :param num_singularvalues: int, number of singular values used when truncating the SVD matrices
    :return:
    """

    NUM_QUARTERS = 17

    for ch in channels:

        print('##### CHANNEL {} ####'.format(ch))

        # initialize variables
        data_mat = {q: {'x': [], 'y': []} for q in range(1, NUM_QUARTERS + 1)}
        centr_tend = {q: {'x': [], 'y': []} for q in range(1, NUM_QUARTERS + 1)}
        idxs_nnan = {q: [] for q in range(1, NUM_QUARTERS + 1)}
        singular_values = {q: {'x': [], 'y': []} for q in range(1, NUM_QUARTERS + 1)}
        kepids_aux = {q: [] for q in range(1, NUM_QUARTERS + 1)}
        raw_centroid_data = {q: {} for q in range(1, NUM_QUARTERS + 1)}
        centroid_data = {q: {} for q in range(1, NUM_QUARTERS + 1)}

        # for kepid_i, kepid in enumerate(kepids):
        #
        #     # print('Channel {}/{} | Kepler ID {} {}/{}'.format(ch, channels[-1], kepid, kepid_i, len(kepids)))
        #
        #     # get fits filenames for the Kepler ID
        #     kepid_fits_filenames = kepler_io.kepler_filenames(lc_data_dir, kepid)
        #
        #     for filename in kepid_fits_filenames:  # kepids_fits_filenames[kepid]:
        #
        #         # get header of the fits file
        #         fits_header = fits.getheader(filename)
        #
        #         if fits_header['CHANNEL'] != ch or fits_header['QUARTER'] == 0:
        #             continue
        #
        #         with fits.open(gfile.Open(filename, "rb")) as hdu_list:
        #
        #             # channel = hdu_list["PRIMARY"].header["CHANNEL"]
        #             quarter = hdu_list["PRIMARY"].header["QUARTER"]
        #
        #             # # TODO: what to do with quarter 0?
        #             # if quarter == 0:
        #             #     continue
        #
        #             print('Kepler ID {} {}/{} | Channel {} | Quarter {}/{} - {} % read'.format(kepid, kepid_i,
        #                                                                                        len(kepids),
        #                                                                                        ch, quarter, 17,
        #                                                                                        kepid_i / len(kepids)
        #                                                                                        * 100))
        #             # print('Channel {} Quarter {}'.format(ch, quarter))
        #
        #             centroid_x, centroid_y = hdu_list['LIGHTCURVE'].data.MOM_CENTR1, hdu_list[
        #                 'LIGHTCURVE'].data.MOM_CENTR2
        #
        #             # if _has_finite(light_curve.PSF_CENTR1):
        #             #     centroid_x, centroid_y = light_curve.PSF_CENTR1, light_curve.PSF_CENTR2
        #             # else:
        #             #     if _has_finite(light_curve.MOM_CENTR1):
        #             #         centroid_x, centroid_y = light_curve.MOM_CENTR1, light_curve.MOM_CENTR2
        #             #     else:
        #             #         continue  # no data
        #
        #         # check if centroid time series is not all NaNs
        #         if np.any(~np.isfinite(centroid_x)):
        #
        #             # centroid_x, centroid_y = light_curve.MOM_CENTR1, light_curve.MOM_CENTR2
        #
        #             # print(len(np.nonzero(np.isnan(centroid_x))[0]), len(np.nonzero(np.isnan(centroid_y))[0]))
        #             # print(len(centroid_x), len(centroid_y))
        #
        #             # # option 1 - assume centroid time series noise is gaussian, use a robust estimator of the std, and
        #             # # impute it into the time series
        #             # TODO: compute the median in a small window around the NaN values
        #             # med_centroidx = np.median(centroid_x)
        #             # med_centroidy = np.median(centroid_y)
        #             # std_rob_estmx = np.median(np.abs(centroid_x - med_centroidx)) * 1.4826
        #             # std_rob_estmy = np.median(np.abs(centroid_y - med_centroidy)) * 1.4826
        #             # centroid_xnan = np.isnan(centroid_x)
        #             # centroid_ynan = np.isnan(centroid_y)
        #             # centroid_x[centroid_xnan] = med_centroidx - \
        #             #                             np.random.normal(0, std_rob_estmx, np.nonzero(centroid_xnan)[0].shape)
        #             # centroid_y[centroid_ynan] = med_centroidy - \
        #             #                             np.random.normal(0, std_rob_estmy, np.nonzero(centroid_ynan)[0].shape)
        #
        #             data_mat[quarter]['x'].append(centroid_x)
        #             data_mat[quarter]['y'].append(centroid_y)
        #
        #             kepids_aux[quarter].append(kepid)
        #
        #         else:
        #             print('Centroid data for {} is all NaNs'.format(filename))
        #             report_exclusion({'channel': ch, 'quarter': quarter, 'kepid': kepid}, filename,
        #                              id_str='Centroid data is all NaNs',
        #                              savedir=save_dir + 'exclusion_logs/')
        #
        # # get the raw data into data matrices and prepare them to SVD
        # for q in data_mat:
        #
        #     if len(data_mat[q]['x']) == 0:
        #         print('Empty raw data matrix')
        #         report_exclusion({'channel': ch, 'quarter': q}, 'No filename',
        #                          id_str='Empty raw data matrix',
        #                          savedir=save_dir + 'exclusion_logs/')
        #         continue
        #
        #     print('Quarter {} ({})'.format(q, list(data_mat.keys())))
        #
        #     data_mat[q] = {coord: np.array(data_mat[q][coord], dtype='float').T for coord in ['x', 'y']}
        #     print('Matrix shape (x, y): {}, {}'.format(data_mat[q]['x'].shape, data_mat[q]['y'].shape))
        #
        #     # option 2 - remove indices for all target stars if at least one target shows a nan value
        #     idxs_nnanx = np.nonzero(np.all(np.isfinite(data_mat[q]['x']), axis=1))
        #     idxs_nnany = np.nonzero(np.all(np.isfinite(data_mat[q]['y']), axis=1))
        #     idxs_nnan[q] = np.union1d(idxs_nnanx, idxs_nnany)
        #     data_mat[q] = {coord: data_mat[q][coord][idxs_nnan[q]] for coord in ['x', 'y']}
        #     print('Matrix shape after removing nans (x, y): {}, {}'.format(data_mat[q]['x'].shape,
        #                                                                    data_mat[q]['y'].shape))
        #
        #     # get central tendency - median is more robust to outliers than mean
        #     # TODO: use a robust estimator of the mean
        #     centr_tend[q] = {coord: np.nanmedian(data_mat[q][coord], axis=0) for coord in ['x', 'y']}
        #
        #     # remove the central tendency
        #     data_mat[q] = {coord: data_mat[q][coord] - centr_tend[q][coord] for coord in ['x', 'y']}
        #
        #     # get the raw centroid time series for each target star
        #     for kepid_i, kepid in enumerate(kepids_aux[q]):
        #         raw_centroid_data[q][kepid] = {coord: data_mat[q][coord][:, kepid_i] for coord in ['x', 'y']}
        #
        # # saving raw data
        # print('Saving raw data for channel {}...'.format(ch))
        # np.save('{}raw_data/rawdata_ch{}.npy'.format(save_dir, ch), data_mat)
        # np.save('{}raw_data/rawdata_ch{}_centraltendency.npy'.format(save_dir, ch), centr_tend)
        # np.save('{}raw_data/centroidtimeseries_ch{}.npy'.format(save_dir, ch), raw_centroid_data)
        # np.save('{}raw_data/rawdata_ch{}_idxsnan.npy'.format(save_dir, ch), idxs_nnan)
        # np.save('{}raw_data/kepids_ch{}.npy'.format(save_dir, ch), kepids_aux)

        # load raw data
        data_mat = np.load('{}raw_data/rawdata_ch{}.npy'.format(save_dir, ch)).item()
        centr_tend = np.load('{}raw_data/rawdata_ch{}_centraltendency.npy'.format(save_dir, ch)).item()
        kepids_aux = np.load('{}raw_data/kepids_ch{}.npy'.format(save_dir, ch)).item()

        # preprocess the raw data
        for q in data_mat:

            if len(data_mat[q]['x']) == 0:
                print('Empty data matrix before SVD')
                report_exclusion({'channel': ch, 'quarter': q}, 'No filename',
                                 id_str='Empty raw data matrix before SVD',
                                 savedir=save_dir + 'exclusion_logs/')
                continue

            print('Matrix shape (x, y): {}, {}'.format(data_mat[q]['x'].shape, data_mat[q]['y'].shape))

            print('SVD for channel {} quarter {}...'.format(ch, q))
            # Full SVD: A [mxn] = U [mxm] * S [mxn] * V^T [nxn]
            svd_comps = {coord: np.linalg.svd(data_mat[q][coord]) for coord in ['x', 'y']}

            # get the singular values
            singular_values[q] = {coord: svd_comps[coord][1] for coord in ['x', 'y']}

            # TODO: implement criterion to choose number of components

            print('Finding minimization coefficients for channel {} quarter {}...'.format(ch, q))

            # # robust LS
            # beta = {coord: np.zeros((num_singularvalues, data_mat[q][coord].shape[1]), dtype='float')
            #         for coord in ['x', 'y']}
            # for col_i in range(data_mat[q]['x'].shape[1]):
            #
            #     print('Robust LS for channel {} quarter - coefs {}/{}'.format(ch, q, col_i, data_mat[q]['x'].shape[1]))
            #
            #     # TODO: choose a better initialization?
            #     x0 = np.random.rand(num_singularvalues)
            #
            #     loss = 'linear'
            #     lambda_reg = 0
            #     reg_func = l2_reg
            #
            #     # x coordinate
            #     result = optimize.least_squares(minimizer,
            #                                     x0,
            #                                     args=(data_mat[q][:, col_i]['x'],
            #                                           svd_comps['x'][0][:, :num_singularvalues],
            #                                           lambda_reg,
            #                                           reg_func),
            #                                     loss=loss,
            #                                     method='trf')
            #
            #     beta['x'][:, col_i] = result.x
            #
            #     # y coordinate
            #     result = optimize.least_squares(minimizer,
            #                                     x0,
            #                                     args=(data_mat[q][:, col_i]['y'],
            #                                           svd_comps['y'][0][:, :num_singularvalues],
            #                                           lambda_reg,
            #                                           reg_func),
            #                                     loss=loss,
            #                                     method='trf')
            #
            #     beta['y'][:, col_i] = result.x

            # OLS with L2 regularizer (Ridge)
            print('Performing Ridge regularization for channel {} quarter {}...'.format(ch, q))
            alpha = 10
            # beta = {}
            beta = {coord: np.zeros((num_singularvalues, data_mat[q][coord].shape[1]), dtype='float')
                    for coord in ['x', 'y']}
            for coord in ['x', 'y']:
                # clf_ridge = Ridge(alpha=alpha, solver='auto', fit_intercept=True)
                # clf_ridge = Lasso(alpha=alpha, fit_intercept=True)
                # clf_ridge.fit(svd_comps[coord][0][:, :num_singularvalues], data_mat[q][coord])
                # beta[coord] = clf_ridge.coef_.T

                for col_i in range(data_mat[q][coord].shape[1]):
                    clf_ridge = HuberRegressor(alpha=alpha, fit_intercept=True, epsilon=1.35)
                    clf_ridge.fit(svd_comps[coord][0][:, :num_singularvalues], data_mat[q][coord][:, col_i])
                    beta[coord][:, col_i] = clf_ridge.coef_.T

            print('Denoising design matrix for channel {} quarter {}...'.format(ch, q))

            # # Vanilla Truncated SVD: remove the components associated with the largest singular values
            # # A_tr [mxn] = U [mxk] * S [kxk] * V^T [kxn]
            # # A_new [mxn] = A [mxn] - A_tr [mxn]
            # data_mat[q] = {coord: data_mat[q][coord] -
            #                       np.dot(svd_comps[coord][0][:, :num_singularvalues] *
            #                              svd_comps[coord][1][:num_singularvalues],
            #                              svd_comps[coord][2][:num_singularvalues, :])
            #                for coord in ['x', 'y']}

            # Optimized coefficients for SVD
            # A_tr [mxn] = U [mxk] * beta [kxn]
            # A_new [mxn] = A [mxn] - A_tr [mxn]
            data_mat[q] = {coord: data_mat[q][coord] - np.dot(svd_comps[coord][0][:, :num_singularvalues], beta[coord])
                           for coord in ['x', 'y']}

            # add back the central tendency
            data_mat[q] = {coord: data_mat[q][coord] + centr_tend[q][coord] for coord in ['x', 'y']}

            # get the preprocessed centroid time series for each target star
            for kepid_i, kepid in enumerate(kepids_aux[q]):
                centroid_data[q][kepid] = {coord: data_mat[q][coord][:, kepid_i] for coord in ['x', 'y']}

        # save preprocessed data
        print('Saving preprocessed data for channel {}...'.format(ch))
        np.save('{}singular_values/singularvalues_ch{}.npy'.format(save_dir, ch), singular_values)
        np.save('{}svdpreproc_data/ppdata_ch{}.npy'.format(save_dir, ch), data_mat)
        np.save('{}svdpreproc_data/centroidtimeseries_ch{}.npy'.format(save_dir, ch), centroid_data)


if __name__ == '__main__':

    # save_dir = '/home/msaragoc/Projects/Kepler-TESS_exoplanet/Data/centroid_svd_processing/Kepler/'
    save_dir = '/home/msaragoc/Projects/Kepler-TESS_exoplanet/Data/centroid_pca_denoising/Kepler/'
    lc_data_dir = '/data5/tess_project/Data/Kepler-Q1-Q17-DR25/pdc-tce-time-series-fits'

    # get list of unique targets
    # kepid_tbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Stellar parameters/Kepler/'
    #                         'q1_q17_dr25_stellar_gaiadr2.csv')['kepid']

    kepids = kepid_tbl.unique()

    num_singularvalues = 6

    channels = np.arange(1, 85)

    # kepids = kepids[:]
    channels = np.arange(1, 2)
    # get_chdatamatrix(1, lc_data_dir, kepids, save_dir)

    n_procs = 1
    jobs = []

    print('Number of total targets = {}'.format(len(kepids)))
    print('Number of channels (per process) = {} (~{})'.format(len(channels), int(len(channels) / n_procs)))
    print('Number of processes = {}'.format(n_procs))

    boundaries = [int(i) for i in np.linspace(0, len(channels), n_procs + 1)]

    for proc_i in range(n_procs):
        indices = [(boundaries[i], boundaries[i + 1]) for i in range(n_procs)][proc_i]
        channels_proc = channels[indices[0]:indices[1]]
        p = multiprocessing.Process(target=kepler_svdcentroidprocessing, args=(channels_proc, lc_data_dir, kepids,
                                                                                save_dir, num_singularvalues))
        jobs.append(p)
        p.start()

    map(lambda p: p.join(), jobs)
