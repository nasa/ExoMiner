import numpy as np
import os
from astropy.io import fits
# import pandas as pd
from tensorflow import gfile
import multiprocessing
import itertools
import scipy.optimize as optimize
from sklearn.linear_model import Ridge, Huber, Lasso, BayesianRidge, ElasticNet, LassoLarsIC

# local
from src_preprocessing.utils_centroid_svd_preprocessing import minimizer, l2_reg, l1_reg, report_exclusion


def tess_svdcentroidprocessing(pair_seccamccd, lc_data_dir, save_dir, num_singularvalues=6):
    """ TESS SVD performed per CCD, separate design matrices for row and col coordinates (px).

    :param pair_seccam: list of tuples, sector and camera to be analyzed
    :param lc_data_dir: str, root directory for the FITS files
    :param save_dir: str, root directory for saving the data generated
    :param num_singularvalues: int, number of singular values used when truncating the SVD matrices
    :return:
    """

    for sector, camera, ccd in pair_seccamccd:

        print('##### SECTOR {} CAMERA {} CCD {} ####'.format(sector, camera, ccd))

        # get sector directory path
        sector_dir = os.path.join(lc_data_dir, 'sector_{}'.format(sector))

        # get filepaths for the FITS files for that sector
        fits_filepaths = [os.path.join(sector_dir, el) for el in os.listdir(sector_dir) if '.fits' in el]

        # initialize variables
        data_mat = []
        # data_mat = {'x': [], 'y': []}
        # centr_tend = {'x': [], 'y': []}
        # idxs_nnan = {'x': [], 'y': []}
        # singular_values = {'x': [], 'y': []}
        ticids_aux = []
        raw_centroid_data = {}
        centroid_data = {}

        # # iterate over the fits files
        # for fits_file_i, fits_file in enumerate(fits_filepaths):
        #
        #     print('Sector {} Camera {} CCD {} - {} % read'.format(sector, camera, ccd,
        #                                                           fits_file_i / len(fits_filepaths) * 100))
        #
        #     fits_header = fits.getheader(fits_file)
        #     cmr_fits = fits_header['CAMERA']
        #     ccd_fits = fits_header['CCD']
        #
        #     if cmr_fits != camera or ccd_fits != ccd:
        #         continue
        #
        #     ticid = fits_header['TICID']
        #
        #     with fits.open(gfile.Open(fits_file, "rb")) as hdu_list:
        #
        #         try:
        #             centroid_x, centroid_y = hdu_list['LIGHTCURVE'].data.MOM_CENTR1, hdu_list[
        #                 'LIGHTCURVE'].data.MOM_CENTR2
        #         except Exception as e:
        #             print('Error while reading centroid time series {}'.format(fits_file))
        #             report_exclusion((sector, ticid), fits_file,
        #                              id_str='Error while reading centroid time series',
        #                              savedir=save_dir + 'exclusion_logs/', stderr=e)
        #             continue
        #
        #     # check if centroid time series is not all NaNs
        #     # if np.any(~np.isfinite(centroid_x)):
        #     if len(np.nonzero(~np.isfinite(centroid_x))[0]) <= 2500 and \
        #             len(np.nonzero(~np.isfinite(centroid_y))[0]) <= 2500:  # 1/8 of the whole time series
        #
        #         data_mat.append(centroid_x)
        #         data_mat.append(centroid_y)
        #
        #         ticids_aux.append(ticid)
        #
        #     else:
        #         print('Centroid data for {} is more than ~12.5% NaNs'.format(fits_file))
        #         report_exclusion({'sector': sector, 'ticid': ticid}, fits_file,
        #                          id_str='Centroid data for is more than ~12.5% NaNs',
        #                          savedir=save_dir + 'exclusion_logs/')
        #
        # # get the raw data into data matrices and prepare them to SVD
        #
        # if len(data_mat) == 0:
        #     print('Empty raw data matrix')
        #     report_exclusion({'sector': sector, 'camera': camera, 'ccd': ccd}, 'No filename',
        #                      id_str='Empty raw data matrix',
        #                      savedir=save_dir + 'exclusion_logs/')
        #     continue
        #
        # data_mat = np.array(data_mat, dtype='float').T
        # # print('Matrix shape (x, y): {}, {}'.format(data_mat['x'].shape, data_mat['y'].shape))
        # print('Matrix shape: {}'.format(data_mat.shape))
        #
        # # option 2 - remove indices for all target stars in if at least one target shows a nan value
        # idxs_nnanx = np.nonzero(np.all(np.isfinite(data_mat), axis=1))
        # # idxs_nnany = np.nonzero(np.all(np.isfinite(data_mat['y']), axis=1))
        # # idxs_nnan = np.union1d(idxs_nnanx, idxs_nnany)
        # data_mat = data_mat[idxs_nnanx]
        # print('Matrix shape after removing nans: {}'.format(data_mat.shape))
        #
        # # get central tendency - median is more robust to outliers than mean
        # # TODO: use a robust estimator of the mean
        # centr_tend = np.nanmedian(data_mat, axis=0)
        #
        # # remove the central tendency
        # data_mat = data_mat - centr_tend
        #
        # # get the raw centroid time series for each target star
        # for ticid_i, ticid in enumerate(ticids_aux):
        #     raw_centroid_data[ticid] = {'x': data_mat[:, 2 * ticid_i], 'y': data_mat[:, 2 * ticid_i + 1]}
        #
        # # saving raw data
        # print('Saving raw data for sector {} camera {} ccd {}...'.format(sector, camera, ccd))
        # np.save('{}raw_data/rawdata_s{}_c{}_ccd{}.npy'.format(save_dir, sector, camera, ccd), data_mat)
        # np.save('{}raw_data/rawdata_s{}_c{}_ccd{}_centraltendency.npy'.format(save_dir, sector, camera,ccd), centr_tend)
        # np.save('{}raw_data/centroidtimeseries_s{}_c{}_ccd{}.npy'.format(save_dir, sector, camera, ccd),
        #         raw_centroid_data)
        # np.save('{}raw_data/rawdata_s{}_c{}_ccd{}_idxsnan.npy'.format(save_dir, sector, camera, ccd), idxs_nnanx)
        # np.save('{}raw_data/ticids_s{}_c{}_ccd{}.npy'.format(save_dir, sector, camera, ccd), ticids_aux)

        # load raw data
        data_mat = np.load('{}raw_data/per_ccd_jointcoordinates/'
                           'rawdata_s{}_c{}_ccd{}.npy'.format(save_dir, sector, camera, ccd))
        centr_tend = np.load('{}raw_data/per_ccd_jointcoordinates/'
                             'rawdata_s{}_c{}_ccd{}_centraltendency.npy'.format(save_dir, sector, camera, ccd))
        ticids_aux = np.load('{}raw_data/per_ccd_jointcoordinates/'
                             'ticids_s{}_c{}_ccd{}.npy'.format(save_dir, sector, camera, ccd))

        if len(data_mat) == 0:
            print('Empty data matrix before SVD')
            report_exclusion({'sector': sector, 'camera': camera, 'ccd': ccd}, 'No filename',
                             id_str='Empty raw data matrix before SVD',
                             savedir=save_dir + 'exclusion_logs/')
            continue

        print('Matrix shape (x, y): {}, {}'.format(data_mat[q]['x'].shape, data_mat[q]['y'].shape))

        print('SVD for sector {} camera {} ccd {}...'.format(sector, camera, ccd))
        # Full SVD: A [mxn] = U [mxm] * S [mxn] * V^T [nxn]
        svd_comps = np.linalg.svd(data_mat)

        # get the singular values
        singular_values = svd_comps[1]

        # TODO: implement criterion to choose number of components

        print('Finding minimization coefficients for sector {} camera {} ccd {}...'.format(sector, camera, ccd))
        # # robust LS
        # beta = np.zeros((num_singularvalues, data_mat.shape[1]), dtype='float')
        # for col_i in range(data_mat.shape[1]):
        #
        #     print('Robust LS for sector {} camera {} ccd {} - coefs {}/{}'.format(sector, camera, ccd, col_i,
        #                                                                           data_mat.shape[1]))
        #
        #     x0 = np.random.rand(num_singularvalues)
        #
        #     loss = 'linear'
        #     lambda_reg = 0
        #     reg_func = l2_reg
        #
        #     result = optimize.least_squares(minimizer,
        #                                     x0,
        #                                     args=(data_mat[:, col_i],
        #                                           svd_comps[0][:, :num_singularvalues],
        #                                           lambda_reg,
        #                                           reg_func),
        #                                     loss=loss,
        #                                     method='trf')
        #
        #     beta[:, col_i] = result.x

        # # OLS with L2 regularizer (Ridge)
        # print('Performing Ridge regularization for sector {} camera {} ccd {}...'.format(sector, camera, ccd))
        # alpha = 0.001
        # clf_ridge = Ridge(alpha=alpha, solver='auto', fit_intercept=True)
        # clf_ridge.fit(svd_comps[0][:, :num_singularvalues], data_mat)
        # beta = clf_ridge.coef_.T

        print('Denoising design matrix for sector {} camera {} ccd {}...'.format(sector, camera, ccd))
        # Truncated SVD: remove the components associated with the largest singular values
        # A_tr [mxn] = U [mxk] * S [kxk] * V^T [kxn]
        # A_new [mxn] = A [mxn] - A_tr [mxn]
        data_mat = data_mat - np.dot(svd_comps[0][:, :num_singularvalues] *
                                     svd_comps[1][:num_singularvalues],
                                     svd_comps[2][:num_singularvalues, :])
        # # A_tr [mxn] = U [mxk] * beta [kxn]
        # # A_new [mxn] = A [mxn] - A_tr [mxn]
        # data_mat = data_mat - np.dot(svd_comps[0][:, :num_singularvalues], beta)

        # add back the central tendency
        data_mat = data_mat + centr_tend

        # get the preprocessed centroid time series for each target star
        for ticid_i, ticid in enumerate(ticids_aux):
            centroid_data[ticid] = {'x': data_mat[:, 2 * ticid_i], 'y': data_mat[:, 2 * ticid_i + 1]}

        # save preprocessed data
        print('Saving preprocessed data for sector {} camera {} ccd {}...'.format(sector, camera, ccd))
        np.save('{}singular_values/singularvalues_s{}_c{}_ccd{}.npy'.format(save_dir, sector, camera, ccd),
                singular_values)
        np.save('{}svdpreproc_data/ppdata_s{}_c{}_ccd{}.npy'.format(save_dir, sector, camera, ccd), data_mat)
        np.save('{}svdpreproc_data/centroidtimeseries_s{}_c{}_ccd{}.npy'.format(save_dir, sector, camera, ccd),
                centroid_data)


if __name__ == '__main__':

    save_dir = '/home/msaragoc/Projects/Kepler-TESS_exoplanet/Data/centroid_svd_processing/TESS/'
    lc_data_dir = '/data5/tess_project/Data/TESS_TOI_fits(MAST)/'
    # ticid_tbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/'
    #                         'q1_q17_dr25_tce_2019.03.12_updt_tcert.csv')['tic']
    # ticids = ticid_tbl.unique()

    num_singularvalues = 100
    # loss = 'linear'  # 'linear', 'soft_l1', 'huber', 'cauchy', 'arctan'
    # lambda_reg = 1
    # reg_func = l2_reg  # l2_reg, l1_reg

    num_cameras = 1
    num_ccds = 4
    num_sectors = 1
    sectors = np.arange(1, num_sectors + 1)
    cameras = np.arange(1, num_cameras + 1)
    ccds = np.arange(1, num_ccds + 1)

    pair_seccamccd = list(itertools.product(sectors, cameras, ccds))

    n_procs = 4  # len(pair_seccamccd)
    jobs = []

    # print('Number of total targets = {}'.format(len(ticids)))
    print('Number of sector-camera-ccd groups (per process) = {} (~{})'.format(len(pair_seccamccd),
                                                                         int(len(pair_seccamccd) / n_procs)))
    print('Number of processes = {}'.format(n_procs))

    boundaries = [int(i) for i in np.linspace(0, len(pair_seccamccd), n_procs + 1)]

    for proc_i in range(n_procs):
        indices = [(boundaries[i], boundaries[i + 1]) for i in range(n_procs)][proc_i]
        pair_seccamccd_proc = pair_seccamccd[indices[0]:indices[1]]
        print('Group (sector, camera, CCD): {}'.format(pair_seccamccd_proc))
        p = multiprocessing.Process(target=tess_svdcentroidprocessing, args=(pair_seccamccd_proc, lc_data_dir,
                                                                             save_dir, num_singularvalues))
        jobs.append(p)
        p.start()

    map(lambda p: p.join(), jobs)
