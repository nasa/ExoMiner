import numpy as np
import os
from astropy.io import fits
# import pandas as pd
from tensorflow import gfile
import multiprocessing
import itertools


def report_exclusion(fits_id, fits_filep, id_str, savedir, stderr=None):
    """ Creates txt file with information regarding the exclusion of the processing of the fits file.

    :param fits_id: dict, ID description of the data related to the error
    :param fits_filep: str, filepath of the fits file being read
    :param id_str: str, contains info on the cause of exclusion
    :param savedir: str, filepath to directory in where the exclusion logs are saved
    :param stderr: str, error
    :return:
    """

    # # if is_pfe():
    #
    # # create path to exclusion logs directory
    # savedir = os.path.join(config.output_dir, 'exclusion_logs')
    # # create exclusion logs directory if it does not exist
    # os.makedirs(savedir, exist_ok=True)

    # if is_pfe():
    #
    #     # get node id
    #     node_id = socket.gethostbyname(socket.gethostname()).split('.')[-1]
    #
    #     # write to exclusion log pertaining to this process and node
    #     with open(os.path.join(savedir, 'exclusions_%d_%s.txt' % (config.process_i, node_id)), "a") as myfile:
    #         myfile.write('kepid: {}, tce_n: {}, {}\n{}'.format(tce.kepid, tce.tce_plnt_num, id_str,
    #                                                          (stderr, '')[stderr is None]))
    # else:
    #
    # write to exclusion log pertaining to this process and node
    with open(os.path.join(savedir, 'exclusions_{}.txt'.format(fits_id)), "a") as myfile:
        myfile.write('ID: {}: {} | Error {}\n {}'.format(fits_id, id_str, (stderr, 'None')[stderr is None],
                                                         fits_filep))


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

        data_mat = []
        # data_mat = {'x': [], 'y': []}
        # centr_tend = {'x': [], 'y': []}
        # idxs_nnan = {'x': [], 'y': []}
        # singular_values = {'x': [], 'y': []}
        ticids_aux = []
        raw_centroid_data = {}
        centroid_data = {}

        # iterate over the fits files
        for fits_file_i, fits_file in enumerate(fits_filepaths):

            print('Sector {} Camera {} CCD {} - {} % read'.format(sector, camera, ccd,
                                                                  fits_file_i / len(fits_filepaths) * 100))

            fits_header = fits.getheader(fits_file)
            cmr_fits = fits_header['CAMERA']
            ccd_fits = fits_header['CCD']

            if cmr_fits != camera or ccd_fits != ccd:
                continue

            ticid = fits_header['TICID']

            with fits.open(gfile.Open(fits_file, "rb")) as hdu_list:

                try:
                    centroid_x, centroid_y = hdu_list['LIGHTCURVE'].data.MOM_CENTR1, hdu_list[
                        'LIGHTCURVE'].data.MOM_CENTR2
                except Exception as e:
                    print('Error while reading centroid time series {}'.format(fits_file))
                    report_exclusion((sector, ticid), fits_file,
                                     id_str='Error while reading centroid time series',
                                     savedir=save_dir + 'exclusion_logs/', stderr=e)
                    continue

            # check if centroid time series is not all NaNs
            # if np.any(~np.isfinite(centroid_x)):
            if len(np.nonzero(~np.isfinite(centroid_x))[0]) <= 2500 and \
                    len(np.nonzero(~np.isfinite(centroid_y))[0]) <= 2500:  # 1/8 of the whole time series

                data_mat.append(centroid_x)
                data_mat.append(centroid_y)

                ticids_aux.append(ticid)

            else:
                print('Centroid data for {} is more than ~12.5% NaNs'.format(fits_file))
                report_exclusion({'sector': sector, 'ticid': ticid}, fits_file,
                                 id_str='Centroid data for is more than ~12.5% NaNs',
                                 savedir=save_dir + 'exclusion_logs/')

        # get the raw data into data matrices and prepare them to SVD

        if len(data_mat) == 0:
            print('Empty raw data matrix')
            report_exclusion({'sector': sector, 'camera': camera, 'ccd': ccd}, fits_file,
                             id_str='Empty raw data matrix',
                             savedir=save_dir + 'exclusion_logs/')
            continue

        data_mat = np.array(data_mat, dtype='float').T
        # print('Matrix shape (x, y): {}, {}'.format(data_mat['x'].shape, data_mat['y'].shape))
        print('Matrix shape: {}'.format(data_mat.shape))

        # option 2 - remove indices for all target stars in if at least one target shows a nan value
        idxs_nnanx = np.nonzero(np.all(np.isfinite(data_mat), axis=1))
        # idxs_nnany = np.nonzero(np.all(np.isfinite(data_mat['y']), axis=1))
        # idxs_nnan = np.union1d(idxs_nnanx, idxs_nnany)
        data_mat = data_mat[idxs_nnanx]
        print('Matrix shape after removing nans: {}'.format(data_mat.shape))

        # get central tendency - median is more robust to outliers than mean
        # TODO: use a robust estimator of the mean
        centr_tend = np.nanmedian(data_mat, axis=0)

        # remove the central tendency
        data_mat = data_mat - centr_tend

        # get the raw centroid time series for each target star
        for ticid_i, ticid in enumerate(ticids_aux):
            raw_centroid_data[ticid] = {'x': data_mat[:, 2 * ticid_i], 'y': data_mat[:, 2 * ticid_i + 1]}

        # saving raw data
        print('Saving raw data for sector {} camera {} ccd {}...'.format(sector, camera, ccd))
        np.save('{}raw_data/rawdata_s{}_c{}_ccd{}.npy'.format(save_dir, sector, camera, ccd), data_mat)
        np.save('{}raw_data/rawdata_s{}_c{}_ccd{}_centraltendency.npy'.format(save_dir, sector, camera,ccd), centr_tend)
        np.save('{}raw_data/centroidtimeseries_s{}_c{}_ccd{}.npy'.format(save_dir, sector, camera, ccd),
                raw_centroid_data)
        np.save('{}raw_data/rawdata_s{}_c{}_ccd{}_idxsnan.npy'.format(save_dir, sector, camera, ccd), idxs_nnanx)
        np.save('{}raw_data/ticids_s{}_c{}_ccd{}.npy'.format(save_dir, sector, camera, ccd), ticids_aux)
        # data_mat = np.load('{}raw_data/rawdata_s{}_c{}_ccd{}.npy'.format(save_dir, sector, camera, ccd)).item()
        # centr_tend = np.load('{}raw_data/rawdata_s{}_c{}_ccd{}_centraltendency.npy'.format(save_dir, sector, camera, ccd)).item()

        print('SVD for sector {} camera {} ccd {}...'.format(sector, camera, ccd))

        if len(data_mat) == 0:
            print('Empty data matrix before SVD')
            report_exclusion({'sector': sector, 'camera': camera, 'ccd': ccd}, fits_file,
                             id_str='Empty raw data matrix before SVD',
                             savedir=save_dir + 'exclusion_logs/')
            continue

        # Full SVD: A [mxn] = U [mxm] * S [mxn] * V^T [nxn]
        svd_comps = np.linalg.svd(data_mat)

        # get the singular values
        singular_values = svd_comps[1]

        # Truncated SVD: remove the components associated with the largest singular values
        # A_tr [mxn] = U [mxk] * S [kxk] * V^T [kxn]
        # A_new [mxn] = A [mxn] - A_tr [mxn]
        data_mat = data_mat - np.dot(svd_comps[0][:, :num_singularvalues] *
                                     svd_comps[1][:num_singularvalues],
                                     svd_comps[2][:num_singularvalues, :])

        # add back the central tendency
        data_mat = data_mat + centr_tend

        # get the preprocessed centroid time series for each target star
        for ticid_i, ticid in enumerate(ticids_aux):
            centroid_data[ticid] = {'x': data_mat[:, 2 * ticid_i], 'y': data_mat[:, 2 * ticid_i + 1]}

        # save processed data
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

    num_singularvalues = 6
    num_cameras = 4
    num_ccds = 4
    num_sectors = 1
    sectors = np.arange(1, num_sectors + 1)
    cameras = np.arange(1, num_cameras + 1)
    ccds = np.arange(1, num_ccds + 1)

    pair_seccamccd = list(itertools.product(sectors, cameras, ccds))

    n_procs = 16
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
                                                                             save_dir))
        jobs.append(p)
        p.start()

    map(lambda p: p.join(), jobs)
