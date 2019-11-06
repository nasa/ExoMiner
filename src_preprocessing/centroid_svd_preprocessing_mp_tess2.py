import numpy as np
import os
from astropy.io import fits
# import pandas as pd
from tensorflow import gfile
import multiprocessing
import itertools


def report_exclusion(fits_id, fits_filep, id_str, savedir, stderr=None):
    """ Creates txt file with information regarding the exclusion of the processing of the fits file.

    :param fits_id: tuple, (sector, camera, ticid) of the fits file being read
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
        myfile.write('Sector {}, Camera {}, TICID {}: {} | Error {}\n {}'.format(fits_id[0], fits_id[1], fits_id[2],
                                                                                 id_str,
                                                                                 (stderr, 'None')[stderr is None],
                                                                                 fits_filep))


def tess_svdcentroidprocessing(pair_seccam, lc_data_dir, save_dir, num_singularvalues=6):
    """ TESS SVD performed per CCD, separate design matrices for row and col coordinates (px).

    :param pair_seccam: list of tuples, sector and camera to be analyzed
    :param lc_data_dir: str, root directory for the FITS files
    :param save_dir: str, root directory for saving the data generated
    :param num_singularvalues: int, number of singular values used when truncating the SVD matrices
    :return:
    """

    for sector, camera in pair_seccam:

        print('##### SECTOR {} CAMERA {} ####'.format(sector, camera))

        # get sector directory path
        sector_dir = os.path.join(lc_data_dir, 'sector_{}'.format(sector))

        # get filepaths for the FITS files for that sector
        fits_filepaths = [os.path.join(sector_dir, el) for el in os.listdir(sector_dir) if '.fits' in el]

        data_mat = {ccd: {'x': [], 'y': []} for ccd in range(1, 5)}
        centr_tend = {ccd: {'x': [], 'y': []} for ccd in range(1, 5)}
        idxs_nnan = {ccd: {'x': [], 'y': []} for ccd in range(1, 5)}
        singular_values = {ccd: {'x': [], 'y': []} for ccd in range(1, 5)}
        ticids_aux = {ccd: [] for ccd in range(1, 5)}
        raw_centroid_data = {ccd: {} for ccd in range(1, 5)}
        centroid_data = {ccd: {} for ccd in range(1, 5)}

        for fits_file_i, fits_file in enumerate(fits_filepaths):

            print('Sector {} Camera {} - {} % read'.format(sector, camera, fits_file_i / len(fits_filepaths) * 100))

            fits_header = fits.getheader(fits_file)
            cmr_fits = fits_header['CAMERA']
            ccd_fits = fits_header['CCD']

            if cmr_fits != camera:
                continue

            ticid = fits_header['TICID']

            with fits.open(gfile.Open(fits_file, "rb")) as hdu_list:

                try:
                    centroid_x, centroid_y = hdu_list['LIGHTCURVE'].data.MOM_CENTR1, hdu_list[
                        'LIGHTCURVE'].data.MOM_CENTR2
                except Exception as e:
                    print('Error while reading centroid time series {}'.format(fits_file))
                    report_exclusion((sector, camera, ticid), fits_file,
                                     id_str='Error while reading centroid time series',
                                     savedir=save_dir + 'exclusion_logs/', stderr=e)
                    continue

            # check if centroid time series is not all NaNs
            # if np.any(~np.isfinite(centroid_x)):
            if len(np.nonzero(~np.isfinite(centroid_x))[0]) <= 2500 and \
                    len(np.nonzero(~np.isfinite(centroid_y))[0]) <= 2500:  # 1/8 of the whole time series

                data_mat[ccd_fits]['x'].append(centroid_x)
                data_mat[ccd_fits]['y'].append(centroid_y)

                ticids_aux[ccd_fits].append(ticid)

            else:
                print('Centroid data for {} is more than ~12.5% NaNs'.format(fits_file))
                report_exclusion((sector, camera, ticid), fits_file,
                                 id_str='Centroid data for is more than ~12.5% NaNs',
                                 savedir=save_dir + 'exclusion_logs/')

        # get the raw data into data matrices and prepare them to SVD
        for ccd in range(1, 5):

            if len(data_mat[ccd]['x']) == 0:
                continue

            data_mat[ccd] = {coord: np.array(data_mat[ccd][coord], dtype='float').T for coord in ['x', 'y']}
            print('Matrix shape (x, y): {}, {}'.format(data_mat[ccd]['x'].shape, data_mat[ccd]['y'].shape))

            # option 2 - remove indices for all target stars in if at least one target shows a nan value
            idxs_nnanx = np.nonzero(np.all(np.isfinite(data_mat[ccd]['x']), axis=1))
            idxs_nnany = np.nonzero(np.all(np.isfinite(data_mat[ccd]['y']), axis=1))
            idxs_nnan[ccd] = np.union1d(idxs_nnanx, idxs_nnany)
            data_mat[ccd] = {coord: data_mat[ccd][coord][idxs_nnan[ccd]] for coord in ['x', 'y']}
            print('Matrix shape after removing nans (x, y): {}, {}'.format(data_mat[ccd]['x'].shape,
                                                                           data_mat[ccd]['y'].shape))

            # get central tendency - median is more robust to outliers than mean
            # TODO: use a robust estimator of the mean
            centr_tend[ccd] = {coord: np.nanmedian(data_mat[ccd][coord], axis=0) for coord in ['x', 'y']}

            # remove the central tendency
            data_mat[ccd] = {coord: data_mat[ccd][coord] - centr_tend[ccd][coord] for coord in ['x', 'y']}

            # get the raw centroid time series for each target star
            for ticid_i, ticid in enumerate(ticids_aux[ccd]):
                raw_centroid_data[ccd][ticid] = {coord: data_mat[ccd][coord][:, ticid_i] for coord in ['x', 'y']}

        # saving raw data
        print('Saving raw data for sector {} camera {}...'.format(sector, camera))
        np.save('{}raw_data/rawdata_s{}_c{}.npy'.format(save_dir, sector, camera), data_mat)
        np.save('{}raw_data/rawdata_s{}_c{}_centraltendency.npy'.format(save_dir, sector, camera), centr_tend)
        np.save('{}raw_data/centroidtimeseries_s{}_c{}.npy'.format(save_dir, sector, camera), raw_centroid_data)
        np.save('{}raw_data/rawdata_s{}_c{}_idxsnan.npy'.format(save_dir, sector, camera), idxs_nnan)
        np.save('{}raw_data/ticids_s{}_c{}.npy'.format(save_dir, sector, camera), ticids_aux)
        # data_mat = np.load('{}raw_data/rawdata_ch{}.npy'.format(save_dir, ch)).item()
        # centr_tend = np.load('{}raw_data/rawdata_ch{}_centraltendency.npy'.format(save_dir, ch)).item()

        for ccd in range(1, 5):

            if len(data_mat[ccd]['x']) == 0:
                continue

            # Full SVD: A [mxn] = U [mxm] * S [mxn] * V^T [nxn]
            svd_comps = {coord: np.linalg.svd(data_mat[ccd][coord]) for coord in ['x', 'y']}

            # get the singular values
            singular_values[ccd] = {coord: svd_comps[coord][1] for coord in ['x', 'y']}

            # Truncated SVD: remove the components associated with the largest singular values
            # A_tr [mxn] = U [mxk] * S [kxk] * V^T [kxn]
            # A_new [mxn] = A [mxn] - A_tr [mxn]
            data_mat[ccd] = {coord: data_mat[ccd][coord] -
                               np.dot(svd_comps[coord][0][:, :num_singularvalues] *
                                      svd_comps[coord][1][:num_singularvalues],
                                      svd_comps[coord][2][:num_singularvalues, :])
                           for coord in ['x', 'y']}

            # add back the central tendency
            data_mat[ccd] = {coord: data_mat[ccd][coord] + centr_tend[ccd][coord] for coord in ['x', 'y']}

            # get the preprocessed centroid time series for each target star
            for ticid_i, ticid in enumerate(ticids_aux[ccd]):
                centroid_data[ccd][ticid] = {coord: data_mat[ccd][coord][:, ticid_i] for coord in ['x', 'y']}

        # save processed data
        print('Saving preprocessed data for sector {} camera {}...'.format(sector, camera))
        np.save('{}singular_values/singularvalues_s{}_c{}.npy'.format(save_dir, sector, camera), singular_values)
        np.save('{}svdpreproc_data/ppdata_s{}_c{}.npy'.format(save_dir, sector, camera), data_mat)
        np.save('{}svdpreproc_data/centroidtimeseries_s{}_c{}.npy'.format(save_dir, sector, camera), centroid_data)


if __name__ == '__main__':

    save_dir = '/home/msaragoc/Projects/Kepler-TESS_exoplanet/Data/centroid_svd_processing/TESS/'
    lc_data_dir = '/data5/tess_project/Data/TESS_TOI_fits(MAST)/'
    # ticid_tbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/'
    #                         'q1_q17_dr25_tce_2019.03.12_updt_tcert.csv')['tic']
    # ticids = ticid_tbl.unique()

    num_singularvalues = 6
    num_cameras = 4
    num_sectors = 1
    sectors = np.arange(1, num_sectors + 1)
    cameras = np.arange(1, num_cameras + 1)

    pair_seccam = list( itertools.product(sectors, cameras))

    n_procs = 4
    jobs = []

    # print('Number of total targets = {}'.format(len(ticids)))
    print('Number of sector-camera pair (per process) = {} (~{})'.format(len(pair_seccam),
                                                                         int(len(pair_seccam) / n_procs)))
    print('Number of processes = {}'.format(n_procs))

    boundaries = [int(i) for i in np.linspace(0, len(pair_seccam), n_procs + 1)]

    for proc_i in range(n_procs):
        indices = [(boundaries[i], boundaries[i + 1]) for i in range(n_procs)][proc_i]
        pair_seccam_proc = pair_seccam[indices[0]:indices[1]]
        p = multiprocessing.Process(target=tess_svdcentroidprocessing, args=(pair_seccam_proc, lc_data_dir, save_dir))
        jobs.append(p)
        p.start()

    map(lambda p: p.join(), jobs)
