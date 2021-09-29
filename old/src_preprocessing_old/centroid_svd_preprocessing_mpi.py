from mpi4py import MPI
import pandas as pd
import numpy as np
from astropy.io import fits
from tensorflow import gfile
import sys

from src_preprocessing.light_curve import kepler_io


def get_chdatamatrix(channels, lc_data_dir, kepids, save_dir):
    for ch in channels:

        print('##### CHANNEL {} ####'.format(ch))
        data_mat = {q: {'x': [], 'y': []} for q in range(1, 18)}
        centr_tend = {q: {'x': [], 'y': []} for q in range(1, 18)}
        # idxs_nan = {q: {'x': [], 'y': []} for q in range(1, 18)}

        for kepid_i, kepid in enumerate(kepids):

            print('Channel {}/{} | Kepler ID {} {}/{}'.format(ch, channels[-1], kepid, kepid_i, len(kepids)))

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

                    # # TODO: what to do with quarter 0?
                    # if quarter == 0:
                    #     continue

                    print('Channel {} Quarter {}'.format(ch, quarter))

                    centroid_x, centroid_y = hdu_list['LIGHTCURVE'].data.MOM_CENTR1, hdu_list[
                        'LIGHTCURVE'].data.MOM_CENTR2

                    # if _has_finite(light_curve.PSF_CENTR1):
                    #     centroid_x, centroid_y = light_curve.PSF_CENTR1, light_curve.PSF_CENTR2
                    # else:
                    #     if _has_finite(light_curve.MOM_CENTR1):
                    #         centroid_x, centroid_y = light_curve.MOM_CENTR1, light_curve.MOM_CENTR2
                    #     else:
                    #         continue  # no data

                # check if centroid time series is not all NaNs
                if np.any(~np.isfinite(centroid_x)):

                    # centroid_x, centroid_y = light_curve.MOM_CENTR1, light_curve.MOM_CENTR2

                    print(len(np.nonzero(np.isnan(centroid_x))[0]), len(np.nonzero(np.isnan(centroid_y))[0]))
                    print(len(centroid_x), len(centroid_y))

                    # TODO: take care of NaNs
                    # option 1 - set them to zero, why?
                    # centroid_x[np.isnan(centroid_x)] = 0
                    # centroid_y[np.isnan(centroid_y)] = 0

                    # option 2 - assume centroid time series noise is gaussian, use a robust estimator of the std, and
                    # impute it into the time series
                    med_centroidx = np.median(centroid_x)
                    med_centroidy = np.median(centroid_y)
                    std_rob_estmx = np.median(np.abs(centroid_x - med_centroidx)) * 1.4826
                    std_rob_estmy = np.median(np.abs(centroid_y - med_centroidy)) * 1.4826
                    centroid_xnan = np.isnan(centroid_x)
                    centroid_ynan = np.isnan(centroid_y)
                    centroid_x[centroid_xnan] = med_centroidx - \
                                                np.random.normal(0, std_rob_estmx, np.nonzero(centroid_xnan)[0].shape)
                    centroid_y[centroid_ynan] = med_centroidy - \
                                                np.random.normal(0, std_rob_estmy, np.nonzero(centroid_ynan)[0].shape)

                    data_mat[quarter]['x'].append(centroid_x)
                    data_mat[quarter]['y'].append(centroid_y)

                else:
                    print('Centroid data for {} is all NaNs'.format(filename))

        for q in data_mat:
            print('Quarter {}'.format(q))

            data_mat[q] = {coord: np.array(data_mat[q][coord], dtype='float').T for coord in ['x', 'y']}
            print('Matrix shape (x, y): {}, {}'.format(data_mat[q]['x'].shape, data_mat[q]['y'].shape))

            # get central tendency - median is more robust to outliers than mean
            # TODO: use a robust estimator of the mean
            centr_tend[q] = {coord: np.nanmedian(data_mat[q][coord], axis=0) for coord in ['x', 'y']}

            # # option 3 - remove indices
            # idxs_nan[q]['x'] = np.nonzero(np.all(np.isfinite(data_mat[q]['x']), axis=1))
            # idxs_nan[q]['y'] = np.nonzero(np.all(np.isfinite(data_mat[q]['y']), axis=1))
            # data_mat[q]['x'] = data_mat[q]['x'][idxs_nan[q]['x']]
            # data_mat[q]['y'] = data_mat[q]['y'][idxs_nan[q]['y']]
            #

            # remove the central tendency
            data_mat[q] = {coord: data_mat[q][coord] - centr_tend[q][coord] for coord in ['x', 'y']}

        np.save('{}rawdata_ch{}.npy'.format(save_dir, ch), data_mat)
        np.save('{}rawdata_ch_centraltendency{}.npy'.format(save_dir, ch), centr_tend)
        # np.save('{}rawdata_ch{}_idxsnan.npy'.format(save_dir, ch), idxs_nan)


if __name__ == '__main__':

    mpiproc_rank = MPI.COMM_WORLD.rank
    n_mpiprocs = MPI.COMM_WORLD.size
    print('Rank MPI process: {}/{}'.format(mpiproc_rank, n_mpiprocs))

    sys.stdout.flush()

    save_dir = '/home/msaragoc/Projects/Kepler-TESS_exoplanet/Data/svd_processing/Kepler/raw_data_matrix/'
    lc_data_dir = '/data5/tess_project/Data/Kepler-Q1-Q17-DR25/pdc-tce-time-series-fits'

    kepid_tbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Stellar parameters/Kepler/'
                            'q1_q17_dr25_stellar.csv')['kepid']
    kepids = kepid_tbl.unique()

    channels = np.arange(1, 85)

    boundaries = [int(i) for i in np.linspace(0, len(channels), n_mpiprocs + 1)]

    indices = [(boundaries[i], boundaries[i + 1]) for i in range(n_mpiprocs)][mpiproc_rank]

    channels_mpiproc = channels[indices[0]:indices[1]]

    get_chdatamatrix(channels, lc_data_dir, kepids, save_dir)
