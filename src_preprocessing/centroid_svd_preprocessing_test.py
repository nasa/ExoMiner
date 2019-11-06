from tensorflow import gfile
import pandas as pd
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import os
import itertools

# import src_preprocessing.preprocess as preprocess
import src_preprocessing.light_curve.kepler_io as kepler_io


# def _has_finite(array):
#     for i in array:
#         if np.isfinite(i):
#             return True
#
#     return False


#%% Build data matrices - Kepler

lc_data_dir = '/data5/tess_project/Data/Kepler-Q1-Q17-DR25/pdc-tce-time-series-fits'
# kepid_tbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/'
#                         'q1_q17_dr25_tce_2019.03.12_updt_tcert.csv')['kepid']
# kepid_tbl2 = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/180k_tce.csv')['kepid']
# kepids = pd.concat([kepid_tbl, kepid_tbl2]).unique()
kepid_tbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Stellar parameters/Kepler/'
                        'q1_q17_dr25_stellar.csv')['kepid']
kepids = kepid_tbl.unique()

# # get fits filenames for the Kepler IDs
# kepids_fits_filenames = {kepid: kepler_io.kepler_filenames(lc_data_dir, kepid) for kepid in kepids}

channels = np.arange(1, 85)

for ch in channels:

    print('##### CHANNEL {} ####'.format(ch))
    data_mat = {q: {'x': [], 'y': []} for q in range(1, 18)}

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

                centroid_x, centroid_y = hdu_list['LIGHTCURVE'].data.MOM_CENTR1, hdu_list['LIGHTCURVE'].data.MOM_CENTR2

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

                # option 3 - remove indices

            else:
                print('Centroid data for {} is all NaNs'.format(filename))

    for q in data_mat:
        print('Quarter {}'.format(q))
        data_mat[q]['x'] = np.array(data_mat[q]['x'], dtype='float').T
        data_mat[q]['y'] = np.array(data_mat[q]['y'], dtype='float').T
        print('Matrix shape (x, y): {}, {}'.format(data_mat[q]['x'].shape, data_mat[q]['y'].shape))

    np.save('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Data/svd_processing/Kepler/raw_data_matrix/'
            'rawdata_ch{}'.format(ch), data_mat)

# for ch in data_mat:
#     for q in data_mat[ch]:
#         print('Channel {}, quarter {}'.format(ch, q))
#         data_mat[ch][q]['x'] = np.array(data_mat[ch][q]['x'], dtype='float').T
#         data_mat[ch][q]['y'] = np.array(data_mat[ch][q]['y'], dtype='float').T
#         print('Matrix shape (x, y): {}, {}'.format(data_mat[ch][q]['x'].shape, data_mat[ch][q]['y'].shape))
#
# np.save('/home/msaragoc/Downlaods/centroid_svd.npy', data_mat)

#%% Build data matrices - TESS

save_dir = '/home/msaragoc/Projects/Kepler-TESS_exoplanet/Data/centroid_svd_processing/TESS/'
lc_data_dir = '/data5/tess_project/Data/TESS_TOI_fits(MAST)/'
ticid_tbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/'
                        'q1_q17_dr25_tce_2019.03.12_updt_tcert.csv')['tic']
ticids = ticid_tbl.unique()

# fits_header = fits.getheader('/data5/tess_project/Data/TESS_TOI_fits(MAST)/sector_1/tess2018206045859-s0001-0000000008196285-0120-s_lc.fits')
# with fits.open(gfile.Open('/data5/tess_project/Data/TESS_TOI_fits(MAST)/sector_1/tess2018206045859-s0001-0000000008196285-0120-s_lc.fits', "rb")) as hdu_list:
#     centroid_x, centroid_y = hdu_list['LIGHTCURVE'].data.MOM_CENTR1, hdu_list[
#         'LIGHTCURVE'].data.MOM_CENTR2

num_singularvalues = 6
NUM_CAMERAS = 4
num_sectors = 13

for sector in np.arange(1, num_sectors + 1):

    print('##### SECTOR {} ####'.format(sector))

    # get sector directory path
    sector_dir = os.path.join(lc_data_dir, 'sector_{}'.format(sector))

    # get filepaths for the FITS files for that sector
    fits_filepaths = [os.path.join(sector_dir, el) for el in os.listdir(sector_dir)]

    data_mat = {q: {'x': [], 'y': []} for q in range(1, NUM_CAMERAS + 1)}
    centr_tend = {q: {'x': [], 'y': []} for q in range(1, NUM_CAMERAS + 1)}
    idxs_nan = {q: {'x': [], 'y': []} for q in range(1, NUM_CAMERAS + 1)}
    singular_values = {q: {'x': [], 'y': []} for q in range(1, NUM_CAMERAS + 1)}
    ticids_aux = {q: [] for q in range(1, NUM_CAMERAS + 1)}
    raw_centroid_data = {q: {} for q in range(1, NUM_CAMERAS + 1)}
    centroid_data = {q: {} for q in range(1, NUM_CAMERAS + 1)}

    for fits_file in fits_filepaths:

        fits_header = fits.getheader(fits_file)
        camera = fits_header['CAMERA']
        ticid = fits_header['TICID']

        with fits.open(gfile.Open(fits_file, "rb")) as hdu_list:

            centroid_x, centroid_y = hdu_list['LIGHTCURVE'].data.MOM_CENTR1, hdu_list[
                'LIGHTCURVE'].data.MOM_CENTR2

        # check if centroid time series is not all NaNs
        if np.any(~np.isfinite(centroid_x)):

            data_mat[sector]['x'].append(centroid_x)
            data_mat[sector]['y'].append(centroid_y)

            ticids_aux[sector].append(ticid)

        else:
            print('Centroid data for {} is all NaNs'.format(fits_file))

    # get the raw data into data matrices and prepare them to SVD
    for q in data_mat:

        if len(data_mat[q]['x']) == 0:
            continue

        print('Camera {} ({})'.format(q, list(data_mat.keys())))

        data_mat[q] = {coord: np.array(data_mat[q][coord], dtype='float').T for coord in ['x', 'y']}
        print('Matrix shape (x, y): {}, {}'.format(data_mat[q]['x'].shape, data_mat[q]['y'].shape))

        # option 2 - remove indices for all target stars if at least one target shows a nan value for that cadence
        idxs_nanx = np.nonzero(np.all(np.isfinite(data_mat[q]['x']), axis=1))
        idxs_nany = np.nonzero(np.all(np.isfinite(data_mat[q]['y']), axis=1))
        idxs_nan[q] = np.union1d(idxs_nanx, idxs_nany)
        data_mat[q]['x'] = data_mat[q]['x'][idxs_nan[q]]
        data_mat[q]['y'] = data_mat[q]['y'][idxs_nan[q]]

        # get central tendency - median is more robust to outliers than mean
        # TODO: use a robust estimator of the mean
        centr_tend[q] = {coord: np.nanmedian(data_mat[q][coord], axis=0) for coord in ['x', 'y']}

        # remove the central tendency
        data_mat[q] = {coord: data_mat[q][coord] - centr_tend[q][coord] for coord in ['x', 'y']}

        # get the raw centroid time series for each target star
        for ticid_i, ticid in enumerate(ticids_aux[q]):
            raw_centroid_data[q][ticid] = {coord: data_mat[q][coord][:, ticid_i] for coord in ['x', 'y']}

    # saving raw data
    print('Saving raw data for sector {}...'.format(sector))
    np.save('{}raw_data/rawdata_s{}.npy'.format(save_dir, sector), data_mat)
    np.save('{}raw_data/rawdata_s{}_centraltendency.npy'.format(save_dir, sector), centr_tend)
    np.save('{}raw_data/centroidtimeseries_s{}.npy'.format(save_dir, sector), raw_centroid_data)
    np.save('{}raw_data/rawdata_s{}_idxsnan.npy'.format(save_dir, sector), idxs_nan)
    np.save('{}raw_data/ticids_s{}.npy'.format(save_dir, sector), ticids_aux)
    # data_mat = np.load('{}raw_data/rawdata_ch{}.npy'.format(save_dir, ch)).item()
    # centr_tend = np.load('{}raw_data/rawdata_ch{}_centraltendency.npy'.format(save_dir, ch)).item()

    # preprocess the raw data
    for q in data_mat:

        if len(data_mat[q]['x']) == 0:
            continue

        # Full SVD: A [mxn] = U [mxm] * S [mxn] * V^T [nxn]
        svd_comps = {coord: np.linalg.svd(data_mat[q][coord]) for coord in ['x', 'y']}

        # get the singular values
        singular_values[q] = {coord: svd_comps[coord][1] for coord in ['x', 'y']}

        # Truncated SVD: remove the components associated with the largest singular values
        # A_tr [mxn] = U [mxk] * S [mxk] * V^T [kxn]
        # A_new [mxn] = A [mxn] - A_tr [mxn]
        data_mat[q] = {coord: data_mat[q][coord] -
                              np.dot(svd_comps[coord][0][:, :num_singularvalues] *
                                     svd_comps[coord][1][:num_singularvalues],
                                     svd_comps[coord][2][:num_singularvalues, :])
                       for coord in ['x', 'y']}

        # add back the central tendency
        data_mat[q] = {coord: data_mat[q][coord] + centr_tend[q][coord] for coord in ['x', 'y']}

        # get the preprocessed centroid time series for each target star
        for ticid_i, ticid in enumerate(ticids_aux[q]):
            centroid_data[q][ticid] = {coord: data_mat[q][coord][:, ticid_i] for coord in ['x', 'y']}

    # save processed data
    print('Saving preprocessed data for sector {}...'.format(sector))
    np.save('{}singular_values/singularvalues_s{}.npy'.format(save_dir, sector), singular_values)
    np.save('{}svdpreproc_data/ppdata_s{}.npy'.format(save_dir, sector), data_mat)
    np.save('{}svdpreproc_data/centroidtimeseries_s{}.npy'.format(save_dir, sector), centroid_data)


#%% Plot raw and processed centroid data - Kepler

channel = 1

save_dir = '/home/msaragoc/Projects/Kepler-TESS_exoplanet/Data/centroid_svd_processing/Kepler/'

raw_centroids = np.load('{}raw_data/centroidtimeseries_ch{}.npy'.format(save_dir, channel)).item()
centr_tend = np.load('{}raw_data/rawdata_ch{}_centraltendency.npy'.format(save_dir, channel)).item()
kepids_aux = np.load('{}raw_data/kepids_ch{}.npy'.format(save_dir, channel)).item()
prep_centroids = np.load('{}svdpreproc_data/centroidtimeseries_ch{}.npy'.format(save_dir, channel)).item()

plot_dir = '/home/msaragoc/Projects/Kepler-TESS_exoplanet/Data/centroid_svd_processing/Kepler/' \
           'centroid_timeseries_plots/'

for quarter in range(1, 18):

    print('Quarter {}/{}'.format(quarter, 17))

    for kepid in raw_centroids[quarter]:

        f, ax = plt.subplots(1, 2,  sharex='col', sharey='col', figsize=(14, 9))
        ax[0].plot(raw_centroids[quarter][kepid]['x'] + centr_tend[quarter]['x'][np.where(kepids_aux[quarter] == kepid)], label='raw')
        ax[0].set_ylabel('Amplitude')
        ax[0].set_title('Coordinate x')
        ax[1].plot(raw_centroids[quarter][kepid]['y'] + centr_tend[quarter]['y'][np.where(kepids_aux[quarter] == kepid)], label='raw')
        ax[1].set_title('Coordinate y')
        ax[0].plot(prep_centroids[quarter][kepid]['x'], label='processed')
        ax[0].set_xlabel('Sample number')
        ax[0].legend()
        # ax[1, 0].set_ylabel('Amplitude')
        # ax[1, 0].set_title('Preprocessed centroid x')
        ax[1].plot(prep_centroids[quarter][kepid]['y'], label='processed')
        ax[1].set_xlabel('Sample number')
        # ax[1, 1].set_title('Preprocessed centroid y')
        ax[1].legend()
        f.suptitle('Channel {} | Quarter {} | Kepler ID {}'.format(channel, quarter, kepid))
        # aaaa
        f.savefig('{}centroidtimeseries_ch{}_q{}_kepid{}.svg'.format(plot_dir, channel, quarter, kepid))
        # plt.waitforbuttonpress()
        plt.close()
        # aaaa

#%% Plot singular values - Kepler

channels = range(1, 85)
quarters = range(1, 18)

save_dir = '/home/msaragoc/Projects/Kepler-TESS_exoplanet/Data/centroid_svd_processing/Kepler/'

plot_dir = '/home/msaragoc/Projects/Kepler-TESS_exoplanet/Data/centroid_svd_processing/Kepler/' \
           'singularvalues_plots/'

for channel in channels:

    singular_values = np.load('{}singular_values/singularvalues_ch{}.npy'.format(save_dir, channel)).item()

    f, ax = plt.subplots(2, 1, figsize=(14, 9))
    # f, ax = plt.subplots(1, 1, sharex=True, figsize=(14, 9))
    for quarter in quarters:

        # if quarter != 1 or channel != 4:
        #     continue

        print('Channel {}/{} | Quarter {}/{}'.format(channel, channels[-1], quarter, quarters[-1]))

        ax[0].plot(singular_values[quarter]['x'])
        ax[1].plot(singular_values[quarter]['y'])
        # ax.scatter(np.arange(len(singular_values[quarter]['x'])), singular_values[quarter]['x'])

        # TODO: logy scale?
        # f, ax = plt.subplots(2, 1, sharex=True, figsize=(14, 9))
        # ax[0].plot(singular_values[quarter]['x'])
        # # ax[0].scatter(np.arange(len(singular_values[quarter]['x'])), singular_values[quarter]['x'])
        # ax[0].set_ylabel('Singular value')
        # ax[0].set_title('Coordinate x')
        # ax[0].set_xticks(np.arange(0, 20))
        # ax[0].set_xlim([0, 20])
        # ax[0].set_yscale('log')
        # ax[1].plot(singular_values[quarter]['y'])
        # # ax[1].scatter(np.arange(len(singular_values[quarter]['y'])), singular_values[quarter]['y'])
        # ax[1].set_ylabel('Singular value')
        # ax[1].set_title('Coordinate y')
        # ax[1].set_xlabel('Singular number')
        # ax[1].set_xticks(np.arange(0, 20))
        # ax[1].set_xlim([0, 20])
        # ax[1].set_yscale('log')
        # f.suptitle('Channel {} | Quarter {}'.format(channel, quarter))
        # # aaaa
        # plt.waitforbuttonpress()
        # # f.savefig('{}singularvalues_ch{}_q{}.png'.format(plot_dir, channel, quarter))
        # plt.close()

    ax[0].set_ylabel('Singular value')
    ax[0].set_title('Coordinate x')
    ax[0].set_xticks(np.arange(0, 20))
    ax[0].set_xlim([0, 20])
    ax[1].set_xlabel('Singular number')
    ax[1].set_ylabel('Singular value')
    ax[1].set_title('Coordinate y')
    ax[1].set_xticks(np.arange(0, 20))
    ax[1].set_xlim([0, 20])
    f.suptitle('Channel {}'.format(channel))
    # aaaa
    plt.waitforbuttonpress()
    f.savefig('{}singularvalues_ch{}.png'.format(plot_dir, channel))
    plt.close()

#%% Plot raw and processed centroid data - TESS

sector = 1
camera = 1
ccd = 2

save_dir = '/home/msaragoc/Projects/Kepler-TESS_exoplanet/Data/centroid_svd_processing/TESS/'

raw_centroids = np.load('{}raw_data/centroidtimeseries_s{}_c{}_ccd{}.npy'.format(save_dir, sector, camera, ccd)).item()
prep_centroids = np.load('{}svdpreproc_data/centroidtimeseries_s{}_c{}_ccd{}.npy'.format(save_dir, sector, camera,
                                                                                         ccd)).item()
centr_tend = np.load('{}raw_data/rawdata_s{}_c{}_ccd{}_centraltendency.npy'.format(save_dir, sector, camera, ccd))
ticids_aux = np.load('{}raw_data/ticids_s{}_c{}_ccd{}.npy'.format(save_dir, sector, camera, ccd))

plot_dir = '/home/msaragoc/Projects/Kepler-TESS_exoplanet/Data/centroid_svd_processing/TESS/' \
           'centroid_timeseries_plots/'

for ticid in raw_centroids:

    f, ax = plt.subplots(1, 2,  sharex='col', sharey='col', figsize=(14, 9))
    ax[0].plot(raw_centroids[ticid]['x'] + centr_tend[2 * np.where(ticids_aux == ticid)[0][0]], label='raw')
    ax[0].set_ylabel('Amplitude')
    ax[0].set_title('Coordinate x')
    ax[1].plot(raw_centroids[ticid]['y'] + centr_tend[2 * np.where(ticids_aux == ticid)[0][0] + 1], label='raw')
    ax[1].set_title('Coordinate y')
    ax[0].plot(prep_centroids[ticid]['x'], label='processed')
    ax[0].set_xlabel('Sample number')
    ax[0].legend()
    # ax[1, 0].set_ylabel('Amplitude')
    # ax[1, 0].set_title('Preprocessed centroid x')
    ax[1].plot(prep_centroids[ticid]['y'], label='processed')
    ax[1].set_xlabel('Sample number')
    # ax[1, 1].set_title('Preprocessed centroid y')
    ax[1].legend()
    f.suptitle('Sector {} | Camera {} | CCD {} | TIC ID {}'.format(sector, camera, ccd, ticid))
    # aaaa
    f.savefig('{}centroidtimeseries_s{}_c{}_ccd{}_ticid{}.svg'.format(plot_dir, sector, camera, ccd, ticid))
    plt.waitforbuttonpress()
    plt.close()
    # aaaa

#%% Plot singular values - TESS

sectors = range(1, 2)
cameras = range(1, 5)
ccds = range(1, 4)

save_dir = '/home/msaragoc/Projects/Kepler-TESS_exoplanet/Data/centroid_svd_processing/TESS/'

plot_dir = '/home/msaragoc/Projects/Kepler-TESS_exoplanet/Data/centroid_svd_processing/TESS/' \
           'singularvalues_plots/'

for sector in sectors:

    # f, ax = plt.subplots(2, 1, figsize=(14, 9))
    f, ax = plt.subplots(1, 1, figsize=(14, 9))

    for camera in cameras:
        for ccd in ccds:

            print('Sector {}/{} | Camera {}/{} CCD {}/{}'.format(sector, sectors[-1], camera, cameras[-1], ccd, ccds[-1]))

            try:
                singular_values = np.load('{}singular_values/singularvalues_s{}_c{}_ccd{}.npy'.format(save_dir, sector, camera, ccd))
            except Exception as e:
                print(e)
                print('No singular values.')
                continue

            ax.plot(singular_values)
            # ax[0].plot(singular_values['x'])
            # ax[1].plot(singular_values['y'])
            # ax.scatter(np.arange(len(singular_values[quarter]['x'])), singular_values[quarter]['x'])

    # TODO: logy scale?
    ax.set_yscale('log')
    ax.set_ylabel('Singular value')
    ax.set_xlabel('Singular number')
    # ax.set_title()
    ax.set_xticks(np.arange(0, 20))
    ax.set_xlim([0, 20])
    # ax[0].set_yscale('log')
    # ax[1].set_yscale('log')
    # ax[0].set_ylabel('Singular value')
    # ax[0].set_title('Coordinate x')
    # ax[0].set_xticks(np.arange(0, 20))
    # ax[0].set_xlim([0, 20])
    # ax[1].set_xlabel('Singular number')
    # ax[1].set_ylabel('Singular value')
    # ax[1].set_title('Coordinate y')
    # ax[1].set_xticks(np.arange(0, 20))
    # ax[1].set_xlim([0, 20])
    f.suptitle('Sector {}'.format(sector))
    # aaaa
    plt.waitforbuttonpress()
    f.savefig('{}singularvalues_s{}.png'.format(plot_dir, sector))
    plt.close()
