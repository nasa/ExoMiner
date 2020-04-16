import matplotlib.pyplot as plt
import numpy as np
import os

#%% Scree plot

channel = 5
singular_values = np.load('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Data/centroid_pca_denoising/Kepler/test/'
                          'pcadenoising/singularvalues_ch5.npy', allow_pickle=True).item()

f, ax = plt.subplots()
for quarter in singular_values:

    ax.plot(np.arange(1, len(singular_values[quarter]) + 1, 1), singular_values[quarter], label='q{}'.format(quarter))
    ax.scatter(np.arange(1, len(singular_values[quarter]) + 1, 1), singular_values[quarter], c='r')
ax.grid(True)
ax.set_xlim(left=1, right=20)
ax.set_yscale('log')
ax.set_ylabel('Singular value')
ax.set_xlabel('Component number')
ax.set_title('Channel {} | Quarter {}'.format(channel, quarter))
ax.legend()
f.savefig(os.path.join('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Data/centroid_pca_denoising/Kepler/test/'
                       'pcadenoising', 'screeplot_ch{}.png'.format(channel)))

#%% Plot raw vs denoised centroid time-series

channel = 5
centr_tend = np.load('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Data/centroid_pca_denoising/Kepler/test/ppdata/'
                     'centraltendency_ch5.npy', allow_pickle=True).item()
raw_centroids = np.load('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Data/centroid_pca_denoising/Kepler/test/ppdata/'
                        'centroidtimeseries_ch5.npy', allow_pickle=True).item()
denoised_centroids = np.load('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Data/centroid_pca_denoising/Kepler/test/'
                             'pcadenoising/denoisedcentroidtimeseries_ch5.npy', allow_pickle=True).item()
kepids_aux = np.load('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Data/centroid_pca_denoising/Kepler/test/ppdata/'
                     'kepids_ch5.npy', allow_pickle=True).item()

i = 0
for kepid in raw_centroids:
    f, ax = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
    start_idx = 0

    ax[0].set_title('Channel {} | Quarter(s) {}\n'
                    'Centroid time-series Kepler ID {:.0f}'.format(channel, list(raw_centroids[kepid].keys()), kepid))
    for quarter in raw_centroids[kepid]:
        ax[0].plot(np.arange(start_idx, len(raw_centroids[kepid][quarter]['x'])),
                   raw_centroids[kepid][quarter]['x'] + centr_tend[quarter][kepids_aux[kepid][quarter]],
                   'b',
                   label='raw')
        ax[0].plot(np.arange(start_idx, len(raw_centroids[kepid][quarter]['x'])),
                   denoised_centroids[kepid][quarter]['x'],
                   'r',
                   label='denoised')

        ax[1].plot(np.arange(start_idx, len(raw_centroids[kepid][quarter]['y'])),
                   raw_centroids[kepid][quarter]['y'] + centr_tend[quarter][kepids_aux[kepid][quarter] + 1],
                   'b',
                   label='raw')
        ax[1].plot(np.arange(start_idx, len(raw_centroids[kepid][quarter]['y'])),
                   denoised_centroids[kepid][quarter]['y'],
                   'r',
                   label='denoised')

        start_idx += len(raw_centroids[kepid][quarter]['x'])

    ax[0].set_ylabel('CCD Row Coordinate [px]')
    ax[0].legend()

    ax[1].set_ylabel('CCD Col Coordinate [px]')
    ax[1].set_xlabel('Sample number')
    ax[1].legend()
    f.savefig(os.path.join('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Data/centroid_pca_denoising/Kepler/test/plots/'
                           'centroidtimeseries-rawvsdenoised',
                           'kepid{}-ch{}-q{}_centroidtimeseries-rawvsdenoised.png'.format(kepid, channel, quarter)))

    i += 1
    if i > 3:
        break
