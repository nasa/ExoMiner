""" Auxiliary functions used to plot outcome from different steps along the preprocessing pipeline. """

# 3rd party
import matplotlib as mpl
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from astropy.stats import mad_std
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astropy import units as u
from PIL import Image
import glob
import matplotlib.lines as mlines
import matplotlib.ticker as ticker

plt.switch_backend('agg')
# plt.rcParams['text.usetex'] = True


DEGREETOARCSEC = 3600


def plot_intransit_binary_timeseries(all_time, all_flux, intransit_cadences_target, intransit_cadences_tce, tce,
                                     savefp):
    """ Creates and saves a 2x1 figure with plots that show the ephemeris pulse train and the flux time-series for
    a given TCE.

    :param all_time: list of numpy arrays, time
    :param all_flux: list of numpy arrays, flux time-series
    :param intransit_cadences_target: list of numpy arrays, binary arrays with 1 for in-transit cadences and 0 otherwise
    for all detected TCEs in the target star
    :param intransit_cadences_tce: list of numpy arrays, binary arrays with 1 for in-transit cadences and 0 otherwise
    for the TCE of interest
    :param tce: Pandas Series, row of the input TCE table Pandas DataFrame.
    :param savefp: Path, filepath used to save figure
    :return:
    """

    # if not centroid:
    f, ax = plt.subplots(2, 1, sharex=True, figsize=(14, 8))

    n_arrs = len(all_time)
    for i in range(n_arrs):
        ax[0].plot(all_time[i], intransit_cadences_target[i], 'b', zorder=1,
                   label=None if i < n_arrs - 1 else 'Detected target TCEs', alpha=0.2)
        ax[0].plot(all_time[i], intransit_cadences_tce[i], 'k--', zorder=2,
                   label=None if i < n_arrs - 1 else 'TCE of interest', linewidth=2)
        ax[0].axvline(x=all_time[i][-1], ymax=1, ymin=0, c='r')
    ax[0].legend()
    ax[0].set_title('Binary timeseries')
    ax[0].set_ylabel('In-transit Cadences Flag')
    ax[0].set_xlim([all_time[0][0], all_time[-1][-1]])

    for i in range(len(all_time)):
        ax[1].scatter(all_time[i], all_flux[i], c='k', s=4)
        ax[1].axvline(x=all_time[i][-1], ymax=1, ymin=0, c='r')
    ax[1].set_title('Flux')
    ax[1].set_xlim([all_time[0][0], all_time[-1][-1]])
    ax[1].set_ylabel('Amplitude')
    ax[1].set_xlabel('Time [day]')
    # else:
    #     f, ax = plt.subplots(3, 1, sharex=True, figsize=(14, 8))
    #
    #     for i in range(len(all_time)):
    #         ax[0].plot(all_time[i], binary_time_all[i], 'b')
    #         ax[0].axvline(x=all_time[i][-1], ymax=1, ymin=0, c='r')
    #     ax[0].set_title('Binary timeseries')
    #     ax[0].set_ylabel('In-transit Cadences Flag')
    #     ax[0].set_xlim([all_time[0][0], all_time[-1][-1]])
    #
    #     for i in range(len(all_time)):
    #         ax[1].scatter(all_time[i], all_flux['x'][i], c='k', s=4)
    #         ax[1].axvline(x=all_time[i][-1], ymax=1, ymin=0, c='r')
    #     ax[1].set_ylabel('RA [deg]')
    #     ax[1].set_title('Centroid')
    #     ax[1].set_xlim([all_time[0][0], all_time[-1][-1]])
    #     for i in range(len(all_time)):
    #         # ax[2].plot(all_time[i], all_flux['y'][i], 'b')
    #         ax[2].scatter(all_time[i], all_flux['x'][i], c='k', s=4)
    #         ax[2].axvline(x=all_time[i][-1], ymax=1, ymin=0, c='r')
    #     ax[2].set_ylabel('Dec [deg]')
    #     ax[2].set_xlabel('Time [day]')
    #     ax[2].set_xlim([all_time[0][0], all_time[-1][-1]])

    f.suptitle(f'{tce.uid} {tce.label}')
    plt.savefig(savefp)
    plt.close()


def plot_centroids(time, centroids, detrended_centroids, target_uid, config, savefp, pxcoordinates=False,
                   target_position=None, delta_dec=None):
    """ Creates and saves a figure with plots that show the centroid, trend, and detrended centroid timeseries for a
    given TCE.

    :param time: numpy array, time
    :param centroids: dict with 'x' and 'y' keys for the coordinates, and values are numpy arrays. Holds the raw
        centroid timeseries
    :param detrended_centroids: dict with 'x' and 'y' keys for the coordinates, and values are dictionaries with the
        detrended centroid 'detrended', removed trend 'trend', residual time series 'residual', and, optionally, the
        linearly interpolated raw centroid timeseries used for fitting 'linear_interp'
    :param target_uid: str, target unique identifier (UID)
    :param config: dict, preprocessing parameters
    :param savefp: Path, filepath to saved figure
    :param pxcoordinates: bool, whether centroid values are in row/col pixel values or celestial coordinates
    :param target_position: list, position of the target [row, col] (or [RA, Dec], if centroid is in celestial
    coordinates)
    :param delta_dec: float, target declination correction

    :return:
    """

    # copy centroid data to plot it
    centroids_plot = {coord: np.array(centroid_arr) for coord, centroid_arr in centroids.items()}
    detrended_centroids_plot = {}
    target_position_plot = np.array(target_position)
    target_position_unit = 'deg'
    for coord in detrended_centroids:
        detrended_centroids_plot[coord] = {}
        for timeseries_name, timeseries_arr in detrended_centroids[coord].items():
            detrended_centroids_plot[coord][timeseries_name] = np.array(timeseries_arr)

    if target_position is not None:  # center centroid on target position
        centroids_plot = {coord: centroid_arr - target_position[coord_i]
                          for coord_i, (coord, centroid_arr) in enumerate(centroids_plot.items())}
        centroids_plot['x'] *= delta_dec

        for coord_i, coord in enumerate(detrended_centroids_plot):
            for timeseries_name, timeseries_arr in detrended_centroids_plot[coord].items():
                detrended_centroids_plot[coord][timeseries_name] = timeseries_arr - target_position[coord_i]
                if coord == 'x':
                    detrended_centroids_plot[coord][timeseries_name] *= delta_dec

    # convert from degrees to arcsec for when centroid is in celestial coordinates
    if not config['px_coordinates'] and not pxcoordinates:
        centroids_plot = {coord: DEGREETOARCSEC * centroid_arr for coord, centroid_arr in centroids_plot.items()}

        for coord_i, coord in enumerate(detrended_centroids_plot):
            for timeseries_name, timeseries_arr in detrended_centroids_plot[coord].items():
                detrended_centroids_plot[coord][timeseries_name] *= DEGREETOARCSEC

        target_position_plot *= DEGREETOARCSEC

        target_position_unit = 'arcsec'

    f, ax = plt.subplots(2, 2, figsize=(18, 12))

    ax[0, 0].plot(time, centroids_plot['x'], 'b', zorder=0)
    # ax[0, 0].plot(time, detrended_centroids_plot['x']['trend'], 'orange', linestyle='--', label='Trend', zorder=1)
    if 'linear_interp' in detrended_centroids_plot['x']:
        ax[0, 0].plot(time, detrended_centroids_plot['x']['linear_interp'], 'g', label='Linear Interp.', zorder=0)
    # ax[0, 0].legend()
    ax[0, 0].set_xlim(time[[0, -1]])
    ax[0, 1].plot(time, detrended_centroids_plot['x']['detrended'], 'b', zorder=0)
    ax[0, 1].plot(time, detrended_centroids_plot['x']['trend'], 'orange', linestyle='--', label='Trend', zorder=1)
    ax[0, 1].set_xlim(time[[0, -1]])
    ax[0, 1].legend()

    ax[1, 0].plot(time, centroids_plot['y'], 'b', zorder=0)
    # ax[1, 0].plot(time, detrended_centroids_plot['y']['trend'], 'orange', linestyle='--', label='Trend', zorder=1)
    if 'linear_interp' in detrended_centroids_plot['x']:
        ax[1, 0].plot(time, detrended_centroids_plot['y']['linear_interp'], 'g', label='Linear Interp.', zorder=0)
    ax[1, 0].set_xlim(time[[0, -1]])
    # ax[1, 0].legend()
    ax[1, 1].plot(time, detrended_centroids_plot['y']['detrended'], 'b', zorder=0)
    ax[1, 1].plot(time, detrended_centroids_plot['y']['trend'], 'orange', linestyle='--', label='Trend', zorder=1)
    ax[1, 1].legend()
    ax[1, 1].set_xlim(time[[0, -1]])

    if config['px_coordinates'] or pxcoordinates:
        ax[0, 0].set_ylabel('Col pixel')
        ax[1, 0].set_ylabel('Row pixel')
    else:
        ax[0, 0].set_ylabel(f'RA [{target_position_unit}] {("", " to target")[target_position is not None]}')
        ax[1, 0].set_ylabel(f'Dec [{target_position_unit}] {("", " to target")[target_position is not None]}')

    # ax[0, 1].set_ylabel('Normalized Value')
    # ax[1, 1].set_ylabel('Normalized Value')

    ax[1, 0].set_xlabel('Time [day]')
    ax[1, 1].set_xlabel('Time [day]')

    ax[0, 0].set_title('Raw Centroids')
    ax[0, 1].set_title('Detrended Centroids')

    f.suptitle(f'{target_uid}\nTarget: {target_position_plot[0]:.3f}, {target_position_plot[1]:.3f} '
               f'({target_position_unit})')
    plt.savefig(savefp)
    plt.close()


def plot_flux_detrend(time, flux, trend, detrended_flux, target_uid, savedir, basename, flux_interp=None):
    """ Creates and saves a 2x1 figure with plots that show the flux time series and the fitted trend and
     the respective detrended flux time series for a given TCE.

    :param time: numpy array, time
    :param flux: numpy array, flux
    :param trend: numpy array, fitted trend
    :param detrended_flux: numpy array, detrended flux
    :param target_uid: str, target unique identifier (UID)
    :param savedir: str, filepath to directory in which the figure is saved
    :param basename: str, added to the figure filename
    :param flux_interp: numpy array, linearly interpolated flux used for detrending
    :return:
    """

    f, ax = plt.subplots(2, 1, figsize=(16, 10))
    ax[0].plot(time, flux, 'b', zorder=0)
    # ax[0].plot(time, trend, 'orange', linestyle='--', label='Trend', zorder=1)
    if flux_interp is not None:
        ax[0].plot(time, flux_interp, 'g', label='Flux w/ lin. interpolated across transits', zorder=0)
    # ax[0].legend()
    ax[0].set_xlim(time[[0, -1]])
    ax[1].plot(time, detrended_flux, 'b', zorder=1)
    ax[1].plot(time, trend, 'orange', linestyle='--', label='Trend', zorder=2)
    ax[1].legend()
    ax[0].set_ylabel(fr'Amplitude [$e^-s^-1$]')
    ax[0].set_title('Raw Flux')
    ax[1].set_ylabel('Normalized Amplitude')
    ax[1].set_xlabel('Time [day]')
    ax[1].set_title('Detrended Flux')
    ax[1].set_xlim(time[[0, -1]])
    f.suptitle(f'Target {target_uid}')
    plt.savefig(savedir / f'{target_uid}_{basename}.png')
    plt.close()


def plot_centroids_it_oot(all_time, binary_time_all, all_centroids, avg_centroid_oot, target_coords, tce, config,
                          savedir, basename, target_center=True):
    """ Creates and saves a 2x3 figure with plots that show the out-of-transit and in-transit centroid time-series and
    their averages, as well as the target position, for a given TCE.

    :param all_time: list of numpy arrays, time
    :param binary_time_all: list of numpy arrays, binary arrays with 1 for in-transit cadences and 0 otherwise
    :param all_centroids: dict ('x' and 'y' keys, values are lists of numpy arrays), centroid time-series
    :param centroid_oot: dict ('x' and 'y' keys, values are lists of numpy arrays), out-of-transit centroid time-series
    :param avg_centroid_oot: dict ('x' and 'y' keys), coordinates of the average out-of-transit centroid
    :param target_coords: list, RA and Dec coordinates of the target
    :param tce: pandas Series, row of the input TCE table Pandas DataFrame
    :param config: dict, preprocessing parameters
    :param savedir: str, filepath to directory in which the figure is saved
    :param basename: str, added to the figure filename
    :return:
    """

    # provide coordinate relative to the target
    if target_center:
        all_centroids = {'x': [(centroids_arr - target_coords[0])  # * np.cos(target_coords[1] * np.pi / 180)
                               for centroids_arr in all_centroids['x']],
                         'y': [(centroids_arr - target_coords[1]) for centroids_arr in all_centroids['y']]}
        avg_centroid_oot = {'x': avg_centroid_oot['x'] - target_coords[0],
                            'y': avg_centroid_oot['y'] - target_coords[1]}

    # convert from degree to arcsec
    if not config['px_coordinates']:
        all_centroids = {coord: [DEGREETOARCSEC * centroids_arr for centroids_arr in all_centroids[coord]]
                         for coord in all_centroids}
        target_coords = [coord * DEGREETOARCSEC for coord in target_coords]

        avg_centroid_oot = {coord: DEGREETOARCSEC * avg_centroid_oot[coord] for coord in avg_centroid_oot}
        # avg_centroid_oot = {coord: [DEGREETOARCSEC * avg_centroid_oot[coord][i] for i in range(len(avg_centroid_oot[coord]))]
        #                     for coord in avg_centroid_oot}

    centroid_oot = {coord: [centroids[np.where(binary_time == 0)] for binary_time, centroids in
                            zip(binary_time_all, all_centroids[coord])] for coord in all_centroids}
    centroid_it = {coord: [centroids[np.where(binary_time == 1)] for binary_time, centroids in
                           zip(binary_time_all, all_centroids[coord])] for coord in all_centroids}

    all_time_oot = [time[np.where(binary_time == 0)] for time, binary_time in zip(all_time, binary_time_all)]
    all_time_it = [time[np.where(binary_time == 1)] for time, binary_time in zip(all_time, binary_time_all)]

    avg_centroid_it = {coord: np.median(np.concatenate(centroid_it[coord])) for coord in centroid_it}
    # avg_centroid_it = {coord: [np.median(centroid_it[coord][i]) for i in range(len(centroid_it[coord]))]
    #                    for coord in centroid_it}

    f, ax = plt.subplots(2, 2, figsize=(18, 8))

    for i in range(len(all_time_oot)):
        ax[0, 0].plot(all_time_oot[i], centroid_oot['x'][i], 'b', zorder=0)
    # plt.plot(np.concatenate(all_time), np.concatenate(all_centroids['x']))
    ax[0, 0].plot(np.concatenate(all_time_oot), avg_centroid_oot['x'] * np.ones(len(np.concatenate(all_time_oot))),
                  'r--',
             label='avg oot', zorder=1)
    # plt.plot(np.concatenate(all_time),
    #          np.concatenate([avg_centroid_oot['x'][i] * np.ones(len(all_time[i])) for i in range(len(all_time))]),
    #          'r--', label='avg oot', zorder=1)
    ax[0, 0].legend()
    if config['px_coordinates']:
        ax[0, 0].set_ylabel('Col pixel')
    else:
        ax[0, 0].set_ylabel(f'RA [arcsec] {"from target" if target_center else ""}')
    ax[0, 0].set_title('Out-of-transit points')

    for i in range(len(all_time_it)):
        # plt.scatter(all_time_it[i], centroid_it['x'][i], color='c', zorder=0)
        ax[0, 1].plot(all_time_it[i], centroid_it['x'][i], 'b', zorder=0)
    ax[0, 1].plot(np.concatenate(all_time_it), avg_centroid_it['x'] * np.ones(len(np.concatenate(all_time_it))), 'g--',
             label='avg it', zorder=1)
    # plt.plot(np.concatenate(all_time),
    #          np.concatenate([avg_centroid_it['x'][i] * np.ones(len(all_time[i])) for i in range(len(all_time))]),
    #          'g--', label='avg it', zorder=1)
    ax[0, 1].legend()
    ax[0, 1].set_title('In-transit points')

    for i in range(len(all_time_oot)):
        ax[1, 0].plot(all_time_oot[i], centroid_oot['y'][i], 'b', zorder=0)
    # plt.plot(np.concatenate(all_time), np.concatenate(all_centroids['y']))
    ax[1, 0].plot(np.concatenate(all_time_oot), avg_centroid_oot['y'] * np.ones(len(np.concatenate(all_time_oot))),
                  'r--',
             label='avg oot', zorder=1)
    # plt.plot(np.concatenate(all_time),
    #          np.concatenate([avg_centroid_oot['y'][i] * np.ones(len(all_time[i])) for i in range(len(all_time))]),
    #          'r--', label='avg oot', zorder=1)

    ax[1, 0].legend()
    ax[1, 0].set_xlabel('Time [day]')
    if config['px_coordinates']:
        ax[1, 0].set_ylabel('Row pixel')
    else:
        ax[1, 0].set_ylabel(f'Dec [arcsec] {"from target" if target_center else ""}')

    for i in range(len(all_time_it)):
        # plt.scatter(all_time_it[i], centroid_it['y'][i], color='c', zorder=0)
        ax[1, 1].plot(all_time_it[i], centroid_it['y'][i], 'b', zorder=0)
    ax[1, 1].plot(np.concatenate(all_time_it), avg_centroid_it['y'] * np.ones(len(np.concatenate(all_time_it))), 'g--',
                  label='avg it', zorder=1)
    # plt.plot(np.concatenate(all_time),
    #          np.concatenate([avg_centroid_it['y'][i] * np.ones(len(all_time[i])) for i in range(len(all_time))]),
    #          'g--', label='avg it', zorder=1)
    ax[1, 1].legend()
    ax[1, 1].set_xlabel('Time [day]')

    f.suptitle('Centroid time-series\n TCE {} {}\nTarget: {} [arcsec]'.format(tce['uid'],
                                                                              tce['label'],
                                                                              target_coords))
    plt.savefig(os.path.join(savedir, '{}_{}_{}.png'.format(tce.uid, tce.label, basename)))
    plt.close()


def plot_corrected_centroids(all_time, all_centroids, avg_centroid_oot, tce, config, savefp, pxcoordinates,
                             target_position=None, delta_dec=None):
    """ Creates and saves a 2x2 figure with plots that show the corrected centroid timeseries and the respective
    out-of-transit centroid, as well as the target position, for a given TCE.

    :param all_time: numpy array, time
    :param all_centroids: dict ('x' and 'y' keys, values are numpy arrays), centroid timeseries
    :param avg_centroid_oot: dict ('x' and 'y' keys), coordinates of the average out-of-transit centroid
    :param tce: pandas Series, row of the input TCE table Pandas DataFrame.
    :param config: dict, preprocessing parameters.
    :param savedir: Path, filepath to save figure
    :param pxcoordinates: bool, whether centroid values are in row/col pixel values or celestial coordinates
    :param target_position: list, position of the target [row, col] (or [RA, Dec], if centroid is in celestial
    coordinates)
    :param delta_dec: float, target declination correction

    :return:
    """

    # copy centroid data to plot it
    all_centroids_plot = {coord: np.array(centroid_arr) for coord, centroid_arr in all_centroids.items()}
    target_position_plot = np.array(target_position)
    avg_centroid_oot_plot = {coord: coord_val for coord, coord_val in avg_centroid_oot.items()}
    target_position_unit = 'deg'

    if target_position is not None:  # center centroid on target position
        all_centroids_plot = {coord: centroid_arr - target_position[coord_i]
                              for coord_i, (coord, centroid_arr) in enumerate(all_centroids_plot.items())}
        all_centroids_plot['x'] *= delta_dec

        avg_centroid_oot_plot = {coord: centroid_arr - target_position[coord_i]
                                 for coord_i, (coord, centroid_arr) in enumerate(avg_centroid_oot_plot.items())}
        avg_centroid_oot_plot['x'] *= delta_dec

    # convert from degrees to arcsec for when centroid is in celestial coordinates
    if not config['px_coordinates'] and not pxcoordinates:
        all_centroids_plot = {coord: DEGREETOARCSEC * centroid_arr
                              for coord, centroid_arr in all_centroids_plot.items()}

        avg_centroid_oot_plot = {coord: DEGREETOARCSEC * centroid_arr
                                 for coord, centroid_arr in avg_centroid_oot_plot.items()}

        target_position_plot *= DEGREETOARCSEC
        target_position_unit = 'arcsec'

    f, ax = plt.subplots(2, 1, figsize=(20, 8))

    ax[0].plot(all_time, all_centroids_plot['x'], 'b', zorder=0)
    ax[0].plot(all_time, avg_centroid_oot_plot['x'] * np.ones(len(all_time)), 'r--', label='avg oot', zorder=1)
    ax[0].legend()
    if config['px_coordinates']:
        ax[0].set_ylabel('Col pixel')
    else:
        ax[0].set_ylabel(f'RA [arcsec]{(""," to target")[target_position is not None]}')
    ax[0].set_title('Corrected Centroids')
    ax[0].set_xlim(all_time[[0, -1]])

    ax[1].plot(all_time, all_centroids_plot['y'], 'b', zorder=0)
    ax[1].plot(all_time, avg_centroid_oot_plot['y'] * np.ones(len(all_time)), 'r--', label='avg oot', zorder=1)
    ax[1].legend()
    if config['px_coordinates']:
        ax[1].set_ylabel('Row pixel')
    else:
        ax[1].set_ylabel(f'Dec [arcsec]{(""," to target")[target_position is not None]}')
    ax[1].set_xlabel('Time [day]')
    ax[1].set_xlim(all_time[[0, -1]])

    f.suptitle(f'{tce.uid} {tce.label}'
               f'\nTarget: {target_position_plot[0]:.3f}, {target_position_plot[1]:.3f} ({target_position_unit})')
    plt.savefig(savefp)
    plt.close()


def plot_dist_centroids(time, centroid_dist, tce, config, savefp, pxcoordinates=False):
    """ Creates and saves a figure with plots that show the centroid-to-target distance and, if desired, the fitted
    spline and the respective spline normalized centroid-to-target distance, for a given TCE.

    :param time: numpy array, time
    :param centroid_dist: numpy array, centroid-to-target distance
    :param tce: Pandas Series, row of the input TCE table Pandas DataFrame
    :param config: dict, preprocessing parameters
    :param savefp: Path, filepath to save figure
    :param pxcoordinates: bool, if True sets label to pixel instead of arcsec
    :return:
    """

    f, ax = plt.subplots(figsize=(16, 10))
    ax.scatter(time, centroid_dist, c='k', s=4)
    if config['px_coordinates'] or pxcoordinates:
        ax.set_ylabel('Pixel Distance [pixel]')
    else:
        ax.set_ylabel('Angular Distance [arcsec]')
    ax.set_title('Transit Source Distance to Target')
    ax.set_xlabel('Time [day]')
    ax.set_xlim(time[[0, -1]])

    f.suptitle(f'{tce.uid} {tce.label}')
    plt.savefig(savefp)
    plt.close()


def plot_centroids_views(glob_view_centr, loc_view_centr, tce, config, savedir, basename):
    """ Creates and saves a 2x1 figure with plots that show the global and local views for the centroid time-series for
     a given TCE.

    :param glob_view_centr: numpy array, global centroid view
    :param loc_view_centr: numpy array, local centroid view
    :param tce: pandas Series, row of the input TCE table Pandas DataFrame
    :param config: dict, preprocessing parameters
    :param savedir: str, filepath to directory in which the figure is saved
    :param basename: str, added to the figure filename
    :return:
    """

    f, ax = plt.subplots(2, 1, figsize=(12, 10))
    ax[0].plot(glob_view_centr)
    ax[0].set_ylabel('Amplitude')
    ax[0].set_title('Global view')
    ax[1].plot(loc_view_centr)
    ax[1].set_ylabel('Amplitude')
    ax[1].set_xlabel('Bin number')
    ax[1].set_title('Local view')

    f.suptitle('{} {}'.format(tce.uid, tce.label))
    plt.savefig(os.path.join(savedir, '{}_{}_{}.png'.format(tce.uid, tce.label, basename)))
    plt.close()


def plot_fluxandcentroids_views(glob_view, loc_view, glob_view_centr, loc_view_centr, tce, config, savedir, basename):
    """ Creates and saves a 2x2 figure with plots that show the global and local views for the flux and centroid
    time-series for a given TCE.

    :param glob_view: numpy array, global flux view
    :param loc_view: numpy array, local flux view
    :param glob_view_centr: numpy array, global centroid view
    :param loc_view_centr: numpy array, local centroid view
    :param tce: pandas Series, row of the input TCE table Pandas DataFrame
    :param config: dict, preprocessing parameters.
    :param savedir: str, filepath to directory in which the figure is saved
    :param basename: str, added to the figure filename
    :return:
    """

    f, ax = plt.subplots(2, 2, figsize=(12, 10))
    ax[0, 0].plot(glob_view)
    ax[0, 0].set_ylabel('Amplitude')
    ax[0, 0].set_title('Global view')
    ax[0, 1].plot(loc_view)
    ax[0, 1].set_title('Local view')
    ax[1, 0].plot(glob_view_centr)
    ax[1, 0].set_ylabel('Amplitude')
    ax[1, 0].set_xlabel('Bin number')
    ax[1, 1].plot(loc_view_centr)
    ax[1, 1].set_xlabel('Bin number')

    f.suptitle('{} {}'.format(tce.uid, tce.label))
    plt.savefig(os.path.join(savedir, '{}_{}_{}.png'.format(tce.uid, tce.label, basename)))
    plt.close()


def plot_all_views(views, tce, config, scheme, savefp, plot_var, draw_lines=False):
    """ Creates and saves a figure with plots that show views for a given TCE.

    :param views: dict, views to be plotted
    :param tce: pandas Series, row of the input TCE table Pandas DataFrame
    :param config: dict, preprocessing parameters.
    :param scheme: list, defines the number and position of the view plots in the figure ([number of plots per row,
        number of plots per column])
    :param savefp: Path, filepath to saved figure
    :param plot_var: bool, if True then dispersion-like time series (+- central tendency) are also plotted
    :return:
    """

    # global_phase = np.linspace(-tce['tce_period'] / 2, tce['tce_period'] / 2, config.num_bins_glob, endpoint=True)
    # local_phase = np.linspace(-tce['tce_duration'] * config.num_durations, tce['tce_duration'] * config.num_durations,
    #                           config.num_bins_loc, endpoint=True)

    scalarParamsStr = ''
    for scalarParam_i in range(len(config['scalar_params'])):
        if scalarParam_i % 7 == 0:
            scalarParamsStr += '\n'
        if config['scalar_params'][scalarParam_i] == 'sectors':
            scalarParamsStr += f'Sectors: {tce["sectors"]} \n'
        elif config['scalar_params'][scalarParam_i] in ['boot_fap']:
            scalarParamsStr += '{}={:.4E}  '.format(config['scalar_params'][scalarParam_i],
                                                    tce[config['scalar_params'][scalarParam_i]])
        elif config['scalar_params'][scalarParam_i] in ['tce_rb_tcount0', 'tce_steff']:
            scalarParamsStr += '{}={}  '.format(config['scalar_params'][scalarParam_i],
                                                tce[config['scalar_params'][scalarParam_i]])
        else:
            scalarParamsStr += '{}={:.4f}  '.format(config['scalar_params'][scalarParam_i],
                                                    tce[config['scalar_params'][scalarParam_i]])

    ephemerisStr = 'Epoch={:.4f}, Period={:.4f}, Transit Duration={:.4f}'.format(
        tce['tce_time0bk'],
        tce['tce_period'],
        tce['tce_duration'] * 24
    )

    f, ax = plt.subplots(scheme[0], scheme[1], figsize=(20, 10))
    k = 0
    views_list = list(views.keys())
    for i in range(scheme[0]):
        for j in range(scheme[1]):
            if k < len(views_list):
                # ax[i, j].plot(views[views_list[k]][1], zorder=2, color='k')
                # ax[i, j].scatter(np.arange(len(views[views_list[k]][1])), views[views_list[k]][1], s=10, color='k',
                #                  zorder=2)

                if draw_lines:
                    ax[i, j].plot(views[views_list[k]][0], views[views_list[k]][1], zorder=2, color='k')
                ax[i, j].scatter(views[views_list[k]][0], views[views_list[k]][1], s=10, color='k',
                                 zorder=2)
                if plot_var:
                    # ax[i, j].plot(views[views_list[k]][1] + views[views_list[k]][2], 'r--', alpha=0.7, zorder=1)
                    # ax[i, j].plot(views[views_list[k]][1] - views[views_list[k]][2], 'r--', alpha=0.7, zorder=1)

                    ax[i, j].plot(views[views_list[k]][0], views[views_list[k]][1] + views[views_list[k]][2], 'r--',
                                  alpha=0.3, zorder=1)
                    ax[i, j].plot(views[views_list[k]][0], views[views_list[k]][1] - views[views_list[k]][2], 'r--',
                                  alpha=0.3, zorder=1)

                ax[i, j].set_title(f'{views_list[k]} num transits={views[views_list[k]][3]}', pad=20)

                # ax[i, j].set_xlim([0, len(views[views_list[k]])])
                ax[i, j].set_xlim(views[views_list[k]][0][[0, -1]])
            if i == scheme[0] - 1:
                # ax[i, j].set_xlabel('Bin Number')
                ax[i, j].set_xlabel('Phase [day]')
            if j == 0:
                ax[i, j].set_ylabel('Amplitude')

            k += 1

    f.suptitle(f'{tce["uid"]} {tce["label"] if tce["label"] != "UNK" else ""} | {ephemerisStr}\n{scalarParamsStr}')
    plt.subplots_adjust(hspace=0.5, wspace=0.37, top=0.83, right=0.974, bottom=0.07, left=0.05)
    if plot_var:
        f.text(0.974, 0.97, 'Red dashed lines: ±1σ uncertainty envelope',
               ha='right', va='top', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='white', edgecolor='red', alpha=0.8))
    plt.savefig(savefp)
    plt.close()


def plot_all_views_var(views, views_var, tce, config, scheme, savedir, basename, num_transits):
    """ Creates and saves a figure with plots that show views for a given TCE.

    :param views: dict, views to be plotted
    :param tce: pandas Series, row of the input TCE table Pandas DataFrame
    :param config: dict, preprocessing parameters.
    :param scheme: list, defines the number and position of the view plots in the figure ([number of plots per row,
        number of plots per column])
    :param savedir: str, filepath to directory in which the figure is saved
    :param basename: str, added to the figure filename
    :param num_transits: dict, number of transits for each view
    :return:
    """

    scalarParamsStr = ''
    for scalarParam_i in range(len(config['scalar_params'])):
        if scalarParam_i % 7 == 0:
            scalarParamsStr += '\n'
        if config['scalar_params'][scalarParam_i] == 'sectors':
            scalarParamsStr += f'Sectors: {tce["sectors"]} \n'
        elif config['scalar_params'][scalarParam_i] in ['boot_fap']:
            scalarParamsStr += '{}={:.4E}  '.format(config['scalar_params'][scalarParam_i],
                                                    tce[config['scalar_params'][scalarParam_i]])
        elif config['scalar_params'][scalarParam_i] in ['tce_rb_tcount0', 'tce_steff']:
            scalarParamsStr += '{}={}  '.format(config['scalar_params'][scalarParam_i],
                                                tce[config['scalar_params'][scalarParam_i]])
        else:
            scalarParamsStr += '{}={:.4f}  '.format(config['scalar_params'][scalarParam_i],
                                                    tce[config['scalar_params'][scalarParam_i]])

    ephemerisStr = 'Epoch={:.4f}, Period={:.4f}, Transit Duration={:.4f}, Transit Depth={:.4f}'.format(
        tce['tce_time0bk'],
        tce['tce_period'],
        tce['tce_duration'] * 24,
        tce['tce_depth'],
        # tce['transit_depth'],
    )

    f, ax = plt.subplots(scheme[0], scheme[1], figsize=(20, 10))
    k = 0
    views_list = list(views.keys())
    for i in range(scheme[0]):
        for j in range(scheme[1]):
            if k < len(views_list):
                ax[i, j].plot(views[views_list[k]])
                ax[i, j].plot(views[views_list[k]] + views_var[views_list[k]], 'r--')
                ax[i, j].plot(views[views_list[k]] - views_var[views_list[k]], 'r--')
                ax[i, j].scatter(np.arange(len(views[views_list[k]])), views[views_list[k]], s=10, color='k', alpha=0.2)
                if views_list[k] == 'global_flux_view':
                    ax[i, j].set_title('{} N_transits={}'.format(views_list[k], num_transits['flux']), pad=20)
                elif views_list[k] == 'local_flux_odd_view':
                    ax[i, j].set_title('{} N_transits={}'.format(views_list[k], num_transits['flux_odd']), pad=20)
                elif views_list[k] == 'local_flux_even_view':
                    ax[i, j].set_title('{} N_transits={}'.format(views_list[k], num_transits['flux_even']),
                                       pad=20)
                elif views_list[k] == 'local_weak_secondary_view':
                    ax[i, j].set_title('{} N_transits={}'.format(views_list[k], num_transits['wks']), pad=20)
                elif views_list[k] == 'global_centr_view':
                    ax[i, j].set_title('{} N_transits={}'.format(views_list[k], num_transits['centroid']),
                                       pad=20)
                elif views_list[k] == 'global_centr_fdl_view':
                    ax[i, j].set_title('{} N_transits={}'.format(views_list[k], num_transits['centroid_fdl']),
                                       pad=20)
                else:
                    ax[i, j].set_title('{}'.format(views_list[k]), pad=20)
                ax[i, j].set_xlim([0, len(views[views_list[k]])])
            if i == scheme[0] - 1:
                ax[i, j].set_xlabel('Bin number')
            if j == 0:
                ax[i, j].set_ylabel('Amplitude')

            k += 1

    f.suptitle('{} {} | {}\n{}'.format(tce.uid, tce.label, ephemerisStr, scalarParamsStr))
    plt.subplots_adjust(hspace=0.5, wspace=0.37, top=0.83, right=0.974, bottom=0.07, left=0.05)
    plt.savefig(os.path.join(savedir, '{}_{}_{}.png'.format(tce.uid, tce.label, basename)))
    plt.close()


def plot_view_exominer_pipeline(views, tce, config, scheme, savefp, plot_var, draw_lines=False):
    """ Creates and saves a figure with plots that show views for a given TCE.

    :param views: dict, views to be plotted
    :param tce: pandas Series, row of the input TCE table Pandas DataFrame
    :param config: dict, preprocessing parameters.
    :param scheme: list, defines the number and position of the view plots in the figure ([number of plots per row,
        number of plots per column])
    :param savefp: Path, filepath to saved figure
    :param plot_var: bool, if True then dispersion-like time series (+- central tendency) are also plotted
    :return:
    """

    # exclude normalized views
    views_to_plot = [view_name for view_name in views if 'norm' not in view_name]
    views_names = {
        'flux_local': 'Transit-view flux',
        'flux_global': 'Full-orbit flux',
        # 'flux_trend_local': 'Transit-view trend',
        'flux_trend_global': 'Full-orbit trend',
        'flux_odd_local': 'Transit-view odd flux',
        'flux_even_local': 'Transit-view even flux',
        'flux_weak_secondary_local': 'Transit-view weak secondary flux',
        'centroid_offset_distance_to_target_global': 'Full-orbit centroid offset to target',
        'centroid_offset_distance_to_target_local': 'Transit-view centroid offset to target',
        'momentum_dump_local': 'Transit-view momentum dump',
    }

    views_units = {
        'flux_local': 'Relative Flux',
        'flux_global': 'Relative Flux',
        # 'flux_trend_local': 'Transit-view trend',
        'flux_trend_global': 'Relative Flux',
        'flux_odd_local': 'Relative Flux',
        'flux_even_local': 'Relative Flux',
        'flux_weak_secondary_local': 'Relative Flux',
        'centroid_offset_distance_to_target_global': 'Centroid Offset [arcsec]',
        'centroid_offset_distance_to_target_local': 'Centroid Offset [arcsec]',
        'momentum_dump_local': 'Median Mom. Dump Flag',
    }

    scalarParamsStr = ''
    for scalarParam_i in range(len(config['scalar_params'])):
        if scalarParam_i % 7 == 0 and scalarParam_i != 0:
            scalarParamsStr += '\n'
        if config['scalar_params'][scalarParam_i] == 'sectors':
            scalarParamsStr += f'Sectors: {tce["sectors"]} \n'
        elif config['scalar_params'][scalarParam_i] in ['boot_fap']:
            scalarParamsStr += '{}={:.3E}  '.format(config['scalar_params'][scalarParam_i],
                                                    tce[config['scalar_params'][scalarParam_i]])
        elif config['scalar_params'][scalarParam_i] in ['tce_rb_tcount0', 'tce_steff']:
            scalarParamsStr += '{}={}  '.format(config['scalar_params'][scalarParam_i],
                                                tce[config['scalar_params'][scalarParam_i]])
        else:
            scalarParamsStr += '{}={:.3f}  '.format(config['scalar_params'][scalarParam_i],
                                                    tce[config['scalar_params'][scalarParam_i]])

    ephemerisStr = 'Epoch (BTJD)={:.3f}, Period (day)={:.3f}, Transit Duration (hour)={:.3f}'.format(
        tce['tce_time0bk'],
        tce['tce_period'],
        tce['tce_duration'] * 24
    )

    f, ax = plt.subplots(scheme[0], scheme[1], figsize=(20, 14))
    k = 0
    for i in range(scheme[0]):
        for j in range(scheme[1]):
            if k < len(views_to_plot):
                if draw_lines:
                    ax[i, j].plot(views[views_to_plot[k]][0], views[views_to_plot[k]][1], zorder=2, color='k')
                ax[i, j].scatter(views[views_to_plot[k]][0], views[views_to_plot[k]][1], s=10, color='k',
                                 zorder=2)
                if plot_var:
                    ax[i, j].plot(views[views_to_plot[k]][0], views[views_to_plot[k]][1] + views[views_to_plot[k]][2], 'r--',
                                  alpha=0.1, zorder=1)
                    ax[i, j].plot(views[views_to_plot[k]][0], views[views_to_plot[k]][1] - views[views_to_plot[k]][2], 'r--',
                                  alpha=0.1, zorder=1)

                if views_to_plot[k] == 'momentum_dump_local':
                    ax[i, j].set_title(f'{views_names[views_to_plot[k]]}', pad=20)
                else:
                    ax[i, j].set_title(f'{views_names[views_to_plot[k]]}\nN_transits={views[views_to_plot[k]][3]}', pad=20)

                ax[i, j].set_xlim(views[views_to_plot[k]][0][[0, -1]])
            if i == scheme[0] - 1:
                ax[i, j].set_xlabel('Phase [day]')
            # if j == 0:
            #     ax[i, j].set_ylabel('Amplitude')
            ax[i, j].set_ylabel(views_units[views_to_plot[k]])

            k += 1

    f.suptitle(r'$\mathbf{Phase-folded\ and\ binned\ flux\ and\ centroid\ views}$' + '\n' + f'==== TCE {tce["uid"]} {tce["label"] if tce["label"] != "UNK" else ""} ====' + \
               '\n' + f'Ephemerides: {ephemerisStr}' + '\n' + '==== TCE and stellar parameters ====' + '\n' + f'{scalarParamsStr}')
    plt.subplots_adjust(hspace=0.5, wspace=0.37, top=0.75, right=0.974, bottom=0.07, left=0.05)
    if plot_var:
        f.text(0.974, 0.97, 'Red dashed lines: ±1σ SEM envelope',
               ha='right', va='top', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='white', edgecolor='red', alpha=0.8))
    plt.savefig(savefp)
    plt.close()


def plot_wks(glob_view, glob_view_weak_secondary, tce, savedir, basename):
    """ Creates and saves a figure with plots of the global flux view and global weak secondary flux view for a given
    TCE.

    :param glob_view: NumPy array, global flux view
    :param glob_view_weak_secondary: NumPy array, global weak secondary flux view
    :param tce: pandas Series, row of the input TCE table Pandas DataFrame
    :param savedir: str, filepath to directory in which the figure is saved
    :param basename: str, added to the figure filename
    :return:
    """

    f, ax = plt.subplots()
    ax.plot(glob_view, color='b', label='primary')
    ax.plot(glob_view_weak_secondary, 'r--', label='secondary')
    ax.set_ylabel('Normalized amplitude')
    ax.set_xlabel('Bins')
    ax.legend()
    ax.set_title('{} {}'.format(tce.uid, tce.label))
    plt.savefig(os.path.join(savedir, '{}_{}_{}.png'.format(tce.uid, tce.label, basename)))
    plt.close()


def plot_all_phasefoldedtimeseries(timeseries, tce, scheme, savefp, timeseries_outliers=None):
    """ Creates and saves a figure with plots that show phase folded timeseries for a given TCE.

    :param timeseries: dict, views to be plotted
    :param tce: Pandas Series, row of the input TCE table Pandas DataFrame
    :param scheme: list, defines the number and position of the view plots in the figure ([number of plots per row,
    number of plots per column])
    :param savefp: Path, filepath to figure
    :param timeseries_outliers: dict, outliers of the time series. If it is not None, these outliers are plotted on top
     of the time series
    :return:
    """

    # SIGMA_FACTOR = 6

    f, ax = plt.subplots(scheme[0], scheme[1], figsize=(18, 10))
    k = 0
    views_list = list(timeseries.keys())
    for i in range(scheme[0]):
        for j in range(scheme[1]):
            if k < len(views_list):
                if len(timeseries[views_list[k]][0]) > 0:
                    ax[i, j].scatter(timeseries[views_list[k]][0], timeseries[views_list[k]][1], zorder=1, c='k', s=5)
                    if timeseries_outliers is not None and views_list[k] in timeseries_outliers:
                        ax[i, j].scatter(timeseries_outliers[views_list[k]][0], timeseries_outliers[views_list[k]][1],
                                         c='r', s=5, zorder=2)
                    ax[i, j].set_title(views_list[k], pad=20)
                    ax[i, j].set_xlim([timeseries[views_list[k]][0][0], timeseries[views_list[k]][0][-1]])
                    # timeseries_madstd, timeseries_med = mad_std(timeseries[views_list[k]][1], ignore_nan=True), \
                    #                                     np.nanmedian(timeseries[views_list[k]][1])
                    # std_range = SIGMA_FACTOR * timeseries_madstd
                    # range_timeseries = [timeseries_med - std_range, timeseries_med + std_range]
                    # ax[i, j].set_ylim(range_timeseries)
                    if 'FDL' in views_list[k]:
                        ax[i, j].set_ylim(bottom=0)
            if i == scheme[0] - 1:
                ax[i, j].set_xlabel('Phase [day]')
            if j == 0:
                ax[i, j].set_ylabel('Amplitude')
            k += 1

    # f.subplots_adjust(left=0.055, right=0.979, bottom=0.071, top=0.917, wspace=0.2, hspace=0.383)
    f.suptitle(f'{tce["uid"]} {tce["label"]}')
    f.tight_layout()
    plt.savefig(savefp)
    plt.close()


def plot_diff_oddeven(timeseries, tce, savedir, basename):
    """ Creates and saves a figure with plots that show the absolute difference between odd and even views.

    :param timeseries: dict, views to be plotted
    :param tce: pandas Series, row of the input TCE table Pandas DataFrame
    :param savedir: str, filepath to directory in which the figure is saved
    :param basename: str, added to the figure filename
    :return:
    """

    f, ax = plt.subplots(2, 1)
    ax[0].plot(np.abs(timeseries['global_flux_odd_view'] - timeseries['global_flux_even_view']))
    ax[0].set_title('Global odd-even views')
    ax[0].set_ylabel('Amplitude')
    ax[1].plot(np.abs(timeseries['local_flux_odd_view'] - timeseries['local_flux_even_view']))
    ax[1].set_title('Local odd-even views')
    ax[1].set_xlabel('Bin Number')
    ax[1].set_ylabel('Amplitude')

    f.suptitle('{} {}'.format(tce.uid, tce.label))
    plt.savefig(os.path.join(savedir, '{}_{}_{}.png'.format(tce.uid, tce.label, basename)))
    plt.close()


def plot_phasefolded_and_binned(timeseries, binned_timeseries, tce, config, savefp):
    """ Creates and saves a figure with plots that show phase folded and binned time series for a given TCE.

    :param timeseries: dict, phase folded time series
    :param binned_timeseries: dict, binned views
    :param tce: Pandas Series, row of the input TCE table Pandas DataFrame
    :param config: dict, preprocessing parameters.
    :param savefp: Path, filepath for saved figure
    :return:
    """

    # SIGMA_FACTOR = 6
    local_view_time_interval = tce['tce_duration'] * (config['num_durations'])

    gs = gridspec.GridSpec(6, 2)

    f = plt.figure(figsize=(20, 14))

    ax = plt.subplot(gs[0, :])
    ax.scatter(timeseries['flux'][0], timeseries['flux'][1], color='k', s=5, alpha=0.1)
    ax.scatter(binned_timeseries['flux_global'][0], binned_timeseries['flux_global'][1], color='b')
    ax.plot(binned_timeseries['flux_global'][0], binned_timeseries['flux_global'][1], 'b')
    ax.set_ylabel('Relative Flux')
    ax.set_xlabel('Phase (day)')
    # ax.set_xlim([timeseries['Flux'][0][0], timeseries['Flux'][0][-1]])
    ax.set_xlim([- tce['tce_period'] / 2, tce['tce_period'] / 2])
    # timeseries_madstd, timeseries_med = mad_std(timeseries['flux'][1]), np.median(timeseries['Flux'][1])
    # std_range = SIGMA_FACTOR * timeseries_madstd
    # ts_len = len(timeseries['flux'][1])
    # idxs_transit = np.arange(ts_len)[int(ts_len // 2 - ts_len // config['num_durations']):int(
    #     ts_len // 2 + ts_len // config['num_durations'])]
    # min_val = min(timeseries['flux'][1][idxs_transit])
    # range_timeseries = [min_val, timeseries_med + std_range]
    # ax.set_ylim(range_timeseries)
    ax.set_title('Flux')

    # left_idx = np.where(timeseries['Flux'][0] > -local_view_time_interval)[0][0]
    # right_idx = np.where(timeseries['Flux'][0] < local_view_time_interval)[0][-1]
    ax = plt.subplot(gs[1, 0])
    # ax.scatter(timeseries['Flux'][0][left_idx:right_idx] * 24, timeseries['Flux'][1][left_idx:right_idx],
    #            color='k', s=5)
    ax.scatter(timeseries['flux'][0] * 24, timeseries['flux'][1], color='k', s=5, alpha=0.1)
    ax.scatter(binned_timeseries['flux_local'][0] * 24, binned_timeseries['flux_local'][1], color='b')
    ax.plot(binned_timeseries['flux_local'][0] * 24, binned_timeseries['flux_local'][1], 'b')
    ax.set_ylabel('Relative Flux')
    ax.set_xlabel('Phase [hour]')
    # ax.set_xlim([timeseries['Flux'][0][left_idx] * 24, timeseries['Flux'][0][right_idx] * 24])
    ax.set_xlim([- local_view_time_interval * 24, local_view_time_interval * 24])
    # timeseries_madstd, timeseries_med = mad_std(timeseries['flux'][1]), np.median(timeseries['Flux'][1])
    # std_range = SIGMA_FACTOR * timeseries_madstd
    # ts_len = len(timeseries['flux'][1])
    # idxs_transit = np.arange(ts_len)[int(ts_len // 2 - ts_len // config['num_durations']):int(
    #     ts_len // 2 + ts_len // config['num_durations'])]
    # min_val = min(timeseries['flux'][1][idxs_transit])
    # range_timeseries = [min_val, timeseries_med + std_range]
    # ax.set_ylim(range_timeseries)

    if 'flux_weak_secondary' in timeseries:
        # left_idx = np.where(timeseries['Weak Secondary Flux'][0] > -local_view_time_interval)[0][0]
        # right_idx = np.where(timeseries['Weak Secondary Flux'][0] < local_view_time_interval)[0][-1]
        ax = plt.subplot(gs[1, 1])
        # ax.scatter(timeseries['Weak Secondary Flux'][0][left_idx:right_idx] * 24,
        #            timeseries['Weak Secondary Flux'][1][left_idx:right_idx], color='k', s=5)
        ax.scatter(timeseries['flux_weak_secondary'][0] * 24,
                   timeseries['flux_weak_secondary'][1], color='k', s=5, alpha=0.1)
        ax.scatter(binned_timeseries['flux_weak_secondary_local'][0] * 24,
                   binned_timeseries['flux_weak_secondary_local'][1], color='b')
        ax.plot(binned_timeseries['flux_weak_secondary_local'][0] * 24,
                binned_timeseries['flux_weak_secondary_local'][1], 'b')
        ax.set_ylabel('Relative Flux')
        ax.set_xlabel('Phase [hour]')
        # ax.set_xlim([timeseries['Weak Secondary Flux'][0][left_idx] * 24,
        #              timeseries['Weak Secondary Flux'][0][right_idx] * 24])
        ax.set_xlim([- local_view_time_interval * 24, local_view_time_interval * 24])
        ax.set_title('Weak Secondary Phase : {:.4f} Days'.format(tce['tce_maxmesd']))
        # timeseries_madstd, timeseries_med = mad_std(timeseries['weak_secondary_flux'][1]), \
        #                                     np.median(timeseries['weak_secondary_flux'][1])
        # std_range = SIGMA_FACTOR * timeseries_madstd
        # ts_len = len(timeseries['weak_secondary_flux'][1])
        # idxs_transit = np.arange(ts_len)[int(ts_len // 2 - ts_len // config['num_durations']):int(
        #     ts_len // 2 + ts_len // config['num_durations'])]
        # min_val = min(timeseries['weak_secondary_flux'][1][idxs_transit])
        # range_timeseries = [min_val, timeseries_med + std_range]
        # ax.set_ylim(range_timeseries)

    ax = plt.subplot(gs[2, 0])
    if len(timeseries['flux_odd'][0]) > 0:
        # left_idx = np.where(timeseries['Odd Flux'][0] > -local_view_time_interval)[0][0]
        # right_idx = np.where(timeseries['Odd Flux'][0] < local_view_time_interval)[0][-1]
        # ax.scatter(timeseries['Odd Flux'][0][left_idx:right_idx] * 24, timeseries['Odd Flux'][1][left_idx:right_idx],
        #            color='k', s=5)
        ax.scatter(timeseries['flux_odd'][0] * 24, timeseries['flux_odd'][1], color='k', s=5, alpha=0.1)
        ax.scatter(binned_timeseries['flux_odd_local'][0] * 24, binned_timeseries['flux_odd_local'][1], color='b')
        ax.plot(binned_timeseries['flux_odd_local'][0] * 24, binned_timeseries['flux_odd_local'][1], 'b')
        ax.set_ylabel('Relative Flux')
        ax.set_xlabel('Phase [hour]')
        # ax.set_xlim([timeseries['Odd Flux'][0][left_idx] * 24, timeseries['Odd Flux'][0][right_idx] * 24])
        ax.set_xlim([- local_view_time_interval * 24, local_view_time_interval * 24])
        ax.set_title('Odd')
        timeseries_madstd, timeseries_med = mad_std(timeseries['flux_odd'][1]), \
                                            np.median(timeseries['flux_odd'][1])
        # std_range = SIGMA_FACTOR * timeseries_madstd
        # ts_len = len(timeseries['odd_flux'][1])
        # idxs_transit = np.arange(ts_len)[int(ts_len // 2 - ts_len // config['num_durations']):int(
        #     ts_len // 2 + ts_len // config['num_durations'])]
        # min_val = min(timeseries['odd_flux'][1][idxs_transit])
        # range_timeseries = [min_val, timeseries_med + std_range]
        # ax.set_ylim(range_timeseries)

    ax = plt.subplot(gs[2, 1])
    if len(timeseries['flux_even'][0]) > 0:
        # left_idx = np.where(timeseries['Even Flux'][0] > -local_view_time_interval)[0][0]
        # right_idx = np.where(timeseries['Even Flux'][0] < local_view_time_interval)[0][-1]
        # ax.scatter(timeseries['Even Flux'][0][left_idx:right_idx] * 24, timeseries['Even Flux'][1][left_idx:right_idx],
        #            color='k', s=5)
        ax.scatter(timeseries['flux_even'][0] * 24, timeseries['flux_even'][1], color='k', s=5, alpha=0.1)
        ax.scatter(binned_timeseries['flux_even_local'][0] * 24, binned_timeseries['flux_even_local'][1], color='b')
        ax.plot(binned_timeseries['flux_even_local'][0] * 24, binned_timeseries['flux_even_local'][1], 'b')
        ax.set_ylabel('Relative Flux')
        ax.set_xlabel('Phase [hour]')
        # ax.set_xlim([timeseries['Even Flux'][0][left_idx] * 24, timeseries['Even Flux'][0][right_idx] * 24])
        ax.set_xlim([- local_view_time_interval * 24, local_view_time_interval * 24])
        ax.set_title('Even')
        # timeseries_madstd, timeseries_med = mad_std(timeseries['even_flux'][1]), \
        #                                     np.median(timeseries['even_flux'][1])
        # std_range = SIGMA_FACTOR * timeseries_madstd
        # ts_len = len(timeseries['even_flux'][1])
        # idxs_transit = np.arange(ts_len)[int(ts_len // 2 - ts_len // config['num_durations']):int(
        #     ts_len // 2 + ts_len // config['num_durations'])]
        # min_val = min(timeseries['Even Flux'][1][idxs_transit])
        # range_timeseries = [min_val, timeseries_med + std_range]
        # ax.set_ylim(range_timeseries)

    ax = plt.subplot(gs[3, :])
    ax.scatter(timeseries['flux_trend'][0], timeseries['flux_trend'][1], color='k', s=5, alpha=0.1)
    ax.scatter(binned_timeseries['flux_trend_global'][0], binned_timeseries['flux_trend_global'][1], color='b')
    ax.plot(binned_timeseries['flux_trend_global'][0], binned_timeseries['flux_trend_global'][1], 'b')
    ax.set_ylabel('Relative Flux')
    ax.set_xlabel('Phase [day]')
    # ax.set_xlim([timeseries['Flux'][0][0], timeseries['Flux'][0][-1]])
    ax.set_xlim([- tce['tce_period'] / 2, tce['tce_period'] / 2])
    ax.set_title('Flux Trend')

    ax = plt.subplot(gs[4, :])
    ax.scatter(timeseries['centroid_offset_distance_to_target'][0],
               timeseries['centroid_offset_distance_to_target'][1], color='k', s=5, alpha=0.1)
    ax.scatter(binned_timeseries['centroid_offset_distance_to_target_global'][0],
               binned_timeseries['centroid_offset_distance_to_target_global'][1], color='b')
    ax.plot(binned_timeseries['centroid_offset_distance_to_target_global'][0],
            binned_timeseries['centroid_offset_distance_to_target_global'][1], 'b')
    ax.set_ylabel('Offset distance [arcsec]')
    ax.set_xlabel('Phase [day]')
    # ax.set_xlim([timeseries['Centroid Offset Distance'][0][0],
    #              timeseries['Centroid Offset Distance'][0][-1]])
    ax.set_xlim([- tce['tce_period'] / 2, tce['tce_period'] / 2])
    # timeseries_madstd, timeseries_med = mad_std(timeseries['Centroid Offset Distance'][1]), \
    #                                     np.median(timeseries['Centroid Offset Distance'][1])
    # std_range = SIGMA_FACTOR * timeseries_madstd
    # ts_len = len(timeseries['Centroid Offset Distance'][1])
    # idxs_transit = np.arange(ts_len)[int(ts_len // 2 - ts_len // config['num_durations']):int(
    #     ts_len // 2 + ts_len // config['num_durations'])]
    # min_val = min(timeseries['Centroid Offset Distance'][1][idxs_transit])
    # max_val = max(timeseries['Centroid Offset Distance'][1][idxs_transit])
    # range_timeseries = [min_val, max_val]
    # ax.set_ylim(range_timeseries)
    ax.set_title('Flux-weighted Centroid Motion')

    # left_idx = np.where(timeseries['Centroid Offset Distance'][0] > -local_view_time_interval)[0][0]
    # right_idx = np.where(timeseries['Centroid Offset Distance'][0] < local_view_time_interval)[0][-1]
    ax = plt.subplot(gs[5, 0])
    # ax.scatter(timeseries['Centroid Offset Distance'][0][left_idx:right_idx] * 24,
    #         timeseries['Centroid Offset Distance'][1][left_idx:right_idx],
    #            color='k', s=5)
    ax.scatter(timeseries['centroid_offset_distance_to_target'][0] * 24,
               timeseries['centroid_offset_distance_to_target'][1],
               color='k', s=5, alpha=0.1)
    ax.scatter(binned_timeseries['centroid_offset_distance_to_target_local'][0] * 24,
               binned_timeseries['centroid_offset_distance_to_target_local'][1], color='b')
    ax.plot(binned_timeseries['centroid_offset_distance_to_target_local'][0] * 24,
            binned_timeseries['centroid_offset_distance_to_target_local'][1], 'b')
    ax.set_ylabel('Offset distance [arcsec]')
    ax.set_xlabel('Phase [hour]')
    # ax.set_xlim([timeseries['Centroid Offset Distance'][0][left_idx] * 24,
    #              timeseries['Centroid Offset Distance'][0][right_idx] * 24])
    ax.set_xlim([- local_view_time_interval * 24, local_view_time_interval * 24])
    # timeseries_madstd, timeseries_med = mad_std(timeseries['Centroid Offset Distance'][1]), \
    #                                     np.median(timeseries['Centroid Offset Distance'][1])
    # std_range = SIGMA_FACTOR * timeseries_madstd
    # range_timeseries = [timeseries_med - std_range, timeseries_med + std_range]
    # ts_len = len(timeseries['Centroid Offset Distance'][1])
    # idxs_transit = np.arange(ts_len)[int(ts_len // 2 - ts_len // config['num_durations']):int(
    #     ts_len // 2 + ts_len // config['num_durations'])]
    # min_val = min(timeseries['Centroid Offset Distance'][1][idxs_transit])
    # max_val = max(timeseries['Centroid Offset Distance'][1][idxs_transit])
    # range_timeseries = [min_val, max_val]
    # ax.set_ylim(range_timeseries)

    # plt.subplots_adjust(
    #     hspace=0.526,
    #     wspace=0.202,
    #     top=0.943,
    #     bottom=0.06,
    #     left=0.057,
    #     right=0.98
    # )
    f.suptitle(f'{tce["uid"]} {tce["label"]}')
    f.tight_layout()
    plt.savefig(savefp)
    plt.close()


def plot_odd_even(binned_timeseries, phasefolded_timeseries, tce, config, savefp,
                  exclusion_prim_factor=1.5, outlier_sigma_high=5, outlier_sigma_low=3):
    """ Creates and saves a figure with plots for odd-even transit depth test for a given TCE.

    :param binned_timeseries: dict, binned views
    :param phasefolded_timeseries: dict, phase-folded time series
    :param tce: Pandas Series, row of the input TCE table Pandas DataFrame
    :param config: dict, preprocessing parameters.
    :param savefp: Path, filepath
    :param exclusion_prim_factor: float, factor by which to exclude primary transit from the calculation of the baseline.
    :param outlier_sigma_high: float, sigma threshold above which a point is considered an outlier.
    :param outliter_sigma_low: float, sigma threshold below which a point is considered an outlier.
    :return:
    """

    local_view_time_interval = tce['tce_duration'] * (config['num_durations'])

    # --- 1. Calculate robust Y-axis limits using MAD ---
    x_odd = phasefolded_timeseries['flux_odd'][0]
    y_odd = phasefolded_timeseries['flux_odd'][1]

    x_even = phasefolded_timeseries['flux_even'][0]
    y_even = phasefolded_timeseries['flux_even'][1]

    # Extract binned data early for limit calculation
    x_even_binned = binned_timeseries['flux_even_local'][0]
    y_even_binned = binned_timeseries['flux_even_local'][1]
    err_even_binned = binned_timeseries['flux_even_local'][2]

    x_odd_binned = binned_timeseries['flux_odd_local'][0]
    y_odd_binned = binned_timeseries['flux_odd_local'][1]
    err_odd_binned = binned_timeseries['flux_odd_local'][2]

    # Combine out-of-transit data to calculate the true baseline scatter
    duration_days = tce['tce_duration']

    mask_odd = np.abs(x_odd) > (exclusion_prim_factor * duration_days)
    mask_even = np.abs(x_even) > (exclusion_prim_factor * duration_days)

    # Also create masks for binned data
    mask_odd_binned = np.abs(x_odd_binned) > (exclusion_prim_factor * duration_days)
    mask_even_binned = np.abs(x_even_binned) > (exclusion_prim_factor * duration_days)

    # Include BOTH unbinned and binned out-of-transit data
    y_valid_combined = np.concatenate([
        y_odd[mask_odd], 
        y_even[mask_even],
        y_odd_binned[mask_odd_binned],
        y_even_binned[mask_even_binned]
    ])

    if len(y_valid_combined) > 0:
        # Calculate robust scatter using Median Absolute Deviation (MAD)
        median_flux = np.nanmedian(y_valid_combined)
        mad_flux = np.nanmedian(np.abs(y_valid_combined - median_flux))
        robust_std = mad_flux * 1.4826
        
        # Upper Bound: Consider binned data + error envelopes
        y_high_candidates = [
            median_flux + (outlier_sigma_high * robust_std),
            np.nanmax(y_even_binned + err_even_binned),
            np.nanmax(y_odd_binned + err_odd_binned)
        ]
        y_high = max(y_high_candidates)
        
        # Lower Bound: Consider binned data - error envelopes
        binned_min_odd = np.nanmin(y_odd_binned - err_odd_binned)
        binned_min_even = np.nanmin(y_even_binned - err_even_binned)
        binned_min = min(binned_min_odd, binned_min_even)
        
        y_low = binned_min - (outlier_sigma_low * robust_std)
        
        # # Fallback safety
        # y_low = max(y_low, median_flux - (10 * robust_std))
        
        ylim_bounds = [y_low, y_high]
    else:
        ylim_bounds = None

    # --- 2. Create the Plot ---
    f = plt.figure(figsize=(20, 14))
    ax = plt.subplot()

    # Scatter unbinned phase-folded data
    ax.scatter(x_odd * 24, y_odd, color='k', s=5, alpha=0.1, zorder=1)
    ax.scatter(x_even * 24, y_even, color='k', s=5, alpha=0.1, zorder=1)

    # Plot binned data (already extracted above, just convert to hours)
    # even_flux_local (binned)
    ax.scatter(x_even_binned * 24, y_even_binned, color='r', zorder=3)
    ax.plot(x_even_binned * 24, y_even_binned, 'r', label='Even', zorder=3)
    ax.plot(x_even_binned * 24, y_even_binned + err_even_binned, 'r--', alpha=0.3, zorder=2)
    ax.plot(x_even_binned * 24, y_even_binned - err_even_binned, 'r--', alpha=0.3, zorder=2)

    # odd_flux_local (binned)
    ax.scatter(x_odd_binned * 24, y_odd_binned, color='g', zorder=3)
    ax.plot(x_odd_binned * 24, y_odd_binned, 'g', label='Odd', zorder=3 )
    ax.plot(x_odd_binned * 24, y_odd_binned + err_odd_binned, 'g--', alpha=0.3, zorder=2)
    ax.plot(x_odd_binned * 24, y_odd_binned - err_odd_binned, 'g--', alpha=0.3, zorder=2)

    # add a dummy line specifically for the custom legend entry.
    ax.plot([], [], color='gray', linestyle='--', label=r'$\pm 1 \sigma$ SEM envelope')

    ax.legend()
    ax.set_ylabel('Relative Flux')
    ax.set_xlabel('Phase [hour]')
    
    # Apply X and Y limits
    ax.set_xlim([-local_view_time_interval * 24, local_view_time_interval * 24])
    
    if ylim_bounds:
        ax.set_ylim(ylim_bounds)

    f.suptitle(f'Odd vs Even TCE {tce["uid"]}') 
    plt.savefig(savefp)
    plt.close()


def plot_residual(time, res_timeseries, tce, savedir, basename):
    """ Creates and saves a figure with plot for the residual timeseries after detrending
    (i.e., res = time_series - trend) a given TCE.

    :param time: numpy array, timestamps
    :param res_timeseries: numpy array, residual timeseries
    :param tce: Pandas Series, row of the input TCE table Pandas DataFrame
    :param savedir: str, filepath to directory in which the figure is saved
    :param basename: Path, added to the figure filename
    :return:
    """

    f, ax = plt.subplots(figsize=(16, 6))
    ax.scatter(time, res_timeseries, c='k', s=4)
    ax.set_ylabel('Amplitude')
    ax.set_xlabel('Time [day]')
    ax.set_xlim([time[0], time[-1]])
    ax.set_title('Residual')
    plt.subplots_adjust(left=0.048, right=0.983)
    f.suptitle(f'{tce.uid} {tce.label}')
    plt.savefig(savedir / f'{tce.uid}_{tce.label}_{basename}.png')
    plt.close()


def plot_riverplot(binned_fluxes, n_bins, tce, savefp):
    """ Plot riverplot from a set of binned flux phases.

    :param binned_fluxes: list, each element is a NumPy array for a phase of flux
    :param n_bins: int, number of bins
    :param tce: Pandas Series, row of the input TCE table Pandas DataFrame
    :param savefp: Path, filepath
    :return:
    """

    n_obs_phases = len(binned_fluxes)

    period_factors = [4]

    bins_idxs_ticks = [0] + \
                      [- n_bins // f + n_bins // 2 for f in period_factors] + \
                      [0 + n_bins // 2] + \
                      [n_bins // f + n_bins // 2 for f in period_factors[::-1]] + \
                      [n_bins - 1]
    bins_idxs_ticks = [el - 0.5 for el in bins_idxs_ticks]
    bins_idxs_lbls = [-tce['tce_period'] / 2] + \
                     [- tce['tce_period'] / f for f in period_factors] + \
                     [0] + \
                     [tce['tce_period'] / f for f in period_factors[::-1]] + \
                     [tce['tce_period'] / 2]

    f, ax = plt.subplots()
    im = ax.imshow(binned_fluxes.T, aspect='auto')
    ax.set_xlim([-0.5, n_obs_phases - 0.5])
    ax.set_xticks(np.arange(n_obs_phases) + 0.5)
    ax.set_xticklabels(np.arange(1, n_obs_phases + 1))
    ax.set_yticks(bins_idxs_ticks)
    ax.set_yticklabels(FormatStrFormatter('%.2f').format_ticks(bins_idxs_lbls))
    ax.set_ylim(bottom=bins_idxs_ticks[-1])
    ax.set_ylabel('Phase [day]')
    ax.set_xlabel('Phase Number')
    ax.grid(axis='x')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    f.tight_layout()
    plt.savefig(savefp)
    plt.close()


def plot_momentum_dump(loc_mom_dump_view, loc_mom_dump_view_var, binned_time, momentum_dump, time, tce, savefp):
    """ Plot phase-folded and binned momentum dump timeseries.

    Args:
        loc_mom_dump_view: NumPy array, local view of binned momentum dump time series
        loc_mom_dump_view_var: NumPy array, local view of binned momentum dump variability
        binned_time: NumPy array, binned time
        momentum_dump: NumPy array, phase-folded momentum dump time series
        time: NumPy array, phase-folded time
        tce: Pandas Series, TCE information
        savefp: str, save directory

    Returns:

    """

    f, ax = plt.subplots(2, 1)
    ax[0].plot(time, momentum_dump)
    ax[0].set_xlim(time[[0, -1]])
    ax[0].set_ylabel('Flag')
    ax[0].set_xlabel('Phase [day]')
    ax[0].set_title('Full-orbit View')
    ax[1].plot(binned_time, loc_mom_dump_view)
    ax[1].plot(binned_time, loc_mom_dump_view + loc_mom_dump_view_var, 'r--')
    ax[1].set_xlim(binned_time[[0, -1]])
    ax[1].set_ylabel('Momentum Dump Flag')
    ax[1].set_xlabel('Binned Time [day]')
    ax[1].set_title('Transit View')
    f.tight_layout()
    plt.savefig(savefp)
    plt.close()

    # f, ax = plt.subplots(2, 1)
    # ax[0].plot(time * 24, momentum_dump, c='b', linewidth=3)
    # # ax[0].set_xlim(time[[0, -1]])
    # ax[0].set_xlim(binned_time[[0, -1]] * 24)
    # ax[0].set_ylabel('Mom. Dump Flag', fontsize=16)
    # # ax[0].set_xlabel('Phase [day]')
    # ax[0].set_ylim(bottom=0)
    # # ax[0].set_title('Full-orbit View')
    # ax[1].plot(binned_time * 24, loc_mom_dump_view, c='b', linewidth=3)
    # ax[1].plot(binned_time * 24, loc_mom_dump_view + loc_mom_dump_view_var, 'r--', linewidth=3)
    # # ax[1].plot(binned_time * 24, loc_mom_dump_view - loc_mom_dump_view_var, 'r--')
    # ax[1].set_xlim(binned_time[[0, -1]] * 24)
    # ax[1].set_ylabel('Binned Value', fontsize=16)
    # ax[1].set_xlabel('Phase [hour]', fontsize=16)
    # ax[1].set_ylim(bottom=0)
    # # ax[1].set_title('Transit View')
    # f.tight_layout()
    # plt.savefig(savefp)
    # plt.close()


def plot_momentum_dump_timeseries(time_momentum_dump, momentum_dump, savefp):
    """ Plot momentum dump timeseries.

    Args:
        time_momentum_dump: NumPy array, time array
        momentum_dump: NumPy array, momentum dump
        savefp: Path, save filepath

    Returns:

    """

    f, ax = plt.subplots()
    ax.plot(time_momentum_dump, momentum_dump)
    ax.set_xlim(time_momentum_dump[[0, -1]])
    ax.set_xlabel('Time [day]')
    ax.set_ylabel('Momentum Dump Flag')
    f.tight_layout()
    plt.savefig(savefp)
    plt.close()


def plot_periodogram(tce_data, save_fp, pgram_res, n_harmonics=5):
    """ Creates figure with plots of 1) the raw flux time series and transit pulse model, 2) the periodograms for both
    time series (not-smoothed and smoothed versions), and 3) the corresponding normalized periodograms.

    Args:
        tce_data: pandas Series, TCE parameters
        save_fp: Path, save filepath
        pgram_res: dict that maps to different computed periodograms 
        n_harmonics: int, number of harmonics to display
    """

    f, ax = plt.subplots(2, 1, figsize=(20, 14))

    # --- Pre-calculate PERIODS in [day] ---
    p_raw = (1 / pgram_res['pgram'].frequency).to(u.day).value
    p_smooth = (1 / pgram_res['pgram_smooth'].frequency).to(u.day).value
    p_tpm = (1 / pgram_res['pgram_tpm'].frequency).to(u.day).value
    p_tpm_smooth = (1 / pgram_res['pgram_tpm_smooth'].frequency).to(u.day).value
    
    p_norm = (1 / pgram_res['pgram_norm'].frequency).to(u.day).value
    p_smooth_norm = (1 / pgram_res['pgram_smooth_norm'].frequency).to(u.day).value
    p_tpm_norm = (1 / pgram_res['pgram_tpm_norm'].frequency).to(u.day).value
    p_tpm_smooth_norm = (1 / pgram_res['pgram_tpm_smooth_norm'].frequency).to(u.day).value

    # --- TOP PLOT: Raw Periodogram ---
    ax[0].plot(p_raw, pgram_res['pgram'].power, zorder=2, color='b', linestyle='-')
    ax[0].plot(p_smooth, pgram_res['pgram_smooth'].power, zorder=3, color='b', linestyle='--', alpha=0.6)
    ax[0].plot(p_tpm, pgram_res['pgram_tpm'].power, zorder=2, color='tab:orange', linestyle='-')
    ax[0].plot(p_tpm_smooth, pgram_res['pgram_tpm_smooth'].power, zorder=3, color='tab:orange', linestyle='--', alpha=0.6)

    # --- BOTTOM PLOT: Normalized Periodogram ---
    ax[1].plot(p_norm, pgram_res['pgram_norm'].power, zorder=2, color='b', linestyle='-')
    ax[1].plot(p_smooth_norm, pgram_res['pgram_smooth_norm'].power, zorder=3, color='b', linestyle='--', alpha=0.6)
    ax[1].plot(p_tpm_norm, pgram_res['pgram_tpm_norm'].power, zorder=2, color='tab:orange', linestyle='-')
    ax[1].plot(p_tpm_smooth_norm, pgram_res['pgram_tpm_smooth_norm'].power, zorder=3, color='tab:orange', linestyle='--', alpha=0.6)

    # --- Add Harmonics (Triangles on the X-axis) ---
    for harmonic_i in range(1, n_harmonics + 1):
        period_val = tce_data['tce_period'] / harmonic_i
        ax[0].plot(period_val, 0.02, marker='^', color='red', markersize=12,
                   transform=ax[0].get_xaxis_transform(), clip_on=False, linestyle='None')
        ax[1].plot(period_val, 0.02, marker='^', color='red', markersize=12,
                   transform=ax[1].get_xaxis_transform(), clip_on=False, linestyle='None')

    # --- Create a Custom Legend ---
    legend_elements = [
        mlines.Line2D([], [], color='b', lw=2, label='Data'),
        mlines.Line2D([], [], color='tab:orange', lw=2, label='TPM'),
        mlines.Line2D([], [], color='gray', lw=2, linestyle='-', label='Raw'),
        mlines.Line2D([], [], color='gray', lw=2, linestyle='--', label='Smoothed'),
        mlines.Line2D([], [], color='red', marker='^', linestyle='None', markersize=10, label='TCE Period Harmonics')
    ]

    # --- Extract Max Frequencies & Periods for the Title ---
    f_max = pgram_res["pgram"].frequency_at_max_power.to(1 / u.day).value
    p_max = 1 / f_max
    f_tpm_max = pgram_res["pgram_tpm"].frequency_at_max_power.to(1 / u.day).value
    p_tpm_max = 1 / f_tpm_max

    # ==========================================
    # --- FORMATTING: TOP PLOT ---
    # ==========================================
    ax[0].legend(handles=legend_elements, loc='upper right')
    ax[0].set_ylabel('Amplitude')
    ax[0].set_xlabel('Period [day]')
    ax[0].set_xscale('log')
    ax[0].set_yscale('log')
    ax[0].set_xlim(p_raw.min(), p_raw.max())
    
    # 1. Format major ticks as plain numbers (0.1, 1, 10) instead of 10^-1
    ax[0].xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{x:g}"))
    
    # 2. Add X-Axis Grid Lines
    ax[0].grid(True, which='major', axis='x', linestyle='-', linewidth=1.2, alpha=0.5, color='gray')
    ax[0].grid(True, which='minor', axis='x', linestyle='--', linewidth=0.8, alpha=0.3, color='gray')

    ax[0].set_title(fr'Peak Amplitude @ '  
                    fr'$f_{{max}}={f_max:.3f}$ d$^{{-1}}$ ($P={p_max:.3f}$ d): ' 
                    fr'${pgram_res["pgram"].max_power:.3e}$ | TPM Peak @ ' 
                    fr'$f_{{TPM, max}}={f_tpm_max:.3f}$ d$^{{-1}}$ ($P={p_tpm_max:.3f}$ d): ' 
                    fr'${pgram_res["pgram_tpm"].max_power:.3e}$')

    # ==========================================
    # --- FORMATTING: BOTTOM PLOT ---
    # ==========================================
    ax[1].legend(handles=legend_elements, loc='upper right')
    ax[1].set_ylabel('Normalized Amplitude')
    ax[1].set_xlabel('Period [day]')
    ax[1].set_xscale('log')
    ax[1].set_yscale('log')
    ax[1].set_xlim(p_norm.min(), p_norm.max())
    
    # 1. Format major ticks as plain numbers
    ax[1].xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{x:g}"))
    
    # 2. Add X-Axis Grid Lines
    ax[1].grid(True, which='major', axis='x', linestyle='-', linewidth=1.2, alpha=0.5, color='gray')
    ax[1].grid(True, which='minor', axis='x', linestyle='--', linewidth=0.8, alpha=0.3, color='gray')

    f.suptitle(fr'Periodogram TCE {tce_data["uid"]} | Period: {tce_data["tce_period"]:.3f} day')
    
    f.tight_layout()
    plt.savefig(save_fp)
    plt.close()


def plot_phasefolded_and_binned_trend(phasefolded_data, binned_data, tce, save_fp):
    """ Plot phase folded and binned trend time series (before and after normalization).

    Args:
        phasefolded_data: tuple, phase folded time series
        binned_data: tuple, binned time series
        tce: pandas Series, TCE data
        save_fp: Path, figure filepath

    Returns:

    """

    gs = gridspec.GridSpec(3, 2)

    f = plt.figure(figsize=(20, 14))

    ax = plt.subplot(gs[0, :])
    ax.scatter(phasefolded_data['flux_trend'][0], phasefolded_data['flux_trend'][1], s=5, c='k', zorder=1, alpha=0.1)
    ax.scatter(binned_data['flux_trend_global'][0], binned_data['flux_trend_global'][1], s=8, c='r', zorder=3)
    ax.plot(binned_data['flux_trend_global'][0], binned_data['flux_trend_global'][1], 'c', zorder=2)
    ax.set_ylabel(fr'Amplitude')
    ax.set_xlim([- tce['tce_period'] / 2, tce['tce_period'] / 2])
    ax.set_xlabel('Phase [day]')
    ax = plt.subplot(gs[1, :])
    ax.scatter(binned_data['flux_trend_global_norm'][0], binned_data['flux_trend_global_norm'][1], s=8,
               c='r', zorder=3)
    ax.plot(binned_data['flux_trend_global_norm'][0], binned_data['flux_trend_global_norm'][1], 'c', zorder=2)
    ax.set_ylabel('Normalized Amplitude')
    ax.set_xlabel('Phase [day]')
    ax.set_xlim([- tce['tce_period'] / 2, tce['tce_period'] / 2])
    ax = plt.subplot(gs[2, 0])
    ax.scatter(phasefolded_data['flux_trend'][0] * 24, phasefolded_data['flux_trend'][1], s=5, c='k', zorder=1,
               alpha=0.1)
    ax.scatter(binned_data['flux_trend_local'][0] * 24, binned_data['flux_trend_local'][1], s=10,
               c='r', zorder=3, alpha=1)
    ax.plot(binned_data['flux_trend_local'][0] * 24, binned_data['flux_trend_local'][1], 'c', zorder=2)
    ax.set_xlim([- 2.5 * tce['tce_duration'] * 24, 2.5 * tce['tce_duration'] * 24])
    ax.set_ylabel(fr'Amplitude')
    ax.set_xlabel('Phase [hour]')
    ax = plt.subplot(gs[2, 1])
    ax.scatter(binned_data['flux_trend_local_norm'][0] * 24, binned_data['flux_trend_local_norm'][1], s=8,
               c='r', zorder=3)
    ax.plot(binned_data['flux_trend_local_norm'][0] * 24, binned_data['flux_trend_local_norm'][1], 'c', zorder=2)
    ax.set_ylabel('Normalized Amplitude')
    ax.set_xlabel('Phase [hour]')
    ax.set_xlim([- 2.5 * tce['tce_duration'] * 24, 2.5 * tce['tce_duration'] * 24])

    f.suptitle(f'{tce["uid"]} {tce["label"]}')
    f.tight_layout()
    plt.savefig(save_fp)
    plt.close()


def plot_phasefolded_and_binned_weak_secondary_flux(phasefolded_data, binned_data, tce, save_fp, 
                                                    exclusion_prim_factor=1.5, outlier_sigma_high=5, outliter_sigma_low=3):
    """ Plot phase folded and binned weak secondary flux time series (before and after normalization).

    Args:
        phasefolded_data: tuple, phase folded time series
        binned_data: tuple, binned time series
        tce: pandas Series, TCE data
        save_fp: Path, figure filepath
        exclusion_prim_factor: float, factor to exclude primary transit from the plot
        outlier_sigma_high: float, sigma threshold for high outliers
        outliter_sigma_low: float, sigma threshold for low outliers
    """
    
    primary_transit_midpoint = -tce['tce_maxmesd']
    half_tce_period = tce['tce_period'] / 2
    primary_transit_midpoint = (primary_transit_midpoint + half_tce_period) % tce['tce_period'] - half_tce_period

    gs = gridspec.GridSpec(2, 1)
    f = plt.figure(figsize=(20, 14))

    # --- 1. Calculate robust Y-axis limits for the Top Plot ---
    x_data = phasefolded_data['flux_weak_secondary'][0]
    y_data = phasefolded_data['flux_weak_secondary'][1]
    
    # Identify points outside the primary transit exclusion zone (+- 1.5 durations)
    duration_days = tce['tce_duration']
    primary_mask = np.abs(x_data - primary_transit_midpoint) > (exclusion_prim_factor * duration_days)
    y_valid = y_data[primary_mask]
    
    if len(y_valid) > 0:
        # 1. Calculate robust scatter using Median Absolute Deviation (MAD)
        median_flux = np.nanmedian(y_valid)
        mad_flux = np.nanmedian(np.abs(y_valid - median_flux))
        robust_std = mad_flux * 1.4826  # Convert MAD to equivalent standard deviation
        
        # 2. Upper Bound: Clip extreme positive outliers (e.g., stellar flares / cosmic rays)
        # 5 sigma above the median is a very safe cutoff for positive outliers
        y_high = median_flux + (outlier_sigma_high * robust_std) 
        
        # 3. Lower Bound: Anchor to the lowest point of the *binned* data (the true physical dip)
        # and subtract 3 "sigmas" of scatter so the unbinned points around the bottom of the dip aren't cut off.
        binned_min = np.nanmin(binned_data['flux_weak_secondary_local'][1])
        y_low = binned_min - (outliter_sigma_low * robust_std)
        
        # # 4. Optional: If there's a massive negative outlier in the binned data itself, 
        # # fallback to a statistical floor just in case.
        # y_low = max(y_low, median_flux - (10 * robust_std)) 
        
        ylim_top = [y_low, y_high]
    else:
        ylim_top = None

    # --- TOP PLOT (Full-Orbit) ---
    ax = plt.subplot(gs[0, 0])
    ax.scatter(x_data, y_data, s=8, c='k', zorder=1, alpha=1)
    
    # Replace axvline with a triangle pointing up, anchored to the bottom spine (Y=0 in axes coordinates)
    ax.plot(primary_transit_midpoint, 0.02, marker='^', color='red', markersize=14, 
            transform=ax.get_xaxis_transform(), clip_on=False, linestyle='None', 
            label='Primary Transit Midpoint')
    
    if ylim_top:
        ax.set_ylim(ylim_top)
        
    ax.set_ylabel('Relative Flux')
    ax.set_xlabel('Phase [day]')
    ax.set_title('Full-Orbit')
    ax.legend()

    # --- BOTTOM PLOT (Transit-View) ---
    ax = plt.subplot(gs[1, 0])
    x_bottom = x_data * 24
    xlim_bottom = [-2.5 * duration_days * 24, 2.5 * duration_days * 24]
    
    ax.scatter(x_bottom, y_data, s=5, c='k', zorder=1, alpha=0.1)
    ax.scatter(binned_data['flux_weak_secondary_local'][0] * 24, binned_data['flux_weak_secondary_local'][1], 
               s=8, c='r', zorder=3)
    ax.plot(binned_data['flux_weak_secondary_local'][0] * 24, binned_data['flux_weak_secondary_local'][1], 
            'c', zorder=2, label='Binned')
    
    # Only add the primary transit marker to the bottom plot IF it is inside the zoomed-in view
    pt_x_bottom = primary_transit_midpoint * 24
    if xlim_bottom[0] <= pt_x_bottom <= xlim_bottom[1]:
        ax.plot(pt_x_bottom, 0.02, marker='^', color='red', markersize=14, 
                transform=ax.get_xaxis_transform(), clip_on=False, linestyle='None', 
                label='Primary Transit Midpoint')
        ax.legend()

    # Calculate y-limits for the bottom plot to prevent zoomed-in outliers from stretching it
    zoom_mask = (x_bottom >= xlim_bottom[0]) & (x_bottom <= xlim_bottom[1])
    if np.any(zoom_mask):
        y_zoom = y_data[zoom_mask]
        y_low_b, y_high_b = np.nanpercentile(y_zoom, [0.5, 99.5])
        y_margin_b = (y_high_b - y_low_b) * 0.15
        ax.set_ylim([y_low_b - y_margin_b, y_high_b + y_margin_b])

    ax.set_xlim(xlim_bottom)
    ax.set_ylabel('Relative Flux')
    ax.set_xlabel('Phase [hour]')
    ax.set_title('Transit-View')
    ax.legend()

    f.suptitle(f'Weak Secondary TCE {tce["uid"]}\n' 
               f'Secondary Transit Depth: {tce["wst_depth"]:.3f} (ppm) | Offset: {tce["tce_maxmesd"]:.3f} (day) | MES: {tce["tce_maxmes"]:.3f}')
    f.tight_layout()
    plt.savefig(save_fp)
    plt.close()


def compile_preprocessing_figures_to_pdf(target_uid, tce_tbl, plot_dir, save_fp, delete_plots=False):
    """
    Compiles preprocessing PNG figures for a target and its TCEs into a single PDF.
    
    Args:
        target_uid (str): Target unique ID (e.g. TIC-Sector)
        tce_tbl (pandas.DataFrame): Table of TCEs for this target
        plot_dir (pathlib.Path): Directory containing the PNGs
        save_fp (pathlib.Path): Output filepath for the PDF
    """

    images_list = []
    
    # 1. Target-level figures
    target_prefixes = [
        f"{target_uid}_2_detrendedflux.png",
        f"{target_uid}_3_1_detrendedcentroids.png",
        f"{target_uid}_4_momentum_dump_timeseries.png"
    ]
    
    for prefix in target_prefixes:
        matches = glob.glob(str(plot_dir / f"*{prefix}"))
        if matches:
            images_list.append(matches[0])
            
    # 2. TCE-level figures
    for _, tce in tce_tbl.iterrows():
        tce_uid = tce["uid"]
        tce_label = tce["label"] if "label" in tce else "UNK"
        
        # Expected suffixes based on typical generation order
        tce_suffixes = [
            "1_intransit_cadences.png",
            "2_lc_periodogram.png",
            "3_2_correctedcentroids.png",
            "3_3_distcentr.png",
            "5_phasefolded_timeseries.png",
            "5_phasefolded_timeseries_outlierrem.png",
            "6_riverplot_flux_aug.png",
            "7_1_riverplot_flux_trend.png",
            "7_2_phasefoldedbinned_trend.png",
            "8_1_oddeven_transitdepth_phasefoldedbinned_timeseries.png",
            "8_2_momentum_dump_phase_and_binned.png",
            "8_3_flux_weak_secondary.png",
            "9_1_phasefoldedbinned_timeseries.png",
            "9_2_binned_timeseries.png"
        ]
        
        for suffix in tce_suffixes:
            matches = glob.glob(str(plot_dir / f"{tce_uid}_{tce_label}_{suffix}"))
            # Fallback in case label is different or missing
            if not matches:
                matches = glob.glob(str(plot_dir / f"{tce_uid}_*_{suffix}"))
            
            if matches:
                images_list.append(matches[0])
                
    if not images_list:
        return
        
    try:
        imgs = [Image.open(img).convert('RGB') for img in images_list]
        imgs[0].save(str(save_fp), save_all=True, append_images=imgs[1:])
        for img in imgs:
            img.close()
    except Exception as e:
        print(f"Failed to compile PDF for {target_uid}: {e}")
    
    if delete_plots:
        for image_fp in images_list:
            os.remove(image_fp)
