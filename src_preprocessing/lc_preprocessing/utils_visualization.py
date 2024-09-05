""" Auxiliary functions used to plot outcome from different steps along the preprocessing pipeline. """

# 3rd party
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from astropy.stats import mad_std
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astropy import units as u

plt.switch_backend('agg')


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
                   label=None if i < n_arrs - 1 else 'Detected target TCEs')
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


def plot_centroids(time, centroids, detrended_centroids, tce, config, savefp, pxcoordinates=False,
                   target_position=None, delta_dec=None):
    """ Creates and saves a figure with plots that show the centroid, trend, and detrended centroid timeseries for a
    given TCE.

    :param time: numpy array, time
    :param centroids: dict with 'x' and 'y' keys for the coordinates, and values are numpy arrays. Holds the raw
    centroid timeseries
    :param detrended_centroids: dict with 'x' and 'y' keys for the coordinates, and values are dictionaries with the
    detrended centroid 'detrended', removed trend 'trend', residual time series 'residual', and, optionally, the
    linearly interpolated raw centroid timeseries used for fitting 'linear_interp'
    :param tce: pandas Series, row of the input TCE table Pandas DataFrame
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

    f.suptitle(f'{tce.uid} {tce.label}\nTarget: {target_position_plot[0]:.3f}, {target_position_plot[1]:.3f} '
               f'({target_position_unit})')
    plt.savefig(savefp)
    plt.close()


def plot_flux_detrend(time, flux, trend, detrended_flux, tce, savedir, basename, flux_interp=None):
    """ Creates and saves a 2x1 figure with plots that show the flux time series and the fitted trend and
     the respective detrended flux time series for a given TCE.

    :param time: numpy array, time
    :param flux: numpy array, flux
    :param trend: numpy array, fitted trend
    :param detrended_flux: numpy array, detrended flux
    :param tce: pandas Series, row of the input TCE table Pandas DataFrame
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
    ax[1].plot(time, detrended_flux, 'b')
    ax[1].plot(time, trend, 'orange', linestyle='--', label='Trend', zorder=1)
    ax[1].legend()
    ax[0].set_ylabel(fr'Amplitude [$e^-s^-1$]')
    ax[0].set_title('Raw Flux')
    ax[1].set_ylabel('Normalized Amplitude')
    ax[1].set_xlabel('Time [day]')
    ax[1].set_title('Detrended Flux')
    ax[1].set_xlim(time[[0, -1]])
    f.suptitle(f'TCE {tce.uid} {tce.label}')
    plt.savefig(savedir / f'{tce.uid}_{tce.label}_{basename}.png')
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
    ax[0, 0].plot(np.concatenate(all_time_oot), avg_centroid_oot['x'] * np.ones(len(np.concatenate(all_time_oot))), 'r--',
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
    ax[1, 0].plot(np.concatenate(all_time_oot), avg_centroid_oot['y'] * np.ones(len(np.concatenate(all_time_oot))), 'r--',
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


def plot_all_views(views, tce, config, scheme, savefp, plot_var):
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

                ax[i, j].plot(views[views_list[k]][0], views[views_list[k]][1], zorder=2, color='k')
                ax[i, j].scatter(views[views_list[k]][0], views[views_list[k]][1], s=10, color='k',
                                 zorder=2)
                if plot_var:
                    # ax[i, j].plot(views[views_list[k]][1] + views[views_list[k]][2], 'r--', alpha=0.7, zorder=1)
                    # ax[i, j].plot(views[views_list[k]][1] - views[views_list[k]][2], 'r--', alpha=0.7, zorder=1)

                    ax[i, j].plot(views[views_list[k]][0], views[views_list[k]][1] + views[views_list[k]][2], 'r--', alpha=0.7, zorder=1)
                    ax[i, j].plot(views[views_list[k]][0], views[views_list[k]][1] - views[views_list[k]][2], 'r--', alpha=0.7, zorder=1)

                ax[i, j].set_title(f'{views_list[k]} num transits={views[views_list[k]][3]}', pad=20)

                # ax[i, j].set_xlim([0, len(views[views_list[k]])])
                ax[i, j].set_xlim(views[views_list[k]][0][[0, -1]])
            if i == scheme[0] - 1:
                # ax[i, j].set_xlabel('Bin Number')
                ax[i, j].set_xlabel('Phase [day]')
            if j == 0:
                ax[i, j].set_ylabel('Amplitude')

            k += 1

    f.suptitle(f'{tce["uid"]} {tce["label"]} | {ephemerisStr}\n{scalarParamsStr}')
    plt.subplots_adjust(hspace=0.5, wspace=0.37, top=0.83, right=0.974, bottom=0.07, left=0.05)
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
                    ax[i, j].set_title('{} N_transits={}'.format(views_list[k], num_transits['flux_even']), pad=20)
                elif views_list[k] == 'local_weak_secondary_view':
                    ax[i, j].set_title('{} N_transits={}'.format(views_list[k], num_transits['wks']), pad=20)
                elif views_list[k] == 'global_centr_view':
                    ax[i, j].set_title('{} N_transits={}'.format(views_list[k], num_transits['centroid']), pad=20)
                elif views_list[k] == 'global_centr_fdl_view':
                    ax[i, j].set_title('{} N_transits={}'.format(views_list[k], num_transits['centroid_fdl']), pad=20)
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


def plot_wks(glob_view, glob_view_weak_secondary, tce, config, savedir, basename):
    """ Creates and saves a figure with plots of the global flux view and global weak secondary flux view for a given
    TCE.

    :param glob_view: NumPy array, global flux view
    :param glob_view_weak_secondary: NumPy array, global weak secondary flux view
    :param tce: pandas Series, row of the input TCE table Pandas DataFrame
    :param config: dict, preprocessing parameters.
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
    ax.scatter(timeseries['flux'][0], timeseries['flux'][1], color='k', s=5)
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
    ax.scatter(timeseries['flux'][0] * 24, timeseries['flux'][1], color='k', s=5)
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
                   timeseries['flux_weak_secondary'][1], color='k', s=5)
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
        ax.scatter(timeseries['flux_odd'][0] * 24, timeseries['flux_odd'][1], color='k', s=5)
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
        ax.scatter(timeseries['flux_even'][0] * 24, timeseries['flux_even'][1], color='k', s=5)
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
    ax.scatter(timeseries['flux_trend'][0], timeseries['flux_trend'][1], color='k', s=5)
    ax.scatter(binned_timeseries['flux_trend_global'][0], binned_timeseries['flux_trend_global'][1], color='b')
    ax.plot(binned_timeseries['flux_trend_global'][0], binned_timeseries['flux_trend_global'][1], 'b')
    ax.set_ylabel('Relative Flux')
    ax.set_xlabel('Phase [day]')
    # ax.set_xlim([timeseries['Flux'][0][0], timeseries['Flux'][0][-1]])
    ax.set_xlim([- tce['tce_period'] / 2, tce['tce_period'] / 2])
    ax.set_title('Flux Trend')

    ax = plt.subplot(gs[4, :])
    ax.scatter(timeseries['centroid_offset_distance_to_target'][0],
               timeseries['centroid_offset_distance_to_target'][1], color='k', s=5)
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
               color='k', s=5)
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


def plot_odd_even(binned_timeseries, phasefolded_timeseries, tce, config, savefp, sigma_factor=6,
                  delta_factor=0.998):
    """ Creates and saves a figure with plots for odd-even transit depth test for a given TCE.

    :param binned_timeseries: dict, binned views
    :param phasefolded_timeseries: dict, phase-folded time series
    :param tce: Pandas Series, row of the input TCE table Pandas DataFrame
    :param config: dict, preprocessing parameters.
    :param savefp: Path, filepath
    :param sigma_factor: float, used for defining the maximum amplitude range in the plots
    (med(phase_folded_flux) + sigma_factor * mad_std(phase_folded flux))
    :param delta_factor: float, used for defining the minimum amplitude in the plots
    (delta_factor * min(min value among all binned time series)
    :return:
    """

    # timeseries_madstd, timeseries_med = (mad_std(timeseries['Flux'][1], ignore_nan=True),
    #                                      np.nanmedian(timeseries['Flux'][1]))
    # std_range = sigma_factor * timeseries_madstd
    # min_range = delta_factor * np.nanmin(np.concatenate([phasefolded_timeseries['odd_flux'][1],
    #                                                      phasefolded_timeseries['even_flux'][1],
    #                                                      binned_timeseries['odd_flux_local'][1],
    #                                                      binned_timeseries['even_flux_local'][1]]))
    # range_timeseries = [min_range, timeseries_med + std_range]

    local_view_time_interval = tce['tce_duration'] * (config['num_durations'])

    gs = gridspec.GridSpec(2, 2)
    f = plt.figure(figsize=(20, 14))

    # odd_flux_local (phase-folded + binned ts)
    ax = plt.subplot(gs[0, 0])
    # if len(timeseries['Odd Flux'][0]) > 0:
    ax.scatter(phasefolded_timeseries['flux_odd'][0] * 24, phasefolded_timeseries['flux_odd'][1], color='k', s=5,
               alpha=0.8)
    ax.scatter(binned_timeseries['flux_odd_local'][0] * 24, binned_timeseries['flux_odd_local'][1], color='r')
    ax.plot(binned_timeseries['flux_odd_local'][0] * 24, binned_timeseries['flux_odd_local'][1], 'c')
    ax.set_ylabel('Relative Flux')
    ax.set_xlabel('Phase [hour]')
    ax.set_xlim([- local_view_time_interval * 24, local_view_time_interval * 24])
    ax.set_title('Odd')
    # if not np.isnan(range_timeseries).any():
    #     ax.set_ylim(range_timeseries)

    # even_flux_local (phase-folded + binned ts)
    ax = plt.subplot(gs[0, 1])
    # if len(timeseries['Even Flux'][0]) > 0:
    ax.scatter(phasefolded_timeseries['flux_even'][0] * 24, phasefolded_timeseries['flux_even'][1], color='k', s=5,
               alpha=0.8)
    ax.scatter(binned_timeseries['flux_even_local'][0] * 24, binned_timeseries['flux_even_local'][1], color='r')
    ax.plot(binned_timeseries['flux_even_local'][0] * 24, binned_timeseries['flux_even_local'][1], 'c')
    ax.set_ylabel('Relative Flux')
    ax.set_xlabel('Phase [hour]')
    ax.set_xlim([- local_view_time_interval * 24, local_view_time_interval * 24])
    ax.set_title('Even')
    # if not np.isnan(range_timeseries).any():
    #     ax.set_ylim(range_timeseries)

    ax = plt.subplot(gs[1, :])

    ax.scatter(phasefolded_timeseries['flux_odd'][0] * 24, phasefolded_timeseries['flux_odd'][1], color='k', s=5,
               alpha=0.1, zorder=1)
    ax.scatter(phasefolded_timeseries['flux_even'][0] * 24, phasefolded_timeseries['flux_even'][1], color='k', s=5,
               alpha=0.1, zorder=1)

    # ax.scatter(binned_timeseries['Local Flux'][0] * 24, binned_timeseries['Local Flux'][1], color='c', zorder=2)
    # ax.plot(binned_timeseries['Local Flux'][0] * 24, binned_timeseries['Local Flux'][1], 'c', zorder=2)

    # even_flux_local (binned)
    ax.scatter(binned_timeseries['flux_odd_local'][0] * 24, binned_timeseries['flux_even_local'][1], color='r',
               zorder=2)
    ax.plot(binned_timeseries['flux_odd_local'][0] * 24, binned_timeseries['flux_even_local'][1], 'r', label='Even',
            zorder=2)

    # odd_flux_local (binned)
    ax.scatter(binned_timeseries['flux_odd_local'][0] * 24, binned_timeseries['flux_odd_local'][1], color='g',
               zorder=2)
    ax.plot(binned_timeseries['flux_odd_local'][0] * 24, binned_timeseries['flux_odd_local'][1], 'g', label='Odd',
            zorder=2)

    ax.legend()
    ax.set_ylabel('Relative Flux')
    ax.set_xlabel('Phase [hour]')
    ax.set_xlim([- local_view_time_interval * 24, local_view_time_interval * 24])
    # if not np.isnan(range_timeseries).any():
    #     ax.set_ylim(range_timeseries)

    f.suptitle('{} {}'.format(tce.uid, tce.label))
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
    ax[1].plot(binned_time, loc_mom_dump_view - loc_mom_dump_view_var, 'r--')
    ax[1].set_xlim(binned_time[[0, -1]])
    ax[1].set_ylabel('Momentum Dump Flag')
    ax[1].set_xlabel('Binned Time [day]')
    ax[1].set_title('Transit View')
    f.tight_layout()
    plt.savefig(savefp)
    plt.close()


def plot_momentum_dump_timeseries(time_momentum_dump, momentum_dump, tce, savefp):
    """ Plot momentum dump timeseries.

    Args:
        time_momentum_dump: NumPy array, time array
        momentum_dump: NumPy array, momentum dump
        tce: Pandas Series, TCE information
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


def plot_periodogram(tce_data, save_fp, lc_data, lc_tpm_data, pgram_res, n_harmonics=5):
    """ Creates figure with plots of 1) the raw flux time series and transit pulse model, 2) the periodograms for both
    time series (not-smoothed and smoothed versions), and 3) the corresponding normalized periodograms.

    Args:
        tce_data: pandas Series, TCE parameters
        save_fp: Path, save filepath
        lc_data: lightkurve LightCurve object with raw light curve data
        lc_tpm_data: lightkurve LightCurve object with transit pulse model data
        pgram_res: dict that maps to different computed periodograms (e.g., whether for the lc or transit pulse model
        data, whether smoothed, and whether normalized)
        n_harmonics: int, number of harmonics to display

    Returns:
    """

    f0_tce = 1 / (tce_data['tce_period'] * 24 * 3600)  # in 1/s

    f, ax = plt.subplots(3, 1, figsize=(14, 10))

    ax[0].scatter(lc_data.time.value, lc_data.flux.value, s=6, color='b')

    ax[0].plot(lc_tpm_data.time.value, lc_tpm_data.flux.value, linestyle='--', label='Transit Pulse Model (TPM)',
               color='tab:orange')

    ax[0].legend()
    ax[0].set_xlabel('Time [day]')
    ax[0].set_ylabel(r'PDCSAP Flux [$e^-s^{-1}$]')
    ax[0].set_xlim(lc_tpm_data.time.value[[0, -1]])

    # plot periodogram of the data
    ax[1].plot(pgram_res['pgram'].frequency.to(1 / u.s), pgram_res['pgram'].power, zorder=3, color='b', label='Data')
    ax[1].plot(pgram_res['pgram_smooth'].frequency.to(1 / u.s), pgram_res['pgram_smooth'].power, zorder=3,
               linestyle='--', color='c', alpha=0.5, label='Data Smoothed')
    # add vertical line for frequency at maximum power
    # ax[1].axvline(x=pgram_res['pgram'].frequency_at_max_power.to(1 / u.s).value, c='m', linestyle='-',
    #               label=None, zorder=2)

    # add vertical lines for f0 and harmonics of the TCE
    # ax[1].axvline(x=f0_tce, c='k', linestyle='-.',
    #               label=fr'$f_0={f0_tce:.3e} /s$', zorder=2, linewidth=2)
    for harmonic_i in range(1, n_harmonics + 1):
        ax[1].axvline(x=f0_tce * harmonic_i, c='r', linestyle='--',
                      label='Harmonics' if harmonic_i == n_harmonics + 1 else None, zorder=2, linewidth=2, alpha=0.5)

    # plot periodograms of TPM model
    ax[1].plot(pgram_res['pgram_tpm'].frequency.to(1 / u.s), pgram_res['pgram_tpm'].power, linestyle='-',
               zorder=3, alpha=1, label='TPM', color='tab:orange')
    ax[1].plot(pgram_res['pgram_tpm_smooth'].frequency.to(1 / u.s), pgram_res['pgram_tpm_smooth'].power, linestyle='--',
               zorder=3, alpha=0.5, label='TPM Smoothed', color='k')
    # # add vertical line for frequency at maximum power for TPM model
    # ax[1].axvline(x=pgram_res['pgram_tpm_smooth'].frequency_at_max_power.to(1 / u.s).value, c='k', linestyle='-',
    #               label=None,
    #               zorder=2)

    ax[1].legend()
    ax[1].set_ylabel('Amplitude')
    ax[1].set_xlabel('Frequency [Hz]')
    ax[1].set_xlim(pgram_res['pgram'].frequency.to(1 / u.s).value[[0, -1]])
    ax[1].set_yscale('log')
    ax[1].set_xscale('log')

    ax[1].set_title(fr'Peak Amplitude @ '  
                    fr'$f_{{Data, max}}={pgram_res["pgram"].frequency_at_max_power.to(1 / u.s):.3e}$:' 
                    fr'$={pgram_res["pgram"].max_power:.3e}$ | TPM Peak Amplitude @ ' 
                    fr'$f_{{TPM, max}}={pgram_res["pgram_tpm"].frequency_at_max_power.to(1 / u.s):.3e}$:' 
                    fr'${pgram_res["pgram_tpm"].max_power:.3e}$')

    ax[2].plot(pgram_res['pgram_norm'].frequency.to(1 / u.s), pgram_res['pgram_norm'].power, zorder=2, linestyle='-',
               color='b', label='Data')
    ax[2].plot(pgram_res['pgram_smooth_norm'].frequency.to(1 / u.s), pgram_res['pgram_smooth_norm'].power, zorder=3,
               linestyle='--', color='c',
               alpha=0.5, label='Data Smoothed')

    ax[2].plot(pgram_res['pgram_tpm_norm'].frequency.to(1 / u.s), pgram_res['pgram_tpm_norm'].power, 'tab:orange',
               label='TPM', alpha=0.5,
               linestyle='-', zorder=2)
    ax[2].plot(pgram_res['pgram_tpm_smooth_norm'].frequency.to(1 / u.s), pgram_res['pgram_tpm_smooth_norm'].power,
               'k', label='TPM Smoothed', alpha=0.5, linestyle='dashed', zorder=3)

    ax[2].legend()
    ax[2].set_ylabel('Normalized Amplitude')
    ax[2].set_xlabel('Frequency [Hz]')
    ax[2].set_xlim(pgram_res['pgram_norm'].frequency.to(1 / u.s).value[[0, -1]])
    ax[2].set_yscale('log')
    ax[2].set_xscale('log')

    f.suptitle(fr'{tce_data["uid"]} {tce_data["label"]}' 
               fr'Period: {tce_data["tce_period"]:.3f} day | $f_0={f0_tce:.3e} /s$')
    f.tight_layout()
    f.savefig(save_fp)
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
    ax.scatter(phasefolded_data['flux_trend'][0], phasefolded_data['flux_trend'][1], s=8, c='k', zorder=1)
    ax.scatter(binned_data['flux_trend_global'][0], binned_data['flux_trend_global'][1], s=8, c='b', zorder=2)
    ax.set_ylabel(fr'Amplitude [$e^-s^-1$]')
    ax.set_xlim([- tce['tce_period'] / 2, tce['tce_period'] / 2])
    ax.set_xlabel('Phase [day]')
    ax = plt.subplot(gs[1, :])
    ax.scatter(binned_data['flux_trend_global_norm'][0], binned_data['flux_trend_global_norm'][1], s=8,
               c='b')
    ax.set_ylabel('Normalized Amplitude')
    ax.set_xlabel('Phase [day]')
    ax.set_xlim([- tce['tce_period'] / 2, tce['tce_period'] / 2])
    ax = plt.subplot(gs[2, 0])
    ax.scatter(phasefolded_data['flux_trend'][0] * 24, phasefolded_data['flux_trend'][1], s=8, c='k', zorder=1)
    ax.scatter(binned_data['flux_trend_local'][0] * 24, binned_data['flux_trend_local'][1], s=8,
               c='b', zorder=2)
    ax.set_xlim([- 2.5 * tce['tce_duration'] * 24, 2.5 * tce['tce_duration'] * 24])
    ax.set_ylabel(fr'Amplitude [$e^-s^-1$]')
    ax.set_xlabel('Phase [hour]')
    ax = plt.subplot(gs[2, 1])
    ax.scatter(binned_data['flux_trend_local_norm'][0] * 24, binned_data['flux_trend_local_norm'][1], s=8,
               c='b')
    ax.set_ylabel('Normalized Amplitude')
    ax.set_xlabel('Phase [hour]')
    ax.set_xlim([- 2.5 * tce['tce_duration'] * 24, 2.5 * tce['tce_duration'] * 24])

    f.suptitle(f'{tce["uid"]} {tce["label"]}')
    f.tight_layout()
    plt.savefig(save_fp)
    plt.close()


def plot_phasefolded_and_binned_weak_secondary_flux(phasefolded_data, binned_data, tce, save_fp):
    """ Plot phase folded and binned weak secondary flux time series (before and after normalization).

    Args:
        phasefolded_data: tuple, phase folded time series
        binned_data: tuple, binned time series
        tce: pandas Series, TCE data
        save_fp: Path, figure filepath

    Returns:

    """

    gs = gridspec.GridSpec(2, 2)

    f = plt.figure(figsize=(20, 14))

    ax = plt.subplot(gs[0, :])
    ax.scatter(phasefolded_data['flux_weak_secondary'][0], phasefolded_data['flux_weak_secondary'][1], s=8, c='k',
               zorder=1)
    ax.axvline(x=-tce['tce_maxmesd'], c='r', label='Primary', zorder=2, linestyle='--', alpha=0.5)
    ax.set_ylabel('Amplitude')
    ax.set_xlim([- tce['tce_period'] / 2, tce['tce_period'] / 2])
    ax.set_xlabel('Phase [day]')
    ax.legend()

    ax = plt.subplot(gs[1, 0])
    ax.scatter(phasefolded_data['flux_weak_secondary'][0] * 24, phasefolded_data['flux_weak_secondary'][1], s=8, c='k',
               zorder=1)
    ax.scatter(binned_data['flux_weak_secondary_local'][0] * 24, binned_data['flux_weak_secondary_local'][1], s=8,
               c='b')
    ax.set_ylabel('Amplitude')
    ax.set_xlabel('Phase [hour]')
    ax.set_xlim([- 2.5 * tce['tce_duration'] * 24, 2.5 * tce['tce_duration'] * 24])

    ax = plt.subplot(gs[1, 1])
    ax.scatter(binned_data['flux_weak_secondary_local_norm'][0] * 24, binned_data['flux_weak_secondary_local_norm'][1],
               s=8, c='b', zorder=2)
    ax.set_xlim([- 2.5 * tce['tce_duration'] * 24, 2.5 * tce['tce_duration'] * 24])
    ax.set_ylabel('Normalized Amplitude')
    ax.set_xlabel('Phase [hour]')

    f.suptitle(f'{tce["uid"]} {tce["label"]}')
    f.tight_layout()
    plt.savefig(save_fp)
    plt.close()
