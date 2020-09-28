"""
Auxiliary functions used to plot outcome from different steps along the preprocessing pipeline.
"""

# 3rd party
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
# if 'nobackup' in os.path.abspath(__file__):
plt.switch_backend('agg')


DEGREETOARCSEC = 3600


def plot_binseries_flux(all_time, all_flux, binary_time_all, tce, config, savedir, basename):
    """ Creates and saves a 2x1 figure with plots that show the ephemeris pulse train and the flux time-series for
    a given TCE.

    :param all_time: list of numpy arrays, time
    :param all_flux: list of numpy arrays, flux time-series
    :param binary_time_all: list of numpy arrays, binary arrays with 1 for in-transit cadences and 0 otherwise
    :param tce: pandas Series, row of the input TCE table Pandas DataFrame.
    :param config: Config object; preprocessing parameters.
    :param savedir: str, filepath to directory in which the figure is saved
    :param basename: str, added to the figure filename
    :return:
    """

    f, ax = plt.subplots(2, 1, sharex=True, figsize=(14, 8))

    plt.suptitle('TCE {} | {} | {}'.format(tce['target_id'], tce[config.tce_identifier], tce['label']))

    for i in range(len(all_time)):
        ax[0].plot(all_time[i], binary_time_all[i], 'b')
        ax[0].axvline(x=all_time[i][-1], ymax=1, ymin=0, c='r')
    ax[0].set_title('Binary time-series')
    ax[0].set_ylabel('Binary amplitude (it-oot)')

    for i in range(len(all_time)):
        ax[1].plot(all_time[i], all_flux[i], 'b')
        ax[1].axvline(x=all_time[i][-1], ymax=1, ymin=0, c='r')
    ax[1].set_title('Flux')
    ax[1].set_ylabel('Amplitude')
    ax[1].set_xlabel('Time [day]')

    plt.savefig(os.path.join(savedir, '{}_{}_{}_{}.png'.format(tce.target_id, tce[config.tce_identifier], tce.label,
                                                               basename)))

    plt.close()


def plot_centroids(time, centroids, centroids_spline, tce, config, savedir, basename, add_info=None,
                   pxcoordinates=False, target_position=None, **kwargs):
    """ Creates and saves a figure with plots that show the centroid time-series and, if desired, the fitted spline and
     the respective spline normalized centroid time-series, for a given TCE.

    :param time: list of numpy arrays, time
    :param centroids: dict ('x' and 'y' keys, values are lists of numpy arrays), centroid time-series
    :param centroids_spline: dict ('x' and 'y' keys, values are lists of numpy arrays), spline fitted to the centroid
    time-series
    :param tce: pandas Series, row of the input TCE table Pandas DataFrame.
    :param config: Config object; preprocessing parameters.
    :param savedir: str, filepath to directory in which the figure is saved
    :param basename: str, added to the figure filename
    :param add_info: dict, 'quarter' and 'module' are lists with the quarters and modules in which the target shows up,
    respectively
    :return:
    """

    if not config.px_coordinates and not pxcoordinates:
        if target_position is None:
            centroids = {coord: [DEGREETOARCSEC * centroids_arr for centroids_arr in centroids[coord]] for coord in centroids}
        else:
            centroids = {'x': [DEGREETOARCSEC * (centroids_arr - target_position[0]) * np.cos(target_position[1] * np.pi / 180)
                               for centroids_arr in centroids['x']],
                         'y': [DEGREETOARCSEC * (centroids_arr - target_position[1]) for centroids_arr in centroids['y']]}

        if centroids_spline is not None:
            centroids_spline = {coord: [DEGREETOARCSEC * spline_arr for spline_arr in centroids_spline[coord]]
                                for coord in centroids_spline}

        if 'centroid_interp' in kwargs:
            kwargs['centroid_interp'] = {coord: [DEGREETOARCSEC * arr for arr in kwargs['centroid_interp'][coord]]
                                         for coord in kwargs['centroid_interp']}

    if centroids_spline is None:

        f, ax = plt.subplots(2, 1, figsize=(16, 14))

        for i, centroids_arr in enumerate(zip(centroids['x'], centroids['y'])):
            ax[0].plot(time[i], centroids_arr[0], 'b')
            ax[1].plot(time[i], centroids_arr[1], 'b')

        if config.px_coordinates or pxcoordinates:
            ax[0].set_ylabel('Col pixel')
            ax[1].set_ylabel('Row pixel')
        else:
            ax[0].set_ylabel('RA [arcsec]')
            ax[1].set_ylabel('Dec [arcsec]')

        ax[1].set_xlabel('Time [day]')

        if config.satellite == 'kepler':

            if add_info is not None:
                f.suptitle('Quarters: {}\nModules: {}'.format(add_info['quarter'], add_info['module']))

        ax[0].set_title('TCE {} {} {}'.format(tce.target_id, tce[config.tce_identifier], tce.label))

        plt.savefig(os.path.join(savedir, '{}_{}_{}_{}.png'.format(tce.target_id, tce[config.tce_identifier], tce.label,
                                                                   basename)))

        plt.close()

    else:

        f, ax = plt.subplots(2, 2, figsize=(18, 12))

        for i, centroids_arr in enumerate(zip(centroids['x'], centroids['y'])):
            ax[0, 0].plot(time[i], centroids_arr[0], 'b', zorder=0)
            if i == 0:
                ax[0, 0].plot(time[i], centroids_spline['x'][i], 'orange', linestyle='--', label='Fitted spline',
                              zorder=1)
                if 'centroid_interp' in kwargs:
                    ax[0, 0].plot(time[i], kwargs['centroid_interp']['x'][i], 'g', label='Gapped transits', zorder=0)
            else:
                ax[0, 0].plot(time[i], centroids_spline['x'][i], 'orange', linestyle='--', zorder=1)
                if 'centroid_interp' in kwargs:
                    ax[0, 0].plot(time[i], kwargs['centroid_interp']['x'][i], 'g', zorder=0)

            ax[0, 0].legend()

            ax[1, 0].plot(time[i], centroids_arr[1], 'b', zorder=0)
            if i == 0:
                ax[1, 0].plot(time[i], centroids_spline['y'][i], 'orange', linestyle='--', label='Fitted spline',
                              zorder=1)
                if 'centroid_interp' in kwargs:
                    ax[1, 0].plot(time[i], kwargs['centroid_interp']['y'][i], 'g', label='Gapped transits', zorder=0)
            else:
                ax[1, 0].plot(time[i], centroids_spline['y'][i], 'orange', linestyle='--', zorder=1)
                if 'centroid_interp' in kwargs:
                    ax[1, 0].plot(time[i], kwargs['centroid_interp']['y'][i], 'g', zorder=0)

            ax[1, 0].legend()

            ax[0, 1].plot(time[i], centroids_arr[0] / centroids_spline['x'][i], 'b')
            ax[1, 1].plot(time[i], centroids_arr[1] / centroids_spline['y'][i], 'b')

        if config.px_coordinates or pxcoordinates:
            ax[0, 0].set_ylabel('Col pixel')
            ax[1, 0].set_ylabel('Row pixel')
        else:
            ax[0, 0].set_ylabel('RA [arcsec]')
            ax[1, 0].set_ylabel('Dec [arcsec]')

        ax[0, 1].set_ylabel('Normalized amplitude')
        ax[1, 1].set_ylabel('Normalized amplitude')

        ax[1, 0].set_xlabel('Time [day]')
        ax[1, 1].set_xlabel('Time [day]')

        ax[0, 0].set_title('Non-normalized centroid time-series')
        ax[0, 1].set_title('Normalized centroid time-series')

        f.suptitle('TCE {} {} {}'.format(tce.target_id, tce[config.tce_identifier], tce.label))

        plt.savefig(os.path.join(savedir, '{}_{}_{}_{}.png'.format(tce.target_id, tce[config.tce_identifier], tce.label,
                                                                   basename)))

        plt.close()


def plot_flux_fit_spline(time, flux, spline_flux, tce, config, savedir, basename, **kwargs):
    """ Creates and saves a 2x1 figure with plots that show the flux time-series and the fitted spline and
     the respective spline normalized flux time-series, for a given TCE.

    :param time: list of numpy arrays, time
    :param flux: list of numpy arrays, flux time-series
    :param flux_spline: list of numpy arrays, spline fitted to the flux time-series
    :param tce: pandas Series, row of the input TCE table Pandas DataFrame
    :param config: Config object; preprocessing parameters.
    :param savedir: str, filepath to directory in which the figure is saved
    :param basename: str, added to the figure filename
    :return:
    """

    f, ax = plt.subplots(2, 1, figsize=(16, 10))
    for i in range(len(flux)):
        ax[0].plot(time[i], flux[i], 'b', zorder=0)
        if i == 0:
            ax[0].plot(time[i], spline_flux[i], 'orange', linestyle='--', label='Fitted spline', zorder=1)
            if 'flux_interp' in kwargs:
                ax[0].plot(time[i], kwargs['flux_interp'][i], 'g', label='Gapped transits', zorder=0)
        else:
            ax[0].plot(time[i], spline_flux[i], 'orange', linestyle='--', zorder=1)
            if 'flux_interp' in kwargs:
                ax[0].plot(time[i], kwargs['flux_interp'][i], 'g', zorder=0)
        ax[0].legend()
        ax[1].plot(time[i], flux[i] / spline_flux[i], 'b')

    ax[0].set_ylabel('Amplitude')
    ax[1].set_ylabel('Normalized amplitude')
    ax[1].set_xlabel('Time [day]')
    ax[1].set_title('Spline normalized flux time-series')
    ax[0].set_title('Non-normalized flux time-series')

    f.suptitle('TCE {} {} {}'.format(tce.target_id, tce[config.tce_identifier], tce.label))

    plt.savefig(os.path.join(savedir, '{}_{}_{}_{}.png'.format(tce.target_id, tce[config.tce_identifier], tce.label,
                                                               basename)))

    plt.close()


def plot_centroids_it_oot(all_time, binary_time_all, all_centroids, centroid_oot, avg_centroid_oot, target_coords, tce,
                          config, savedir, basename):
    """ Creates and saves a 2x3 figure with plots that show the out-of-transit and in-transit centroid time-series and
    their averages, as well as the target position, for a given TCE.

    :param all_time: list of numpy arrays, time
    :param binary_time_all: list of numpy arrays, binary arrays with 1 for in-transit cadences and 0 otherwise
    :param all_centroids: dict ('x' and 'y' keys, values are lists of numpy arrays), centroid time-series
    :param centroid_oot: dict ('x' and 'y' keys, values are lists of numpy arrays), out-of-transit centroid time-series
    :param avg_centroid_oot: dict ('x' and 'y' keys), coordinates of the average out-of-transit centroid
    :param target_coords: list, RA and Dec coordinates of the target
    :param tce: pandas Series, row of the input TCE table Pandas DataFrame
    :param config: Config object; preprocessing parameters
    :param savedir: str, filepath to directory in which the figure is saved
    :param basename: str, added to the figure filename
    :return:
    """

    if not config.px_coordinates:
        all_centroids = {coord: [3600 * centroids_arr for centroids_arr in all_centroids[coord]]
                         for coord in all_centroids}
        centroid_oot = {coord: [3600 * centroids_arr for centroids_arr in centroid_oot[coord]]
                        for coord in centroid_oot}
        avg_centroid_oot = {coord: 3600 * avg_centroid_oot[coord] for coord in avg_centroid_oot}
        # avg_centroid_oot = {coord: [3600 * avg_centroid_oot[coord][i] for i in range(len(avg_centroid_oot[coord]))]
        #                     for coord in avg_centroid_oot}
        target_coords = [coord * 3600 for coord in target_coords]

    all_time_oot = [time[np.where(binary_time == 0)] for time, binary_time in zip(all_time, binary_time_all)]

    all_time_it = [time[np.where(binary_time == 1)] for time, binary_time in zip(all_time, binary_time_all)]
    centroid_it = {coord: [centroids[np.where(binary_time == 1)] for binary_time, centroids in
                    zip(binary_time_all, all_centroids[coord])] for coord in all_centroids}
    avg_centroid_it = {coord: np.median(np.concatenate(centroid_it[coord])) for coord in centroid_it}
    # avg_centroid_it = {coord: [np.median(centroid_it[coord][i]) for i in range(len(centroid_it[coord]))]
    #                    for coord in centroid_it}

    plt.figure(figsize=(18, 8))

    plt.suptitle('Centroid time-series\n {} | TCE {} | {}\nTarget: {} (arcsec)'.format(tce['target_id'],
                                                                                       tce[config.tce_identifier],
                                                                                       tce['label'],
                                                                                       target_coords))

    plt.subplot(221)
    for i in range(len(all_time_oot)):
        plt.plot(all_time_oot[i], centroid_oot['x'][i], 'b', zorder=0)
    # plt.plot(np.concatenate(all_time), np.concatenate(all_centroids['x']))
    plt.plot(np.concatenate(all_time_oot), avg_centroid_oot['x'] * np.ones(len(np.concatenate(all_time_oot))), 'r--',
             label='avg oot', zorder=1)
    # plt.plot(np.concatenate(all_time),
    #          np.concatenate([avg_centroid_oot['x'][i] * np.ones(len(all_time[i])) for i in range(len(all_time))]),
    #          'r--', label='avg oot', zorder=1)
    plt.legend()
    if config.px_coordinates:
        plt.ylabel('Col pixel')
    else:
        plt.ylabel('RA [arcsec]')
    plt.title('Out-of-transit points')

    plt.subplot(222)
    for i in range(len(all_time_it)):
        # plt.scatter(all_time_it[i], centroid_it['x'][i], color='c', zorder=0)
        plt.plot(all_time_it[i], centroid_it['x'][i], 'b', zorder=0)
    plt.plot(np.concatenate(all_time_it), avg_centroid_it['x'] * np.ones(len(np.concatenate(all_time_it))), 'g--',
             label='avg it', zorder=1)
    # plt.plot(np.concatenate(all_time),
    #          np.concatenate([avg_centroid_it['x'][i] * np.ones(len(all_time[i])) for i in range(len(all_time))]),
    #          'g--', label='avg it', zorder=1)
    plt.legend()
    plt.title('In-transit points')

    plt.subplot(223)
    for i in range(len(all_time_oot)):
        plt.plot(all_time_oot[i], centroid_oot['y'][i], 'b', zorder=0)
    # plt.plot(np.concatenate(all_time), np.concatenate(all_centroids['y']))
    plt.plot(np.concatenate(all_time_oot), avg_centroid_oot['y'] * np.ones(len(np.concatenate(all_time_oot))), 'r--',
             label='avg oot', zorder=1)
    # plt.plot(np.concatenate(all_time),
    #          np.concatenate([avg_centroid_oot['y'][i] * np.ones(len(all_time[i])) for i in range(len(all_time))]),
    #          'r--', label='avg oot', zorder=1)

    plt.legend()
    plt.xlabel('Time [day]')
    if config.px_coordinates:
        plt.ylabel('Row pixel')
    else:
        plt.ylabel('Dec [arcsec]')

    plt.subplot(224)
    for i in range(len(all_time_it)):
        # plt.scatter(all_time_it[i], centroid_it['y'][i], color='c', zorder=0)
        plt.plot(all_time_it[i], centroid_it['y'][i], 'b', zorder=0)
    plt.plot(np.concatenate(all_time_it), avg_centroid_it['y'] * np.ones(len(np.concatenate(all_time_it))), 'g--',
             label='avg it', zorder=1)
    # plt.plot(np.concatenate(all_time),
    #          np.concatenate([avg_centroid_it['y'][i] * np.ones(len(all_time[i])) for i in range(len(all_time))]),
    #          'g--', label='avg it', zorder=1)
    plt.legend()
    plt.xlabel('Time [day]')

    plt.savefig(os.path.join(savedir, '{}_{}_{}_{}.png'.format(tce.target_id, tce[config.tce_identifier], tce.label,
                                                               basename)))

    plt.close()


def plot_corrected_centroids(all_time, all_centroids, avg_centroid_oot, target_coords, tce, config, savedir,
                             basename):
    """ Creates and saves a 2x2 figure with plots that show the corrected centroid time-series and the respective
    out-of-transit centroid, as well as the target position, for a given TCE.

    :param all_time: list of numpy arrays, time
    :param all_centroids: dict ('x' and 'y' keys, values are lists of numpy arrays), centroid time-series
    :param avg_centroid_oot: dict ('x' and 'y' keys), coordinates of the average out-of-transit centroid
    :param target_coords: list, RA and Dec coordinates of the target
    :param tce: pandas Series, row of the input TCE table Pandas DataFrame.
    :param config: Config object; preprocessing parameters.
    :param savedir: str, filepath to directory in which the figure is saved
    :param basename: str, added to the figure filename
    :return:
    """

    if not config.px_coordinates:
        all_centroids = {coord: [3600 * centroids_arr for centroids_arr in all_centroids[coord]]
                         for coord in all_centroids}
        avg_centroid_oot = {coord: 3600 * avg_centroid_oot[coord] for coord in avg_centroid_oot}
        # avg_centroid_oot = {coord: [3600 * avg_centroid_oot[coord][i] for i in range(len(avg_centroid_oot[coord]))]
        #                     for coord in avg_centroid_oot}
        target_coords = [coord * 3600 for coord in target_coords]

    plt.figure(figsize=(20, 8))

    plt.suptitle('{} | TCE {} | {}\nTarget: {} (arcsec)'.format(tce['target_id'], tce[config.tce_identifier],
                                                                tce['label'], target_coords))

    plt.subplot(211)
    for i in range(len(all_time)):
        plt.plot(all_time[i], all_centroids['x'][i], 'b', zorder=0)
    plt.plot(np.concatenate(all_time), avg_centroid_oot['x'] * np.ones(len(np.concatenate(all_time))), 'r--',
             label='avg oot', zorder=1)
    # plt.plot(np.concatenate(all_time),
    #          np.concatenate([avg_centroid_oot['x'][i] * np.ones(len(all_time[i])) for i in range(len(all_time))]),
    #          'r--', label='avg oot', zorder=1)
    plt.legend()
    if config.px_coordinates:
        plt.ylabel('Col pixel')
    else:
        plt.ylabel('RA [arcsec]')
    plt.title('Corrected centroid time-series')

    plt.subplot(212)
    for i in range(len(all_time)):
        plt.plot(all_time[i], all_centroids['y'][i], 'b', zorder=0)
    plt.plot(np.concatenate(all_time), avg_centroid_oot['y'] * np.ones(len(np.concatenate(all_time))), 'r--',
             label='avg oot', zorder=1)
    # plt.plot(np.concatenate(all_time),
    #          np.concatenate([avg_centroid_oot['y'][i] * np.ones(len(all_time[i])) for i in range(len(all_time))]),
    #          'r--', label='avg oot', zorder=1)
    plt.legend()
    if config.px_coordinates:
        plt.ylabel('Row pixel')
    else:
        plt.ylabel('Dec [arcsec]')
    plt.xlabel('Time [day]')

    plt.savefig(os.path.join(savedir, '{}_{}_{}_{}.png'.format(tce.target_id, tce[config.tce_identifier], tce.label,
                                                               basename)))

    plt.close()


def plot_dist_centroids(time, centroid_dist, centroid_dist_spline, avg_centroid_dist_oot, tce,
                        config, savedir, basename, pxcoordinates=False):
    """ Creates and saves a figure with plots that show the centroid-to-target distance and, if desired, the fitted
    spline and the respective spline normalized centroid-to-target distance, for a given TCE.

    :param time: list of numpy arrays, time
    :param centroid_dist: list of numpy arrays, centroid-to-target distance
    :param centroid_dist_spline: list of numpy arrays, spline fitted to centroid-to-target distance
    :param avg_centroid_dist_oot: float, average out-of-transit centroid-to-target distance
    :param tce: pandas Series, row of the input TCE table Pandas DataFrame
    :param config: Config object; preprocessing parameters
    :param savedir: str, filepath to directory in which the figure is saved
    :param basename: str, added to the figure filename
    :return:
    """

    if centroid_dist_spline is None:

        f, ax = plt.subplots(figsize=(16, 10))

        for i in range(len(centroid_dist)):
            ax.plot(time[i], centroid_dist[i], 'b')

        if config.px_coordinates or pxcoordinates:
            ax.set_ylabel('Euclidean distance [pixel]')
        else:
            ax.set_ylabel('Angular distance [arcsec]')
        ax.set_title('Centroid-to-target distance time-series')
        ax.set_xlabel('Time [day]')

        f.suptitle('TCE {} {} {}'.format(tce.target_id, tce[config.tce_identifier], tce.label))
        plt.savefig(os.path.join(savedir, '{}_{}_{}_{}.png'.format(tce.target_id, tce[config.tce_identifier], tce.label,
                                                                   basename)))

        plt.close()

    else:
        time, centroid_quadr, centroid_quadr_spline = [np.concatenate(time)], [np.concatenate(centroid_dist)], \
                                                      [np.concatenate(centroid_dist_spline)]

        f, ax = plt.subplots(2, 1, figsize=(16, 10))

        for i in range(len(time)):
            ax[0].plot(time[i], centroid_dist[i], 'b', zorder=0)
            if i == 0:
                ax[0].plot(time[i], centroid_dist_spline[i], linestyle='--', label='spline', zorder=1)
            else:
                ax[0].plot(time[i], centroid_dist_spline[i], linestyle='--', zorder=1)
            ax[1].plot(time[i], centroid_dist[i] / centroid_dist_spline[i] * avg_centroid_dist_oot, 'b')

        if config.px_coordinates or pxcoordinates:
            ax[0].set_ylabel('Euclidean distance [pixel]')
        else:
            ax[0].set_ylabel('Angular distance [arcsec]')
        ax[0].set_title('Centroid-to-target distance time-series')
        ax[0].legend()
        ax[1].set_ylabel('Centroid-to-target distance')
        ax[1].set_xlabel('Time [day]')
        ax[1].set_title('Spline normalized centroid-to-target distance time-series')

        f.suptitle('TCE {} {} {}'.format(tce.target_id, tce[config.tce_identifier], tce.label))
        plt.savefig(os.path.join(savedir, '{}_{}_{}_{}.png'.format(tce.target_id, tce[config.tce_identifier], tce.label,
                                                                   basename)))

        plt.close()


def plot_centroids_views(glob_view_centr, loc_view_centr, tce, config, savedir, basename):
    """ Creates and saves a 2x1 figure with plots that show the global and local views for the centroid time-series for
     a given TCE.

    :param glob_view_centr: numpy array, global centroid view
    :param loc_view_centr: numpy array, local centroid view
    :param tce: pandas Series, row of the input TCE table Pandas DataFrame
    :param config: Config object; preprocessing parameters
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

    f.suptitle('TCE {} {} {}'.format(tce.target_id, tce[config.tce_identifier], tce.label))
    plt.savefig(os.path.join(savedir, '{}_{}_{}_{}.png'.format(tce.target_id, tce[config.tce_identifier], tce.label,
                                                               basename)))

    plt.close()


def plot_fluxandcentroids_views(glob_view, loc_view, glob_view_centr, loc_view_centr, tce, config, savedir, basename):
    """ Creates and saves a 2x2 figure with plots that show the global and local views for the flux and centroid
    time-series for a given TCE.

    :param glob_view: numpy array, global flux view
    :param loc_view: numpy array, local flux view
    :param glob_view_centr: numpy array, global centroid view
    :param loc_view_centr: numpy array, local centroid view
    :param tce: pandas Series, row of the input TCE table Pandas DataFrame
    :param config: Config object; preprocessing parameters.
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

    f.suptitle('TCE {} {} {}'.format(tce.target_id, tce[config.tce_identifier], tce.label))
    plt.savefig(os.path.join(savedir, '{}_{}_{}_{}.png'.format(tce.target_id, tce[config.tce_identifier], tce.label,
                                                               basename)))

    plt.close()


def plot_all_views(views, tce, config, scheme, savedir, basename, num_transits):
    """ Creates and saves a figure with plots that show views for a given TCE.

    :param views: dict, views to be plotted
    :param tce: pandas Series, row of the input TCE table Pandas DataFrame
    :param config: Config object; preprocessing parameters.
    :param scheme: list, defines the number and position of the view plots in the figure ([number of plots per row,
    number of plots per column])
    :param savedir: str, filepath to directory in which the figure is saved
    :param basename: str, added to the figure filename
    :param num_transits: dict, number of transits for each view
    :return:
    """

    global_phase = np.linspace(-tce['tce_period'] / 2, tce['tce_period'] / 2, config.num_bins_glob, endpoint=True)
    local_phase = np.linspace(-tce['tce_duration'] * config.num_durations, tce['tce_duration'] * config.num_durations,
                              config.num_bins_loc, endpoint=True)

    scalarParamsStr = ''
    for scalarParam_i in range(len(config.scalar_params)):
        if scalarParam_i % 5 == 0:
            scalarParamsStr += '\n'
        if config.scalar_params[scalarParam_i] in ['boot_fap']:
            scalarParamsStr += '{}={:.4E}  '.format(config.scalar_params[scalarParam_i],
                                                    tce[config.scalar_params[scalarParam_i]])
        elif config.scalar_params[scalarParam_i] in ['tce_rb_tcount0', 'tce_steff']:
            scalarParamsStr += '{}={}  '.format(config.scalar_params[scalarParam_i],
                                                int(tce[config.scalar_params[scalarParam_i]]))
        else:
            scalarParamsStr += '{}={:.4f}  '.format(config.scalar_params[scalarParam_i],
                                                    tce[config.scalar_params[scalarParam_i]])

    ephemerisStr = 'Epoch={:.4f}, Period={:.4f}, Transit Duration={:.4f}, Transit Depth={:.4f}'.format(
        tce['tce_time0bk'],
        tce['tce_period'],
        tce['tce_duration'] * 24,
        tce['transit_depth'])

    f, ax = plt.subplots(scheme[0], scheme[1], figsize=(20, 14))
    k = 0
    views_list = list(views.keys())
    for i in range(scheme[0]):
        for j in range(scheme[1]):
            if k < len(views_list):
                ax[i, j].plot(views[views_list[k]])
                ax[i, j].scatter(np.arange(len(views[views_list[k]])), views[views_list[k]], s=10, color='k', alpha=0.2)
                # ax2 = ax[i, j].twiny()
                # if 'global' in views_list[k]:
                #     ax2.plot(global_phase, views[views_list[k]])
                #     ax2.set_xlim([global_phase[0], global_phase[-1]])
                # else:
                #     ax2.plot(local_phase, views[views_list[k]])
                #     ax2.set_xlim([local_phase[0], local_phase[-11]])
                # ax2.grid(True)
                if views_list[k] in num_transits:
                    ax[i, j].set_title('{} N_transits={}'.format(views_list[k], num_transits[views_list[k]]), pad=20)
                else:
                    ax[i, j].set_title('{}'.format(views_list[k]), pad=20)
                ax[i, j].set_xlim([0, len(views[views_list[k]])])
            if i == scheme[0] - 1:
                ax[i, j].set_xlabel('Bin number')
            if j == 0:
                ax[i, j].set_ylabel('Amplitude')

            k += 1

    f.suptitle('TCE {}-{} {} | {}\n{}'.format(tce.target_id, tce[config.tce_identifier], tce.label, ephemerisStr,
                                              scalarParamsStr))
    plt.subplots_adjust(hspace=0.36, wspace=0.37, top=0.85, right=0.95, bottom=0.07, left=0.07)
    plt.savefig(os.path.join(savedir, '{}_{}_{}_{}.png'.format(tce.target_id, tce[config.tce_identifier], tce.label,
                                                               basename)))
    plt.close()


def plot_wks(glob_view, glob_view_weak_secondary, tce, config, savedir, basename):
    """ Creates and saves a figure with plots of the global flux view and global weak secondary flux view for a given
    TCE.

    :param glob_view: NumPy array, global flux view
    :param glob_view_weak_secondary: NumPy array, global weak secondary flux view
    :param tce: pandas Series, row of the input TCE table Pandas DataFrame
    :param config: Config object; preprocessing parameters.
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
    ax.set_title('TCE {} {} {}'.format(tce.target_id, tce[config.tce_identifier], tce.label))
    plt.savefig(os.path.join(savedir, '{}_{}_{}_{}.png'.format(tce.target_id, tce[config.tce_identifier], tce.label,
                                                               basename)))
    plt.close()


def plot_phasefolded(time, timeseries, tce, config, savedir, basename):
    """ Creates and saves a figure with plots of the phase folded time series for a given TCE.

    :param time: Numpy array, timestamps
    :param timeseries: NumPy array, timeseries
    :param tce: pandas Series, row of the input TCE table Pandas DataFrame
    :param config: Config object; preprocessing parameters.
    :param savedir: str, filepath to directory in which the figure is saved
    :param basename: str, added to the figure filename
    :return:
    """

    f, ax = plt.subplots()
    ax.plot(time, timeseries)
    ax.set_xlim([time[0], time[-1]])
    ax.set_ylabel('Amplitude')
    ax.set_xlabel('Phase')
    ax.set_title('TCE {} {} {}'.format(tce.target_id, tce[config.tce_identifier], tce.label))
    plt.savefig(os.path.join(savedir, '{}_{}_{}_{}.png'.format(tce.target_id, tce[config.tce_identifier], tce.label,
                                                               basename)))
    plt.close()


def plot_all_phasefoldedtimeseries(timeseries, tce, config, scheme, savedir, basename):
    """ Creates and saves a figure with plots that show phase folded and binned time series for a given TCE.

    :param timeseries: dict, views to be plotted
    :param tce: pandas Series, row of the input TCE table Pandas DataFrame
    :param config: Config object; preprocessing parameters.
    :param scheme: list, defines the number and position of the view plots in the figure ([number of plots per row,
    number of plots per column])
    :param savedir: str, filepath to directory in which the figure is saved
    :param basename: str, added to the figure filename
    :return:
    """

    f, ax = plt.subplots(scheme[0], scheme[1], figsize=(20, 14))
    k = 0
    views_list = list(timeseries.keys())
    for i in range(scheme[0]):
        for j in range(scheme[1]):
            if k < len(views_list):
                ax[i, j].plot(timeseries[views_list[k]][0], timeseries[views_list[k]][1])
                ax[i, j].set_title(views_list[k], pad=20)
                ax[i, j].set_xlim([timeseries[views_list[k]][0][0], timeseries[views_list[k]][0][-1]])
            if i == scheme[0] - 1:
                ax[i, j].set_xlabel('Phase')
            if j == 0:
                ax[i, j].set_ylabel('Amplitude')
            k += 1

    f.suptitle('TCE {} {} {}'.format(tce.target_id, tce[config.tce_identifier], tce.label))
    # plt.subplots_adjust(hspace=0.3)
    plt.savefig(os.path.join(savedir, '{}_{}_{}_{}.png'.format(tce.target_id, tce[config.tce_identifier], tce.label,
                                                               basename)))

    # f.tight_layout(rect=[0, 0.03, 1, 0.95])

    plt.close()


def plot_diff_oddeven(timeseries, tce, config, savedir, basename):
    """ Creates and saves a figure with plots that show the absolute difference between odd and even views.

    :param timeseries: dict, views to be plotted
    :param tce: pandas Series, row of the input TCE table Pandas DataFrame
    :param config: Config object; preprocessing parameters.
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

    f.suptitle('TCE {} {} {}'.format(tce.target_id, tce[config.tce_identifier], tce.label))
    plt.savefig(os.path.join(savedir, '{}_{}_{}_{}.png'.format(tce.target_id, tce[config.tce_identifier], tce.label,
                                                               basename)))

    plt.close()


def plot_phasefolded_and_binned(timeseries, binned_timeseries, tce, config, scheme, savedir, basename):
    """ Creates and saves a figure with plots that show phase folded and binned time series for a given TCE.

    :param timeseries: dict, phase folded time series
    :param binned_timeseries: dict, binned views
    :param tce: pandas Series, row of the input TCE table Pandas DataFrame
    :param config: Config object; preprocessing parameters.
    :param scheme: list, defines the number and position of the view plots in the figure ([number of plots per row,
    number of plots per column])
    :param savedir: str, filepath to directory in which the figure is saved
    :param basename: str, added to the figure filename
    :return:
    """

    gs = gridspec.GridSpec(4, 2)

    local_view_time_interval = tce['tce_duration'] * (config.num_durations)

    f = plt.figure(figsize=(20, 14))

    ax = plt.subplot(gs[0, :])
    ax.scatter(timeseries['Flux'][0], timeseries['Flux'][1], color='k', s=5)
    ax.scatter(binned_timeseries['Global Flux'][0], binned_timeseries['Global Flux'][1], color='b')
    ax.plot(binned_timeseries['Global Flux'][0], binned_timeseries['Global Flux'][1], 'b')
    ax.set_ylabel('Relative Flux')
    ax.set_xlabel('Phase (day)')
    ax.set_xlim([timeseries['Flux'][0][0], timeseries['Flux'][0][-1]])

    left_idx = np.where(timeseries['Flux'][0] > -local_view_time_interval)[0][0]
    right_idx = np.where(timeseries['Flux'][0] < local_view_time_interval)[0][-1]
    ax = plt.subplot(gs[1, 0])
    ax.scatter(timeseries['Flux'][0][left_idx:right_idx] * 24, timeseries['Flux'][1][left_idx:right_idx],
               color='k', s=5)
    ax.scatter(binned_timeseries['Local Flux'][0] * 24, binned_timeseries['Local Flux'][1], color='b')
    ax.plot(binned_timeseries['Local Flux'][0] * 24, binned_timeseries['Local Flux'][1], 'b')
    ax.set_ylabel('Relative Flux')
    ax.set_xlabel('Phase (hour)')
    ax.set_xlim([timeseries['Flux'][0][left_idx] * 24, timeseries['Flux'][0][right_idx] * 24])

    if 'Weak Secondary Flux' in timeseries:
        left_idx = np.where(timeseries['Weak Secondary Flux'][0] > -local_view_time_interval)[0][0]
        right_idx = np.where(timeseries['Weak Secondary Flux'][0] < local_view_time_interval)[0][-1]
        ax = plt.subplot(gs[1, 1])
        ax.scatter(timeseries['Weak Secondary Flux'][0][left_idx:right_idx] * 24,
                   timeseries['Weak Secondary Flux'][1][left_idx:right_idx], color='k', s=5)
        ax.scatter(binned_timeseries['Local Weak Secondary Flux'][0] * 24,
                   binned_timeseries['Local Weak Secondary Flux'][1], color='b')
        ax.plot(binned_timeseries['Local Weak Secondary Flux'][0] * 24,
                binned_timeseries['Local Weak Secondary Flux'][1], 'b')
        ax.set_ylabel('Relative Flux')
        ax.set_xlabel('Phase (hour)')
        ax.set_xlim([timeseries['Weak Secondary Flux'][0][left_idx] * 24,
                     timeseries['Weak Secondary Flux'][0][right_idx] * 24])
        ax.set_title('Weak Secondary Phase : {} Days'.format(tce['tce_maxmesd']))

    left_idx = np.where(timeseries['Odd Flux'][0] > -local_view_time_interval)[0][0]
    right_idx = np.where(timeseries['Odd Flux'][0] < local_view_time_interval)[0][-1]
    ax = plt.subplot(gs[2, 0])
    ax.scatter(timeseries['Odd Flux'][0][left_idx:right_idx] * 24, timeseries['Odd Flux'][1][left_idx:right_idx],
               color='k', s=5)
    ax.scatter(binned_timeseries['Local Odd Flux'][0] * 24, binned_timeseries['Local Odd Flux'][1], color='b')
    ax.plot(binned_timeseries['Local Odd Flux'][0] * 24, binned_timeseries['Local Odd Flux'][1], 'b')
    ax.set_ylabel('Relative Flux')
    ax.set_xlabel('Phase (hour)')
    ax.set_xlim([timeseries['Odd Flux'][0][left_idx] * 24, timeseries['Odd Flux'][0][right_idx] * 24])
    ax.set_title('Odd')

    left_idx = np.where(timeseries['Even Flux'][0] > -local_view_time_interval)[0][0]
    right_idx = np.where(timeseries['Even Flux'][0] < local_view_time_interval)[0][-1]
    ax = plt.subplot(gs[2, 1])
    ax.scatter(timeseries['Even Flux'][0][left_idx:right_idx] * 24, timeseries['Even Flux'][1][left_idx:right_idx],
               color='k', s=5)
    ax.scatter(binned_timeseries['Local Even Flux'][0] * 24, binned_timeseries['Local Even Flux'][1], color='b')
    ax.plot(binned_timeseries['Local Even Flux'][0] * 24, binned_timeseries['Local Even Flux'][1], 'b')
    ax.set_ylabel('Relative Flux')
    ax.set_xlabel('Phase (hour)')
    ax.set_xlim([timeseries['Even Flux'][0][left_idx] * 24, timeseries['Even Flux'][0][right_idx] * 24])
    ax.set_title('Even')

    ax = plt.subplot(gs[3, 0])
    ax.scatter(timeseries['Centroid Offset Distance'][0], timeseries['Centroid Offset Distance'][1], color='k', s=5)
    ax.scatter(binned_timeseries['Global Centroid Offset Distance'][0],
               binned_timeseries['Global Centroid Offset Distance'][1], color='b')
    ax.plot(binned_timeseries['Global Centroid Offset Distance'][0],
            binned_timeseries['Global Centroid Offset Distance'][1], 'b')
    ax.set_ylabel('Offset distance (arcsec)')
    ax.set_xlabel('Phase (day)')
    ax.set_xlim([timeseries['Centroid Offset Distance'][0][0],
                 timeseries['Centroid Offset Distance'][0][-1]])

    left_idx = np.where(timeseries['Centroid Offset Distance'][0] > -local_view_time_interval)[0][0]
    right_idx = np.where(timeseries['Centroid Offset Distance'][0] < local_view_time_interval)[0][-1]
    ax = plt.subplot(gs[3, 1])
    ax.scatter(timeseries['Centroid Offset Distance'][0][left_idx:right_idx] * 24,
            timeseries['Centroid Offset Distance'][1][left_idx:right_idx],
            color='k', s=5)
    ax.scatter(binned_timeseries['Local Centroid Offset Distance'][0] * 24,
               binned_timeseries['Local Centroid Offset Distance'][1], color='b')
    ax.plot(binned_timeseries['Local Centroid Offset Distance'][0] * 24,
            binned_timeseries['Local Centroid Offset Distance'][1], 'b')
    ax.set_ylabel('Offset distance (arcsec)')
    ax.set_xlabel('Phase (hour)')
    ax.set_xlim([timeseries['Centroid Offset Distance'][0][left_idx] * 24,
                 timeseries['Centroid Offset Distance'][0][right_idx] * 24])

    f.suptitle('TCE {} {} {}'.format(tce.target_id, tce[config.tce_identifier], tce.label))
    plt.subplots_adjust(
        hspace=0.526,
        wspace=0.202,
        top=0.943,
        bottom=0.06,
        left=0.057,
        right=0.98
    )
    plt.savefig(os.path.join(savedir, '{}_{}_{}_{}.png'.format(tce.target_id, tce[config.tce_identifier], tce.label,
                                                               basename)))
    plt.close()
