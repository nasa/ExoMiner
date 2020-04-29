"""
Auxiliary functions used to plot outcome from different steps along the preprocessing pipeline

# TODO: generalize this code for TESS - change keywords, quarters and modules, ...
"""

# 3rd party
import os
import numpy as np
import matplotlib.pyplot as plt
if 'nobackup' in os.path.abspath(__file__):
    plt.switch_backend('agg')


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

    # if config.satellite == 'kepler':
    #     plt.suptitle('Kepler ID {} | TCE {} | {}'.format(tce['kepid'], tce['tce_plnt_num'], tce['av_training_set']))
    # else:
    #     plt.suptitle('TIC ID {} | TCE {} | {}'.format(tce['tic'], tce['tce_plnt_num'], tce['disposition']))

    plt.suptitle('TCE {} | {} | {}'.format(tce['target_id'], tce[config.tce_identifier], tce['label']))

    ax[0].plot(np.concatenate(all_time), np.concatenate(binary_time_all))
    for time in all_time:
        ax[0].axvline(x=time[-1], ymax=1, ymin=0, c='r')
    ax[0].set_title('Binary time-series')
    ax[0].set_ylabel('Binary amplitude (it-oot)')

    ax[1].plot(np.concatenate(all_time), np.concatenate(all_flux))
    for time in all_time:
        ax[1].axvline(x=time[-1], ymax=1, ymin=0, c='r')
    ax[1].set_title('Flux')
    ax[1].set_ylabel('Amplitude')
    ax[1].set_xlabel('Time [day]')

    # if config.satellite == 'kepler':
    #     plt.savefig('{}{}_{}_{}_{}.png'.format(savedir, tce.kepid, tce.tce_plnt_num, tce.av_training_set, basename))
    # else:
    #     plt.savefig('{}{}_{}_{}_{}.png'.format(savedir, tce.tic, tce.tce_plnt_num, tce.disposition, basename))

    plt.savefig(os.path.join(savedir, '{}_{}_{}_{}.png'.format(tce.target_id, tce[config.tce_identifier], tce.label,
                                                               basename)))

    plt.close()


def plot_centroids(time, centroids, centroids_spline, tce, config, savedir, basename, add_info=None):
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

    if not config.px_coordinates:
        centroids = {coord: [3600 * centroids_arr for centroids_arr in centroids[coord]] for coord in centroids}
        if centroids_spline is not None:
            centroids_spline = {coord: [3600 * spline_arr for spline_arr in centroids_spline[coord]]
                                for coord in centroids_spline}

    if centroids_spline is None:
        f, ax = plt.subplots(2, 1, figsize=(16, 14))

        for i, centroids_arr in enumerate(zip(centroids['x'], centroids['y'])):
            ax[0].plot(time[i], centroids_arr[0])
            ax[1].plot(time[i], centroids_arr[1])

        if config.px_coordinates:
            ax[0].set_ylabel('Col pixel')
            ax[1].set_ylabel('Row pixel')
        else:
            ax[0].set_ylabel('RA [arcsec]')
            ax[1].set_ylabel('Dec [arcsec]')

        ax[1].set_xlabel('Time [day]')

        # if max_gap is not None:
        #     ax[0].set_title('TCE {} {} {}\n{}'.format(tce.kepid, tce.tce_plnt_num, tce.av_training_set,
        #                                               [round(el, 5) for el in max_gap]))
        #     ax[0].title.set_fontsize(10)
        # else:
        if config.satellite == 'kepler':
            # # ax[0].set_title('TCE {} {} {}'.format(tce.kepid, tce.tce_plnt_num, tce.av_training_set))
            # ax[0].set_title('TCE {} {} {}'.format(tce.target_id, tce.tce_plnt_num, tce.label))

            if add_info is not None:
                f.suptitle('Quarters: {}\nModules: {}'.format(add_info['quarter'], add_info['module']))

        #     # plt.savefig('{}{}_{}_{}_{}.png'.format(savedir, tce.kepid, tce.tce_plnt_num, tce.av_training_set, basename))
        #     plt.savefig('{}{}_{}_{}_{}.png'.format(savedir, tce.target_id, tce.tce_plnt_num, tce.label, basename))
        # else:
        #     # ax[0].set_title('TCE {} {} {}'.format(tce.tic, tce.tce_plnt_num, tce.disposition))
        #     # ax[0].set_title('TCE {} {} {}'.format(tce.target_id, tce.tce_plnt_num, tce.label))
        #     ax[0].set_title('TCE {} {} {}'.format(tce.target_id, tce[config.tce_identifier], tce.label))
        #
        #     # plt.savefig('{}{}_{}_{}_{}.png'.format(savedir, tce.tic, tce.tce_plnt_num, tce.disposition, basename))
        #     # plt.savefig('{}{}_{}_{}_{}.png'.format(savedir, tce.target_id, tce.tce_plnt_num, tce.label, basename))
        #     plt.savefig('{}{}_{}_{}_{}.png'.format(savedir, tce.target_id, tce[config.tce_identifier], tce.label, basename))

        ax[0].set_title('TCE {} {} {}'.format(tce.target_id, tce[config.tce_identifier], tce.label))

        plt.savefig(os.path.join(savedir, '{}_{}_{}_{}.png'.format(tce.target_id, tce[config.tce_identifier], tce.label,
                                                                   basename)))

        plt.close()

    else:
        time, centroids, centroids_spline = [np.concatenate(time)], \
                                            {coord: [np.concatenate(centroids[coord])] for coord in centroids}, \
                                            {coord: [np.concatenate(centroids_spline[coord])] for coord in centroids}

        f, ax = plt.subplots(2, 2, figsize=(18, 12))

        for i, centroids_arr in enumerate(zip(centroids['x'], centroids['y'])):
            ax[0, 0].plot(time[i], centroids_arr[0], zorder=0)
            ax[0, 0].plot(time[i], centroids_spline['x'][i], linestyle='--', label='Fitted spline', zorder=1)
            ax[0, 0].legend()
            ax[1, 0].plot(time[i], centroids_arr[1], zorder=0)
            ax[1, 0].plot(time[i], centroids_spline['y'][i], linestyle='--', label='Fitted spline', zorder=1)
            ax[1, 0].legend()

            ax[0, 1].plot(time[i], centroids_arr[0] / centroids_spline['x'][i])
            ax[1, 1].plot(time[i], centroids_arr[1] / centroids_spline['y'][i])

        if config.px_coordinates:
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

        # if config.satellite == 'kepler':
        #     if add_info is not None:
        #         # f.suptitle('Quarters: {}\nModules: {}\nKepler ID {}'.format(add_info['quarter'], add_info['module'],
        #         #                                                             tce.kepid))
        #         f.suptitle('Quarters: {}\nModules: {}\nKepler ID {}'.format(add_info['quarter'], add_info['module'],
        #                                                                     tce.target_id))
        #     else:
        #         # f.suptitle('TCE {} {} {}'.format(tce.kepid, tce.tce_plnt_num, tce.av_training_set))
        #         f.suptitle('TCE {} {} {}'.format(tce.target_id, tce.tce_plnt_num, tce.label))
        #
        #     # plt.savefig('{}{}_{}_{}_{}.png'.format(savedir, tce.kepid, tce.tce_plnt_num,
        #     #                                        tce.av_training_set, basename))
        #     plt.savefig('{}{}_{}_{}_{}.png'.format(savedir, tce.target_id, tce.tce_plnt_num, tce.label, basename))
        # else:
        #     # f.suptitle('TCE {} {} {}'.format(tce.disposition, tce.tce_plnt_num, tce.disposition))
        #     f.suptitle('TCE {} {} {}'.format(tce.target_id, tce[config.tce_identifier], tce.label))
        #
        #     # plt.savefig('{}{}_{}_{}_{}.png'.format(savedir, tce.tic, tce.tce_plnt_num, tce.disposition, basename))
        #     plt.savefig('{}{}_{}_{}_{}.png'.format(savedir, tce.target_id, tce[config.tce_identifier], tce.label, basename))

        f.suptitle('TCE {} {} {}'.format(tce.target_id, tce[config.tce_identifier], tce.label))

        plt.savefig(os.path.join(savedir, '{}_{}_{}_{}.png'.format(tce.target_id, tce[config.tce_identifier], tce.label,
                                                                   basename)))

        plt.close()


def plot_flux_fit_spline(time, flux, spline_flux, tce, config, savedir, basename):
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

    time, flux, spline_flux = [np.concatenate(time)], [np.concatenate(flux)], [np.concatenate(spline_flux)]

    f, ax = plt.subplots(2, 1, figsize=(16, 10))
    for i in range(len(flux)):
        ax[0].plot(time[i], flux[i], zorder=0)
        ax[0].plot(time[i], spline_flux[i], linestyle='--', label='Fitted spline', zorder=1)
        ax[0].legend()
        ax[1].plot(time[i], flux[i] / spline_flux[i])

    ax[0].set_ylabel('Amplitude')
    ax[1].set_ylabel('Normalized amplitude')
    ax[1].set_xlabel('Time [day]')
    ax[1].set_title('Spline normalized flux time-series')
    ax[0].set_title('Non-normalized flux time-series')

    # if config.satellite == 'kepler':
    #     # f.suptitle('TCE {} {} {}'.format(tce.kepid, tce.tce_plnt_num, tce.av_training_set))
    #     #
    #     # plt.savefig('{}{}_{}_{}_{}.png'.format(savedir, tce.kepid, tce.tce_plnt_num,
    #     #                                        tce.av_training_set, basename))
    #
    #     f.suptitle('TCE {} {} {}'.format(tce.target_id, tce.tce_plnt_num, tce.label))
    #
    #     plt.savefig('{}{}_{}_{}_{}.png'.format(savedir, tce.target_id, tce.tce_plnt_num, tce.label, basename))
    # else:
    #     # f.suptitle('TCE {} {} {}'.format(tce.tic, tce.tce_plnt_num, tce.disposition))
    #     #
    #     # plt.savefig('{}{}_{}_{}_{}.png'.format(savedir, tce.tic, tce.tce_plnt_num, tce.disposition, basename))
    #
    #     f.suptitle('TCE {} {} {}'.format(tce.target_id, tce[config.tce_identifier], tce.label))
    #
    #     plt.savefig('{}{}_{}_{}_{}.png'.format(savedir, tce.target_id, tce[config.tce_identifier], tce.label, basename))

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

    # if config.satellite == 'kepler':
    #     # plt.suptitle('Centroid time-series\n'
    #     #              'Kepler ID {} | TCE {} | {}\nTarget: {} (arcsec)'.format(tce['kepid'], tce['tce_plnt_num'],
    #     #                                                                       tce['av_training_set'], target_coords))
    #     plt.suptitle('Centroid time-series\n'
    #                  'Kepler ID {} | TCE {} | {}\nTarget: {} (arcsec)'.format(tce['target_id'], tce['tce_plnt_num'],
    #                                                                           tce['label'], target_coords))
    # else:
    #     # plt.suptitle('Centroid time-series\nTIC ID {} | TCE {} | {}\nTarget: {} (arcsec)'.format(tce['tic'],
    #     #                                                                                         tce['tce_plnt_num'],
    #     #                                                                                         tce['disposition'],
    #     #                                                                                         target_coords))
    #
    #     plt.suptitle('Centroid time-series\nTIC ID {} | TCE {} | {}\nTarget: {} (arcsec)'.format(tce['target_id'],
    #                                                                                              tce['Full TOI ID'],
    #                                                                                              tce['label'],
    #                                                                                              target_coords))

    plt.suptitle('Centroid time-series\n {} | TCE {} | {}\nTarget: {} (arcsec)'.format(tce['target_id'],
                                                                                             tce[config.tce_identifier],
                                                                                             tce['label'],
                                                                                             target_coords))

    plt.subplot(221)
    # plt.plot(np.concatenate(all_time), np.concatenate(all_centroids['x']))
    plt.plot(np.concatenate(all_time), avg_centroid_oot['x'] * np.ones(len(np.concatenate(all_time))), 'r--',
             label='avg oot', zorder=1)
    # plt.plot(np.concatenate(all_time),
    #          np.concatenate([avg_centroid_oot['x'][i] * np.ones(len(all_time[i])) for i in range(len(all_time))]),
    #          'r--', label='avg oot', zorder=1)
    plt.plot(np.concatenate(all_time_oot), np.concatenate(centroid_oot['x']), zorder=0)
    plt.legend()
    if config.px_coordinates:
        plt.ylabel('Col pixel')
    else:
        plt.ylabel('RA [arcsec]')
    plt.title('Out-of-transit points')
    plt.subplot(222)
    plt.scatter(np.concatenate(all_time_it), np.concatenate(centroid_it['x']), color='c', zorder=0)
    plt.plot(np.concatenate(all_time_it), avg_centroid_it['x'] * np.ones(len(np.concatenate(all_time_it))), 'g--',
             label='avg it', zorder=1)
    # plt.plot(np.concatenate(all_time),
    #          np.concatenate([avg_centroid_it['x'][i] * np.ones(len(all_time[i])) for i in range(len(all_time))]),
    #          'g--', label='avg it', zorder=1)
    plt.legend()
    plt.title('In-transit points')
    # plt.subplot(223)
    # plt.plot(np.concatenate(all_time), target_coords[0] * np.ones(len(np.concatenate(all_time))), c='y', label='target')
    # plt.title('Target')
    plt.subplot(223)
    plt.plot(np.concatenate(all_time), np.concatenate(all_centroids['y']))
    plt.plot(np.concatenate(all_time), avg_centroid_oot['y'] * np.ones(len(np.concatenate(all_time))), 'r--',
             label='avg oot', zorder=1)
    # plt.plot(np.concatenate(all_time),
    #          np.concatenate([avg_centroid_oot['y'][i] * np.ones(len(all_time[i])) for i in range(len(all_time))]),
    #          'r--', label='avg oot', zorder=1)
    plt.plot(np.concatenate(all_time_oot), np.concatenate(centroid_oot['y']), zorder=0)
    plt.legend()
    plt.xlabel('Time [day]')
    if config.px_coordinates:
        plt.ylabel('Row pixel')
    else:
        plt.ylabel('Dec [arcsec]')
    plt.subplot(224)
    plt.scatter(np.concatenate(all_time_it), np.concatenate(centroid_it['y']), color='c', zorder=0)
    plt.plot(np.concatenate(all_time_it), avg_centroid_it['y'] * np.ones(len(np.concatenate(all_time_it))), 'g--',
             label='avg it', zorder=1)
    # plt.plot(np.concatenate(all_time),
    #          np.concatenate([avg_centroid_it['y'][i] * np.ones(len(all_time[i])) for i in range(len(all_time))]),
    #          'g--', label='avg it', zorder=1)
    plt.legend()
    plt.xlabel('Time [day]')
    # plt.subplot(236)
    # plt.plot(np.concatenate(all_time), target_coords[1] * np.ones(len(np.concatenate(all_time))), c='y', label='target')
    # plt.xlabel('Time [day]')

    # if config.satellite == 'kepler':
    #     # plt.savefig('{}{}_{}_{}_{}.png'.format(savedir, tce.kepid, tce.tce_plnt_num, tce.av_training_set, basename))
    #     plt.savefig('{}{}_{}_{}_{}.png'.format(savedir, tce.target_id, tce.tce_plnt_num, tce.label, basename))
    #
    # else:
    #     # plt.savefig('{}{}_{}_{}_{}.png'.format(savedir, tce.tic, tce.tce_plnt_num, tce.disposition, basename))
    #     plt.savefig('{}{}_{}_{}_{}.png'.format(savedir, tce.target_id, tce['toi_id'], tce.label, basename))

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

    # if config.satellite == 'kepler':
    #     # plt.suptitle('Kepid {} | TCE {} | {}\nTarget: {} (arcsec)'.format(tce['kepid'], tce['tce_plnt_num'],
    #     #                                                                   tce['av_training_set'], target_coords))
    #     plt.suptitle('Kepid {} | TCE {} | {}\nTarget: {} (arcsec)'.format(tce['target_id'], tce['tce_plnt_num'],
    #                                                                       tce['label'], target_coords))
    # else:
    #     # plt.suptitle('TIC ID {} | TCE {} | {}\nTarget: {} (arcsec)'.format(tce['tic'], tce['tce_plnt_num'],
    #     #                                                                   tce['disposition'], target_coords))
    #     plt.suptitle('TIC ID {} | TCE {} | {}\nTarget: {} (arcsec)'.format(tce['target_id'], tce['toi_id'],
    #                                                                        tce['label'], target_coords))

    plt.suptitle('{} | TCE {} | {}\nTarget: {} (arcsec)'.format(tce['target_id'], tce[config.tce_identifier],
                                                                       tce['label'], target_coords))

    plt.subplot(211)
    plt.plot(np.concatenate(all_time), np.concatenate(all_centroids['x']), zorder=0)
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

    # plt.subplot(222)
    # plt.plot(np.concatenate(all_time), target_coords[0] * np.ones(len(np.concatenate(all_time))), c='y')
    # plt.title('Target')

    plt.subplot(212)
    plt.plot(np.concatenate(all_time), np.concatenate(all_centroids['y']), zorder=0)
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

    # plt.subplot(224)
    # plt.plot(np.concatenate(all_time), target_coords[1] * np.ones(len(np.concatenate(all_time))), c='y')
    # plt.xlabel('Time [day]')

    # if config.satellite == 'kepler':
    #     # plt.savefig('{}{}_{}_{}_{}.png'.format(savedir, tce.kepid, tce.tce_plnt_num, tce.av_training_set, basename))
    #     plt.savefig('{}{}_{}_{}_{}.png'.format(savedir, tce.target_id, tce.tce_plnt_num, tce.label, basename))
    # else:
    #     # plt.savefig('{}{}_{}_{}_{}.png'.format(savedir, tce.tic, tce.tce_plnt_num, tce.disposition, basename))
    #     plt.savefig('{}{}_{}_{}_{}.png'.format(savedir, tce.target_id, tce['toi_id'], tce.label, basename))

    plt.savefig(os.path.join(savedir, '{}_{}_{}_{}.png'.format(tce.target_id, tce[config.tce_identifier], tce.label,
                                                               basename)))

    plt.close()


def plot_dist_centroids(time, centroid_dist, centroid_dist_spline, avg_centroid_dist_oot, tce,
                        config, savedir, basename):
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
            ax.plot(time[i], centroid_dist[i])

        if config.px_coordinates:
            ax.set_ylabel('Euclidean distance [pixel')
        else:
            ax.set_ylabel('Angular distance [arcsec]')
        ax.set_title('Centroid-to-target distance time-series')
        ax.set_xlabel('Time [day]')

        # if config.satellite == 'kepler':
        #     f.suptitle('TCE {} {} {}'.format(tce.kepid, tce.tce_plnt_num, tce.av_training_set))
        #     plt.savefig('{}{}_{}_{}_{}.png'.format(savedir, tce.kepid, tce.tce_plnt_num, tce.av_training_set, basename))
        # else:
        #     f.suptitle('TCE {} {} {}'.format(tce.tic, tce.tce_plnt_num, tce.disposition))
        #     plt.savefig('{}{}_{}_{}_{}.png'.format(savedir, tce.tic, tce.tce_plnt_num, tce.disposition, basename))

        f.suptitle('TCE {} {} {}'.format(tce.target_id, tce[config.tce_identifier], tce.label))
        plt.savefig(os.path.join(savedir, '{}_{}_{}_{}.png'.format(tce.target_id, tce[config.tce_identifier], tce.label,
                                                                   basename)))

        plt.close()

    else:
        time, centroid_quadr, centroid_quadr_spline = [np.concatenate(time)], [np.concatenate(centroid_dist)], \
                                                      [np.concatenate(centroid_dist_spline)]

        f, ax = plt.subplots(2, 1, figsize=(16, 10))

        for i in range(len(centroid_quadr)):
            ax[0].plot(time[i], centroid_quadr[i], zorder=0)
            ax[0].plot(time[i], centroid_quadr_spline[i], linestyle='--', label='spline', zorder=1)
            ax[1].plot(time[i], centroid_quadr[i] / centroid_quadr_spline[i] * avg_centroid_dist_oot)

        if config.px_coordinates:
            ax[0].set_ylabel('Euclidean distance [pixel]')
        else:
            ax[0].set_ylabel('Angular distance [arcsec]')
        ax[0].set_title('Centroid-to-target distance time-series')
        ax[0].legend()
        ax[1].set_ylabel('Centroid-to-target distance')
        ax[1].set_xlabel('Time [day]')
        ax[1].set_title('Spline normalized centroid-to-target distance time-series')

        # if config.satellite == 'kepler':
        #     f.suptitle('TCE {} {} {}'.format(tce.kepid, tce.tce_plnt_num, tce.av_training_set))
        #     plt.savefig('{}{}_{}_{}_{}.png'.format(savedir, tce.kepid, tce.tce_plnt_num, tce.av_training_set, basename))
        # else:
        #     f.suptitle('TCE {} {} {}'.format(tce.tic, tce.tce_plnt_num, tce.disposition))
        #     plt.savefig('{}{}_{}_{}_{}.png'.format(savedir, tce.tic, tce.tce_plnt_num, tce.disposition, basename))

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

    # if config.satellite:
    #     f.suptitle('TCE {} {} {}'.format(tce.kepid, tce.tce_plnt_num, tce.av_training_set))
    #     plt.savefig('{}{}_{}_{}_{}.png'.format(savedir, tce.kepid, tce.tce_plnt_num,
    #                                            tce.av_training_set, basename))
    # else:
    #     f.suptitle('TCE {} {} {}'.format(tce.tic, tce.tce_plnt_num, tce.disposition))
    #     plt.savefig('{}{}_{}_{}_{}.png'.format(savedir, tce.tic, tce.tce_plnt_num,
    #                                            tce.disposition, basename))

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

    # if config.satellite == 'kepler':
    #     f.suptitle('TCE {} {} {}'.format(tce.kepid, tce.tce_plnt_num, tce.av_training_set))
    #     plt.savefig('{}{}_{}_{}_{}.png'.format(savedir, tce.kepid, tce.tce_plnt_num,
    #                                            tce.av_training_set, basename))
    # else:
    #     f.suptitle('TCE {} {} {}'.format(tce.tic, tce.tce_plnt_num, tce.disposition))
    #     plt.savefig('{}{}_{}_{}_{}.png'.format(savedir, tce.tic, tce.tce_plnt_num,
    #                                            tce.disposition, basename))

    f.suptitle('TCE {} {} {}'.format(tce.target_id, tce[config.tce_identifier], tce.label))
    plt.savefig(os.path.join(savedir, '{}_{}_{}_{}.png'.format(tce.target_id, tce[config.tce_identifier], tce.label,
                                                               basename)))

    plt.close()


def plot_all_views(views, tce, config, scheme, savedir, basename):
    """ Creates and saves a figure with plots that show views for a given TCE.

    :param views: dict, views to be plotted
    :param tce: pandas Series, row of the input TCE table Pandas DataFrame
    :param config: Config object; preprocessing parameters.
    :param scheme: list, defines the number and position of the view plots in the figure ([number of plots per row,
    number of plots per column])
    :param savedir: str, filepath to directory in which the figure is saved
    :param basename: str, added to the figure filename
    :return:
    """

    f, ax = plt.subplots(scheme[0], scheme[1], figsize=(17, 11))
    k = 0
    views_list = list(views.keys())
    for i in range(scheme[0]):
        for j in range(scheme[1]):
            if k < len(views_list):
                ax[i, j].plot(views[views_list[k]])
                ax[i, j].set_title(views_list[k], pad=20)
            if i == scheme[0] - 1:
                ax[i, j].set_xlabel('Bin number')
            if j == 0:
                ax[i, j].set_ylabel('Amplitude')
            k += 1

    # if config.satellite == 'kepler':
    #     f.suptitle('TCE {} {} {}'.format(tce.kepid, tce.tce_plnt_num, tce.av_training_set))
    #
    #     plt.savefig('{}{}_{}_{}_{}.png'.format(savedir, tce.kepid, tce.tce_plnt_num,
    #                                        tce.av_training_set, basename))
    # else:
    #     f.suptitle('TCE {} {} {}'.format(tce.tic, tce.tce_plnt_num, tce.disposition))
    #
    #     plt.savefig('{}{}_{}_{}_{}.png'.format(savedir, tce.tic, tce.tce_plnt_num,
    #                                            tce.disposition, basename))

    f.suptitle('TCE {} {} {}'.format(tce.target_id, tce[config.tce_identifier], tce.label))
    plt.subplots_adjust(hspace=0.3)
    plt.savefig(os.path.join(savedir, '{}_{}_{}_{}.png'.format(tce.target_id, tce[config.tce_identifier], tce.label,
                                                               basename)))

    f.tight_layout(rect=[0, 0.03, 1, 0.95])

    plt.close()


def plot_wks(glob_view, glob_view_weak_secondary, tce, config, savedir, basename):

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
