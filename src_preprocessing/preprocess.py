""" Preprocessing module adapted from:

# Copyright 2018 The Exoplanet ML Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

Functions for reading and preprocessing light curves and related data.

Authors:
- Laurent Wilkens
- Nikash Walia
- Miguel Martinho
- Sam Donald

"""

# 3rd party
import numpy as np
import tensorflow as tf
import os
if 'home6' in os.path.dirname(os.path.dirname(os.path.abspath(__file__))):
    import matplotlib; matplotlib.use('agg')
import socket

# local
from src_preprocessing.light_curve import kepler_io
from src_preprocessing.light_curve import median_filter
from src_preprocessing.light_curve import util
from src_preprocessing.tf_util import example_util
from src_preprocessing.third_party.kepler_spline import kepler_spline
from src_preprocessing import utils_visualization
from src_preprocessing.utils_centroid_preprocessing import kepler_transform_pxcoordinates_mod13, \
    synchronize_centroids_with_flux
from src_preprocessing.utils_ephemeris import create_binary_time_series, find_first_epoch_after_this_time
from src_preprocessing import tess_io


def is_pfe():
    """
    Returns boolean which indicates whether this script is being run on Pleiades or local computer.
    """

    nodename = os.uname().nodename

    if nodename[:3] == 'pfe':
        return True

    if nodename[0] == 'r':
        try:
            int(nodename[-1])
            return True
        except ValueError:
            return False

    return False


def report_exclusion(config, tce, id_str, stderr=None):
    """ Error log is saved into a txt file with the reasons why a given TCE was not preprocessed.

    :param config: Config object with parameters for the preprocessing. Check the Config class
    :param tce: pandas Series, row of the input TCE table Pandas DataFrame.
    :param id_str: str, contains info on the cause of exclusion
    :param stderr: str, error
    :return:
    """

    # create path to exclusion logs directory
    savedir = os.path.join(config.output_dir, 'exclusion_logs')

    # create exclusion logs directory if it does not exist
    os.makedirs(savedir, exist_ok=True)

    # TODO: what if TESS changes to multi-sector analysis; sector becomes irrelevant...
    if config.satellite == 'kepler':
        main_str = 'KepID {} TCE {}'.format(tce.kepid, tce.tce_plnt_num)
    else:  # 'tess'
        main_str = 'TICID {} TCE {} Sector {}'.format(tce.target_id, tce[config.tce_identifier], tce.sector)

    if is_pfe():

        # get node id
        node_id = socket.gethostbyname(socket.gethostname()).split('.')[-1]

        # write to exclusion log pertaining to this process and node
        with open(os.path.join(savedir, 'exclusions_{}_{}-{}.txt'.format(config.process_i, node_id,
                                                                         main_str.replace(" ", ""))),
                  "a") as myfile:
            myfile.write('{}, {}\n{}'.format(main_str, id_str, (stderr, '')[stderr is None]))
    else:
        # write to exclusion log locally
        with open(os.path.join(savedir, 'exclusions-{}.txt'.format(main_str.replace(" ", ""))), "a") as myfile:
            myfile.write('{}, {}\n{}'.format(main_str, id_str, (stderr, '')[stderr is None]))


def read_light_curve(tce, config):
    """ Reads the FITS files pertaining to a Kepler/TESS target.

    Args:
        tce: row of DataFrame, information on the TCE (ID, ephemeris, ...).
        config: Config object, preprocessing parameters

    Returns:
        all_time: A list of numpy arrays; the time values of the raw light curve
        all_flux: A list of numpy arrays corresponding to the PDC flux time series
        all_centroid: A list of numpy arrays corresponding to the raw centroid time series
        add_info: A dict with additional data extracted from the FITS files; 'quarter' is a list of quarters for each numpy
        array of the light curve; 'module' is the same but for the module in which the target is in every quarter;
        'target position' is a list of two elements which correspond to the target star position, either in world (RA, Dec)
        or local CCD (x, y) pixel coordinates

    Raises:
      IOError: If the light curve files for this target cannot be found.
    """

    # gets all filepaths for the FITS files for this target
    if config.satellite == 'kepler':

        # file_names = kepler_io.kepler_filenames(config.lc_data_dir, tce.kepid)
        file_names = kepler_io.kepler_filenames(config.lc_data_dir, tce.target_id)

        if not file_names:
            if not config.omit_missing:
                raise IOError("Failed to find .fits files in {} for Kepler ID {}".format(config.lc_data_dir, tce.kepid))
            else:
                report_exclusion(config, tce, 'No available lightcurve .fits files')
                return None, None, None, None

        return kepler_io.read_kepler_light_curve(file_names, centroid_radec=not config.px_coordinates)

    else:
        # file_names = tess_io.tess_filenames(config.lc_data_dir, tce.tic, tce.sector, config.multisector)

        # the sector field decides which sectors to look for the TCE
        # can be single- or multi-sector run
        if 'sector' in tce:  # if there is a 'sector' field in the TCE table
            # overlap between sectors defined in the config and sector in the TCE table for the TCE
            if config.sectors is not None:
                sectors = [int(sector) for sector in tce.sector if int(sector) in config.sectors]
            # get sectors in the TCE table for the TCE
            else:
                sectors = [int(sector) for sector in tce.sector]
            # multisector = 'table'
        else:  # get sectors defined in the config
            sectors = config.sectors
            # multisector = 'no-table'

        file_names, tce_sectors = tess_io.tess_filenames(config.lc_data_dir, tce.target_id, sectors)

        if 'sector' not in tce:
            tce.sector = tce_sectors

        if not file_names:
            if not config.omit_missing:
                raise IOError("Failed to find .fits files in {} for TESS ID {}".format(config.lc_data_dir,
                                                                                       tce.target_id))
            else:
                report_exclusion(config, tce, 'No available lightcurve .fits files')
                return None, None, None, None

        return tess_io.read_tess_light_curve(file_names, centroid_radec=not config.px_coordinates)


def get_gap_indices(flux, checkfuncs=None):
    """ Finds gaps in time series data (where time series is one of [0, nan, inf, -inf])

    :param flux:  flux time-series
    :return:
        id_dict: dict with start and end indices of each gap in flux time series
    """

    # maybe only the checks requested should be performed
    checkfuncdict = {0: flux == 0,
                     'nan': np.isnan(flux),
                     'inf': np.isinf(flux),
                     '-inf': np.isinf(-flux)}

    # set which checks to do
    checkfuncs = checkfuncdict if not checkfuncs else {i: checkfuncdict[i] for i in checkfuncs}

    id_dict = {}
    for checkstr, checkfunc in checkfuncs.items():
        arr = np.where(checkfunc)[0]

    #     id_dict_type = {}
    #     count = 0
    #     for i in range(len(arr)):
    #         if count not in id_dict_type:
    #             id_dict_type[count] = [arr[i], -1]
    #         if i + 1 <= len(arr) - 1 and arr[i + 1] - arr[i] > 1:
    #             id_dict_type[count] = [id_dict_type[count][0], arr[i]]
    #             count += 1
    #         else:
    #             if arr[i] - arr[i - 1] > 1:
    #                 id_dict_type[count] = [arr[i], arr[i]]
    #             else:
    #                 id_dict_type[count] = [id_dict_type[count][0], arr[i]]
    #     id_dict[checkstr] = id_dict_type

        ####
        # CASE NO GAPS
        if len(arr) == 0:
            id_dict[checkstr] = []  # {}  # EMPTY DICT, NONE?
        # CASE ONLY ONE SINGLE IDX GAP
        elif len(arr) == 1:
            id_dict[checkstr] = [[arr[0], arr[0] + 1]]  # {0: [arr[0], arr[0] + 1]}
        # CASE TWO OR MORE IDXS GAP OR MORE THAN ONE GAP
        # id_dict_type = {}  # WHY IS IT A DICT??? IT COULD BE A SIMPLE LIST!!!!!!!!!
        id_dict_type = []
        arr_diff = np.diff(arr)
        jump_idxs = np.where(arr_diff > 1)[0]
        jump_idxs += 1
        jump_idxs = np.insert(jump_idxs, [0, len(jump_idxs)], [0, len(arr)])
        for start, end in zip(jump_idxs[:-1], jump_idxs[1:]):
            # id_dict_type[len(id_dict_type)] = [start, end]
            id_dict_type.append([start, end])
        id_dict[checkstr] = id_dict_type

    return id_dict


def lininterp_transits(timeseries, transit_pulse_train, centroid=False):
    """ Linearly interpolate the timeseries across the in-transit cadences.

    :param timeseries: list of numpy arrays, time-series; if centroid is True, then it is a dictionary with a list of
    numpy arrays for each coordinate ('x' and 'y')
    :param transit_pulse_train: list of numpy arrays, binary arrays that are 1's for in-transit cadences and 0 otherwise
    :param centroid: bool, treats time-series as centroid time-series ('x' and 'y') if True
    :return:
        timeseries_interp: list of numpy arrays, time-series linearly interpolated at the transits; if centroid is True,
         then it is a dictionary with a list of numpy arrays for each coordinate ('x' and 'y')
    """

    # initialize variables
    if centroid:
        num_arrs = len(timeseries['x'])
        timeseries_interp = {'x': [], 'y': []}
    else:
        num_arrs = len(timeseries)
        timeseries_interp = []

    for i in range(num_arrs):

        if centroid:
            timeseries_interp['x'].append(np.array(timeseries['x'][i]))
            timeseries_interp['y'].append(np.array(timeseries['y'][i]))
        else:
            timeseries_interp.append(np.array(timeseries[i]))

        idxs_it = np.where(transit_pulse_train[i] == 1)[0]
        if len(idxs_it) == 0:  # no transits in the array
            continue

        idxs_lim = np.where(np.diff(idxs_it) > 1)[0] + 1

        start_idxs = np.insert(idxs_lim, 0, 0)
        end_idxs = np.append(idxs_lim, len(idxs_it))

        for start_idx, end_idx in zip(start_idxs, end_idxs):

            # boundary issue - do nothing, since the whole array is a transit; does this happen?
            if idxs_it[start_idx] == 0 and idxs_it[end_idx - 1] == len(transit_pulse_train[i]) - 1:
                continue

            if idxs_it[start_idx] == 0:  # boundary issue start - constant value end
                if centroid:
                    timeseries_interp['x'][i][idxs_it[start_idx]:idxs_it[end_idx - 1] + 1] = \
                        timeseries['x'][i][idxs_it[end_idx - 1] + 1]
                    timeseries_interp['y'][i][idxs_it[start_idx]:idxs_it[end_idx - 1] + 1] = \
                        timeseries['y'][i][idxs_it[end_idx - 1] + 1]
                else:
                    timeseries_interp[i][idxs_it[start_idx]:idxs_it[end_idx - 1] + 1] = \
                        timeseries[i][idxs_it[end_idx - 1] + 1]

            elif idxs_it[end_idx - 1] == len(transit_pulse_train[i]) - 1:  # boundary issue end - constant value start
                if centroid:
                    timeseries_interp['x'][i][idxs_it[start_idx]:idxs_it[end_idx - 1] + 1] = timeseries['x'][i][
                        idxs_it[start_idx] - 1]
                    timeseries_interp['y'][i][idxs_it[start_idx]:idxs_it[end_idx - 1] + 1] = timeseries['y'][i][
                        idxs_it[start_idx] - 1]
                else:
                    timeseries_interp[i][idxs_it[start_idx]:idxs_it[end_idx - 1] + 1] = \
                        timeseries[i][idxs_it[start_idx] - 1]

            else:  # linear interpolation
                idxs_interp = np.array([idxs_it[start_idx] - 1, idxs_it[end_idx - 1] + 1])
                idxs_to_interp = np.arange(idxs_it[start_idx], idxs_it[end_idx - 1] + 1)

                if centroid:
                    timeseries_interp['x'][i][idxs_it[start_idx]:idxs_it[end_idx - 1] + 1] = \
                        np.interp(idxs_to_interp, idxs_interp, timeseries['x'][i][idxs_interp])
                    timeseries_interp['y'][i][idxs_it[start_idx]:idxs_it[end_idx - 1] + 1] = \
                        np.interp(idxs_to_interp, idxs_interp, timeseries['y'][i][idxs_interp])
                else:
                    timeseries_interp[i][idxs_it[start_idx]:idxs_it[end_idx - 1] + 1] = \
                        np.interp(idxs_to_interp, idxs_interp, timeseries[i][idxs_interp])

    return timeseries_interp


def gap_other_tces(all_time, all_flux, all_centroids, add_info, tce, table, config, conf_dict, gap_pad=0):
    """ Remove from the time series the cadences that belong to other TCEs in the light curve. These values are set to
    NaN.

    :param all_time: list of numpy arrays, cadences
    :param all_flux: list of numpy arrays, flux time series
    :param all_centroids: list of numpy arrays, centroid time series
    :param tce: row of pandas DataFrame, main TCE ephemeris
    :param table: pandas DataFrame, TCE ephemeris table
    :param config: Config object, preprocessing parameters
    :param conf_dict: dict, keys are a tuple (Kepler ID, TCE planet number) and the values are the confidence level used
     when gapping (between 0 and 1)
    :param gap_pad: extra pad on both sides of the gapped TCE transit duration
    :return:
        all_time: list of numpy arrays, cadences
        all_flux: list of numpy arrays, flux time series
        all_centroids: list of numpy arrays, flux centroid time series
        imputed_time: list of numpy arrays, gapped cadences
    """

    # get gapped TCEs ephemeris
    if config.satellite == 'kepler':
        # gap_ephems = table.loc[(table['kepid'] == tce.kepid) &
        #                        (table['tce_plnt_num'] != tce.tce_plnt_num)][['tce_period', 'tce_duration',
        #                                                                      'tce_time0bk']]
        gap_ephems = table.loc[(table['target_id'] == tce.target_id) &
                               (table[config.tce_identifier] != tce[config.tce_identifier])][['tce_period',
                                                                                              'tce_duration',
                                                                                              'tce_time0bk']]
    else:  # TESS
        # gap_ephems = table.loc[(table['tic'] == tce.tic) &
        #                           (table['sector'] == tce.sector) &
        #                           (table[config.tce_identifier] != tce[config.tce_identifier])]
        # gap_ephems = table.loc[(table['target_id'] == tce.tic) &
        #                        (table['sector'] == tce.sector) &
        #                        (table[config.tce_identifier] != tce[config.tce_identifier])]

        # if sector column exists in the TCE table
        if 'sector' in table:

            # get observed sectors for the current TCE
            sectors = tce.sector.split(' ')

            # get ephemeris for the TCEs which belong to the same target star
            candidateGapTceTable = table.loc[(table['target_id'] == tce.target_id) &
                                             (table[config.tce_identifier] !=
                                              tce[config.tce_identifier])][['tce_period',
                                                                            'tce_duration',
                                                                            'tce_time0bk']]

            # get TCEs whose observed sectors overlap with the observed sectors for the current TCE
            candidatesRemoved = []
            gapSectors = {}
            for i, candidate in candidateGapTceTable.iterrows():
                candidateSectors = candidate.sector.split(' ')
                # get only overlapping sectors
                sectorsIntersection = np.intersect1d(sectors, candidateSectors)
                if len(sectorsIntersection) > 0:
                    gapSectors[len(gapSectors)] = sectorsIntersection
                else:
                    candidatesRemoved.append(i)
            # remove candidates that do not have any overlapping sectors
            gap_ephems = candidateGapTceTable.drop(candidateGapTceTable.index[candidatesRemoved], inplace=False)

        else:  # if not, gap all TCEs that belong to the same target star

            gap_ephems = table.loc[(table['target_id'] == tce.target_id) &
                                   (table[config.tce_identifier] != tce[config.tce_identifier])][['tce_period',
                                                                                                  'tce_duration',
                                                                                                  'tce_time0bk']]

            gapSectors = {gtce_i: add_info['sector'] for gtce_i in range(len(gap_ephems))}

            print('#####' * 50)
            print('1', len(gap_ephems), print(gapSectors))

    # # rename column names to same as Kepler - it would be easier if the uniformization was done in the TCE tables
    # if config.satellite == 'tess':
    #     gap_ephems = gap_ephems.rename(columns={'transitDurationHours': 'tce_duration', 'orbitalPeriodDays':
    #         'tce_period', 'transitEpochBtjd': 'tce_time0bk'})

    # if gapping with confidence level, remove those gapped TCEs that are not in the confidence dict
    # TODO: currently only implemented for Kepler
    if config.gap_with_confidence_level and config.satellite == 'kepler':
        poplist = []
        for index, gapped_tce in gap_ephems.iterrows():
            if (tce.kepid, gapped_tce[config.tce_identifier]) not in conf_dict or \
                    conf_dict[(tce.target_id, gapped_tce[config.tce_identifier])] < config.gap_confidence_level:
                poplist += [gapped_tce[config.tce_identifier]]

        gap_ephems = gap_ephems.loc[gap_ephems[config.tce_identifier].isin(poplist)]
    elif config.gap_with_confidence_level and config.satellite == 'tess':
        raise NotImplementedError('Using confidence level for gapping TCEs not implemented for TESS.')

    # initialize array to get timestamps for imputing
    imputed_time = [] if config.gap_imputed else None

    # begin_time, end_time = all_time[0][0], all_time[-1][-1]

    # find gapped cadences for each TCE
    real_ephem_i = 0
    for ephem_i, ephem in gap_ephems.iterrows():

        ephem['tce_duration'] = ephem['tce_duration'] * (1 + 2 * gap_pad)

        for i in range(len(all_time)):

            # for TESS, check if this time array belongs to one of the overlapping sectors
            if config.satellite != 'kepler' and add_info['sector'][i] not in gapSectors[real_ephem_i]:
                continue

            begin_time, end_time = all_time[i][0], all_time[i][-1]

            # get timestamp of the first transit of the gapped TCE in the current time array
            epoch = find_first_epoch_after_this_time(ephem['tce_time0bk'], ephem['tce_period'], begin_time)

            # create binary time-series for this time interval based on the ephemeris of the gapped TCE
            bintransit_ts = create_binary_time_series(all_time[i], epoch, ephem['tce_duration'], ephem['tce_period'])

            # get indexes of in-transit cadences for the gapped TCE in this time array
            transit_idxs = np.where(bintransit_ts == 1)

            # get gapped cadences to be imputed
            if config.gap_imputed and len(transit_idxs) > 0:
                imputed_time.append(all_time[i][transit_idxs])

            # set in-transit cadences to NaN
            all_flux[i][transit_idxs] = np.nan
            all_centroids['x'][i][transit_idxs] = np.nan
            all_centroids['y'][i][transit_idxs] = np.nan

        real_ephem_i += 1

    return all_time, all_flux, all_centroids, imputed_time


def _process_tce(tce, table, all_flux, all_time, config, conf_dict, gap_time=None):
    """ Processes the time-series features for the TCE and returns an Example proto.

    Args:
        tce: row of the input TCE table.
        table: pandas DataFrame, ephemeris information on other TCEs used when gapping
        all_flux: list of numpy arrays, flux time series for TCEs; None when not using whitened data
        all_time: list of numpy arrays, cadences time series for TCEs; None when not using whitened data
        config: Config object, holds preprocessing parameters
        conf_dict: dict, keys are a tuple (Kepler ID, TCE planet number) and the values are the confidence level used
        when gapping (between 0 and 1)
        gap_ids: list of numpy arrays, gapped cadences

    Returns:
        A tensorflow.train.Example proto containing TCE features.
    """

    # if tce['av_training_set'] == 'PC' and tce['kepid'] == 3236705 and tce['tce_plnt_num'] == 1:
    #     print(tce[['kepid', 'tce_plnt_num', 'ra', 'dec', 'av_training_set']])
    # else:
    #     return None
    # if tce['disposition'] == 'PC' and tce['tic'] == 100608026 and tce['tce_plnt_num'] == 1:  # what about sector?
    #     print(tce[['tic', 'tce_plnt_num', 'disposition']])
    # else:
    #     return None

    if [tce['target_id'], tce['tce_plnt_num']] in [[11017901, 1], [10358759, 2], [11037818, 1], [10863608, 1],
                                                   [10777591, 1], [10975146, 1], [10904857, 1], [10028792, 2],
                                                   [10585738, 1], [10717220, 1]]:  #  and tce['tce_plnt_num'] == 1:  # what about sector?
        print(tce[['target_id', 'label']])
    else:
        return None

    # check if preprocessing pipeline figures are saved for the TCE
    plot_preprocessing_tce = True
    if np.random.random() < 0.01:
        plot_preprocessing_tce = config.plot_figures

    centroid_time = None  # pre-fill

    if all_flux is None:  # all_flux = None if satellite = kepler and non-whitened light curves

        # get cadence, flux and centroid data for the tce
        all_time, all_flux, all_centroids, add_info = read_light_curve(tce, config)

        if all_time is None:
            return None

        if plot_preprocessing_tce:
            utils_visualization.plot_centroids(all_time, all_centroids, None, tce, config,
                                               os.path.join(config.output_dir, 'plots/'), '1_raw',
                                               add_info=add_info)

    else:  # only keep centroids as whitened source does not have centroid information
        centroid_time, _, all_centroids, add_info = read_light_curve(tce, config)

    if all_time is None:
        report_exclusion(config, tce, 'Empty arrays')
        return None

    # for the whitened data series we need to do some extra data manipulation and processing
    if config.whitened:

        # FIXME does it give the expected structure in terms of centroids?
        all_centroids['x'] = np.concatenate(all_centroids['x'])
        all_centroids['y'] = np.concatenate(all_centroids['y'])
        centroid_time = np.concatenate(centroid_time)

        all_flux = np.array(all_flux)
        all_time = np.array(all_time)

        if np.max(all_flux) == 0:
            report_exclusion(config, tce, 'All whitened flux import values are 0')
            return None

        # remove patches of 0's (whitened time series are filled with 0's instead of nan's)
        zero_gaps = get_gap_indices(all_flux, checkfuncs=[0])[0]
        # if removing different valued gaps, we would need an additional step to order the gaps
        # starting from the last gap to the first one in order to not mess with the indexing
        for gap_n in range(len(zero_gaps) - 1, -1, -1):
            gap = zero_gaps[gap_n]
            gap_length = gap[-1] - gap[0] + 1
            if gap_length > 2:  # avoiding gaps smaller than 3 cadences
                all_flux = np.concatenate((all_flux[:gap[0]], all_flux[gap[-1]:]))
                all_time = np.concatenate((all_time[:gap[0]], all_time[gap[-1]:]))

        # align cadences of centroid and time
        all_centroids = synchronize_centroids_with_flux(all_time, centroid_time, all_centroids)
        if len(all_centroids['x']) != len(all_flux):
            report_exclusion(config, tce, 'after sychronizing the centroid with the flux time series, the two time '
                                          'series do not have the same length.')

        # if not config.gapped:
        all_time, all_flux, all_centroids['x'], all_centroids['y'] = \
            [all_time], [all_flux], [all_centroids['x']], [all_centroids['y']]

    # FIXME: what if removes a whole quarter? need to adjust all_additional_info to it
    # gap other TCEs in the light curve
    if config.gapped:
        all_time, all_flux, all_centroids, gap_time = \
            gap_other_tces(all_time, all_flux, all_centroids, add_info, tce, table, config, conf_dict, gap_pad=0)

    # remove timestamps with NaN or infinite time, flux or centroid values in each quarter
    # at least some of these NaNs come from gapping the time series
    # other NaNs can come from missing time values from the fits files
    for i, time_series_set in enumerate(zip(all_flux, all_centroids['x'], all_centroids['y'])):
        finite_id = np.isfinite(all_time[i])
        for time_series in time_series_set:
            finite_id = np.logical_and(finite_id, np.isfinite(time_series))
        all_time[i] = all_time[i][finite_id]
        all_flux[i] = time_series_set[0][finite_id]
        all_centroids['x'][i] = time_series_set[1][finite_id]
        all_centroids['y'][i] = time_series_set[2][finite_id]

    if plot_preprocessing_tce:
        utils_visualization.plot_centroids(all_time, all_centroids, None, tce, config,
                                           os.path.join(config.output_dir, 'plots/'), '2_rawwithoutnans',
                                           add_info=add_info)

    # preprocess the flux and centroid time series
    time, flux, centroids = process_light_curve(all_time, all_flux, all_centroids, gap_time, config, add_info,
                                                tce, plot_preprocessing_tce)

    # create the different channels
    return generate_example_for_tce(time, flux, centroids, tce, config, plot_preprocessing_tce)


def process_light_curve(all_time, all_flux, all_centroids, gap_time, config, add_info, tce,
                        plot_preprocessing_tce=False):
    """ Removes low-frequency variability from the flux and centroid time-series.

    Args:
      all_time: A list of numpy arrays; the cadences of the raw light curve in each quarter
      all_flux: A list of numpy arrays corresponding to the PDC flux time series in each quarter
      all_centroids: A list of numpy arrays corresponding to the raw centroid time series in each quarter
      gap_time: A list of numpy arrays; the cadences for the imputed intervals
      config: Config object, holds preprocessing parameters
      add_info: A dict with additional data extracted from the FITS files; 'quarter' is a list of quarters for each
      numpy array of the light curve; 'module' is the same but for the module in which the target is in every quarter;
      'target position' is a list of two elements which correspond to the target star position, either in world
      (RA, Dec) or local CCD (x, y) pixel coordinates
      tce: pandas Series, row of the TCE table for the TCE that is being processed; it contains data such as the
      ephemeris
      plot_preprocessing_tce: bool, if True plots figures for several steps in the preprocessing pipeline

    Returns:
      time: 1D NumPy array; the time values of the light curve
      flux: 1D NumPy array; the normalized flux values of the light curve
      centroid_dist NumPy array; the distance of the corrected centroid time series to the target position
    """

    # FIXME: why 0.75 gap?
    # Split on gaps.
    all_time, all_flux, all_centroids, add_info = util.split_wcentroids(all_time, all_flux, all_centroids, add_info,
                                                                        config.satellite, gap_width=config.gapWidth)

    # pixel coordinate transformation for targets on module 13 for Kepler
    if config.px_coordinates and config.satellite == 'kepler':
        if add_info['module'][0] == 13:
            all_centroids = kepler_transform_pxcoordinates_mod13(all_centroids, add_info)

            if plot_preprocessing_tce:
                utils_visualization.plot_centroids(all_time, all_centroids, None, tce, config,
                                                   os.path.join(config.output_dir, 'plots/'),
                                                   '2_rawaftertransformation', add_info=add_info)

    # TODO: standardize keywords in the TCE tables for TESS and Kepler so that we do not need to do this check..
    # if config.satellite == 'kepler':
    #     period, duration, t0, transit_depth = tce["tce_period"], tce["tce_duration"], tce["tce_time0bk"], \
    #                                           tce['tce_depth']
    # else:
    #     period, duration, t0, transit_depth = tce["orbitalPeriodDays"], tce["transitDurationHours"], \
    #                                           tce["transitEpochBtjd"], tce['transitDepthPpm']
    period, duration, t0, transit_depth = tce["tce_period"], tce["tce_duration"], tce["tce_time0bk"], \
                                          tce['transit_depth']

    # get epoch of first transit for each time array
    first_transit_time_all = [find_first_epoch_after_this_time(t0, period, time[0])
                              for time in all_time]

    # create binary time series for each time array in which in-transit points are labeled as 1's, otherwise as 0's
    binary_time_all = [create_binary_time_series(time, first_transit_time, duration, period)
                       for first_transit_time, time in zip(first_transit_time_all, all_time)]

    if plot_preprocessing_tce:
        utils_visualization.plot_binseries_flux(all_time, all_flux, binary_time_all, tce, config,
                                                os.path.join(config.output_dir, 'plots/'), '3_binarytimeseriesandflux')

    # get centroid time series out-of-transit points
    centroid_oot = {coord: [centroids[np.where(binary_time == 0)] for binary_time, centroids in
                     zip(binary_time_all, all_centroids[coord])] for coord in all_centroids}

    # TODO: how to compute the average oot? mean, median, other...
    # average the centroid time series values for the oot points
    avg_centroid_oot = {coord: np.median(np.concatenate(centroid_oot[coord])) for coord in centroid_oot}

    # spline fitting and normalization - fit a piecewise-cubic spline with default arguments
    # FIXME: wouldn't it be better to fit a spline to oot values or a Savitzky Golay filter
    #  (as Jeff and Doug mentioned)?
    # FIXME: fit spline only to the oot values - linearly interpolate the transits; same thing for flux
    all_centroids_lininterp = lininterp_transits(all_centroids, binary_time_all, centroid=True)
    spline_centroid = {coord: kepler_spline.fit_kepler_spline(all_time, all_centroids_lininterp[coord],
                                                              verbose=False)[0] for coord in all_centroids_lininterp}
    # spline_centroid = {coord: kepler_spline.fit_kepler_spline(all_time, all_centroids[coord], verbose=False)[0]
    #                    for coord in all_centroids}
    if not config.whitened:
        # fit a piecewise-cubic spline with default arguments
        all_flux_lininterp = lininterp_transits(all_flux, binary_time_all, centroid=False)
        spline_flux = kepler_spline.fit_kepler_spline(all_time, all_flux_lininterp, verbose=False)[0]
        # spline_flux = kepler_spline.fit_kepler_spline(all_time, all_flux, verbose=False)[0]

    if plot_preprocessing_tce:
        if not config.whitened:
            utils_visualization.plot_flux_fit_spline(all_time, all_flux, spline_flux, tce, config,
                                                     os.path.join(config.output_dir, 'plots/'),
                                                     '4_smoothingandnormalizationflux')

        utils_visualization.plot_centroids(all_time, all_centroids, spline_centroid, tce, config,
                                           os.path.join(config.output_dir, 'plots/'),
                                           '4_smoothingandnormalizationcentroid')

    # In rare cases the piecewise spline contains NaNs in places the spline could not be fit. We can't normalize those
    # points if the spline isn't defined there. Instead we just remove them.
    finite_i_centroid = [np.logical_and(np.isfinite(spline_centroid['x'][i]), np.isfinite(spline_centroid['y'][i]))
                         for i in range(len(spline_centroid['x']))]
    if not config.whitened:
        finite_i_flux = [np.isfinite(spline_flux[i]) for i in range(len(spline_flux))]
        finite_i = [np.logical_and(finite_i_centroid[i], finite_i_flux[i]) for i in range(len(finite_i_centroid))]
    else:
        finite_i = finite_i_centroid

    # remove cadences for which the spline did not fit
    all_time = [all_time[i][finite_i[i]] for i in range(len(all_time))]
    binary_time_all = [binary_time_all[i][finite_i[i]] for i in range(len(binary_time_all))]

    # normalize by the spline
    all_centroids = {coord: [all_centroids[coord][i][finite_i[i]] / spline_centroid[coord][i][finite_i[i]] *
                             avg_centroid_oot[coord] for i in range(len(spline_centroid[coord]))]
                     for coord in all_centroids}

    if not config.whitened:
        all_flux = [all_flux[i][finite_i[i]] / spline_flux[i][finite_i[i]] for i in range(len(spline_flux))]

    # get the position of the target star
    target_coords = add_info['target position']

    # compute the new average oot after the spline fitting and normalization
    # TODO: how to compute the average oot? mean, median, other...
    centroid_oot = {coord: [centroids[np.where(binary_time == 0)] for binary_time, centroids in
                            zip(binary_time_all, all_centroids[coord])] for coord in all_centroids}
    avg_centroid_oot = {coord: np.median(np.concatenate(centroid_oot[coord])) for coord in all_centroids}

    if plot_preprocessing_tce:
        utils_visualization.plot_centroids_it_oot(all_time, binary_time_all, all_centroids, centroid_oot,
                                                  avg_centroid_oot, target_coords, tce, config,
                                                  os.path.join(config.output_dir, 'plots/'),
                                                  '5_centroidtimeseries_it-ot-target')

    # compute the corrected centroid time-series normalized by the transit depth fraction and centered on the avg oot
    # centroid position
    transitdepth_term = transit_depth / (1e6 - transit_depth)
    all_centroids = {coord: [-((all_centroids[coord][i] - avg_centroid_oot[coord]) / transitdepth_term) +
                             avg_centroid_oot[coord] for i in range(len(all_centroids[coord]))]
                     for coord in all_centroids}

    if plot_preprocessing_tce:
        utils_visualization.plot_corrected_centroids(all_time, all_centroids, avg_centroid_oot, target_coords, tce,
                                                     config, os.path.join(config.output_dir, 'plots/'),
                                                     '6_correctedcentroidtimeseries')

    if config.px_coordinates:
        # compute the euclidean distance of the corrected centroid time series to the target star position
        centroid_dist = [np.sqrt(np.square(all_centroids['x'][i] - target_coords[0]) +
                          np.square(all_centroids['y'][i] - target_coords[1]))
                          for i in range(len(all_centroids['x']))]
    else:

        # compute the angular distance of the corrected centroid time series to the target star position
        centroid_dist = [np.sqrt(np.square((all_centroids['x'][i] - target_coords[0]) *
                                           np.cos(all_centroids['y'][i] - target_coords[1])) +
                                 np.square(all_centroids['y'][i] - target_coords[1]))
                         for i in range(len(all_centroids['x']))]

    # # get the average oot distance
    # centroid_dist_oot = [centroid_dist_block[np.where(binary_time == 0)] for binary_time, centroid_dist_block
    #                      in zip(binary_time_all, centroid_dist)]
    # avg_centroid_dist_oot = np.median(np.concatenate(centroid_dist_oot))

    # convert from degree to arcsec
    if not config.px_coordinates:
        centroid_dist = [centroid_dist_arr * 3600 for centroid_dist_arr in centroid_dist]

    if plot_preprocessing_tce:
        utils_visualization.plot_dist_centroids(all_time, centroid_dist, None, None, tce, config,
                                                os.path.join(config.output_dir, 'plots/'), '7_distcentr')

    # # spline fitting and normalization - fit a piecewise-cubic spline with default arguments
    # # FIXME: wouldn't it be better to fit a spline to oot values or a Savitzky Golay filter
    # #  (as Jeff and Doug mentioned)?
    # # FIXME: fit spline only to the oot values - linearly interpolate the transits; same thing for flux
    # spline_centroid = kepler_spline.fit_kepler_spline(all_time, centroid_dist, verbose=False)[0]
    # if not config.whitened:
    #     # Fit a piecewise-cubic spline with default arguments.
    #     spline_flux = kepler_spline.fit_kepler_spline(all_time, all_flux, verbose=False)[0]
    #
    # if plot_preprocessing_tce:
    #     utils_visualization.plot_flux_fit_spline(all_time, all_flux, spline_flux, tce,
    #                                              os.path.join(config.output_dir, 'plots/'),
    #                                              '7_smoothingandnormalizationflux')
    #     utils_visualization.plot_dist_centroids(all_time, centroid_dist, spline_centroid, avg_centroid_dist_oot,
    #                                             tce, config.px_coordinates, os.path.join(config.output_dir, 'plots/'),
    #                                             '7_smoothingandnormalizationcentroid')
    #
    # # In rare cases the piecewise spline contains NaNs in places the spline could not be fit. We can't normalize those
    # # points if the spline isn't defined there. Instead we just remove them.
    # finite_i_centroid = [np.isfinite(spline_centroid[i]) for i in range(len(spline_centroid))]
    # if not config.whitened:
    #     finite_i_flux = [np.isfinite(spline_flux[i]) for i in range(len(spline_flux))]
    #     finite_i = [np.logical_or(finite_i_centroid[i], finite_i_flux[i]) for i in range(len(finite_i_centroid))]
    # else:
    #     finite_i = finite_i_centroid
    #
    # # remove cadences for which the spline did not fit
    # all_time = [all_time[i][finite_i[i]] for i in range(len(all_time))]
    #
    # # normalize by the spline
    # centroid_dist = [centroid_dist[i][finite_i[i]] / spline_centroid[i][finite_i[i]] * avg_centroid_dist_oot
    #                  for i in range(len(spline_centroid))]
    #
    # if not config.whitened:
    #     all_flux = [all_flux[i][finite_i[i]] / spline_flux[i][finite_i[i]] for i in range(len(spline_flux))]

    centroid_dist = np.concatenate(centroid_dist)
    flux = np.concatenate(all_flux)
    time = np.concatenate(all_time)

    # impute the time series with noise based on global estimates of median and std
    if gap_time:

        med = {'flux': np.median(flux), 'centroid': np.median(centroid_dist)}
        # robust std estimator of the time series
        std_rob_estm = {'flux': np.median(np.abs(flux - med['flux'])) * 1.4826,
                        'centroid': np.median(np.abs(centroid_dist - med['centroid'])) * 1.4826}

        for time in gap_time:

            imputed_flux = med['flux'] + np.random.normal(0, std_rob_estm['flux'], time.shape)
            flux = np.append(flux, imputed_flux.astype(flux.dtype))

            imputed_centr = med['centroid'] + np.random.normal(0, std_rob_estm['centroid'], time.shape)
            centroid_dist = np.append(centroid_dist, imputed_centr.astype(centroid_dist.dtype))

            time = np.append(time, time.astype(time.dtype))

    return time, flux, centroid_dist


def min_max_normalization(arr, max_val, min_val):
    """ Min-max normalization.

    :param arr: array
    :param max_val: float, max val
    :param min_val: float, min val
    :return:
        normalized array
    """

    return (arr - min_val) / (max_val - min_val)


def phase_fold_and_sort_light_curve(time, flux, centroids, period, t0):
    """Phase folds a light curve and sorts by ascending time.

    Args:
      time: 1D NumPy array of time values.
      flux: 1D NumPy array of flux values.
      centroids: 1D NumPy array of centroid values
      period: A positive real scalar; the period to fold over.
      t0: The center of the resulting folded vector; this value is mapped to 0.

    Returns:
      folded_time: 1D NumPy array of phase folded time values in
          [-period / 2, period / 2), where 0 corresponds to t0 in the original
          time array. Values are sorted in ascending order.
      folded_flux: 1D NumPy array. Values are the same as the original input
          array, but sorted by folded_time.
      folded_centroids: 1D NumPy array. Values are the same as the original input array, but sorted by folded_time.
    """

    # Phase fold time.
    time = util.phase_fold_time(time, period, t0)

    # Sort by ascending time.
    sorted_i = np.argsort(time)
    time = time[sorted_i]
    flux = flux[sorted_i]
    centroids = centroids[sorted_i]

    return time, flux, centroids


def phase_fold_and_sort_light_curve_odd_even(time, flux, centroids, period, t0):
    """Creates separate phase-folded time vectors for odd and even periods.

    :param time: 1D NumPy array of time values
    :param flux: 1D NumPy array of flux values
    :param centroids: 1D NumPy array of centroid values
    :param period: A positive real scalar; the period to fold over.
    :param t0: The center of the resulting folded vector; this value is mapped to 0.
    :return:
        Two tuples of size three, each of which contains odd or even time values, flux values, and centroid values.
        time: 1D NumPy array of phase folded time values in [-period / 2, period / 2), where 0 corresponds to t0
            in the original time array. Values are sorted in ascending order.
        folded_flux: 1D NumPy array. Values are the same as the original input array, but sorted by folded_time.
        folded_centroids: 1D NumPy array. Values are the same as the original input array, but sorted by folded_time.
    """

    half_period = period / 2
    odd_indices = np.array([], dtype='int64')
    even_indices = np.array([], dtype='int64')

    switcher = 1
    i = t0

    # FIXME: using switcher here just makes sure that the period centered at t0 is odd - why is this relevant?
    # is the number of periods between t0 and tmin even (switcher=1) or odd (switcher=-1)?
    # starts counting in the period centered in t0 (assumed as odd period)
    # this is done so that we get the first valid period
    # if np.min(time) < i - half_period:
    if np.min(time) < i + half_period:
        # is the right side of the current period interval after minimum time?
        # while i - half_period >= np.min(time):
        # while i + half_period >= np.min(time):
        while i + half_period > np.min(time):
            # iterate to the previous period
            i -= period
            switcher *= -1

    # start in the first valid period interval
    i += period

    # starting from the first valid period interval, is the left side of the current period interval before the maximum
    # time?
    while i - half_period <= np.amax(time):
        if switcher == 1:
            # add odd indices
            odd_indices = np.concatenate(
                (odd_indices, np.where(
                    np.logical_and(
                        time < i + half_period,
                        time >= i - half_period
                    ))),
                axis=None
            )
        else:
            # add even indices
            even_indices = np.concatenate(
                (even_indices, np.where(
                    np.logical_and(
                        time < i + half_period,
                        time >= i - half_period
                    ))),
                axis=None
            )

        # iterate to the next period
        i += period
        # alternate between odd and even periods
        switcher *= -1

    # get odd and even values
    odd_time = np.take(time, odd_indices)
    even_time = np.take(time, even_indices)
    odd_flux = np.take(flux, odd_indices)
    even_flux = np.take(flux, even_indices)
    odd_centroids = np.take(centroids, odd_indices)
    even_centroids = np.take(centroids, even_indices)

    # Phase fold time.
    odd_time = util.phase_fold_time(odd_time, period, t0)
    even_time = util.phase_fold_time(even_time, period, t0)

    # Sort by ascending time.
    sorted_i_odd = np.argsort(odd_time)
    sorted_i_even = np.argsort(even_time)

    odd_result = odd_time[sorted_i_odd]
    even_result = even_time[sorted_i_even]
    odd_flux = odd_flux[sorted_i_odd]
    even_flux = even_flux[sorted_i_even]
    odd_centroids = odd_centroids[sorted_i_odd]
    even_centroids = even_centroids[sorted_i_even]

    assert(len(odd_time) + len(even_time) - len(time) == 0 and len(set(odd_time) & set(even_time)) == 0)

    return (odd_result, odd_flux, odd_centroids), (even_result, even_flux, even_centroids)


def generate_view(time, flux, num_bins, bin_width, t_min, t_max,
                  centering=True, normalize=True, centroid=False, eps=1e-32):
    """Generates a view of a phase-folded light curve using a median filter.

    Args:
      time: 1D array of time values, sorted in ascending order.
      flux: 1D array of flux/centroid values.
      num_bins: The number of intervals to divide the time axis into.
      bin_width: The width of each bin on the time axis.
      t_min: The inclusive leftmost value to consider on the time axis.
      t_max: The exclusive rightmost value to consider on the time axis.
      centering: bool, whether to center the view by subtracting the median
      normalize: Whether to perform normalization
      centroid: bool, if True considers these view a centroid time series
      eps: float, non-zero normalization term

    Returns:
      1D NumPy array of size num_bins containing the median flux values of
      uniformly spaced bins on the phase-folded time axis.
    """

    # binning using median
    view = median_filter.median_filter(time, flux, num_bins, bin_width, t_min, t_max)

    # global median centering
    if centering:
        view -= np.median(view)

    # normalization
    if normalize:
        view = normalize_view(view, val=None, centroid=centroid, eps=eps)

    return view


def normalize_view(view, val=None, centroid=False, eps=1e-32):
    """ Normalize the phase-folded time series.

    :param view: array, phase-folded time series
    :param val: float, value used to normalize the time series
    :param centroid: bool, True for centroid time series
    :param eps: float, non-zero normalization term
    :return:
        array, normalized phase-folded time series
    """

    # for the centroid time series
    # range [new_min_val, 1], assuming max is positive, which should be since we are removing the median from a
    # non-negative time series
    # for the flux time series
    # range [-1, new_max_val], assuming min is negative, if not [1, new_max_val]
    if val is None:
        val = np.abs(np.max(view)) if centroid else np.abs(np.min(view))

    # if val == 0:
    #     print('Dividing view by 0. Returning the non-normalized view.')
    #     return view
    # else:
    #     view = view / val
    #     # view /= val

    return view / (val + eps)


def global_view(time, flux, period, num_bins=2001, bin_width_factor=1/2001, centroid=False, normalize=True,
                centering=True, eps=1e-32):
    """Generates a 'global view' of a phase folded light curve.

    See Section 3.3 of Shallue & Vanderburg, 2018, The Astronomical Journal.
    http://iopscience.iop.org/article/10.3847/1538-3881/aa9e09/meta

    Args:
      time: 1D array of time values, sorted in ascending order.
      flux: 1D array of flux values.
      period: The period of the event (in days).
      num_bins: The number of intervals to divide the time axis into.
      bin_width_factor: Width of the bins, as a fraction of period.
      centering: bool, whether to center the view by subtracting the median
      normalize: Whether to perform normalization
      centroid: bool, if True considers these view a centroid time series
      eps: float, non-zero normalization term


    Returns:
      1D NumPy array of size num_bins containing the median flux values of
      uniformly spaced bins on the phase-folded time axis.
    """
    return generate_view(
        time,
        flux,
        num_bins=num_bins,
        bin_width=period * bin_width_factor,
        t_min=-period / 2,
        t_max=period / 2,
        centroid=centroid,
        normalize=normalize,
        centering=centering,
        eps=eps)


def local_view(time,
               flux,
               period,
               duration,
               num_bins=201,
               bin_width_factor=0.16,
               num_durations=4,
               centroid=False,
               normalize=True,
               centering=True,
               eps=1e-32):
    """Generates a 'local view' of a phase folded light curve.

    See Section 3.3 of Shallue & Vanderburg, 2018, The Astronomical Journal.
    http://iopscience.iop.org/article/10.3847/1538-3881/aa9e09/meta

    Args:
      time: 1D array of time values, sorted in ascending order.
      flux: 1D array of flux values.
      period: The period of the event (in days).
      duration: The duration of the event (in days).
      num_bins: The number of intervals to divide the time axis into.
      bin_width_factor: Width of the bins, as a fraction of duration.
      num_durations: The number of durations to consider on either side of 0 (the event is assumed to be centered at 0).
      centering: bool, whether to center the view by subtracting the median
      normalize: Whether to perform normalization
      centroid: bool, if True considers these view a centroid time series


    Returns:
      1D NumPy array of size num_bins containing the median flux values of
      uniformly spaced bins on the phase-folded time axis.
    """
    return generate_view(
        time,
        flux,
        num_bins=num_bins,
        bin_width=duration * bin_width_factor,
        t_min=max(-period / 2, -duration * num_durations),
        t_max=min(period / 2, duration * num_durations),
        centroid=centroid,
        normalize=normalize,
        centering=centering,
        eps=eps)


# def generate_example_for_tce(time, flux, centroids, tce, config, transit_times):
def generate_example_for_tce(time, flux, centroids, tce, config, plot_preprocessing_tce=False):
    """ Generates a tf.train.Example representing an input TCE.

    Args:
      time: 1D NumPy array; the time values of the light curve.
      flux: 1D NumPy array; the flux values of the light curve.
      centroids: 1D NumPy array; the flux values of the light curve.
      tce: Dict-like object containing at least 'tce_period', 'tce_duration', and
        'tce_time0bk'. Additional items are included as features in the output.
      config: Config object; preprocessing parameters.
      plot_preprocessing_tce: bool, if True plots figures for some steps while generating the inputs.

    Returns:
      A tf.train.Example containing features. These features can be time series, stellar parameters, statistical
      quantities, .... it returns None if some exception while creating these features occurs
    """

    # get ephemeris
    # TODO: standardize keywords in the TCE tables for TESS and Kepler so that we do not need to do this check..
    # if config.satellite == 'kepler':
    #     period, duration, t0 = tce["tce_period"], tce["tce_duration"], tce["tce_time0bk"]
    # else:
    #     period, duration, t0 = tce["orbitalPeriodDays"], tce["transitDurationHours"], tce["transitEpochBtjd"]
    period, duration, t0 = tce["tce_period"], tce["tce_duration"], tce["tce_time0bk"]

    # phase folding for odd and even time series
    (odd_time, odd_flux, odd_centroid), \
        (even_time, even_flux, even_centroid) = phase_fold_and_sort_light_curve_odd_even(time,
                                                                                         flux,
                                                                                         centroids,
                                                                                         period,
                                                                                         t0)

    # phase folding for flux and centroid time series
    time, flux, centroid = phase_fold_and_sort_light_curve(time, flux, centroids, period, t0)

    # make output proto
    ex = tf.train.Example()

    # set time series features
    try:
        # TODO: take into account that eps value depends on the kind of data and preprocessing done previously

        # get flux views
        glob_view = global_view(time, flux, period, normalize=True, centering=True,
                                num_bins=config.num_bins_glob, bin_width_factor=config.bin_width_factor_glob, eps=1e-32)
        loc_view = local_view(time, flux, period, duration, normalize=True, centering=True,
                              num_bins=config.num_bins_loc, bin_width_factor=config.bin_width_factor_loc, eps=1e-32)

        # get centroid views
        glob_view_centr = global_view(time, centroid, period, centroid=True, normalize=False, centering=False,
                                      num_bins=config.num_bins_glob, bin_width_factor=config.bin_width_factor_glob,
                                      eps=1e-32)
        loc_view_centr = local_view(time, centroid, period, duration, centroid=True, normalize=False, centering=False,
                                    num_bins=config.num_bins_loc, bin_width_factor=config.bin_width_factor_loc,
                                    eps=1e-32)

        # get odd views (flux and centroid)
        glob_view_odd = global_view(odd_time, odd_flux, period, normalize=False, centering=True,
                                    num_bins=config.num_bins_glob, bin_width_factor=config.bin_width_factor_glob,
                                    eps=1e-32)
        loc_view_odd = local_view(odd_time, odd_flux, period, duration, normalize=False, centering=True,
                                  num_bins=config.num_bins_loc, bin_width_factor=config.bin_width_factor_loc, eps=1e-32)

        # glob_view_odd_centr = global_view(odd_time, odd_centroid, period, centroid=True, normalize=False,
        #                                   num_bins=config.num_bins_glob, bin_width_factor=config.bin_width_factor_glob)
        # loc_view_odd_centr = local_view(odd_time, odd_centroid, period, duration, centroid=True, normalize=False,
        #                                 num_bins=config.num_bins_loc, bin_width_factor=config.bin_width_factor_loc)

        # get even views (flux and centroid)
        glob_view_even = global_view(even_time, even_flux, period, normalize=False, centering=True,
                                     num_bins=config.num_bins_glob, bin_width_factor=config.bin_width_factor_glob,
                                     eps=1e-32)
        loc_view_even = local_view(even_time, even_flux, period, duration, normalize=False, centering=True,
                                   num_bins=config.num_bins_loc, bin_width_factor=config.bin_width_factor_loc,
                                   eps=1e-32)

        # glob_view_even_centr = global_view(even_time, even_centroid, period, centroid=True, normalize=False,
        #                                    num_bins=config.num_bins_glob, bin_width_factor=config.bin_width_factor_glob)
        # loc_view_even_centr = local_view(even_time, even_centroid, period, duration, centroid=True, normalize=False,
        #                                  num_bins=config.num_bins_loc, bin_width_factor=config.bin_width_factor_loc)

        # normalize odd and even views by their joint minimum (flux)/maximum (centroid)
        val_norm = np.abs(min(np.min(glob_view_even), np.min(glob_view_odd)))
        glob_view_even = normalize_view(glob_view_even, val=val_norm, eps=1e-32)
        glob_view_odd = normalize_view(glob_view_odd, val=val_norm, eps=1e-32)

        val_norm = np.abs(min(np.min(loc_view_even), np.min(loc_view_odd)))
        loc_view_even = normalize_view(loc_view_even, val=val_norm, eps=1e-32)
        loc_view_odd = normalize_view(loc_view_odd, val=val_norm, eps=1e-32)

        # # TODO: should we normalize the odd and even centroid jointly or use the normalization factor used for the
        #  main centroid time series?
        # # val_norm = np.abs(max(np.max(glob_view_even_centr), np.max(glob_view_odd_centr)))
        # # glob_view_even_centr = normalize_view(glob_view_even_centr, val=val_norm, centroid=True)
        # # glob_view_odd_centr = normalize_view(glob_view_odd_centr, val=val_norm, centroid=True)
        # #
        # # val_norm = np.abs(max(np.max(loc_view_even_centr), np.max(loc_view_odd_centr)))
        # # loc_view_odd_centr = normalize_view(loc_view_odd_centr, val=val_norm, centroid=True)
        # # loc_view_even_centr = normalize_view(loc_view_even_centr, val=val_norm, centroid=True)

        # # # normalize the global and local views centroids by the maximum value in the training set
        # # max_centr = np.load('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Kepler_planet_finder/src_preprocessing/'
        # #                     'max_centr_trainingset.npy').item()
        # # glob_view_centr = normalize_view(glob_view_centr, val=max_centr['global_view'], centroid=True)
        # # loc_view_centr = normalize_view(loc_view_centr, val=max_centr['local_view'], centroid=True)

        # # if plot_preprocessing_tce:
        # #     views = {'global_view': glob_view, 'local_view': loc_view, 'global_view_odd': glob_view_odd,
        # #              'local_view_odd': loc_view_odd, 'global_view_even': glob_view_even,
        # #              'local_view_even': loc_view_even, 'global_view_centr': glob_view_centr,
        # #              'local_view_centr': loc_view_centr}
        # #
        # #     plot_all_views(views, tce, '5nonnormalized_centroids', (2, 4))
        #

        # load training set statistics for normalization of the centroid time-series views
        # stats_trainset = np.load(config.stats_preproc_filepath).item()

        # if plot_preprocessing_tce:
        #     views = {'global_view': glob_view, 'local_view': loc_view,
        #              'global_view_centr': glob_view_centr, 'local_view_centr': loc_view_centr}
        #     # views = {'global_view': glob_view, 'local_view': loc_view, 'global_view_odd': glob_view_odd,
        #     #          'local_view_odd': loc_view_odd, 'global_view_even': glob_view_even,
        #     #          'local_view_even': loc_view_even, 'global_view_centr': glob_view_centr,
        #     #          'local_view_centr': loc_view_centr}
        #
        #     utils_visualization.plot_all_views(views, tce, (2, 2), os.path.join(config.output_dir, 'plots/'),
        #                                        '8_views_nonnormalizedcentroids')

        # # FDL centroid normalization
        # # median centering using median computed based on the training set
        # glob_view_centr1 = glob_view_centr - stats_trainset['global_view_centr']['median_oot']
        # loc_view_centr1 = loc_view_centr - stats_trainset['local_view_centr']['median_oot']
        # # normalization using std of the centroid and flux views computed based on the training set
        # glob_view_centr1 = normalize_view(glob_view_centr1, val=stats_trainset['global_view_centr']['std_oot'] /
        #                                                        np.std(glob_view), centroid=True)
        # loc_view_centr1 = normalize_view(loc_view_centr1, val=stats_trainset['local_view_centr']['std_oot'] /
        #                                                      np.std(loc_view), centroid=True)

        # min-max normalization of the centroid views computed based on the training set
        # glob_view_centr2 = min_max_normalization(glob_view_centr,
        #                                          stats_trainset['global_view_centr']['max'],
        #                                          stats_trainset['global_view_centr']['min'])
        # loc_view_centr2 = min_max_normalization(loc_view_centr,
        #                                         stats_trainset['local_view_centr']['max'],
        #                                         stats_trainset['local_view_centr']['min'])
        # glob_view_centr2 = min_max_normalization(glob_view_centr,
        #                                          glob_view_centr.max(),
        #                                          glob_view_centr.min())
        # loc_view_centr2 = min_max_normalization(loc_view_centr,
        #                                         loc_view_centr.max(),
        #                                         loc_view_centr.min())

        # # individual median centering
        # glob_view_centr2 = glob_view_centr2 - np.median(glob_view_centr2)
        # loc_view_centr2 = loc_view_centr2 - np.median(loc_view_centr2)

        # individual median oot normalization
        nr_transit_durations = 2 * 4 + 1
        transit_duration_bins_loc = config.num_bins_loc / nr_transit_durations
        nontransitcadences_loc = np.array([True] * config.num_bins_loc)
        nontransitcadences_loc[np.arange(int(np.floor((config.num_bins_loc - transit_duration_bins_loc) / 2)),
                                         int(np.ceil((config.num_bins_loc + transit_duration_bins_loc) / 2)))] = False
        idxs_nontransitcadences_loc = np.where(nontransitcadences_loc)

        frac_durper = duration / period
        nontransitcadences_glob = np.array([True] * config.num_bins_glob)
        nontransitcadences_glob[np.arange(int(config.num_bins_glob / 2 * (1 - frac_durper)),
                                          int(config.num_bins_glob / 2 * (1 + frac_durper)))] = False
        idxs_nontransitcadences_glob = np.where(nontransitcadences_glob)

        glob_avg_oot_centr = np.median(glob_view_centr[idxs_nontransitcadences_glob])
        loc_avg_oot_centr = np.median(loc_view_centr[idxs_nontransitcadences_loc])

        glob_view_centr /= glob_avg_oot_centr
        loc_view_centr /= loc_avg_oot_centr

        # views = {'global_view': glob_view, 'local_view': loc_view, 'global_view_centr_fdl': glob_view_centr1,
        #          'local_view_centr_fdl': loc_view_centr1, 'global_view_centr': glob_view_centr2,
        #          'local_view_centr': loc_view_centr2, 'global_view_odd': glob_view_odd,
        #          'local_view_odd': loc_view_odd, 'global_view_even': glob_view_even,
        #          'local_view_even': loc_view_even}

        # views = {'global_view': glob_view, 'local_view': loc_view, 'global_view_centr': glob_view_centr,
        #          'local_view_centr': loc_view_centr, 'global_view_odd': glob_view_odd,
        #          'local_view_odd': loc_view_odd, 'global_view_even': glob_view_even,
        #          'local_view_even': loc_view_even}
        views = {'global_view': glob_view, 'local_view': loc_view, 'global_view_centr': glob_view_centr,
                 'local_view_centr': loc_view_centr, 'global_view_odd': glob_view_odd, 'local_view_odd': loc_view_odd,
                 'global_view_even': glob_view_even, 'local_view_even': loc_view_even}

        # compute oot mean and std for each view
        auxiliary_data = {}
        for view in views:
            auxiliary_data['{}_meanoot'.format(view)] = \
                np.mean(views[view][(idxs_nontransitcadences_loc, idxs_nontransitcadences_glob)['glob' in view]])
            auxiliary_data['{}_stdoot'.format(view)] = \
                np.std(views[view][(idxs_nontransitcadences_loc, idxs_nontransitcadences_glob)['glob' in view]])

        if plot_preprocessing_tce:
            # CHANGE NUMBER OF VIEWS PLOTTED!!!
            utils_visualization.plot_all_views(views, tce, config, (2, 4), os.path.join(config.output_dir, 'plots/'),
                                               '8_views_normalized')

    except Exception as e:
        report_exclusion(config, tce, 'Error when creating views', stderr=e)
        return None

    # TODO: exclude TCEs that have at least one view with at least a NaN/infinite value? Right now is checking just for
    #  one finite value...
    for view in views:
        if not np.any(np.isfinite(views[view])):  # at least one point is not infinite or NaN
            report_exclusion(config, tce, 'Views are NaNs.')
            return None

    # set time series features in the tfrecord
    for view in views:
        example_util.set_float_feature(ex, view, views[view])

        # add auxiliary data
        example_util.set_float_feature(ex, '{}_meanoot'.format(view), [auxiliary_data['{}_meanoot'.format(view)]])
        example_util.set_float_feature(ex, '{}_stdoot'.format(view), [auxiliary_data['{}_stdoot'.format(view)]])

    # # save statistics about the non-normalized views in numpy files, one per TCE
    # extrema_centr = {'glob_view_centr': {'max': np.nanmax(glob_view_centr), 'min': np.nanmin(glob_view_centr)},
    #                  'loc_view_centr': {'max': np.nanmax(loc_view_centr), 'min': np.nanmin(loc_view_centr)}}
    # np.save(os.path.join(config.output_dir, 'stats',
    #                      'extrema_centroid-{}_{}'.format(tce['kepid'], tce['tce_plnt_num'])), extrema_centr)

    # set other features from the TCE table - TCE and stellar parameters
    scalar_params = []
    for name, value in tce.items():
        if name in config.scalar_params:
            scalar_params.append(value)
        else:  # set individual parameters
            example_util.set_feature(ex, name, [value])

    if len(scalar_params) > 0:
        example_util.set_feature(ex, 'scalar_params', scalar_params)

    return ex
