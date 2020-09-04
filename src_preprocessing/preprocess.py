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
import socket

# local
from src_preprocessing.light_curve import kepler_io
from src_preprocessing.light_curve import median_filter
from src_preprocessing.light_curve import util
from src_preprocessing.tf_util import example_util
from src_preprocessing.third_party.kepler_spline import kepler_spline
from src_preprocessing import utils_visualization
from src_preprocessing.utils_centroid_preprocessing import kepler_transform_pxcoordinates_mod13
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
        # main_str = 'KepID {} TCE {}'.format(tce.kepid, tce.tce_plnt_num)
        main_str = 'KeplerID {} TCE {}'.format(tce.target_id, tce[config.tce_identifier])
    else:  # 'tess'
        main_str = 'TICID {} TCE {} Sector(s) {}'.format(tce.target_id, tce[config.tce_identifier], tce.sectors)

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
        file_names = kepler_io.kepler_filenames(config.lc_data_dir,
                                                tce.target_id,
                                                injected_group=config.injected_group)

        if not file_names:
            if not config.omit_missing:
                raise IOError("Failed to find .fits files in {} for Kepler ID {}".format(config.lc_data_dir,
                                                                                         tce.target_id))
            else:
                report_exclusion(config, tce, 'No available lightcurve .fits files')
                return None

        return kepler_io.read_kepler_light_curve(file_names,
                                                 centroid_radec=not config.px_coordinates,
                                                 prefer_psfcentr=config.prefer_psfcentr,
                                                 light_curve_extension=config.light_curve_extension,
                                                 scramble_type=config.scramble_type,
                                                 invert=config.invert)

    else:
        # file_names = tess_io.tess_filenames(config.lc_data_dir, tce.tic, tce.sector, config.multisector)

        # # the sector field decides which sectors to look for the TCE
        # # can be single- or multi-sector run
        # if 'sectors' in tce:  # if there is a 'sector' field in the TCE table
        #     # overlap between sectors defined in the config and sector in the TCE table for the TCE
        #     if config.sectors is not None:
        #         sectors = [int(sector) for sector in tce.sector if int(sector) in config.sectors]
        #     # get sectors in the TCE table for the TCE
        #     else:
        #         sectors = [int(sector) for sector in tce.sector]
        #     # multisector = 'table'
        # else:  # get sectors defined in the config
        #     sectors = config.sectors
        #     # multisector = 'no-table'

        sectors = [int(sect) for sect in tce['sectors'].split(' ')]

        # file_names, tce_sectors = tess_io.tess_filenames(config.lc_data_dir, tce.target_id, sectors)
        file_names = tess_io.tess_filenames(config.lc_data_dir, tce.target_id, sectors)

        # if 'sector' not in tce:
        #     tce.sector = tce_sectors

        if not file_names:
            if not config.omit_missing:
                raise IOError("Failed to find .fits files in {} for TESS ID {}".format(config.lc_data_dir,
                                                                                       tce.target_id))
            else:
                report_exclusion(config, tce, 'No available lightcurve .fits files')
                return None

        return tess_io.read_tess_light_curve(file_names,
                                             centroid_radec=not config.px_coordinates,
                                             prefer_psfcentr=config.prefer_psfcentr,
                                             light_curve_extension=config.light_curve_extension,
                                             invert=config.invert
                                             )


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
    """ Linearly interpolate the timeseries across the in-transit cadences. The interpolation is performed between the
    boundary points using their values.

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


def impute_transits(timeseries, transit_pulse_train, centroid=False):
    """ Impute transits by linearly interpolating them using as boundary values the out-of-transit median in windows
    around the transits. These windows have the same width as transit.

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

            numcadences_win = idxs_it[end_idx - 1] - idxs_it[start_idx] + 1

            # boundary issue - do nothing, since the whole array is a transit; does this happen?
            if idxs_it[start_idx] == 0 and idxs_it[end_idx - 1] == len(transit_pulse_train[i]) - 1:
                continue

            if idxs_it[start_idx] == 0:  # boundary issue start - constant value end

                if centroid:
                    val_interp = {'x': np.median(timeseries['x'][i][idxs_it[end_idx - 1] + 1:min(len(timeseries['x'][i]), idxs_it[end_idx - 1] + 1 + numcadences_win)]),
                                  'y': np.median(timeseries['y'][i][idxs_it[end_idx - 1] + 1:min(len(timeseries['y'][i]), idxs_it[end_idx - 1] + 1 + numcadences_win)])}
                    timeseries_interp['x'][i][idxs_it[start_idx]:idxs_it[end_idx - 1] + 1] = val_interp['x']
                    timeseries_interp['y'][i][idxs_it[start_idx]:idxs_it[end_idx - 1] + 1] = val_interp['y']

                else:
                    val_interp = np.median(timeseries[i][idxs_it[end_idx - 1] + 1:min(len(timeseries[i]), idxs_it[end_idx - 1] + 1 + numcadences_win)])
                    timeseries_interp[i][idxs_it[start_idx]:idxs_it[end_idx - 1] + 1] = val_interp

            elif idxs_it[end_idx - 1] == len(transit_pulse_train[i]) - 1:  # boundary issue end - constant value start

                if centroid:
                    val_interp = {'x': np.median(timeseries['x'][i][max(0, idxs_it[start_idx] - numcadences_win):idxs_it[start_idx] + 1]),
                                  'y': np.median(timeseries['y'][i][max(0, idxs_it[start_idx] - numcadences_win):idxs_it[start_idx] + 1])}

                    timeseries_interp['x'][i][idxs_it[start_idx]:idxs_it[end_idx - 1] + 1] = val_interp['x']
                    timeseries_interp['y'][i][idxs_it[start_idx]:idxs_it[end_idx - 1] + 1] = val_interp['y']
                else:
                    val_interp = np.median(timeseries[i][max(0, idxs_it[start_idx] - numcadences_win):idxs_it[start_idx] + 1])
                    timeseries_interp[i][idxs_it[start_idx]:idxs_it[end_idx - 1] + 1] = val_interp

            else:  # linear interpolation
                idxs_interp = np.array([idxs_it[start_idx] - 1, idxs_it[end_idx - 1] + 1])
                idxs_to_interp = np.arange(idxs_it[start_idx], idxs_it[end_idx - 1] + 1)

                if centroid:
                    val_interp = {'x': np.array([np.median(timeseries['x'][i][max(0, idxs_interp[0] - numcadences_win):idxs_interp[0] + 1]),
                                                np.median(timeseries['x'][i][idxs_interp[1]:min(len(timeseries['x'][i]), idxs_interp[1] + numcadences_win) + 1])]),
                                  'y': np.array([np.median(timeseries['y'][i][max(0, idxs_interp[0] - numcadences_win):idxs_interp[0] + 1]),
                                                np.median(timeseries['y'][i][idxs_interp[1]:min(len(timeseries['y'][i]), idxs_interp[1] + numcadences_win) + 1])])}
                    timeseries_interp['x'][i][idxs_it[start_idx]:idxs_it[end_idx - 1] + 1] = \
                        np.interp(idxs_to_interp, idxs_interp, val_interp['x'])
                    timeseries_interp['y'][i][idxs_it[start_idx]:idxs_it[end_idx - 1] + 1] = \
                        np.interp(idxs_to_interp, idxs_interp, val_interp['y'])

                else:
                    val_interp = np.array([np.median(timeseries[i][max(0, idxs_interp[0] - numcadences_win):idxs_interp[0] + 1]),
                                           np.median(timeseries[i][idxs_interp[1]:min(len(timeseries[i]), idxs_interp[1] + numcadences_win) + 1])])

                    timeseries_interp[i][idxs_it[start_idx]:idxs_it[end_idx - 1] + 1] = \
                        np.interp(idxs_to_interp, idxs_interp, val_interp)

    return timeseries_interp


def gap_this_tce(all_time, tce, gap_pad=0):
    """ Remove from the time series the cadences that belong to the TCE in the light curve. These values are set to
    NaN.

    :param all_time: list of NumPy arrays, cadences
    :param tce: row of pandas DataFrame, main TCE ephemeris
    :param gap_pad: extra pad on both sides of the gapped TCE transit duration
    :return:
        gapped_idxs: list of numpy arrays, gapped indices
    """

    gapped_idxs = []

    # find gapped cadences for the TCE
    # transit_duration = tce['tce_duration'] * (1 + 2 * gap_pad)
    # transit_duration = max(min(tce['tce_duration'] * (1 + 2 * gap_pad), tce['tce_period'] / 10),
    #                        tce['tce_duration'])
    transit_duration = min(tce['tce_duration'] * (1 + 2 * gap_pad), np.abs(tce['tce_maxmesd']))

    for i in range(len(all_time)):

        begin_time, end_time = all_time[i][0], all_time[i][-1]

        # get timestamp of the first transit of the gapped TCE in the current time array
        epoch = find_first_epoch_after_this_time(tce['tce_time0bk'], tce['tce_period'], begin_time)

        # create binary time-series for this time interval based on the ephemeris of the gapped TCE
        bintransit_ts = create_binary_time_series(all_time[i], epoch, transit_duration, tce['tce_period'])

        # get indexes of in-transit cadences for the gapped TCE in this time array
        transit_idxs = np.where(bintransit_ts == 1)

        gapped_idxs.append(transit_idxs)

    return gapped_idxs


def gap_other_tces(all_time, add_info, tce, table, config, conf_dict, gap_pad=0, keep_overlap=False):
    """ Remove from the time series the cadences that belong to other TCEs in the light curve. These values are set to
    NaN.

    :param all_time: list of numpy arrays, cadences
    :param tce: row of pandas DataFrame, main TCE ephemeris
    :param table: pandas DataFrame, TCE ephemeris table
    :param config: Config object, preprocessing parameters
    :param conf_dict: dict, keys are a tuple (Kepler ID, TCE planet number) and the values are the confidence level used
     when gapping (between 0 and 1)
    :param gap_pad: extra pad on both sides of the gapped TCE transit duration
    :param keep_overlap: bool, if True overlapping cadences between TCE of interest and gapped TCEs are preserved
    :return:
        gapped_idxs: list of NumPy arrays, gapped cadences
    """

    gapped_idxs = []

    # get gapped TCEs ephemeris
    if config.satellite == 'kepler':  # Kepler
        # get the ephemeris for TCEs in the same target star
        gap_ephems = table.loc[(table['target_id'] == tce.target_id) &
                               (table[config.tce_identifier] != tce[config.tce_identifier])][['tce_period',
                                                                                              'tce_duration',
                                                                                              'tce_time0bk']]
    else:  # TESS
        # gap_ephems = table.loc[(table['target_id'] == tce.tic) &
        #                        (table['sector'] == tce.sector) &
        #                        (table[config.tce_identifier] != tce[config.tce_identifier])]

        # # if sector column exists in the TCE table
        # if 'sector' in table:

        # get observed sectors for the current TCE
        sectors = np.array([int(sect) for sect in tce['sectors'].split(' ')])

        # get ephemeris for the TCEs which belong to the same target star
        candidateGapTceTable = table.loc[(table['target_id'] == tce.target_id) &
                                         (table[config.tce_identifier] !=
                                          tce[config.tce_identifier])][['tce_period',
                                                                        'tce_duration',
                                                                        'tce_time0bk',
                                                                        'sectors']]

        # get TCEs whose observed sectors overlap with the observed sectors for the current TCE
        candidatesRemoved = []
        gapSectors = []
        for i, candidate in candidateGapTceTable.iterrows():

            candidateSectors = np.array([int(sect) for sect in candidate['sectors'].split(' ')])

            # get only overlapping sectors
            sectorsIntersection = np.intersect1d(sectors, candidateSectors)
            if len(sectorsIntersection) > 0:
                gapSectors.append(sectorsIntersection)
            else:
                candidatesRemoved.append(i)

        # remove candidates that do not have any overlapping sectors
        gap_ephems = candidateGapTceTable.drop(candidateGapTceTable.index[candidatesRemoved],
                                               inplace=False).reset_index()

        # else:  # if not, gap all TCEs that belong to the same target star
        #
        #     gap_ephems = table.loc[(table['target_id'] == tce.target_id) &
        #                            (table[config.tce_identifier] != tce[config.tce_identifier])][['tce_period',
        #                                                                                           'tce_duration',
        #                                                                                           'tce_time0bk']]
        #
        #     gapSectors = {gtce_i: add_info['sector'] for gtce_i in range(len(gap_ephems))}

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

    # get transit duration of the TCE of interest
    transit_duration_main = tce['tce_duration'] * (1 + 2 * gap_pad)

    # gap cadences for all other TCEs in the same target star
    for i in range(len(all_time)):

        begin_time, end_time = all_time[i][0], all_time[i][-1]

        transit_idxs_gappedtces = []

        # find gapped cadences for each TCE
        for ephem_i, ephem in gap_ephems.iterrows():

            # gap more than transit duration to account for inaccuracies in the ephemeris estimates
            # ephem['tce_duration'] = ephem['tce_duration'] * (1 + 2 * gap_pad)
            # ephem['tce_duration'] = max(min(ephem['tce_duration'] * (1 + 2 * gap_pad), ephem['tce_period'] / 10),
            #                             ephem['tce_duration'])
            duration_gapped = ephem['tce_duration'] * (1 + 2 * gap_pad)

            # for TESS, check if this time array belongs to one of the overlapping sectors
            if config.satellite != 'kepler' and add_info['sectors'][i] not in gapSectors[ephem_i]:
                continue

            # get timestamp of the first transit of the gapped TCE in the current time array
            epoch = find_first_epoch_after_this_time(ephem['tce_time0bk'], ephem['tce_period'], begin_time)

            # create binary time-series for this time interval based on the ephemeris of the gapped TCE
            bintransit_ts = create_binary_time_series(all_time[i], epoch, duration_gapped, ephem['tce_period'])

            # get indexes of in-transit cadences for the gapped TCE in this time array
            transit_idxs = np.where(bintransit_ts == 1)[0]

            transit_idxs_gappedtces.extend(list(transit_idxs))

        if keep_overlap:  # do not gap cadences that belong to the TCE of interest

            # get timestamp of the first transit of the main TCE in the current time array
            epoch_main = find_first_epoch_after_this_time(tce['tce_time0bk'], tce['tce_period'], begin_time)

            # create binary time-series for this time interval based on the ephemeris of the main TCE
            bintransit_ts_main = create_binary_time_series(all_time[i], epoch_main, transit_duration_main,
                                                           tce['tce_period'])

            # get indexes of in-transit cadences for the main TCE in this time array
            transit_idxs_main = np.where(bintransit_ts_main == 1)[0]

            # exclude from gapped indices the indices of the in-transit cadences for the main TCE
            transit_idxs_gappedtces = np.setdiff1d(transit_idxs_gappedtces, transit_idxs_main)

        gapped_idxs.append(np.array(transit_idxs_gappedtces))

    return gapped_idxs


def gap_tces(all_time, ephem, config):
    """ Get gapped indices for a given set of one or more TCEs to be gapped.

    :param all_time: list of Numpy arrays, timestamps
    :param ephem: list of ephemeris of TCEs to be gapped
    :param config: Config object, preprocessing parameters
    :return:
        all_transit_idxs: list of Numpy arrays containing boolean arrays with True value for gapped indices
        imputed_time: list of Numpy arrays, timestamps for gapped time
    """

    all_transit_idxs = []
    imputed_time = []

    for time_i in range(len(all_time)):

        transit_idxs = np.zeros(len(all_time[time_i]), dtype='bool')
        for ephem_i in range(len(ephem)):

            # # for TESS, check if this time array belongs to one of the overlapping sectors
            # if config.satellite != 'kepler' and add_info['sector'][i] not in gapSectors[real_ephem_i]:
            #     continue

            begin_time, end_time = all_time[time_i][0], all_time[time_i][-1]

            # get timestamp of the first transit of the gapped TCE in the current time array
            epoch = find_first_epoch_after_this_time(ephem['tce_time0bk'][ephem_i], ephem['tce_period'][ephem_i],
                                                     begin_time)

            # create binary time-series for this time interval based on the ephemeris of the gapped TCE
            bintransit_ts = create_binary_time_series(all_time[time_i], epoch, ephem['tce_duration'][ephem_i],
                                                      ephem_i['tce_period'])

            # add indices of in-transit cadences for the gapped TCE in this time array
            transit_idxs = np.logical_or(transit_idxs, bintransit_ts)

        # get gapped cadences to be imputed
        if config.gap_imputed and len(transit_idxs) > 0:
            imputed_time.append(all_time[time_i][transit_idxs])

        all_transit_idxs.append(transit_idxs)

    # real_ephem_i += 1

    return all_transit_idxs, imputed_time


def imputing_gaps(time, timeseries, all_gap_time):
    """ Imputing missing time with Gaussian noise computed taking into account global statistics of the data.

    :param time: list of Numpy arrays, contains the non-gapped timestamps
    :param timeseries: list of Numpy arrays, 1D time-series
    :param all_gap_time: list of Numpy arrays, gapped timestamps
    :return:
        time: list of Numpy arrays, now with added gapped timestamps
        timeseries: list of Numpy arrays, now with imputed data

    TODO: compute statistics using only datapoints from cadences close to the gapped time intervals and also in o.o.t
          values.
    """

    med = np.median(timeseries)
    # robust std estimator of the time series
    std_rob_estm = np.median(np.abs(timeseries - med['flux'])) * 1.4826

    for gap_time in all_gap_time:
        imputed_timeseries = med + np.random.normal(0, std_rob_estm, time.shape)
        timeseries.append(imputed_timeseries.astype(timeseries[0].dtype))
        time.append(gap_time, time.astype(time[0].dtype))

    return time, timeseries


def _process_tce(tce, table, config, conf_dict):
    """ Processes the time-series and scalar features for the TCE and returns an Example proto.

    Args:
        tce: row of the input TCE table.
        table: pandas DataFrame, ephemeris information on other TCEs used when gapping
        config: Config object, holds preprocessing parameters
        conf_dict: dict, keys are a tuple (Kepler ID, TCE planet number) and the values are the confidence level used
        when gapping (between 0 and 1)
        # gap_ids: list of numpy arrays, gapped cadences

    Returns:
        A tensorflow.train.Example proto containing TCE features.
    """

    import pandas as pd
    # # rankingTbl = pd.read_csv(
    # #     '/home/msaragoc/Projects/Kepler-TESS_exoplanet/Kepler_planet_finder/results_ensemble/'
    # #     'keplerdr25_nontces_g2001-l201_spline_gbal_nongapped_norobovetterkois_starshuffle_configE_glflux-glcentr-loe-6stellar_prelu/'
    # #     'ensemble_ranked_predictions_predictset.csv')
    # # rankingTbl = pd.read_csv('/home/msaragoc/Downloads/ml_tces_first_19.csv')
    # rankingTbl = pd.read_csv('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Analysis/odd-even_min_comparison/'
    #                          'keplerq1q17dr25_norobovetterkois_normalized_oddeven_views_diff_koicomment_koidisp_fpwgdisp.csv')
    # rankingTbl = rankingTbl.loc[rankingTbl['label'] == 'PC']
    # rankingTbl = rankingTbl[-20:]
    # rankingTbl = pd.read_csv('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Analysis/num_tces_per_disp/tce_counts_per_target.csv')
    # rankingTbl = rankingTbl.loc[rankingTbl['label'] == 'PC']
    #
    # if len(rankingTbl.loc[(rankingTbl['target_id'] == tce['target_id']) &
    #                       (rankingTbl['tce_plnt_num'] == tce['tce_plnt_num'])]) == 1:
    # # if tce['target_id'] in rankingTbl[0:30]['target_id'].values:
    # # if tce['target_id'] in rankingTbl['KICID'].values and tce['tce_plnt_num'] == 1:
    # if tce['target_id'] in rankingTbl[0:10]['target_id'].values:
    # if tce['target_id'] == 11442793 and tce['tce_plnt_num'] == 1:  # tce['av_training_set'] == 'PC' and
    # # if tce['oi'] == 541.01:
    #     print(tce)
    # else:
    #     return None

    # check if preprocessing pipeline figures are saved for the TCE
    plot_preprocessing_tce = False  # False
    if np.random.random() < 0.01:
        plot_preprocessing_tce = config.plot_figures

    # sample TCE ephemeris using uncertainty interval
    if tce['augmentation_idx'] != 0 and config.augmentation:  # one of the TCEs does not use uncertainty values

        # tce_tbl_ephem = {ephemeris: tce[ephemeris] for ephemeris in ['tce_period', 'tce_duration', 'tce_time0bk']}
        #
        # # fill missing values in uncertainty
        # for ephemeris in ['tce_period', 'tce_duration', 'tce_time0bk']:
        #     if tce['{}_err'.format(ephemeris)] == -1:
        #         tce['{}_err'.format(ephemeris)] = config.tce_min_err[ephemeris]
        #
        #     tce[ephemeris] = np.random.normal(tce[ephemeris], tce['{}_err'.format(ephemeris)])
        #
        # # sample ephemerides until they are positive
        # for ephemeris in ['tce_period', 'tce_duration']:
        #     while tce[ephemeris] <= 0:
        #         tce[ephemeris] = np.random.normal(tce_tbl_ephem[ephemeris], tce['{}_err'.format(ephemeris)])

        # add noise to the ephemerides; assuming augmenting data with 2 more TCEs (ephemeris * (1 +- aug_noise_factor))
        for ephemeris in ['tce_period', 'tce_duration', 'tce_time0bk']:
            tce[ephemeris] = tce[ephemeris] * (1 - config.aug_noise_factor,
                                               1 + config.aug_noise_factor)[tce['augmentation_idx'] == 1]

    # initialize data dict
    data = {}

    # get cadence, flux and centroid data for the tce
    data_fits = read_light_curve(tce, config)

    if data_fits is None:
        report_exclusion(config, tce, 'Empty arrays')
        return None

    data.update(data_fits)

    if config.satellite == 'kepler':
        add_info = {'quarter': data['quarter'], 'module': data['module']}
    else:
        add_info = {'sectors': data['sectors']}

    # # TODO: do the same thing for TESS
    # if config.get_denoised_centroids:
    #     all_centroids, idxs_nan_centroids = get_denoised_centroids_kepler(tce.target_id, config.denoised_centroids_dir)

    if plot_preprocessing_tce:
        utils_visualization.plot_centroids(data['all_time'],
                                           data['all_centroids'],
                                           None,
                                           tce,
                                           config,
                                           os.path.join(config.output_dir, 'plots'),
                                           '1_raw_aug{}'.format(tce['augmentation_idx']),
                                           add_info=add_info,
                                           target_position=None
                                           )

        if not config.px_coordinates:
            utils_visualization.plot_centroids(data['all_time'],
                                               data['all_centroids'],
                                               None,
                                               tce,
                                               config,
                                               os.path.join(config.output_dir, 'plots'),
                                               '1_raw_target_aug{}'.format(tce['augmentation_idx']),
                                               add_info=add_info,
                                               target_position=data['target_position']
                                               )

        utils_visualization.plot_centroids(data['all_time'], data['all_centroids_px'], None, tce, config,
                                           os.path.join(config.output_dir, 'plots'),
                                           '1_raw_fdl_aug{}'.format(tce['augmentation_idx']),
                                           add_info=add_info,
                                           pxcoordinates=True)

    # gap TCE to get weak secondary test time-series
    # get gapped indices
    data['gap_time_noprimary'] = None
    # initialize arrays for gapped time series
    timeseries_gapped = ['all_time_noprimary', 'all_flux_noprimary']
    data['all_time_noprimary'] = [np.array(arr) for arr in data['all_time']]
    data['all_flux_noprimary'] = [np.array(arr) for arr in data['all_flux']]
    gapped_idxs_noprimary = gap_this_tce(data['all_time_noprimary'], tce, gap_pad=config.gap_padding)
    if config.gap_imputed:
        data['gap_time_noprimary'] = []
    # set to NaN gapped cadences
    for arr_i in range(len(gapped_idxs_noprimary)):
        for timeseries in timeseries_gapped:
            data[timeseries][arr_i][gapped_idxs_noprimary[arr_i]] = np.nan
        if config.gap_imputed and len(gapped_idxs_noprimary[arr_i]) > 0:
            data['gap_time_noprimary'].append(data['all_time'][arr_i][gapped_idxs_noprimary[arr_i]])

    # non-gapped time array required for FDL centroid time-series
    data['all_time_nongapped'] = [np.array(el) for el in data['all_time']]

    # estimate the global (across quarters) average out-of-transit centroid position before gapping TCEs
    # duration_gapped = max(min(tce['tce_duration'] * (1 + 2 * 1), tce['tce_period'] / 10), tce['tce_duration'])
    duration_gapped = min(3 * tce['tce_duration'], np.abs(tce['tce_maxmesd']))
    # remove NaNs
    # centroid time-series
    joint_timeseries = ['all_time', 'all_centroids']
    finite_idxs = []
    for arr_i in range(len(data[joint_timeseries[0]])):
        finite_idxs_aux = []
        for timeseries in joint_timeseries:
            if 'time' not in timeseries:
                finite_idxs_aux.append(np.isfinite(data[timeseries]['x'][arr_i]))
                finite_idxs_aux.append(np.isfinite(data[timeseries]['y'][arr_i]))
            else:
                finite_idxs_aux.append(np.isfinite(data[timeseries][arr_i]))
        finite_idxs.append(np.logical_and.reduce(finite_idxs_aux))

    # create binary time series for each time array in which in-transit points are labeled as 1's, otherwise as 0's
    binary_time_all = [create_binary_time_series(time[finite_idx],
                                                 find_first_epoch_after_this_time(tce['tce_time0bk'],
                                                                                  tce['tce_duration'],
                                                                                  time[finite_idx][0]),
                                                 duration_gapped,
                                                 tce['tce_period'])
                       for time, finite_idx in zip(data['all_time'], finite_idxs)]

    # get out-of-transit indices for the centroid time series
    centroid_oot = {coord: [centroids[finite_idx][np.where(binary_time == 0)] for binary_time, centroids, finite_idx in
                            zip(binary_time_all, data['all_centroids'][coord], finite_idxs)]
                    for coord in data['all_centroids']}
    # estimate average out-of-transit centroid as the median across quarters - same as they do in the Kepler pipeline
    # TODO: how to compute the average oot? mean, median, other...
    data['avg_centroid_oot'] = {coord: np.nanmedian(np.concatenate(centroid_oot[coord])) for coord in centroid_oot}

    # gap cadences belonging to the transits of other TCEs in the same target star
    # FIXME: what if removes a whole quarter? need to adjust all_additional_info to it
    data['gap_time'] = None
    if config.gapped:
        gapped_idxs = gap_other_tces(data['all_time'],
                                     add_info,
                                     tce,
                                     table,
                                     config,
                                     conf_dict,
                                     gap_pad=config.gap_padding,
                                     keep_overlap=config.gap_keep_ovelap)
        timeseries_gapped = ['all_time', 'all_flux', 'all_centroids']
        if config.gap_imputed:
            data['gap_time'] = []
        # set to NaN gapped cadences in the time series
        for arr_i in range(len(gapped_idxs)):
            for timeseries in timeseries_gapped:
                if 'centroids' in timeseries:
                    data[timeseries]['x'][arr_i][gapped_idxs[arr_i]] = np.nan
                    data[timeseries]['y'][arr_i][gapped_idxs[arr_i]] = np.nan
                else:
                    data[timeseries][arr_i][gapped_idxs[arr_i]] = np.nan
            if config.gap_imputed and len(gapped_idxs[arr_i]) > 0:
                data['gap_time'].append(data['all_time'][arr_i][gapped_idxs[arr_i]])

    data['all_time_centroids'] = [np.array(time) for time in data['all_time']]

    # remove timestamps with NaN or infinite time, or time-series value in each quarter
    # at least some of these NaNs come from gapping the time series
    # other NaNs can come from missing time values from the FITS files
    # flux time-series
    joint_timeseries = ['all_time', 'all_flux']
    for arr_i in range(len(data[joint_timeseries[0]])):
        finite_idxs = []
        for timeseries in joint_timeseries:
            finite_idxs.append(np.isfinite(data[timeseries][arr_i]))
        finite_idxs = np.logical_and.reduce(finite_idxs)
        for timeseries in joint_timeseries:
            data[timeseries][arr_i] = data[timeseries][arr_i][finite_idxs]

    # centroid time-series
    joint_timeseries = ['all_time_centroids', 'all_centroids']
    for arr_i in range(len(data[joint_timeseries[0]])):
        finite_idxs = []
        for timeseries in joint_timeseries:
            if 'time' not in timeseries:
                finite_idxs.append(np.isfinite(data[timeseries]['x'][arr_i]))
                finite_idxs.append(np.isfinite(data[timeseries]['y'][arr_i]))
            else:
                finite_idxs.append(np.isfinite(data[timeseries][arr_i]))
        finite_idxs = np.logical_and.reduce(finite_idxs)
        for timeseries in joint_timeseries:
            if 'time' not in timeseries:
                data[timeseries]['x'][arr_i] = data[timeseries]['x'][arr_i][finite_idxs]
                data[timeseries]['y'][arr_i] = data[timeseries]['y'][arr_i][finite_idxs]
            else:
                data[timeseries][arr_i] = data[timeseries][arr_i][finite_idxs]

    # weak secondary flux time-series
    joint_timeseries = ['all_time_noprimary', 'all_flux_noprimary']
    for arr_i in range(len(data[joint_timeseries[0]])):
        finite_idxs = []
        for timeseries in joint_timeseries:
            finite_idxs.append(np.isfinite(data[timeseries][arr_i]))
        finite_idxs = np.logical_and.reduce(finite_idxs)
        for timeseries in joint_timeseries:
            data[timeseries][arr_i] = data[timeseries][arr_i][finite_idxs]

    # same for FDL centroid time-series and non-gapped time array
    joint_timeseries = ['all_time_nongapped', 'all_centroids_px']
    for arr_i in range(len(data[joint_timeseries[0]])):
        finite_idxs = []
        for timeseries in joint_timeseries:
            if 'time' not in timeseries:
                finite_idxs.append(np.isfinite(data[timeseries]['x'][arr_i]))
                finite_idxs.append(np.isfinite(data[timeseries]['y'][arr_i]))
            else:
                finite_idxs.append(np.isfinite(data[timeseries][arr_i]))
        finite_idxs = np.logical_and.reduce(finite_idxs)
        for timeseries in joint_timeseries:
            if 'time' not in timeseries:
                data[timeseries]['x'][arr_i] = data[timeseries]['x'][arr_i][finite_idxs]
                data[timeseries]['y'][arr_i] = data[timeseries]['y'][arr_i][finite_idxs]
            else:
                data[timeseries][arr_i] = data[timeseries][arr_i][finite_idxs]

    # if plot_preprocessing_tce:
    #     utils_visualization.plot_centroids(data['all_time_centroids'],
    #                                        data['all_centroids'],
    #                                        None,
    #                                        tce,
    #                                        config,
    #                                        os.path.join(config.output_dir, 'plots'),
    #                                        '2_rawwithoutnans_aug{}'.format(tce['augmentation_idx']),
    #                                        add_info=add_info)

    # preprocess the flux and centroid time series
    data_processed = process_light_curve(data, config, tce, plot_preprocessing_tce)

    # generate TCE example based on the preprocessed data
    return generate_example_for_tce(data_processed, tce, config, plot_preprocessing_tce)


def flux_preprocessing(all_time, all_flux, gap_time, tce, config, plot_preprocessing_tce):
    """ Preprocess the flux time series.

    :param all_time: list of NumPy arrays, timestamps
    :param all_flux: list of NumPy arrays, flux time series
    :param gap_time: list of NumPy arrays, gapped timestamps
    :param tce: pandas Series, TCE parameters
    :param config: Config object, preprocessing parameters
    :param plot_preprocessing_tce: bool, set to True to plot figures related to different preprocessing steps
    :return:
        time: list of NumPy arrays, timestamps for preprocessed flux time seriesed
        flux: list of NumPy arrays, preprocessed flux time series
    """

    # split on gaps
    # flux time-series
    all_time, all_flux, _ = util.split(all_time, all_flux, gap_width=config.gapWidth)

    # add gap after and before transit based on transit duration
    # duration_gapped = max(min(tce['tce_duration'] * (1 + 2 * 1), tce['tce_period'] / 10), tce['tce_duration'])
    duration_gapped = min(3 * tce['tce_duration'], np.abs(tce['tce_maxmesd']))

    # get epoch of first transit for each time array
    first_transit_time_all = [find_first_epoch_after_this_time(tce['tce_time0bk'], tce['tce_period'], time[0])
                              for time in all_time]

    # create binary time series for each time array in which in-transit points are labeled as 1's, otherwise as 0's
    binary_time_all = [create_binary_time_series(time, first_transit_time, duration_gapped, tce['tce_period'])
                       for first_transit_time, time in zip(first_transit_time_all, all_time)]

    if plot_preprocessing_tce:
        utils_visualization.plot_binseries_flux(all_time, all_flux, binary_time_all, tce, config,
                                                os.path.join(config.output_dir, 'plots'), '2_binarytimeseriesandflux')

    # linearly interpolate across TCE transits
    all_flux_lininterp = lininterp_transits(all_flux, binary_time_all, centroid=False)

    # fit a spline to the flux time-series
    spline_flux = kepler_spline.fit_kepler_spline(all_time, all_flux_lininterp, verbose=False)[0]

    if plot_preprocessing_tce:
        utils_visualization.plot_flux_fit_spline(all_time,
                                                 all_flux,
                                                 spline_flux,
                                                 tce,
                                                 config,
                                                 os.path.join(config.output_dir, 'plots'),
                                                 '3_smoothingandnormalizationflux_aug{}'.format(tce['augmentation_idx']),
                                                 flux_interp=all_flux_lininterp)

    # get indices for which the spline has finite values
    finite_i = [np.isfinite(spline_flux[i]) for i in range(len(spline_flux))]
    # normalize flux time-series by the fitted spline
    all_flux = [all_flux[i][finite_i[i]] / spline_flux[i][finite_i[i]] for i in range(len(spline_flux))
                if len(finite_i[i]) > 0]

    all_time = [all_time[i][finite_i[i]] for i in range(len(all_time)) if len(finite_i[i]) > 0]

    # impute the time series with Gaussian noise based on global estimates of median and std
    if config.gap_imputed:
        all_time, all_flux = imputing_gaps(all_time, all_flux, gap_time)

    time = np.concatenate(all_time)
    flux = np.concatenate(all_flux)

    return time, flux


def weak_secondary_flux_preprocessing(all_time, all_flux_noprimary, gap_time, tce, config, plot_preprocessing_tce):
    """ Preprocess the weak secondary flux time series.

    :param all_time: list of NumPy arrays, timestamps
    :param all_flux_noprimary: list of NumPy arrays, weak secondary flux time series
    :param gap_time: list of NumPy arrays, gapped timestamps
    :param tce: pandas Series, TCE parameters
    :param config: Config object, preprocessing parameters
    :param plot_preprocessing_tce: bool, set to True to plot figures related to different preprocessing steps
    :return:
        time: list of NumPy arrays, timestamps for preprocessed flux time seriesed
        flux_noprimary: list of NumPy arrays, preprocessed weak secondary flux time series
    """

    all_time, all_flux_noprimary, _ = util.split(all_time, all_flux_noprimary, gap_width=config.gapWidth)

    # add gap after and before transit based on transit duration
    # duration_gapped = max(min(tce['tce_duration'] * (1 + 2 * 1), tce['tce_period'] / 10), tce['tce_duration'])
    duration_gapped = min(3 * tce['tce_duration'], np.abs(tce['tce_maxmesd']))

    first_transit_time_all_noprimary = [find_first_epoch_after_this_time(tce['tce_time0bk'] + tce['tce_maxmesd'],
                                                                         tce['tce_period'], time[0])
                                        for time in all_time]
    binary_time_all_noprimary = [create_binary_time_series(time, first_transit_time, duration_gapped, tce['tce_period'])
                                 for first_transit_time, time in
                                 zip(first_transit_time_all_noprimary, all_time)]

    if plot_preprocessing_tce:
        utils_visualization.plot_binseries_flux(all_time, all_flux_noprimary, binary_time_all_noprimary,
                                                tce, config,
                                                os.path.join(config.output_dir, 'plots'),
                                                '2_binarytimeseries_wksflux_aug{}'.format(tce['augmentation_idx']))

    # spline fitting for the secondary flux time-series
    all_flux_noprimary_lininterp = lininterp_transits(all_flux_noprimary, binary_time_all_noprimary, centroid=False)

    spline_flux_noprimary = kepler_spline.fit_kepler_spline(all_time, all_flux_noprimary_lininterp, verbose=False)[0]

    if plot_preprocessing_tce:
        utils_visualization.plot_flux_fit_spline(all_time,
                                                 all_flux_noprimary,
                                                 spline_flux_noprimary,
                                                 tce,
                                                 config,
                                                 os.path.join(config.output_dir, 'plots'),
                                                 '3_smoothingandnormalization_wksflux_aug{}'.format(tce['augmentation_idx']),
                                                 flux_interp= all_flux_noprimary_lininterp)

    finite_i = [np.isfinite(spline_flux_noprimary[i]) for i in range(len(spline_flux_noprimary))]

    all_time = [all_time[i][finite_i[i]] for i in range(len(all_time)) if len(finite_i[i]) > 0]

    all_flux_noprimary = [all_flux_noprimary[i][finite_i[i]] /
                          spline_flux_noprimary[i][finite_i[i]]
                          for i in range(len(spline_flux_noprimary)) if len(finite_i[i]) > 0]

    # impute the time series with Gaussian noise based on global estimates of median and std
    if config.gap_imputed:
        all_time, all_flux_noprimary = imputing_gaps(all_time, all_flux_noprimary, gap_time)

    time = np.concatenate(all_time)
    flux_noprimary = np.concatenate(all_flux_noprimary)

    return time, flux_noprimary


def centroid_preprocessing(all_time, all_centroids, avg_centroid_oot, target_position, add_info, gap_time, tce, config,
                           plot_preprocessing_tce):
    """ Preprocess the centroid time series.

    :param all_time: list of NumPy arrays, timestamps
    :param all_centroids: dictionary for the two centroid coordinates coded as 'x' and 'y'. Each key maps to a list of
    NumPy arrays for the respective centroid coordinate time series
    :param avg_centroid_oot: dictionary for the two centroid coordinates coded as 'x' and 'y'. Each key maps to the
    estimate of the average out-of-transit centroid
    :param target_position: list, target star position in 'x' and 'y'
    :param add_info: dictionary, additional information such as quarters and modules
    :param gap_time: list of NumPy arrays, gapped timestamps
    :param tce: pandas Series, TCE parameters
    :param config: Config object, preprocessing parameters
    :param plot_preprocessing_tce: bool, set to True to plot figures related to different preprocessing steps
    :return:
        time: NumPy array, timestamps for preprocessed centroid time series
        centroid_dist: NumPy array, preprocessed centroid time series which is an estimate of the distance of the
        transit to the target
    """

    all_time, all_centroids, add_info = util.split(all_time, all_centroids, add_info=add_info, centroid=True,
                                                   gap_width=config.gapWidth)

    # pixel coordinate transformation for targets on module 13 for Kepler
    if config.px_coordinates and config.satellite == 'kepler':
        if add_info['module'][0] == 13:
            all_centroids = kepler_transform_pxcoordinates_mod13(all_centroids, add_info)

            if plot_preprocessing_tce:
                utils_visualization.plot_centroids(all_time,
                                                   all_centroids,
                                                   None,
                                                   tce,
                                                   config,
                                                   os.path.join(config.output_dir, 'plots'),
                                                   '2_rawaftertransformation_aug{}'.format(tce['augmentation_idx']),
                                                   add_info=add_info)

    # add gap after and before transit based on transit duration
    # duration_gapped = max(min(tce['tce_duration'] * (1 + 2 * 1), tce['tce_period'] / 10), tce['tce_duration'])
    duration_gapped = min(3 * tce['tce_duration'], np.abs(tce['tce_maxmesd']))

    # get epoch of first transit for each time array
    first_transit_time_all = [find_first_epoch_after_this_time(tce['tce_time0bk'], tce['tce_period'], time[0])
                              for time in all_time]

    # create binary time series for each time array in which in-transit points are labeled as 1's, otherwise as 0's
    binary_time_all = [create_binary_time_series(time, first_transit_time, duration_gapped, tce['tce_period'])
                       for first_transit_time, time in zip(first_transit_time_all, all_time)]

    # spline fitting and normalization - fit a piecewise-cubic spline with default arguments
    # FIXME: wouldn't it be better to fit a spline to oot values or a Savitzky Golay filter
    #  (as Jeff and Doug mentioned)?
    # FIXME: fit spline only to the oot values - linearly interpolate the transits; same thing for flux
    all_centroids_lininterp = lininterp_transits(all_centroids, binary_time_all, centroid=True)
    spline_centroid = {coord: kepler_spline.fit_kepler_spline(all_time, all_centroids_lininterp[coord],
                                                              verbose=False)[0] for coord in all_centroids_lininterp}

    if plot_preprocessing_tce:
        utils_visualization.plot_centroids(all_time,
                                           all_centroids,
                                           spline_centroid,
                                           tce,
                                           config,
                                           os.path.join(config.output_dir, 'plots'),
                                           '3_smoothingandnormalizationcentroid_aug{}'.format(tce['augmentation_idx']))

    # In rare cases the piecewise spline contains NaNs in places the spline could not be fit. We can't normalize those
    # points if the spline isn't defined there. Instead we just remove them.
    finite_i = [np.logical_and(np.isfinite(spline_centroid['x'][i]), np.isfinite(spline_centroid['y'][i]))
                for i in range(len(spline_centroid['x']))]

    # # get average oot per quarter
    # oot_idxs = [np.where(~binary_time) for binary_time in binary_time_all]  # compute it using only finite and oot values
    # oot_idxs = [np.union1d(oot_idxs[i], finite_i[i]) for i in range(len(oot_idxs))]
    # avg_centroid_oot = {coord: [np.median(all_centroids[coord][i][oot_idxs[i]])
    #                             for i in range(len(all_centroids[coord]))] for coord in all_centroids}

    # normalize by the spline
    all_centroids = {coord: [all_centroids[coord][i][finite_i[i]] / spline_centroid[coord][i][finite_i[i]] *
                             avg_centroid_oot[coord] for i in range(len(spline_centroid[coord]))
                             if len(finite_i[i]) > 0]
                     for coord in all_centroids}
    # normalize by the fitted splines and recover the range by multiplying by the average oot for each quarter
    # all_centroids = {coord: [all_centroids[coord][i][finite_i[i]] / spline_centroid[coord][i][finite_i[i]] *
    #                          avg_centroid_oot[coord][i] for i in range(len(spline_centroid[coord]))]
    #                  for coord in all_centroids}
    # all_centroids = {coord: [all_centroids[coord][i][finite_i[i]] - spline_centroid[coord][i][finite_i[i]]
    #                          for i in range(len(spline_centroid[coord]))]
    #                  for coord in all_centroids}

    binary_time_all = [binary_time_all[i][finite_i[i]] for i in range(len(binary_time_all)) if len(finite_i[i]) > 0]
    all_time = [all_time[i][finite_i[i]] for i in range(len(all_time)) if len(finite_i[i]) > 0]

    # # set outliers to zero using Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    # q25_75 = {'x': {'q25': np.percentile(np.concatenate(all_centroids['x']), 25),
    #                 'q75': np.percentile(np.concatenate(all_centroids['x']), 75)},
    #           'y': {'q25': np.percentile(np.concatenate(all_centroids['y']), 25),
    #                 'q75': np.percentile(np.concatenate(all_centroids['y']), 75)}
    #           }
    # iqr = {'x': q25_75['x']['q75'] - q25_75['x']['q25'],
    #        'y': q25_75['y']['q75'] - q25_75['y']['q25']}
    # outlier_thr = 1.5
    # for coord in all_centroids:
    #     for i in range(len(all_centroids[coord])):
    #         all_centroids[coord][i][np.where(all_centroids[coord][i] > q25_75[coord]['q75'] + outlier_thr * iqr[coord])] = avg_centroid_oot[coord]
    #         all_centroids[coord][i][np.where(all_centroids[coord][i] < q25_75[coord]['q25'] - outlier_thr * iqr[coord])] = avg_centroid_oot[coord]

    # compute the new average oot after the spline fitting and normalization
    # TODO: how to compute the average oot? mean, median, other...
    centroid_oot = {coord: [centroids[np.where(binary_time == 0)] for binary_time, centroids in
                            zip(binary_time_all, all_centroids[coord])] for coord in all_centroids}
    avg_centroid_oot_plt = {coord: np.median(np.concatenate(centroid_oot[coord])) for coord in all_centroids}

    if plot_preprocessing_tce:
        utils_visualization.plot_centroids_it_oot(all_time,
                                                  binary_time_all,
                                                  all_centroids,
                                                  centroid_oot,
                                                  avg_centroid_oot_plt,
                                                  target_position,
                                                  tce, config,
                                                  os.path.join(config.output_dir, 'plots'),
                                                  '4_centroidtimeseries_it-ot-target_aug{}'.format(tce['augmentation_idx']))

    # compute the corrected centroid time-series normalized by the transit depth fraction and centered on the avg oot
    # centroid position
    transit_depth = tce['transit_depth'] + 1
    transitdepth_term = (1e6 - transit_depth) / transit_depth
    # avg_centroid_oot = {coord: avg_centroid_oot[coord] * 1.15 for coord in avg_centroid_oot}
    # all_centroids_corr = {coord: [-((all_centroids[coord][i] - avg_centroid_oot[coord]) * transitdepth_term) /
    #                               (1, np.cos(all_centroids['y'][i] * np.pi / 180))[coord == 'x'] +
    #                               avg_centroid_oot[coord] for i in range(len(all_centroids[coord]))]
    #                       for coord in all_centroids}
    all_centroids_corr = {coord: [-((all_centroids[coord][i] - avg_centroid_oot[coord]) * transitdepth_term) /
                                  (1, np.cos(target_position[1] * np.pi / 180))[coord == 'x'] +
                                  avg_centroid_oot[coord] for i in range(len(all_centroids[coord]))]
                          for coord in all_centroids}
    # correction performed using quarter average oot estimates
    # all_centroids_corr = {coord: [-((all_centroids[coord][i] - avg_centroid_oot[coord][i]) * transitdepth_term) /
    #                               (1, np.cos(all_centroids['y'][i] * np.pi / 180))[coord == 'x'] +
    #                               avg_centroid_oot[coord][i] for i in range(len(all_centroids[coord]))]
    #                       for coord in all_centroids}
    # all_centroids_corr = {coord: [-(all_centroids[coord][i] * transitdepth_term) /
    #                               (1, np.cos((all_centroids['y'][i]) * np.pi / 180))[coord == 'x'] +
    #                               avg_centroid_oot[coord] for i in range(len(all_centroids[coord]))]
    #                       for coord in all_centroids}

    if plot_preprocessing_tce:
        utils_visualization.plot_corrected_centroids(all_time,
                                                     all_centroids_corr,
                                                     avg_centroid_oot,
                                                     target_position,
                                                     tce,
                                                     config,
                                                     os.path.join(config.output_dir, 'plots'),
                                                     '5_correctedcentroidtimeseries_aug{}'.format(tce['augmentation_idx']))

    if config.px_coordinates:
        # compute the euclidean distance of the corrected centroid time series to the target star position
        all_centroid_dist = [np.sqrt(np.square(all_centroids['x'][i] - target_position[0]) +
                                     np.square(all_centroids['y'][i] - target_position[1]))
                             for i in range(len(all_centroids['x']))]
    else:
        # compute the angular distance of the corrected centroid time series to the target star position
        all_centroid_dist = [np.sqrt(np.square((all_centroids_corr['x'][i] - target_position[0]) *
                                               np.cos(all_centroids['y'][i] * np.pi / 180)) +
                                     np.square(all_centroids_corr['y'][i] - target_position[1]))
                             for i in range(len(all_centroids_corr['x']))]

    # # get the across quarter average oot estimate using only finite and oot values
    # avg_centroid_oot_dist_global = np.median(np.concatenate([all_centroid_dist[i][oot_idxs[i]]
    #                                                          for i in range(len(all_centroid_dist))]))
    # spline_centroid_dist = kepler_spline.fit_kepler_spline(all_time, all_centroid_dist, verbose=False)[0]
    # # center the offset centroid distance to the across quarter average oot estimate
    # all_centroid_dist = [all_centroid_dist[i] / np.median(all_centroid_dist[i][oot_idxs[i]]) *
    #                      avg_centroid_oot_dist_global
    #                      for i in range(len(all_centroid_dist))]

    # convert from degree to arcsec
    if not config.px_coordinates:
        all_centroid_dist = [centroid_dist_arr * 3600 for centroid_dist_arr in all_centroid_dist]

    if plot_preprocessing_tce:
        utils_visualization.plot_dist_centroids(all_time,
                                                all_centroid_dist,
                                                None,
                                                None,
                                                tce,
                                                config,
                                                os.path.join(config.output_dir, 'plots'),
                                                '6_distcentr_aug{}'.format(tce['augmentation_idx']))

    # impute the time series with Gaussian noise based on global estimates of median and std
    if config.gap_imputed:
        time, all_centroid_dist = imputing_gaps(all_time, all_centroid_dist, gap_time)

    time = np.concatenate(all_time)
    centroid_dist = np.concatenate(all_centroid_dist)

    return time, centroid_dist  # , add_info


def centroidFDL_preprocessing(all_time, all_centroids, add_info, gap_time, tce, config, plot_preprocessing_tce):
    """ Preprocess the centroid time series following FDL preprocessing pipeline [1].

     :param all_time: list of NumPy arrays, timestamps
     :param all_centroids: dictionary for the two centroid coordinates coded as 'x' and 'y'. Each key maps to a list of
     NumPy arrays for the respective centroid coordinate time series
     :param add_info: dictionary, additional information such as quarters and modules
     :param gap_time: list of NumPy arrays, gapped timestamps
     :param tce: pandas Series, TCE parameters
     :param config: Config object, preprocessing parameters
     :param plot_preprocessing_tce: bool, set to True to plot figures related to different preprocessing steps
     :return:
         time: NumPy array, timestamps for preprocessed centroid time series
         centroid_dist: NumPy array, preprocessed centroid time series according to FDL

     [1] Ansdell, Megan, et al. "Scientific Domain Knowledge Improves Exoplanet Transit Classification with Deep
     Learning." The Astrophysical Journal Letters 869.1 (2018): L7.
     """

    all_time, all_centroids, add_info = util.split(all_time, all_centroids, gap_width=config.gapWidth, centroid=True,
                                                   add_info=add_info)

    if add_info['quarter'][0] == 13:
        all_centroids = kepler_transform_pxcoordinates_mod13(all_centroids, add_info)

    first_transit_time_all = [find_first_epoch_after_this_time(tce['tce_time0bk'], tce['tce_period'], time[0])
                              for time in all_time]

    # duration_gapped = max(min(tce['tce_duration'] * (1 + 2 * 1), tce['tce_period'] / 10), tce['tce_duration'])
    duration_gapped = min(3 * tce['tce_duration'], np.abs(tce['tce_maxmesd']))
    # duration_gapped = 3 * tce['tce_duration']

    binary_time_all = [create_binary_time_series(time, first_transit_time, duration_gapped, tce['tce_period'])
                       for first_transit_time, time in zip(first_transit_time_all, all_time)]

    all_centroids_lininterp = lininterp_transits(all_centroids, binary_time_all, centroid=True)

    spline_centroid = {coord: kepler_spline.fit_kepler_spline(all_time,
                                                              all_centroids_lininterp[coord],
                                                              verbose=False)[0] for coord in all_centroids_lininterp}

    finite_i_centroid = [np.logical_and(np.isfinite(spline_centroid['x'][i]), np.isfinite(spline_centroid['y'][i]))
                         for i in range(len(spline_centroid['x']))]

    # all_centroids = {coord: [all_centroids[coord][i][finite_i_centroid[i]] /
    #                          spline_centroid[coord][i][finite_i_centroid[i]]
    #                          for i in range(len(spline_centroid[coord])) if len(finite_i_centroid[i] > 0)]
    #                  for coord in all_centroids}
    all_centroids = {coord: [all_centroids[coord][i][finite_i_centroid[i]] -
                             spline_centroid[coord][i][finite_i_centroid[i]]
                             for i in range(len(spline_centroid[coord])) if len(finite_i_centroid[i] > 0)]
                     for coord in all_centroids}

    all_time = [all_time[i][finite_i_centroid[i]] for i in range(len(all_time)) if len(finite_i_centroid[i]) > 0]

    all_centroid_dist = [np.sqrt(np.square(all_centroids['x'][i]) + np.square(all_centroids['y'][i]))
                         for i in range(len(all_centroids['x']))]

    if plot_preprocessing_tce:
        utils_visualization.plot_dist_centroids(all_time,
                                                all_centroid_dist,
                                                None,
                                                None,
                                                tce,
                                                config,
                                                os.path.join(config.output_dir, 'plots'),
                                                '6_distcentrfdl_aug{}'.format(tce['augmentation_idx']),
                                                pxcoordinates=True)

    # impute the time series with Gaussian noise based on global estimates of median and std
    if config.gap_imputed:
        all_time, all_centroid_dist = imputing_gaps(all_time, all_centroid_dist, gap_time)

    time = np.concatenate(all_time)
    centroid_dist = np.concatenate(all_centroid_dist)

    return time, centroid_dist


def process_light_curve(data, config, tce, plot_preprocessing_tce=False):
    """ Removes low-frequency variability from the flux and centroid time-series.

    Args:
      data: dictionary containing data related to the time-series to be preprocessed before generating views
      config: Config object, holds preprocessing parameters
      tce: pandas Series, row of the TCE table for the TCE that is being processed; it contains data such as the
      ephemeris
      plot_preprocessing_tce: bool, if True plots figures for several steps in the preprocessing pipeline

    Returns:
      data_views: dictionary containing data used to generate the views
    """

    if config.satellite == 'kepler':
        add_info_centr = {'quarter': data['quarter'], 'module': data['module']}
    else:
        add_info_centr = None

    # preprocess flux time series
    time, flux = flux_preprocessing(data['all_time'], data['all_flux'], data['gap_time'], tce, config,
                                    plot_preprocessing_tce)

    # preprocess weak secondary flux time series
    time_wksecondaryflux, wksecondaryflux = weak_secondary_flux_preprocessing(data['all_time_noprimary'],
                                                                              data['all_flux_noprimary'],
                                                                              data['gap_time_noprimary'],
                                                                              tce, config, plot_preprocessing_tce)

    # preprocess centroid time series
    time_centroid, centroid_dist = centroid_preprocessing(data['all_time_centroids'], data['all_centroids'],
                                                          data['avg_centroid_oot'], data['target_position'],
                                                          add_info_centr, data['gap_time'], tce,
                                                          config, plot_preprocessing_tce)

    # preprocess FDL centroid time series
    time_centroidFDL, centroid_distFDL = centroidFDL_preprocessing(data['all_time_nongapped'], data['all_centroids_px'],
                                                                   add_info_centr, [], tce, config,
                                                                   plot_preprocessing_tce)

    # dictionary with preprocessed time series used to generate views
    data_views = {}
    data_views['time'] = time
    data_views['flux'] = flux
    data_views['time_centroid_dist'] = time_centroid
    data_views['centroid_dist'] = centroid_dist
    data_views['time_wksecondaryflux'] = time_wksecondaryflux
    data_views['wksecondaryflux'] = wksecondaryflux
    data_views['time_centroid_distFDL'] = time_centroidFDL
    data_views['centroid_distFDL'] = centroid_distFDL

    return data_views


def min_max_normalization(arr, max_val, min_val):
    """ Min-max normalization.

    :param arr: array
    :param max_val: float, max val
    :param min_val: float, min val
    :return:
        normalized array
    """

    return (arr - min_val) / (max_val - min_val)


def phase_fold_time_aug(time, period, t0):
    """ Creates a phase-folded time vector. Performs data augmentation by sampling transits with replacement for a
    total number equal to the original.

    result[i] is the unique number in [-period / 2, period / 2)
    such that result[i] = time[i] - t0 + k_i * period, for some integer k_i.

    Args:
    time: 1D numpy array of time values.
    period: A positive real scalar; the period to fold over.
    t0: The center of the resulting folded vector; this value is mapped to 0.

    Returns:
    result_sampled: 1D numpy array with the sampled phase-folded time array.
    sampled_idxs: 1D nunpy array with the indices of the sampled transits.
    num_transits: int, number of transits in the time series.
    """

    half_period = period / 2
    result = np.mod(time + (half_period - t0), period)
    result -= half_period

    # get the first index after -pi/2
    edge_idxs = np.where(np.diff(result) < 0)[0] + 1
    # add the first and last indices in the array
    edge_idxs = np.concatenate([[0], edge_idxs, [len(result)]])

    # define the boundary indices for each transit
    start_end_idxs = list(zip(edge_idxs[:-1], edge_idxs[1:]))

    # number of transits in the time series
    num_transits = len(start_end_idxs)

    # sample with replacement transits
    transit_idxs = np.random.randint(0, num_transits, size=num_transits)
    sampled_idxs = np.concatenate([np.arange(start_end_idxs[transit_idx][0], start_end_idxs[transit_idx][1])
                                   for transit_idx in transit_idxs])

    result_sampled = result[sampled_idxs]

    return result_sampled, sampled_idxs, num_transits


def phase_fold_and_sort_light_curve(time, timeseries, period, t0, augmentation=False):
    """ Phase folds a light curve and sorts by ascending time.

    Args:
      time: 1D NumPy array of time values.
      timeseries: 1D NumPy array of time series values.
      period: A positive real scalar; the period to fold over.
      t0: The center of the resulting folded vector; this value is mapped to 0.

    Returns:
      time: 1D NumPy array of phase folded time values in
          [-period / 2, period / 2), where 0 corresponds to t0 in the original
          time array. Values are sorted in ascending order.
      timeseries: 1D NumPy array. Values are the same as the original input
          array, but sorted by folded_time.
      num_transits: int, number of transits in the time series.
    """

    # Phase fold time.
    if not augmentation:
        time = util.phase_fold_time(time, period, t0)
    else:
        time, sampled_idxs = phase_fold_time_aug(time, period, t0)
        timeseries = timeseries[sampled_idxs]

    # count number of transits in the phase domain
    num_transits = np.sum(np.diff(time) < 0) + 1

    # Sort by ascending time.
    sorted_i = np.argsort(time)
    time = time[sorted_i]
    timeseries = timeseries[sorted_i]

    return time, timeseries, num_transits


def phase_fold_and_sort_light_curve_odd_even(time, timeseries, period, t0, augmentation=False):
    """ Creates separate phase-folded time vectors for odd and even periods.

    :param time: 1D NumPy array of time values
    :param timeseries: 1D NumPy array of time series values
    :param period: A positive real scalar; the period to fold over.
    :param t0: The center of the resulting folded vector; this value is mapped to 0.
    :return:
        Two tuples of size three, each of which contains odd or even time values, flux values, and centroid values.
        time: 1D NumPy array of phase folded time values in [-period / 2, period / 2), where 0 corresponds to t0
            in the original time array. Values are sorted in ascending order.
        folded_timeseries: 1D NumPy array. Values are the same as the original input array, but sorted by folded_time.
        num_transits_even: int, number of even transits in the time series.
        num_transits_odd: int, number of odd transits in the time series.
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
    odd_timeseries = np.take(timeseries, odd_indices)
    even_timeseries = np.take(timeseries, even_indices)
    # odd_centroids = np.take(centroids, odd_indices)
    # even_centroids = np.take(centroids, even_indices)

    # Phase fold time.
    if not augmentation:
        odd_time = util.phase_fold_time(odd_time, period, t0)
        even_time = util.phase_fold_time(even_time, period, t0)
    else:
        odd_time, odd_sampled_idxs = phase_fold_time_aug(odd_time, period, t0)
        even_time, even_sampled_idxs = phase_fold_time_aug(even_time, period, t0)
        odd_timeseries = odd_timeseries[odd_sampled_idxs]
        even_timeseries = even_timeseries[even_sampled_idxs]
        # odd_flux, odd_centroids = odd_flux[odd_sampled_idxs], odd_centroids[odd_sampled_idxs]
        # even_flux, even_centroids = even_flux[even_sampled_idxs], even_centroids[even_sampled_idxs]

    # count number of transits in the phase domain
    num_transits_odd = np.sum(np.diff(odd_time) < 0) + 1
    num_transits_even = np.sum(np.diff(even_time) < 0) + 1

    # Sort by ascending time.
    sorted_i_odd = np.argsort(odd_time)
    sorted_i_even = np.argsort(even_time)

    odd_result = odd_time[sorted_i_odd]
    even_result = even_time[sorted_i_even]
    odd_timeseries = odd_timeseries[sorted_i_odd]
    even_timeseries = even_timeseries[sorted_i_even]
    # odd_centroids = odd_centroids[sorted_i_odd]
    # even_centroids = even_centroids[sorted_i_even]

    # assert(len(odd_time) + len(even_time) - len(time) == 0 and len(set(odd_time) & set(even_time)) == 0)

    return (odd_result, odd_timeseries, num_transits_odd), (even_result, even_timeseries, num_transits_even)


def generate_view(time, flux, num_bins, bin_width, t_min, t_max,
                  centering=True, normalize=True, centroid=False, **kwargs):
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

    Returns:
      1D NumPy array of size num_bins containing the median flux values of
      uniformly spaced bins on the phase-folded time axis.
    """

    # binning using median
    view = median_filter.median_filter(time, flux, num_bins, bin_width, t_min, t_max)

    # TODO: what happens with empty (NaN) bins?
    # empty bins fall back to the global median
    # view[np.where(np.isnan(view)] = np.median(flux)

    # global median centering
    if centering:
        view -= np.median(view)

    # normalization
    if normalize:
        view = normalize_view(view, val=None, centroid=centroid, **kwargs)

    return view


def normalize_view(view, val=None, centroid=False, **kwargs):
    """ Normalize the phase-folded time series.

    :param view: array, phase-folded time series
    :param val: float, value used to normalize the time series
    :param centroid: bool, True for centroid time series
    :param kwargs: dict, extra keyword parameters
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

    if val == 0:
        print('Dividing view by 0. Returning the non-normalized view.')
        report_exclusion(kwargs['report']['config'], kwargs['report']['tce'],
                         'Dividing view by 0. Returning the non-normalized view {}.'.format(kwargs['report']['view']))

        return view

    return view / val


def centering_and_normalization(view, val_centr, val_norm, **kwargs):
    """ Center and normalize a 1D time series.

    :param view: array, 1D time series
    :param val_centr: float, value used to center the time series
    :param val_norm: float, value used normalize the time series
    :param kwargs: dict, extra keyword parameters
    :return:
        array, centered and normalized time series
    """

    if val_norm == 0:
        print('Dividing view by 0. Returning the non-normalized view.')
        report_exclusion(kwargs['report']['config'], kwargs['report']['tce'],
                         'Dividing view by 0. Returning the non-normalized view {}.'.format(kwargs['report']['view']))

        return view - val_centr

    return (view - val_centr) / val_norm


def global_view(time, flux, period, num_bins=2001, bin_width_factor=1/2001, centroid=False, normalize=True,
                centering=True, **kwargs):
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


    Returns:
      1D NumPy array of size num_bins containing the median flux values of
      uniformly spaced bins on the phase-folded time axis.
    """
    return generate_view(
        time,
        flux,
        num_bins=num_bins,
        bin_width=max(period * bin_width_factor, kwargs['tce_duration'] * 0.16),
        t_min=-period / 2,
        t_max=period / 2,
        centroid=centroid,
        normalize=normalize,
        centering=centering,
        **kwargs)


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
               **kwargs):
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
        **kwargs)


def get_out_of_transit_idxs_loc(num_bins_loc, num_transit_durations):
    """ Get indices of out-of-transit cadences for the local views.

    :param num_bins_loc: int, number of bins for the local views
    :param num_transit_durations: int, number of transit durations in the local views
    :return:
        idxs_nontransitcadences_loc: list with out-of-transit indices for the local views
    """

    # get out-of-transit indices for local views
    transit_duration_bins_loc = num_bins_loc / num_transit_durations  # number of bins per transit duration
    nontransitcadences_loc = np.array([True] * num_bins_loc)
    # set to False the indices outside of the transit which is in the center of the view
    nontransitcadences_loc[np.arange(int(np.floor((num_bins_loc - transit_duration_bins_loc) / 2)),
                                     int(np.ceil((num_bins_loc + transit_duration_bins_loc) / 2)))] = False
    idxs_nontransitcadences_loc = np.where(nontransitcadences_loc)

    return idxs_nontransitcadences_loc


def get_out_of_transit_idxs_glob(num_bins_glob, transit_duration, orbital_period):
    """ Get indices of out-of-transit cadences for the global views.

    :param num_bins_glob: int, number of bins for the global views
    :param transit_duration: float, transit duration
    :param orbital_period: float, orbital period
    :return:
        idxs_nontransitcadences_glob: list with out-of-transit indices for the global views
    """

    # get out-of-transit indices for global views
    frac_durper = transit_duration / orbital_period  # ratio transit duration to orbital period
    nontransitcadences_glob = np.array([True] * num_bins_glob)
    # set To False the indices outside of the transit which is in the center of the view
    nontransitcadences_glob[np.arange(int(num_bins_glob / 2 * (1 - frac_durper)),
                                      int(num_bins_glob / 2 * (1 + frac_durper)))] = False
    idxs_nontransitcadences_glob = np.where(nontransitcadences_glob)

    return idxs_nontransitcadences_glob


def generate_example_for_tce(data, tce, config, plot_preprocessing_tce=False):
    """ Generates a tf.train.Example representing an input TCE.

    Args:
      data: dictionary containing preprocessed time-series used to generate the views
      tce: Dict-like object containing at least 'tce_period', 'tce_duration', and
        'tce_time0bk'. Additional items are included as features in the output.
      config: Config object; preprocessing parameters.
      plot_preprocessing_tce: bool, if True plots figures for some steps while generating the inputs.

    Returns:
      A tf.train.Example containing features. These features can be time series, stellar parameters, statistical
      quantities, .... it returns None if some exception while creating these features occurs
    """

    if len(data['time']) == 0:
        print(tce)
        report_exclusion(config, tce, 'No data[time] before creating views.')
        return None

    # phase folding for odd and even time series
    (odd_time, odd_flux, num_transits_odd), \
    (even_time, even_flux, num_transits_even) = phase_fold_and_sort_light_curve_odd_even(data['time'],
                                                                                         data['flux'],
                                                                                         tce['tce_period'],
                                                                                         tce['tce_time0bk'],
                                                                                         augmentation=False)

    # phase folding for flux time series
    time, flux, num_transits_flux = phase_fold_and_sort_light_curve(data['time'],
                                                                    data['flux'],
                                                                    tce['tce_period'],
                                                                    tce['tce_time0bk'],
                                                                    augmentation=False)

    # phase folding for centroid time series
    time_centroid_dist, centroid_dist, \
    num_transits_centroid = phase_fold_and_sort_light_curve(data['time_centroid_dist'],
                                                            data['centroid_dist'],
                                                            tce['tce_period'],
                                                            tce['tce_time0bk'],
                                                            augmentation=False)

    # phase folding for the weak secondary flux time series
    time_noprimary, flux_noprimary, num_transits_wks = phase_fold_and_sort_light_curve(data['time_wksecondaryflux'],
                                                                                       data['wksecondaryflux'],
                                                                                       tce['tce_period'],
                                                                                       tce['tce_time0bk'] + tce[
                                                                                           'tce_maxmesd'],
                                                                                       augmentation=False)

    # same for FDL centroid time-series
    # phase folding for flux and centroid time series
    time_centroid_dist_fdl, centroid_dist_fdl, \
    num_transits_centrfdl = phase_fold_and_sort_light_curve(data['time_centroid_distFDL'],
                                                            data['centroid_distFDL'],
                                                            tce['tce_period'],
                                                            tce['tce_time0bk'],
                                                            augmentation=False)

    phasefolded_timeseries = {'Flux': (time, flux),
                              'Odd Flux': (odd_time, odd_flux),
                              'Even Flux': (even_time, even_flux),
                              'Weak Secondary Flux': (time_noprimary, flux_noprimary),
                              'Centroid dist': (time_centroid_dist, centroid_dist),
                              'Centroid dist FDL': (time_centroid_dist_fdl, centroid_dist_fdl)
                              }

    if plot_preprocessing_tce:
        # CHANGE NUMBER OF PLOTS AS FUNCTION OF THE TIME SERIES PREPROCESSED
        utils_visualization.plot_all_phasefoldedtimeseries(phasefolded_timeseries,
                                                           tce,
                                                           config,
                                                           (3, 2),
                                                           os.path.join(config.output_dir, 'plots'),
                                                           '7_phasefoldedbinned_timeseries_aug{}'.format(tce['augmentation_idx'])
                                                           )

    # make output proto
    ex = tf.train.Example()

    # set time series features
    try:

        # initialize number of transits per time series dictionary
        num_transits = {}

        # get flux views
        glob_flux_view = global_view(time, flux,
                                     tce['tce_period'],
                                     normalize=False,
                                     centering=False,
                                     num_bins=config.num_bins_glob,
                                     bin_width_factor=config.bin_width_factor_glob,
                                     report={'config': config, 'tce': tce, 'view': 'global_flux_view'},
                                     tce_duration=tce['tce_duration'])
        loc_flux_view = local_view(time, flux,
                                   tce['tce_period'], tce['tce_duration'],
                                   normalize=False,
                                   centering=False,
                                   num_durations=config.num_durations,
                                   num_bins=config.num_bins_loc,
                                   bin_width_factor=config.bin_width_factor_loc,
                                   report={'config': config, 'tce': tce, 'view': 'local_flux_view'})

        # get odd flux views
        glob_flux_odd_view = global_view(odd_time, odd_flux,
                                         tce['tce_period'],
                                         normalize=False,
                                         centering=False,
                                         num_bins=config.num_bins_glob,
                                         bin_width_factor=config.bin_width_factor_glob,
                                         report={'config': config, 'tce': tce, 'view': 'global_flux_odd_view'},
                                         tce_duration=tce['tce_duration'])
        loc_flux_odd_view = local_view(odd_time, odd_flux,
                                       tce['tce_period'], tce['tce_duration'],
                                       normalize=False,
                                       centering=False,
                                       num_durations=config.num_durations,
                                       num_bins=config.num_bins_loc,
                                       bin_width_factor=config.bin_width_factor_loc,
                                       report={'config': config, 'tce': tce, 'view': 'local_flux_odd_view'})

        # get even flux views
        glob_flux_even_view = global_view(even_time, even_flux,
                                          tce['tce_period'],
                                          normalize=False,
                                          centering=False,
                                          num_bins=config.num_bins_glob,
                                          bin_width_factor=config.bin_width_factor_glob,
                                          report={'config': config, 'tce': tce, 'view': 'global_flux_even_view'},
                                          tce_duration=tce['tce_duration'])
        loc_flux_even_view = local_view(even_time, even_flux,
                                        tce['tce_period'], tce['tce_duration'],
                                        normalize=False,
                                        centering=False,
                                        num_durations=config.num_durations,
                                        num_bins=config.num_bins_loc,
                                        bin_width_factor=config.bin_width_factor_loc,
                                        report={'config': config, 'tce': tce, 'view': 'local_flux_even_view'})

        # get local and global view for the weak secondary flux
        glob_weak_secondary_view = global_view(time_noprimary, flux_noprimary,
                                               tce['tce_period'],
                                               normalize=False,
                                               centering=False,
                                               num_durations=config.num_durations,
                                               num_bins=config.num_bins_glob,
                                               bin_width_factor=config.bin_width_factor_glob,
                                               report={'config': config, 'tce': tce, 'view': 'global_wks_view'},
                                               tce_duration=tce['tce_duration'])
        loc_weak_secondary_view = local_view(time_noprimary, flux_noprimary,
                                             tce['tce_period'], tce['tce_duration'],
                                             normalize=False,
                                             centering=False,
                                             num_durations=config.num_durations,
                                             num_bins=config.num_bins_loc,
                                             bin_width_factor=config.bin_width_factor_loc,
                                             report={'config': config, 'tce': tce, 'view': 'local_wks_view'})

        # if plot_preprocessing_tce:
        #     utils_visualization.plot_wks(glob_view, glob_view_weak_secondary, tce, config,
        #                                  os.path.join(config.output_dir, 'plots'), '8_wks_test')

        flux_views = {'global_flux_view': glob_flux_view,
                      'local_flux_view': loc_flux_view,
                      'global_flux_odd_view': glob_flux_odd_view,
                      'local_flux_odd_view': loc_flux_odd_view,
                      'global_flux_even_view': glob_flux_even_view,
                      'local_flux_even_view': loc_flux_even_view,
                      }
        num_transits.update({'global_flux_view': num_transits_flux,
                             'local_flux_odd_view': num_transits_odd,
                             'local_flux_even_view': num_transits_even})

        # get median and std statistics for global and local flux views
        flux_views_stats = {'median': {'global': np.median(flux_views['global_flux_view']),
                                       'local': np.median(flux_views['local_flux_view'])}}
        flux_views_stats['min'] = {'global': np.abs(np.min(flux_views['global_flux_view'] -
                                                         flux_views_stats['median']['global'])),
                                   'local': np.abs(np.min(flux_views['local_flux_view'] -
                                                         flux_views_stats['median']['local']))}

        # center by the flux view median and normalize by the flux view absolute minimum
        views_aux = {}
        for flux_view in flux_views:
            views_aux['{}_fluxnorm'.format(flux_view)] = \
                centering_and_normalization(flux_views[flux_view],
                                            flux_views_stats['median'][('local', 'global')['global' in flux_view]],
                                            flux_views_stats['min'][('local', 'global')['global' in flux_view]],
                                            report={'config': config, 'tce': tce, 'view': flux_view}
                                            )
        flux_views.update(views_aux)

        # # add odd-even views absolute difference
        # flux_views['local_flux_oddeven_view_diff_dir'] = flux_views['local_flux_even_view'] - \
        #                                                  flux_views['local_flux_odd_view']
        # if np.min(flux_views['local_flux_even_view']) <= np.min(flux_views['local_flux_odd_view']):
        #     flux_views['local_flux_oddeven_view_diff'] = flux_views['local_flux_even_view'] -\
        #                                                  flux_views['local_flux_odd_view']
        # else:
        #     flux_views['local_flux_oddeven_view_diff'] = flux_views['local_flux_odd_view'] - \
        #                                                  flux_views['local_flux_even_view']

        # center by the weak secondary flux view median and normalize by the weak secondary flux view absolute minimum
        weak_secondary_flux_views = {'global_weak_secondary_view': glob_weak_secondary_view,
                                     'local_weak_secondary_view': loc_weak_secondary_view
                                     }
        views_aux = {}
        for flux_view in weak_secondary_flux_views:
            # normalize by self absolute minimum
            weak_secondary_view_median = np.median(weak_secondary_flux_views[flux_view])
            views_aux['{}_selfnorm'.format(flux_view)] = \
                centering_and_normalization(weak_secondary_flux_views[flux_view],
                                            weak_secondary_view_median,
                                            np.abs(np.min(weak_secondary_flux_views[flux_view] -
                                                          weak_secondary_view_median)),
                                            report={'config': config, 'tce': tce, 'view': flux_view}
                                            )
            # normalize by flux absolute minimum
            views_aux['{}_fluxnorm'.format(flux_view)] = \
                centering_and_normalization(weak_secondary_flux_views[flux_view],
                                            flux_views_stats['median'][('local', 'global')['global' in flux_view]],
                                            flux_views_stats['min'][('local', 'global')['global' in flux_view]],
                                            report={'config': config, 'tce': tce, 'view': flux_view}
                                            )
        weak_secondary_flux_views.update(views_aux)
        num_transits.update({'local_weak_secondary_view': num_transits_wks})

        # get centroid views
        glob_centr_view = global_view(time_centroid_dist, centroid_dist,
                                      tce['tce_period'],
                                      centroid=True,
                                      normalize=False,
                                      centering=False,
                                      num_bins=config.num_bins_glob,
                                      bin_width_factor=config.bin_width_factor_glob,
                                      report={'config': config, 'tce': tce, 'view': 'global_centr_view'},
                                      tce_duration=tce['tce_duration'])
        loc_centr_view = local_view(time_centroid_dist, centroid_dist,
                                    tce['tce_period'], tce['tce_duration'],
                                    centroid=True,
                                    normalize=False,
                                    centering=False,
                                    num_durations=config.num_durations,
                                    num_bins=config.num_bins_loc,
                                    bin_width_factor=config.bin_width_factor_loc,
                                    report={'config': config, 'tce': tce, 'view': 'local_centr_view'})

        # if plot_preprocessing_tce:
        #     utils_visualization.plot_centroids_views(glob_centr_view, loc_centr_view, tce, config,
        #                                              os.path.join(config.output_dir, 'plots'),
        #                                              '7_non-normalized_centroid_views_aug{}'.format(tce['augmentation_idx']))

        # get median statistics for global and local centroid views
        centr_views_stats = {'median': {'global': np.median(glob_centr_view),
                                        'local': np.median(loc_centr_view)}
                             }

        # median centering, absolute and max normalization of the centroid views
        # removes magnitude and direction of shift; describes shift relative to oot only
        # defined in [0, 1]
        glob_centr_view_medc_abs_maxn = centering_and_normalization(glob_centr_view,
                                                                    centr_views_stats['median']['global'],
                                                                    1,  # np.max(glob_centr_view - centr_views_stats['median']['global']),
                                                                    report={'config': config, 'tce': tce,
                                                                    'view': 'global_centr_view_medcmaxn'}
                                                                    )
        glob_centr_view_medc_abs_maxn = np.abs(glob_centr_view_medc_abs_maxn)
        glob_centr_view_medc_abs_maxn /= np.max(glob_centr_view_medc_abs_maxn)

        loc_centr_view_medc_abs_maxn = centering_and_normalization(loc_centr_view,
                                                                   centr_views_stats['median']['local'],
                                                                   1,  # np.max(loc_centr_view - centr_views_stats['median']['local']),
                                                                   report={'config': config, 'tce': tce,
                                                                           'view': 'local_centr_view_medcmaxn'}
                                                                   )
        loc_centr_view_medc_abs_maxn = np.abs(loc_centr_view_medc_abs_maxn)
        loc_centr_view_medc_abs_maxn /= np.max(loc_centr_view_medc_abs_maxn)

        # median centering, max absolute normalization of the centroid views
        # removes magnitude of shift, but keeps direction of shift from oot relative to the target star
        # closer to -1, closer to target star; closer to 0, closer to oot; closer to 1, further from target star
        # defined in [-1, 1]
        glob_centr_view_medc_maxabsn = centering_and_normalization(glob_centr_view,
                                                                   centr_views_stats['median']['global'],
                                                                   1,  # np.max(glob_centr_view - centr_views_stats['median']['global']),
                                                                   report={'config': config, 'tce': tce,
                                                                           'view': 'global_centr_view_medcmaxn'}
                                                                   )
        glob_centr_view_medc_maxabsn /= np.max(np.abs(glob_centr_view_medc_maxabsn))

        loc_centr_view_medc_maxabsn = centering_and_normalization(loc_centr_view,
                                                                  centr_views_stats['median']['local'],
                                                                  1,  # np.max(loc_centr_view - centr_views_stats['median']['local']),
                                                                  report={'config': config, 'tce': tce,
                                                                          'view': 'local_centr_view_medcmaxn'}
                                                                  )
        loc_centr_view_medc_maxabsn /= np.max(np.abs(loc_centr_view_medc_maxabsn))

        # median normalization for centroid views
        # centroid is normalized by oot value, which is set to around 1
        # the shift is scaled by oot, so bound is not defined [P/N, 1] or [1, P/N]
        glob_centr_view_mn = centering_and_normalization(glob_centr_view, 0, centr_views_stats['median']['global'],
                                                         report={'config': config, 'tce': tce,
                                                                 'view': 'global_centr_view_medn'}
                                                         )
        loc_centr_view_mn = centering_and_normalization(loc_centr_view, 0, centr_views_stats['median']['local'],
                                                        report={'config': config, 'tce': tce,
                                                                'view': 'local_centr_view_medn'}
                                                        )

        # FDL centroid normalization
        # non-normalized views that need to be normalized using statistics from the training set
        glob_centr_fdl_view = global_view(time_centroid_dist_fdl, centroid_dist_fdl,
                                          tce['tce_period'],
                                          centroid=True,
                                          normalize=False,
                                          centering=False,
                                          num_bins=config.num_bins_glob,
                                          bin_width_factor=config.bin_width_factor_glob,
                                          report={'config': config, 'tce': tce, 'view': 'global_centr_fdl_view'},
                                          tce_duration=tce['tce_duration'])
        loc_centr_fdl_view = local_view(time_centroid_dist_fdl, centroid_dist_fdl,
                                        tce['tce_period'], tce['tce_duration'],
                                        centroid=True,
                                        normalize=False,
                                        centering=False,
                                        num_durations=config.num_durations,
                                        num_bins=config.num_bins_loc,
                                        bin_width_factor=config.bin_width_factor_loc,
                                        report={'config': config, 'tce': tce, 'view': 'local_centr_fdl_view'})

        # if plot_preprocessing_tce:
        #     utils_visualization.plot_centroids_views(glob_centr_fdl_view, loc_centr_fdl_view, tce, config,
        #                                              os.path.join(config.output_dir, 'plots'),
        #                                              '7_non-normalized_centroid_fdl_views_aug{}'.format(tce['augmentation_idx']))

        centr_views = {'global_centr_view': glob_centr_view,
                       'local_centr_view': loc_centr_view,
                       'global_centr_view_medcmaxn': glob_centr_view_medc_abs_maxn,
                       'local_centr_view_medcmaxn': loc_centr_view_medc_abs_maxn,
                       'global_centr_view_medn': glob_centr_view_mn,
                       'local_centr_view_medn': loc_centr_view_mn,
                       'global_centr_view_medcmaxn_dir': glob_centr_view_medc_maxabsn,
                       'local_centr_view_medcmaxn_dir': loc_centr_view_medc_maxabsn,
                       'global_centr_fdl_view': glob_centr_fdl_view,
                       'local_centr_fdl_view': loc_centr_fdl_view
                       }
        num_transits.update({'global_centr_view': num_transits_centroid,
                             'global_centr_fdl_view': num_transits_centrfdl
                             })

        # initialize dictionary with the views that will be stored in the TFRecords example for this TCE
        views = {}
        views.update(flux_views)
        views.update(weak_secondary_flux_views)
        views.update(centr_views)

        # utils_visualization.plot_diff_oddeven(views, tce, config, os.path.join(config.output_dir, 'plots'),
        #                                       '8_diffoddeven_aug{}'.format(tce['augmentation_idx']))

        if plot_preprocessing_tce:
            # CHANGE NUMBER OF VIEWS PLOTTED!!!
            views_to_plot = ['global_flux_view', 'local_flux_view', 'local_flux_odd_view', 'local_flux_even_view',
                          'global_flux_view_fluxnorm', 'local_flux_view_fluxnorm', 'local_flux_odd_view_fluxnorm',
                          'local_flux_even_view_fluxnorm',
                          'local_weak_secondary_view', 'local_weak_secondary_view_selfnorm',
                          'local_weak_secondary_view_fluxnorm', 'global_centr_view', 'local_centr_view',
                          'global_centr_fdl_view', 'local_centr_fdl_view']
            views_plot = {view_name: view for view_name, view in views.items() if view_name in views_to_plot}
            utils_visualization.plot_all_views(views_plot, tce, config, (4, 4),
                                               os.path.join(config.output_dir, 'plots'),
                                               '8_final_views_aug{}'.format(tce['augmentation_idx']),
                                               num_transits)

    except Exception as e:
        report_exclusion(config, tce, 'Error when creating views', stderr=e)
        return None

    for view in views:
        if np.any(~np.isfinite(views[view])):  # at least one point is non-finite (infinite or NaN)
            report_exclusion(config, tce, 'View has at least one non-finite data point in view {}.'.format(view))
            return None

    # set time series features in the example to be written to a TFRecord
    for view in views:
        example_util.set_float_feature(ex, view, views[view])

    # set other features from the TCE table - diagnostic statistics, transit fits, stellar parameters...
    for name, value in tce.items():
        if name == 'Public Comment':  # can add special characters that are not serializable in the TFRecords
            continue
        try:
            example_util.set_feature(ex, name, [value])
        except:
            report_exclusion(config, tce, 'Couldnt set up this TCE table parameter: {}.'.format(name))
            print(name, value)

    # set scalar parameters
    scalar_params = len(config.scalar_params) * [None]
    if len(config.scalar_params) > 0:
        for scalar_param_i in range(len(config.scalar_params)):
            scalar_params[scalar_param_i] = tce[config.scalar_params[scalar_param_i]]

        example_util.set_feature(ex, 'scalar_params', scalar_params)

    return ex
