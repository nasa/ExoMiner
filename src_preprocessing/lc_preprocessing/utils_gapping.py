""" Utility functions used to gap TCEs from time series. """

# 3rd party
import numpy as np

# local
from src_preprocessing.lc_preprocessing.utils_ephemeris import create_binary_time_series, find_first_epoch_after_this_time


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

    if 'tce_maxmesd' in tce:
        # transit_duration = min(tce['tce_duration'] * (1 + 2 * gap_pad), np.abs(tce['tce_maxmesd']))
        # transit_duration = min(tce['tce_duration'] * gap_pad, np.abs(tce['tce_maxmesd']))
        phase_sec = np.abs(tce['tce_maxmesd'])
        if 2 * tce['tce_duration'] < phase_sec:
            transit_duration = gap_pad * tce['tce_duration']
        else:
            transit_duration = tce['tce_duration'] + phase_sec

        transit_duration = min(transit_duration, phase_sec - 0.5 * tce['tce_duration'])
    else:
        transit_duration = tce['tce_duration'] * gap_pad

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


def gap_tces(all_time, ephem, config):
    """ Get gapped indices for a given set of one or more TCEs to be gapped.

    :param all_time: list of Numpy arrays, timestamps
    :param ephem: list of ephemeris of TCEs to be gapped
    :param config: dict, preprocessing parameters
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
            # if config['satellite'] != 'kepler' and add_info['sector'][i] not in gapSectors[real_ephem_i]:
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
        if config['gap_imputed'] and len(transit_idxs) > 0:
            imputed_time.append(all_time[time_i][transit_idxs])

        all_transit_idxs.append(transit_idxs)

    return all_transit_idxs, imputed_time


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
