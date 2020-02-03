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

"""Functions for reading Kepler data."""

import os.path

from astropy.io import fits
import numpy as np

from src_preprocessing.light_curve import util
from tensorflow import gfile

# Quarter index to filename prefix for long cadence Kepler data.
# Reference: https://archive.stsci.edu/kepler/software/get_kepler.py
LONG_CADENCE_QUARTER_PREFIXES = {
    0: ["2009131105131"],
    1: ["2009166043257"],
    2: ["2009259160929"],
    3: ["2009350155506"],
    4: ["2010078095331", "2010009091648"],
    5: ["2010174085026"],
    6: ["2010265121752"],
    7: ["2010355172524"],
    8: ["2011073133259"],
    9: ["2011177032512"],
    10: ["2011271113734"],
    11: ["2012004120508"],
    12: ["2012088054726"],
    13: ["2012179063303"],
    14: ["2012277125453"],
    15: ["2013011073258"],
    16: ["2013098041711"],
    17: ["2013131215648"]
}

# Quarter index to filename prefix for short cadence Kepler data.
# Reference: https://archive.stsci.edu/kepler/software/get_kepler.py
SHORT_CADENCE_QUARTER_PREFIXES = {
    0: ["2009131110544"],
    1: ["2009166044711"],
    2: ["2009201121230", "2009231120729", "2009259162342"],
    3: ["2009291181958", "2009322144938", "2009350160919"],
    4: ["2010009094841", "2010019161129", "2010049094358", "2010078100744"],
    5: ["2010111051353", "2010140023957", "2010174090439"],
    6: ["2010203174610", "2010234115140", "2010265121752"],
    7: ["2010296114515", "2010326094124", "2010355172524"],
    8: ["2011024051157", "2011053090032", "2011073133259"],
    9: ["2011116030358", "2011145075126", "2011177032512"],
    10: ["2011208035123", "2011240104155", "2011271113734"],
    11: ["2011303113607", "2011334093404", "2012004120508"],
    12: ["2012032013838", "2012060035710", "2012088054726"],
    13: ["2012121044856", "2012151031540", "2012179063303"],
    14: ["2012211050319", "2012242122129", "2012277125453"],
    15: ["2012310112549", "2012341132017", "2013011073258"],
    16: ["2013017113907", "2013065031647", "2013098041711"],
    17: ["2013121191144", "2013131215648"]
}

# Quarter order for different scrambling procedures.
# Page 9: https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/20170009549.pdf.
SIMULATED_DATA_SCRAMBLE_ORDERS = {
    "SCR1": [0, 13, 14, 15, 16, 9, 10, 11, 12, 5, 6, 7, 8, 1, 2, 3, 4, 17],
    "SCR2": [0, 1, 2, 3, 4, 13, 14, 15, 16, 9, 10, 11, 12, 5, 6, 7, 8, 17],
    "SCR3": [0, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 17],
}


def kepler_filenames(base_dir,
                     kep_id,
                     long_cadence=True,
                     quarters=None,
                     injected_group=None,
                     check_existence=True):
    """Returns the light curve filenames for a Kepler target star.

  This function assumes the directory structure of the Mikulski Archive for
  Space Telescopes (http://archive.stsci.edu/pub/kepler/lightcurves).
  Specifically, the filenames for a particular Kepler target star have the
  following format:

    ${kep_id:0:4}/${kep_id}/kplr${kep_id}-${quarter_prefix}_${type}.fits,

  where:
    kep_id is the Kepler id left-padded with zeros to length 9;
    quarter_prefix is the filename quarter prefix;
    type is one of "llc" (long cadence light curve) or "slc" (short cadence
        light curve).

  Args:
    base_dir: Base directory containing Kepler data.
    kep_id: Id of the Kepler target star. May be an int or a possibly zero-
      padded string.
    long_cadence: Whether to read a long cadence (~29.4 min / measurement) light
      curve as opposed to a short cadence (~1 min / measurement) light curve.
    quarters: Optional list of integers in [0, 17]; the quarters of the Kepler
      mission to return.
    injected_group: Optional string indicating injected light curves. One of
      "inj1", "inj2", "inj3".
    check_existence: If True, only return filenames corresponding to files that
      exist (not all stars have data for all quarters).

  Returns:
    A list of filenames.
  """

    # Pad the Kepler id with zeros to length 9.
    kep_id = "{:09d}".format(int(kep_id))

    quarter_prefixes, cadence_suffix = ((LONG_CADENCE_QUARTER_PREFIXES, "llc")
                                        if long_cadence else
                                        (SHORT_CADENCE_QUARTER_PREFIXES, "slc"))

    if quarters is None:
        quarters = quarter_prefixes.keys()

    quarters = sorted(quarters)  # Sort quarters chronologically.

    filenames = []
    base_dir = os.path.join(base_dir, kep_id[0:4], kep_id)
    for quarter in quarters:
        for quarter_prefix in quarter_prefixes[quarter]:
            if injected_group:
                base_name = "kplr{}-{}_INJECTED-{}_{}.fits".format(
                    kep_id, quarter_prefix, injected_group, cadence_suffix)
            else:
                base_name = "kplr{}-{}_{}.fits".format(kep_id, quarter_prefix,
                                                       cadence_suffix)
            filename = os.path.join(base_dir, base_name)
            # Not all stars have data for all quarters.
            if not check_existence or gfile.Exists(filename):
                filenames.append(filename)

    return filenames


def scramble_light_curve(all_time, all_flux, all_quarters, scramble_type):
    """Scrambles a light curve according to a given scrambling procedure.

  Args:
    all_time: List holding arrays of time values, each containing a quarter of
      time data.
    all_flux: List holding arrays of flux values, each containing a quarter of
      flux data.
    all_quarters: List of integers specifying which quarters are present in
      the light curve (max is 18: Q0...Q17).
    scramble_type: String specifying the scramble order, one of {'SCR1', 'SCR2',
      'SCR3'}.

  Returns:
    scr_flux: Scrambled flux values; the same list as the input flux in another
      order.
    scr_time: Time values, re-partitioned to match sizes of the scr_flux lists.
  """
    order = SIMULATED_DATA_SCRAMBLE_ORDERS[scramble_type]
    scr_flux = []
    for quarter in order:
        # Ignore missing quarters in the scramble order.
        if quarter in all_quarters:
            scr_flux.append(all_flux[all_quarters.index(quarter)])

    scr_time = util.reshard_arrays(all_time, scr_flux)

    return scr_time, scr_flux


def get_gap_ephems_for_DR25readouttbl(table, tce):
    """ Get ephemeris for the gapped TCEs.

    :param table: pandas DataFrame, TCE ephemeris table
    :param tce: row of pandas DataFrame, TCE of interest
    :return:
        gap_ephems: dict, each item contains ephemeris information for a gapped TCE
    """

    ephem_keys = {'epoch': 'tce_time0bk', 'period': 'tce_period', 'duration': 'tce_duration'}

    # initialize empty dictionary for gapped TCEs
    gap_ephems = {}

    # FIXME: is it already in days?
    d_factor = 1.0 / 24.0  # if satellite_id == 'kepler' else 1.0  # for tess, duration is already in [day] units

    # search for TCEs belonging to the same Kepler ID but with a different TCE planet number
    for tce_i, tce_i_ephem in table[tce.kepid].items():
        if tce.tce_plnt_num != tce_i:  # if it is not the TCE of interest
            # gap_ephems[len(gap_ephems)] = {'epoch': tce_i_ephem['epoch_corr'],
            #                                'period': tce_i_ephem['period'],
            #                                'duration': tce_i_ephem['duration'] * d_factor,
            #                                'tce_n': tce_i}
            gap_ephems[len(gap_ephems)] = {'epoch': tce_i_ephem[ephem_keys['epoch']],
                                           'period': tce_i_ephem[ephem_keys['period']],
                                           'duration': tce_i_ephem[ephem_keys['duration']] * d_factor,
                                           'tce_n': tce_i}

    return gap_ephems




# def transit_points(all_time, tce, table):
def transit_points(all_time, tce):
    gap_pad = 0
    d_factor = 1.0 / 24.0
    ephem = {}
    # for tce_i, tce_i_ephem in table[tce.kepid].items():
    #     if tce.tce_plnt_num == tce_i:
    #         ephem = {'epoch': tce_i_ephem['epoch_corr'],
    #                  'period': tce_i_ephem['period'],
    #                  'duration': tce_i_ephem['duration'] * d_factor,
    #                  'tce_n': tce_i
    #                  }
    ephem = {'epoch': tce['tce_time0bk'],
             'period': tce['tce_period'],
             'duration': tce['tce_duration'] * d_factor,
             }

    begin_time, end_time = all_time[0][0], all_time[-1][-1]

    if ephem['epoch'] < begin_time:
        ephem['epoch'] = ephem['epoch'] + ephem['period'] * np.ceil((begin_time - ephem['epoch']) / ephem['period'])
    else:
        ephem['epoch'] = ephem['epoch'] - ephem['period'] * np.floor((ephem['epoch'] - begin_time) / ephem['period'])
    ephem['duration'] = ephem['duration'] * (1 + 2 * gap_pad)

    if ephem['epoch'] <= end_time:
        midTransitTimes = np.arange(ephem['epoch'], end_time, ephem['period'])
        midTransitTimeBefore = midTransitTimes[0] - ephem['period']
        midTransitTimeAfter = midTransitTimes[-1] + ephem['period']
    else:
        midTransitTimes = []
        midTransitTimeBefore = ephem['epoch'] - ephem['period']
        midTransitTimeAfter = ephem['epoch']

    extendedMidTransitTimes = np.concatenate([[midTransitTimeBefore], midTransitTimes, [midTransitTimeAfter]])

    return extendedMidTransitTimes


def gap_other_tces(all_time, all_flux, tce, table, config, conf_dict, gap_pad=0):
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
        imputed_time: list of numpy arrays,
    """

    # get gapped TCEs ephemeris
    if config.satellite == 'kepler':
        gap_ephems = table.loc[(table['kepid'] == tce.kepid) &
                               (table['tce_plnt_num'] != tce.tce_plnt_num)][['tce_period', 'tce_duration',
                                                                             'tce_time0bk']]
    else:
        raise NotImplementedError('Gapping is still not implemented for TESS.')

        # gap_ephems = table.loc[(table['tic'] == tce.kepid) &
        #                        (table['tce_plnt_num'] != tce.tce_plnt_num)][['tce_period', 'tce_duration',
        #                                                                      'tce_time0bk']]

    # if gapping with confidence level, remove those gapped TCEs that are not in the confidence dict
    if config.gap_with_confidence_level:
        poplist = []
        for index, gapped_tce in gap_ephems.iterrows():
            if (tce.kepid, gapped_tce.tce_plnt_num) not in conf_dict or \
                    conf_dict[(tce.kepid, gapped_tce.tce_plnt_num)] < config.gap_confidence_level:
                poplist += [gapped_tce.tce_plnt_num]

        gap_ephems = gap_ephems.loc[gap_ephems['tce_plnt_num'].isin(poplist)]

    imputed_time = [] if config.gap_imputed else None

    begin_time, end_time = all_time[0][0], all_time[-1][-1]

    for ephem_i, ephem in gap_ephems.iterrows():

        if ephem['tce_time0bk'] < begin_time:  # when the epoch of the gapped TCE occurs before the first cadence
            ephem['tce_time0bk'] = ephem['tce_time0bk'] + \
                                   ephem['tce_period'] * np.ceil((begin_time - ephem['tce_time0bk']) /
                                                                 ephem['tce_period'])
        else:
            ephem['tce_time0bk'] = ephem['tce_time0bk'] - ephem['tce_period'] * np.floor(
                (ephem['tce_time0bk'] - begin_time) / ephem['tce_period'])

        ephem['tce_duration'] = ephem['tce_duration'] * (1 + 2 * gap_pad)

        if ephem['tce_time0bk'] <= end_time:
            midTransitTimes = np.arange(ephem['tce_time0bk'], end_time, ephem['tce_period'])
            midTransitTimeBefore = midTransitTimes[0] - ephem['tce_period']
            midTransitTimeAfter = midTransitTimes[-1] + ephem['tce_period']
        else:
            midTransitTimes = []
            midTransitTimeBefore = ephem['tce_time0bk'] - ephem['tce_period']
            midTransitTimeAfter = ephem['tce_time0bk']

        extendedMidTransitTimes = np.concatenate([[midTransitTimeBefore], midTransitTimes, [midTransitTimeAfter]])

        startTransitTimes = (extendedMidTransitTimes - 0.5 * ephem['tce_duration'])
        endTransitTimes = (extendedMidTransitTimes + 0.5 * ephem['tce_duration'])
        nTransits = len(startTransitTimes)

        for quarter_i, time_i in enumerate(all_time):
            transit_boolean = np.full(time_i.shape[0], False)

            index = 0
            for i in range(time_i.shape[0]):
                for j in [index, index + 1]:
                    if startTransitTimes[j] <= time_i[i] <= endTransitTimes[j]:
                        transit_boolean[i] = True
                        if j > index:
                            index += 1
                        break
                if index > nTransits - 1:
                    break

            if config.gap_imputed and np.any(transit_boolean):  # if gaps need to be imputed and True in transit_boolean
                # imputed_time += [[all_time[quarter_i][transit_boolean], all_flux[quarter_i][transit_boolean],
                #                   {all_centroids[coord][quarter_i][transit_boolean] for coord in all_centroids}]]
                imputed_time += [[all_time[quarter_i][transit_boolean], all_flux[quarter_i][transit_boolean]]]

            all_flux[quarter_i][transit_boolean] = np.nan
            # all_centroids['x'][quarter_i][transit_boolean] = np.nan
            # all_centroids['y'][quarter_i][transit_boolean] = np.nan

    return all_time, all_flux, imputed_time


def convertpxtoradec_centr(centroid_x, centroid_y, cd_transform_matrix, ref_px_apert, ref_angcoord):
    """ Convert the centroid time series from pixel coordinates to world coordinates right ascension (RA) and
    declination (Dec).

    :param centroid_x: list [num cadences], column centroid position time series [pixel] in the CCD frame
    :param centroid_y: list, row centroid position time series [pixel] in the CCD frame
    :param cd_transform_matrix: numpy array [2x2], coordinates transformation matrix from col, row aperture frame
    to world coordinates RA and Dec
    :param ref_px:  numpy array [2x1], reference pixel [pixel] coordinates in the aperture frame of the target star
    frame
    :param ref_px_apert: numpy array [2x1], reference pixel [pixel] coordinates of the origin of the aperture frame in
    the CCD frame
    :param ref_angcoord: numpy array [2x1], RA and Dec at reference pixel [RA, Dec]
    :return:
        ra: numpy array [num cadences], right ascension coordinate centroid time series
        dec: numpy array [num cadences], declination coordinate centroid time series
    """

    px_coords = np.reshape(np.concatenate((centroid_x, centroid_y)), (2, len(centroid_x)))

    ra, dec = np.matmul(cd_transform_matrix, px_coords - ref_px_apert + np.array([[1], [1]])) + ref_angcoord

    return ra, dec


def read_kepler_light_curve(filenames,
                            light_curve_extension="LIGHTCURVE",
                            interpolate_missing_time=False):
    """ Reads data from FITS files for a Kepler target star.

  Args:
    filenames: A list of .fits files containing time and flux measurements.
    light_curve_extension: Name of the HDU 1 extension containing light curves.
    scramble_type: What scrambling procedure to use: 'SCR1', 'SCR2', or 'SCR3'
      (pg 9: https://exoplanetarchive.ipac.caltech.edu/docs/KSCI-19114-002.pdf).
    interpolate_missing_time: Whether to interpolate missing (NaN) time values.
      This should only affect the output if scramble_type is specified (NaN time
      values typically come with NaN flux values, which are removed anyway, but
      scrambing decouples NaN time values from NaN flux values).
    centroid_radec: bool, whether to transform the centroid time series from the CCD module pixel coordinates to RA
      and Dec, or not

  Returns:
    all_time: A list of numpy arrays; the time values of the light curve.
    all_flux: A list of numpy arrays; the flux values of the light curve.
    all_centroid: A dict, 'x' is a list of numpy arrays with either the col or RA coordinates of the centroid values of
    the light curve; 'y' is a list of numpy arrays with either the row or Dec coordinates.
    add_info: A dict with additional data extracted from the FITS files; 'quarter' is a list of quarters for each numpy
    array of the light curve; 'module' is the same but for the module in which the target is in every quarter;
    'target position' is a list of two elements which correspond to the target star position, either in world (RA, Dec)
    or local CCD (x, y) pixel coordinates
  """

    # initialize variables
    all_time = []
    all_flux = []
    # all_centroid = {'x': [], 'y': []}
    # add_info = {'quarter': [], 'module': [], 'target position': []}

    # def _has_finite(array):
    #     for i in array:
    #         if np.isfinite(i):
    #             return True
    #
    #     return False

    # iterate through the FITS files for the target star
    for filename in filenames:
        with fits.open(gfile.Open(filename, "rb")) as hdu_list:

            # # needed to transform coordinates when target is in module 13 and when using pixel coordinates
            # quarter = hdu_list["PRIMARY"].header["QUARTER"]
            # if quarter == 0:  # ignore quarter 0
            #     continue
            # module = hdu_list["PRIMARY"].header["MODULE"]

            # kepid_coord1, kepid_coord2 = hdu_list["PRIMARY"].header["RA_OBJ"], hdu_list["PRIMARY"].header["DEC_OBJ"]

            # if len(add_info['target position']) == 0:
            #     add_info['target position'] = [kepid_coord1, kepid_coord2]

            # # TODO: convert target position from RA and Dec to local CCD pixel coordinates
            # if not centroid_radec:
            #     pass

            light_curve = hdu_list[light_curve_extension].data

            # if _has_finite(light_curve.PSF_CENTR1):
            #     centroid_x, centroid_y = light_curve.PSF_CENTR1, light_curve.PSF_CENTR2
            # else:
            #     if _has_finite(light_curve.MOM_CENTR1):
            #         centroid_x, centroid_y = light_curve.MOM_CENTR1, light_curve.MOM_CENTR2
            #     else:
            #         continue  # no data

            # # get components required for the transformation from CCD pixel coordinates to world coordinates RA and Dec
            # if centroid_radec:
            #
            #     # transformation matrix from aperture coordinate frame to RA and Dec
            #     cd_transform_matrix = np.zeros((2, 2))
            #     cd_transform_matrix[0] = hdu_list['APERTURE'].header['PC1_1'] * hdu_list['APERTURE'].header['CDELT1'], \
            #                              hdu_list['APERTURE'].header['PC1_2'] * hdu_list['APERTURE'].header['CDELT1']
            #     cd_transform_matrix[1] = hdu_list['APERTURE'].header['PC2_1'] * hdu_list['APERTURE'].header['CDELT2'], \
            #                              hdu_list['APERTURE'].header['PC2_2'] * hdu_list['APERTURE'].header['CDELT2']
            #
            #     # # reference pixel in the aperture coordinate frame
            #     # ref_px = np.array([[hdu_list['APERTURE'].header['CRPIX1']], [hdu_list['APERTURE'].header['CRPIX2']]])
            #
            #     # reference pixel in CCD coordinate frame
            #     ref_px_apert = np.array([[hdu_list['APERTURE'].header['CRVAL1P']],
            #                              [hdu_list['APERTURE'].header['CRVAL2P']]])
            #
            #     # RA and Dec at reference pixel
            #     ref_angcoord = np.array([[hdu_list['APERTURE'].header['CRVAL1']],
            #                              [hdu_list['APERTURE'].header['CRVAL2']]])

        # # convert from CCD pixel coordinates to world coordinates RA and Dec
        # if centroid_radec:
        #     centroid_x, centroid_y = convertpxtoradec_centr(centroid_x, centroid_y, cd_transform_matrix, ref_px_apert,
        #                                                     ref_angcoord)

        # get time and PDC-SAP flux arrays
        time = light_curve.TIME
        flux = light_curve.PDCSAP_FLUX

        if not time.size:
            continue  # No data.

        # Possibly interpolate missing time values.
        if interpolate_missing_time:
            time = util.interpolate_missing_time(time, light_curve.CADENCENO)

        all_time.append(time)
        all_flux.append(flux)
        # all_centroid['x'].append(centroid_x)
        # all_centroid['y'].append(centroid_y)

        # add_info['quarter'].append(quarter)
        # add_info['module'].append(module)

    # # TODO: adapt this to the centroid time series as well?
    # if scramble_type:
    #     all_time, all_flux = scramble_light_curve(all_time, all_flux, add_info['quarter'], scramble_type)

    return all_time, all_flux  # , all_centroid, add_info
