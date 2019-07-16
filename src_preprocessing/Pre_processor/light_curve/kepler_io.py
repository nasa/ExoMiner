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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path

from astropy.io import fits
import numpy as np

from src_preprocessing.Pre_processor.light_curve import util
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


def get_gap_ephems(table, tce, satellite_id):
    """

    :param table: pandas DataFrame, TCE table
    :param tce:  row of table, target TCE
    :param satellite_id: str, either 'kepler' or 'tess'
    :return:
        - gap_ephems: dict, each entry contains as value a dict with ephemeris information of a TCE that belongs to the
        same Kepler ID as tce.
    """

    gap_ephems = {}

    d_factor = 1.0 / 24.0 if satellite_id == 'kepler' else 1.0  # for tess, duration is already in [day] units

    # iterate over all TCEs with the same Kepler ID and extract the ephemeris of those that are not the target TCE
    for tce_i, tce_i_ephem in table[tce.kepid].items():
        if tce.tce_plnt_num != tce_i:
            gap_ephems[len(gap_ephems)] = {'epoch': tce_i_ephem['epoch_corr'],
                                           'period': tce_i_ephem['period'],
                                           'duration': tce_i_ephem['duration'] * d_factor,
                                           'tce_n': tce_i}

    return gap_ephems


def get_centr_oot_rms(all_centroids, all_time, tce, table, config):
    gap_pad = 0

    gap_ephems = {}
    d_factor = 1.0 / 24.0 if config.satellite == 'kepler' else 1.0  # for tess, duration is already in [day] units

    for tce_i, tce_i_ephem in table[tce.kepid].items():
        gap_ephems[len(gap_ephems)] = {'epoch': tce_i_ephem['epoch_corr'],
                                       'period': tce_i_ephem['period'],
                                       'duration': tce_i_ephem['duration'] * d_factor,
                                       'tce_n': tce_i}

    begin_time, end_time = all_time[0][0], all_time[-1][-1]
    for ephem in gap_ephems.values():
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

        startTransitTimes = (extendedMidTransitTimes - 0.5 * ephem['duration'])
        endTransitTimes = (extendedMidTransitTimes + 0.5 * ephem['duration'])
        nTransits = len(startTransitTimes)

        for quarter_i, time_i in enumerate(all_time):
            transit_boolean = np.full(time_i.shape[0], False)

            index = 0
            for i in range(time_i.shape[0]):
                for j in [index, index+1]:
                    if startTransitTimes[j] <= time_i[i] <= endTransitTimes[j]:
                        transit_boolean[i] = True
                        if j > index:
                            index += 1
                        break
                if index > nTransits-1:
                    break

            all_centroids['x'][quarter_i][transit_boolean] = np.nan
            all_centroids['y'][quarter_i][transit_boolean] = np.nan

    all_centroids_2 = {}
    for dim, array in all_centroids.items():
        all_centroids_2[dim] = np.concatenate(array)

    def _get_rms(array):
        return np.sqrt(np.square(np.nanmean(array)) + np.nanvar(array))

    return {key: _get_rms(centr_array) for key, centr_array in all_centroids_2.items()}


def gap_other_tces(all_time, all_flux, all_centroids, tce, table, config, conf_dict):
    """

    :param all_time:
    :param all_flux:
    :param all_centroids:
    :param tce:
    :param table:
    :param config:
    :param conf_dict:
    :return:
        all_time: list of numpy arrays, cadences with NaNs for the gapped transits
        all_flux: list of numpy arrays, flux time series with NaNs for the gapped transits
        all_centroids: list of numpy arrays, centroid time series with NaNs for the gapped transits
        imputed_time: list of list of numpy arrays with cadences and flux for the gapped transits
    """

    # FIXME: shouldn't it be an argument?
    # padding used when removing the TCE. The formula is tce_duration + 2 * gap_pad
    gap_pad = 0

    # get ephemeris information on the other TCEs that belong to the same Kepler ID
    gap_ephems = get_gap_ephems(table, tce, config.satellite)

    # when using confidence level based gapping
    if config.gap_with_confidence_level:
        poplist = []
        for ephem_i, ephem in gap_ephems.items():
            # if the TCE number is not in the config dict or is below the confidence level, do not consider it in the
            # gapping process
            if (tce.kepid, ephem['tce_n']) not in conf_dict or \
                    conf_dict[(tce.kepid, ephem['tce_n'])] < config.gap_confidence_level:
                poplist += [ephem_i]
        for i in poplist:
            gap_ephems.pop(i)

    # WHY ARE WE DOING THIS IN HERE???
    # if config.whitened or config.satellite == 'tess':
    if config.satellite == 'tess':
        all_time, all_flux, all_centroids['x'], all_centroids['y'] = \
            [all_time], [all_flux], [all_centroids['x']], [all_centroids['y']]

    # get cadences and flux of gapped TCEs in order to add noise if imputing the light curve
    imputed_time = [] if config.gap_imputed else None
    begin_time, end_time = all_time[0][0], all_time[-1][-1]
    for ephem in gap_ephems.values():
        # update the epoch for the gapped TCEs
        if ephem['epoch'] < begin_time:  # if the epoch for this TCE happens before the first time stamp in the flux
            ephem['epoch'] = ephem['epoch'] + ephem['period'] * np.ceil((begin_time - ephem['epoch']) / ephem['period'])
        else:  # if it happens after
            ephem['epoch'] = ephem['epoch'] - ephem['period'] * np.floor((ephem['epoch'] - begin_time) / ephem['period'])

        # update duration with padding
        ephem['duration'] = ephem['duration'] * (1 + 2 * gap_pad)

        # get the mid-transit times
        if ephem['epoch'] <= end_time:
            midTransitTimes = np.arange(ephem['epoch'], end_time, ephem['period'])
            midTransitTimeBefore = midTransitTimes[0] - ephem['period']
            midTransitTimeAfter = midTransitTimes[-1] + ephem['period']
        # FIXME: this can only happen if the period is larger than the time interval between the epoch and begin_time, right?
        else:
            midTransitTimes = []
            midTransitTimeBefore = ephem['epoch'] - ephem['period']
            midTransitTimeAfter = ephem['epoch']

        extendedMidTransitTimes = np.concatenate([[midTransitTimeBefore], midTransitTimes, [midTransitTimeAfter]])

        # get the start and end cadences of the gapped transits
        startTransitTimes = (extendedMidTransitTimes - 0.5 * ephem['duration'])
        endTransitTimes = (extendedMidTransitTimes + 0.5 * ephem['duration'])
        nTransits = len(startTransitTimes)

        for quarter_i, time_i in enumerate(all_time):

            transit_boolean = np.full(time_i.shape[0], False)

            index = 0
            for i in range(time_i.shape[0]):
                for j in [index, index+1]:  # test if it is in the current transit interval or in the next one
                    if startTransitTimes[j] <= time_i[i] <= endTransitTimes[j]:
                        transit_boolean[i] = True
                        if j > index:  # if it is in the next one, the next one becomes the current interval
                            index += 1
                        break
                if index > nTransits-1:
                    break

            # if gaps need to be imputed and True in transit_boolean
            if config.gap_imputed and np.any(transit_boolean):
                imputed_time += [[all_time[quarter_i][transit_boolean], all_flux[quarter_i][transit_boolean]]]

            # replace the gapped transits by NaN
            all_flux[quarter_i][transit_boolean] = np.nan
            all_centroids['x'][quarter_i][transit_boolean] = np.nan
            all_centroids['y'][quarter_i][transit_boolean] = np.nan

    return all_time, all_flux, all_centroids, imputed_time


def read_kepler_light_curve(filenames,
                            light_curve_extension="LIGHTCURVE",
                            scramble_type=None,
                            interpolate_missing_time=False):
  """Reads time and flux measurements for a Kepler target star.

  Args:
    filenames: A list of .fits files containing time and flux measurements.
    light_curve_extension: Name of the HDU 1 extension containing light curves.
    scramble_type: What scrambling procedure to use: 'SCR1', 'SCR2', or 'SCR3'
      (pg 9: https://exoplanetarchive.ipac.caltech.edu/docs/KSCI-19114-002.pdf).
    interpolate_missing_time: Whether to interpolate missing (NaN) time values.
      This should only affect the output if scramble_type is specified (NaN time
      values typically come with NaN flux values, which are removed anyway, but
      scrambing decouples NaN time values from NaN flux values).

  Returns:
    all_time: A list of numpy arrays; the time values of the light curve.
    all_flux: A list of numpy arrays; the flux values of the light curve.
  """
  all_time = []
  all_flux = []
  all_quarters = []

  all_centroid = {'x': [], 'y': []}

  def _has_finite(array):
    for i in array:
      if np.isfinite(i):
        return True

    return False

  for filename in filenames:
    with fits.open(gfile.Open(filename, "rb")) as hdu_list:
      quarter = hdu_list["PRIMARY"].header["QUARTER"]
      light_curve = hdu_list[light_curve_extension].data

      if _has_finite(light_curve.PSF_CENTR1):
        centroid_x, centroid_y = light_curve.PSF_CENTR1, light_curve.PSF_CENTR2
      else:
        if _has_finite(light_curve.MOM_CENTR1):
          centroid_x, centroid_y = light_curve.MOM_CENTR1, light_curve.MOM_CENTR2
        else:
          continue  # no data

    all_centroid['x'].append(centroid_x)
    all_centroid['y'].append(centroid_y)

    time = light_curve.TIME
    flux = light_curve.PDCSAP_FLUX
    if not time.size:
      continue  # No data.

    # Possibly interpolate missing time values.
    if interpolate_missing_time:
      time = util.interpolate_missing_time(time, light_curve.CADENCENO)

    all_time.append(time)
    all_flux.append(flux)
    all_quarters.append(quarter)

  if scramble_type:
    all_time, all_flux = scramble_light_curve(all_time, all_flux, all_quarters, scramble_type)

  return all_time, all_flux, all_centroid
