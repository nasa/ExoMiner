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

# 3rd party
import os.path
import pandas as pd
from astropy.io import fits
import numpy as np
from tensorflow.io import gfile
from astropy import wcs

# local
from src_preprocessing.light_curve import util
# from src_preprocessing.utils_centroid_preprocessing import convertpxtoradec_centr

MOMENTUM_DUMP_VALUE = 32  # momentum dump value in the DQ array
MAX_BIT = 12  # max number of bits in the DQ array
N_TOTAL_QUARTERS = 17  # total number of quarters

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
# SIMULATED_DATA_SCRAMBLE_ORDERS = {
#     "SCR1": [0, 13, 14, 15, 16, 9, 10, 11, 12, 5, 6, 7, 8, 1, 2, 3, 4, 17],
#     "SCR2": [0, 1, 2, 3, 4, 13, 14, 15, 16, 9, 10, 11, 12, 5, 6, 7, 8, 17],
#     "SCR3": [0, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 17],
# }
SIMULATED_DATA_SCRAMBLE_ORDERS = {
    "SCR1": [13, 14, 15, 16, 9, 10, 11, 12, 5, 6, 7, 8, 1, 2, 3, 4, 17],
    "SCR2": [1, 2, 3, 4, 13, 14, 15, 16, 9, 10, 11, 12, 5, 6, 7, 8, 17],
    "SCR3": [16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 17],
}


def kepler_filenames(base_dir,
                     kep_id,
                     long_cadence=True,
                     quarters=None,
                     injected_group=None,
                     check_existence=True):
    """ Returns the light curve filenames for a Kepler target star.

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
    if not injected_group:
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
            if not check_existence or gfile.exists(filename):
                filenames.append(filename)

    return filenames


def scramble_light_curve(all_time, all_flux, all_quarters, scramble_type):
    """ Scrambles a light curve according to a given scrambling procedure.

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


def scramble_light_curve_with_centroids(fluxes, centroids, quarters_target, scramble_type):
    """ Scrambles light curve data according to a given scrambling procedure.

        Args:
        # all_time: List holding arrays of time values, each containing a quarter of
        #   time data.
        fluxes: dict, each key maps to a type of flux data, which is a list of lists containing flux data for each
        quarter
        centroids: dict, each key maps to a type of centroid data, which is a dictionary of row and col centroid, each
        with lists holding arrays of centroid values, each list containing a quarter of centroid data.
        quarters_target: List of integers specifying which quarters are present in the light curve
        (max is 18: Q0...Q17).
        scramble_type: String specifying the scramble order, one of {'SCR1', 'SCR2', 'SCR3'}.

        Returns:
        # scr_time: Time values, re-partitioned to match sizes of the scr_flux lists.
        scr_fluxes: Scrambled flux values; the same dict as the input `fluxes` in another order.
        scr_centroids: Scrambled centroid values; the same dict as the input `centroids` in another order.
        quarter_idx_arr: list, quarter index order after scrambling
    """

    quarter_idx_arr = []  # store new quarter index order
    order = SIMULATED_DATA_SCRAMBLE_ORDERS[scramble_type]
    # initialize data array for scrambled data
    scr_fluxes = {flux_name: [] for flux_name in fluxes}
    scr_centroids = {centroids_name: {'x': [], 'y': []} for centroids_name in centroids}

    for quarter in order:

        if quarter in quarters_target:
            quarter_idx_arr.append(quarters_target.index(quarter))

        for flux_name in fluxes:
            scr_fluxes[flux_name].append(fluxes[flux_name][quarters_target.index(quarter)])
        for centroids_name in centroids:
            scr_centroids[centroids_name]['x'].append((centroids[centroids_name]['x'][quarters_target.index(quarter)]))
            scr_centroids[centroids_name]['y'].append((centroids[centroids_name]['y'][quarters_target.index(quarter)]))

    # scr_time = util.reshard_arrays(all_time, scr_fluxes[list(scr_fluxes)[0]])

    return scr_fluxes, scr_centroids, quarter_idx_arr


def scramble_data(data, cadence_no_quarters_tbl_fp, scramble_type, timeseries_fields, centroid_radec):
    """ Scramble data following a scrambling procedure

    Args:
        data: dict, data
        cadence_no_quarters_tbl_fp: Path, file path to table that, for each quarter, gives information on the first and
        last cadences, and on the number of cadences in the interquarter gap.
        scramble_type: str, one from 'SCR1', 'SCR2', 'SCR2'
        timeseries_fields: list, time series fields' names
        centroid_radec: bool, whether to transform the centroid time series from the CCD module pixel coordinates to RA
        and Dec, or not
    Returns: scrambled data, dict

    """

    scr_data = {
        'all_time': [],
        'all_flux': [],
        'all_centroids': {'x': [], 'y': []},
        'all_centroids_px': {'x': [], 'y': []},
        'flag_keep': [],
    }

    # read table with cadence numbers for each quarter
    interquarter_cadence_tbl = pd.read_csv(cadence_no_quarters_tbl_fp, index_col='quarter')

    total_n_cadences = interquarter_cadence_tbl.loc[1:, 'total_no_cadences_with_interquarter_gap'].sum()
    cadence_no_arr = np.arange(total_n_cadences)
    all_time, all_quarters = [], []
    for quarter in range(1, N_TOTAL_QUARTERS + 1):

        # ncadences_interquarter = int(interquarter_cadence_tbl.loc[quarter, 'interquarter_cadence_gap'])
        total_n_cadences_quarter = interquarter_cadence_tbl.loc[quarter, 'total_no_cadences_with_interquarter_gap']

        if quarter in data['quarter']:  # for the quarters with data
            quarter_idx = data['quarter'].index(quarter)

            n_cadences_data = len(data['all_time'][quarter_idx])
            n_cadences_to_add = total_n_cadences_quarter - n_cadences_data

            # extend time series data arrays with NaNs to account for interquarter gaps
            all_time.append(np.concatenate([data['all_time'][quarter_idx], np.nan * np.ones(n_cadences_to_add)]))

        else:  # quarter not present in the data
            all_time.append(np.nan * np.ones(total_n_cadences_quarter))

        all_quarters.append(quarter * np.ones(len(all_time[-1]), dtype='uint8'))

    # interpolate timestamps array with cadence number array
    all_time, all_quarters = np.concatenate(all_time), np.concatenate(all_quarters)
    all_time_interp = util.interpolate_missing_time(all_time, cadence_no_arr)
    scr_all_quarters = np.concatenate([all_quarters[all_quarters == quarter]
                                       for quarter in SIMULATED_DATA_SCRAMBLE_ORDERS[scramble_type]])

    scr_data['quarter'] = [quarter for quarter in SIMULATED_DATA_SCRAMBLE_ORDERS[scramble_type]
                           if quarter in data['quarter']]
    scr_quarters_order = [data['quarter'].index(quarter) for quarter in scr_data['quarter']]

    # assign new timestamps after scrambling
    scr_data['all_time'] = []
    for quarter in scr_data['quarter']:  # SIMULATED_DATA_SCRAMBLE_ORDERS[scramble_type]:
        # get number of cadences with data (valid)
        ncadences_valid_quarter = interquarter_cadence_tbl.loc[quarter, 'total_no_cadences']
        # ncadences_valid_quarter = len(data['all_flux'][data['quarter'].index(quarter)])
        idxs_quarter = scr_all_quarters == quarter  # get indices for quarter in scrambled array
        # add timestamps for valid cadences for the quarter
        scr_data['all_time'].append(all_time_interp[idxs_quarter][:ncadences_valid_quarter])

    # scramble time series data
    scr_data.update({field: [] for field in ['all_flux', 'flag_keep']})
    scr_data.update({field: {'x': [], 'y': []} for field in ['all_centroids', 'all_centroids_px']})
    for quarter_idx in scr_quarters_order:
        for data_field in [field for field in timeseries_fields if field != 'all_time']:
            if 'centroid' in data_field:
                scr_data[data_field]['x'].append(data[data_field]['x'][quarter_idx])
                scr_data[data_field]['y'].append(data[data_field]['y'][quarter_idx])
            else:
                scr_data[data_field].append(data[data_field][quarter_idx])

    # add NaNs to time series that did not have the expected dimension (number of cadences)
    for quarter_idx in range(len(scr_data['all_time'])):
        n_cadences_data = len(scr_data['all_flux'][quarter_idx])
        n_total_cadences_quarter = len(scr_data['all_time'][quarter_idx])
        n_cadences_to_add = n_total_cadences_quarter - n_cadences_data
        for data_field in [field for field in timeseries_fields if field != 'all_time']:

            if 'centroid' in data_field:
                scr_data[data_field]['x'][quarter_idx] = np.concatenate([scr_data[data_field]['x'][quarter_idx],
                                                                         np.nan * np.ones(n_cadences_to_add)])
                scr_data[data_field]['y'][quarter_idx] = np.concatenate([scr_data[data_field]['y'][quarter_idx],
                                                                         np.nan * np.ones(n_cadences_to_add)])
            elif data_field == 'flag_keep':
                scr_data[data_field][quarter_idx] = np.concatenate([scr_data[data_field][quarter_idx],
                                                                    False * np.ones(n_cadences_to_add, dtype='bool')])
            else:
                scr_data[data_field][quarter_idx] = np.concatenate([scr_data[data_field][quarter_idx],
                                                                    np.nan * np.ones(n_cadences_to_add)])

    # other fields
    scr_data['module'] = np.array(data['module'])[scr_quarters_order]
    if centroid_radec:
        scr_data['target_position_px'] = np.array(data['target_position_px'])[scr_quarters_order]

    return scr_data


def read_kepler_light_curve(filenames,
                            light_curve_extension="LIGHTCURVE",
                            scramble_type=None,
                            cadence_no_quarters_tbl_fp=None,
                            interpolate_missing_time=False,
                            centroid_radec=False,
                            prefer_psfcentr=False,
                            invert=False,
                            dq_values_filter=None,
                            get_momentum_dump=False,
                            # get_px_centr=False
                            ):
    """ Reads data from FITS files for a Kepler target star.

    Args:
        filenames: A list of .fits files containing time and flux measurements.
        light_curve_extension: Name of the HDU 1 extension containing light curves.
        scramble_type: What scrambling procedure to use: 'SCR1', 'SCR2', or 'SCR3'
          (pg 9: https://exoplanetarchive.ipac.caltech.edu/docs/KSCI-19114-002.pdf).
        cadence_no_quarters_tbl_fp: Path, file path to table that, for each quarter, gives information on the first and
        last cadences, and on the number of cadences in the interquarter gap.
        interpolate_missing_time: Whether to interpolate missing (non-finite) time values.
        centroid_radec: bool, whether to transform the centroid time series from the CCD module pixel coordinates to RA
          and Dec, or not
        prefer_psfcentr: bool, if True, uses PSF centroids when available
        invert: bool, if True, inverts time series
        dq_values_filter: list, values (integers) in the data quality flag array for a set of anomalies. Cadences with
        the associated bit active are excluded. See the Kepler documentation on DQ flags for more information. If set
        to `None`, no anomalies in the DQ array are filtered.
        get_momentum_dump: bool, if True the momentum dump information is extracted from the FITS file


    Returns:
        data: dictionary with data extracted from the FITS files
            - all_time: A list of numpy arrays; the time values of the light curve.
            - all_flux: A list of numpy arrays; the flux values of the light curve.
            - all_centroid: A dict, 'x' is a list of numpy arrays with either the col or RA coordinates of the centroid
            values of the light curve; 'y' is a list of numpy arrays with either the row or Dec coordinates.
            - target_position: A list of two elements which correspond to the target star position in world
            coordinates (RA, Dec)
            - target_position_px: A list of two elements for each quarter which correspond to the target star position
            in the CCD pixel frame
            (RA, Dec) or local CCD (x, y) pixel coordinates
            - module: A list with the module IDs
            - quarter: A list with the observed quarters
        files_not_read: list of FITS files that were not read correctly

  """

    # initialize data dict for time series data
    data = {
        'all_time': [],
        'all_flux': [],
        'all_flux_err': [],
        'all_centroids': {'x': [], 'y': []},
        'all_centroids_px': {'x': [], 'y': []},
        'flag_keep': [],
    }
    timeseries_fields = list(data.keys())

    if get_momentum_dump:
        data['momentum_dump'] = []
        data['time_momentum_dump'] = []

    # add fields for additional data
    data.update({
        'quarter': [],
        'module': [],
        'target_position': [],
        'target_position_px': [],
        'quarter_timestamps': {},
    }
    )

    files_not_read = []

    # iterate through the FITS files for the target star
    for filename in filenames:

        with fits.open(filename) as hdu_list:

            # needed to transform coordinates when target is in module 13 and when using pixel coordinates
            quarter = hdu_list["PRIMARY"].header["QUARTER"]
            if quarter == 0:  # ignore quarter 0
                continue
            module = hdu_list["PRIMARY"].header["MODULE"]

            kepid_coord1, kepid_coord2 = hdu_list["PRIMARY"].header["RA_OBJ"], hdu_list["PRIMARY"].header["DEC_OBJ"]

            if len(data['target_position']) == 0:
                data['target_position'] = [kepid_coord1, kepid_coord2]

            if not centroid_radec:
                pass

            light_curve = hdu_list[light_curve_extension].data

            # if _has_finite(light_curve.PSF_CENTR1) and prefer_psfcentr:
            if prefer_psfcentr:
                centroid_x, centroid_y = light_curve.PSF_CENTR1, light_curve.PSF_CENTR2
            else:
                # if _has_finite(light_curve.MOM_CENTR1):
                # correct centroid time series using the POS_CORR (position correction) time series which take into
                # account DVA, pointing drift and thermal transients - based on PSF centroids, so not the best
                # approach to correct systematics in the MOM (flux-weighted) centroids
                centroid_x, centroid_y = light_curve.MOM_CENTR1 - light_curve.POS_CORR1, \
                                         light_curve.MOM_CENTR2 - light_curve.POS_CORR2
                # centroid_x, centroid_y = light_curve.MOM_CENTR1, \
                #                          light_curve.MOM_CENTR2

            centroid_fdl_x, centroid_fdl_y = light_curve.MOM_CENTR1, light_curve.MOM_CENTR2
            # else:
            #     continue  # no data

            # get components required for the transformation from CCD pixel coordinates to world coordinates RA and Dec
            if centroid_radec:
                # transformation matrix from aperture coordinate frame to RA and Dec
                # cd_transform_matrix = np.zeros((2, 2))
                # cd_transform_matrix[0] = hdu_list['APERTURE'].header['PC1_1'] * hdu_list['APERTURE'].header['CDELT1'], \
                #                          hdu_list['APERTURE'].header['PC1_2'] * hdu_list['APERTURE'].header['CDELT1']
                # cd_transform_matrix[1] = hdu_list['APERTURE'].header['PC2_1'] * hdu_list['APERTURE'].header['CDELT2'], \
                #                          hdu_list['APERTURE'].header['PC2_2'] * hdu_list['APERTURE'].header['CDELT2']

                # # reference pixel in the aperture coordinate frame
                # ref_px_apf = np.array([[hdu_list['APERTURE'].header['CRPIX1']],
                #                        [hdu_list['APERTURE'].header['CRPIX2']]])

                # reference pixel in CCD coordinate frame
                ref_px_ccdf = np.array([[hdu_list['APERTURE'].header['CRVAL1P']],
                                        [hdu_list['APERTURE'].header['CRVAL2P']]])

                # # RA and Dec at reference pixel
                # ref_angcoord = np.array([[hdu_list['APERTURE'].header['CRVAL1']],
                #                          [hdu_list['APERTURE'].header['CRVAL2']]])

                # convert from CCD pixel coordinates to world coordinates RA and Dec
        # if centroid_radec:
                # centroid_ra, centroid_dec = convertpxtoradec_centr(centroid_x, centroid_y,
                #                                                    cd_transform_matrix,
                #                                                    ref_px_ccdf,  # np.array([[0], [0]])
                #                                                    ref_px_apf,  # np.array([[0], [0]])
                #                                                    ref_angcoord
                #                                                    )

                w = wcs.WCS(hdu_list['APERTURE'].header)
                pixcrd = np.vstack((centroid_x - ref_px_ccdf[0], centroid_y - ref_px_ccdf[1])).T
                world = w.wcs_pix2world(pixcrd, 1, ra_dec_order=True)
                centroid_x, centroid_y = world[:, 0], world[:, 1]
                target_pos_px = w.wcs_world2pix(np.expand_dims(data['target_position'], 0), 1,
                                                ra_dec_order=True)
                data['target_position_px'].append(target_pos_px)

        # if get_px_centr:
        # required for FDL centroid time-series; center centroid time-series relative to the reference pixel in the
        # aperture
        # 1st attempt at reproducing FDL centroid; they mention centroid relative to the Target Pixel File center,
        # It is not clear what exactly that means
        # centroid_fdl_x = centroid_fdl_x - ref_px_ccdf[0])
        # centroid_fdl_y = centroid_fdl_y - ref_px_ccdf[1])

        # get time and PDC-SAP flux arrays
        time = light_curve.TIME
        flux = light_curve.PDCSAP_FLUX
        flux_err = light_curve.PDCSAP_FLUX_ERR

        if not time.size:
            continue  # No data.

        # check if arrays have the same size
        if not len(time) == len(flux) == len(centroid_x) == len(centroid_y):
            files_not_read.append(filename)
            continue

        data['quarter_timestamps'].update({quarter: [time[0], time[-1]]})

        inds_keep = True * np.ones(len(time), dtype='bool')
        # set indices for missing flux values to false
        inds_keep[np.isnan(flux)] = False  # keep cadences for which the PDC-SAP flux was not gapped

        # use quality flags to exclude cadences
        if dq_values_filter:
            dq_values_filter = dq_values_filter if dq_values_filter else []
            flags = {dq_value: np.binary_repr(dq_value).zfill(MAX_BIT).find('1') for dq_value in dq_values_filter}
            qflags = np.array([np.binary_repr(el).zfill(MAX_BIT) for el in light_curve.SAP_QUALITY])
            inds_keep = True * np.ones(len(qflags), dtype='bool')

            for flag_bit in flags:
                qflags_bit = [el[flags[flag_bit]] == '1' for el in qflags]
                inds_keep[qflags_bit] = False

        if get_momentum_dump:
            if not dq_values_filter:
                qflags = np.array([np.binary_repr(el).zfill(MAX_BIT) for el in light_curve.SAP_QUALITY])
            momentum_dump_bit = np.binary_repr(MOMENTUM_DUMP_VALUE).zfill(MAX_BIT).find('1')
            momentum_dump_arr = np.array([el[momentum_dump_bit] == '1' for el in qflags]).astype('uint')

        # Possibly interpolate missing time values IN EXISTING ARRAY.
        if interpolate_missing_time:
            time = util.interpolate_missing_time(time, light_curve.CADENCENO)

        # add data
        data['all_time'].append(time)
        data['all_flux'].append(flux)
        data['all_flux_err'].append(flux_err)
        data['all_centroids']['x'].append(centroid_x)
        data['all_centroids']['y'].append(centroid_y)
        data['all_centroids_px']['x'].append(centroid_fdl_x)
        data['all_centroids_px']['y'].append(centroid_fdl_y)
        data['quarter'].append(quarter)
        data['module'].append(module)
        data['flag_keep'].append(inds_keep)

        if get_momentum_dump:
            data['momentum_dump'].append(momentum_dump_arr)
            data['time_momentum_dump'].append(np.array(time))

    if scramble_type:  # scrambles data according to the scramble type selected

        scr_data = scramble_data(data, cadence_no_quarters_tbl_fp, scramble_type, timeseries_fields, centroid_radec)

        scr_data['target_position'] = data['target_position']
        scr_data['quarter_timestamps'] = data['quarter_timestamps']

        data = scr_data

    if invert:  # inverts light curve for inverted group runs
        data['all_flux'] = [flux - 2 * np.median(flux) for flux in data['all_flux']]
        data['all_flux'] = [-1 * flux for flux in data['all_flux']]

    # exclude data points based on keep flags
    for arr_i, inds_keep in enumerate(data['flag_keep']):
        for data_field in timeseries_fields:
            if 'centroids' in data_field:
                data[data_field]['x'][arr_i] = data[data_field]['x'][arr_i][inds_keep]
                data[data_field]['y'][arr_i] = data[data_field]['y'][arr_i][inds_keep]
            else:
                data[data_field][arr_i] = data[data_field][arr_i][inds_keep]

    return data, files_not_read
