"""
Utility I/O functions for TESS data.
"""

# 3rd party
import os
import numpy as np
from astropy import wcs
from astropy.io import fits
from tensorflow.io import gfile
from pathlib import Path

# local
from src_preprocessing.light_curve import util
# from src_preprocessing.utils_centroid_preprocessing import convertpxtoradec_centr


MOMENTUM_DUMP_VALUE = 32  # momentum dump value in the DQ array
MAX_BIT = 12  # max number of bits in the DQ array

# mapping sector number to date and id
SECTOR_ID = {1: ("2018206045859", "120"),
             2: ("2018234235059", "121"),
             3: ("2018263035959", "123"),
             4: ("2018292075959", "124"),
             5: ("2018319095959", "125"),
             6: ("2018349182500", "126"),
             7: ("2019006130736", "131"),
             8: ("2019032160000", "136"),
             9: ("2019058134432", "139"),
             10: ("2019085135100", "140"),
             11: ("2019112060037", "143"),
             12: ("2019140104343", "144"),
             13: ("2019169103026", "146"),
             14: ("2019198215352", "150"),
             15: ("2019226182529", "151"),
             16: ("2019253231442", "152"),
             17: ("2019279210107", "161"),
             18: ("2019306063752", "162"),
             19: ("2019331140908", "164"),
             20: ("2019357164649", "165"),
             21: ("2020020091053", "167"),
             22: ("2020049080258", "174"),
             23: ("2020078014623", "177"),
             24: ("2020106103520", "180"),
             25: ("2020133194932", "182"),
             26: ("2020160202036", "188"),
             27: ("2020186164531", "189"),
             28: ("2020212050318", "190"),
             29: ("2020238165205", "193"),
             30: ("2020266004630", "195"),
             31: ("2020294194027", "198"),
             32: ("2020324010417", "200"),
             33: ("2020351194500", "203"),
             34: ("2021014023720", "204"),
             35: ("2021039152502", "205"),
             36: ("2021065132309", "207"),
             37: ("2021091135823", "208"),
             38: ("2021118034608", "209"),
             39: ("2021146024351", "210"),
             40: ("2021175071901", "211"),
             41: ("2021204101404", "212"),
             42: ("2021232031932", "213"),
             43: ("2021258175143", "214"),
             44: ("2021284114741", "215"),
             45: ("2021310001228", "216"),
             46: ("2021336043614", "217"),
             47: ("2021364111932", "218"),
             48: ("2022027120115", "219"),
             49: ("2022057073128", "221"),
             50: ("2022085151738", "222"),
             51: ("2022112184951", "223"),
             52: ("2022138205153", "224"),
             53: ("2022164095748", "226"),
             54: ("2022190063128", "227"),
             55: ("2022217014003", "242"),
             56: ("2022244194134", "243"),
             57: ("2022273165103", "245"),
             58: ("2022302161335", "247"),
             59: ("2022330142927", "248"),
             60: ("2022357055054", "249"),
             61: ("2023018032328", "250"),
             62: ("2023043185947", "254"),
             63: ("2023069172124", "255"),
             64: ("2023096110322", "257"),
             65: ("2023124020739", "259"),
             66: ("2023153011303", "260"),
             67: ("2023181235917", "261"),
             68: ("", ""),
             69: ("", ""),
             70: ("", ""),
             71: ("", ""),
             72: ("", ""),
             73: ("2023341045131", "268"),
             74: ("2024003055635", "269"),
             75: ("024030031500", "270"),
             76: ("2024058030222", "271"),
             77: ("2024085201119", "272"),
             78: ("2024114025118", "273"),
             79: ("2024142205832", "274"),
             80: ("2024170053053", "275"),
             81: ("2024196212429", "276"),
             82: ("", ""),
             83: ("2024249191853", "280"),
             84: ("2024274222008", "281"),
             85: ("2024300212641", "282"),
             }


def tess_filenames(base_dir, ticid, sectors):
    """ Returns the light curve filenames for a TESS target star in 2-min cadence data.

    This function assumes the file structure of the Mikulski Archive for Space Telescopes
    (https://archive.stsci.edu/missions-and-data/transiting-exoplanet-survey-satellite-tess/data-products.html).
    Specifically, the filenames for a particular TESS target star have the following format:

    sector_${sector}/tess${sector_obs_date}-s00${sector}-${tic_id}-0${scid}-s_lc.fits,

    where:
        sector is the observed sector;
        sector_obs_date is the timestamp associated with the file (yyyydddhhmmss format);
        tic_id is target identifier that refers to an object in the TESS Input Catalog;
        scid is the identifier of the spacecraft configuration map used to process this data;
        s denotes the cosmic ray mitigation procedure performed on the spacecraft

    Args:
    base_dir: Base directory containing TESS 2-min cadence data
    ticid: ID of the TESS target star. It can be an int or a possibly zero-padded string
    sectors: list of observation sector(s)

    Returns:
    A list of filepaths to the FITS files for a given TIC and observation sector(s)
    """

    # initialize variables
    filenames = []

    # # a zero-padded, 16-digit target identifier that refers to an object in the TESS Input Catalog.
    # tess_id = str(ticid).zfill(16)

    for sector in sectors:

        sector_dir = Path(base_dir) / f'sector_{sector}'
        fps_lst = list(sector_dir.glob(f'*{ticid}*lc.fits'))
        if len(fps_lst) == 0:
            filename = None
        else:
            filename = fps_lst[0]

        # sector_timestamp = SECTOR_ID[sector][0]  # timestamp associated with the file (yyyydddhhmmss format)
        #
        # # a zero - padded, four - digit identifier of the spacecraft configuration map used to process this data.
        # scft_configmapid = SECTOR_ID[sector][1]
        #
        # # zero-padded 2-digit integer indicating the sector in which the data were collected
        # sector_string = str(sector).zfill(2)
        #
        # base_name = f"sector_{sector}/tess{sector_timestamp}-s00{sector_string}-{tess_id}-0{scft_configmapid}-s_lc.fits"
        # filename = os.path.join(base_dir, base_name)
        # # lc_dir = Path(f'{base_dir}/sector_{sector}')
        # # filename = list(lc_dir.glob(f"tess*-s0[0-9][0-9][0-9]-{tess_id}*"))[0]

        # if not check_existence or gfile.exists(filename):
        if filename:
            filenames.append(filename)

    return filenames


def tess_ffi_filenames(base_dir, tic_id, sector_run, check_existence=True):
    """ Returns the light curve filenames for a TESS target star observed in the TESS SPOC FFI data.

        This function assumes the filenames for a particular TESS target star have the following format:

        ${tic_id:0:4}/${tic_id:4:8}/tic_id:8:12}/tic_id:12:16}/
            hlsp_tess-spoc_tess_phot_${tic_id}-$s{sector_id}_tess_v1_lc.fits,

        where:
            tic_id is the TESS id left-padded with zeros to length 16;
            sector_id is the filename sector run left-padded with zeros to length 4;

        Args:
        base_dir: str, base directory containing TESS SPOC FFI data.
        tic_id: int, id of the TIC target star. It may be an int or a possibly zero-padded string.
        sector_run: list of ints, for all the sector runs the TIC was observed.
        check_existence: If True, only return filenames corresponding to files that exist (not all stars have data for
        all sector runs).

        Returns:
        A list of filenames.
    """

    # pad the TIC id with zeros to length 16
    tic_id = f'{tic_id}'.zfill(16)
    # pad the sector runs ids with zeros to length 4
    sector_runs_ids = [f'{sector_i}'.zfill(4) for sector_i in sector_run]

    filenames = []
    for sector_run_id in sector_runs_ids:
        base_dir_sector_run = os.path.join(base_dir, f's{sector_run_id}', 'target', tic_id[0:4], tic_id[4:8],
                                           tic_id[8:12], tic_id[12:16])
        base_name = f'hlsp_tess-spoc_tess_phot_{tic_id}-s{sector_run_id}_tess_v1_lc.fits'
        filename = os.path.join(base_dir_sector_run, base_name)
        # not all stars have data for all sector runs
        if not check_existence or gfile.exists(filename):
            filenames.append(filename)

    return filenames


def read_tess_light_curve(filenames,
                          light_curve_extension="LIGHTCURVE",
                          interpolate_missing_time=False,
                          centroid_radec=False,
                          prefer_psfcentr=False,
                          get_momentum_dump=False,
                          dq_values_filter=None,
                          ):
    """ Reads data from FITS files for a TESS target star.

    Args:
        filenames: A list of .fits files containing time and flux measurements.
        light_curve_extension: Name of the HDU 1 extension containing light curves.
        interpolate_missing_time: Whether to interpolate missing (NaN) time values.
          This should only affect the output if scramble_type is specified (NaN time
          values typically come with NaN flux values, which are removed anyway, but
          scrambing decouples NaN time values from NaN flux values).
        centroid_radec: bool, whether to transform the centroid time series from the CCD module pixel coordinates to RA
          and Dec, or not
        prefer_psfcentr: bool, if True, uses PSF centroids when available
        get_momentum_dump: bool, if True the momentum dump information is extracted from the FITS file
        dq_values_filter: list, values (integers) in the data quality flag array for a set of anomalies. Cadences with
        the associated bit active are excluded. See 'TESS Science Data Products Description Document' for more
        information. If set to `None`, no anomalies in the DQ array are filtered.

    Returns:
        data: dictionary with data extracted from the FITS files
            - all_time: A list of numpy arrays; the time values of the light curve.
            - all_flux: A list of numpy arrays; the flux values of the light curve.
            - all_centroid: A dict, 'x' is a list of numpy arrays with either the col or RA coordinates of the centroid
            values of the light curve; 'y' is a list of numpy arrays with either the row or Dec coordinates.
            - sectors: A list with the observation sectors
            - target_postiion: A list of two elements which correspond to the target star position, either in world
            (RA, Dec) or local CCD (x, y) pixel coordinates
            - camera: A list with the camera IDs
            - ccd: A list with the CCD IDs
        files_not_read: list with file paths for FITS files that were not read correctly
    """

    # initialize data dict for time series data
    data = {
        'all_time': [],
        'all_flux': [],
        'all_flux_err': [],
        'all_centroids': {'x': [], 'y': []},
        'all_centroids_px': {'x': [], 'y': []},
    }

    timeseries_fields = list(data.keys())

    if get_momentum_dump:
        data['momentum_dump'] = []
        data['time_momentum_dump'] = []

    # add fields for additional data
    data.update({
        'flag_keep': [],
        'sectors': [],
        'module': [],
        'camera': [],
        'ccd': [],
        'target_position': [],
    }
    )

    files_not_read = []

    # iterate through the FITS files for the target star
    for filename in filenames:

        basename = os.path.basename(filename)

        try:
            with fits.open(filename, ignoring_missing_end=True) as hdu_list:

                camera = hdu_list["PRIMARY"].header["CAMERA"]
                ccd = hdu_list["PRIMARY"].header["CCD"]
                sector = hdu_list["PRIMARY"].header["SECTOR"]

                if len(data['target_position']) == 0:
                    data['target_position'] = [hdu_list["PRIMARY"].header["RA_OBJ"],
                                               hdu_list["PRIMARY"].header["DEC_OBJ"]]

                # TODO: convert target position from RA and Dec to local CCD pixel coordinates
                if not centroid_radec:
                    pass

                light_curve = hdu_list[light_curve_extension].data

                if prefer_psfcentr:
                    centroid_x, centroid_y = light_curve.PSF_CENTR1, light_curve.PSF_CENTR2
                else:
                    # if _has_finite(light_curve.MOM_CENTR1):
                    centroid_x, centroid_y = light_curve.MOM_CENTR1 - light_curve.POS_CORR1, \
                                             light_curve.MOM_CENTR2 - light_curve.POS_CORR2

                centroid_fdl_x, centroid_fdl_y = light_curve.MOM_CENTR1, light_curve.MOM_CENTR2
                # else:
                #     continue  # no data

                # get components required for the transformation from CCD pixel coordinates to world coordinates RA and
                # Dec
                if centroid_radec:
                    # transformation matrix from aperture coordinate frame to RA and Dec
                    # cd_transform_matrix = np.zeros((2, 2))
                    # cd_transform_matrix[0] = hdu_list['APERTURE'].header['PC1_1'] * hdu_list['APERTURE'].header[
                    #     'CDELT1'], \
                    #                          hdu_list['APERTURE'].header['PC1_2'] * hdu_list['APERTURE'].header[
                    #                              'CDELT1']
                    # cd_transform_matrix[1] = hdu_list['APERTURE'].header['PC2_1'] * hdu_list['APERTURE'].header[
                    #     'CDELT2'], \
                    #                          hdu_list['APERTURE'].header['PC2_2'] * hdu_list['APERTURE'].header[
                    #                              'CDELT2']

                    # # reference pixel in the aperture coordinate frame
                    # ref_px_apf = np.array([[hdu_list['APERTURE'].header['CRPIX1']], [hdu_list['APERTURE'].header['CRPIX2']]])

                    # reference pixel in CCD coordinate frame
                    ref_px_ccdf = np.array([[hdu_list['APERTURE'].header['CRVAL1P']],
                                            [hdu_list['APERTURE'].header['CRVAL2P']]])

                    # # RA and Dec at reference pixel
                    # ref_angcoord = np.array([[hdu_list['APERTURE'].header['CRVAL1']],
                    #                          [hdu_list['APERTURE'].header['CRVAL2']]])

            # convert from CCD pixel coordinates to world coordinates RA and Dec
            if centroid_radec:
                # centroid_x, centroid_y = convertpxtoradec_centr(centroid_x,
                #                                                 centroid_y,
                #                                                 cd_transform_matrix,
                #                                                 ref_px_apert,
                #                                                 ref_angcoord
                #                                                 )

                w = wcs.WCS(hdu_list['APERTURE'].header)
                pixcrd = np.vstack((centroid_x - ref_px_ccdf[0], centroid_y - ref_px_ccdf[1])).T
                world = w.wcs_pix2world(pixcrd, 0, ra_dec_order=True)
                # RA and Dec centroids
                centroid_x, centroid_y = world[:, 0], world[:, 1]

            time = light_curve.TIME
            flux = light_curve.PDCSAP_FLUX
            flux_err = light_curve.PDCSAP_FLUX_ERR

            if not time.size:
                files_not_read.append((basename, 'No data available.'))
                continue  # No data.

            # check if arrays have the same size
            if not len(time) == len(flux) == len(centroid_x) == len(centroid_y):
                files_not_read.append((basename, f'Time series do not have the same size (Timestamps {time.size}, '
                                                 f'PDC flux {flux.size}, '
                                                 f'FW centroid {centroid_x.size}|{centroid_y.size}).'))
                continue

            inds_keep = True * np.ones(len(flux), dtype='bool')
            inds_keep[np.isnan(flux)] = False  # exclude cadences that are NaN in the PDCSAP flux

            # use quality flags to exclude cadences
            if dq_values_filter:
                dq_values_filter = dq_values_filter if dq_values_filter else []  # [2048, 4096, 32768]
                flags = {dq_value: np.binary_repr(dq_value).zfill(MAX_BIT).find('1') for dq_value in dq_values_filter}
                qflags = np.array([np.binary_repr(el).zfill(MAX_BIT) for el in light_curve.QUALITY])
                for flag_bit in flags:  # set cadences to be excluded based on selected dq flags
                    qflags_bit = [el[flags[flag_bit]] == '1' for el in qflags]
                    inds_keep[qflags_bit] = False

            if get_momentum_dump:
                if not dq_values_filter:
                    qflags = np.array([np.binary_repr(el).zfill(MAX_BIT) for el in light_curve.QUALITY])
                momentum_dump_bit = np.binary_repr(MOMENTUM_DUMP_VALUE).zfill(MAX_BIT).find('1')
                momentum_dump_arr = np.array([el[momentum_dump_bit] == '1' for el in qflags]).astype('uint')

            # Possibly interpolate missing time values.
            if interpolate_missing_time:
                time = util.interpolate_missing_time(time, light_curve.CADENCENO)

            data['all_time'].append(time)
            data['all_flux'].append(flux)
            data['all_flux_err'].append(flux_err)
            data['all_centroids']['x'].append(centroid_x)
            data['all_centroids']['y'].append(centroid_y)
            data['all_centroids_px']['x'].append(centroid_fdl_x)
            data['all_centroids_px']['y'].append(centroid_fdl_y)
            data['camera'].append(camera)
            data['ccd'].append(ccd)
            data['sectors'].append(sector)
            data['flag_keep'].append(inds_keep)

            if get_momentum_dump:
                data['momentum_dump'].append(momentum_dump_arr)
                data['time_momentum_dump'].append(np.array(time))

        except:
            files_not_read.append((basename, 'FITS file not read correctly.'))

    # exclude data points based on keep flags
    for arr_i, inds_keep in enumerate(data['flag_keep']):
        for data_field in timeseries_fields:
            if 'centroids' in data_field:
                data[data_field]['x'][arr_i] = data[data_field]['x'][arr_i][inds_keep]
                data[data_field]['y'][arr_i] = data[data_field]['y'][arr_i][inds_keep]
            else:
                data[data_field][arr_i] = data[data_field][arr_i][inds_keep]

    return data, files_not_read
