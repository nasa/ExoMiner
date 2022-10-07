"""
Utility I/O functions for TESS data.
"""

# 3rd party
import os
import numpy as np
from astropy import wcs
from astropy.io import fits
from tensorflow.io import gfile

# local
from src_preprocessing.light_curve import util
# from src_preprocessing.utils_centroid_preprocessing import convertpxtoradec_centr

# mapping sector number to date and id
SECTOR_ID = {1: ("2018206045859", "120"),
             2: ("2018234235059", "121"),
             3: ("2018263035959", "123"),
             4: ("2018292075959", "124"),
             5: ("2018319095959", "125"),
             6: ("2018349182459", "126"),
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
             53: ("202216409574", "226"),
             54: ("2022190063128", "227"),
             55: ("2022217014003", "242"),
             }


def tess_filenames(base_dir,
                   ticid,
                   sectors,
                   check_existence=True):
    """ Returns the light curve filenames for a TESS target star.

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
    base_dir: Base directory containing TESS data
    ticid: Id of the TESS target star. May be an int or a possibly zero-padded string
    sectors: list of observation sector(s)
    # multisector: str, either 'table' or 'no-table'; if 'table', the sectors list defines from which sectors to extract
    # the TCE; if 'no-table', then looks for the target star in all the sectors in `sectors`.
    check_existence: If True, only return filenames corresponding to files that exist

    Returns:
    A list of filepaths to the FITS files for a given TIC and observation sector(s)
    # A string containing all the sectors in which the TCE was observed separated by a space
    """

    # initialize variables
    filenames = []
    # tce_sectors = ''

    # a zero-padded, 16-digit target identifier that refers to an object in the TESS Input Catalog.
    tess_id = str(ticid).zfill(16)

    # if multisector:  # multi-sector run
    #     sectors = [int(sector) for sector in sectors.split(' ')]
    # else:  # single-sector run
    #     sectors = [sectors]

    for sector in sectors:

        sector_timestamp = SECTOR_ID[sector][0]  # timestamp associated with the file (yyyydddhhmmss format)

        # A zero - padded, four - digit identifier of the spacecraft configuration map used to process this data.
        scft_configmapid = SECTOR_ID[sector][1]

        # zero-padded 2-digit integer indicating the sector in which the data were collected
        sector_string = str(sector).zfill(2)

        base_name = f"sector_{sector}/tess{sector_timestamp}-s00{sector_string}-{tess_id}-0{scft_configmapid}-s_lc.fits"
        filename = os.path.join(base_dir, base_name)

        if not check_existence or gfile.exists(filename):
            filenames.append(filename)
            # tce_sectors += '{} '.format(sector)
    # else:
    #     print("File {} does not exist.".format(filename))

    return filenames  # , tce_sectors[:-1]


def read_tess_light_curve(filenames,
                          light_curve_extension="LIGHTCURVE",
                          interpolate_missing_time=False,
                          centroid_radec=False,
                          prefer_psfcentr=False,
                          invert=False):
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
        invert: bool, if True, inverts time series

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

    # initialize data dict
    data = {'all_time': [],
            'all_flux': [],
            'all_centroids': {'x': [], 'y': []},
            'all_centroids_px': {'x': [], 'y': []},
            'sectors': [],
            'target_position': [],
            'camera': [],
            'ccd': []}

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
                    cd_transform_matrix = np.zeros((2, 2))
                    cd_transform_matrix[0] = hdu_list['APERTURE'].header['PC1_1'] * hdu_list['APERTURE'].header[
                        'CDELT1'], \
                                             hdu_list['APERTURE'].header['PC1_2'] * hdu_list['APERTURE'].header[
                                                 'CDELT1']
                    cd_transform_matrix[1] = hdu_list['APERTURE'].header['PC2_1'] * hdu_list['APERTURE'].header[
                        'CDELT2'], \
                                             hdu_list['APERTURE'].header['PC2_2'] * hdu_list['APERTURE'].header[
                                                 'CDELT2']

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
                world = w.wcs_pix2world(pixcrd, 0, ra_dec_order=False)
                # RA and Dec centroids
                centroid_x, centroid_y = world[:, 0], world[:, 1]

            time = light_curve.TIME
            flux = light_curve.PDCSAP_FLUX

            if not time.size:
                files_not_read.append((basename, 'No data available.'))
                continue  # No data.

            # check if arrays have the same size
            if not len(time) == len(flux) == len(centroid_x) == len(centroid_y):
                files_not_read.append((basename, f'Time series do not have the same size (Timestamps {time.size}, '
                                                 f'PDC flux {flux.size}, '
                                                 f'FW centroid {centroid_x.size}|{centroid_y.size}).'))
                continue

            # use quality flags to remove cadences
            MAX_BIT = 16
            BITS = []  # [2048, 4096, 32768]
            flags = {bit: np.binary_repr(bit).zfill(MAX_BIT).find('1') for bit in BITS}
            qflags = np.array([np.binary_repr(el).zfill(MAX_BIT) for el in light_curve.QUALITY])
            inds_keep = True * np.ones(len(qflags), dtype='bool')

            for flag_bit in flags:
                qflags_bit = [el[flags[flag_bit]] == '1' for el in qflags]
                inds_keep[qflags_bit] = False

            inds_keep[np.isnan(flux)] = False  # keep cadences for which the PDC-SAP flux was not gapped
            time = time[inds_keep]
            flux = flux[inds_keep]
            centroid_x = centroid_x[inds_keep]
            centroid_y = centroid_y[inds_keep]
            centroid_fdl_x = centroid_fdl_x[inds_keep]
            centroid_fdl_y = centroid_fdl_y[inds_keep]

            # Possibly interpolate missing time values.
            if interpolate_missing_time:
                time = util.interpolate_missing_time(time, light_curve.CADENCENO)

            data['all_time'].append(time)
            data['all_flux'].append(flux)
            data['all_centroids']['x'].append(centroid_x)
            data['all_centroids']['y'].append(centroid_y)

            data['all_centroids_px']['x'].append(centroid_fdl_x)
            data['all_centroids_px']['y'].append(centroid_fdl_y)

            data['camera'].append(camera)
            data['ccd'].append(ccd)
            data['sectors'].append(sector)

        except:
            files_not_read.append((basename, 'FITS file not read correctly.'))

    # inverts light curve
    if invert:
        data['all_flux'] = [flux - 2 * np.median(flux) for flux in data['all_flux']]
        data['all_flux'] = [-1 * flux for flux in data['all_flux']]

    return data, files_not_read
