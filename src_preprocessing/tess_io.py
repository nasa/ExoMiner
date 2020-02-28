"""
Utility I/O functions for TESS data.
"""

# 3rd party
import os
from tensorflow import gfile
from astropy.io import fits
import numpy as np

# local
from src_preprocessing.light_curve import util
from src_preprocessing.utils_centroid_preprocessing import convertpxtoradec_centr

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
             18: ("2019306063752", "162")}


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
    base_dir: Base directory containing Kepler data
    ticid: Id of the TESS target star. May be an int or a possibly zero-padded string
    sectors: list of observation sector(s)
    # multisector: str, either 'table' or 'no-table'; if 'table', the sectors list defines from which sectors to extract
    the TCE; if 'no-table', then looks for the target star in all the sectors in `sectors`.
    check_existence: If True, only return filenames corresponding to files that exist

    Returns:
    A list of filepaths to the FITS files for a given TIC and observation sector(s)
    A string containing all the sectors in which the TCE was observed separated by a space
    """

    # initialize variables
    filenames = []
    tce_sectors = ''

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

        # TODO: why do we want the check_existence flag?
        if not check_existence or gfile.Exists(filename):
            filenames.append(filename)
            tce_sectors += '{} '.format(sector)
    # else:
    #     print("File {} does not exist.".format(filename))

    return filenames, tce_sectors[:-1]


def read_tess_light_curve(filenames,
                          light_curve_extension="LIGHTCURVE",
                          interpolate_missing_time=False,
                          centroid_radec=False):
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

  Returns:
    all_time: A list of numpy arrays; the time values of the light curve.
    all_flux: A list of numpy arrays; the flux values of the light curve.
    all_centroid: A dict, 'x' is a list of numpy arrays with either the col or RA coordinates of the centroid values of
    the light curve; 'y' is a list of numpy arrays with either the row or Dec coordinates.
    add_info: A dict with additional data extracted from the FITS files; 'module' is the same but for the module in
    which the target is in every quarter; 'target position' is a list of two elements which correspond to the target
    star position, either in world (RA, Dec) or local CCD (x, y) pixel coordinates
  """

    # initialize variables
    all_time = []
    all_flux = []
    all_centroid = {'x': [], 'y': []}
    add_info = {'camera': [], 'ccd': [], 'sector': [], 'target position': []}

    def _has_finite(array):
        for i in array:
            if np.isfinite(i):
                return True

        return False

    # iterate through the FITS files for the target star
    for filename in filenames:
        with fits.open(gfile.Open(filename, "rb")) as hdu_list:

            add_info['camera'].append(hdu_list["PRIMARY"].header["CAMERA"])
            add_info['ccd'].append(hdu_list["PRIMARY"].header["CCD"])
            add_info['sector'].append(hdu_list["PRIMARY"].header["SECTOR"])

            if len(add_info['target position']) == 0:
                add_info['target position'] = [hdu_list["PRIMARY"].header["RA_OBJ"],
                                               hdu_list["PRIMARY"].header["DEC_OBJ"]]

                # TODO: convert target position from RA and Dec to local CCD pixel coordinates
                if not centroid_radec:
                    pass

            light_curve = hdu_list[light_curve_extension].data

            if _has_finite(light_curve.PSF_CENTR1):
                centroid_x, centroid_y = light_curve.PSF_CENTR1, light_curve.PSF_CENTR2
            else:
                if _has_finite(light_curve.MOM_CENTR1):
                    centroid_x, centroid_y = light_curve.MOM_CENTR1, light_curve.MOM_CENTR2
                else:
                    continue  # no data

            # get components required for the transformation from CCD pixel coordinates to world coordinates RA and Dec
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
                # ref_px = np.array([[hdu_list['APERTURE'].header['CRPIX1']], [hdu_list['APERTURE'].header['CRPIX2']]])

                # reference pixel in CCD coordinate frame
                ref_px_apert = np.array([[hdu_list['APERTURE'].header['CRVAL1P']],
                                         [hdu_list['APERTURE'].header['CRVAL2P']]])

                # RA and Dec at reference pixel
                ref_angcoord = np.array([[hdu_list['APERTURE'].header['CRVAL1']],
                                         [hdu_list['APERTURE'].header['CRVAL2']]])

        # convert from CCD pixel coordinates to world coordinates RA and Dec
        if centroid_radec:
            centroid_x, centroid_y = convertpxtoradec_centr(centroid_x, centroid_y, cd_transform_matrix,
                                                            ref_px_apert,
                                                            ref_angcoord, 'tess')

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

    # TODO: adapt this to the centroid time series as well?
    #if scramble_type:
     #   all_time, all_flux = scramble_light_curve(all_time, all_flux, all_quarters, scramble_type)

    return all_time, all_flux, all_centroid, add_info
