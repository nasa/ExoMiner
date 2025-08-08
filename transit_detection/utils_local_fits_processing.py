from pathlib import Path
import lightkurve as lk
from astropy.io import fits
from typing import Union
import numpy as np
from astropy.table import Table
import warnings
from astropy.units import UnitsWarning


def fill_missing_tess_lc_timestamps(
    time: np.array, strategy: str = ""
) -> tuple[np.array, np.array]:
    """
    Given a time series from a TESS light curve file, where missing cadences are represented with nans,
    interpolates between start and end times. If the time series has missing edges, start and times
    are extrapolated using the provided strategy.
    Args:
        time: NumPy array of floats, timestamps
        strategy: str: defining strategy to compute cadence for start/end time extrapolation.
                    'average': finds the average difference between cadences, ignoring gaps
                    'median': finds the median difference between cadences, ignoring gaps
                    '' or other: defaults to expected 2 min cadence for TESS mission
    Returns:
        time_filled: NumPy array of floats, with missing cadences filled
        missing_cadence_mask: Numpy array of bools, with originally missing cadences marked as True
    """

    if not isinstance(time, np.ndarray):
        time = np.array(time)

    missing_cadence_mask = ~np.isfinite(time)

    n_cadences = len(time)
    valid_idxs = np.where(np.isfinite(time))[0]  # returns actual idxs, not mask

    if n_cadences == len(valid_idxs):
        return time, np.zeros_like(time).astype(bool)  # time has no missing values

    if strategy == "average":
        cadence = np.nanmean(np.diff(time[valid_idxs]))
    elif strategy == "median":
        cadence = np.nanmedian(np.diff(time[valid_idxs]))
    else:
        cadence = 2 / 1440  # 2mins in days for TESS

    # if first/last idx are valid, use directly else offset by cadnece * idx dist to start/end
    t_start = time[valid_idxs][0] - valid_idxs[0] * cadence
    t_end = time[valid_idxs][-1] + ((n_cadences - 1) - valid_idxs[-1]) * cadence

    time_filled = np.linspace(t_start, t_end, n_cadences)

    return time_filled, missing_cadence_mask


def search_and_read_lcfs_and_tpfs(
    target: Union[int, str],
    sectors: list[int, str],
    lcf_dir: str,
    tpf_dir: str,
):
    """
    Searches for locally stored lightcurve and targetpixel fits files for a given target_id and
    list of sectors. Only returns info for sectors in which both the lcf and tpf are valid/found.

        Arguments:
            target: int or str, specifying target star tic_id.
            sectors: List of ints or strs, of sectors to download.
            tpf_dir: str, of directory with tess target pixel file, in mast e (.fits)
            lcf_dir: str, of directory with tess light curve file, in mast e (.fits)
        Returns:
            found_sectors: List [int] of found sectors
            found_lcfs: List [lk.LightCurve objs] of valid light curve file objects
            found_tpfs: List [lk.TargetPixelFile objs] of valid target pixel file objects
    """
    # Suppress all astropy unit warnings ( not using units for conversions directly)
    warnings.filterwarnings("ignore", category=UnitsWarning)

    tpf_sector_paths = (
        [Path(f"{tpf_dir}/sector_{sector}") for sector in sectors] if sectors else []
    )
    lcf_sector_paths = (
        [Path(f"{lcf_dir}/sector_{sector}") for sector in sectors] if sectors else []
    )

    found_sectors = []
    found_lcfs = []
    found_tpfs = []

    for sector, lcf_sector_path, tpf_sector_path in zip(
        sectors, lcf_sector_paths, tpf_sector_paths
    ):
        try:
            lcf_fps = list(lcf_sector_path.rglob(f"*{str(target).zfill(16)}*lc.fits"))
            if not lcf_fps:  # target not found in sector
                continue
            tpf_fps = list(tpf_sector_path.rglob(f"*{str(target).zfill(16)}*tp.fits"))
            if not tpf_fps:  # target not found in sector
                continue

            lcf_fp = lcf_fps[0]  # should only be 1 instance
            tpf_fp = tpf_fps[0]  # should only be 1 instance

            lcf = read_tess_lcf_with_astropy_table(lcf_fp)
            tpf = lk.read(tpf_fp)

            found_lcfs.append(lcf)
            found_tpfs.append(tpf)

            found_sectors.append(int(sector))
        except fits.VerifyError as e:
            print(f"ERROR: Corrupted fits file - {e.args[0]}")
            continue
        except Exception as e:
            print(f"ERROR: Unexpected exception while reading file - {e}")
            continue
    return found_sectors, found_lcfs, found_tpfs


# def search_and_read_tess_tpfs_with_lk(
#     target: Union[int, str], sectors: list[int, str], tpf_dir: str
# ):
#     """
#     Searches lcf_dir in the format of: lcf_dir/
#                                                 sector_1/
#                                                         *tic_id*_tp.fits
#                                                 sector_2/
#                                              ...sector_n/
#     for a given target tic_id fits file and reads it for
#         Arguments:
#             target: int or str, specifying target star tic_id.
#             sectors: List of ints or strs, of sectors to download.
#             tpf_dir: str, of directory with tess target pixel file, in mast e (.fits)
#         Returns:
#             found_sectors: List of found sectors
#             light_curve_files: List of valid light curve files correponding to found_sectors
#     """

#     tpf_sector_paths = (
#         [Path(f"{tpf_dir}/sector_{sector}") for sector in sectors] if sectors else []
#     )
#     lcf_sector_paths = (
#         [Path(f"{lcf_dir}/sector_{sector}") for sector in sectors] if sectors else []
#     )

#     found_sectors = []
#     found_lcfs = []
#     found_tpfs = []

#     for sector, lcf_sector_path, tpf_sector_path in zip(
#         sectors, lcf_sector_paths, tpf_sector_paths
#     ):
#         try:
#             tpf_fps = list(tpf_sector_path.rglob(f"*{str(target).zfill(16)}*tp.fits"))
#             if not tpf_fps:  # target not found in sector
#                 continue

#             lcf_fps = list(lcf_sector_path.rglob(f"*{str(target).zfill(16)}*lc.fits"))
#             if not lcf_fps:  # target not found in sector
#                 continue

#             tpf_fp = tpf_fps[0]  # should only be 1 instance
#             lcf_fp = lcf_fps[0]  # should only be 1 instance

#             tpf = lk.read(tpf_fp)
#             lcf = read_tess_lcf_with_astropy_table(lcf_fp)

#             found_tpfs.append(tpf)
#             found_lcfs.append(lcf)

#             found_sectors.append(sector)

#         except fits.VerifyError as e:
#             print(f"ERROR: Corrupted fits file - {e}")
#             continue
#         except Exception as e:
#             print(f"ERROR: While reading lcf - {e}")
#             continue
#     return found_sectors, tpfs


def read_tess_lcf_with_astropy_table(
    lcf_fp: str,
    quality_bitmask="default",
):
    """
    Searches lcf_dir in the format of: lcf_dir/
                                                sector_1/
                                                        *tic_id*_lc.fits
                                                sector_2/
                                             ...sector_n/
    for a given target tic_id fits file and reads it for
        Arguments:
            target: int or str, specifying target star tic_id.
            sectors: List of ints or strs, of sectors to download.
            lcf_dir: str, of directory with tess lightcurve data (.fits)
        Returns:
            found_sectors: List of found sectors
            lcs: List of valid light curve objects, from lcfs corresponding to found_sectors
    """

    # -------- CODE pipelined from https://lightkurve.github.io/lightkurve/_modules/lightkurve/utils.html#TessQualityFlags -------- #

    AttitudeTweak = 1
    SafeMode = 2
    CoarsePoint = 4
    EarthPoint = 8
    Argabrightening = 16
    Desat = 32
    ApertureCosmic = 64
    ManualExclude = 128
    Discontinuity = 256
    ImpulsiveOutlier = 512
    CollateralCosmic = 1024
    #: The first stray light flag is set manually by MIT based on visual inspection.
    Straylight = 2048
    #: The second stray light flag is set automatically by Ames/SPOC based on background level thresholds.
    Straylight2 = 4096
    # See TESS Science Data Products Description Document
    PlanetSearchExclude = 8192
    BadCalibrationExclude = 16384
    # Set in the sector 20 data release notes
    InsufficientTargets = 32768

    #: DEFAULT bitmask identifies all cadences which are definitely useless.
    # See https://outerspace.stsci.edu/display/TESS/2.0+-+Data+Product+Overview
    DEFAULT_BITMASK = (
        AttitudeTweak
        | SafeMode
        | CoarsePoint
        | EarthPoint
        | Argabrightening
        | Desat
        | ManualExclude
        | ImpulsiveOutlier
        | BadCalibrationExclude
    )
    #: HARD bitmask is conservative and may identify cadences which are useful.
    HARD_BITMASK = (
        DEFAULT_BITMASK | ApertureCosmic | CollateralCosmic | Straylight | Straylight2
    )
    #: HARDEST bitmask identifies cadences with any flag set. Its use is not recommended.
    HARDEST_BITMASK = 65535

    #: Dictionary which provides friendly names for the various bitmasks.
    OPTIONS = {
        "none": 0,
        "default": DEFAULT_BITMASK,
        "hard": HARD_BITMASK,
        "hardest": HARDEST_BITMASK,
    }
    # -------- END CODE pipelined from https://lightkurve.github.io/lightkurve/_modules/lightkurve/utils.html#TessQualityFlags -------- #
    table = Table.read(lcf_fp, format="fits")

    # masked vals replaced w/ nan
    time = table["TIME"].value.filled(np.nan)
    flux = table["PDCSAP_FLUX"].value.filled(np.nan)
    quality = table["QUALITY"].value

    assert (
        len(time) == len(flux) == len(quality)
    ), "ERROR: expected time, flux, quality columns to be eq length from table"

    time, cadence_mask = fill_missing_tess_lc_timestamps(time, strategy="average")

    valid_quality_mask = (OPTIONS[quality_bitmask] & quality) == 0
    valid_flux_mask = np.isfinite(flux)

    cadence_mask |= ~valid_quality_mask
    cadence_mask |= ~valid_flux_mask

    flux[cadence_mask] = np.nan

    lc = lk.LightCurve({"time": time, "flux": flux})

    return lc


# def search_and_read_tess_lcfs_with_astropy_table(
#     target: Union[int, str],
#     sectors: list[int, str],
#     lcf_dir: str,
#     quality_bitmask="default",
# ) -> list:
#     """
#     Searches lcf_dir in the format of: lcf_dir/
#                                                 sector_1/
#                                                         *tic_id*_lc.fits
#                                                 sector_2/
#                                              ...sector_n/
#     for a given target tic_id fits file and reads it for
#         Arguments:
#             target: int or str, specifying target star tic_id.
#             sectors: List of ints or strs, of sectors to download.
#             lcf_dir: str, of directory with tess lightcurve data (.fits)
#         Returns:
#             found_sectors: List of found sectors
#             lcs: List of valid light curve objects, from lcfs corresponding to found_sectors
#     """

#     # -------- CODE pipelined from https://lightkurve.github.io/lightkurve/_modules/lightkurve/utils.html#TessQualityFlags -------- #

#     AttitudeTweak = 1
#     SafeMode = 2
#     CoarsePoint = 4
#     EarthPoint = 8
#     Argabrightening = 16
#     Desat = 32
#     ApertureCosmic = 64
#     ManualExclude = 128
#     Discontinuity = 256
#     ImpulsiveOutlier = 512
#     CollateralCosmic = 1024
#     #: The first stray light flag is set manually by MIT based on visual inspection.
#     Straylight = 2048
#     #: The second stray light flag is set automatically by Ames/SPOC based on background level thresholds.
#     Straylight2 = 4096
#     # See TESS Science Data Products Description Document
#     PlanetSearchExclude = 8192
#     BadCalibrationExclude = 16384
#     # Set in the sector 20 data release notes
#     InsufficientTargets = 32768

#     #: DEFAULT bitmask identifies all cadences which are definitely useless.
#     # See https://outerspace.stsci.edu/display/TESS/2.0+-+Data+Product+Overview
#     DEFAULT_BITMASK = (
#         AttitudeTweak
#         | SafeMode
#         | CoarsePoint
#         | EarthPoint
#         | Argabrightening
#         | Desat
#         | ManualExclude
#         | ImpulsiveOutlier
#         | BadCalibrationExclude
#     )
#     #: HARD bitmask is conservative and may identify cadences which are useful.
#     HARD_BITMASK = (
#         DEFAULT_BITMASK | ApertureCosmic | CollateralCosmic | Straylight | Straylight2
#     )
#     #: HARDEST bitmask identifies cadences with any flag set. Its use is not recommended.
#     HARDEST_BITMASK = 65535

#     #: Dictionary which provides friendly names for the various bitmasks.
#     OPTIONS = {
#         "none": 0,
#         "default": DEFAULT_BITMASK,
#         "hard": HARD_BITMASK,
#         "hardest": HARDEST_BITMASK,
#     }
#     # -------- END CODE pipelined from https://lightkurve.github.io/lightkurve/_modules/lightkurve/utils.html#TessQualityFlags -------- #

#     if quality_bitmask.lower() not in OPTIONS:
#         quality_bitmask = "default"

#     lc_sector_paths = (
#         [Path(f"{lcf_dir}/sector_{sector}") for sector in sectors] if sectors else []
#     )
#     found_sectors = []
#     lcs = []

#     for sector, lc_sector_path in zip(sectors, lc_sector_paths):
#         try:
#             lcf_fps = list(lc_sector_path.rglob(f"*{str(target).zfill(16)}*lc.fits"))
#             if not lcf_fps:  # target not found in sector
#                 continue
#             lcf_fp = lcf_fps[0]  # should only be 1 instance

#             table = Table.read(lcf_fp, format="fits")

#             # masked vals replaced w/ nan
#             time = table["TIME"].value.filled(np.nan)
#             flux = table["PDCSAP_FLUX"].value.filled(np.nan)
#             quality = table["QUALITY"].value.filled(np.nan)

#             assert (
#                 len(time) == len(flux) == len(quality)
#             ), "ERROR: expected time, flux, quality columns to be eq length from table"

#             time, cadence_mask = fill_missing_tess_cadences(time, strategy="average")

#             valid_quality_idxs = (OPTIONS[quality_bitmask] & quality) == 0
#             valid_flux_idxs = np.isfinite(flux)

#             cadence_mask |= ~valid_quality_idxs
#             cadence_mask |= ~valid_flux_idxs

#             flux[cadence_mask] = np.nan

#             lc = lk.LightCurve({"time": time, "flux": flux})

#             lcs.append(lc)
#             found_sectors.append(sector)
#         except fits.VerifyError as e:
#             print(f"ERROR: Corrupted fits file - {e}")
#             continue
#         except Exception as e:
#             print(f"ERROR: While reading lcf - {e}")
#             continue


# def search_and_read_tess_lightcurvefile_with_fits(target: int, sectors: list[int], lcf_dir):
#     """
#     Searches lcf_dir in the format of: lcf_dir/
#                                                 sector_1/
#                                                         *tic_id*_lc.fits
#                                                 sector_2/
#                                              ...sector_n/
#     for a given target tic_id fits file and reads it for
#         Arguments:
#             target: int, specifying target star tic_id
#             sectors: int or List of ints, of sectors to download
#             lcf_dir: str, of directory with tess lightcurve data (.fits)
#         Returns:
#             found_sectors: List of found sectors
#             light_curve_files: List of valid light curve files correponding to found_sectors
#     """


# def search_and_read_tess_lightcurvefile_with_lk(target, sectors, lcf_dir):
#     """
#     Searches lcf_dir in the format of: lcf_dir/
#                                                 sector_1/
#                                                         *tic_id*_lc.fits
#                                                 sector_2/
#                                              ...sector_n/
#     for a given target tic_id fits file and reads it for
#         Arguments:
#             target: int, specifying target star tic_id
#             sectors: int or List of ints, of sectors to download
#             lcf_dir: str, of directory with tess lightcurve data (.fits)
#         Returns:
#             found_sectors: List of found sectors
#             light_curve_files: List of valid light curve files correponding to found_sectors
#     """
#     if isinstance(sectors, int):
#         sectors = [sectors]

#     sector_paths = (
#         [Path(f"{lcf_dir}/sector_{sector}") for sector in sectors] if sectors else []
#     )
#     found_sectors = []
#     light_curve_files = []
#     for sector, sector_path in zip(sectors, sector_paths):
#         try:
#             fps = list(sector_path.rglob(f"*{str(target).zfill(16)}*lc.fits"))
#             if not fps:
#                 # target not found in sector
#                 continue
#             fits_fp = fps[0]  # should only be 1 instance
#             lcf = lk.read(fits_fp)
#             light_curve_files.append(lcf)
#             found_sectors.append(sector)
#         except fits.VerifyError as e:
#             print(f"ERROR: Corrupted fits file - {e}")
#             continue
#         except Exception as e:
#             print(f"ERROR: While reading lcf - {e}")
#             continue
#     return found_sectors, light_curve_files


def search_and_read_tess_targetpixelfile_with_lk(target, sectors, tpf_dir):
    """
    Searches tpf_dir in the format of: tpf_dir/
                                            sector_1/
                                                    *tic_id*_tpf.fits
                                            sector_2/
                                         ...sector_n/
    Arguments:
        target: int, specifying target star tic_id
        sectors: int or List of ints, of sectors to download
        tpf_dir: str, of directory with tess target pixel data (.fits)
    Returns:
        found_sectors: List of found sectors
        target_pixel_files: List of valid target pixel files correponding to found_sectors
    """
    if isinstance(sectors, int):
        sectors = [sectors]

    sector_paths = (
        [Path(f"{tpf_dir}/sector_{sector}") for sector in sectors] if sectors else []
    )
    found_sectors = []
    target_pixel_files = []
    for sector, sector_path in zip(sectors, sector_paths):
        try:
            fps = list(sector_path.rglob(f"*{str(target).zfill(16)}*tp.fits"))
            if not fps:
                continue
            fits_fp = fps[0]  # should only be 1 instance
            tpf = lk.read(fits_fp)
            target_pixel_files.append(tpf)
            found_sectors.append(sector)
        except fits.VerifyError as e:
            print(f"ERROR: corrupted fits file -{e}")
            continue
        except Exception as e:
            # target not found in sector
            print(f"ERROR: While reading tpf - {e}")
            continue
    return found_sectors, target_pixel_files
