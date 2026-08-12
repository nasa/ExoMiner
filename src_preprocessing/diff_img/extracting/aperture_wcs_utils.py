""" Utilities to extract WCS from TESS SPOC light-curve files' APERTURE."""

# 3rd party
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Optional, Tuple, List
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
import logging

logger = logging.getLogger(__name__)

@dataclass
class ApertureWCSResult:
    """ Result of extracting WCS from TESS SPOC light-curve APERTURE extension.
    """
    wcs: WCS
    aperture_shape: Tuple[int, int]      # (n_rows, n_cols)
    extname: str                         # e.g., 'APERTURE'
    file_path: Path


# Classic SPOC:
# tessYYYY... - sNNNN - TIC16 - #### - s_lc.fits
RE_CLASSIC = re.compile(
    r"^tess[0-9]{13}-s(?P<sector>\d{4})-(?P<tic>\d{16})-[0-9]+-s_lc\.fits$"
)

# HLSP mirror:
# hlsp_tess-spoc_tess_phot_TIC16 - sNNNN _ tess_v* _ lc.fits
RE_HLSP = re.compile(
    r"^hlsp_tess-spoc_tess_phot_(?P<tic>\d{16})-s(?P<sector>\d{4})_tess_v\d+_lc\.fits$"
)


def _pad_tic(tic_id: int | str) -> str:
    """ Pad TIC ID to 16 digits with leading zeros.

    :param int | str tic_id: TIC ID
    :raises ValueError: if TIC ID is not numeric
    :return str: zero-padded TIC ID (16 digits)
    """
    
    tic_str = str(tic_id).strip()
    
    if not tic_str.isdigit():
        raise ValueError(f"TIC ID must be numeric, got: {tic_id}")
    
    return tic_str.zfill(16)


def _extract_tic_sector_from_name(fp: Path) -> Optional[Tuple[str, int]]:
    """Try both regexes; return (tic16, sector_int) or None if no match.
    
    param Path fp: file path
    :return Optional[Tuple[str, int]]: (tic16, sector_id) or None
    """
    
    name = fp.name

    for rex in (RE_CLASSIC, RE_HLSP):
        m = rex.match(name)
        if m:
            tic16 = m.group("tic")
            sector = int(m.group("sector"))
            return tic16, sector
        
    return None


def _find_lc_file_by_name(
    lc_dir: Path, tic16: str, sector_id: int
) -> List[Path]:
    """
    Collect candidates that look like LC files for TIC/sector by
    filename pattern only. Supports both styles.
    
    :param Path lc_dir: light curve directory
    :param str tic16: zero-padded TIC ID (16 digits)
    :param int sector_id: Sector ID
    :return List[Path]: list of candidate file paths
    """
    
    s_tag = f"s{sector_id:04d}"

    # Patterns that catch both classic and HLSP styles
    patterns = [
        f"tess*-{s_tag}-{tic16}-*-s_lc.fits",                  # classic
        f"hlsp_tess-spoc_tess_phot_{tic16}-{s_tag}_*_lc.fits", # HLSP
        # relaxed fallbacks (just in case)
        f"*{tic16}*{s_tag}*lc.fits*",
    ]
    
    cand = []
    for pat in patterns:
        cand.extend(list(Path(lc_dir).rglob(pat)))

    # Filter by strict regex check to avoid false positives
    filtered = []
    for fp in cand:
        ts = _extract_tic_sector_from_name(fp)
        if ts is not None:
            t16, sec = ts
            if t16 == tic16 and sec == sector_id:
                filtered.append(fp)
                
    return sorted(set(filtered))


def _prefer_latest(candidates: List[Path]) -> Path:
    """
    Heuristic: prefer file with the highest 'drNN' (data release) in name
    if present; otherwise choose by most recent mtime.
    
    :param List[Path] candidates: list of candidate file paths
    :return Path: preferred file path
    """
    
    def _dr_tag(fp: Path) -> int:
        m = re.search(r"dr(\d+)", fp.stem)
        return int(m.group(1)) if m else -1

    candidates.sort(key=lambda p: (_dr_tag(p), p.stat().st_mtime))
    
    return candidates[-1]


def _find_lc_file_by_header(
    lc_dir: Path, tic16: str, sector_id: int, scan_limit: int = 2000
) -> Optional[Path]:
    """
    Fallback: scan *.fits in lc_dir and check primary/header keywords such as
    TICID, SECTOR (or TSECTOR), and ensure it is a light-curve (LC) product.
    We cap the scan to 'scan_limit' files for safety.
    
    :param Path lc_dir: light curve directory
    :param str tic16: zero-padded TIC ID (16 digits)
    :param int sector_id: Sector ID
    :param int scan_limit: maximum number of FITS files to scan
    :return Optional[Path]: path to located LC file, or None if not found
    """
    
    all_fits = list(Path(lc_dir).glob("*.fits*"))
    if len(all_fits) == 0:
        return None

    # Prioritize files that contain 'lc' in the name
    all_fits.sort(key=lambda p: ("lc" not in p.name.lower(), p.name.lower()))
    for fp in all_fits[:scan_limit]:
        
        try:
            with fits.open(fp) as hdul:
                hdr = hdul[0].header  # primary
                header_tic = str(hdr.get("TICID", hdr.get("OBJECT", ""))).strip()
                header_sec = hdr.get("SECTOR", hdr.get("TSECTOR", None))

                # TIC in header may be unpadded; normalize
                if header_tic and header_tic.isdigit():
                    header_tic16 = header_tic.zfill(16)
                else:
                    header_tic16 = None

                # Sector may be string or int
                if isinstance(header_sec, str) and header_sec.isdigit():
                    header_sector = int(header_sec)
                elif isinstance(header_sec, (int, np.integer)):
                    header_sector = int(header_sec)
                else:
                    header_sector = None

                # Is this an LC product?
                prod = (hdr.get("TELESCOP") or "") + " " + (hdr.get("PROCVER") or "")
                name_lc_like = ("lc" in fp.name.lower())
                lc_hdu_present = any(
                    (h.header.get("EXTNAME", "").upper().endswith("LC") or
                     "APERTURE" in (h.header.get("EXTNAME", "").upper()))
                    for h in hdul
                )

                if header_tic16 == tic16 and header_sector == sector_id and (name_lc_like or lc_hdu_present):
                    return fp
                
        except Exception:
            
            # ignore unreadable files
            continue
        
    return None


def _get_aperture_extension(hdul: fits.HDUList) -> Tuple[int, str]:
    """
    Prefer EXTNAME containing 'APERTURE'. Fallback: first image HDU with
    celestial WCS keywords present.
    
    param fits.HDUList hdul: opened FITS file HDU list
    :return Tuple[int, str]: (extension index, extension name)
    """
    
    # 1) Prefer explicit APERTURE
    for i, hdu in enumerate(hdul):
        extname = (hdu.header.get("EXTNAME") or "").upper()
        if hdu.data is not None and "APERTURE" in extname:
            return i, extname

    # 2) Fallback: first WCS-bearing image HDU
    for i, hdu in enumerate(hdul):
        if hdu.data is None:
            continue
        hdr = hdu.header
        has_wcs = (hdr.get("CTYPE1") and hdr.get("CTYPE2")) or \
                  any(k in hdr for k in ("CRVAL1", "CRVAL2", "CD1_1", "CD2_2", "PC1_1", "PC2_2"))
        if has_wcs:
            return i, hdr.get("EXTNAME", f"HDU{i}")

    raise RuntimeError("Could not locate an APERTURE/WCS-bearing image extension.")


def _resolve_ffi_hlsp_candidates(root: Path, tic16: str, sector_id: int) -> List[Path]:
    """
    Resolve FFI HLSP path(s) deterministically from (tic16, sector_id).
    Does not recurse; only checks the precise target subdir.
    """
    s_tag = f"s{sector_id:04d}"
    # Build the nested TIC path components: 0000/0010/0410/3418
    subdirs = [tic16[i:i+4] for i in range(0, 16, 4)]
    target_dir = root / s_tag / "target" / subdirs[0] / subdirs[1] / subdirs[2] / subdirs[3]

    # Filename base (version is wildcard)
    base = f"hlsp_tess-spoc_tess_phot_{tic16}-{s_tag}_tess_v"
    # Non-recursive glob only in that leaf directory
    if not target_dir.exists():
        return []
    return list(target_dir.glob(base + "*_lc.fits"))

def _resolve_2min_spoc_candidates(root: Path, tic16: str, sector_id: int) -> List[Path]:
    """
    Resolve 2-min SPOC path(s) deterministically from (tic16, sector_id).
    Does not recurse; only checks the sector_{id} directory.
    """
    s_tag = f"s{sector_id:04d}"
    sector_dir = root / f"sector_{sector_id}"
    if not sector_dir.exists():
        return []

    # SPOC classic pattern: tess*-sNNNN-TIC16-*-s_lc.fits
    pat = f"tess*-{s_tag}-{tic16}-*-s_lc.fits"
    return list(sector_dir.glob(pat))

def _prefer_latest(candidates: List[Path]) -> Path:
    """
    Same logic as your current function, but try to prefer drNN if present.
    If no drNN in names, use mtime.
    """
    def _dr_tag(fp: Path) -> int:
        m = re.search(r"dr(\d+)", fp.stem)
        return int(m.group(1)) if m else -1
    # Sort by (DR, mtime) and pick the last
    candidates.sort(key=lambda p: (_dr_tag(p), p.stat().st_mtime))
    return candidates[-1]

def _find_lc_file_fast(lc_root: Path, tic_id: int | str, sector_id: int) -> Path:
    """
    Fast resolver that tries FFI HLSP location, then 2-min SPOC location,
    using deterministic paths and non-recursive globs.
    """
    tic16 = str(tic_id).strip().zfill(16)

    # 1) Try FFI HLSP
    cand = _resolve_ffi_hlsp_candidates(lc_root, tic16, sector_id)
    if cand:
        return _prefer_latest(cand)

    # 2) Try 2-min SPOC
    cand = _resolve_2min_spoc_candidates(lc_root, tic16, sector_id)
    if cand:
        return _prefer_latest(cand)

    # 3) As an optional, narrow fallback: glob only in these two dirs for safety
    s_tag = f"s{sector_id:04d}"
    narrow = []
    # Leaf target directory for FFI
    subdirs = [tic16[i:i+4] for i in range(0, 16, 4)]
    target_dir = lc_root / s_tag / "target" / subdirs[0] / subdirs[1] / subdirs[2] / subdirs[3]
    if target_dir.exists():
        narrow.extend(target_dir.glob(f"*{tic16}*{s_tag}*lc.fits*"))
    sector_dir = lc_root / f"sector_{sector_id}"
    if sector_dir.exists():
        narrow.extend(sector_dir.glob(f"*{tic16}*{s_tag}*lc.fits*"))

    if narrow:
        return _prefer_latest(narrow)

    raise FileNotFoundError(
        f"No LC FITS under {lc_root} for TIC {tic16} sector {sector_id} "
        f"(checked FFI HLSP and 2-min SPOC locations)."
    )

# def _find_lc_file(
#     lc_dir: Path, tic_id: int | str, sector_id: int, prefer_latest: bool = True
# ) -> Path:
#     """
#     Unifies the two strategies:
#     1) filename-based search (handles both classic and HLSP)
#     2) header-based fallback (TICID/SECTOR)
    
#     param Path lc_dir: light curve directory
#     :param int | str tic_id: TIC ID
#     :param int sector_id: Sector ID
#     :param bool prefer_latest: if multiple candidates found by name, prefer latest
#     :return Path: path to located LC file
#     """
    
#     tic16 = _pad_tic(tic_id)

#     # Try filename approach first
#     candidates = _find_lc_file_by_name(lc_dir, tic16, sector_id)
#     if len(candidates) > 0:
#         return _prefer_latest(candidates) if prefer_latest else candidates[0]

#     # Fallback: header scan
#     fp = _find_lc_file_by_header(lc_dir, tic16, sector_id)
#     if fp is not None:
#         return fp

#     raise FileNotFoundError(
#         f"No LC FITS found in {lc_dir} for TIC {tic16} sector {sector_id}."
#     )


def get_aperture_wcs_from_lc(
    lc_dir: Path,
    tic_id: int | str,
    sector_id: int
) -> ApertureWCSResult:
    """
    Locate LC file for (tic_id, sector_id), read APERTURE extension,
    and return its WCS + image shape.
    
    :param Path lc_dir: light curve directory
    :param int | str tic_id: TIC ID
    :param int sector_id: Sector ID
    :return ApertureWCSResult: dataclass with WCS, aperture shape, extname, and file path
    """
    
    lc_fp = _find_lc_file_fast(lc_dir, tic_id, sector_id)

    with fits.open(lc_fp) as hdul:
        ext_idx, extname = _get_aperture_extension(hdul)
        hdr = hdul[ext_idx].header
        data = hdul[ext_idx].data
        if data is None:
            raise RuntimeError(f"Extension '{extname}' has no data.")
        wcs = WCS(hdr, relax=True)
        if wcs.naxis < 2:
            raise RuntimeError(f"WCS in extension '{extname}' is not 2D.")
        shape = (data.shape[0], data.shape[1])  # (rows, cols)

    return ApertureWCSResult(wcs=wcs, aperture_shape=shape, extname=extname, file_path=lc_fp)


def sky_to_aperture_pixels(
    wcs: WCS,
    ra_deg: float,
    dec_deg: float,
) -> dict:
    """Map sky coordinates (RA, Dec) to aperture pixel coordinates (col, row).

    :param WCS wcs: WCS object
    :param float ra_deg: Right Ascension in degrees
    :param float dec_deg: Declination in degrees
    :return dict: pixel coordinates (col, row)
    """
    
    sc = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
    
    col_pix, row_pix = wcs.world_to_pixel(sc)  # (x,y) == (col,row)

    return {'col': float(col_pix), 'row': float(row_pix)}


def sky_to_aperture(
    lc_dir: Path,
    tic_id: int | str,
    sector_id: int,
    ra_deg: float,
    dec_deg: float
) -> dict:
    """ Get aperture pixel coordinates from sky coordinates using WCS in target light curve aperture header

    :param Path lc_dir: light curve directory
    :param int | str tic_id: TIC ID
    :param int sector_id: Sector ID
    :param float ra_deg: Right Ascension in degrees
    :param float dec_deg: Declination in degrees
    :return dict: dictionary with pixel coordinates {'col': float, 'row': float}
    """

    try:
        res = get_aperture_wcs_from_lc(lc_dir, tic_id, sector_id)

        pix = sky_to_aperture_pixels(res.wcs, ra_deg, dec_deg)  # , res.aperture_shape)
    except Exception as e:
        logger.error(f'Found issue while getting aperture WCS from target {tic_id} light curve in {lc_dir}: {e}\n'
                    'Setting pixel coordinates to NaN.')
        pix = {'col': np.nan, 'row': np.nan}

    return pix

