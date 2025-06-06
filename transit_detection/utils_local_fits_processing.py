from pathlib import Path
import lightkurve as lk
from astropy.io import fits

def search_and_read_tess_lightcurvefile(target, sectors, lcf_dir):
    """
    Searches lcf_dir in the format of: lcf_dir/
                                                sector_1/
                                                        *tic_id*_lc.fits
                                                sector_2/
                                             ...sector_n/
    for a given target tic_id fits file and reads it for 
        Arguments:
            target: int, specifying target star tic_id
            sectors: int or List of ints, of sectors to download
            lcf_dir: str, of directory with tess lightcurve data (.fits)
        Returns:
            found_sectors: List of found sectors
            light_curve_files: List of valid light curve files correponding to found_sectors
    """
    if isinstance(sectors, int):
        sectors = [sectors]

    sector_paths = [Path(f"{lcf_dir}/sector_{sector}") for sector in sectors] if sectors else []
    found_sectors = []
    light_curve_files = []
    for sector, sector_path in zip(sectors, sector_paths):
        try:
            fps = list(sector_path.rglob(f"*{str(target).zfill(16)}*lc.fits"))
            if not fps:
                # target not found in sector 
                continue
            fits_fp = fps[0] #should only be 1 instance
            lcf = lk.read(fits_fp)
            light_curve_files.append(lcf)
            found_sectors.append(sector)
        except fits.VerifyError as e:
            print(f"ERROR: Corrupted fits file - {e}")
            continue
        except Exception as e:
            print(f"ERROR: While reading lcf - {e}")
            continue
    return found_sectors, light_curve_files

def search_and_read_tess_targetpixelfile(target, sectors, tpf_dir):
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

    sector_paths = [Path(f"{tpf_dir}/sector_{sector}") for sector in sectors] if sectors else []
    found_sectors = []
    target_pixel_files = []
    for sector, sector_path in zip(sectors, sector_paths):
        try:
            fps = list(sector_path.rglob(f"*{str(target).zfill(16)}*tp.fits")) 
            if not fps:
                continue
            fits_fp = fps[0] #should only be 1 instance
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
