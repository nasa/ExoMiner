from pathlib import Path
import lightkurve as lk
from astropy.io import fits

def search_and_read_tess_lightcurve(target, sectors, lc_dir):
    """
    Searches lc_dir in the format of: lc_dir/
                                                sector_1/
                                                        *tic_id*_lc.fits
                                                sector_2/
                                             ...sector_n/
    for a given target tic_id fits file and reads it for 
        Arguments:
            target: int, specifying target star tic_id
            sectors: int or List of ints, of sectors to download
            lc_dir: str, of directory with tess lightcurve data (.fits)
        Returns:
            found_sectors: List of found sectors
            light_curve_files: List of valid light curve files correponding to found_sectors
    """
    if isinstance(sectors, int):
        sectors = [sectors]

    sector_paths = [Path(f"{lc_dir}/sector_{sector}") for sector in sectors] if sectors else []
    found_sectors = []
    light_curve_files = []
    for sector, sector_path in zip(sectors, sector_paths):
        try:
            fits_file_path = list(sector_path.rglob(f"*{f'{target}'.zfill(16)}*lc.fits"))[0] #should only be 1 instance
            lcf = lk.read(fits_file_path)
            light_curve_files.append(lcf)
            found_sectors.append(sector)
        except:
            # target not found in sector 
            pass
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
            fits_file_path = list(sector_path.rglob(f"*{f'{target}'.zfill(16)}*tp.fits"))[0] #should only be 1 instance
            tpf = lk.read(fits_file_path)
            target_pixel_files.append(tpf)
            found_sectors.append(sector)
        except fits.VerifyError as e:
            print(f"Corrupted fits file: {e}")
        except Exception as e:
            # target not found in sector 
            pass
    return found_sectors, target_pixel_files
