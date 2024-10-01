from pathlib import Path
import lightkurve as lk

def search_and_read_tess_lightcurve(target, sectors, data_dir):
    """
        Arguments:
            target: int, specifying target star
            sectors: int or List of ints, of sectors to download
            data_dir: str, of directory with tess lightcurve data (.fits)
        Returns:
            None
    """
    if isinstance(sectors, int):
        sectors = [sectors]

    sector_paths = [Path(f"{data_dir}/sector_{sector}") for sector in sectors] if sectors else []
    found_sectors = []
    light_curve_files = []
    for sector, sector_path in zip(sectors, sector_paths):
        try:
            fits_file_path = list(sector_path.rglob(f"*{f'{target}'.zfill(16)}*lc.fits"))[0] #should only be 1 instance
            lcf = lk.read(fits_file_path)
            light_curve_files.append(lcf)
            found_sectors.append(sector)
        except:
            print(f"Error finding file {sector_path}.")
    return found_sectors, light_curve_files

def search_and_read_tess_targetpixelfile(target, sectors, data_dir):
    """
        Arguments:
            target: int, specifying target star
            sectors: int or List of ints, of sectors to download
            data_dir: str, of directory with tess target pixel data (.fits)
        Returns:
            None
    """
    if isinstance(sectors, int):
        sectors = [sectors]

    sector_paths = [Path(f"{data_dir}/sector_{sector}") for sector in sectors] if sectors else []
    found_sectors = []
    target_pixel_files = []
    for sector, sector_path in zip(sectors, sector_paths):
        try:
            fits_file_path = list(sector_path.rglob(f"*{f'{target}'.zfill(16)}*tp.fits"))[0] #should only be 1 instance
            tpf = lk.read(fits_file_path)
            target_pixel_files.append(tpf)
            found_sectors.append(sector)
        except:
            print(f"Error finding file {sector_path}.")
    return found_sectors, target_pixel_files
