from astropy.io import fits
import os
import warnings


def validate_fits_file(file_path):
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always', fits.verify.VerifyWarning)
                
            with fits.open(file_path, ignore_missing_end=True) as hdul:
                hdul.verify('silentfix')
                # print(f"{file_path}: FITS file is valid.")
                return True
            
            for warning in w:
                if issubclass(warning.category, fits.verify.VerifyWarning):
                    print(f"Warning: {warning.message}")

            header = hdul[0].header
            return header
        
    except fits.VerifyError as e:
        print(f'Verify Error: {file_path}: {e}')
        return False
    except Exception as e:
        print(f"{file_path}: Corrupted or invalid FITS file")
        return False
    
def process_sector_files(sector_dir):
    sector_name = os.path.basename(sector_dir)
    failed_files_log = os.path.join(sector_dir, f"{sector_name}_failed_files_1.txt")

    with open(failed_files_log, 'w') as log_file:
        for file_name in os.listdir(sector_dir):
            file_path = os.path.join(sector_dir, file_name)
            if file_name.endswith('.fits'):
                is_valid = validate_fits_file(file_path)
                if not is_valid:
                    log_file.write(f"{file_name}\n")
                    print(f"Validation failed for {file_name} in sector {sector_name}")

def validate_sectors(root_dir):
    for sector_name in os.listdir(root_dir):
        sector_dir = os.path.join(root_dir, sector_name)
        if os.path.isdir(sector_dir):
            print(f"Processing sector {sector_name}...")
            process_sector_files(sector_dir)
    
# fits_folder = "/Users/jochoa4/Desktop/ExoMiner/exoplanet_dl/transit_detection/bash_scripts/fits"

# for file_name in os.listdir(fits_folder):
#     file_path = os.path.join(fits_folder, file_name)
#     if file_name.endswith('.fits'):
#         validate_fits_file(file_path)

if __name__ == "__main__":
    root_dir = "/nobackup/jochoa4/TESS/fits_files/spoc_2min/tp"
    validate_sectors(root_dir)