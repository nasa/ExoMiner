#!/bin/bash

#execute download_failed_files_sector_X.sh for each sector

ROOT_DIR="/nobackup/jochoa4/TESS/fits_files/spoc_2min/tp"

for sector_number in {1..67}; do
    SECTOR_DIR="$ROOT_DIR/sector_$sector_number"
    DOWNLOAD_SCRIPT="$SECTOR_DIR/download_failed_files_run_2_sector_$sector_number.sh"

    if [[ -f "$DOWNLOAD_SCRIPT" ]]; then # if directory does not exist
        echo "Running download script for sector $sector_number..."
        bash "$DOWNLOAD_SCRIPT"

    else
        echo "No download script found for sector $sector_number"
    fi
done

echo "All sector scripts have been processed."