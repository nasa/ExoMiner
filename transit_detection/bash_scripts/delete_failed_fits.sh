#!/bin/bash

# failed file format: sector_1/sector_1_failed_files.txt 

ROOT_DIR="/nobackup/jochoa4/TESS/fits_files/spoc_2min/tp"

# deletes failed fits files from sectors 1-67:

for sector_number in {1..67}; do
    SECTOR_DIR="$ROOT_DIR/sector_$sector_number"
    FAILED_FILES="$SECTOR_DIR/sector_${sector_number}_failed_files_run_2.txt"

    if [[ ! -d "$SECTOR_DIR" ]]; then # if directory does not exist
        echo "Sector directory $SECTOR_DIR not found!"
        continue
    fi

    if [[ ! -f "$FAILED_FILES" ]]; then #if file does not exist
        echo "Failed file list not found for $SECTOR_DIR!"
        continue
    fi

    while IFS= read -r file; do # while reading from 
        file_path="$SECTOR_DIR/$file"

        if [[ -f "$file_path" ]]; then
            echo "Deleting $file_path"
            rm "$file_path"
        else
            echo "File not found: $file_path"
        fi
    done < "$FAILED_FILES" #read from failed files for sector

    echo "Done processing failed files for $SECTOR_DIR."
done

echo "All sectors processed."