#!/bin/bash

# failed file format: sector_1/sector_1_failed_files.txt 
# curl command example: curl -C - -L -o tess2021204101404-s0041-0000000011113164-0212-s_tp.fits https://mast.stsci.edu/api/v0.1/Download/file/?uri=mast:TESS/product/tess2021204101404-s0041-0000000011113164-0212-s_tp.fits

ROOT_DIR="/nobackup/jochoa4/TESS/fits_files/spoc_2min/tp"
BASE_URL="https://mast.stsci.edu/api/v0.1/Download/file/?uri=mast:TESS/product/"

for sector_number in {1..67}; do
    SECTOR_DIR="$ROOT_DIR/sector_$sector_number"
    FAILED_FILES="$SECTOR_DIR/sector_${sector_number}_failed_files.txt"
    OUTPUT_SCRIPT="$SECTOR_DIR/download_failed_files_sector_$sector_number.sh"

    if [[ ! -d "$SECTOR_DIR" ]]; then # if directory does not exist
        echo "Sector directory $SECTOR_DIR not found!"
        continue
    fi

    if [[ ! -f "$FAILED_FILES" ]]; then #if file does not exist
        echo "Failed file list not found for $SECTOR_DIR!"
        continue
    fi

    echo "#!/bin/bash" > "$OUTPUT_SCRIPT" #prepend shebang

    while IFS= read -r file; do
        echo "curl -C - -L -o $file $BASE_URL/$file" >> "$OUTPUT_SCRIPT"
    done < "$FAILED_FILES"

    chmod +x "$OUTPUT_SCRIPT"

    echo "Created script: $OUTPUT_SCRIPT."
done

echo "All scripts generated."