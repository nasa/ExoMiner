#!/bin/bash

# Download TESS SPOC DV XML files using curl statements in sh files. Works for both 2-min and FFI data.
# Assumes the curl sh files have already been filtered to only include XML files.
# Assumes the curl sh files are named "tesscurl*dv.sh" for 2-min and "hlsp_tess-spoc*dv.sh" for FFI, and that they are inside directory $TARGET_SH_DIR.

# Directory with .sh files
TARGET_SH_DIR="/u/msaragoc/work_dir/Kepler-TESS_exoplanet/data/FITS_files/TESS/spoc_ffi/dv/xml_files/target_sh_xml_only"
# Destination directory for XML files
DEST_DIR="/u/msaragoc/work_dir/Kepler-TESS_exoplanet/data/FITS_files/TESS/spoc_ffi/dv/xml_files/sector_runs/"
CHANGE_PERMISSIONS_AND_GROUP=false
# use "tesscurl*dv.sh" for 2-min and "hlsp_tess-spoc*dv.sh" for FFI
CURL_FILE_PATTERN="hlsp_tess-spoc*dv.sh"  

# Create completed directory if it doesn't exist
mkdir -p "$TARGET_SH_DIR/completed"

# Loop through each curl shell script
for sector_shfile in "$TARGET_SH_DIR"/$CURL_FILE_PATTERN; do
    
    echo "Processing: $sector_shfile"

    SH_FILENAME=$(basename "$sector_shfile")

    if [[ "$SH_FILENAME" =~ (sector_[0-9]+) ]]; then  # for 2-min
        SECTOR_RUN="${BASH_REMATCH[1]}"
    elif [[ "$SH_FILENAME" =~ (s[0-9]{4}) ]]; then  # for FFI
        SECTOR_RUN="${BASH_REMATCH[1]}"
    else
        echo "❌ Error: Could not extract sector run from filename '$SH_FILENAME'" >&2
        exit 1
    fi

    echo "✅ Extracted sector run: $SECTOR_RUN"

    # # Extract filename and sector info
    # SH_FILENAME=$(basename "$sector_shfile")
    # SECTOR_RUN="${SH_FILENAME:9:-6}"

    DEST_DIR_SECTOR="$DEST_DIR/$SECTOR_RUN"
    # # Determine if it's a multi-sector or single-sector file
    # if [[ "$SH_FILENAME" == *"multi"* ]]; then
    #     echo "Detected multi-sector: $SECTOR_RUN"
    #     DEST_DIR_SECTOR="$DEST_DIR/multi-sector/$SECTOR_RUN"
    # else
    #     echo "Detected single-sector: $SECTOR_RUN"
    #     DEST_DIR_SECTOR="$DEST_DIR/single-sector/$SECTOR_RUN"
    # fi

    echo "Destination: $DEST_DIR_SECTOR"
    mkdir -p "$DEST_DIR_SECTOR"

    # Copy and execute the shell script
    cp "$sector_shfile" "$DEST_DIR_SECTOR"
    cd "$DEST_DIR_SECTOR" || { echo "Failed to cd into $DEST_DIR_SECTOR"; continue; }

    if bash "$SH_FILENAME"; then
        echo "Successfully ran $SH_FILENAME"
        rm -f $CURL_FILE_PATTERN
        mv "$sector_shfile" "$TARGET_SH_DIR/completed/"
    else
        echo "Error running $SH_FILENAME"
    fi
done

echo "Finished downloading DV XML files."

# Set permissions and group if requested
if [[ "$CHANGE_PERMISSIONS_AND_GROUP" == "true" ]]; then
    echo "Changing group to $GROUP and setting permissions..."
    chgrp -R "$GROUP" "$DEST_DIR"
    chmod -R 770 "$DEST_DIR"
fi

