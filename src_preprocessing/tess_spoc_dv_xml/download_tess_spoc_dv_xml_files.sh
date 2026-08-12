#!/bin/bash

# Download TESS SPOC DV XML files using curl statements in sh files. Works for both 2-min and FFI data.
# Assumes the curl sh files have already been filtered to only include XML files.
# Assumes the curl sh files are named "tesscurl*dv.sh" for 2-min and "hlsp_tess-spoc*dv.sh" for FFI, and that they are inside directory $TARGET_SH_DIR.

# Directory with .sh files
TARGET_SH_DIR="/u/msaragoc/work_dir/Kepler-TESS_exoplanet/data/FITS_files/TESS/spoc_2min/dv/s14s86_sh/dv_xml_only/"
# Destination directory for XML files
DEST_DIR="/u/msaragoc/work_dir/Kepler-TESS_exoplanet/data/FITS_files/TESS/spoc_2min/dv/xml_files/sector_runs/"
CHANGE_PERMISSIONS_AND_GROUP=false
# use "tesscurl*dv.sh" for 2-min and "hlsp_tess-spoc*dv.sh" for FFI
CURL_FILE_PATTERN="tesscurl*dv.sh"  

# Create completed directory if it doesn't exist
mkdir -p "$TARGET_SH_DIR/completed"

# Loop through each curl shell script
for sector_shfile in "$TARGET_SH_DIR"/$CURL_FILE_PATTERN; do
    
    echo "Processing: $sector_shfile"

    SH_FILENAME=$(basename "$sector_shfile")

    # Extract SECTOR_RUN for both 2-min and FFI forms
    # Covers:
    #   - 2-min multisector:  multisector_s0001-s0092
    #   - 2-min single-sector: sector_90
    #   - FFI multisector:    s0056-s0069
    #   - FFI single-sector:  s0049

    if [[ "$SH_FILENAME" =~ (multisector_s[0-9]{4}-s[0-9]{4}|sector_[0-9]{1,3}|s[0-9]{4}-s[0-9]{4}|s[0-9]{4}) ]]; then
        SECTOR_RUN="${BASH_REMATCH[1]}"
    else
        echo "❌ Error: Could not extract sector run from filename '$SH_FILENAME'" >&2
        exit 1
    fi


    echo "✅ Extracted sector run: $SECTOR_RUN"

    if [[ $CURL_FILE_PATTERN == tesscurl* ]]; then  # for 2-min
        # Determine if it's a multi-sector or single-sector file
        if [[ $SH_FILENAME == *multisector* ]]; then
            echo "Detected multi-sector: $SECTOR_RUN"
            DEST_DIR_SECTOR="$DEST_DIR/multi-sector/$SECTOR_RUN"
        else
            echo "Detected single-sector: $SECTOR_RUN"
            DEST_DIR_SECTOR="$DEST_DIR/single-sector/$SECTOR_RUN"
        fi
    else  # for ffi
        DEST_DIR_SECTOR="$DEST_DIR/$SECTOR_RUN"
    fi

    echo "Destination: $DEST_DIR_SECTOR"
    mkdir -p "$DEST_DIR_SECTOR"

    # copy shell script
    cp "$sector_shfile" "$DEST_DIR_SECTOR"
    chmod +x "$DEST_DIR_SECTOR/$(basename "$sector_shfile")"

    # move into sector directory for 2-min
    if [[ $CURL_FILE_PATTERN == tesscurl* ]]; then  # for 2-min
        cd "$DEST_DIR_SECTOR" || { echo "Failed to cd into $DEST_DIR_SECTOR"; continue; }
    fi

    if bash "$DEST_DIR_SECTOR/$(basename "$sector_shfile")"; then
        echo "Successfully ran $SH_FILENAME"
        # rm -f $CURL_FILE_PATTERN
        rm -f "$DEST_DIR_SECTOR/$(basename "$sector_shfile")" # delete sh file inside the folder
        mv "$sector_shfile" "$TARGET_SH_DIR/completed/" # move original sh file to completed
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

