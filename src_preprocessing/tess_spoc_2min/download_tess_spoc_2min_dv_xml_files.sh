#!/bin/bash

# Download XML files using curl statements in sh files.

# Directory with .sh files
TARGET_SH_DIR="/data3/exoplnt_dl/dv_xml/tesscurl_spoc/dv_xml_only/"
# Destination directory for XML files
DEST_DIR="/data3/exoplnt_dl/dv_xml/2-min/"

# Create completed directory if it doesn't exist
mkdir -p "$TARGET_SH_DIR/completed"

# Loop through each curl shell script
for sector_shfile in "$TARGET_SH_DIR"/tesscurl*dv.sh; do
    echo "Processing: $sector_shfile"

    # Extract filename and sector info
    SH_FILENAME=$(basename "$sector_shfile")
    SECTOR_RUN="${SH_FILENAME:9:-6}"

    # Determine if it's a multi-sector or single-sector file
    if [[ "$SH_FILENAME" == *"multi"* ]]; then
        echo "Detected multi-sector: $SECTOR_RUN"
        DEST_DIR_SECTOR="$DEST_DIR/multi-sector/$SECTOR_RUN"
    else
        echo "Detected single-sector: $SECTOR_RUN"
        DEST_DIR_SECTOR="$DEST_DIR/single-sector/$SECTOR_RUN"
    fi

    echo "Destination: $DEST_DIR_SECTOR"
    mkdir -p "$DEST_DIR_SECTOR"

    # Copy and execute the shell script
    cp "$sector_shfile" "$DEST_DIR_SECTOR"
    cd "$DEST_DIR_SECTOR" || { echo "Failed to cd into $DEST_DIR_SECTOR"; continue; }

    if bash "$SH_FILENAME"; then
        echo "Successfully ran $SH_FILENAME"
        rm -f tesscurl*dv.sh
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

