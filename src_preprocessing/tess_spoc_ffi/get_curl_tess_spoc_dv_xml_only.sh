#!/bin/bash

# Create new sh script with curl commands only for xml files.

# define directories
TARGET_SH_DIR="/data3/exoplnt_dl/tess_ffi_source_data/dv_xml/tesscurl_spoc"
DEST_DIR="$TARGET_SH_DIR/dv_xml_only"
COMPLETED_DIR="$TARGET_SH_DIR/completed"
CHANGE_PERMISSIONS_AND_GROUP=true

# create necessary directories
mkdir -p "$DEST_DIR"
mkdir -p "$COMPLETED_DIR"

# Process each dv.sh file
for sector_shfile in "$TARGET_SH_DIR"/*dv.sh; do
    # Extract filename from path
    FILENAME=$(basename "$sector_shfile")

    echo "Processing: $FILENAME"

    # Extract curl lines containing 'xml' and save to DEST_DIR
    grep 'xml' "$sector_shfile" > "$DEST_DIR/$FILENAME"

    # Move processed file to completed directory
    mv "$sector_shfile" "$COMPLETED_DIR/"
done

# Set permissions and group if requested
if [[ "$CHANGE_PERMISSIONS_AND_GROUP" == "true" ]]; then
    echo "Changing group to $GROUP and setting permissions..."
    chgrp -R "$GROUP" "$DEST_DIR"
    chmod -R 770 "$DEST_DIR"
fi
