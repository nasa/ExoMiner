#!/bin/bash

# Create new sh script with curl commands only for TESS SPOC DV XML files. Works for both 2-min and FFI data.
# Assumes the curl sh files are named "*dv.sh" and inside a directory $TARGET_SH_DIR.

TARGET_SH_DIR=/u/msaragoc/work_dir/Kepler-TESS_exoplanet/data/FITS_files/TESS/spoc_ffi/dv_sh_files/
DEST_DIR=/u/msaragoc/work_dir/Kepler-TESS_exoplanet/data/FITS_files/TESS/spoc_ffi/dv_sh_files/dv_xml_only/
# set permissions and group if needed
CHANGE_PERMISSIONS_AND_GROUP=false
GROUP=ar-gg-ti-tess-dsg

mkdir -p $DEST_DIR
mkdir -p "$TARGET_SH_DIR"/completed

for sector_shfile in "$TARGET_SH_DIR"/*dv.sh
do
   FILENAME=$(basename "$sector_shfile")
   echo "$FILENAME"
   grep xml "$sector_shfile" > $DEST_DIR/"$FILENAME"
   mv "$sector_shfile" "$TARGET_SH_DIR"/completed
done

echo "Finished filtering TESS curl sh files for DV XML files only."

# set permissions and group
if [[ $CHANGE_PERMISSIONS_AND_GROUP == true ]]

then
  chgrp -R $GROUP $DEST_DIR
  chmod -R 770 $DEST_DIR
fi