# Download xml files using curl statements in sh files.

# directory with sh files
TARGET_SH_DIR=/u/msaragoc/work_dir/Kepler-TESS_exoplanet/data/FITS_files/TESS/spoc_ffi/lc_sh_files/download_targets_4-16-2025_1014/
# destination directory for lc files
DEST_DIR=/home6/msaragoc/work_dir/Kepler-TESS_exoplanet/data/FITS_files/TESS/spoc_ffi/lc/
# set permissions and group if needed
CHANGE_PERMISSIONS_AND_GROUP=false
GROUP=ar-gg-ti-tess-dsg

# create directory for completed sh scripts
COMPLETED_DIR=$TARGET_SH_DIR/completed
mkdir -p $COMPLETED_DIR
for SECTOR_SHFILE in "$TARGET_SH_DIR"/*lc.sh
do
    echo "$SECTOR_SHFILE"

    cd $DEST_DIR

    bash "$SECTOR_SHFILE"  # download files for sector run; sh script creates directory for sector run

    cd ../

    mv "$SECTOR_SHFILE" "$COMPLETED_DIR"  # move completed sh script

done

echo "Finished downloading light curve FITS files"

# set permissions and group
if [[ $CHANGE_PERMISSIONS_AND_GROUP == true ]]

then
  chgrp -R $GROUP $DEST_DIR
  chmod -R 770 $DEST_DIR
fi