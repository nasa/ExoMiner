# Download xml files using curl statements in sh files.

# directory with sh files
TARGET_SH_DIR=/Users/msaragoc/Downloads/tess-spoc_ffi_s71_s72/target_sh_xml_only
# destination directory for xml files
DEST_DIR=/Users/msaragoc/Downloads/tess-spoc_ffi_s71_s72
# set permissions and group if needed
CHANGE_PERMISSIONS_AND_GROUP=true
GROUP=ar-gg-ti-tess-dsg

# create directory for completed sh scripts
COMPLETED_DIR=$TARGET_SH_DIR/completed
mkdir -p $COMPLETED_DIR

for SECTOR_SHFILE in "$TARGET_SH_DIR"/*.sh
do
    echo "AAA" "$SECTOR_SHFILE"
    SH_FILENAME_INDEX=$(echo "$SECTOR_SHFILE" | grep -bo "hlsp_tess" | grep -oe "[0-9]*")
    SH_FILENAME=${SECTOR_SHFILE:$SH_FILENAME_INDEX}
    SECTOR_RUN=${SH_FILENAME:25:5}
    echo "BBB" "$SECTOR_RUN"

    if [[ $SH_FILENAME == *"multi"* ]]

    then
        echo multi "$SECTOR_RUN"
        DEST_DIR_SECTOR=$DEST_DIR/multi-sector  # /"$SECTOR_RUN"
    else
        echo single "$SECTOR_RUN"
        DEST_DIR_SECTOR=$DEST_DIR/single-sector  # /"$SECTOR_RUN"
    fi

    echo "$DEST_DIR_SECTOR"
    mkdir -p "$DEST_DIR_SECTOR"

    cd $DEST_DIR_SECTOR

    bash $SECTOR_SHFILE

    cd ../

    mv "$SECTOR_SHFILE" "$COMPLETED_DIR"  # move completed sh script
done

# set permissions and group
if [[ $CHANGE_PERMISSIONS_AND_GROUP == true ]]

then
  chgrp -R $GROUP $DEST_DIR
  chmod -R 770 $DEST_DIR
fi