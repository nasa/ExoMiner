# Download xml files using curl statements in sh files.

# directory with sh files
TARGET_SH_DIR=null
# destination directory for xml files
DEST_DIR=null
# set permissions and group if needed
CHANGE_PERMISSIONS_AND_GROUP=false
GROUP=null

mkdir -p "$TARGET_SH_DIR"/completed

for sector_shfile in "$TARGET_SH_DIR"/tesscurl*dv.sh
do
    echo "$sector_shfile"
    SH_FILENAME_INDEX=$(echo "$sector_shfile" | grep -bo "tesscurl" | grep -oe "[0-9]*")
    SH_FILENAME=${sector_shfile:$SH_FILENAME_INDEX}
    SECTOR_RUN=${SH_FILENAME:9:-6}

    if [[ $SH_FILENAME == *"multi"* ]]

    then
        echo multi "$SECTOR_RUN"
        DEST_DIR_SECTOR=$DEST_DIR/multi-sector/"$SECTOR_RUN"
    else
        echo single "$SECTOR_RUN"
        DEST_DIR_SECTOR=$DEST_DIR/single-sector/"$SECTOR_RUN"
    fi

    echo "$DEST_DIR_SECTOR"
    mkdir -p "$DEST_DIR_SECTOR"
    cp "$sector_shfile" "$DEST_DIR_SECTOR"
    cd $DEST_DIR_SECTOR
    bash *.sh
    rm tesscurl*
    mv "$sector_shfile" "$TARGET_SH_DIR"/completed

done

# set permissions and group
if [[ $CHANGE_PERMISSIONS_AND_GROUP == true ]]

then
  chgrp -R $GROUP "$DEST_DIR"
  chmod -R 770 "$DEST_DIR"
fi
