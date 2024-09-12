# Download xml files using curl statements in sh files.

# directory with sh files
TARGET_SH_DIR=/data5/tess_project/Data/tess_spoc_2min_data/dv/target_sh_xml_only
# destination directory for xml files
DEST_DIR=/data5/tess_project/Data/tess_spoc_2min_data/dv/xml_files/sector_runs

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

    # set permissions and group
    chgrp -R ar-gg-ti-tess-dsg "$DEST_DIR_SECTOR"
    chmod -R 770 "$DEST_DIR_SECTOR"

done


