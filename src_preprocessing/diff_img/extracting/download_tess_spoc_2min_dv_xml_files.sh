# Download xml files using curl statements in sh files.

# directory with sh files
TARGET_SH_DIR=/data5/tess_project/Data/tess_spoc_2min_data/dv/tesscurl_dv_xml_only
# destination directory for xml files
DEST_DIR=/data5/tess_project/Data/tess_spoc_2min_data/dv/dv_xml/sector_runs

# set permissions and group
chgrp -R ar-gg-ti-tess-dsg $DEST_DIR
chmod -R 770 $DEST_DIR

for sector_shfile in "$TARGET_SH_DIR"/tesscurl*dv.sh
do
    SECTOR_RUN=${sector_shfile:30:-6}
    if [[ $SECTOR_RUN == *"multi"* ]]
    then
        echo multi "$SECTOR_RUN"
        DEST_DIR_SECTOR=$DEST_DIR/multi-sector/"$SECTOR_RUN"
    else
        echo single $SECTOR_RUN
        DEST_DIR_SECTOR=$DEST_DIR/single-sector/"$SECTOR_RUN"
    fi

    mkdir -p "$DEST_DIR_SECTOR"
    cp "$sector_shfile" "$DEST_DIR_SECTOR"
    cd $DEST_DIR_SECTOR
    bash *.sh
    rm tesscurl*

    chgrp -R ar-gg-ti-tess-dsg "$DEST_DIR_SECTOR"
    chmod -R 770 "$DEST_DIR_SECTOR"

done
