# Download xml files using curl statements in sh files.

# directory with sh files
TARGET_SH_DIR=/data5/tess_project/Data/tess_spoc_ffi_data/dv/target_sh_xml_only
# destination directory for xml files
DEST_DIR=/data5/tess_project/Data/tess_spoc_ffi_data/dv/xml_files

for sector_shfile in "$TARGET_SH_DIR"/*.sh
do 
    echo Sector run "${sector_shfile:46:-17}"
    cp "$sector_shfile" $DEST_DIR
    cd $DEST_DIR
    bash *.sh  # download files for sector run
    rm hlsp_tess-spoc*  # remove copied sh file
    cd ../
done

# set permissions and group
chgrp -R ar-gg-ti-tess-dsg $DEST_DIR
chmod -R 770 $DEST_DIR