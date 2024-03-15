# Download xml files using curl statements in sh files.

# directory with sh files
TARGET_SH_DIR=/data5/tess_project/Data/tess_spoc_ffi_data/lc/dv_target_list_sh/
# destination directory for lc files
DEST_DIR=/data5/tess_project/Data/tess_spoc_ffi_data/lc/fits_files

for sector_shfile in "$TARGET_SH_DIR"/*lc.sh
do 
    echo Sector run "${sector_shfile:46:-17}"
    cp "$sector_shfile" $DEST_DIR
    cd $DEST_DIR
    bash *.sh  # download files for sector run; sh script creates directory for sector run
    rm hlsp_tess-spoc*  # remove copied sh file
    cd ../
done

# set group and permissions for rwx in group
chgrp -R ar-gg-ti-tess-dsg DEST_DIR
chmod -R 770 $DEST_DIR