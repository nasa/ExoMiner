# Download xml files using curl statements in sh files.

# directory with sh files
TARGET_SH_DIR=/data5/tess_project/Data/tess_spoc_ffi_data/dv/target_sh_xml_only

for sector_shfile in "$TARGET_SH_DIR"/*.sh
do 
    echo Sector run "${sector_shfile:46:-17}"
    # mkdir fits_files/sector_${sector_shfile:46:-17}
    cp "$sector_shfile" xml_files/  # sector_${sector_shfile:46:-17}
    cd xml_files/  # sector_${sector_shfile:46:-17}
    bash *.sh
    rm hlsp_tess-spoc*
    cd ../
done
