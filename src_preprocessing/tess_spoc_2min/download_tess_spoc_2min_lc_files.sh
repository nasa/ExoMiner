# Download lc fits files to `LC_DIR` based on sh files in `SH_DIR`

# directory with lc sh files
SH_DIR=/data5/tess_project/Data/tess_spoc_2min_data/lc/dv_target_list_sh
# destination directory
LC_DIR=/data5/tess_project/Data/tess_spoc_2min_data/lc/fits_files

for sector_shfile in "$SH_DIR"/*lc.sh
do
    FILENAME_INDEX=$(echo "$sector_shfile" | grep -bo "sector*" | grep -oe "[0-9]*")
    SECTOR_RUN=${sector_shfile:$FILENAME_INDEX:-6}

    LC_SECTOR_DIR=$LC_DIR/$SECTOR_RUN
    echo "Downloading data for sector $SECTOR_RUN ..."
    mkdir -p "$LC_SECTOR_DIR"
    cp "$sector_shfile" "$LC_SECTOR_DIR"
    cd $LC_SECTOR_DIR
    bash *.sh
    rm tesscurl*
    cd ../
    mv "$sector_shfile" "$SH_DIR"/completed

    # set group and permissions for rwx in group
    chgrp -R ar-gg-ti-tess-dsg "$LC_SECTOR_DIR"
    chmod -R 770 "$LC_SECTOR_DIR"
done
