# Download lc fits files to `LC_DIR` based on sh files in `SH_DIR`

# directory with lc sh files
SH_DIR="/data5/tess_project/Data/tess_spoc_2min_data/lc/dv_target_list_sh_09-13-2023_1606"
# destination directory
LC_DIR="/data5/tess_project/Data/tess_spoc_2min_data/lc/fits_files"

# set group and permissions for rwx in group
chgrp -R ar-gg-ti-tess-dsg $LC_DIR
chmod -R 770 $LC_DIR

for sector_shfile in SH_DIR/*lc.sh
do
    LC_SECTOR_DIR=$LC_DIR/${sector_shfile:9:-6}
    echo "Downloading data for sector $LC_SECTOR_DIR ..."
    mkdir -p "$LC_SECTOR_DIR"
    cp "$sector_shfile" "$LC_SECTOR_DIR"
    cd $LC_SECTOR_DIR
    bash *.sh
    rm tesscurl*

    # set group and permissions for rwx in group
    chgrp -R ar-gg-ti-tess-dsg "$LC_SECTOR_DIR"
    chmod -R 770 "$LC_SECTOR_DIR"
done

