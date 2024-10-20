# Download target pixel (tp) fits files to `TP_DIR` based on sh files in `SH_DIR`

# directory with tp sh files
SH_DIR="/nobackupp27/jochoa4/tp_sh_cleaned"

# destination directory
TP_DIR="/nobackupp27/jochoa4/TESS/fits_files/spoc_2min/tp"

for sector_shfile in "$SH_DIR"/*sector_*_tp.sh;
do
    X=$(basename "$sector_shfile" | sed -n 's|.*sector_\([0-9]*\)_tp.sh|\1|p') #get sector_num

    TP_SECTOR_DIR=$TP_DIR/sector_${X}
    
    echo "Downloading data for sector $TP_SECTOR_DIR"

    mkdir $TP_SECTOR_DIR
    if [ $? -eq 0 ]; then
        echo "Directory '$TP_SECTOR_DIR' created successfully."
    # echo $TP_DIR/sector_${X}
    else
        echo "Failed to create directory '$TP_SECTOR_DIR'"
    fi

    cp "$sector_shfile" "$TP_SECTOR_DIR"
    cd $TP_SECTOR_DIR
    bash *.sh
    rm tesscurl*
done
