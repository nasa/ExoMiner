# Download lc fits files to `LC_DIR` based on sh files in `SH_DIR`

# directory with lc sh files
SH_DIR=
# destination directory
LC_DIR=
# set permissions and group if needed
CHANGE_PERMISSIONS_AND_GROUP=false
GROUP=null

mkdir -p "$SH_DIR"/completed

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

done

echo "Finished downloading light curve FITS files"

# set permissions and group
if [[ $CHANGE_PERMISSIONS_AND_GROUP == true ]]

then
  chgrp -R $GROUP "$DEST_DIR"
  chmod -R 770 "$DEST_DIR"
fi