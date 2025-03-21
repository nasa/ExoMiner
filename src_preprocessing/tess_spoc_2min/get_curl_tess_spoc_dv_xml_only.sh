# Create new sh script with curl commands only for xml files.

TARGET_SH_DIR=/nobackupp19/msaragoc/work_dir/Kepler-TESS_exoplanet/data/FITS_files/TESS/spoc_2min/s81-s88_dv_sh/dv_xml/
DEST_DIR=/nobackupp19/msaragoc/work_dir/Kepler-TESS_exoplanet/data/FITS_files/TESS/spoc_2min/s81-s88_dv_sh/dv_xml_only/

mkdir -p "$TARGET_SH_DIR"/completed

for sector_shfile in "$TARGET_SH_DIR"/*dv.sh
do
   FILENAME_INDEX=$(echo "$sector_shfile" | grep -bo "tesscurl_*" | grep -oe "[0-9]*")
   FILENAME=${sector_shfile:$FILENAME_INDEX}
   echo "$FILENAME"
   grep xml "$sector_shfile" > $DEST_DIR/"$FILENAME"
   mv "$sector_shfile" "$TARGET_SH_DIR"/completed
done

# set permissions and group when in ranokau
chgrp -R ar-gg-ti-tess-dsg $DEST_DIR
chmod -R 770 $DEST_DIR
