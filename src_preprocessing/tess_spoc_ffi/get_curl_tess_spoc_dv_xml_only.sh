# Create new sh script with curl commands only for xml files.

TARGET_SH_DIR=/home6/msaragoc/work_dir/Kepler-TESS_exoplanet/data/FITS_files/TESS/spoc_ffi/dv/xml_files/target_sh
DEST_DIR=/home6/msaragoc/work_dir/Kepler-TESS_exoplanet/data/FITS_files/TESS/spoc_ffi/dv/xml_files/target_sh_xml_only

COMPLETED_DIR=$TARGET_SH_DIR/completed
mkdir -p $COMPLETED_DIR

for sector_shfile in "$TARGET_SH_DIR"/*dv.sh
do
   FILENAME_INDEX=$(echo "$sector_shfile" | grep -bo "hlsp_*" | grep -oe "[0-9]*")
   FILENAME=${sector_shfile:$FILENAME_INDEX}
   echo "$FILENAME"
   grep xml "$sector_shfile" > $DEST_DIR/"$FILENAME"
   mv "$sector_shfile" "$COMPLETED_DIR"
done

# set permissions and group when in ranokau
chgrp -R ar-gg-ti-tess-dsg $DEST_DIR
chmod -R 770 $DEST_DIR
