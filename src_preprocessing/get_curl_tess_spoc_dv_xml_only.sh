# Create new sh script with curl commands only for xml files.

TARGET_SH_DIR=/Users/msaragoc/Downloads/tess_spoc_ffi_fits_sh/dv
DEST_DIR=/Users/msaragoc/Downloads/tess_spoc_ffi_fits_sh/dv_xml

for sector_shfile in "$TARGET_SH_DIR"/*dv.sh
do
   FILENAME_INDEX=$(echo $sector_shfile | grep -bo "hlsp_*" | grep -oe "[0-9]*")
   FILENAME=${sector_shfile:$FILENAME_INDEX}
   echo "$FILENAME"
   grep xml "$sector_shfile" > $DEST_DIR/"$FILENAME"
done

# set permissions and group when in ranokau
chgrp -R ar-gg-ti-tess-dsg $DEST_DIR
chmod -R 770 $DEST_DIR
