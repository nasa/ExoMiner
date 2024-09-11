# Create new sh script with curl commands only for xml files.

TARGET_SH_DIR=/data5/tess_project/Data/tess_spoc_ffi_data/dv/target_sh
DEST_DIR=/data5/tess_project/Data/tess_spoc_ffi_data/dv/target_sh_xml_only

for sector_shfile in "$TARGET_SH_DIR"/*dv.sh
do
   FILENAME_INDEX=$(echo "$sector_shfile" | grep -bo "hlsp_*" | grep -oe "[0-9]*")
   FILENAME=${sector_shfile:$FILENAME_INDEX}
   echo "$FILENAME"
   grep xml "$sector_shfile" > $DEST_DIR/"$FILENAME"
   mv "$sector_shfile" "$TARGET_SH_DIR"/completed
done

# set permissions and group when in ranokau
chgrp -R ar-gg-ti-tess-dsg $DEST_DIR
chmod -R 770 $DEST_DIR
