# Create new sh script with curl commands only for xml files.

TARGET_SH_DIR=/data5/tess_project/Data/tess_spoc_ffi_data/dv/target_sh/
DEST_DIR=/data5/tess_project/Data/tess_spoc_ffi_data/dv/target_sh_xml_only/

for sector_shfile in "$TARGET_SH_DIR"/*dv.sh
do
   FILENAME=${sector_shfile:10}
   echo "$FILENAME"
   grep xml "$sector_shfile" > $DEST_DIR/"$FILENAME"
done

# set permissions and group
chgrp -R ar-gg-ti-tess-dsg $DEST_DIR
chmod -R 770 $DEST_DIR
