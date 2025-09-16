# Create new sh script with curl commands only for xml files.

TARGET_SH_DIR=/data3/exoplnt_dl/dv_xml/tesscurl_spoc/
DEST_DIR=/data3/exoplnt_dl/dv_xml/tesscurl_spoc/dv_xml_only/
# set permissions and group if needed
CHANGE_PERMISSIONS_AND_GROUP=false
GROUP=ar-gg-ti-tess-dsg

mkdir -p $DEST_DIR
mkdir -p "$TARGET_SH_DIR"/completed

for sector_shfile in "$TARGET_SH_DIR"/*dv.sh
do
   FILENAME=$(basename "$sector_shfile")
   echo "$FILENAME"
   grep xml "$sector_shfile" > $DEST_DIR/"$FILENAME"
   mv "$sector_shfile" "$TARGET_SH_DIR"/completed
done

echo "Finished filtering TESS curl sh files for DV XML files only."

# set permissions and group
if [[ $CHANGE_PERMISSIONS_AND_GROUP == true ]]

then
  chgrp -R $GROUP $DEST_DIR
  chmod -R 770 $DEST_DIR
fi