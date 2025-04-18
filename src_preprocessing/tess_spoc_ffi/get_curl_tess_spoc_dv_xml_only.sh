# Create new sh script with curl commands only for xml files.

TARGET_SH_DIR=
DEST_DIR=
# set permissions and group if needed
CHANGE_PERMISSIONS_AND_GROUP=false
GROUP=

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

# set permissions and group
if [[ $CHANGE_PERMISSIONS_AND_GROUP == true ]]

then
  chgrp -R $GROUP $DEST_DIR
  chmod -R 770 $DEST_DIR
fi
