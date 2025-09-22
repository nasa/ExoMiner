#!/bin/bash

# Download TESS 2-min/HLSP FFI light curve FITS files using curl statements in sh files.
# Assumes the curl sh files are named "*lc.sh" and inside a directory $SH_DIR. 

SH_DIR="/data3/exoplnt_dl/lc_fits/tesscurl_sectors_lcs/download_missing_targets_sectors_lcs_s1-s92_9-16-2025_1257"
LC_DIR="/data3/exoplnt_dl/lc_fits/2-min"
CHANGE_PERMISSIONS_AND_GROUP=false
GROUP="ar-gg-ti-tess-dsg"
MAX_RETRIES=3
LOG_FILE=$SH_DIR/download_log_$(date +%Y%m%d_%H%M%S).log

mkdir -p "$SH_DIR/completed"
echo "📝 Starting download process at $(date)" | tee -a "$LOG_FILE"

for sector_shfile in "$SH_DIR"/*lc.sh; do
    # SECTOR_RUN=$(basename "$sector_shfile" | grep -oP 'sector\S*')
    SECTOR_RUN=$(basename "$sector_shfile" | sed -E 's/(_lc\.sh)$//')

    NUM_TARGETS=$(wc -l < "$sector_shfile")
    echo "🎯 $NUM_TARGETS target light curve FITS files to download for sector $SECTOR_RUN" | tee -a "$LOG_FILE"


    if [[ -z "$SECTOR_RUN" ]]; then
        echo "⚠️ Could not extract sector from $sector_shfile" | tee -a "$LOG_FILE"
        continue
    fi

    LC_SECTOR_DIR="$LC_DIR/$SECTOR_RUN"
    echo "📥 Downloading data for sector $SECTOR_RUN ..." | tee -a "$LOG_FILE"
    mkdir -p "$LC_SECTOR_DIR"
    cp "$sector_shfile" "$LC_SECTOR_DIR"

    pushd "$LC_SECTOR_DIR" > /dev/null

    RETRY_COUNT=0
    SUCCESS=false
    while [[ $RETRY_COUNT -lt $MAX_RETRIES ]]; do
        echo "🔄 Attempt $((RETRY_COUNT + 1)) for sector $SECTOR_RUN" | tee -a "$LOG_FILE"
        bash "$(basename "$sector_shfile")"
        if [[ $? -eq 0 ]]; then
            SUCCESS=true
            echo "✅ Sector $SECTOR_RUN downloaded successfully." | tee -a "$LOG_FILE"
            break
        else
            echo "❌ Attempt $((RETRY_COUNT + 1)) failed for sector $SECTOR_RUN." | tee -a "$LOG_FILE"
            ((RETRY_COUNT++))
            sleep 5
        fi
    done

    if [[ "$SUCCESS" == true ]]; then
        mv "$sector_shfile" "$SH_DIR/completed/"
        rm -f "$(basename "$sector_shfile")"  # Remove the copied .sh file

    else
        echo "🚫 Failed to download sector $SECTOR_RUN after $MAX_RETRIES attempts." | tee -a "$LOG_FILE"
    fi

    popd > /dev/null
done
echo "🏁 Finished downloading light curve FITS files at $(date)" | tee -a "$LOG_FILE"

if [[ "$CHANGE_PERMISSIONS_AND_GROUP" == true ]]; then
    echo "🔧 Setting group to $GROUP and permissions to 770 for $LC_DIR ..." | tee -a "$LOG_FILE"
    chgrp -R "$GROUP" "$LC_DIR"
    chmod -R 770 "$LC_DIR"
    echo "✅ Permissions and group updated." | tee -a "$LOG_FILE"
fi
