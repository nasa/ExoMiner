# Script used to aggregate all reports for a set of examples into a single folder

SRC_DIR=/Users/msaragoc/Projects/exoplanet_transit_classification/data/dv_reports/mastDownload/TESS/misclassified_ntps_8-8-2023/
DEST_DIR=/Users/msaragoc/Downloads/misclassified_ntps_8-8-2023

mkdir -p $DEST_DIR

for target_dir in $SRC_DIR*
do
    for report in $target_dir/*
    do
        cp $report $DEST_DIR
    done
done