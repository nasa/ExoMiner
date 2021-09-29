""" Concatenate tables for TFRecords files. """

# 3rd party
from pathlib import Path
import pandas as pd

# source TFRecord directory
srcTfrecDir = Path('/data5/tess_project/Data/tfrecords/'
                   'TESS/tfrecordstess-dv_g301-l31_5tr_spline_nongapped_s1-s40_09-21-2021_16-23_data/tfrecordstess-dv_g301-l31_5tr_spline_nongapped_s1-s40_09-21-2021_16-23')

# get csv files for each TFRecord file
srcTfrecTblsFps = sorted([file for file in srcTfrecDir.iterdir() if file.suffix == '.csv' and
                          file.stem.startswith('shard')])

# concatenate TFRecord tables
srcTfrecTblMerge = pd.concat([pd.read_csv(srcTfrecTblFp) for srcTfrecTblFp in srcTfrecTblsFps])

srcTfrecTblMerge.to_csv(srcTfrecDir / 'merged_shards.csv', index=False)
