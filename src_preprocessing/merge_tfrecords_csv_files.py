""" Concatenate tables for TFRecords files. Keeps track of examples stored in each file. """

# 3rd party
from pathlib import Path
import pandas as pd


def create_shards_table(srcTfrecDir):
    """ Create table that merges tables for each TFRecord file. Keeps track of examples stored in each file.

    :param srcTfrecDir: str, source TFRecord directory filepath
    :return:
    """

    srcTfrecDir = Path(srcTfrecDir)

    # get csv files for each TFRecord file
    srcTfrecTblsFps = sorted([file for file in srcTfrecDir.iterdir() if file.suffix == '.csv' and
                              file.stem.startswith('shard')])

    # concatenate TFRecord tables
    srcTfrecTblMerge = pd.concat([pd.read_csv(srcTfrecTblFp) for srcTfrecTblFp in srcTfrecTblsFps])

    srcTfrecTblMerge.to_csv(srcTfrecDir / 'merged_shards.csv', index=False)


if __name__ == "__main__":
    # source TFRecord directory
    srcTfrecDir = Path(
        '/data5/tess_project/Data/tfrecords/Kepler/Q1-Q17_DR25/tfrecordskeplerdr25-dv_g301-l31_spline_nongapped_newvalpcs_tessfeaturesadjs_12-1-2021_data/tfrecordskeplerdr25-dv_g301-l31_spline_nongapped_newvalpcs_tessfeaturesadjs_12-1-2021')

    create_shards_table(srcTfrecDir)
