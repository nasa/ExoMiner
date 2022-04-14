""" Concatenate tables for TFRecords files. Keeps track of examples stored in each file. """

# 3rd party
from pathlib import Path
import pandas as pd


def create_shards_table(srcTfrecDir):
    """ Create table that merges tables for each TFRecord file. Keeps track of examples stored in each file.

    :param srcTfrecDir: str, source TFRecord directory filepath
    :return:
        bool, True if the table was created successfully
    """

    srcTfrecDir = Path(srcTfrecDir)

    # get csv files for each TFRecord file
    srcTfrecTblsFps = sorted([file for file in srcTfrecDir.iterdir() if file.suffix == '.csv' and
                              file.stem.startswith('shard')])

    if len(srcTfrecTblsFps) == 0:  # no shard csv files found in the directory
        return False

    # concatenate TFRecord tables
    srcTfrecTblMerge = pd.concat([pd.read_csv(srcTfrecTblFp) for srcTfrecTblFp in srcTfrecTblsFps])

    srcTfrecTblMerge.to_csv(srcTfrecDir / 'merged_shards.csv', index=False)

    return True


if __name__ == "__main__":
    # source TFRecord directory
    srcTfrecDir = Path(
        '/data5/tess_project/Data/tfrecords/Kepler/Q1-Q17_DR25/tfrecordskeplerq1q17dr25-dv_g301-l31_5tr_spline_nongapped_4-6-2022_data/tfrecordskeplerq1q17dr25-dv_g301-l31_5tr_spline_nongapped_4-6-2022')

    create_shards_tbl_flag = create_shards_table(srcTfrecDir)

    print(f'Created shard table: {create_shards_tbl_flag}')
