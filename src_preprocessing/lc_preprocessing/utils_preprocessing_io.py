""" I/O utility functions for the preprocessing pipeline. """

# 3rd party
import os
# import socket
import pandas as pd
import traceback


def is_pfe():
    """ Returns boolean which indicates whether this script is being run on Pleiades or local computer. """

    nodename = os.uname().nodename

    if nodename[:3] == 'pfe':
        return True

    if nodename[0] == 'r':
        try:
            int(nodename[-1])
            return True
        except ValueError:
            return False

    return False


def report_exclusion(id_str, save_fp, error_log=None):
    """ Error log is saved into a txt file with the reasons why a given TCE was not preprocessed.

    # :param config: dict with parameters for the preprocessing. Check the Config class
    # :param tce: Pandas Series, row of the input TCE table Pandas DataFrame.
    :param id_str: str, contains info on the cause of exclusion
    :param save_fp: str, file path to save the error log
    :param error_log: Error, exception error
    # :param error_log: dict, exception type, value, and traceback
    :return:
    """

    with open(save_fp, "a") as excl_file:
        # if first_exclusion:
        #     excl_file.write(uid_str)
        excl_file.write(f'Info: {id_str}\n')
        if error_log:
            excl_file.write(f'Traceback for the exception:\n')
            # traceback.print_exception(**error_log, file=excl_file)
            traceback.print_exception(error_log, file=excl_file)

        excl_file.write('##############\n')


def create_tbl_from_exclusion_logs(excl_fps, max_n_errors_logged):
    """ Create table for examples with exclusion logs that occurred while preprocessing the data.

    Args:
        excl_fps: list, file paths to exclusion logs.
        max_n_errors_logged: int, maximum number of logged exclusion logs.

    Returns: exclusion_tbl, pandas DataFrame, table with examples and corresponding exclusion events that occurred when
    preprocessing.

    """

    n_examples = len(excl_fps)

    data_to_tbl = {
        'uid': [''] * n_examples,
        'filename': [''] * n_examples,
    }

    for error_i in range(max_n_errors_logged):
        data_to_tbl[f'exclusion_{error_i}'] = [''] * n_examples
        data_to_tbl[f'error_{error_i}'] = [''] * n_examples

    for excl_fp_i, excl_fp in enumerate(excl_fps):

        data_to_tbl['filename'][excl_fp_i] = excl_fp.name

        with open(excl_fp, 'r') as excl_file:

            line = excl_file.readline()
            data_to_tbl['uid'][excl_fp_i] = line.split(' ')[1][:-1]

            cnt_lines = 0
            line = excl_file.readline()
            while line:
                if cnt_lines % 2 == 0:  # exclusion
                    data_to_tbl[f'exclusion_{cnt_lines // 2}'][excl_fp_i] = line[10:-1]
                elif cnt_lines % 2 == 1:  # error associated with exclusion
                    data_to_tbl[f'error_{(cnt_lines - 1) // 2}'][excl_fp_i] = line[6:-1]

                line = excl_file.readline()
                cnt_lines += 1

    exclusion_tbl = pd.DataFrame(data_to_tbl)

    return exclusion_tbl
