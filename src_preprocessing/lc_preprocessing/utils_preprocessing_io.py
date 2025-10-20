""" I/O utility functions for the preprocessing pipeline. """

# 3rd party
import os
# import socket
import pandas as pd
import traceback

# local
from src_preprocessing.lc_preprocessing import utils_visualization, kepler_io, tess_io

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

    :param id_str: str, contains info on the cause of exclusion
    :param save_fp: str, file path to save the error log
    :param error_log: Error, exception error

    :return:
    """

    with open(save_fp, "a") as excl_file:

        excl_file.write(f'Info: {id_str}\n')
        if error_log:
            excl_file.write(f'Traceback for the exception:\n')
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

def read_light_curve(target_dict, config):
    """ Reads the FITS files pertaining to a Kepler/TESS target.

    Args:
        target_dict: target dictionary with ID (KIC/TIC), and also 'sector_run' and 'sectors_observed' for TESS
        config: Config object, preprocessing parameters

    Returns: dictionary with data extracted from the FITS files
        all_time: A list of numpy arrays; the time values of the raw light curve
        all_flux: A list of numpy arrays corresponding to the PDC flux time series
        all_centroid: A list of numpy arrays corresponding to the raw centroid time series
        add_info: A dict with additional data extracted from the FITS files; 'quarter' is a list of quarters for each
        NumPy
        array of the light curve; 'module' is the same but for the module in which the target is in every quarter;
        'target position' is a list of two elements which correspond to the target star position, either in world
        (RA, Dec) or local CCD (x, y) pixel coordinates

    Raises:
        IOError: If the light curve files for this target cannot be found.
    """

    # gets data from the lc FITS files for the TCE's target star
    if config['satellite'] == 'kepler':  # Kepler

        # get lc FITS files for the respective target star
        file_names = kepler_io.kepler_filenames(config['lc_data_dir'],
                                                target_dict['target_id'],
                                                injected_group=config['injected_group'])

        if not file_names:
            raise FileNotFoundError(f'No available lightcurve FITS files in {config["lc_data_dir"]} for '
                                    f'KIC {target_dict["target_id"]}')

        fits_data, fits_files_not_read = kepler_io.read_kepler_light_curve(
            file_names,
            centroid_radec=not config['px_coordinates'],
            prefer_psfcentr=config['prefer_psfcentr'],
            light_curve_extension=config['light_curve_extension'],
            scramble_type=config['scramble_type'],
            cadence_no_quarters_tbl_fp=config['cadence_no_quarters_tbl_fp'],
            invert=config['invert'],
            dq_values_filter=config['dq_values_filter'],
            get_momentum_dump=config['get_momentum_dump'],
        )

    else:  # TESS

        # # get sectors for the run
        # if '-' in target_dict['sector_run']:
        #     s_sector, e_sector = [int(sector) for sector in target_dict['sector_run'].split('-')]
        # else:
        #     s_sector, e_sector = [int(target_dict['sector_run'])] * 2
        # sectors = range(s_sector, e_sector + 1)

        # get lc FITS files for the respective target star if it was observed for that modality in the given sectors
        if config['using_exominer_pipeline']:
            file_names = tess_io.get_tess_light_curve_files(config['lc_data_dir'], target_dict['target_id'], target_dict['sectors_observed'])
        else:
            if config['ffi_data']:
                file_names = tess_io.tess_ffi_filenames(config['lc_data_dir'], target_dict['target_id'], target_dict['sectors_observed'])
            else:
                file_names = tess_io.tess_filenames(config['lc_data_dir'], target_dict['target_id'], target_dict['sectors_observed'])

            if not file_names:

                raise FileNotFoundError(f'No available lightcurve FITS files in {config["lc_data_dir"]} for '
                                        f'TIC {target_dict["target_id"]}')

        fits_data, fits_files_not_read = tess_io.read_tess_light_curve(file_names,
                                                                       centroid_radec=not config['px_coordinates'],
                                                                       prefer_psfcentr=config['prefer_psfcentr'],
                                                                       light_curve_extension=config['light_curve_extension'],
                                                                       get_momentum_dump=config['get_momentum_dump'],
                                                                       dq_values_filter=config['dq_values_filter'],
                                                                       )

    if len(fits_files_not_read) > 0:
        raise IOError(f'FITS files not read correctly for target {target_dict["target_id"]}: {fits_files_not_read}')

    return fits_data
