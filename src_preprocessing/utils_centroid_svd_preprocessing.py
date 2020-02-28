import numpy as np
import os


def _has_finite(array):
    for i in array:
        if np.isfinite(i):
            return True

    return False


def minimizer(x, design_matrix_col, u_trunc, lambda_reg, reg_func):
    # print(np.linalg.norm(design_matrix_col - np.dot(u_trunc, x), ord=2))

    return np.linalg.norm(design_matrix_col - np.dot(u_trunc, x), ord=2) + lambda_reg * reg_func(x)


def l2_reg(x):
    return np.linalg.norm(x, ord=2)


def l1_reg(x):
    return np.linalg.norm(x, ord=1)


def report_exclusion(fits_id, fits_filep, id_str, savedir, stderr=None):
    """ Creates txt file with information regarding the exclusion of the processing of the fits file.

    :param fits_id: dict, ID description of the data related to the error
    :param fits_filep: str, filepath of the fits file being read
    :param id_str: str, contains info on the cause of exclusion
    :param savedir: str, filepath to directory in where the exclusion logs are saved
    :param stderr: str, error
    :return:
    """

    # # if is_pfe():
    #
    # # create path to exclusion logs directory
    # savedir = os.path.join(config.output_dir, 'exclusion_logs')
    # # create exclusion logs directory if it does not exist
    # os.makedirs(savedir, exist_ok=True)

    # if is_pfe():
    #
    #     # get node id
    #     node_id = socket.gethostbyname(socket.gethostname()).split('.')[-1]
    #
    #     # write to exclusion log pertaining to this process and node
    #     with open(os.path.join(savedir, 'exclusions_%d_%s.txt' % (config.process_i, node_id)), "a") as myfile:
    #         myfile.write('kepid: {}, tce_n: {}, {}\n{}'.format(tce.kepid, tce.tce_plnt_num, id_str,
    #                                                          (stderr, '')[stderr is None]))
    # else:
    #
    # write to exclusion log pertaining to this process and node
    with open(os.path.join(savedir, 'exclusions_{}.txt'.format(fits_id)), "a") as myfile:
        myfile.write('ID: {}: {} | Error {}\n {}'.format(fits_id, id_str, (stderr, 'None')[stderr is None],
                                                         fits_filep))
