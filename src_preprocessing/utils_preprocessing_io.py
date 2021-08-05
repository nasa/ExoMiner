""" I/O utility functions for the preprocessing pipeline. """

# 3rd party
import os
import socket


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


def report_exclusion(config, tce, id_str, stderr=None):
    """ Error log is saved into a txt file with the reasons why a given TCE was not preprocessed.

    :param config: dict with parameters for the preprocessing. Check the Config class
    :param tce: Pandas Series, row of the input TCE table Pandas DataFrame.
    :param id_str: str, contains info on the cause of exclusion
    :param stderr: str, error
    :return:
    """

    # create path to exclusion logs directory
    savedir = os.path.join(config['output_dir'], 'exclusion_logs')

    # create exclusion logs directory if it does not exist
    os.makedirs(savedir, exist_ok=True)

    # TODO: what if TESS changes to multi-sector analysis; sector becomes irrelevant...
    if config['satellite'] == 'kepler':
        main_str = f'Kepler ID {tce.target_id} TCE {tce[config["tce_identifier"]]}'
    else:  # 'tess'
        main_str = f'TIC ID {tce.target_id} TCE {tce[config["tce_identifier"]]} Sector(s) {tce.sectors}'

    if is_pfe():

        # get node id
        node_id = socket.gethostbyname(socket.gethostname()).split('.')[-1]

        # write to exclusion log pertaining to this process and node
        with open(os.path.join(savedir, 'exclusions_{}_{}-{}.txt'.format(config['process_i'], node_id,
                                                                         main_str.replace(" ", ""))),
                  "a") as myfile:
            myfile.write('{}\n{}\n{}'.format(main_str, id_str, (stderr, '')[stderr is None]))
    else:
        # write to exclusion log locally
        with open(os.path.join(savedir, 'exclusions-{}.txt'.format(main_str.replace(" ", ""))), "a") as myfile:
            myfile.write('{}\n{}\n{}'.format(main_str, id_str, (stderr, '')[stderr is None]))
