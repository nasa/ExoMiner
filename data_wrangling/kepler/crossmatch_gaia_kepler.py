"""
Get revised stellar effective temperature and radius from Gaia DR2 for KIC targets.
"""

# 3rd party
from astropy.table import Table
import msgpack


def check_duplicate_kepids():
    """
    Checks if kepid occurs multiple times in cross match file. If so, prints kepid with most occurrences, the number of
    occurrences and the total number of kepids which occur multiple times.

    :return:
    """
    pd_data = Table.read('/home/lswilken/Downloads/kepler_dr2_1arcsec.fits', format='fits').to_pandas()

    max_n = 1
    max_n_kepid = 0
    crossref_dict = {}
    for _, row in pd_data.iterrows():
        kepid = int(row['kepid'])
        info_i = {'radius': row['radius'], 'teff': row['teff'], 'ang_dist': row['kepler_gaia_ang_dist']}

        if kepid not in crossref_dict:
            crossref_dict[kepid] = info_i
        elif isinstance(crossref_dict[kepid], dict):
            crossref_dict[kepid] = [crossref_dict[kepid], info_i]
            max_n = 2
            max_n_kepid = kepid
        else:
            crossref_dict[kepid] = crossref_dict[kepid] + [info_i]
            max_n = len(crossref_dict[kepid])
            max_n_kepid = kepid

    print('highest duplicate n: %d, kepid: %d' % (max_n, max_n_kepid))

    mult_count = 0
    for kepid, info_i in crossref_dict.items():
        if isinstance(info_i, list):
            mult_count += 1

    print('Number of duplicate kepids: ' + str(mult_count))  # ~ 3529 kepids have two matched Gaia targets (1 arcsec)


def get_custom_crossmatch():
    """
    Loads downloaded Gaia dr2-Kepler dr25 target cross-referenced data kepler_dr2_1arcsec.fits and saves into easily
    readable python dict.
    Source of .fits file: https://gaia-kepler.fun/
    :return:
    """
    pd_data = Table.read('/home/lswilken/Downloads/kepler_dr2_1arcsec.fits', format='fits').to_pandas()

    # Create kepid-keyed dictionary with strictly necessary data and keep only best matched Gaia target per kepid
    crossref_dict = {}
    for _, row in pd_data.iterrows():
        kepid = int(row['kepid'])
        info_i = {'radius': row['radius'], 'teff': row['teff'], 'feh': row['feh'],
                  'ang_dist': row['kepler_gaia_ang_dist']}

        if kepid not in crossref_dict:
            crossref_dict[kepid] = info_i
        elif info_i['ang_dist'] < crossref_dict[kepid]['ang_dist']:
            crossref_dict[kepid] = info_i

    # Write msgpack file
    crossref_dict_file = '/home/lswilken/Documents/DV/structs/gaia_kepler_crossref_dict.msgp'

    with open(crossref_dict_file, 'wb') as outfile:
        msgpack.pack(crossref_dict, outfile)

    # # Read msgpack file
    # with open(crossref_dict_file, 'rb') as data_file:
    #     crossref_dict_loaded = msgpack.unpack(data_file)


def get_huber_crossmatch():
    """
    Loads downloaded Gaia dr2-Kepler dr25 target cross-referenced data from paper: Berger & Huber: Revised Radii of
    Kepler Stars and Planets Using Gaia Data Release 2 and saves into easily readable python dict.
    Paper: https://arxiv.org/pdf/1805.00231.pdf
    :return:
    """
    crossref_dict = {}
    with open('DR2PapTablfe1.txt', 'r') as file:
        cols_map = {str_id: i for i, str_id in enumerate(file.__next__().split('&'))}

        for line in file:
            col = line.split('&')
            kepid = int(col[cols_map['KIC']])

            assert kepid not in crossref_dict

            crossref_dict[kepid] = {'radius': col[cols_map['rad']], 'teff': col[cols_map['teff']]}

    # gaia_dr_dict = {}
    # with open('gaia_dr2_import.csv', 'r') as f:
    #     cols_map = {str_id: i for i, str_id in enumerate(f.__next__().split(','))}
    #
    #     for row in csv.reader(f, delimiter='\t'):
    #         gaia_dr_dict[row[cols_map['source_id']]] = {'teff': col[cols_map['teff_val']],
    #                                                     'radius': col[cols_map['radius_val']]}

    crossref_dict_file = '/home/lswilken/Documents/DV/structs/gaia_kepler_crossref_dict_huber.msgp'

    # Write msgpack file
    with open(crossref_dict_file, 'wb') as outfile:
        msgpack.pack(crossref_dict, outfile)

    # # Read msgpack file
    # with open(crossref_dict_file, 'rb') as data_file:
    #     crossref_dict_loaded = msgpack.unpack(data_file)


if __name__ == '__main__':
    # get_custom_crossmatch()

    get_huber_crossmatch()

    # # Read msgpack file
    # with open('/home/lswilken/Documents/DV/structs/gaia_kepler_crossref_dict_huber.msgp', 'rb') as data_file:
    #     crossref_dict_loaded = msgpack.unpack(data_file)
    # pass
