""" Utility functions for data manipulation. """

# 3rd party
import pandas as pd
import numpy as np
from astropy.io import fits
import os
from scipy.io import loadmat
import itertools


def crossref_ephem(ephem, ephem_upd):

    matched_tce = None

    return matched_tce


def update_labels_KeplerDR25(tce_table, upd_table):
    """ Update the TCE table with new labels.

    :param tce_table: pandas DataFrame, original TCE table
    :param upd_table: pandas DataFrame, table with updated labels
    :return:
        tce_table: pandas DataFrame, TCE table with updated labels
    """

    # # fill everything as NTP
    # tce_table['av_training_set'] = 'NTP'

    # iterate over the table with the labels
    for i in range(len(upd_table)):

        # comp = tce_table.loc[(tce_table['kepid'] == upd_table.iloc[i]['kepid']) &
        #                      (tce_table['tce_plnt_num'] == upd_table.iloc[i]['koi_tce_plnt_num'])]
        ephem = tce_table.loc[(tce_table['kepid'] == upd_table.iloc[i]['kepid'])][['tce_period', 'tce_time0bk',
                                                                                   'tce_duration']]
        ephem_upd = upd_table.iloc[i]['koi_period', 'koi_time0bk', 'koi_duration']


        matched_tce = crossref_ephem(ephem, ephem_upd)
        # if len(comp) >= 2:
        #     raise ValueError('{} rows matched (same Kepler ID and TCE planet number)'.format(len(comp)))

        # if len(comp) == 1:
        #     if upd_table.iloc[i]['koi_disposition'] in ['CONFIRMED', 'CANDIDATE']:
        #         # print('Setting label to PC')
        #         tce_table.loc[comp.index, 'av_training_set'] = 'PC'
        #     else:
        #         # print('Setting label to AFP')
        #         tce_table.loc[comp.index, 'av_training_set'] = 'AFP'

        if upd_table.iloc[i]['koi_disposition'] in ['CONFIRMED', 'CANDIDATE']:
            # print('Setting label to PC')
            tce_table.loc[(tce_table['kepid'] == matched_tce['kepid']) &
                          (tce_table['tce_plnt_num'] == matched_tce['tce_plnt_num']), 'av_training_set'] = 'PC'
        else:
            # print('Setting label to AFP')
            tce_table.loc[(tce_table['kepid'] == matched_tce['kepid']) &
                          (tce_table['tce_plnt_num'] == matched_tce['tce_plnt_num']), 'av_training_set'] = 'AFP'

    return tce_table


def get_rankedtces_bykepid(inputranking, kepids, outputranking, fields=['kepid', 'MES', 'rank', 'output']):
    """ Selects TCEs based on their Kepler IDs from the ranking given as input and creates a csv file with the ranking
    for the selected TCEs and fields.

    :param inputranking: str, filepath to ranking in csv format
    :param kepids: list, Kepler ids of interest
    :param outputranking: str, filepath to new ranking with Kepler ids of interest
    :param fields: list, fields to be extracted from the input ranking csv file
    :return:
    """

    # read the csv score ranking file
    ranked_tces = pd.read_csv(inputranking)
    # ranked_tces = pd.read_csv(inputranking, header=1)  # for Laurent and Pedro's ranking using weighted loss trained models

    # Add ranking index to DataFrame
    ranked_tces['rank'] = np.arange(1, len(ranked_tces) + 1)

    # filter TCEs based on Kepler ID
    filt_ranked_tces = ranked_tces.loc[ranked_tces['kepid'].isin(kepids)]

    # get only the fields of interest
    filt_ranked_tces = filt_ranked_tces[fields]
    # filt_ranked_tces = filt_ranked_tces[['kepid', 'MES', 'rank', 'output']]  # for the new rankings
    # filt_ranked_tces = filt_ranked_tces[['kepid', 'MES', 'rank', 'ensemble mean']]  # for Laurent and Pedro's ranking using weighted loss trained models

    # save to a new csv file
    filt_ranked_tces.to_csv(outputranking, index=False)


def get_confirmedpcs(koi_tbl, pc_list):
    """ Cross-check TCEs against a KOI table (preferably the latest one...)

    :param koi_tbl: str, filepath to csv file with KOI table
    :param pc_list: list, TCEs to be checked against the KOI table. Each entry is a string kepid_tce_n, where kepid is
    the Kepler ID and tce_n is the TCE planet number
    :return:
        dict, with list of confirmed, false positives, candidates and not candidates according to the KOI table
    """

    pc_dict = {'confirmed': [], 'false positive': [], 'candidate': [], 'not candidate': pc_list}

    koi_tbl = pd.read_csv(koi_tbl, header=53)[['kepid', 'koi_tce_plnt_num', 'koi_disposition']]

    for pc in pc_list:
        pc_kepid, pc_tce_n = pc.split('_')
        pc_in_koi_tbl = koi_tbl.loc[(koi_tbl['kepid'] == int(pc_kepid)) & (koi_tbl['koi_tce_plnt_num'] == int(pc_tce_n))]

        if len(pc_in_koi_tbl) == 1:
            pc_dict['not candidate'].remove(pc)

            if pc_in_koi_tbl['koi_disposition'].item() == 'CONFIRMED':
                pc_dict['confirmed'].append(pc)
            elif pc_in_koi_tbl['koi_disposition'].item() == 'FALSE POSITIVE':
                pc_dict['false positive'].append(pc)
            elif pc_in_koi_tbl['koi_disposition'].item() == 'CANDIDATE':
                pc_dict['candidate'].append(pc)

        elif len(pc_in_koi_tbl) > 1:
            raise ValueError('There are {} entries in the KOI table that match the TCE {} Kepler {}'.format(
                len(pc_in_koi_tbl), pc_tce_n, pc_kepid))
        else:
            print('TCE {} in Kepler {} was not found in the database'.format(pc_tce_n, pc_kepid))

    return pc_dict


def get_kp_fits(kepid, lc_dir):
    """ Get Kepler magnitude for the Kepler ID from the fits files in the directory lc.

    :param kepid: int, Kepler ID
    :param lc_dir: str, fits files root directory
    :return:
        kp: float, Kepler magnitude for the given Kepler ID
    """

    # print('Kepler ID {}'.format(kepid))

    kepid_str = "{:09d}".format(kepid)

    # get light curve fits files paths that belong to the target Kepler ID
    lc_fits_path = lc_dir + kepid_str[:4] + '/' + kepid_str
    fitsfile = [os.path.join(lc_fits_path, filepath) for filepath in os.listdir(lc_fits_path)][0]

    with fits.open(fitsfile, mode="readonly") as hdulist:
        head = hdulist[0].header
        kp = head['KEPMAG']

    return kp


def get_kp_tps(mat_file):
    """ Get Kepler magnitude for the entire set of Kepler IDs in the mat_file.

    :param mat_file: str, filepath to mat file (e.g. tpsTceStructV4_KSOP2536.mat)
    :return:
        dict, keys are Kepler IDs (int) and values are the respective Kepler magnitudes (float)
    """

    mat = loadmat(mat_file)['tpsTceStructV4_KSOP2536'][0][0]

    kepid_arr, kp_arr = np.squeeze(mat['keplerId']), np.squeeze(mat['keplerMag'])

    return {kepid_arr[i]: kp_arr[i] for i in range(len(kepid_arr))}


def get_dv_ranking(kepid_list, dv_root_dir, save_dir):
    """ Extract parameters for a set of TCEs (identified by their Kepler IDs) and write them to a csv table.

    :param kepid_list: list, Kepler IDs of TCEs to get information from their DV runs
    :param dv_root_dir: str, DV root directory with a folder for each TCE
    :param save_dir: str, save directory
    :return:
    """

    fields_idxs = [3, 6, 7, 10, 15]
    fieldsnames = ['Rp (Earth radii)', 'Rs (Solar radii)', 'Transit duration (h)', 'Orbital Period (d)',
                   'Insolation Flux']
    subfields = ['', ' uncertainty']
    fields = [''.join(feature_name_tuple)
                      for feature_name_tuple in itertools.product(fieldsnames, subfields)]

    dvtable_dict = {field: [] for field in fields}

    other_fields = ['KepID', 'Kp', 'MES', 'SNR']
    for field in other_fields:
        if field == 'KepID':
            dvtable_dict[field] = kepid_list
        else:
            dvtable_dict[field] = []

    for kepid in kepid_list:

        kepid_str = "{:09d}".format(kepid)

        dv_mat_targetResults = loadmat(
            os.path.join(dv_root_dir,
                         'kepid-{}/'
                         'dv_post_fit_workspace.mat'.format(kepid_str)))['dvResultsStruct']['targetResults' \
                                                                                            'Struct'][0][0][0, 0]

        # dv_mat_targetResults_kp = dv_mat_targetResults['keplerMag'][0]['value'][0][0][0]
        dvtable_dict['Kp'].append(dv_mat_targetResults['keplerMag'][0]['value'][0][0][0])

        dvtable_dict['MES'].append(
            dv_mat_targetResults['planetResultsStruct']['planetCandidate'][0][0]['maxMultipleEventSigma'][0, 0][0, 0])

        dvtable_dict['SNR'].append(
            dv_mat_targetResults['planetResultsStruct']['allTransitsFit'][0, 0][0, 0]['modelFitSnr'][0, 0])

        dv_mat_targetResults_params = dv_mat_targetResults['planetResultsStruct']['allTransits' \
                                                                                  'Fit'][0, 0][0, 0]['modelParameters']
        for i in range(len(fields_idxs)):
            dvtable_dict[fieldsnames[i]].append(dv_mat_targetResults_params[0, fields_idxs[i]]['value'][0][0])
            dvtable_dict[fieldsnames[i] +
                         ' uncertainty'].append(dv_mat_targetResults_params[0, fields_idxs[i]]['uncertainty'][0][0])

    dvtable_df = pd.DataFrame(dvtable_dict)
    dvtable_df = dvtable_df[other_fields + fieldsnames]

    dvtable_df.to_csv('{}dvtable.csv'.format(save_dir), index=False)


if __name__ == '__main__':

    # UPDATE LABELS IN THE KEPLER TCE TABLE DOWNLOADED FROM THE NASA EXOPLANET ARCHIVE USING THE TCERT TABLE
    # save_fp = '/data5/tess_project/Data/Ephemeris_tables/q1_q17_dr25_tce_2019.08.20_11.34.45_updt_tcert.csv'
    #
    # upd_labels_table = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/dr25_koi.csv', header=18)
    # tce_table = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/q1_q17_dr25_tce_2019.08.20_11.34.45.csv',
    #                         header=34)
    #
    # print('Saving updated TCE table to file {}'.format(save_fp))
    #
    # upd_tce_table = update_labels_tcetable(tce_table, upd_labels_table)
    #
    # upd_tce_table.to_csv(save_fp)

    ###############################
    #
    # # inputranking = '/home/msaragoc/Projects/Kepler-TESS_exoplanet/Kepler_planet_finder/results_ensemble/' \
    # #                'study_bohb_dr25_tcert_spline2/180k_ES_300-p20_34k/ranked_predictions_predict'
    # inputranking = '/home/msaragoc/Projects/Kepler-TESS_exoplanet/Kepler_planet_finder/results_ensemble/' \
    #                'study_bohb_dr25_tcert_spline2/180k_ES_300-p20_34k//cand_list_batched.csv'
    #
    # kepids = [10666285, 6846844, 7269798, 5788044, 4070414, 5976620, 6024567, 6668663, 7117186, 7334793, 8105243,
    #               9520009, 10483732, 11904336, 2423008, 6288438, 6859648, 9962599, 10526615]
    #
    # outputranking = '/home/msaragoc/Projects/Kepler-TESS_exoplanet/Kepler_planet_finder/results_ensemble/'\
    #                 'study_bohb_dr25_tcert_spline2/180k_ES_300-p20_34k/ranked_predictions_predict_filteredweighted.csv'
    #
    # get_rankedtces_bykepid(inputranking, kepids, outputranking)

    # url = 'https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=cumulative'
    #
    # import wget
    # wget.download()
    # pc_list = ['10666592_1']
    # koi_tbl = '/home/msaragoc/Downloads/q1_q17_dr25_koi_2019.08.16_13.46.54.csv'
    # print(get_confirmedpcs(koi_tbl, pc_list))

    ###############################
    # # CREATE DICT WITH KEPLER MAGNITUDES FOR THE 200K KEPLER IDS
    # kp_dict = get_kp_tps('/data5/tess_project/Data/Ephemeris_tables/tpsTceStructV4_KSOP2536.mat')
    # np.save('/data5/tess_project/Data/Ephemeris_tables/kp_KSOP2536.npy', kp_dict)

    ###############################

    dv_root_dir = '/home/msaragoc/Projects/Kepler-TESS_exoplanet/DV/files_for_DV/'
    save_dir = ''

    kepid_list = [1433534, 9775362, 9274173, 6964115, 4562023, 6790250, 4373201, 10453635, 8609534, 4995716, 3962698,
                  12316241, 8491719, 7119412, 9823926]

    get_dv_ranking(kepid_list, dv_root_dir, save_dir)