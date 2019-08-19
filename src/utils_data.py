"""
Utility functions for data manipulation.
"""

# 3rd party
import pandas as pd
import numpy as np


def update_labels_tcetable(tce_table, upd_table):
    """ Update the TCE table with new labels.

    :param tce_table: pandas DataFrame, original TCE table
    :param upd_table: pandas DataFrame, table with updated labels
    :return:
        tce_table: pandas DataFrame, TCE table with updated labels
    """

    tce_table['av_training_set'] = 'NTP'

    for i in range(len(upd_table)):

        comp = tce_table.loc[(tce_table['kepid'] == upd_table.iloc[i]['kepid']) &
                             (tce_table['tce_plnt_num'] == upd_table.iloc[i]['koi_tce_plnt_num'])]

        if len(comp) >= 2:
            raise ValueError('{} rows matched (same Kepler ID and TCE planet number)'.format(len(comp)))

        if len(comp) == 1:
            if upd_table.iloc[i]['koi_disposition'] in ['CONFIRMED', 'CANDIDATE']:
                print('Setting label to PC')
                tce_table.loc[comp.index, 'av_training_set'] = 'PC'
            else:
                print('Setting label to AFP')
                tce_table.loc[comp.index, 'av_training_set'] = 'AFP'

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


if __name__ == '__main__':

    # save_fp = '/home/msaragoc/Downloads/q1_q17_dr25_tce_2019.08.07_16.23.32_updt_tcert.csv'
    #
    # upd_labels_table = pd.read_csv('/home/msaragoc/Downloads/dr25_koi.csv', header=18)
    # tce_table = pd.read_csv('/home/msaragoc/Downloads/q1_q17_dr25_tce_2019.08.07_16.23.32.csv', header=16)
    #
    # print('Saving updated TCE table to file {}'.format(save_fp))
    #
    # upd_tce_table = update_labels_tcetable(tce_table, upd_labels_table)
    #
    # upd_tce_table.to_csv(save_fp)

    ###############################3

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
    pc_list = ['10666592_1']
    koi_tbl = '/home/msaragoc/Downloads/q1_q17_dr25_koi_2019.08.16_13.46.54.csv'
    print(get_confirmedpcs(koi_tbl, pc_list))
