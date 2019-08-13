"""
Utility functions for data manipulation.
"""

# 3rd party
import pandas as pd


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


if __name__ == '__main__':

    save_fp = '/home/msaragoc/Downloads/q1_q17_dr25_tce_2019.08.07_16.23.32_updt_tcert.csv'

    upd_labels_table = pd.read_csv('/home/msaragoc/Downloads/dr25_koi.csv', header=18)
    tce_table = pd.read_csv('/home/msaragoc/Downloads/q1_q17_dr25_tce_2019.08.07_16.23.32.csv', header=16)

    print('Saving updated TCE table to file {}'.format(save_fp))

    upd_tce_table = update_labels_tcetable(tce_table, upd_labels_table)

    upd_tce_table.to_csv(save_fp)