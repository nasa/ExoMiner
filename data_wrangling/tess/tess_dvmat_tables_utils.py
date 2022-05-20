""" Utility functions used to create TESS TCE tables from TESS DV TCE and TOI catalogs. """

# 3rd party
import numpy as np
import pandas as pd


def match_tces_to_tois(res_tbl_cols, sector, tce_tbl, toi_tbl, matching_tbl):
    """ Create a TCE table for a given sector run (single- or multi-sector run) where TCEs are matched to TOIs.

    :param res_tbl_cols: list, result TCE table columns
    :param sector: str, sector run
    :param tce_tbl: pandas DataFrame, TCE table for the sector run
    :param toi_tbl: pandas DataFrame, TOI table
    :param matching_tbl: pandas DataFrame, TOI-TCE matching table
    :return:
        res_tbl: pandas DataFrame, result TCE table with TCEs matched to TOIs for the sector run
    """

    res_tbl = pd.DataFrame(columns=res_tbl_cols)

    for tce_i, tce in tce_tbl.iterrows():

        if tce_i % 50 == 0:
            print(f'[Sector {sector}] Iterating over TCE {tce_i + 1}/{len(tce_tbl)}')

        tce['tce_sectors'] = sector
        for col in toi_tbl.columns:
            tce[col] = np.nan
        tce['match_dist'] = np.nan

        # get TOIs from the same TIC as the TCE
        tois_tic = matching_tbl.loc[matching_tbl['TIC'] == tce['target_id']]

        if len(tois_tic) > 0:

            toi_for_this_tce = (np.nan, np.inf)

            for toi_i, toi in tois_tic.iterrows():

                # get TCEs in the same run that are matched to the TOI
                tces_toi_sector = [(float(tce_toi.split('_')[1]), tce_toi_i)
                                   for tce_toi_i, tce_toi in enumerate(toi['Matched TCEs'].split(' '))
                                   if tce_toi.split('_')[0] == sector]

                # check if this TCE is the closest match to the TOI in this run
                if len(tces_toi_sector) > 0 and tces_toi_sector[0][0] == tce['tce_plnt_num']:

                    # check if this TOI is the closest match to the TCE so far
                    if toi[f'matching_dist_{tces_toi_sector[0][1]}'] < toi_for_this_tce[1]:
                        toi_for_this_tce = (toi['TOI ID'], toi[f'matching_dist_{tces_toi_sector[0][1]}'])

            if not np.isnan(toi_for_this_tce[0]):  # if there is a matched TOI to the TCE, add the TOI data to the TCE
                matched_toi = toi_tbl.loc[toi_tbl['TOI'] == toi_for_this_tce[0]]

                for col in toi_tbl.columns:
                    tce[col] = matched_toi[col].item()

                tce[f'match_dist'] = toi_for_this_tce[1]

        # add TCE to TCE table
        res_tbl = res_tbl.append(tce[res_tbl_cols], ignore_index=True)

    return res_tbl


def _map_to_sector_string(sector_run):
    """ Map set of sectors to string.

    :param sector_run: str, sector string
    :return:
        str, set of sectors mapped to string
    """

    if '-' not in sector_run.values[0]:
        return sector_run.values[0]
    else:
        s_sector, e_sector = sector_run.values[0].split('-')
        multi_sector = [str(el) for el in list(range(int(s_sector), int(e_sector) + 1))]
        return ' '.join(multi_sector)
