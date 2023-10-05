""" Prepare TSO EB catalog to be used for EB disposition source - extract sector run from vetting information. """

# 3rd party
import re
import numpy as np
import pandas as pd
from pathlib import Path
import logging

#%% Get sector run for each EB in the TSO EB catalog

tso_eb_catalog_fp = Path('/Users/msaragoc/OneDrive - NASA/Projects/exoplanet_transit_classification/data/ephemeris_tables/tess/eb_catalogs/eb_catalog_tso/eb_list_tso_12-16-2021.csv')  # load TSO EB catalog
tso_eb_catalog_tbl = pd.read_csv(tso_eb_catalog_fp)

tso_eb_catalog_tbl['sector_run'] = 'N/A'  # initialize sector run column; default is no sector run

# reg1 = tso_eb_catalog_tbl['vetting'].str.contains('-s[0-9][0-9]-')


def _get_sector_run_reg1(x):
    """ Get sector run from the vetting column information.

    :param x: pandas Series, EB
    :return:
        str, sector run for EB (e.g. '1' for single-sector runs, '1-13' for multi-sector runs)
    """

    # test matching to different regex
    re_found_case1 = re.findall('-s[0-9][0-9]s[0-9][0-9]-', x['vetting'])
    re_found_case2 = re.findall('-s[0-9][0-9]-s[0-9][0-9]', x['vetting'])
    re_found_case3 = re.findall('-s[0-9][0-9]', x['vetting'])
    re_found_case4 = re.findall('sector-[0-9][0-9]', x['vetting'])
    re_found_case5 = re.findall('sector-[0-9]', x['vetting'])
    re_found_case6 = re.findall('^s12-', x['vetting'])

    if len(re_found_case1) == 1:
        try:
            return f'{re_found_case1[0][2:4].lstrip("0")}-{re_found_case1[0][5:7].lstrip("0")}'
        except:
            print(x['vetting'])
            aa
    elif len(re_found_case2) == 1:
        return f'{re_found_case2[0][2:4].lstrip("0")}-{re_found_case2[0][6:8].lstrip("0")}'
    elif len(re_found_case3) == 1:
        return re_found_case3[0][2:4].lstrip('0')
    elif len(re_found_case4) == 1:
        try:
            return f'{re_found_case4[0][7:9].lstrip("0")}'
        except:
            print(x['vetting'])
            aa
    elif len(re_found_case5) == 1:
        try:
            return f'{re_found_case5[0][7:8].lstrip("0")}'
        except:
            print(x['vetting'])
            aa
    elif len(re_found_case6) == 1:
        return  f'{re_found_case6[0][1:3].lstrip("0")}'
    else:
        # print(x)
        return x['sector_run']


# extract sector run from vetting information using set of regular expressions
tso_eb_catalog_tbl['sector_run'] = tso_eb_catalog_tbl[['vetting', 'sector_run']].apply(_get_sector_run_reg1, axis=1)

# get unique IDs for each EB in TSO EB catalog
tso_eb_catalog_tbl['UID'] = tso_eb_catalog_tbl[['tic_id', 'candidate_n', 'sector_run']].apply(lambda x: f'{x["tic_id"]}-s{x["sector_run"]}-{x["candidate_n"]}', axis=1)
tso_eb_catalog_tbl.drop_duplicates(subset='UID', keep='first', inplace=True)

# remove EBs whose sector run is N/A
tso_eb_catalog_tbl = tso_eb_catalog_tbl.loc[tso_eb_catalog_tbl['sector_run'] != 'N/A']

tso_eb_catalog_tbl.to_csv(tso_eb_catalog_fp.parent / f'{tso_eb_catalog_fp.stem}_processed.csv', index=False)

#%% Add to TCE table TSO EB match

tce_tbl_fp = Path('/Users/msaragoc/Downloads/tess_tces_s1-s40_11-23-2021_1409_stellarparams_updated_tecfluxtriage_eb_label_tecsec.csv')
tce_tbl = pd.read_csv(tce_tbl_fp)
tso_eb_catalog_tbl = pd.read_csv('/Users/msaragoc/Downloads/eb_list_tso_12-16-2021_processed.csv')
tso_eb_catalog_tbl.rename(columns={'tic_id': 'target_id', 'candidate_n': 'tce_plnt_num'}, inplace=True)

tso_eb_catalog_tbl['in_tso_eb_catalog'] = 'yes'
# tce_tbl['in_tso_eb_catalog'] = 'no'

# get unique IDs for each EB in TSO EB catalog
tso_eb_catalog_tbl['UID'] = tso_eb_catalog_tbl[['target_id', 'tce_plnt_num', 'sector_run']].apply(lambda x: f'{x["target_id"]}-s{x["sector_run"]}-{x["tce_plnt_num"]}', axis=1)
tso_eb_catalog_tbl.drop_duplicates(subset='UID', keep='first', inplace=True)

# remove EBs that do not have a valid sector run
tso_eb_catalog_tbl = tso_eb_catalog_tbl[tso_eb_catalog_tbl['sector_run'] != 'N/A']

tce_tbl_tso = tce_tbl.merge(tso_eb_catalog_tbl[['target_id', 'tce_plnt_num', 'sector_run', 'in_tso_eb_catalog']],
                            on=['target_id', 'tce_plnt_num', 'sector_run'], how='left', validate='one_to_one')

tce_tbl_tso.to_csv(tce_tbl_fp.parent / f'{tce_tbl_fp.stem}_tso.csv', index=False)

# check TCEs that are matched to both KP/CP TOIs and are also in the TSO EB catalog
cols = ['target_id', 'sector_run', 'tce_plnt_num', 'tce_period', 'Period (days)', 'tce_time0bk', 'Epoch (TBJD)', 'TOI', 'match_dist', 'TFOPWG Disposition', 'TESS Disposition', 'Comments']
pcs_ebs_matched = tce_tbl_tso.loc[(tce_tbl_tso['TFOPWG Disposition'].isin(['KP', 'CP'])) & (tce_tbl_tso['in_tso_eb_catalog'] == 'yes'), cols]
pcs_ebs_matched.to_csv(tce_tbl_fp.parent / 'pcs_ebs_matched.csv', index=False)

#%% Correct matching using Jon's table


def _create_uid_for_jon_tbl(x):
    """ Create UID for Jon's SPOC EB table that matched TSO EBs with SPOC TESS TCEs.

    :param x: pandas Series, EB
    :return:
        str, '{tic_id}-{tce_plnt_num}-{sector_run}'. E.g., TCE TIC 123456-1-S3 (single-sector run),
        TCE TIC 123456-2-S1-26 (multi-sector run)
    """

    sector_run = [int(el[1:]) for el in x['sectors'].split('-')]
    if len(sector_run) == 1:
        sector_run = f'S{sector_run[0]}'
    elif len(sector_run) == 2:
        sector_run = f'S{sector_run[0]}-{sector_run[1]}'
    else:
        raise ValueError(f'Sector run does not have the expected template.')

    return f'{x["ticid"]}-{x["tce_plnt_num"]}-{sector_run}'


res_dir = Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/data/ephemeris_tables/tess/DV_SPOC_mat_files/10-05-2022_1338')

# set up logger
logger = logging.getLogger(name='add_tso_ebs_to_tce_tbl')
logger_handler = logging.FileHandler(filename=res_dir / f'add_tso_ebs_to_tce_tbl.log', mode='w')
logger_formatter = logging.Formatter('%(asctime)s - %(message)s')
logger.setLevel(logging.INFO)
logger_handler.setFormatter(logger_formatter)
logger.addHandler(logger_handler)
logger.info(f'Starting run...')

tce_tbl_fp = res_dir / Path('tess_tces_dv_s1-s55_10-05-2022_1338_ticstellar_ruwe_tec.csv')
tce_tbl = pd.read_csv(tce_tbl_fp)
logger.info(f'Using TCE table {tce_tbl_fp}')

# add uid to Jon SPOC TCE-to-TSO EB table
jon_tbl_fp = Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/data/ephemeris_tables/tess/eb_catalogs/eb_catalog_tso/spocEBs.csv')
jon_tbl = pd.read_csv(jon_tbl_fp)
logger.info(f'Using TSO EB catalog in {jon_tbl_fp}')
jon_tbl['uid'] = jon_tbl.apply(_create_uid_for_jon_tbl, axis=1)
jon_tbl.drop_duplicates(subset='uid', inplace=True)

tce_tbl['in_jon_spoc_ebs'] = 'no'
tce_tbl.loc[tce_tbl['uid'].isin(jon_tbl['uid']), 'in_jon_spoc_ebs'] = 'yes'
# print(f'Number of TCEs previously matched to TSO EBs: {(tce_tbl["label_source"] == "TSO EB").sum()}')
# print(f'Number of SPOC EBs in the TCE table: {(tce_tbl["in_jon_spoc_ebs"] == "yes").sum()}')
# print(f'Number of TCEs mismatched to TSO EBs : {((tce_tbl["in_jon_spoc_ebs"] == "no") & (tce_tbl["label_source"] == "TSO EB")).sum()}')
# print(f'Number of TCEs correctly matched to TSO EBs : {((tce_tbl["in_jon_spoc_ebs"] == "yes") & (tce_tbl["label_source"] == "TSO EB")).sum()}')
logger.info(f'Number of TCEs in TSO EB catalog: {(tce_tbl["in_jon_spoc_ebs"] == "yes").sum()}')

# # set TCEs that are in Jon's SPOC TSO EBs and were not associated with a TOI to EBs from TSO
# tce_tbl.loc[(tce_tbl['in_jon_spoc_ebs'] == 'yes') & (~tce_tbl['label_source'].isin(['TFOPWG Disposition'])), ['label', 'label_source']] = 'EB', 'TSO EB'
# # set TCEs that are not in Jon's SPOC TSO EBs to missing label and label source
# tce_tbl.loc[(tce_tbl['in_jon_spoc_ebs'] == 'no') & (tce_tbl['label_source'] == 'TSO EB'), ['label', 'label_source']] = 'UNK', np.nan

tce_tbl.to_csv(tce_tbl_fp.parent / f'{tce_tbl_fp.stem}_tsoebs.csv', index=False)

logger.info('Saved TCE table with TSO EB results')

#%% Checking which misclassified TSO EBs were actually TSO EBs according to Jon's SPOC EB matching table to TSO EBs

misclf_tbl = pd.read_csv('/Users/msaragoc/Downloads/misclf_tess_ebs_jon_4-29-2022 JJ_5-12-2022.csv')


def _find_s_and_capitalize_it(x):
    """ Update TCE UID to match standard.

    :param x: pandas Series, EB
    :return:
        str, TESS SPOC TCE unique identifier, '{tic_id}-{tce_plnt_num}-{sector_run}'. E.g., TCE TIC 123456-1-S3
        (single-sector run), TCE TIC 123456-2-S1-26 (multi-sector run)
    """
    return x['uid'].replace('s', 'S')


misclf_tbl['uid'] = misclf_tbl.apply(_find_s_and_capitalize_it, axis=1)

misclf_tbl['in_jon_spoc_ebs'] = 'no'
misclf_tbl.loc[(misclf_tbl['uid'].isin(tce_tbl['uid']) & (tce_tbl['in_jon_spoc_ebs'] == 'yes')), 'in_jon_spoc_ebs'] = 'yes'
