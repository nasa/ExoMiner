""" Prepare TSO EB catalog to be used for EB disposition source - extract sector run from vetting information. """

# 3rd party
import re
import pandas as pd
from pathlib import Path

#%% Get sector run for each EB in the TSO EB catalog

tso_eb_catalog_fp = Path('/Users/msaragoc/Downloads/eb_list_tso_12-16-2021.csv')  # load TSO EB catalog
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
