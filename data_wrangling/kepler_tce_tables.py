# 3rd party
import pandas as pd
import numpy as np
import random

#%% Add KOI dispositions; need to match KOIs to TCEs

koiTbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/kois_tables/cumulative_2020.02.21_10.29.22.csv',
                     header=90)

# select columns to add from the KOI table to the TCE table
columnsToAdd = ['kepoi_name', 'kepler_name', 'koi_disposition', 'koi_score',  'koi_fpflag_nt', 'koi_fpflag_ss',
                'koi_fpflag_co', 'koi_fpflag_ec', 'koi_comment', 'koi_period', 'koi_time0bk', 'koi_eccen', 'koi_impact',
                'koi_duration', 'koi_ingress', 'koi_depth', 'koi_ror', 'koi_prad', 'koi_sma', 'koi_incl', 'koi_teq',
                'koi_insol', 'koi_dor', 'koi_max_sngle_ev', 'koi_max_mult_ev', 'koi_model_snr', 'koi_count',
                'koi_num_transits', 'koi_tce_plnt_num', 'koi_tce_delivname', 'koi_datalink_dvr', 'koi_datalink_dvs']

# get only KOIs associated with Q1-Q17 DR25 TCE list
koiTbl = koiTbl.loc[koiTbl['koi_tce_delivname'] == 'q1_q17_dr25_tce']
koiTbl.reset_index(drop=True, inplace=True)

tceTbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17_DR25/'
                     'q1_q17_dr25_tce_2020.09.15_15.12.12_stellar.csv')

# initialize KOI columns in the TCE table
tceTbl = pd.concat([tceTbl, pd.DataFrame(columns=columnsToAdd)], axis=1)

# iterate through the Kepler IDs in the KOI table
print('Matching {} KOIs in the Cumulative KOI list to {} TCEs in the TCE table'.format(len(koiTbl), len(tceTbl)))
for koi_i, koi in koiTbl.iterrows():

    if koi_i % 1000 == 0:
        print('Matching KOI {} out of {} KOIs'.format(koi_i, len(koiTbl)))

    # add KOI parameters to the matched TCE
    tce_match = (tceTbl['kepid'] == koi['kepid']) & \
                (tceTbl['tce_plnt_num'] == int(koi['koi_datalink_dvs'].split('-')[1]))
    tceTbl.loc[tce_match, columnsToAdd] = koi[columnsToAdd].values

    # should have found exactly one match
    assert tce_match.sum() == 1

tceTbl.to_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17_DR25/'
              'q1_q17_dr25_tce_2020.09.15_15.12.12_stellar_koi.csv', index=False)

#%% Add FPWG dispositions; need to match KOIs from CFP table to KOIs from cumulative list

cfpTbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/kois_tables/fpwg_2020.03.13_11.37.49.csv',
                     header=13)

# select columns to add from the CFP table to the TCE table
columnsToAdd = ['fpwg_disp_status', 'fpwg_disp_source', 'fpwg_disp_eb', 'fpwg_disp_offst', 'fpwg_disp_perep',
                'fpwg_disp_other']

tceTbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17_DR25/'
                     'q1_q17_dr25_tce_2020.09.15_15.12.12_stellar_koi.csv')

# initialize KOI columns in the TCE table
tceTbl = pd.concat([tceTbl, pd.DataFrame(columns=columnsToAdd)], axis=1)

# iterate through the Kepler IDs in the KOI table
koiMatched = 0
print('Matching {} KOIs in the CFP table to {} TCEs in the TCE table'.format(len(cfpTbl), len(tceTbl)))
for koi_i, koi in cfpTbl.iterrows():

    if koi_i % 1000 == 0:
        print('Matching KOI {} out of {} KOIs'.format(koi_i, len(koiTbl)))

    # add KOI parameters to the matched TCE
    tce_match = (tceTbl['kepoi_name'] == koi['kepoi_name'])

    if tce_match.sum() == 1:
        tceTbl.loc[tce_match, columnsToAdd] = koi[columnsToAdd].values
        koiMatched += 1

print('Number of KOI matched: {}'.format(koiMatched))

tceTbl.to_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17_DR25/'
              'q1_q17_dr25_tce_2020.09.15_15.12.12_stellar_koi_cfp.csv', index=False)

#%% Define the labels for each TCE in the TCE table

# Update labels using CFP list  PPs (PC), CFPs (non-PC) and CFAs (non-PC) KOIs, and Confirmed KOIs (PC) from the
# Cumulative KOI list

tceTbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17_DR25/'
                     'q1_q17_dr25_tce_2020.09.15_15.12.12_stellar_koi_cfp.csv')
print('Original number of TCEs: {}'.format(len(tceTbl)))

# initialize TCE labels as NTP
tceTbl['label'] = 'NTP'

# load Certified False Positive list
cfpTbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/kois_tables/fpwg_2020.03.13_11.37.49.csv',
                     header=13)

map_cfpdisp_to_label = {'CERTIFIED FP': 'AFP', 'CERTIFIED FA': 'NTP', 'POSSIBLE PLANET': 'PC'}


def _cfpdisp_to_label(row, mapDict):

    if isinstance(row['fpwg_disp_status'], str) and row['fpwg_disp_status'] in mapDict:
        return mapDict[row['fpwg_disp_status']]
    else:
        return row['label']


tceTbl['label'] = tceTbl[['label', 'fpwg_disp_status']].apply(_cfpdisp_to_label, axis=1, args=(map_cfpdisp_to_label,))

# load Cumulative KOI list
cumKoiTbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/kois_tables/'
                        'cumulative_2020.02.21_10.29.22.csv', header=90)

map_cumkoidisp_to_label = {'CONFIRMED': 'PC'}


def _cumkoidisp_to_label(row, mapDict):

    if isinstance(row['koi_disposition'], str) and row['koi_disposition'] in mapDict:
        return mapDict[row['koi_disposition']]
    else:
        return row['label']


tceTbl['label'] = tceTbl[['label', 'koi_disposition']].apply(_cumkoidisp_to_label, axis=1,
                                                             args=(map_cumkoidisp_to_label,))

tceTbl.to_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17_DR25/'
              'q1_q17_dr25_tce_2020.09.15_15.12.12_stellar_koi_cfp_norobovetterlabels.csv', index=False)

#%% Rename columns in Q1-Q17 DR25 TCE list

# rawTceTable = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17 DR25/'
#                           'q1_q17_dr25_tce_2020.04.15_23.19.10_cumkoi_2020.02.21.csv')
tceTbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17_DR25/'
                     'q1_q17_dr25_tce_2020.09.15_15.12.12_stellar_koi_cfp_norobovetterlabels.csv')

# columns to be renamed
rawFields = ['kepid', 'kepmag', 'tce_depth']
newFields = ['target_id', 'mag', 'transit_depth']

# rename fields to standardize fieldnames
renameDict = {}
for i in range(len(rawFields)):
    renameDict[rawFields[i]] = newFields[i]
tceTbl.rename(columns=renameDict, inplace=True)

tceTbl.to_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17_DR25/'
              'q1_q17_dr25_tce_2020.09.15_15.12.12_stellar_koi_cfp_norobovetterlabels_renamedcols.csv',
              index=False)

#%% Filter out CANDIDATE and FALSE POSITIVE KOIs dispositioned by Robovetter

tceTbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17_DR25/'
                     'q1_q17_dr25_tce_2020.09.15_15.12.12_stellar_koi_cfp_norobovetterlabels_renamedcols.csv')

# keep only CONFIRMED KOIs, CFP, CFA and POSSIBLE PLANET KOIs, and non-KOI
tceTbl = tceTbl.loc[(tceTbl['koi_disposition'] == 'CONFIRMED') |
                    (tceTbl['fpwg_disp_status'].isin(['CERTIFIED FP', 'CERTIFIED FA', 'POSSIBLE PLANET'])) |
                    tceTbl['kepoi_name'].isna()]

print('Number of TCEs after removing KOIs dispositioned by Robovetter: {}'.format(len(tceTbl)))
print(tceTbl['label'].value_counts())

tceTbl.to_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17_DR25/'
              'q1_q17_dr25_tce_2020.09.15_15.12.12_stellar_koi_cfp_norobovetterlabels_renamedcols_rmcandandfpkois.csv',
              index=False)

#%% Check if there are  TCEs with missing values for the required columns

tceTbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17_DR25/'
                     'q1_q17_dr25_tce_2020.09.15_15.12.12_stellar_koi_cfp_norobovetterlabels_renamedcols_'
                     'rmcandandfpkois.csv')

numTotalTces = len(tceTbl)

colsToCheck = ['tce_duration', 'tce_time0bk', 'tce_period', 'transit_depth', 'ra', 'dec', 'tce_steff', 'tce_slogg',
               'tce_smet', 'tce_sradius', 'boot_fap', 'tce_smass', 'tce_sdens', 'tce_cap_stat', 'tce_hap_stat',
               'tce_rb_tcount0', 'tce_maxmesd', 'tce_max_mult_ev', 'tce_dikco_msky', 'tce_dicco_msky', 'tce_maxmes']

# remove TCEs with any NaN in the required fields
tceTbl.dropna(axis=0, subset=colsToCheck, inplace=True)

print('Number of TCEs removed: {}'.format(numTotalTces - len(tceTbl)))

# remove TCEs with zero period or transit duration
# rawTceTable = rawTceTable.loc[(rawTceTable['tce_period'] > 0) & (rawTceTable['tce_duration'] > 0)]
# print(len(rawTceTable))

# 34032 to 34032 TCEs
# rawTceTable.to_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17 DR25/'
#                    'q1_q17_dr25_tce_2020.04.15_23.19.10_cumkoi_2020.02.21_processed.csv',
#                    index=False)

#%% Filter rogue TCEs in Q1-Q17 DR25

tceTbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17_DR25/'
                     'q1_q17_dr25_tce_2020.09.15_15.12.12_stellar_koi_cfp_norobovetterlabels_renamedcols_'
                     'rmcandandfpkois.csv')

numTotalTces = len(tceTbl)

tceTbl = tceTbl.loc[tceTbl['tce_rogue_flag'] == 0]

print('Number of rogue TCEs removed from the TCE table: {}'.format(numTotalTces - len(tceTbl)))

tceTbl.to_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17_DR25/'
              'q1_q17_dr25_tce_2020.09.15_15.12.12_stellar_koi_cfp_norobovetterlabels_renamedcols_rmcandandfpkois_'
              'norogues.csv', index=False)

#%% Replace TCE ephemeris for KOI ephemeris

tceTbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17_DR25/'
                     'q1_q17_dr25_tce_2020.09.15_15.12.12_stellar_koi_cfp_norobovetterlabels_renamedcols.csv')


def _replace_tce_for_koi_ephem(row):

    if isinstance(row['kepoi_name'], str):
        row[['tce_period', 'tce_duration', 'tce_time0bk']] = row[['koi_period', 'koi_duration', 'koi_time0bk']].values

    return row[['tce_period', 'tce_duration', 'tce_time0bk']]


tceTbl[['tce_period', 'tce_duration', 'tce_time0bk']] = tceTbl.apply(_replace_tce_for_koi_ephem, axis=1)

assert len(np.where(tceTbl['tce_period'] == tceTbl['koi_period'])[0]) == len(tceTbl.loc[~tceTbl['kepoi_name'].isna()])

tceTbl.to_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17_DR25/q1_q17_dr25_tce_2020.09.15_15.12.12_stellar'
              '_koi_cfp_norobovetterlabels_renamedcols_koiephemeris.csv', index=False)

#%% Shuffle TCEs in Q1-Q17 DR25 TCE table

# np.random.seed(24)
random.seed(24)

tceTbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17_DR25/'
                     'q1_q17_dr25_tce_2020.09.15_15.12.12_stellar_koi_cfp_norobovetterlabels_renamedcols_'
                     'rmcandandfpkois_norogues.csv')

# shuffle at target star level
targetStarGroups = [df for _, df in tceTbl.groupby('target_id')]
random.shuffle(targetStarGroups)
tceTblShuffled = pd.concat(targetStarGroups).reset_index(drop=True)

# # shuffle at a TCE level
# tceTblShuffled = tceTbl.iloc[np.random.permutation(len(tceTbl))]

tceTblShuffled.to_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17_DR25/'
                      'q1_q17_dr25_tce_2020.09.15_15.12.12_stellar_koi_cfp_norobovetterlabels_renamedcols_'
                      'rmcandandfpkois_norogues_shuffle.csv', index=False)
