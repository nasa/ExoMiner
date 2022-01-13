"""
Preprocess TPS TCE table.
"""

# 3rd party
import numpy as np
import pandas as pd
from scipy import io

#%% Get TPS TCE table from mat file

tpsMatFp = '/data5/tess_project/Data/Kepler-Q1-Q17-DR25/tpsTceStructV4_KSOP2536.mat'
tpsMat = io.loadmat(tpsMatFp)['tpsTceStructV4_KSOP2536'][0][0]

tpsMatFields = list(tpsMat.dtype.fields)
tpsMatFields = [field for field in tpsMatFields if field not in ['topDir', 'planetCandidateStruct', 'unitOfWorkKjd',
                                                                 'pulseDurations', 'spsdIndices',
                                                                 'positiveOutlierIndices', 'backExponentialPpm',
                                                                 'frontExponentialPpm', 'indexOfSesAdded',
                                                                 'sesCombinedToYieldMes'#, 'quartersPresent'
                                                                 ]
                ]

numTces = len(tpsMat['keplerId'])

tceTpsTbl = pd.DataFrame(columns=tpsMatFields)

for tce_i in range(numTces):

    valuesVec = {}
    for field in tpsMatFields:
        if field in ['maxMesPulseDurationHours']:
            valuesVec[field] = tpsMat[field][0][tce_i]
        elif field in ['taskfile']:
            valuesVec[field] = tpsMat[field][tce_i][0][0]
        elif field in ['quartersPresent']:
            valuesVec[field] = str(tpsMat['quartersPresent'][tce_i])[1:-1]
        else:
            valuesVec[field] = tpsMat[field][tce_i][0]

    tceDf = pd.DataFrame(data=valuesVec, index=[0])

    tceTpsTbl = pd.concat([tceTpsTbl, tceDf])

tceTpsTbl.to_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/keplerTPS_KSOP2536.csv', index=False)

#%% Add stellar parameters to the TPS TCE table

stellar_tbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Stellar parameters/Kepler/'
                          'q1_q17_dr25_stellar_gaiadr2_nanstosolar.csv')

tceTpsTbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/TPS_tables/Q1-Q17 DR25/keplerTPS_KSOP2536.csv')

stellar_fields_out = ['kepmag', 'tce_steff', 'tce_steff_err1', 'tce_steff_err2', 'tce_slogg', 'tce_slogg_err1',
                      'tce_slogg_err2', 'tce_smet', 'tce_smet_err1', 'tce_smet_err2', 'tce_sradius', 'tce_sradius_err1',
                      'tce_sradius_err2', 'tce_smass', 'tce_smass_err1', 'tce_smass_err2', 'tce_sdens',
                      'tce_sdens_err1', 'tce_dens_serr2', 'ra', 'dec']
stellar_fields_in = ['kepmag', 'teff', 'teff_err1', 'teff_err2', 'logg', 'logg_err1', 'logg_err2', 'feh', 'feh_err1',
                     'feh_err2', 'radius', 'radius_err1', 'radius_err2', 'mass', 'mass_err1', 'mass_err2', 'dens',
                     'dens_err1', 'dens_err2', 'ra', 'dec']

# tce_tbl_cols = list(tce_tbl.columns)

for stellar_param in stellar_fields_out:
    # if stellar_param not in tce_tbl_cols:
        tceTpsTbl[stellar_param] = np.nan

count = 0
for row_star_i, row_star in stellar_tbl.iterrows():

    if row_star_i % 100 == 0:
        print('Star {} out of {} ({} %)\n Number of TCEs updated: {}'.format(row_star_i,
                                                                             len(stellar_tbl),
                                                                             row_star_i / len(stellar_tbl) * 100,
                                                                             count))

    target_cond = tceTpsTbl['keplerId'] == row_star['kepid']

    count += target_cond.sum()

    tceTpsTbl.loc[target_cond, stellar_fields_out] = row_star[stellar_fields_in].values

print('Number of TCEs updated: {}'.format(count))

print(tceTpsTbl[stellar_fields_out].isna().any(axis=0))

tceTpsTbl.to_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/keplerTPS_KSOP2536_stellar.csv', index=False)

#%% Standardize fields in the TPS TCE table

tceTpsTbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/TPS_tables/Q1-Q17 DR25/keplerTPS_KSOP2536_stellar.csv')

rawFields = ['keplerId', 'keplerMag', 'epochKjd', 'periodDays', 'fittedDepth', 'weakSecondaryPhase',
             'maxMesPulseDurationHours']
newFields = ['target_id', 'mag', 'tce_time0bk', 'tce_period', 'transit_depth', 'tce_maxmesd', 'tce_duration']

# # remove TCEs with any NaN in the required fields
# rawTceTable.dropna(axis=0, subset=np.array(rawFields)[[0, 1, 2, 3, 5, 7, 9, 11]], inplace=True)
print(len(tceTpsTbl))

# rename fields to standardize fieldnames
renameDict = {}
for i in range(len(rawFields)):
    renameDict[rawFields[i]] = newFields[i]
tceTpsTbl.rename(columns=renameDict, inplace=True)

# remove TCEs with zero period or transit duration
tceTpsTbl = tceTpsTbl.loc[(tceTpsTbl['tce_period'] > 0) & (tceTpsTbl['tce_duration'] > 0)]
print(len(tceTpsTbl))

# convert fractional transit depth to ppm
tceTpsTbl['transit_depth'] = tceTpsTbl['transit_depth'] * 1e6

tceTpsTbl.to_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/TPS_tables/Q1-Q17 DR25/keplerTPS_KSOP2536_processed.csv', index=False)

#%% Shuffle TPS table

np.random.seed(24)

tceTpsTbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/TPS_tables/Q1-Q17 DR25/keplerTPS_KSOP2536_processed.csv')

tceTpsTbl = tceTpsTbl.iloc[np.random.permutation(len(tceTpsTbl))]

tceTpsTbl.to_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/TPS_tables/Q1-Q17 DR25/keplerTPS_KSOP2536_shuffled.csv', index=False)

#%% Filter non-TCEs from the TPS table

tceTpsTbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/TPS_tables/Q1-Q17 DR25/keplerTPS_KSOP2536_shuffled.csv')

tceTpsTbl['tce_plnt_num'] = 1

nonTceTpsTbl = tceTpsTbl.loc[tceTpsTbl['isPlanetACandidate'] == 0]
nonTceTpsTbl['label'] = 'NTP'

tceTpsTbl = tceTpsTbl.loc[tceTpsTbl['isPlanetACandidate'] == 1]

nonTceTpsTbl.to_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/TPS_tables/Q1-Q17 DR25/keplerTPS_KSOP2536_nontces.csv', index=False)
tceTpsTbl.to_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/TPS_tables/Q1-Q17 DR25/keplerTPS_KSOP2536_tces.csv', index=False)

#%% Update TPS TCE table with dispositions

tceTpsTbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/TPS_tables/Q1-Q17 DR25/keplerTPS_KSOP2536_tces.csv')

tceTbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17_DR25/q1_q17_dr25_tce_2020.04.15_23.19.10_cumkoi_2020.02.21_shuffled.csv')

tceTpsTbl['label'] = 'NTP'
tceTpsTbl['tce_rogue_flag'] = 0

tceTpsTbl['tce_datalink_dvs'] = ''
tceTpsTbl['tce_datalink_dvr'] = ''

for tce_i, tce in tceTpsTbl.iterrows():

    tceTpsTbl.loc[tce_i, ['label', 'tce_rogue_flag', 'tce_datalink_dvs', 'tce_datalink_dvr']] = \
        tceTbl.loc[(tceTbl['target_id'] == tce.target_id) &
                   (tceTbl['tce_plnt_num'] == tce.tce_plnt_num)][['label', 'tce_rogue_flag', 'tce_datalink_dvs',
                                                                  'tce_datalink_dvr']].values[0]

tceTpsTbl.to_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/TPS_tables/Q1-Q17 DR25/keplerTPS_KSOP2536_dr25.csv', index=False)

#%% Filter out rogue TCEs

tceTpsTbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/TPS_tables/Q1-Q17 DR25/keplerTPS_KSOP2536_dr25.csv')

tceTpsTbl = tceTpsTbl.loc[tceTpsTbl['tce_rogue_flag'] == 0]

tceTpsTbl.to_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/TPS_tables/Q1-Q17 DR25/keplerTPS_KSOP2536_dr25_noroguetces.csv',
                 index=False)

#%% Order TPS TCE table following order of the DV TCE table

tceTbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17_DR25/'
                     'q1_q17_dr25_tce_2020.04.15_23.19.10_cumkoi_2020.02.21_shuffledstar.csv')

tceTpsTbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/TPS_tables/Q1-Q17 DR25/'
                        'keplerTPS_KSOP2536_dr25.csv')

tceTbl = tceTbl.loc[tceTbl['tce_plnt_num'] == 1]

tceTbl.to_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17_DR25/'
              'q1_q17_dr25_tce_2020.04.15_23.19.10_cumkoi_2020.02.21_shuffledstar_tps.csv', index=False)

assert len(tceTbl) == len(tceTpsTbl)

tceTpsTbl.set_index('target_id', drop=False, inplace=True)

tceTpsTbl = tceTpsTbl.reindex(tceTbl['target_id'])

tceTpsTbl.to_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/TPS_tables/Q1-Q17 DR25/'
                 'keplerTPS_KSOP2536_dr25_sameorder.csv', index=False)

#%% Add labels to TPS TCE table based on the DV TCE table no Robovetter KOI dispositions

tceTpsTbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/TPS_tables/Q1-Q17 DR25/'
                        'keplerTPS_KSOP2536_dr25_sameorder.csv')

tceTbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17_DR25/'
                     'q1_q17_dr25_tce_2020.04.15_23.19.10_cumkoi_2020.02.21_shuffledstar_tps.csv')

# filter out rogue TCEs
tceTpsTbl = tceTpsTbl.loc[tceTpsTbl['tce_rogue_flag'] == 0]
tceTbl = tceTbl.loc[tceTbl['tce_rogue_flag'] == 0]

# reset TCE labels
tceTpsTbl['label'] = 'NTP'
tceTbl['label'] = 'NTP'

# load Certified False Positive list
cfpTbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/kois_tables/fpwg_2020.03.13_11.37.49.csv',
                     header=13)

# initialize column for FPWG disposition
tceTpsTbl['fpwg_disp_status'] = np.nan
tceTbl['fpwg_disp_status'] = np.nan

# add kepoi_name column to TPS TCE table
tceTpsTbl.insert(1, 'kepoi_name', tceTbl['kepoi_name'])

map_cfp_to_bin = {'CERTIFIED FP': 'AFP', 'CERTIFIED FA': 'NTP', 'POSSIBLE PLANET': 'PC'}
for koi_i, koi in cfpTbl.iterrows():
    if koi.fpwg_disp_status in list(map_cfp_to_bin.keys()):
        tceTbl.loc[tceTbl['kepoi_name'] == koi.kepoi_name, 'label'] = map_cfp_to_bin[koi.fpwg_disp_status]
        tceTpsTbl.loc[tceTpsTbl['kepoi_name'] == koi.kepoi_name, 'label'] = map_cfp_to_bin[koi.fpwg_disp_status]

    tceTbl.loc[tceTbl['kepoi_name'] == koi.kepoi_name, 'fpwg_disp_status'] = koi.fpwg_disp_status
    tceTpsTbl.loc[tceTpsTbl['kepoi_name'] == koi.kepoi_name, 'fpwg_disp_status'] = koi.fpwg_disp_status

# load Cumulative KOI list
cumKoiTbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/kois_tables/'
                        'cumulative_2020.02.21_10.29.22.csv', header=90)

# filter CONFIRMED KOIs in the Cumulative KOI list
cumKoiTbl = cumKoiTbl.loc[cumKoiTbl['koi_disposition'] == 'CONFIRMED']

for koi_i, koi in cumKoiTbl.iterrows():
    tceTbl.loc[tceTbl['kepoi_name'] == koi.kepoi_name, 'label'] = 'PC'
    tceTpsTbl.loc[tceTpsTbl['kepoi_name'] == koi.kepoi_name, 'label'] = 'PC'

# add koi_disposition column to TPS TCE table
tceTpsTbl.insert(2, 'koi_disposition', tceTbl['koi_disposition'])

# filter out CANDIDATE and FALSE POSITIVE KOIs dispositioned by Robovetter
# keep only CONFIRMED KOIs, CFP, CFA and POSSIBLE PLANET KOIs, and non-KOI (NTPs)
tceTbl = tceTbl.loc[(tceTbl['koi_disposition'] == 'CONFIRMED') |
                    (tceTbl['fpwg_disp_status'].isin(['CERTIFIED FP', 'CERTIFIED FA', 'POSSIBLE PLANET'])) |
                    tceTbl['kepoi_name'].isna()]
tceTpsTbl = tceTpsTbl.loc[(tceTpsTbl['koi_disposition'] == 'CONFIRMED') |
                          (tceTpsTbl['fpwg_disp_status'].isin(['CERTIFIED FP', 'CERTIFIED FA', 'POSSIBLE PLANET'])) |
                          tceTpsTbl['kepoi_name'].isna()]

print('Number of TCEs after removing KOIs dispositioned by Robovetter: {}'.format(len(tceTbl)))
print(tceTbl['label'].value_counts())

print('Number of TCEs after removing KOIs dispositioned by Robovetter: {}'.format(len(tceTpsTbl)))
print(tceTpsTbl['label'].value_counts())

tceTbl.to_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17_DR25/'
              'q1_q17_dr25_tce_2020.04.15_23.19.10_cumkoi_2020.02.21_shuffledstar_tps_norobovetterkois.csv',
              index=False)

tceTpsTbl.to_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/TPS_tables/Q1-Q17 DR25/'
                 'keplerTPS_KSOP2536_dr25_sameorder_norobovetterkois.csv', index=False)
