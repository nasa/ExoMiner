# 3rd party
import pandas as pd
from scipy import io
import numpy as np
import matplotlib.pyplot as plt

#%% Add TPS TCE parameters to the non-TCE (~180k) Kepler table

tce_tbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/180k_nontce_addedstellarparamswitherrors.csv')

tps_tce_mat = io.loadmat('/data5/tess_project/Data/Ephemeris_tables/Kepler/'
                         'tpsTceStructV4_KSOP2536.mat')['tpsTceStructV4_KSOP2536'][0][0]

# # list of TCE data fields in TPS
# mat_names = ['chiSquare1', 'chiSquare2', 'chiSquareDof1', 'chiSquareDof2', 'chiSquareGof', 'chiSquareGofDof',
#              'keplerId', 'maxMes', 'maxSes', 'periodDays',
#              'robustStatistic', 'thresholdForDesiredPfa', 'maxSes']

# tce_fields_out = ['tce_depth']
# tce_fields_in = ['fittedDepth']  # fittedDepthChi ?

tce_tbl['tce_depth'] = np.nan

count = 0
for kepid_i in range(len(tps_tce_mat['keplerId'])):

    if kepid_i % 100 == 0:
        print('Checked {} Kepler IDs out of {} ({} %)\n'
              'Number of non-TCEs updated: {}'.format(kepid_i, len(tps_tce_mat['keplerId']),
                                                      kepid_i / len(tps_tce_mat['keplerId']) * 100, count))

    target_cond = tce_tbl['kepid'] == tps_tce_mat['keplerId'][kepid_i][0]

    count += target_cond.sum()

    tce_tbl.loc[target_cond, 'tce_depth'] = [tps_tce_mat['fittedDepth'][kepid_i][0] * 1e6]  # convert to ppm

print('Number of non-TCEs updated: {}'.format(count))
tce_tbl.to_csv('/home/msaragoc/Downloads/newtcetbl180k_testwithdepth.csv', index=False)

#%% Normalize parameters in Q1-Q17 DR25 TCE table

tce_tbl_fp = '/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17 DR25/' \
             'q1_q17_dr25_tce_2020.04.15_23.19.10_cumkoi_2020.02.21_shuffled_noroguetces.csv'
tce_tbl = pd.read_csv(tce_tbl_fp)

norm_params = ['tce_sradius', 'tce_steff', 'tce_slogg', 'tce_smet', 'tce_smass', 'tce_sdens', 'wst_robstat',
               'wst_depth', 'tce_bin_oedp_stat', 'boot_fap', 'tce_cap_stat', 'tce_hap_stat']

# # fill in NaNs using Solar parameters
# tce_tbl['tce_smass'] = tce_tbl['tce_smass'].fillna(value=1.0)
# tce_tbl['tce_sdens'] = tce_tbl['tce_sdens'].fillna(value=1.408)

# normalize using statistics computed in the training set
trainingset_idx = int(len(tce_tbl) * 0.8)

# ax = tce_tbl[stellar_params].hist(figsize=(15, 8))
# ax[0, 0].set_title('Stellar density')
# ax[0, 0].set_ylabel('g/cm^3')
# ax[0, 1].set_title('Stellar surface gravity')
# ax[0, 1].set_ylabel('log10(cm/s^2)')
# ax[1, 0].set_title('Stellar mass')
# ax[1, 0].set_ylabel('Solar masses')
# ax[1, 1].set_title('Stellar metallicity (Fe/H)')
# ax[1, 1].set_ylabel('dex')
# ax[2, 0].set_title('Stellar radius')
# ax[2, 0].set_ylabel('Solar radii')
# ax[2, 0].set_xlabel('Bins')
# ax[2, 1].set_title('Stellar effective temperature')
# ax[2, 1].set_ylabel('K')
# ax[2, 1].set_xlabel('Bins')
# plt.suptitle('Stellar parameters')

norm_params_med = tce_tbl[norm_params][:trainingset_idx].median(axis=0, skipna=False)
norm_params_std = tce_tbl[norm_params][:trainingset_idx].std(axis=0, skipna=False)
stats_norm = {'med': norm_params_med, 'std': norm_params_std}

np.save('{}_stats_norm.npy'.format(tce_tbl_fp.replace('.csv', '')), stats_norm)

tce_tbl[norm_params] = (tce_tbl[norm_params] - norm_params_med) / norm_params_std

# ax = tce_tbl[stellar_params].hist(figsize=(15, 8))
# ax[0, 0].set_title('Stellar density')
# ax[0, 0].set_ylabel('Amplitude')
# ax[0, 1].set_title('Stellar surface gravity')
# ax[0, 1].set_ylabel('Amplitude')
# ax[1, 0].set_title('Stellar mass')
# ax[1, 0].set_ylabel('Amplitude')
# ax[1, 1].set_title('Stellar metallicity (Fe/H)')
# ax[1, 1].set_ylabel('Amplitude')
# ax[2, 0].set_title('Stellar radius')
# ax[2, 0].set_ylabel('Amplitude')
# ax[2, 0].set_xlabel('Bins')
# ax[2, 1].set_title('Stellar effective temperature')
# ax[2, 1].set_ylabel('Amplitude')
# ax[2, 1].set_xlabel('Bins')
# plt.suptitle('Normalized stellar parameters')

tce_tbl[norm_params].median(axis=0, skipna=False)
tce_tbl[norm_params].std(axis=0, skipna=False)

tce_tbl.to_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17 DR25/'
               'q1_q17_dr25_tce_2020.04.15_23.19.10_cumkoi_2020.02.21_shuffled_noroguetces_norm.csv',
               index=False)

#%% Check transit depth for Kepler dataset

tce_tbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/'
                      'q1_q17_dr25_tce_2019.03.12_updt_tcert_extendedtceparams_updt_normstellarparamswitherrors.csv')

transit_depth_col = tce_tbl[['tce_depth']]
print(transit_depth_col.min(), transit_depth_col.max())

bins = np.arange(0, 2e6, 1e5)
ax = transit_depth_col.hist(bins=bins, cumulative=False, density=False)[0]
for x in ax:
    x.set_title("TCE transit depth histogram\nMax {}\nMin {}".format(transit_depth_col.min().values[0],
                                                                     transit_depth_col.max().values[0]))

    # Set x-axis label
    x.set_xlabel("Transit depth (ppm)")
    x.set_xticks(bins)

    # Set y-axis label
    x.set_ylabel("Counts")
    # x.set_ylabel("Density")
    x.set_xscale('log')
    x.set_xlim(left=bins[0], right=bins[-1])

#%% Uniformization of the fields

stellar_params_in = ['tce_sradius', 'tce_steff', 'tce_slogg', 'tce_smet', 'tce_smass', 'tce_sdens']

stellar_params_out = ['tce_sradius', 'tce_steff', 'tce_slogg', 'tce_smet', 'tce_smass', 'tce_sdens']

tce_tbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/'
                      'q1_q17_dr25_tce_2019.03.12_updt_tcert_extendedtceparams_updt_stellarparamswitherrors.csv')

tce_tbl[stellar_params_out].hist()
tce_tbl[stellar_params_out].median(axis=0, skipna=False)
tce_tbl[stellar_params_out].std(axis=0, skipna=False)

#%% Getting EBs - need list for the 34k TCEs!!!

tpsStruct = io.loadmat('/data5/tess_project/Data/TPS_output/Kepler/'
                       'tpsTceStructV4_KSOP2536.mat')['tpsTceStructV4_KSOP2536'][0][0]

kepIdList = tpsStruct['keplerId'].ravel()
ebList = tpsStruct['isOnEclipsingBinaryList'].ravel()

plt.hist(ebList)

tceTbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/'
                     'q1_q17_dr25_tce_2019.03.12_updt_tcert_extendedtceparams_updt_normstellarparamswitherrors.csv')

tceTbl['isOnEclipsingBinaryList'] = -1

for i, row in tceTbl.iterrows():

    idxEbList = np.where(kepIdList == row.kepid)[0]
    if row.tce_plnt_num == 1 and len(idxEbList) > 0:
        tceTbl.loc[i, 'isOnEclipsingBinaryList'] = ebList[idxEbList]

tceTbl.to_csv('/home/msaragoc/Downloads/qqqqq.csv', index=False)

#%% Preprocess TEV MIT TCE lists

rawTceTable = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/TESS/TEV_MIT_TOI_lists/raw/'
                          'toi-plus-tev.mit.edu_2020-04-15.csv', header=4)
print(len(rawTceTable))
rawTceTableGroupDisposition = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/TESS/TEV_MIT_TOI_lists/raw/'
                                          'toi-tev.mit.edu_2020-04-15.csv', header=4,
                                          usecols=['Full TOI ID', 'Group Disposition'])
print(len(rawTceTableGroupDisposition))

# add group disposition to the TCE plus list
rawTceTable['Group Disposition'] = np.nan
for tce_i, tce in rawTceTableGroupDisposition.iterrows():
    rawTceTable.loc[rawTceTable['Full TOI ID'] == tce['Full TOI ID'], 'Group Disposition'] = tce['Group Disposition']

rawTceTable.to_csv('/data5/tess_project/Data/Ephemeris_tables/TESS/TEV_MIT_TOI_lists/'
                   'toi-plus-tev.mit.edu_2020-04-15_addedgroupdisposition.csv', index=False)

# changing the field name in rawFields
rawFields = ['TIC', 'Full TOI ID', 'TIC Right Ascension', 'TIC Declination', 'TMag Value',
             'TMag Uncertainty', 'Orbital Epoch Value', 'Orbital Epoch Error', 'Orbital Period Value',
             'Orbital Period Error', 'Transit Duration Value', 'Transit Duration Error', 'Transit Depth Value',
             'Transit Depth Error', 'Sectors', 'Surface Gravity Value', 'Surface Gravity Uncertainty',
             'Star Radius Value', 'Star Radius Error', 'Effective Temperature Value',
             'Effective Temperature Uncertainty']
newFields = ['target_id', 'oi', 'ra', 'dec', 'mag', 'mag_err', 'tce_time0bk', 'tce_time0bk_err',
             'tce_period', 'tce_period_err', 'tce_duration', 'tce_duration_err', 'transit_depth', 'transit_depth_err',
             'sectors', 'tce_slogg', 'tce_slogg_err', 'tce_sradius', 'tce_sradius_err', 'tce_steff', 'tce_steff_err']

# remove TCEs with any NaN in the required fields
rawTceTable.dropna(axis=0, subset=np.array(rawFields)[[0, 1, 2, 3, 6, 8, 10, 12]], inplace=True)
print(len(rawTceTable))

# rename fields to standardize fieldnames
renameDict = {}
for i in range(len(rawFields)):
    renameDict[rawFields[i]] = newFields[i]
rawTceTable.rename(columns=renameDict, inplace=True)

# remove TCEs with zero period or transit duration
rawTceTable = rawTceTable.loc[(rawTceTable['tce_period'] > 0) & (rawTceTable['tce_duration'] > 0)]
print(len(rawTceTable))

# rawTceTable = rawTceTable.drop(['Edited', 'Alerted'], axis=1)

# 1732 TCEs
rawTceTable.to_csv('/data5/tess_project/Data/Ephemeris_tables/TESS/TEV_MIT_TOI_lists/'
                   'toi-plus-tev.mit.edu_2020-04-15_processed.csv', index=False)

# split TCE list into two TCE lists with TOI and Group disposition
rawTceTable = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/TESS/TEV_MIT_TOI_lists/'
                          'toi-plus-tev.mit.edu_2020-04-15_stellar.csv')

disposition_src = ['Group', 'TOI']
for disposition in disposition_src:
    dispTceTable = rawTceTable.rename(columns={'{} Disposition'.format(disposition): 'label'})
    dispTceTable.to_csv('/data5/tess_project/Data/Ephemeris_tables/TESS/TEV_MIT_TOI_lists/'
                        'toi-plus-tev.mit.edu_2020-04-15_{}disposition.csv'.format(disposition), index=False)

#%% Preprocess NASA Exoplanet Archive TESS TCE list to standardized fields

rawTceTable = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/TESS/NASA_Exoplanet_Archive_TOI_lists/raw/'
                          'TOI_2020.04.14_23.04.53.csv', header=69)

# 1766
print(len(rawTceTable))

# Create Group or TOI disposition table by changing the field name in rawFields
rawFields = ['toi', 'tid', 'tfopwg_disp', 'ra', 'dec', 'st_tmag', 'st_tmagerr1', 'st_tmagerr2', 'pl_tranmid',
             'pl_tranmiderr1', 'pl_tranmiderr2', 'pl_orbper',
             'pl_orbpererr1', 'pl_orbpererr2', 'pl_trandurh', 'pl_trandurherr1', 'pl_trandurherr2', 'pl_trandep',
             'pl_trandeperr1', 'pl_trandeperr2', 'st_teff', 'st_tefferr1', 'st_tefferr2', 'st_logg', 'st_loggerr1',
             'st_loggerr2', 'st_rad', 'st_raderr1', 'st_raderr2']
newFields = ['oi', 'target_id', 'label', 'ra', 'dec', 'mag', 'mag_uncert', 'mag_uncert2', 'tce_time0bk',
             'tce_time0bk_err', 'tce_time0bk_err2', 'tce_period', 'tce_period_err', 'tce_period_err2', 'tce_duration',
             'tce_duration_err', 'tce_duration_err2', 'transit_depth', 'transit_depth_err', 'transit_depth_err2',
             'tce_steff', 'tce_steff_err1', 'tce_steff_err2', 'tce_slogg', 'tce_slogg_err1', 'tce_slogg_err2',
             'tce_sradius', 'tce_sradius_err1', 'tce_sradius_err2']

# remove TCEs with any NaN in the required fields - except label field
rawTceTable.dropna(axis=0, subset=np.array(rawFields)[[0, 1, 3, 4, 8, 11, 14, 17]], inplace=True)
# 1730
print(len(rawTceTable))

# rename fields to standardize fieldnames
renameDict = {}
for i in range(len(rawFields)):
    renameDict[rawFields[i]] = newFields[i]
rawTceTable.rename(columns=renameDict, inplace=True)

# remove TCEs with zero period or transit duration
rawTceTable = rawTceTable.loc[(rawTceTable['tce_period'] > 0) & (rawTceTable['tce_duration'] > 0)]
# 1730
print(len(rawTceTable))

# convert epoch value from BJD to TBJD
rawTceTable['tce_time0bk'] -= 2457000
# 1730 TCEs
rawTceTable.to_csv('/data5/tess_project/Data/Ephemeris_tables/TESS/NASA_Exoplanet_Archive_TOI_lists/'
                   'TOI_2020.04.14_23.04.53_processed.csv', index=False)

rawTceTable = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/TESS/NASA_Exoplanet_Archive_TOI_lists/'
                          'TOI_2020.04.14_23.04.53_stellar.csv')
print(len(rawTceTable))

# drop unlabeled TCEs
rawTceTable.dropna(axis=0, subset=['label'], inplace=True)
print(len(rawTceTable))

# 645 TCEs
rawTceTable.to_csv('/data5/tess_project/Data/Ephemeris_tables/TESS/NASA_Exoplanet_Archive_TOI_lists/'
                   'TOI_2020.04.14_23.04.53_TFOPWG.csv', index=False)

#%% Preprocess fields in EXOFOP Community TESS disposition list

rawTceTable = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/TESS/EXOFOP_TOI_lists/exofop_ctoilists.csv',
                          header=0)
print(len(rawTceTable))

rawFields = ['CTOI', 'TIC ID', 'User Disposition', 'RA', 'Dec', 'TESS Mag', 'TESS Mag err', 'Midpoint (BJD)',
             'Midpoint err', 'Period (days)', 'Period (days) Error', 'Duration (hrs)', 'Duration (hrs) Error',
             'Depth ppm', 'Depth ppm Error']
newFields = ['oi', 'target_id', 'label', 'ra', 'dec', 'mag', 'mag_uncert', 'tce_time0bk', 'tce_time0bk_err',
             'tce_period', 'tce_period_err', 'tce_duration', 'tce_duration_err', 'transit_depth', 'transit_depth_err']

# remove TCEs with any NaN in the required fields
rawTceTable.dropna(axis=0, subset=np.array(rawFields)[[0, 1, 2, 3, 4, 7, 9, 11, 13]], inplace=True)
print(len(rawTceTable))

# rename fields to standardize fieldnames
renameDict = {}
for i in range(len(rawFields)):
    renameDict[rawFields[i]] = newFields[i]
rawTceTable.rename(columns=renameDict, inplace=True)

# remove TCEs with zero period or transit duration
rawTceTable = rawTceTable.loc[(rawTceTable['tce_period'] > 0) & (rawTceTable['tce_duration'] > 0)]
print(len(rawTceTable))

# convert epoch value from BJD to TBJD
rawTceTable['tce_time0bk'] -= 2457000

# 321 to 273 TCEs
rawTceTable.to_csv('/data5/tess_project/Data/Ephemeris_tables/TESS/EXOFOP_TOI_lists/final_tce_tables/'
                   'exofop_ctoilists_Community_processed.csv', index=False)

#%% Preprocess Q1-Q17 DR25 TCE list

rawTceTable = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17 DR25/'
                          'q1_q17_dr25_tce_2020.04.15_23.19.10_cumkoi_2020.02.21.csv')
print(len(rawTceTable))

rawFields = ['kepid', 'tce_plnt_num', 'ra', 'dec', 'kepmag', 'tce_time0bk', 'tce_time0bk_err',
             'tce_period', 'tce_period_err', 'tce_duration', 'tce_duration_err',
             'tce_depth', 'tce_depth_err']
newFields = ['target_id', 'tce_plnt_num', 'ra', 'dec', 'mag', 'tce_time0bk', 'tce_time0bk_err',
             'tce_period', 'tce_period_err', 'tce_duration', 'tce_duration_err', 'transit_depth', 'transit_depth_err']

# remove TCEs with any NaN in the required fields
rawTceTable.dropna(axis=0, subset=np.array(rawFields)[[0, 1, 2, 3, 5, 7, 9, 11]], inplace=True)
print(len(rawTceTable))

# rename fields to standardize fieldnames
renameDict = {}
for i in range(len(rawFields)):
    renameDict[rawFields[i]] = newFields[i]
rawTceTable.rename(columns=renameDict, inplace=True)

# remove TCEs with zero period or transit duration
rawTceTable = rawTceTable.loc[(rawTceTable['tce_period'] > 0) & (rawTceTable['tce_duration'] > 0)]
print(len(rawTceTable))

# # remove 'rowid' column
# rawTceTable.drop(columns='rowid', inplace=True)

# 34032 to 34032 TCEs
rawTceTable.to_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17 DR25/'
                   'q1_q17_dr25_tce_2020.04.15_23.19.10_cumkoi_2020.02.21_processed.csv',
                   index=False)

#%% Normalize stellar parameters in the DR24 Q1-Q17 Kepler TCE table

tce_tbl_fp = '/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17_DR24/' \
             'q1_q17_dr24_tce_2020.03.02_17.51.43_nounks_shuffled.csv'
tce_tbl = pd.read_csv(tce_tbl_fp)

stellar_params = ['tce_sradius', 'tce_steff', 'tce_slogg', 'tce_smet', 'tce_smass', 'tce_sdens']

# # fill in NaNs using Solar parameters
# tce_tbl['tce_smass'] = tce_tbl['tce_smass'].fillna(value=1.0)
# tce_tbl['tce_sdens'] = tce_tbl['tce_sdens'].fillna(value=1.408)

# # remove TCEs with 'UNK' label
# tce_tbl = tce_tbl.loc[tce_tbl['av_training_set'] != 'UNK']

# # Randomly shuffle the TCE table using the same seed as Shallue and FDL
# np.random.seed(123)
# tce_table = tce_tbl.iloc[np.random.permutation(len(tce_tbl))]

# normalize using statistics computed in the training set
trainingset_idx = int(len(tce_tbl) * 0.8)

stellar_params_med = tce_tbl[stellar_params][:trainingset_idx].median(axis=0, skipna=False)
stellar_params_std = tce_tbl[stellar_params][:trainingset_idx].std(axis=0, skipna=False)
stats_norm = {'med': stellar_params_med, 'std': stellar_params_std}

np.save('{}_stats_norm.npy'.format(tce_tbl_fp.replace('.csv', '')), stats_norm)

tce_tbl[stellar_params] = (tce_tbl[stellar_params] - stellar_params_med) / stellar_params_std

tce_tbl[stellar_params].median(axis=0, skipna=False)
tce_tbl[stellar_params].std(axis=0, skipna=False)

# rawFields = ['kepid', 'av_training_set', 'tce_plnt_num', 'ra', 'dec', 'kepmag', 'tce_time0bk', 'tce_time0bk_err',
#              'tce_period', 'tce_period_err', 'tce_duration', 'tce_duration_err',
#              'tce_depth', 'tce_depth_err']
# newFields = ['target_id', 'label', 'tce_plnt_num', 'ra', 'dec', 'mag', 'tce_time0bk', 'tce_time0bk_err',
#              'tce_period', 'tce_period_err', 'tce_duration', 'tce_duration_err', 'transit_depth', 'transit_depth_err']

print(len(tce_tbl))

tce_tbl.rename(columns={'tce_depth': 'transit_depth'}, inplace=True)

# remove TCEs with any NaN in the required fields
# tce_tbl.dropna(axis=0, subset=np.array(rawFields)[[0, 1, 2, 3, 4, 6, 8, 10, 12]], inplace=True)
tce_tbl.dropna(axis=0, subset=['target_id', 'tce_plnt_num', 'tce_period', 'tce_time0bk', 'tce_duration',
                               'transit_depth', 'ra', 'dec'], inplace=True)
print(len(tce_tbl))

# # rename fields to standardize fieldnames
# renameDict = {}
# for i in range(len(rawFields)):
#     renameDict[rawFields[i]] = newFields[i]
# tce_tbl.rename(columns=renameDict, inplace=True)

# remove TCEs with zero period or transit duration
tce_tbl = tce_tbl.loc[(tce_tbl['tce_period'] > 0) & (tce_tbl['tce_duration'] > 0)]
print(len(tce_tbl))

# 15737 to 15737 TCEs
tce_tbl.to_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17_DR24/'
               'q1_q17_dr24_tce_2020.03.02_17.51.43_nounks_processed_stellarnorm.csv',
               index=False)

#%% Filter 'UNK' TCEs in Q1-Q17 DR24

tce_tbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17_DR24/'
                      'q1_q17_dr24_tce_2020.03.02_17.51.43_stellar.csv')

print(len(tce_tbl))

stellar_params = ['tce_sradius', 'tce_steff', 'tce_slogg', 'tce_smet', 'tce_smass', 'tce_sdens']

# remove TCEs with 'UNK' label
tce_tbl = tce_tbl.loc[tce_tbl['label'] != 'UNK']

print(len(tce_tbl))

# 20367 to 15737 TCEs
tce_tbl.to_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17_DR24/'
               'q1_q17_dr24_tce_2020.03.02_17.51.43_nounks.csv',
               index=False)

#%% Filter rogue TCEs in Q1-Q17 DR25

tceTbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17 DR25/'
                     'q1_q17_dr25_tce_2020.04.15_23.19.10_cumkoi_2020.02.21_shuffled.csv')

print(len(tceTbl))

tceTbl = tceTbl.loc[tceTbl['tce_rogue_flag'] == 0]

print(len(tceTbl))

tceTbl.to_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17 DR25/'
              'q1_q17_dr25_tce_2020.04.15_23.19.10_cumkoi_2020.02.21_shuffled_noroguetces.csv', index=False)

#%% Shuffle TCEs in Q1-Q17 DR25 TCE table (with and without rogue TCEs)

np.random.seed(24)

tceTbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17 DR25/'
                     'q1_q17_dr25_tce_2020.04.15_23.19.10_cumkoi_2020.02.21_processed.csv')

tceTbl = tceTbl.iloc[np.random.permutation(len(tceTbl))]

tceTbl.to_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17 DR25/'
              'q1_q17_dr25_tce_2020.04.15_23.19.10_cumkoi_2020.02.21_shuffled.csv', index=False)
#
# #%%
#
# tceTbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17 DR25/'
#                      'q1_q17_dr25_tce_2020.04.15_23.19.10_cumkoi_2020.02.21_shuffled.csv')
#
# addFieldsTbl = pd.read_csv('/home/msaragoc/Downloads/q1_q17_dr25_tce_2020.04.20_14.43.23.csv', header=26)
#
# fieldsToAdd = ['tce_maxmesd', 'wst_robstat', 'wst_depth', 'tce_bin_oedp_stat', 'boot_fap', 'tce_cap_stat',
#                'tce_hap_stat']
#
# for field in fieldsToAdd:
#     tceTbl[field] = np.nan
#
# for tce_i, tce in tceTbl.iterrows():
#     tceFound = addFieldsTbl.loc[(addFieldsTbl['kepid'] == tce.target_id) & (addFieldsTbl['tce_plnt_num'] == tce.tce_plnt_num)]
#     tceTbl.loc[tce_i, fieldsToAdd] = tceFound[fieldsToAdd].values[0]
#
# tceTbl.to_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17 DR25/'
#               'q1_q17_dr25_tce_2020.04.15_23.19.10_cumkoi_2020.02.21_shuffled.csv', index=False)

#%% Shuffle TCEs in Q1-Q17 DR24 TCE table

np.random.seed(123)  # same seed as Shallue & Vanderburg and Ansdell et al

tceTbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17_DR24/'
                     'q1_q17_dr24_tce_2020.03.02_17.51.43_nounks.csv')

tceTbl = tceTbl.iloc[np.random.permutation(len(tceTbl))]

tceTbl.to_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17_DR24/'
              'q1_q17_dr24_tce_2020.03.02_17.51.43_nounks_shuffled.csv', index=False)

#%% Shuffle non-TCEs 180k

np.random.seed(42)

tceTbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/180k non-TCEs/180k_nontce_stellar.csv')

tceTbl = tceTbl.iloc[np.random.permutation(len(tceTbl))]

tceTbl.to_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/180k non-TCEs/'
              '180k_nontce_stellar_shuffled.csv', index=False)

#%% Standardize columns in TCE lists

tceTbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17_DR24/'
                     'q1_q17_dr24_tce_2020.03.02_17.51.43_stellar_shuffled.csv')

rawFields = ['kepid', 'av_training_set']
newFields = ['target_id', 'label']
# rawFields = ['kepid', 'av_training_set', 'tce_plnt_num', 'ra', 'dec', 'kepmag', 'tce_time0bk', 'tce_time0bk_err',
#              'tce_period', 'tce_period_err', 'tce_duration', 'tce_duration_err',
#              'tce_depth', 'tce_depth_err']
# newFields = ['target_id', 'label', 'tce_plnt_num', 'ra', 'dec', 'mag', 'tce_time0bk', 'tce_time0bk_err',
#              'tce_period', 'tce_period_err', 'tce_duration', 'tce_duration_err', 'transit_depth', 'transit_depth_err']

print(len(tceTbl))

# remove TCEs with any NaN in the required fields
tceTbl.dropna(axis=0, subset=['tce_period', 'tce_duration', 'tce_time0bk'], inplace=True)
print(len(tceTbl))

# rename fields to standardize fieldnames
renameDict = {}
for i in range(len(rawFields)):
    renameDict[rawFields[i]] = newFields[i]
tceTbl.rename(columns=renameDict, inplace=True)

# tceTbl.drop(columns=['rowid'], inplace=True)

tceTbl.to_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17_DR24/'
              'q1_q17_dr24_tce_2020.03.02_17.51.43_stellar_shuffled.csv', index=False)

#%%

# rawTceTbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/TESS/NASA_Exoplanet_Archive_TOI_lists/'
#                         'TOI_2020.01.21_13.55.10.csv', header=72)
#
# print(len(rawTceTbl))
#
# rawTceTbl = rawTceTbl.dropna(axis=0, subset=['tfopwg_disp'])
#
# print(len(rawTceTbl))
#
# rawTceTbl = rawTceTbl.loc[(rawTceTbl['pl_orbper'] > 0) & (rawTceTbl['pl_trandurh'] > 0)]
# print(len(rawTceTbl))
#
#
# stellar_params = ['st_teff', 'st_logg', 'st_rad', 'ra', 'dec']
#
# print(rawTceTbl[stellar_params].isna().sum(axis=0))

tceTbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/TESS/TEV_MIT_TOI_lists/'
                     'toi-plus-tev.mit.edu_2020-01-15_TOI Disposition_processed_stellar.csv')

ticsFits = np.load('/data5/tess_project/Data/Ephemeris_tables/Stellar parameters/TESS/final_target_list_s1-s20.npy')

print(len(tceTbl))

tceTbl = tceTbl.loc[tceTbl['target_id'].isin(ticsFits)]

print(len(tceTbl))

stellarColumns = ['ra', 'dec', 'tce_steff', 'tce_slogg', 'tce_sradius', 'tce_smass', 'tce_smet', 'tce_sdens']

# count all nan stellar rows
print(tceTbl.loc[tceTbl[stellarColumns].isna().all(axis=1)])

print(tceTbl[stellarColumns].isna().sum())

targetsTbl = tceTbl.drop_duplicates(subset=['target_id'])
print(targetsTbl[stellarColumns].isna().sum())

#%% Updated stellar parameters in TESS TCE using the values found in the TESS FITS files

fitsTbl = pd.read_csv('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Kepler_planet_finder/stellar_parameters_analysis/'
                      'tess_stellar_parameters_fits/fitsStellarTbl_s1-s20_unique.csv')

tceTbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/TESS/TEV_MIT_TOI_lists/'
                     'toi-plus-tev.mit.edu_2020-04-15_TOIdisposition.csv')

count = 0
stellarColumnsTceTbl = np.array(['mag', 'tce_steff', 'tce_slogg', 'tce_smet', 'tce_sradius'])
stellarColumnsFitsTbl = np.array(['TESSMAG', 'TEFF', 'LOGG', 'MH', 'RADIUS'])
print(tceTbl[stellarColumnsTceTbl].isna().sum())
for tce_i, tce in tceTbl.iterrows():

    targetInFitsTbl = fitsTbl.loc[fitsTbl['TICID'] == tce.target_id]

    if len(targetInFitsTbl) == 0:
        # print('TIC ID {} not found in FITS files'.format(tce.target_id))
        count += 1
        continue

    targetInTceTbl = tceTbl.loc[tce_i, stellarColumnsTceTbl].values

    targetInTceTbl = targetInTceTbl.astype('float')
    idxsParamsNan = np.where(np.isnan(targetInTceTbl))[0]

    if len(idxsParamsNan) == 0:
        continue  # no missing values

    tceTbl.loc[tce_i, stellarColumnsTceTbl[idxsParamsNan]] = \
        targetInFitsTbl[stellarColumnsFitsTbl[idxsParamsNan]].values[0]

    # aaaa
print(tceTbl[stellarColumnsTceTbl].isna().sum())

#%% Updated stellar parameters in TESS TCE lists using the values found in the EXOFOP TOI list

fitsTbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/TESS/EXOFOP_TOI_lists/TOI/raw/'
                      'exofop_toilists_4-17-2020.csv')

fitsTbl.drop_duplicates(subset=['TIC ID'], inplace=True)

tceTbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/TESS/TEV_MIT_TOI_lists/'
                     'toi-plus-tev.mit.edu_2020-04-15_TOIdisposition.csv')

count = 0
stellarColumnsTceTbl = np.array(['mag', 'tce_steff', 'tce_slogg', 'tce_smet', 'tce_sradius'])
stellarColumnsFitsTbl = np.array(['TESS Mag', 'Stellar Eff Temp (K)', 'Stellar log(g) (cm/s^2)', 'Stellar Metallicity',
                                  'Stellar Radius (R_Sun)'])
print(tceTbl[stellarColumnsTceTbl].isna().sum())
print(fitsTbl[stellarColumnsFitsTbl].isna().sum())
for tce_i, tce in tceTbl.iterrows():

    targetInFitsTbl = fitsTbl.loc[fitsTbl['TIC ID'] == tce.target_id]

    if len(targetInFitsTbl) == 0:
        # print('TIC ID {} not found in FITS files'.format(tce.target_id))
        count += 1
        continue

    targetInTceTbl = tceTbl.loc[tce_i, stellarColumnsTceTbl].values

    targetInTceTbl = targetInTceTbl.astype('float')
    idxsParamsNan = np.where(np.isnan(targetInTceTbl))[0]

    if len(idxsParamsNan) == 0:
        continue  # no missing values

    tceTbl.loc[tce_i, stellarColumnsTceTbl[idxsParamsNan]] = \
        targetInFitsTbl[stellarColumnsFitsTbl[idxsParamsNan]].values[0]

    # aaaa
print(tceTbl[stellarColumnsTceTbl].isna().sum())

#%% Update labels using CFP list  PPs, CFPs and CFAs KOIs, and Confirmed KOIs from the Cumulative KOI list

tceTbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17 DR25/'
                     'q1_q17_dr25_tce_2020.04.15_23.19.10_cumkoi_2020.02.21_shuffled_noroguetces_norm.csv')

# reset TCE labels
tceTbl['label'] = 'NTP'

# load Certified False Positive list
cfpTbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/kois_tables/fpwg_2020.03.13_11.37.49.csv',
                     header=13)

# initialize column for FPWG disposition
tceTbl['fpwg_disp_status'] = np.nan

# # filter NOT EXAMINED and DATA INCONCLUSIVE KOIs from the CFP list
# cfpTbl = cfpTbl.loc[cfpTbl['fpwg_disp_status'].isin(['NOT EXAMINED', 'DATA INCONCLUSIVE'])]

map_cfp_to_bin = {'CERTIFIED FP': 'AFP', 'CERTIFIED FA': 'NTP', 'POSSIBLE PLANET': 'PC'}
for koi_i, koi in cfpTbl.iterrows():
    if koi.fpwg_disp_status in list(map_cfp_to_bin.keys()):
        tceTbl.loc[tceTbl['kepoi_name'] == koi.kepoi_name, 'label'] = map_cfp_to_bin[koi.fpwg_disp_status]

    tceTbl.loc[tceTbl['kepoi_name'] == koi.kepoi_name, 'fpwg_disp_status'] = koi.fpwg_disp_status

# load Cumulative KOI list
cumKoiTbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/kois_tables/'
                        'cumulative_2020.02.21_10.29.22.csv', header=90)

# filter CONFIRMED KOIs
cumKoiTbl = cumKoiTbl.loc[cumKoiTbl['koi_disposition'] == 'CONFIRMED']

for koi_i, koi in cumKoiTbl.iterrows():
    tceTbl.loc[tceTbl['kepoi_name'] == koi.kepoi_name, 'label'] = 'PC'

tceTbl.to_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17 DR25/'
              'q1_q17_dr25_tce_2020.04.15_23.19.10_cumkoi_2020.02.21_shuffled_noroguetces_norm_bhv.csv', index=False)
