import pandas as pd
import numpy as np
from scipy import io
import matplotlib.pyplot as plt
# from astropy.stats import mad_std
import random

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

#%% add weak secondary phase and depth
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

#%% add rolling band diagnostic values

sourceTbl = pd.read_csv('/home/msaragoc/Downloads/q1_q17_dr25_tce_2020.07.14_17.42.20.csv', header=38)
tceDir = '/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17_DR25/'

tceTblsFiles = [os.path.join(tceDir, tceTblFile) for tceTblFile in os.listdir(tceDir) if tceTblFile.startswith('q1_q17_dr25')]

columnsToAdd = ['tce_rb_tpdur', 'tce_rb_tcount0', 'tce_rb_tcount1', 'tce_rb_tcount2', 'tce_rb_tcount3', 'tce_rb_tcount4']

for tceTblFile in tceTblsFiles:

    print(tceTblFile)

    tceTbl = pd.read_csv(tceTblFile)

    for columnToAdd in columnsToAdd:
        tceTbl[columnToAdd] = np.nan

    for tce_i, tce in tceTbl.iterrows():

        key = 'kepid' if tceTblFile.split('/')[-1] in ['q1_q17_dr25_tce_2020.04.15_23.19.10_stellar.csv',
                                                       'q1_q17_dr25_tce_2020.04.15_23.19.10_cumkoi_2020.02.21.csv'] else 'target_id'

        tceTbl.loc[tce_i, columnsToAdd] = sourceTbl.loc[(sourceTbl['kepid'] == tce[key]) &
                                                        (sourceTbl['tce_plnt_num'] == tce['tce_plnt_num'])][columnsToAdd].values[0]

    tceTbl.to_csv(tceTblFile, index=False)

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

#%% Uniformization of the fields

stellar_params_in = ['tce_sradius', 'tce_steff', 'tce_slogg', 'tce_smet', 'tce_smass', 'tce_sdens']

stellar_params_out = ['tce_sradius', 'tce_steff', 'tce_slogg', 'tce_smet', 'tce_smass', 'tce_sdens']

tce_tbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/'
                      'q1_q17_dr25_tce_2019.03.12_updt_tcert_extendedtceparams_updt_stellarparamswitherrors.csv')

tce_tbl[stellar_params_out].hist()
tce_tbl[stellar_params_out].median(axis=0, skipna=False)
tce_tbl[stellar_params_out].std(axis=0, skipna=False)

#%% Change labels of the preprocessing Kepler Q1-Q17 DR25 TCE table to the ones being used in the final experiment

tceTbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17_DR25/'
                     'q1_q17_dr25_tce_2020.04.15_23.19.10_cumkoi_2020.02.21_shuffled.csv')

labelTbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17_DR25/'
                       'q1_q17_dr25_tce_2020.04.15_23.19.10_cumkoi_2020.02.21_shuffledstar_noroguetces_noRobovetterKOIs.csv')

tceTbl['label'] = 'NTP'

for tcei, tce in tceTbl.iterrows():

    tceFound = labelTbl.loc[(labelTbl['target_id'] == tce['target_id']) &
                            (labelTbl['tce_plnt_num'] == tce['tce_plnt_num'])]

    if len(tceFound) > 0:
        tceTbl.loc[tcei, ['label']] = tceFound['label'].values[0]

# assert labels changed
for tcei, tce in tceTbl.iterrows():

    tceFound = labelTbl.loc[(labelTbl['target_id'] == tce['target_id']) &
                            (labelTbl['tce_plnt_num'] == tce['tce_plnt_num'])]

    if len(tceFound) > 0:
        assert tce['label'] == tceFound['label'].values[0]

tceTbl.to_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17_DR25/'
              'q1_q17_dr25_tce_2020.04.15_23.19.10_cumkoi_2020.02.21_shuffled_norobovetterlabels.csv',
              index=False)

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

#%% Shuffle non-TCEs 180k

np.random.seed(42)

tceTbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/180k non-TCEs/180k_nontce_stellar.csv')

tceTbl = tceTbl.iloc[np.random.permutation(len(tceTbl))]

tceTbl.to_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/180k non-TCEs/'
              '180k_nontce_stellar_shuffled.csv', index=False)

#%% Check if the labels are the same for the new TCE table

newTceTbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17_DR25/'
                        'q1_q17_dr25_tce_2020.09.15_15.12.12_stellar_koi_cfp_norobovetterlabels_rmcandandfpkois_'
                        'norogues_renamedcols.csv')

oldTceTbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17_DR25/'
                        'q1_q17_dr25_tce_2020.04.15_23.19.10_cumkoi_2020.02.21_shuffledstar_noroguetces_'
                        'noRobovetterKOIs.csv')

# they have the same number of TCEs
assert len(newTceTbl) == len(oldTceTbl)

for tce_i, tce in oldTceTbl.iterrows():

    # check that TCE is there and the labels match
    matchedTce = newTceTbl.loc[(newTceTbl['target_id'] == tce['target_id']) &
                               (newTceTbl['tce_plnt_num'] == tce['tce_plnt_num']) &
                               (newTceTbl['label'] == tce['label'])]

    assert len(matchedTce) == 1

#%% add TCE max MES to TCE table

tceTblFp = '/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17_DR25/q1_q17_dr25_tce_2020.09.15_15.12.12_stellar.csv'
tceTbl = pd.read_csv(tceTblFp, header=0)

addTbl = pd.read_csv('/home/msaragoc/Downloads/q1_q17_dr25_tce_2020.09.17_16.33.52.csv', header=9)

tceTbl['tce_maxmes'] = np.nan

for tce_i, tce in tceTbl.iterrows():

    tceTbl.loc[tce_i, 'tce_maxmes'] = addTbl.loc[(addTbl['kepid'] == tce['kepid']) &
                                                  (addTbl['tce_plnt_num'] ==
                                                   tce['tce_plnt_num'])]['tce_maxmes'].values[0]

tceTbl.to_csv(tceTblFp, index=False)

#%% Shuffle TCEs in Q1-Q17 DR25 TCE table

# np.random.seed(24)
random.seed(24)

tceTbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17_DR25/'
                     '{}_stellar_koi_cfp_norobovetterlabels_renamedcols_'
                     'rmcandandfpkois_norogues.csv'.format(baseTceTbl))

# shuffle at target star level
targetStarGroups = [df for _, df in tceTbl.groupby('target_id')]
random.shuffle(targetStarGroups)
tceTblShuffled = pd.concat(targetStarGroups).reset_index(drop=True)

# # shuffle at a TCE level
# tceTblShuffled = tceTbl.iloc[np.random.permutation(len(tceTbl))]

tceTblShuffled.to_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17_DR25/'
                      '{}_stellar_koi_cfp_norobovetterlabels_renamedcols_'
                      'rmcandandfpkois_norogues_shuffle.csv'.format(baseTceTbl), index=False)

#%% Choose random rows from TCE table

tceTbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17_DR25/'
                     'q1_q17_dr25_tce_2020.09.28_10.36.22_stellar_koi_cfp_norobovetterlabels_renamedcols_nomissingval_'
                     'rmcandandfpkois_norogues.csv')

numTces = {
    'CONFIRMED': 10,
    'POSSIBLE PLANET': 10,
    'CERTIFIED FA': 10,
    'CERTIFIED FP': 10,
    'NTP': 10
}
confirmedSubTbl = tceTbl.loc[tceTbl['koi_disposition'] == 'CONFIRMED'].sample(numTces['CONFIRMED'])
ppSubTbl = tceTbl.loc[(tceTbl['label'] == 'PC') & (tceTbl['fpwg_disp_status'] == 'POSSIBLE PLANET')].sample(numTces['POSSIBLE PLANET'])
cfpSubTbl = tceTbl.loc[(tceTbl['label'] == 'AFP') & (tceTbl['fpwg_disp_status'] == 'CERTIFIED FP')].sample(numTces['CERTIFIED FP'])
cfaSubTbl = tceTbl.loc[(tceTbl['label'] == 'NTP') & (tceTbl['fpwg_disp_status'] == 'CERTIFIED FA')].sample(numTces['CERTIFIED FA'])
ntpSubTbl = tceTbl.loc[(tceTbl['label'] == 'NTP') & (tceTbl['fpwg_disp_status'] != 'CERTIFIED FA')].sample(numTces['NTP'])

subTbl = pd.concat([confirmedSubTbl, ppSubTbl, cfpSubTbl, cfaSubTbl, ntpSubTbl], axis=0)

subTbl.to_csv('/home/msaragoc/Downloads/subtbl.csv', index=False)

