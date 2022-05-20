import logging
# from astropy.stats import mad_std
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import io

# %% Check transit depth for Kepler dataset

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

#%% Replace Confirmed KOI TCE period for KOI period

baseTceTbl = 'q1_q17_dr25_tce_2020.09.28_10.36.22'

tceTbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17_DR25/'
                     '{}_stellar_koi_cfp_norobovetterlabels_renamedcols_nomissingval_'
                     'symsecphase.csv'.format(baseTceTbl))


def _replace_tce_for_koi_ephem(row):

    if isinstance(row['kepoi_name'], str) and row['koi_disposition'] == 'CONFIRMED':
        row[['tce_period']] = row[['koi_period']].values

    return row[['tce_period']]


tceTbl[['tce_period']] = tceTbl.apply(_replace_tce_for_koi_ephem, axis=1)

tceTbl.to_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17_DR25/{}_stellar'
              '_koi_cfp_norobovetterlabels_renamedcols_nomissingval_secsymphase_'
              'confirmedkoiperiod.csv'.format(baseTceTbl), index=False)

#%% recalculate secondary parameters


def estimate_sec_geo_albedo(tr_depth, tr_depth_unc, plnt_radius, plnt_radius_unc, sma, sma_unc):

    sg_albedo = tr_depth * (sma / plnt_radius) ** 2
    sg_albedo_unc = sg_albedo * ((tr_depth_unc / tr_depth) ** 2 +
                                 (2 * sma_unc / sma) ** 2 +
                                 (2 * plnt_radius_unc / plnt_radius) ** 2) ** 0.5

    return sg_albedo, sg_albedo_unc


def estimate_plnt_eff_temp(st_eff_temp, st_eff_temp_unc, tr_depth, tr_depth_unc, ror, ror_unc):

    p_efftemp = st_eff_temp * tr_depth ** 0.25 * ror ** (-0.5)
    p_efftemp_unc = p_efftemp * ((st_eff_temp_unc / st_eff_temp) ** 2 +
                                 (tr_depth_unc / (4 * tr_depth) ** 2 +
                                  (ror_unc / (2 * ror)) ** 2)) ** 0.5

    return p_efftemp, p_efftemp_unc


def compute_sec_geo_albedo_stat(sg_albedo, sg_albedo_unc):

    return (sg_albedo - 1) / sg_albedo_unc


def compute_plnt_eff_temp_stat(p_efftemp, p_efftemp_unc, p_eqtemp, p_eqtemp_unc):

    return(p_efftemp - p_eqtemp) / np.sqrt(p_efftemp_unc ** 2 + p_eqtemp_unc ** 2)


tce_tbl_fp = Path('/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17_DR25/q1_q17_dr25_tce_2020.09.28_10.36.22_stellar_koi_cfp_norobovetterlabels_renamedcols_nomissingval_symsecphase_confirmedkoiperiod.csv')
tce_tbl = pd.read_csv(tce_tbl_fp)

# set up logger
logger = logging.getLogger(name='correct_secondary')
logger_handler = logging.FileHandler(filename=Path(tce_tbl_fp.parent / f'correct_secondary_params.log', mode='w'))
logger_formatter = logging.Formatter('%(asctime)s - %(message)s')
logger.setLevel(logging.INFO)
logger_handler.setFormatter(logger_formatter)
logger.addHandler(logger_handler)
logger.info(f'Starting...')

logger.info(f'TCE table: {tce_tbl_fp}')

sec_match_tbl_fp = "/home/msaragoc/Projects/Kepler-TESS_exoplanet/Analysis/wks_tce_match/match_tces_['max_period_diff', 'max_phase_diff_rel']_[0.005, 0.01]_23-10-20_12:54_final.csv"
logger.info(f'Secondary TCE matching table: {sec_match_tbl_fp}')
sec_match_tbl = pd.read_csv(sec_match_tbl_fp)

logger.info(f'Number of secondary TCEs matched out of {len(tce_tbl)} TCEs: {len(sec_match_tbl)}')

cols_sec = ['wst_depth', 'tce_maxmes', 'tce_albedo', 'tce_ptemp', 'tce_albedo_stat', 'tce_ptemp_stat']
tce_tbl_original = tce_tbl[cols_sec].rename(lambda x: f'{x}_original', axis=1).reset_index(drop=True)
tce_tbl = pd.concat([tce_tbl, tce_tbl_original], axis=1)

missval = {'tce_albedo_stat': 0, 'tce_ptemp_stat': 0, 'tce_ptemp': 0, 'tce_ptemp_err': -1, 'tce_albedo': 0,
           'tce_albedo_err': -1}
logger.info(f'Missing values for: {missval}')

count_missval = {'albedo': 0, 'ptemp': 0, 'albedo_stat': 0, 'ptemp_stat': 0}
count_skip_tces = 0
for tce_i, tce in tce_tbl.iterrows():

    matched_tce = sec_match_tbl.loc[(sec_match_tbl['target_id'] == tce['target_id']) &
                                    (sec_match_tbl['secondary_tce'] == tce['tce_plnt_num'])]

    if len(matched_tce) == 0:  # skip non-matched TCEs
        count_skip_tces += 1
        continue

    primary_tce = tce_tbl.loc[(tce_tbl['target_id'] == matched_tce['target_id'].values[0]) &
                              (tce_tbl['tce_plnt_num'] == matched_tce['primary_tce'].values[0])]

    # the secondary depth and MES are replaced by the values of the primary TCE
    tce_tbl.loc[tce_i, ['wst_depth', 'tce_maxmes']] = \
        primary_tce[['transit_depth', 'tce_max_mult_ev']].values[0]

    # the secondary geometric albedo and planet effective compute are estimated again taking into account that the TCE
    # now points to a different secondary

    # deal with missing values: set to (0, -1)
    if primary_tce['transit_depth'].values[0] <= 0 or primary_tce['tce_depth_err'].values[0] == -1 \
        or tce['tce_prad'] <= 0 or tce['tce_prad_err'] == -1 or tce['tce_sma'] == 0 or tce['tce_sma_err'] == -1:
        logger.info(f'Secondary Geometric Albedo not estimated for {tce["target_id"]}-{tce["tce_plnt_num"]} due to '
                     f'missing values')
        count_missval['albedo'] += 1
        tce_albedo_est, tce_albedo_unc_est = missval['tce_albedo'], missval['tce_albedo_err']
    else:
        tce_albedo_est, tce_albedo_unc_est = estimate_sec_geo_albedo(primary_tce['transit_depth'].values[0] * 1e-6,
                                                                     primary_tce['tce_depth_err'].values[0] * 1e-6,
                                                                     tce['tce_prad'] * 6371,
                                                                     tce['tce_prad_err'] * 6371,
                                                                     tce['tce_sma'] * 1.496e8,
                                                                     tce['tce_sma_err'] * 1.496e8)

    tce_tbl.loc[tce_i, ['tce_albedo', 'tce_albedo_err']] = [tce_albedo_est, tce_albedo_unc_est]

    if tce['tce_steff'] == 0 or tce['tce_steff_err'] == -1 or primary_tce['transit_depth'].values[0] <= 0 \
        or primary_tce['tce_depth_err'].values[0] == -1 or tce['tce_ror'] == 0 or tce['tce_ror_err'] == -1:
        logger.info(f'Planet Effective Temperature not estimated for {tce["target_id"]}-{tce["tce_plnt_num"]} due to'
                     f' missing values')
        count_missval['ptemp'] += 1
        tce_plnt_eff_temp_est, tce_plnt_eff_temp_unc_est = missval['tce_ptemp'], missval['tce_ptemp_err']
    else:
        tce_plnt_eff_temp_est, tce_plnt_eff_temp_unc_est = estimate_plnt_eff_temp(tce['tce_steff'],
                                                                                  tce['tce_steff_err'],
                                                                                  primary_tce['transit_depth'].values[
                                                                                      0],
                                                                                  primary_tce['tce_depth_err'].values[
                                                                                      0] * 1e-6,
                                                                                  tce['tce_ror'],
                                                                                  tce['tce_ror_err'])

    tce_tbl.loc[tce_i, ['tce_ptemp', 'tce_ptemp_err']] = [tce_plnt_eff_temp_est, tce_plnt_eff_temp_unc_est]

    # compute secondary geometric albedo comparison statistic
    if tce_albedo_est != 0:
        tce_tbl.loc[tce_i, 'tce_albedo_stat'] = compute_sec_geo_albedo_stat(tce_tbl.loc[tce_i, 'tce_albedo'],
                                                                            tce_tbl.loc[tce_i, 'tce_albedo_err'])
    else:
        logger.info(f'Secondary Geometric Albedo Comparison Statistic not computed for '
                    f'{tce["target_id"]}-{tce["tce_plnt_num"]} due to missing values')
        count_missval['albedo_stat'] += 1
        tce_tbl.loc[tce_i, 'tce_albedo_stat'] = missval['tce_albedo_stat']

    # compute effective temperature comparison statistic
    if tce_plnt_eff_temp_est != 0 and tce_plnt_eff_temp_unc_est != -1 and tce_tbl.loc[tce_i, 'tce_eqt'] != 0 and \
            tce_tbl.loc[tce_i, 'tce_eqt_err'] != -1:
        tce_tbl.loc[tce_i, 'tce_ptemp_stat'] = compute_plnt_eff_temp_stat(tce_tbl.loc[tce_i, 'tce_ptemp'],
                                                                          tce_tbl.loc[tce_i, 'tce_ptemp_err'],
                                                                          tce_tbl.loc[tce_i, 'tce_eqt'],
                                                                          tce_tbl.loc[tce_i, 'tce_eqt_err'])
    else:
        logger.info(f'Effective Temperature Comparison Statistic not computed for '
                    f'{tce["target_id"]}-{tce["tce_plnt_num"]} due to missing values')
        count_missval['ptemp_stat'] += 1
        tce_tbl.loc[tce_i, 'tce_ptemp_stat'] = missval['tce_ptemp_stat']

logger.info(f'Number of TCE with missing values for: {count_missval}')
logger.info(f'Number of TCEs skipped: {count_skip_tces}')
logger.info(f'Number of TCEs not skipped: {len(tce_tbl) - count_skip_tces}')
logger.info(f'Number of matched TCEs to a primary TCE: {len(sec_match_tbl)}')

tce_tbl.to_csv(tce_tbl_fp.parent / (tce_tbl_fp.stem + '_sec.csv'), index=False)

#%% Replace Confirmed KOIs period in the TPS TCE table

tps_tbl_fp = Path('/data5/tess_project/Data/Ephemeris_tables/Kepler/TPS_tables/Q1-Q17_DR25/keplerTPS_KSOP2536_dr25.csv')
tps_tbl = pd.read_csv(tps_tbl_fp)

dv_tbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17_DR25/'
                     'q1_q17_dr25_tce_2020.09.28_10.36.22_stellar_koi_cfp_norobovetterlabels_renamedcols_nomissingval_'
                     'symsecphase_confirmedkoiperiod_sec.csv')

sec_match_tbl = pd.read_csv("/home/msaragoc/Projects/Kepler-TESS_exoplanet/Analysis/wks_tce_match/match_tces_"
                            "['max_period_diff', 'max_phase_diff_rel']_[0.005, 0.01]_23-10-20_12:54_final.csv")

count_tces = {'confirmed_period': 0, 'sec_phase': 0}
for tce_i, tce in tps_tbl.iterrows():

    tce_dv = dv_tbl.loc[(dv_tbl['target_id'] == tce['target_id']) & (dv_tbl['tce_plnt_num'] == tce['tce_plnt_num'])]

    if tce_dv['koi_disposition'].values[0] == 'CONFIRMED':
        tps_tbl.loc[tce_i, 'tce_period'] = tce_dv['tce_period'].values[0]
        count_tces['confirmed_period'] += 1

    tce_sec_match = sec_match_tbl.loc[(sec_match_tbl['target_id'] == tce['target_id']) &
                                      (sec_match_tbl['secondary_tce'] == tce['tce_plnt_num'])]

    # if len(tce_sec_match) > 0:
    #     tps_tbl.loc[tce_i, 'tce_maxmesd'] = sec_match_tbl['tce_maxmesd_secondary_new'].values[0]
    #     count_tces['sec_phase'] += 1

tps_tbl.to_csv(tps_tbl_fp.parent / 'keplerTPS_KSOP2536_dr25_symsecphase_confirmedkoiperiod.csv', index=False)

#%%

from pathlib import Path
import pandas as pd

experiment_dir = Path('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Data/tfrecords/Kepler/Q1-Q17_DR25/plot_confirmed-cfp_3-2-2021_test_centroidnocorr/')

csv_file = pd.concat([pd.read_csv(el) for el in experiment_dir.iterdir() if el.suffix == '.csv'])
csv_file.to_csv(experiment_dir / 'all_shards.csv', index=False)