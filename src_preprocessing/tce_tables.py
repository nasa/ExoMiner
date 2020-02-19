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

#%% Normalize stellar parameters in the Kepler TCE table

tce_tbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/'
                      'q1_q17_dr25_tce_2019.03.12_updt_tcert_extendedtceparams_updt_stellarparamswitherrors.csv')

stellar_params = ['tce_sradius', 'tce_steff', 'tce_slogg', 'tce_smet', 'tce_smass', 'tce_sdens']

# fill in NaNs using Solar parameters
tce_tbl['tce_smass'] = tce_tbl['tce_smass'].fillna(value=1.0)
tce_tbl['tce_sdens'] = tce_tbl['tce_sdens'].fillna(value=1.408)

# normalize using statistics computed in the training set
trainingset_idx = int(len(tce_tbl) * 0.8)

ax = tce_tbl[stellar_params].hist(figsize=(15, 8))
ax[0, 0].set_title('Stellar density')
ax[0, 0].set_ylabel('g/cm^3')
ax[0, 1].set_title('Stellar surface gravity')
ax[0, 1].set_ylabel('log10(cm/s^2)')
ax[1, 0].set_title('Stellar mass')
ax[1, 0].set_ylabel('Solar masses')
ax[1, 1].set_title('Stellar metallicity (Fe/H)')
ax[1, 1].set_ylabel('dex')
ax[2, 0].set_title('Stellar radius')
ax[2, 0].set_ylabel('Solar radii')
ax[2, 0].set_xlabel('Bins')
ax[2, 1].set_title('Stellar effective temperature')
ax[2, 1].set_ylabel('K')
ax[2, 1].set_xlabel('Bins')
plt.suptitle('Stellar parameters')

stellar_params_med = tce_tbl[stellar_params][:trainingset_idx].median(axis=0, skipna=False)
stellar_params_std = tce_tbl[stellar_params][:trainingset_idx].std(axis=0, skipna=False)

tce_tbl[stellar_params] = (tce_tbl[stellar_params] - stellar_params_med) / stellar_params_std

ax = tce_tbl[stellar_params].hist(figsize=(15, 8))
ax[0, 0].set_title('Stellar density')
ax[0, 0].set_ylabel('Amplitude')
ax[0, 1].set_title('Stellar surface gravity')
ax[0, 1].set_ylabel('Amplitude')
ax[1, 0].set_title('Stellar mass')
ax[1, 0].set_ylabel('Amplitude')
ax[1, 1].set_title('Stellar metallicity (Fe/H)')
ax[1, 1].set_ylabel('Amplitude')
ax[2, 0].set_title('Stellar radius')
ax[2, 0].set_ylabel('Amplitude')
ax[2, 0].set_xlabel('Bins')
ax[2, 1].set_title('Stellar effective temperature')
ax[2, 1].set_ylabel('Amplitude')
ax[2, 1].set_xlabel('Bins')
plt.suptitle('Normalized stellar parameters')

tce_tbl[stellar_params].median(axis=0, skipna=False)
tce_tbl[stellar_params].std(axis=0, skipna=False)

tce_tbl.to_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/'
               'q1_q17_dr25_tce_2019.03.12_updt_tcert_extendedtceparams_updt_normstellarparamswitherrors.csv',
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

#%% Update fields in TEV MIT TOI disposition lists to standardized fields; removed TCEs with no value for the required
# parameters

rawTceTable = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/TESS/TEV_MIT_TOI_lists/'
                          'toi-plus-tev.mit.edu_2020-01-15.csv', header=4)
print(len(rawTceTable))

disposition_src = 'Group Disposition'  # 'Group Disposition or 'TOI disposition'
# Create Group or TOI disposition table by changing the field name in rawFields
rawFields = ['TIC', 'Full TOI ID', disposition_src, 'TIC Right Ascension', 'TIC Declination', 'TMag Value',
             'TMag Uncertainty', 'Orbital Epoch Value', 'Orbital Epoch Error', 'Orbital Period Value',
             'Orbital Period Error', 'Transit Duration Value', 'Transit Duration Error', 'Transit Depth Value',
             'Transit Depth Error', 'Sectors']
newFields = ['target_id', 'oi', 'label', 'ra', 'dec', 'mag', 'mag_uncert', 'tce_time0bk', 'tce_time0bk_err',
             'tce_period', 'tce_period_err', 'tce_duration', 'tce_duration_err', 'transit_depth', 'transit_depth_err',
             'sectors']

# remove TCEs with any NaN in the required fields
rawTceTable.dropna(axis=0, subset=np.array(rawFields)[[0, 1, 2, 3, 4, 7, 9, 11, 13]], inplace=True)
print(len(rawTceTable))

# rename fields to standardize fieldnames
renameDict = {}
for i in range(len(rawFields)):
    renameDict[rawFields[i]] = newFields[i]
rawTceTable.rename(columns=renameDict, inplace=True)

# Group disposition: 1604 to 1326 TCEs; TOI disposition: 1604 to 1571 TCEs
rawTceTable.to_csv('/data5/tess_project/Data/Ephemeris_tables/TESS/TEV_MIT_TOI_lists/final_tce_tables/'
                   'toi-plus-tev.mit.edu_2020-01-15_{}_processed.csv'.format(disposition_src), index=False)

#%% Update fields in NASA Exoplanet Archive disposition list to standardized fields; removed TCEs with no value for the
# required parameters

rawTceTable = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/TESS/NASA_Exoplanet_Archive_TOI_lists/'
                          'TOI_2020.01.21_13.55.10.csv', header=72)
print(len(rawTceTable))

# Create Group or TOI disposition table by changing the field name in rawFields
rawFields = ['toi', 'tid', 'tfopwg_disp', 'ra', 'dec', 'st_tmag', 'st_tmagerr1', 'st_tmagerr2', 'pl_tranmid',
             'pl_tranmiderr1', 'pl_tranmiderr2', 'pl_orbper',
             'pl_orbpererr1', 'pl_orbpererr2', 'pl_trandurh', 'pl_trandurherr1', 'pl_trandurherr2', 'pl_trandep',
             'pl_trandeperr1', 'pl_trandeperr2']
newFields = ['oi', 'target_id', 'label', 'ra', 'dec', 'mag', 'mag_uncert', 'mag_uncert2', 'tce_time0bk',
             'tce_time0bk_err', 'tce_time0bk_err2', 'tce_period', 'tce_period_err', 'tce_period_err2', 'tce_duration',
             'tce_duration_err', 'tce_duration_err2', 'transit_depth', 'transit_depth_err', 'transit_depth_err2']

# remove TCEs with any NaN in the required fields
rawTceTable.dropna(axis=0, subset=np.array(rawFields)[[0, 1, 2, 3, 4, 8, 11, 14, 17]], inplace=True)
print(len(rawTceTable))

# rename fields to standardize fieldnames
renameDict = {}
for i in range(len(rawFields)):
    renameDict[rawFields[i]] = newFields[i]
rawTceTable.rename(columns=renameDict, inplace=True)

# 1604 to 536 TCEs
rawTceTable.to_csv('/data5/tess_project/Data/Ephemeris_tables/TESS/NASA_Exoplanet_Archive_TOI_lists/final_tce_tables/'
                   'TOI_2020.01.21_13.55.10.csv_TFOPWG_processed.csv', index=False)

#%% Update fields in EXOFOP Community disposition list to standardized fields; removed TCEs with no value for the
# required parameters

rawTceTable = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/TESS/EXOFOP_TOI_lists/exofop_ctoilists.csv',
                          header=0)
print(len(rawTceTable))

# Create Group or TOI disposition table by changing the field name in rawFields
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

# 321 to 275 TCEs
rawTceTable.to_csv('/data5/tess_project/Data/Ephemeris_tables/TESS/EXOFOP_TOI_lists/final_tce_tables/'
                   'exofop_ctoilists_Community_processed.csv', index=False)
