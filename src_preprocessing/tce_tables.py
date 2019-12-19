# 3rd party
import pandas as pd
from scipy import io
import numpy as np

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

# normalize using statistics computed in the training set
trainingset_idx = int(len(tce_tbl) * 0.8)

tce_tbl[stellar_params].hist()

stellar_params_med = tce_tbl[stellar_params][:trainingset_idx].median(axis=0, skipna=True)
stellar_params_std = tce_tbl[stellar_params][:trainingset_idx].std(axis=0, skipna=True)

tce_tbl[stellar_params] = (tce_tbl[stellar_params] - stellar_params_med) / stellar_params_std

tce_tbl[stellar_params].hist()
tce_tbl[stellar_params].median(axis=0, skipna=True)
tce_tbl[stellar_params].std(axis=0, skipna=True)

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
