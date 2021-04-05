""" Analyze the distribution of scalar features. """

import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from astropy.stats import mad_std

#%% Non-normalized bootstrap FA probability from the TCE table

tceTbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17_DR25/'
                     'q1_q17_dr25_tce_2020.04.15_23.19.10_cumkoi_2020.02.21_shuffled.csv')

print(len(tceTbl.loc[tceTbl['boot_fap'] == -1]))
print(len(tceTbl.loc[(tceTbl['boot_fap'] == -1) & (tceTbl['tce_rogue_flag'] == 0)]))

f, ax = plt.subplots()
ax.boxplot(tceTbl['boot_fap'].values + np.abs(tceTbl['boot_fap'].values.min()))
# ax.set_yscale('log')
ax.set_xticklabels(['Bootstrap FA Probability'])
ax.set_ylim([0, 1])
f.savefig('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Analysis/scalar_params_analysis/boxplot_bfap_nonnorm_translatelogspace.png')

bins = np.logspace(-32, 0, num=100, endpoint=True)
f, ax = plt.subplots()
ax.hist(tceTbl['boot_fap'].values, bins=bins, range=(-1, 1), edgecolor='k')
ax.axvline(x=1.56179999807e-33, c='r')
ax.axvline(x=1.56179999807e-33-0.0000000000313199708567, c='y')
ax.axvline(x=1.56179999807e-33+0.0000000000313199708567, c='y')
ax.set_xscale('log')
ax.set_xlim([-1, 1])
ax.set_ylabel('Counts')
ax.set_xlabel('Bootstrap FA probability')
f.savefig('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Analysis/scalar_params_analysis/hist_bfap_nonorm.png')

#%% Normalized bootstrap FA probability from the TFRecords

tfrecDir = '/data5/tess_project/Data/tfrecords/Kepler/Q1-Q17_DR25/tfrecordskeplerdr25_g2001-l201_spline_gapped_flux-centroid_selfnormalized-oddeven-wks-scalar_data/tfrecordskeplerdr25_g2001-l201_spline_gapped_flux-centroid_selfnormalized-oddeven-wks-scalar_starshuffle_experiment-labels-norm'

tfrecFiles = [os.path.join(tfrecDir, filename) for filename in os.listdir(tfrecDir) if 'shard' in filename]

bfapArr = []

for tfrecFile in tfrecFiles:

        record_iterator = tf.python_io.tf_record_iterator(path=tfrecFile)

        for string_i, string_record in enumerate(record_iterator):
            example = tf.train.Example()
            example.ParseFromString(string_record)

            # normalize scalar parameters
            tceScalarParams = np.array(example.features.feature['scalar_params'].float_list.value)

            bfapArr.append(tceScalarParams[7])

f, ax = plt.subplots()
ax.boxplot(np.array(bfapArr) + np.abs(np.min(bfapArr)))
ax.set_yscale('log')
ax.set_xticklabels(['Bootstrap FA Probability'])
f.savefig('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Analysis/scalar_params_analysis/boxplot_bfap_norm_translatelogspace.png')

# bins = np.logspace(-3, 1, num=1000, endpoint=True)
bins = np.linspace(-10, 10, num=100, endpoint=True)
f, ax = plt.subplots()
ax.hist(bfapArr, bins=bins, edgecolor='k')
# ax.set_xscale('log')
ax.set_yscale('log')
# ax.set_xlim([-1, 1])
ax.set_ylabel('Counts')
ax.set_xlabel('Bootstrap FA probability')
f.savefig('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Analysis/scalar_params_analysis/hist_bfap_norm_translatelogspace.png')

#%% Non-normalized bootstrap FA probability from the TCE table

tceTbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17_DR25/'
                     'q1_q17_dr25_tce_2020.04.15_23.19.10_cumkoi_2020.02.21_shuffled.csv')

f, ax = plt.subplots()
ax.boxplot([tceTbl['tce_cap_stat'].values + np.abs(tceTbl['tce_cap_stat'].values.min()),
            tceTbl['tce_hap_stat'].values + np.abs(tceTbl['tce_hap_stat'].values.min())])
ax.set_yscale('log')
ax.set_xticklabels(['CAP', 'HAP'])
f.savefig('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Analysis/scalar_params_analysis/boxplot_hap-capstats_nonnorm_translatelogspace.png')

# bins = np.logspace(-3, 2, num=100, endpoint=True)
bins = np.linspace(-1e2, 1e2, num=100, endpoint=True)
f, ax = plt.subplots()
ax.hist(tceTbl['tce_cap_stat'].values, bins=bins, edgecolor='k')
# ax.set_xscale('log')
ax.axvline(x=2.625, c='r')
ax.axvline(x=2.625-3.89183082358, c='y')
ax.axvline(x=2.625+3.89183082358, c='y')
ax.set_ylabel('Counts')
ax.set_xlabel('CAP stat')
f.savefig('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Analysis/scalar_params_analysis/hist_capstat_nonnorm_linspace-1e2_1e2.png')

# bins = np.logspace(-3, 2, num=100, endpoint=True)
bins = np.linspace(-1e2, 1e2, num=100, endpoint=True)
f, ax = plt.subplots()
ax.hist(tceTbl['tce_hap_stat'].values, bins=bins, edgecolor='k')
ax.axvline(x=1.80400002003, c='r')
ax.axvline(x=1.80400002003-2.96698358979, c='y')
ax.axvline(x=1.80400002003+2.96698358979, c='y')
# ax.set_xscale('log')
ax.set_ylabel('Counts')
ax.set_xlabel('HAP stat')
f.savefig('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Analysis/scalar_params_analysis/'
          'hist_hapstat_nonnorm_linspace-1e2_1e2.png')

#%% Histogram for each TCE parameter

scalarParams = ['wst_robstat', 'tce_bin_oedp_stat', 'boot_fap', 'tce_cap_stat', 'tce_hap_stat']

tceTbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17_DR25/'
                     'q1_q17_dr25_tce_2020.04.15_23.19.10_cumkoi_2020.02.21_shuffledstar_noroguetces_'
                     'noRobobvetterKOIs.csv', usecols=scalarParams + ['label'])

# scalarParamsNormStats = np.load('/data5/tess_project/Data/tfrecords/Kepler/Q1-Q17_DR25/tfrecordskeplerdr25_g2001-l201_spline_gapped_flux-centroid_selfnormalized-oddeven-wks-scalar_data/tfrecordskeplerdr25_g2001-l201_spline_gapped_flux-centroid_selfnormalized-oddeven-wks-scalar_starshuffle_experiment/train_scalarparam_norm_stats.npy', allow_pickle=True).item()
#
# scalarParamsOrder = ['tce_steff', 'tce_slogg', 'tce_smet', 'tce_sradius', 'wst_robstat', 'wst_depth', 'tce_bin_oedp_stat',
#                      'boot_fap', 'tce_smass', 'tce_sdens', 'tce_cap_stat', 'tce_hap_stat']
# for scalarParam in scalarParamsNormStats:
#     scalarParamIdx = np.where(scalarParamsOrder == scalarParam)[0]
#     tceTbl[scalarParam] = (tceTbl[scalarParam] - scalarParamsNormStats['median'][scalarParamIdx]) / scalarParamsNormStats['std'][scalarParamIdx]

for scalarParam in scalarParams:
    aaa
    scalarParamPC = tceTbl.loc[tceTbl['label'] == 'PC'][scalarParam]
    scalarParamNonPC = tceTbl.loc[tceTbl['label'] != 'PC'][scalarParam]
    bins = np.linspace(-1000, 4000, num=100, endpoint=True)  # tce_hap_stat
    # bins = np.linspace(-1000, 4000, num=100, endpoint=True)  # tce_cap_stat
    # bins = np.linspace(0, 1e-15, num=1000, endpoint=True)  # boot_fap
    # bins = np.linspace(-20, 20, num=100, endpoint=True)  # wst_robstat
    # bins = np.linspace(0, 5, num=100, endpoint=True)  # tce_bin_oedp_stat
    f, ax = plt.subplots()
    ax.hist(scalarParamPC.values, bins=bins, label='PC', color='b', alpha=1, zorder=10, cumulative=False, edgecolor='k')
    ax.hist(scalarParamNonPC.values, bins=bins, label='non-PC', color='r', alpha=0.8, cumulative=False, edgecolor='k')
    ax.set_xlabel('Values')
    ax.set_title(scalarParam)
    ax.set_ylabel('Counts')
    ax.set_yscale('log')
    # ax.set_xscale('log')
    # ax.set_xlim([0, 5])  # tce_bin_oedp_stat
    # ax.set_xlim([0, 1e-15])  # boot_fap
    # ax.set_xlim([-1000, 4000])  # tce_cap_stat
    # ax.set_xlim([-250, 2000])  # tce_hap_stat
    ax.legend()

#%%

# tceTbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17_DR25/q1_q17_dr25_tce_2020.04.15_23.19.10_stellar.csv')
# tceTbl['tce_cap_stat'] = np.nan
# tceTbl['tce_hap_stat'] = np.nan
# tceTbl['tce_maxmesd'] = np.nan
#
# tceTblGhost = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17_DR25/q1_q17_dr25_tce_2020.04.15_23.19.10_cumkoi_2020.02.21_shuffled.csv')
#
# for tce_i, tce in tceTblGhost.iterrows():
#
#     tceTbl.loc[(tceTbl['kepid'] == tce['target_id']) & (tce['tce_plnt_num'] == tce['tce_plnt_num']), ['tce_cap_stat', 'tce_hap_stat', 'tce_maxmesd']] = [tce['tce_cap_stat'], tce['tce_hap_stat'], tce['tce_maxmesd']]
#
# tceTbl.to_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17_DR25/q1_q17_dr25_tce_2020.04.15_23.19.10_stellar.csv', index=False)

#%% plot histogram of scalar parameters values stored in the TFRecords

# TFRecord directory
tfrecDir = '/data5/tess_project/Data/tfrecords/Kepler/Q1-Q17_DR25/tfrecordskeplerdr25_g2001-l201_spline_gapped_flux-centroid_selfnormalized-oddeven-wks-scalar_data/tfrecordskeplerdr25_g2001-l201_spline_gapped_flux-centroid_selfnormalized-oddeven-wks-scalar_starshuffle_experiment-labels-norm'
# plots saving directory
plotDir = '/data5/tess_project/Data/tfrecords/Kepler/Q1-Q17_DR25/tfrecordskeplerdr25_g2001-l201_spline_gapped_flux-centroid_selfnormalized-oddeven-wks-scalar_data/hist_scalarparams_norm'
os.makedirs(plotDir, exist_ok=True)
tfrecFiles = [os.path.join(tfrecDir, file) for file in os.listdir(tfrecDir) if 'shard' in file]

# order of scalar parameters in the scalar parameter array in the TFRecords
scalarParams = np.array(['tce_steff', 'tce_slogg', 'tce_smet', 'tce_sradius', 'wst_robstat', 'wst_depth',
                         'tce_bin_oedp_stat', 'boot_fap', 'tce_smass', 'tce_sdens', 'tce_cap_stat', 'tce_hap_stat'])

scalarParamsMat = []
labelArr = []
for tfrecFile in tfrecFiles:

    record_iterator = tf.python_io.tf_record_iterator(path=tfrecFile)

    for string_i, string_record in enumerate(record_iterator):
        example = tf.train.Example()
        example.ParseFromString(string_record)

        scalarParamsMat.append(np.array(example.features.feature['scalar_params'].float_list.value, dtype='float'))
        labelArr.append(example.features.feature['label'].bytes_list.value[0].decode("utf-8"))

scalarParamsMat = np.array(scalarParamsMat)  # convert list of NumPy arrays to NumPy 2D array
labelArr = np.array(labelArr)

# get indices of PC and non-PC TCEs
scalarParamPCIdxs = np.where(labelArr == 'PC')
# scalarParamNonPCIdxs = np.where(labelArr != 'PC')
scalarParamAFPIdxs = np.where(labelArr == 'AFP')
scalarParamNTPIdxs = np.where(labelArr == 'NTP')

# choose a scalar parameter
scalarParam = 'boot_fap'
scalarParamIdx = np.where(scalarParams == scalarParam)

# plot histogram
# bins = np.linspace(-1000, 4000, num=100, endpoint=True)  # tce_hap_stat
# bins = np.linspace(-1000, 4000, num=100, endpoint=True)  # tce_cap_stat
bins = np.linspace(-2, 2, num=100, endpoint=True)  # boot_fap
# bins = np.linspace(-20, 20, num=100, endpoint=True)  # wst_robstat
# bins = np.linspace(-18, 5, num=100, endpoint=True)  # tce_bin_oedp_stat
f, ax = plt.subplots()
ax.hist(scalarParamsMat[scalarParamPCIdxs, scalarParamIdx].ravel(), bins=bins, label='PC', color='b', alpha=.5, zorder=10, cumulative=False, edgecolor='k')
# ax.hist(scalarParamsMat[scalarParamNonPCIdxs, scalarParamIdx].ravel(), bins=bins, label='non-PC', color='r', alpha=0.8, cumulative=False, edgecolor='k')
ax.hist(scalarParamsMat[scalarParamAFPIdxs, scalarParamIdx].ravel(), bins=bins, label='AFP', color='r', alpha=.75, zorder=5, cumulative=False, edgecolor='k')
ax.hist(scalarParamsMat[scalarParamNTPIdxs, scalarParamIdx].ravel(), bins=bins, label='NTP', color='g', alpha=1, zorder=1, cumulative=False, edgecolor='k')
ax.set_xlabel('Values')
ax.set_title(scalarParam)
ax.set_ylabel('Counts')
ax.set_yscale('log')
# ax.set_xscale('log')
# ax.set_xlim([0, 5])  # tce_bin_oedp_stat
# ax.set_xlim([0, 1e-15])  # boot_fap
# ax.set_xlim([-1000, 4000])  # tce_cap_stat
# ax.set_xlim([-250, 2000])  # tce_hap_stat
ax.legend()

#%%

tceTbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17_DR25/'
                     'q1_q17_dr25_tce_2020.04.15_23.19.10_cumkoi_2020.02.21_shuffledstar_noroguetces_noRobovetterKOIs.csv')

columnName = 'tce_rb_tcount0'

tceTbl[columnName].describe()

diagnostic_var = tceTbl[columnName]  # .apply(lambda x: np.log10(x + 1))
# diagnostic_var = -1 + 2 * (diagnostic_var - diagnostic_var.min()) / (diagnostic_var.max() - diagnostic_var.min())
diagnostic_var = (diagnostic_var - tceTbl[columnName].median()) / tceTbl[columnName].std()
# diagnostic_var = tceTbl[columnName].apply(lambda x: -1 + 2 * (x - tceTbl[columnName].min()) / (tceTbl[columnName].max() - tceTbl[columnName].min()))

minval, maxval = diagnostic_var.min(), diagnostic_var.max()

bins = np.linspace(minval, maxval, 100, endpoint=True)
# bins = np.logspace(-1, 0, 1000, endpoint=True)

f, ax = plt.subplots()
ax.hist(diagnostic_var, bins=bins, edgecolor='k')
ax.set_xlabel('tce_rb_tcount0')
ax.set_ylabel('Counts')
ax.set_title('All TCEs')

bins = np.linspace(minval, maxval, 100, endpoint=True)
smean = diagnostic_var.mean()
smedian = diagnostic_var.median()
sstd = diagnostic_var.std()
smin = diagnostic_var.min()
smax = diagnostic_var.max()
f, ax = plt.subplots()
ax.hist(diagnostic_var, bins=bins, edgecolor='k')
ax.set_xlabel('tce_rb_tcount0')
ax.set_ylabel('Counts')
ax.set_title('All TCEs')
ax.axvline(x=smedian, c='y', label='median')
ax.axvline(x=smean, c='g', label='mean')
ax.axvline(x=smin, c='r', label='min')
ax.axvline(x=smax, c='r', label='max')
ax.set_title('Mean: {:.2f} | Median: {:.2f} |\n Std: {:.2f} | Min: {:.2f} |Max: {:.2f}'.format(smean, smedian, sstd,
                                                                                               smin, smax))
# ax.set_yscale('log')
ax.set_xlim(bins[[0, -1]])
ax.legend()
f.savefig('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Analysis/scalar_params_analysis/normalized_scalar_parameters/hist_{}-norm_keplerq1q7dr25.png'.format(columnName))


#%%

# labels = ['PC', 'AFP', 'NTP']
bins = np.linspace(0, 3000, 100, endpoint=True)
# for label in labels:

    # diagnostic_var = tceTbl.loc[tceTbl['label'] == label, [columnName]]
    # # aaa

f, ax = plt.subplots()
ax.hist(tceTbl.loc[tceTbl['label'] == 'PC', [columnName]].values, bins=bins, edgecolor='k', color='b', label='PC', zorder=10)
ax.hist(tceTbl.loc[tceTbl['label'] == 'AFP', [columnName]].values, bins=bins, edgecolor='k', color='r', label='AFP', zorder=5)
ax.hist(tceTbl.loc[tceTbl['label'] == 'NTP', [columnName]].values, bins=bins, edgecolor='k', color='g', label='NTP', zorder=1)
ax.set_xlabel('tce_rb_tcount0')
ax.set_ylabel('Counts')
ax.set_title('{}'.format(''))
ax.set_yscale('log')
ax.legend()
ax.set_xlim([0, 3000])
ax.grid(True)


#%% Check scalar features after normalization

# tfrecDir = '/data5/tess_project/Data/tfrecords/Kepler/Q1-Q17_DR25/tfrecordskeplerdr25_g2001-l201_spline_nongapped_flux-centroid_selfnormalized-oddeven-wks-scalar_globalbinwidthaslocal_data/tfrecordskeplerdr25_g2001-l201_spline_nongapped_flux-centroid_selfnormalized-oddeven-wks-scalar_globalbinwidthaslocal_starshuffle_experiment-labels-norm'
tfrecDir = '/data5/tess_project/Data/tfrecords/Kepler/Q1-Q17_DR25/tfrecordskeplerdr25_g2001-l201_gbal_splinenew_nongapped_flux-centroid-oddeven-wks-scalar_data/tfrecordskeplerdr25_g2001-l201_gbal_splinenew_nongapped_flux-centroid-oddeven-wks-scalar_starshuffle_experiment-labels-norm_rollingband'
# tfrecDir = '/data5/tess_project/Data/tfrecords/Kepler/Q1-Q17_DR25/tfrecordskeplerdr25_g2001-l201_spline_nongapped_flux-centroid_selfnormalized-oddeven-wks-scalar_data/tfrecordskeplerdr25_g2001-l201_spline_nongapped_flux-centroid_selfnormalized-oddeven-wks-scalar_centrmedcmaxncorr_starshuffle_experiment-labels-norm'

scalarFeaturesNames = ['tce_steff', 'tce_slogg', 'tce_smet', 'tce_sradius', 'boot_fap', 'tce_smass', 'tce_sdens',
                       'tce_cap_stat', 'tce_hap_stat']
# scalarFeaturesNames = ['tce_steff', 'tce_slogg', 'tce_smet', 'tce_sradius', 'boot_fap', 'tce_smass', 'tce_sdens',
#                        'tce_cap_stat', 'tce_hap_stat', 'tce_rb_tcount0']

scalarFeatures = []

tfrecFiles = [os.path.join(tfrecDir, tfrecFile) for tfrecFile in os.listdir(tfrecDir) if 'shard' in tfrecFile]

for tfrecFile in tfrecFiles:

    record_iterator = tf.python_io.tf_record_iterator(path=tfrecFile)

    for string_i, string_record in enumerate(record_iterator):
        example = tf.train.Example()
        example.ParseFromString(string_record)

        scalarFeatures.append(np.array(example.features.feature['scalar_params'].float_list.value, dtype='float')[np.array([0, 1, 2, 3, 7, 8, 9, 10, 11])])
        # scalarFeatures.append(np.array(example.features.feature['scalar_params'].float_list.value, dtype='float')[
        #                           np.array([0, 1, 2, 3, 7, 8, 9, 10, 11, 12])])

scalarFeatures = np.array(scalarFeatures)

bins = {
    'tce_steff': np.linspace(-4, 13, 100, endpoint=True),
    'tce_slogg': np.linspace(-11, 3, 100, endpoint=True),
    'tce_smet': np.linspace(-8, 3, 100, endpoint=True),
    'boot_fap': np.linspace(-2, 2, 100, endpoint=True),
    'tce_smass': np.linspace(-3, 6, 100, endpoint=True),
    'tce_sradius': np.linspace(-2, 15, 100, endpoint=True),
    'tce_sdens': np.linspace(-1, 15, 100, endpoint=True),
    'tce_cap_stat': np.linspace(-15, 40, 100, endpoint=True),
    'tce_hap_stat': np.linspace(-10, 40, 100, endpoint=True),
    'tce_rb_tcount0': np.linspace(-1, 30, 100, endpoint=True)
}
for i in [4]:  # range(len(scalarFeaturesNames)):

    smean = np.mean(scalarFeatures[:, i])
    smedian = np.median(scalarFeatures[:, i])
    sstd = np.std(scalarFeatures[:, i])
    smin = np.min(scalarFeatures[:, i])
    smax = np.max(scalarFeatures[:, i])

    f, ax = plt.subplots()
    ax.hist(scalarFeatures[:, i], bins='auto' if scalarFeaturesNames[i] not in bins else bins[scalarFeaturesNames[i]],
            edgecolor='k')
    ax.axvline(x=smedian, c='y', label='median')
    ax.axvline(x=smean, c='g', label='mean')
    ax.axvline(x=smin, c='r', label='min')
    ax.axvline(x=smax, c='r', label='max')
    ax.set_ylabel('Counts')
    ax.set_xlabel('{}'.format(scalarFeaturesNames[i]))
    ax.set_title('Mean: {:.2f} | Median: {:.2f} |\n Std: {:.2f} | Min: {:.2f} |Max: {:.2f}'.format(smean, smedian, sstd,
                                                                                                   smin, smax))
    # ax.set_yscale('log')
    ax.set_xlim(bins[scalarFeaturesNames[i]][[0, -1]])
    ax.legend()
    # f.savefig('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Analysis/scalar_params_analysis/normalized_scalar_parameters/hist_{}-norm_keplerq1q7dr25.png'.format(scalarFeaturesNames[i]))
    # plt.close('all')

#%% Analyze MES

tceTbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17_DR25/'
                     'q1_q17_dr25_tce_2020.09.15_15.12.12_stellar_koi_cfp_norobovetterlabels_rmcandandfpkois_'
                     'norogues_renamedcols.csv')

bins = np.linspace(0, 20, 100, endpoint=True)

stats = {
    'median': np.median(tceTbl['tce_max_mult_ev']),
    'mean': np.mean(tceTbl['tce_max_mult_ev']),
    'std_rob': mad_std(tceTbl['tce_max_mult_ev']),
    'std': np.std(tceTbl['tce_max_mult_ev'], ddof=1),
    'min': np.min(tceTbl['tce_max_mult_ev']),
    'max': np.max(tceTbl['tce_max_mult_ev'])
}

# tceTbl['tce_max_mult_ev'].clip(0, 20 * stats['std_rob'], inplace=True)  # clip values to see what changes in the histogram
f, ax = plt.subplots()
tceTbl['tce_max_mult_ev'].hist(bins=bins)
ax.set_ylabel('Counts')
ax.set_xlabel('MES')
ax.axvline(x=stats['median'], c='y', label='median')
ax.axvline(x=stats['median'] + stats['std_rob'], c='g', label='med+std_rob')
ax.axvline(x=stats['median'] - stats['std_rob'], c='b', label='med-std_rob')
ax.set_ylabel('Counts')
ax.set_xlabel('MES')
ax.set_title('Mean: {:.2f} | Median: {:.2f} |\n Std: {:.2f} | Std Rob: {:.2f} | Min: {:.2f} |Max: {:.2f}'.format(stats['mean'],
                                                                                               stats['median'],
                                                                                               stats['std'],
                                                                                               stats['std_rob'],
                                                                                               stats['min'],
                                                                                               stats['max']
                                                                                               )
             )
ax.legend()
# ax.set_xscale('log')
f.savefig('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Analysis/scalar_params_analysis/mes/'
          'hist_tce_mes_keplerq1q7dr25.svg')

labels = ['PC', 'AFP', 'NTP']
zorder = {'PC': 3, 'AFP': 2, 'NTP': 1}

f, ax = plt.subplots()
for label in labels:
    tceTbl.loc[tceTbl['label'] == label]['tce_max_mult_ev'].hist(bins=bins, label=label, edgecolor='k',
                                                                 zorder=zorder[label])
ax.set_ylabel('Counts')
ax.set_xlabel('MES')
ax.legend()
# ax.set_xscale('log')
f.savefig('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Analysis/scalar_params_analysis/mes/'
          'hist_tce_mes_keplerq1q7dr25_pc-afp-ntp.svg')

#%% Analyze Weak Secondary MES

# tceTbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17_DR25/'
#                      'q1_q17_dr25_tce_2020.09.15_15.12.12_stellar_koi_cfp_norobovetterlabels_renamedcols_'
#                      'rmcandandfpkois_norogues.csv')
tceTbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17_DR25/'
                     'q1_q17_dr25_tce_2020.09.28_10.36.22_stellar_koi_cfp_norobovetterlabels_renamedcols_nomissingval_'
                     'rmcandandfpkois_norogues.csv')

tceTbl['tce_maxmes'].describe()

bins = np.linspace(0, 20, 100, endpoint=True)

stats = {
    'median': np.median(tceTbl['tce_maxmes']),
    'mean': np.mean(tceTbl['tce_maxmes']),
    'std_rob': mad_std(tceTbl['tce_maxmes']),
    'std': np.std(tceTbl['tce_maxmes'], ddof=1),
    'min': np.min(tceTbl['tce_maxmes']),
    'max': np.max(tceTbl['tce_maxmes'])
}

# tceTbl['tce_max_mult_ev'].clip(0, 20 * stats['std_rob'], inplace=True)  # clip values to see what changes in the histogram
f, ax = plt.subplots()
tceTbl['tce_maxmes'].hist(bins=bins)
ax.set_ylabel('Counts')
ax.set_xlabel('MES')
ax.axvline(x=stats['median'], c='y', label='median')
ax.axvline(x=stats['median'] + stats['std_rob'], c='g', label='med+std_rob')
ax.axvline(x=stats['median'] - stats['std_rob'], c='b', label='med-std_rob')
ax.set_ylabel('Counts')
ax.set_xlabel('Weak Secondary Max MES')
ax.set_title('Mean: {:.2f} | Median: {:.2f} |\n Std: {:.2f} | Std Rob: {:.2f} | Min: {:.2f} |Max: {:.2f}'.format(
    stats['mean'],
    stats['median'],
    stats['std'],
    stats['std_rob'],
    stats['min'],
    stats['max']
)
             )
ax.legend()
# ax.set_xscale('log')
# f.savefig('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Analysis/scalar_params_analysis/wks_mes/'
#           'hist_tce_wksmaxmes_keplerq1q7dr25.svg')

labels = ['PC', 'AFP', 'NTP']
zorder = {'PC': 3, 'AFP': 2, 'NTP': 1}

f, ax = plt.subplots()
for label in labels:
    tceTbl.loc[tceTbl['label'] == label]['tce_maxmes'].hist(bins=bins, label=label, edgecolor='k', zorder=zorder[label])
ax.set_ylabel('Counts')
ax.set_xlabel('Weak Secondary Max MES')
ax.legend()
# ax.set_xscale('log')
# f.savefig('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Analysis/scalar_params_analysis/wks_mes/'
#           'hist_tce_wksmaxmes_keplerq1q7dr25_pc-afp-ntp.svg')

#%% Analyze difference imaging offset features

tceTbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17_DR25/'
                     'q1_q17_dr25_tce_2020.09.15_15.12.12_stellar_koi_cfp_norobovetterlabels_rmcandandfpkois_'
                     'norogues_renamedcols.csv')

feature = 'tce_dikco_msky'  # 'tce_dikco_msky', 'tce_dicco_msky'

tceTbl[feature].describe()

bins = np.linspace(0, 3, 100, endpoint=True)

stats = {
    'median': np.median(tceTbl[feature]),
    'mean': np.mean(tceTbl[feature]),
    'std_rob': mad_std(tceTbl[feature]),
    'std': np.std(tceTbl[feature], ddof=1),
    'min': np.min(tceTbl[feature]),
    'max': np.max(tceTbl[feature])
}

# tceTbl[feature].clip(0, 20 * stats['std_rob'], inplace=True)  # clip values to see what changes in the histogram

f, ax = plt.subplots()
tceTbl[feature].hist(bins=bins)
ax.axvline(x=stats['median'], c='y', label='median')
ax.axvline(x=stats['median'] + stats['std_rob'], c='g', label='med+std_rob')
ax.axvline(x=stats['median'] - stats['std_rob'], c='b', label='med-std_rob')
ax.set_ylabel('Counts')
ax.set_xlabel('{}'.format(feature))
ax.set_title('Mean: {:.2f} | Median: {:.2f} |\n Std: {:.2f} | Std Rob: {:.2f} | Min: {:.2f} '
             '|Max: {:.2f}'.format(stats['mean'],
                                   stats['median'],
                                   stats['std'],
                                   stats['std_rob'],
                                   stats['min'],
                                   stats['max']
                                   )
             )
ax.legend()
# ax.set_xscale('log')
f.savefig('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Analysis/scalar_params_analysis/diff_imaging_centroid_off/'
          'hist_tce_{}_keplerq1q7dr25.svg'.format(feature))

labels = ['PC', 'AFP', 'NTP']
zorder = {'PC': 3, 'AFP': 2, 'NTP': 1}

f, ax = plt.subplots()
for label in labels:
    tceTbl.loc[tceTbl['label'] == label][feature].hist(bins=bins, label=label, edgecolor='k', zorder=zorder[label])
ax.set_ylabel('Counts')
ax.set_xlabel('{}'.format(feature))
ax.set_ylabel('Counts')
ax.set_xlabel('{}'.format(feature))
ax.legend()
# ax.set_xscale('log')
f.savefig('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Analysis/scalar_params_analysis/diff_imaging_centroid_off/'
          'hist_tce_{}_keplerq1q7dr25_pc-afp-ntp.svg'.format(feature))

#%% Plot histograms of non-normalized scalar features from the TCE table

saveDir = '/home/msaragoc/Projects/Kepler-TESS_exoplanet/Analysis/scalar_params_analysis/' \
          'rba_cnt0n_3-26-2021'
os.makedirs(saveDir, exist_ok=True)

tceTbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17_DR25/q1_q17_dr25_tce_2020.09.28_10.36.22_stellar_koi_cfp_norobovetterlabels_renamedcols_nomissingval_symsecphase_confirmedkoiperiod_sec_rba_cnt0n.csv')

# remove rogue TCEs
tceTbl = tceTbl.loc[tceTbl['tce_rogue_flag'] != 1]
# remove Possible Planet KOIs
tceTbl = tceTbl.loc[~((tceTbl['fpwg_disp_status'] == 'POSSIBLE PLANET') & (tceTbl['koi_disposition'] != 'CONFIRMED'))]
# remove KOIs not used (not Confirmed, CFP, CFA KOIs)
tceTbl = tceTbl.loc[(tceTbl['fpwg_disp_status'].isin(['CERTIFIED FA', 'CERTIFIED FP'])) | (tceTbl['koi_disposition'] == 'CONFIRMED') | (tceTbl['koi_disposition'].isna())]

# tceTbl['tce_num_transits'] += 1
# tceTbl['tce_rb_tcount0_trnorm'] = tceTbl['tce_rb_tcount0'] / tceTbl['tce_num_transits']

features = [
    # 'tce_albedo',
    # 'tce_albedo_err',
    # 'tce_albedo_stat',
    # 'tce_ptemp',
    # 'tce_ptemp_err',
    # 'tce_ptemp_stat',
    # 'wst_depth',
    # 'tce_maxmes',
    # 'tce_eqt',
    # 'transit_depth',
    # 'tce_depth_err',
    # 'tce_duration',
    # 'tce_duration_err',
    # 'tce_prad',
    # 'tce_period',
    # 'tce_period_err',
    # 'tce_dikco_msky',
    # 'tce_dikco_msky_err',
    # 'tce_dicco_msky',
    # 'tce_dicco_msky_err',
    # 'tce_fwm_stat',
    # 'tce_fwm_srao',
    # 'tce_fwm_srao_err',
    # 'tce_fwm_sdeco',
    # 'tce_fwm_sdeco_err',
    # 'tce_fwm_prao',
    # 'tce_fwm_prao_err',
    # 'tce_fwm_pdeco',
    # 'tce_fwm_pdeco_err',
    # 'tce_max_mult_ev',
    # 'tce_rb_tcount0_trnorm',
    # 'tce_impact',
    # 'tce_prad',
    # 'mag',
    'tce_rb_tcount0n'
]

bins = {
    # 'tce_eqt': np.linspace(-1e4, 1e4, 50, endpoint=True),
    # 'tce_albedo': np.linspace(0, 100, 50, endpoint=True),
    # 'tce_albedo_err': np.linspace(-10, 100, 50, endpoint=True),
    # 'tce_albedo_stat': np.linspace(-4, 4, 50, endpoint=True),
    # 'tce_ptemp': np.linspace(0, 1e4, 50, endpoint=True),
    # 'tce_ptemp_err': np.linspace(-10, 1e3, 50, endpoint=True),
    # 'tce_ptemp_stat': np.linspace(-10, 10, 50, endpoint=True),
    # 'tce_maxmes': np.linspace(0, 100, 50, endpoint=True),
    # 'wst_depth': np.linspace(-1e4, 1e4, 100, endpoint=True),
    # 'transit_depth': np.linspace(0, 1e4, 50, endpoint=True),
    # 'tce_depth_err': np.linspace(-1, 100, 50, endpoint=True),
    # 'tce_duration': np.linspace(0, 100, 50, endpoint=True),
    # 'tce_duration_err': np.linspace(-1, 20, 50, endpoint=True),
    # 'tce_prad': np.linspace(0, 20, 100, endpoint=True),
    # 'tce_period': np.linspace(0, 800, 100, endpoint=True),
    # 'tce_period_err': np.linspace(0, 0.1, 100, endpoint=True),
    # 'tce_dikco_msky': np.linspace(0, 20, 50, endpoint=True),
    # 'tce_dikco_msky_err': np.linspace(-1, 10, 50, endpoint=True),
    # 'tce_dicco_msky': np.linspace(0, 20, 50, endpoint=True),
    # 'tce_dicco_msky_err': np.linspace(-1, 10, 50, endpoint=True),
    # 'tce_fwm_stat': np.linspace(0, 1000, 50, endpoint=True),
    # 'tce_fwm_srao': np.linspace(-100, 100, 50, endpoint=True),
    # 'tce_fwm_srao_err': np.linspace(-100, 100, 50, endpoint=True),
    # 'tce_fwm_sdeco': np.linspace(-100, 100, 50, endpoint=True),
    # 'tce_fwm_sdeco_err': np.linspace(-10, 100, 50, endpoint=True),
    # 'tce_fwm_prao': np.linspace(-1e-1, 1e-1, 50, endpoint=True),
    # 'tce_fwm_prao_err': np.linspace(0, 1e-1, 50, endpoint=True),
    # 'tce_fwm_pdeco': np.linspace(-1e-1, 1e-1, 50, endpoint=True),
    # 'tce_fwm_pdeco_err': np.linspace(0, 1e-1, 50, endpoint=True),
    # 'tce_max_mult_ev': np.linspace(7.1, 1000, 50, endpoint=True),
    # 'tce_impact': np.linspace(0, 1, 100, endpoint=True),
    # 'tce_rb_tcount0_trnorm': np.linspace(0, 1, 100, endpoint=True),
    # 'mag': np.linspace(0, 20, 100, endpoint=True),
    'tce_rb_tcount0n': np.linspace(0, 1, 100, endpoint=True)
}

log_yscale = [
    'tce_fwm_stat',
    'tce_fwm_sdeco',
    'tce_fwm_sdeco_err',
    'tce_fwm_prao',
    'tce_fwm_prao_err',
    'tce_fwm_pdeco',
    'tce_fwm_pdeco_err',
    'tce_period_err',
    'tce_max_mult_ev',
    'tce_dikco_msky',
    'tce_dikco_msky_err',
    'tce_dicco_msky',
    'tce_dicco_msky_err',
    'transit_depth',
    'tce_duration',
    'tce_period',
    'wst_depth',
    'tce_rb_tcount0n'
]

for feature in features:

    print('Feature {}: '.format(feature), tceTbl[feature].describe())

    # convert from second to arcsecond
    if feature in ['tce_fwm_srao', 'tce_fwm_srao_err', 'tce_fwm_prao', 'tce_fwm_prao_err']:
        tceTbl[feature] *= 15

    stats = {
        'median': np.nanmedian(tceTbl[feature]),
        'mean': np.nanmean(tceTbl[feature]),
        'std_rob': mad_std(tceTbl.loc[~tceTbl[feature].isna(), feature]),
        'std': np.nanstd(tceTbl[feature], ddof=1),
        'min': np.min(tceTbl[feature]),
        'max': np.max(tceTbl[feature])
    }

    # tceTbl[feature].clip(0, 20 * stats['std_rob'], inplace=True)  # clip values to see what changes in the histogram

    f, ax = plt.subplots()
    tceTbl[feature].hist(bins=bins[feature], edgecolor='k')
    ax.axvline(x=stats['median'], c='y', label='median')
    ax.axvline(x=stats['median'] + stats['std_rob'], c='g', label='med+std_rob')
    ax.axvline(x=stats['median'] - stats['std_rob'], c='b', label='med-std_rob')
    ax.set_ylabel('Counts')
    ax.set_xlabel('{}'.format(feature))
    ax.set_title('Mean: {:.2f} | Median: {:.2f} |\n Std: {:.2f} | Std Rob: {:.2f} | Min: {:.2f} '
                 '|Max: {:.2f}'.format(stats['mean'],
                                       stats['median'],
                                       stats['std'],
                                       stats['std_rob'],
                                       stats['min'],
                                       stats['max']
                                       )
                 )
    ax.set_xlim([bins[feature][0], bins[feature][-1]])
    ax.legend()
    if feature in log_yscale:
        ax.set_yscale('log')
    # ax.set_xscale('log')
    f.savefig(os.path.join(saveDir, 'hist_{}_keplerq1q7dr25.svg'.format(feature)))
    # plt.close()

    labels = ['PC', 'AFP', 'NTP']
    zorder = {'PC': 3, 'AFP': 2, 'NTP': 1}

    f, ax = plt.subplots()
    for label in labels:
        tceTbl.loc[tceTbl['label'] == label][feature].hist(bins=bins[feature], label=label, edgecolor='k',
                                                           zorder=zorder[label])
    ax.set_ylabel('Counts')
    ax.set_xlabel('{}'.format(feature))
    ax.set_ylabel('Counts')
    ax.set_xlabel('{}'.format(feature))
    ax.set_xlim([bins[feature][0], bins[feature][-1]])
    ax.legend()
    if feature in log_yscale:
        ax.set_yscale('log')
    # ax.set_xscale('log')
    f.savefig(os.path.join(saveDir, 'hist_{}_keplerq1q7dr25_pc-afp-ntp.svg'.format(feature)))
    # plt.close()
