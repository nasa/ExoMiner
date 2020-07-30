import numpy as np
import os
import matplotlib.pyplot as plt

#%%

trainRootDir = '/home/msaragoc/Projects/Kepler-TESS_exoplanet/Kepler_planet_finder/trained_models/'

results = {}

expTrain = ['keplerdr25_g2001-l201_spline_nongapped_norobovetterkois_starshuffle_configD_glflux-glcentr_prelu',
            'keplerdr25_g2001-l201_spline_nongapped_norobovetterkois_starshuffle_configD_glflux-glcentrmedn_prelu',
            'keplerdr25_g2001-l201_spline_nongapped_norobovetterkois_starshuffle_configD_glflux-glcentrmedcmaxn_prelu',
            'keplerdr25_g2001-l201_spline_nongapped_norobovetterkois_starshuffle_configD_glflux-glcentrmedcmaxn_dir_prelu'
            ]

for experiment in expTrain:

    results[experiment] = {}

    modelsRootDir = os.path.join(trainRootDir, experiment, 'models')

    modelsDir = [os.path.join(modelsRootDir, modelDir) for modelDir in os.listdir(modelsRootDir) if 'model' in modelDir]

    # results = {}
    for modelDir in modelsDir:
        results[experiment][modelDir.split('/')[-1]] = np.load(os.path.join(modelDir, 'results.npy'), allow_pickle=True).item()

#%% Plot results

mapExpName = {
    'keplerdr25_g2001-l201_spline_nongapped_norobovetterkois_starshuffle_configD_glflux-glcentr_prelu': 'trstd-clip',
    'keplerdr25_g2001-l201_spline_nongapped_norobovetterkois_starshuffle_configD_glflux-glcentrmedn_prelu': 'medn',
    'keplerdr25_g2001-l201_spline_nongapped_norobovetterkois_starshuffle_configD_glflux-glcentrmedcmaxn_prelu': 'medcmaxn',
    'keplerdr25_g2001-l201_spline_nongapped_norobovetterkois_starshuffle_configD_glflux-glcentrmedcmaxn_dir_prelu': 'medcmaxabsn'
}

numModels = 10  # len(results)
metric = 'val_auc_pr'

# valMetric = np.array([results[model_i][metric][-1] for model_i in results]) * 100
# meanMetric = np.mean(valMetric)
# stdMetric = np.std(valMetric, ddof=1)

dataExperiments = {experiment: np.array([results[experiment][model_i][metric][-1]
                                         for model_i in results[experiment]]) * 100
                   for experiment in results}

f, ax = plt.subplots()
for experiment in dataExperiments:
    # valMetric = np.array([results[experiment][model_i][metric][-1] for model_i in results[experiment]]) * 100
    ax.plot(np.arange(1, numModels + 1), dataExperiments[experiment], label=mapExpName[experiment])
    # ax.scatter(np.arange(1, numModels + 1), valMetric)
# ax.set_title('{}\nMean +- std = {:.2f} +- {:.2f} %'.format(metric, meanMetric, stdMetric))
ax.set_xlabel('Model id')
ax.set_ylabel('{}'.format(metric))
ax.set_xticks(np.arange(1, numModels + 1))
ax.legend()

#%%
import pandas as pd

testsetTbl = pd.read_csv('/data5/tess_project/Data/tfrecords/Kepler/Q1-Q17_DR25/tfrecordskeplerdr25_g2001-l201_spline_gapped_flux-centroid_selfnormalized-oddeven-wks-scalar_data/tfrecordskeplerdr25_g2001-l201_spline_gapped_flux-centroid_selfnormalized-oddeven-wks-scalar_starshuffle_experiment/trainset.csv')
tceTbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17_DR25/q1_q17_dr25_tce_2020.04.15_23.19.10_cumkoi_2020.02.21_shuffledstar_noroguetces_noRobovetterKOIs.csv')

tceTbl = tceTbl.loc[tceTbl['label'] == 'PC']

tceCount = 0
for tce_i, tce in tceTbl.iterrows():

    tceFound = testsetTbl.loc[(testsetTbl['target_id'] == tce['target_id']) &
                              (testsetTbl['tce_plnt_num'] == tce['tce_plnt_num'])]

    if len(tceFound) == 1:
        tceCount += 1

# test set 283 PC; 257 val set; 2084
print(tceCount)
