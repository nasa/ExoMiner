from sklearn.calibration import calibration_curve
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

#%% Plot reliability curve for each dataset for a given experiment

dataset = 'val'
experimentRootDir = '/home/msaragoc/Projects/Kepler-TESS_exoplanet/Kepler_planet_finder/results_ensemble/'
# experiment = 'keplerdr25_g2001-l201_spline_nongapped_norobovetterkois_starshuffle_configD_glflux_prelu'
experiment = 'keplerdr25_g2001-l201_spline_nongapped_norobovetterkois_starshuffle_configD_glflux-glcentr_std_noclip-loe-lwks-6stellar-bfap-ghost-rollingband_prelu'
rankingTbl = pd.read_csv(os.path.join(experimentRootDir,
                                      experiment,
                                      'ensemble_ranked_predictions_{}set.csv'.format(dataset)))

fractionPositives, meanPredictedVal = calibration_curve(rankingTbl['label'], rankingTbl['score'], n_bins=10,
                                                        normalize=False, strategy='uniform')

f, ax = plt.subplots(2, 1)
ax[0].plot([0, 1], [0, 1], 'k:')
ax[0].plot(meanPredictedVal, fractionPositives)
ax[0].scatter(meanPredictedVal, fractionPositives, c='r')
ax[0].set_xticks(np.linspace(0, 1, 11, endpoint=True))
ax[0].set_yticks(np.linspace(0, 1, 11, endpoint=True))
ax[0].set_xlim([0, 1])
ax[0].set_ylim([0, 1])
ax[0].grid(True)
ax[0].set_ylabel('Fraction of Positives')
ax[1].hist(rankingTbl.loc[rankingTbl['label'] == 1]['score'], bins=10, range=(0, 1), histtype='step', lw=2)
ax[1].set_xticks(np.linspace(0, 1, 11, endpoint=True))
ax[1].set_xlim([0, 1])
# ax[1].set_yscale('log')
ax[1].set_xlabel('Mean Predicted Value')
ax[1].set_ylabel('Counts')
f.savefig(os.path.join(experimentRootDir, experiment, 'reliability_curve_{}.svg'.format(dataset)))

#%% Plot reliability curve for all datasets for a given experiment

datasets = ['train', 'val', 'test']
datasetsLabels = {'train': 'Train', 'val': 'Validation', 'test': 'Test'}
datasetsColors = {'train': 'b', 'val': 'r', 'test': 'orange'}

experimentRootDir = '/home/msaragoc/Projects/Kepler-TESS_exoplanet/Kepler_planet_finder/results_ensemble/'
# experiment = 'keplerdr25_g2001-l201_spline_nongapped_norobovetterkois_starshuffle_configD_glflux_prelu/'
experiment = 'keplerdr25_g2001-l201_spline_nongapped_norobovetterkois_starshuffle_configD_glflux-glcentr_std_noclip-loe-lwks-6stellar-bfap-ghost-rollingband_prelu'

reliabilityCurve = {dataset: {'mean predicted value': 0.0, 'fraction of positives': 0.0, 'positives scores': []}
                    for dataset in datasets}
for dataset in datasets:
    rankingTbl = pd.read_csv(os.path.join(experimentRootDir,
                                          experiment,
                                          'ensemble_ranked_predictions_{}set.csv'.format(dataset)))

    reliabilityCurve[dataset]['fraction of positives'], \
    reliabilityCurve[dataset]['mean predicted value'] = calibration_curve(rankingTbl['label'],
                                                                          rankingTbl['score'],
                                                                          n_bins=10,
                                                                          normalize=False,
                                                                          strategy='uniform')
    reliabilityCurve[dataset]['positives scores'] = rankingTbl.loc[rankingTbl['label'] == 1]['score']

f, ax = plt.subplots(2, 1, figsize=(10, 10))
ax[0].plot([0, 1], [0, 1], 'k:')
for dataset in datasets:
    ax[0].plot(reliabilityCurve[dataset]['mean predicted value'], reliabilityCurve[dataset]['fraction of positives'],
            datasetsColors[dataset], label=datasetsLabels[dataset])
    ax[0].scatter(reliabilityCurve[dataset]['mean predicted value'], reliabilityCurve[dataset]['fraction of positives'],
               c=datasetsColors[dataset])
    ax[1].hist(reliabilityCurve[dataset]['positives scores'], bins=10, range=(0, 1), histtype='step', lw=2,
               label=datasetsLabels[dataset], color=datasetsColors[dataset])
ax[0].set_xticks(np.linspace(0, 1, 11, endpoint=True))
ax[0].set_yticks(np.linspace(0, 1, 11, endpoint=True))
ax[0].set_xlim([0, 1])
ax[0].set_ylim([0, 1])
ax[0].grid(True)
ax[0].set_ylabel('Fraction of Positives')
ax[0].legend()
ax[1].set_xticks(np.linspace(0, 1, 11, endpoint=True))
ax[1].set_xlim([0, 1])
ax[1].set_yscale('log')
ax[1].set_xlabel('Mean Predicted Value')
ax[1].set_ylabel('Counts')
ax[1].legend(loc='upper left')
f.savefig(os.path.join(experimentRootDir, experiment, 'reliability_curve_alldatasets.svg'))
