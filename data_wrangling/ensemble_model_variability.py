""" Study model variability for ensemble. """

import numpy as np
import os
import matplotlib.pyplot as plt

#%%

trainRootDir = '/home/msaragoc/Projects/Kepler-TESS_exoplanet/Kepler_planet_finder/trained_models/'

results = {}

expTrain = [
    # 'keplerdr25_g2001-l201_spline_nongapped_norobovetterkois_starshuffle_configD_glflux-glcentr_prelu',
    # 'keplerdr25_g2001-l201_spline_nongapped_norobovetterkois_starshuffle_configD_glflux-glcentrmedn_prelu',
    # 'keplerdr25_g2001-l201_spline_nongapped_norobovetterkois_starshuffle_configD_glflux-glcentrmedcmaxn_prelu',
    # 'keplerdr25_g2001-l201_spline_nongapped_norobovetterkois_starshuffle_configD_glflux-glcentrmedcmaxn_dir_prelu'
    # 'keplerdr25_g2001-l201_spline_nongapped_globalbinwidthaslocal_norobovetterkois_starshuffle_configC_glflux-glcentrmedcmaxn-lwks-loe-6stellar-bfap-ghost',
    # 'keplerdr25_g2001-l201_spline_gbal_nongapped_norobovetterkois_starshuffle_configE_glflux-glcentr-loe-lwks-6stellar-bfap-ghost-rollingband',
    # 'keplerdr25_g2001-l201_spline_gbal_nongapped_norobovetterkois_starshuffle_configC_glflux-glcentr-loe-lwks-6stellar-bfap-ghost-rollingband'
    # 'keplerdr25_g2001-l201_spline_gbal_nongapped_norobovetterkois_starshuffle_configD_glflux_prelu',
    # 'keplerdr25_g2001-l201_spline_gbal_nongapped_norobovetterkois_starshuffle_configD_glflux-rollingband_prelu',
    # 'keplerdr25_g2001-l201_spline_gbal_nongapped_norobovetterkois_starshuffle_configD_glflux-6stellar_prelu',
    # 'keplerdr25_g2001-l201_spline_gbal_nongapped_norobovetterkois_starshuffle_configD_glflux-bfap_prelu',
    # 'keplerdr25_g2001-l201_spline_gbal_nongapped_norobovetterkois_starshuffle_configD_glflux-ghost_prelu',
    # 'keplerdr25_g2001-l201_spline_gbal_nongapped_norobovetterkois_starshuffle_configD_glflux-loe_prelu',
    # 'keplerdr25_g2001-l201_spline_gbal_nongapped_norobovetterkois_starshuffle_configD_glflux-lwks_prelu',
    # 'keplerdr25_g2001-l201_spline_gbal_nongapped_norobovetterkois_starshuffle_configD_glflux-glcentr_prelu'
    # 'keplerdr25_g2001-l201_spline_nongapped_norobovetterkois_starshuffle_configD_glflux-lwks_prelu',
    'keplerdr25_g2001-l201_spline_nongapped_norobovetterkois_starshuffle_configD_glflux_prelu',
    # 'keplerdr25_g2001-l201_spline_nongapped_norobovetterkois_starshuffle_configD_glflux-loe_prelu',
    # 'keplerdr25_g2001-l201_spline_nongapped_norobovetterkois_starshuffle_configD_glflux-lwks_selfnorm-wksmaxmes_prelu',
    # 'keplerdr25_g2001-l201_spline_nongapped_norobovetterkois_starshuffle_configD_glflux-lwks_fluxnorm-wksmaxmes_prelu',
    # 'keplerdr25_g2001-l201_spline_nongapped_norobovetterkois_starshuffle_configD_glflux-lwks_prelu',
    # 'keplerdr25_g2001-l201_spline_nongapped_norobovetterkois_starshuffle_configD_glflux-lwks_selfnorm_prelu',
    # 'keplerdr25_g2001-l201_spline_nongapped_norobovetterkois_starshuffle_configD_glflux-co_kic_oot_prelu',
    # 'keplerdr25_g2001-l201_spline_nongapped_norobovetterkois_starshuffle_configD_glflux-glcentr_std_noclip_prelu',
    # 'keplerdr25_g2001-l201_spline_nongapped_norobovetterkois_starshuffle_configD_glflux-glcentr_std_noclip-co_kic_oot_prelu',
    'keplerdr25_g2001-l201_spline_nongapped_norobovetterkois_starshuffle_configE_glflux-glcentr_std_noclip-lwks_fluxnorm-loe-6stellar-bfap-ghost-rollingband-co_kic_oot-wksmaxmes_prelu',
    'keplerdr25_g2001-l201_spline_nongapped_norobovetterkois_starshuffle_configE_glflux-glcentr_std_noclip-loe-lwks-6stellar-bfap-ghost-rollingband_prelu',
    'keplerdr25_g2001-l201_spline_nongapped_norobovetterkois_starshuffle_configE_glflux-glcentr_std_noclip-loe-lwks-6stellar_prelu',
    'keplerdr25_g2001-l201_spline_nongapped_norobovetterkois_starshuffle_astronet-300epochs-es20patience_glflux',
    'keplerdr25_g2001-l201_spline_nongapped_norobovetterkois_starshuffle_exonet-300epochs-es20patience_glflux-glcentr_fdl-6stellar'
]

for experiment in expTrain:

    results[experiment] = {}

    modelsRootDir = os.path.join(trainRootDir, experiment, 'models')

    modelsDir = [os.path.join(modelsRootDir, modelDir) for modelDir in os.listdir(modelsRootDir) if 'model' in modelDir]

    # results = {}
    for modelDir in modelsDir:
        results[experiment][modelDir.split('/')[-1]] = np.load(os.path.join(modelDir, 'results.npy'),
                                                               allow_pickle=True).item()

#%% Plot results

mapExpName = {
    # 'keplerdr25_g2001-l201_spline_nongapped_norobovetterkois_starshuffle_configD_glflux-glcentr_prelu': 'trstd-clip',
    # 'keplerdr25_g2001-l201_spline_nongapped_norobovetterkois_starshuffle_configD_glflux-glcentrmedn_prelu': 'medn',
    # 'keplerdr25_g2001-l201_spline_nongapped_norobovetterkois_starshuffle_configD_glflux-glcentrmedcmaxn_prelu': 'medcmaxn',
    # 'keplerdr25_g2001-l201_spline_nongapped_norobovetterkois_starshuffle_configD_glflux-glcentrmedcmaxn_dir_prelu': 'medcmaxabsn'
    # 'keplerdr25_g2001-l201_spline_nongapped_globalbinwidthaslocal_norobovetterkois_starshuffle_configC_glflux-glcentrmedcmaxn-lwks-loe-6stellar-bfap-ghost': 'Config C',
    # 'keplerdr25_g2001-l201_spline_gbal_nongapped_norobovetterkois_starshuffle_configE_glflux-glcentr-loe-lwks-6stellar-bfap-ghost-rollingband': 'Config E',
    # 'keplerdr25_g2001-l201_spline_gbal_nongapped_norobovetterkois_starshuffle_configC_glflux-glcentr-loe-lwks-6stellar-bfap-ghost-rollingband': 'Config C new'
    # 'keplerdr25_g2001-l201_spline_gbal_nongapped_norobovetterkois_starshuffle_configD_glflux_prelu': 'Baseline\n(gl Flux)',
    # 'keplerdr25_g2001-l201_spline_gbal_nongapped_norobovetterkois_starshuffle_configD_glflux-rollingband_prelu': 'Rolling\nBand',
    # 'keplerdr25_g2001-l201_spline_gbal_nongapped_norobovetterkois_starshuffle_configD_glflux-6stellar_prelu': '6 Stellar\nParameters',
    # 'keplerdr25_g2001-l201_spline_gbal_nongapped_norobovetterkois_starshuffle_configD_glflux-bfap_prelu': 'Bootstrap \nFA Prob',
    # 'keplerdr25_g2001-l201_spline_gbal_nongapped_norobovetterkois_starshuffle_configD_glflux-ghost_prelu': 'Ghost\ndiagnostic',
    # 'keplerdr25_g2001-l201_spline_gbal_nongapped_norobovetterkois_starshuffle_configD_glflux-loe_prelu': 'l Odd-even',
    # 'keplerdr25_g2001-l201_spline_gbal_nongapped_norobovetterkois_starshuffle_configD_glflux-lwks_prelu': 'l Weak\nSecondary',
    # 'keplerdr25_g2001-l201_spline_gbal_nongapped_norobovetterkois_starshuffle_configD_glflux-glcentr_prelu': 'gl Centroid',
    # 'keplerdr25_g2001-l201_spline_nongapped_norobovetterkois_starshuffle_configD_glflux-lwks_prelu': 'l Weak\nSecondary',
    # 'keplerdr25_g2001-l201_spline_nongapped_norobovetterkois_starshuffle_configD_glflux_prelu': 'Baseline',
    # 'keplerdr25_g2001-l201_spline_nongapped_norobovetterkois_starshuffle_configD_glflux-loe_prelu': 'l Odd-even',
    # 'keplerdr25_g2001-l201_spline_nongapped_norobovetterkois_starshuffle_configD_glflux-lwks_selfnorm-wksmaxmes_prelu': 'l Weak\nSecondary\nSelfnorm\n+MaxMes',
    # 'keplerdr25_g2001-l201_spline_nongapped_norobovetterkois_starshuffle_configD_glflux-lwks_fluxnorm-wksmaxmes_prelu': 'l Weak\nSecondary\nFluxnorm\n+MaxMes',
    # 'keplerdr25_g2001-l201_spline_nongapped_norobovetterkois_starshuffle_configD_glflux-lwks_prelu': 'l Weak\nSecondary\nFluxnorm',
    # 'keplerdr25_g2001-l201_spline_nongapped_norobovetterkois_starshuffle_configD_glflux-lwks_selfnorm_prelu': 'l Weak\nSecondary\nSelfnorm',
    'keplerdr25_g2001-l201_spline_nongapped_norobovetterkois_starshuffle_configD_glflux_prelu': 'Baseline',
    # 'keplerdr25_g2001-l201_spline_nongapped_norobovetterkois_starshuffle_configD_glflux-co_kic_oot_prelu': 'CO-KIC_OOT',
    # 'keplerdr25_g2001-l201_spline_nongapped_norobovetterkois_starshuffle_configD_glflux-glcentr_std_noclip_prelu': 'gl Centroid\nStd-noclip',
    # 'keplerdr25_g2001-l201_spline_nongapped_norobovetterkois_starshuffle_configD_glflux-glcentr_std_noclip-co_kic_oot_prelu': 'gl Centroid\nStd-noclip\n+CO-KIC_OOT',
    'keplerdr25_g2001-l201_spline_nongapped_norobovetterkois_starshuffle_configE_glflux-glcentr_std_noclip-lwks_fluxnorm-loe-6stellar-bfap-ghost-rollingband-co_kic_oot-wksmaxmes_prelu': 'Exominer_new',
    'keplerdr25_g2001-l201_spline_nongapped_norobovetterkois_starshuffle_configE_glflux-glcentr_std_noclip-loe-lwks-6stellar-bfap-ghost-rollingband_prelu': 'Exominer',
    'keplerdr25_g2001-l201_spline_nongapped_norobovetterkois_starshuffle_configE_glflux-glcentr_std_noclip-loe-lwks-6stellar_prelu': 'Exominer-TPS',
    'keplerdr25_g2001-l201_spline_nongapped_norobovetterkois_starshuffle_astronet-300epochs-es20patience_glflux': 'Astronet',
    'keplerdr25_g2001-l201_spline_nongapped_norobovetterkois_starshuffle_exonet-300epochs-es20patience_glflux-glcentr_fdl-6stellar': 'Exonet'
}

numModels = 10  # len(results)
metric = 'test_precision'

# valMetric = np.array([results[model_i][metric][-1] for model_i in results]) * 100
# meanMetric = np.mean(valMetric)
# stdMetric = np.std(valMetric, ddof=1)

if 'test' in metric or '_at_' in metric:
    dataExperiments = {experiment: np.array([results[experiment][model_i][metric]
                                             for model_i in results[experiment]]) * 100
                       for experiment in mapExpName}
else:
    dataExperiments = {experiment: np.array([results[experiment][model_i][metric][-1]
                                             for model_i in results[experiment]]) * 100
                       for experiment in mapExpName}

f, ax = plt.subplots(figsize=(18, 12))
for experiment in dataExperiments:
    # valMetric = np.array([results[experiment][model_i][metric][-1] for model_i in results[experiment]]) * 100
    ax.plot(np.arange(1, numModels + 1), dataExperiments[experiment], label=mapExpName[experiment])
    # ax.scatter(np.arange(1, numModels + 1), valMetric)
# ax.set_title('{}\nMean +- std = {:.2f} +- {:.2f} %'.format(metric, meanMetric, stdMetric))
ax.set_xlabel('Model id')
ax.set_ylabel('{}'.format(metric))
ax.set_xticks(np.arange(1, numModels + 1))
ax.legend()
f.savefig('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Analysis/model_variability/modelsplot_{}.png'.format(metric))
plt.close()

f, ax = plt.subplots()
ax.boxplot(list(dataExperiments.values()))
ax.set_xticklabels(mapExpName.values(), rotation=45)
ax.set_ylabel('{}'.format(metric))
ax.set_xlabel('Experiment')
plt.subplots_adjust(top=0.975, bottom=0.220, left=0.11, right=0.9, hspace=0.2, wspace=0.2)
f.savefig('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Analysis/model_variability/boxplot_{}.png'.format(metric))
plt.close()
