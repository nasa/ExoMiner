import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

#%%


def compute_precision_at_k(study, dataset, rootDir, k_arr, k_curve_arr, k_curve_arr_plot, plot=False, verbose=False):
    """ Compute precision-at-k and plot related curves.

    :param study: str, study name
    :param dataset: str, dataset. Either 'train', 'val' and 'test'
    :param rootDir: str, root directory with the studies
    :param k_arr: list with k values for which to compute precision-at-k
    :param k_curve_arr: list with k values for which to compute precision-at-k curve
    :param k_curve_arr_plot: list with values for which to draw xticks (k values)
    :param plot: bool, if True plots precision-at-k and misclassified-at-k curves
    :param verbose: bool, if True print precision-at-k values
    :return:
    """

    rankingTbl = pd.read_csv(os.path.join(rootDir, study, 'ensemble_ranked_predictions_{}set.csv'.format(dataset)))

    # order by ascending score
    rankingTblOrd = rankingTbl.sort_values('score', axis=0, ascending=True)

    # compute precision at k
    precision_at_k = {k: np.nan for k in k_arr}
    for k_i in range(len(k_arr)):
        if len(rankingTblOrd) < k_arr[k_i]:
            precision_at_k[k_arr[k_i]] = np.nan
        else:
            precision_at_k[k_arr[k_i]] = \
                np.sum(rankingTblOrd['label'][-k_arr[k_i]:]) / k_arr[k_i]

    np.save(os.path.join(rootDir, study, 'precision_at_k_{}.npy'.format(dataset)), precision_at_k)

    if verbose:
        print('{}: {}'.format(dataset, precision_at_k))

    # compute precision at k curve
    precision_at_k = {k: np.nan for k in k_curve_arr}
    for k_i in range(len(k_curve_arr)):
        if len(rankingTblOrd) < k_curve_arr[k_i]:
            precision_at_k[k_curve_arr[k_i]] = np.nan
        else:
            precision_at_k[k_curve_arr[k_i]] = \
                np.sum(rankingTblOrd['label'][-k_curve_arr[k_i]:]) / k_curve_arr[k_i]

    np.save(os.path.join(rootDir, study, 'precision_at_k_curve_{}.npy'.format(dataset)), precision_at_k)

    if plot:
        # precision at k curve
        f, ax = plt.subplots()
        ax.plot(list(precision_at_k.keys()), list(precision_at_k.values()))
        ax.set_ylabel('Precision')
        ax.set_xlabel('Top-K')
        ax.grid(True)
        ax.set_xticks(k_curve_arr_plot)
        ax.set_xlim([k_curve_arr[0], k_curve_arr[-1]])
        # ax.set_ylim(top=1)
        ax.set_ylim([-0.01, 1.01])
        f.savefig(os.path.join(rootDir, study, 'precision_at_k_{}.svg'.format(dataset)))
        plt.close()

        # misclassified examples at k curve
        f, ax = plt.subplots()
        kvalues = np.array(list(precision_at_k.keys()))
        precvalues = np.array(list(precision_at_k.values()))
        num_misclf_examples = kvalues - kvalues * precvalues
        ax.plot(kvalues, num_misclf_examples)
        ax.set_ylabel('Number Misclassfied TCEs')
        ax.set_xlabel('Top-K')
        ax.grid(True)
        ax.set_xticks(k_curve_arr_plot)
        ax.set_xlim([k_curve_arr[0], k_curve_arr[-1]])
        ax.set_ylim(bottom=-0.01)
        f.savefig(os.path.join(rootDir, study, 'misclassified_at_k_{}.svg'.format(dataset)))
        plt.close()

#%% Compute precision at k for experiments

studies = [
    # 'keplerdr25_tps-tce1_g2001-l201_spline_gbal_nongapped_norobovetterkois_starshuffle_configE_glflux-glcentr-loe-lwks-6stellar_prelu',
    # 'keplerdr25_tce1_g2001-l201_spline_gbal_nongapped_norobovetterkois_starshuffle_configE_glflux-glcentr-loe-lwks-6stellar_prelu',
    # 'keplerdr25_g2001-l201_spline_gbal_nongapped_norobovetterkois_starshuffle_configD_glflux-glcentrFDL_prelu',
    # 'keplerdr25_g2001-l201_spline_gbal_nongapped_norobovetterkois_starshuffle_configD_glflux-lwks_prelu',
    # 'keplerdr25_g2001-l201_spline_gbal_nongapped_norobovetterkois_starshuffle_configE_glflux-glcentr-loe-lwks_prelu',
    # 'keplerdr25_g2001-l201_spline_gbal_nongapped_norobovetterkois_starshuffle_configE_glflux-6stellar-bfap-ghost-rollingband_prelu',
    # 'keplerdr25_g2001-l201_spline_gbal_nongapped_norobovetterkois_starshuffle_configE_glflux-glcentr-loe-6stellar_prelu',
    # 'keplerdr25_g2001-l201_spline_gbal_nongapped_norobovetterkois_starshuffle_configE_glflux-glcentr-loe-lwks-6stellar_prelu',
    # 'keplerdr25_g2001-l201_spline_gbal_nongapped_norobovetterkois_starshuffle_configE_glflux-glcentr-loe-lwks-6stellar-bfap-ghost-rollingband_prelu_augon',
    # 'keplerdr25_g2001-l201_spline_gbal_nongapped_norobovetterkois_starshuffle_configD_glflux_prelu',
    # 'keplerdr25_g2001-l201_spline_gbal_nongapped_norobovetterkois_starshuffle_configD_glflux-rollingband_prelu',
    # 'keplerdr25_g2001-l201_spline_gbal_nongapped_norobovetterkois_starshuffle_configD_glflux-loe_prelu',
    # 'keplerdr25_g2001-l201_spline_gbal_nongapped_norobovetterkois_starshuffle_configD_glflux-ghost_prelu',
    # 'keplerdr25_g2001-l201_spline_gbal_nongapped_norobovetterkois_starshuffle_configD_glflux-bfap_prelu',
    # 'keplerdr25_g2001-l201_spline_gbal_nongapped_norobovetterkois_starshuffle_configD_glflux-6stellar_prelu',
    # 'keplerdr25_g2001-l201_spline_gbal_nongapped_norobovetterkois_starshuffle_configE_glflux-glcentrmedcmaxn-loe-lwks-6stellar-bfap-ghost-rollingband_prelu',
    # 'keplerdr25_g2001-l201_spline_gbal_nongapped_norobovetterkois_starshuffle_configD_glflux-glcentrmedn_prelu',
    # 'keplerdr25_g2001-l201_spline_gbal_nongapped_norobovetterkois_starshuffle_configD_glflux-glcentrmedcmaxn_prelu',
    # 'keplerdr25_g2001-l201_spline_gbal_nongapped_norobovetterkois_starshuffle_configD_glflux-glcentrmedcmaxn_dir_prelu',
    # 'keplerdr25_g2001-l201_spline_gbal_nongapped_norobovetterkois_starshuffle_configD_glflux-glcentr_prelu',
    # 'keplerdr25_g2001-l201_spline_gbal_nongapped_norobovetterkois_starshuffle_configE_glflux-glcentr-loe-lwks-6stellar-bfap-ghost-rollingband',
    # 'keplerdr25_g2001-l201_spline_gbal_nongapped_norobovetterkois_starshuffle_configC_glflux-glcentr-loe-lwks-6stellar-bfap-ghost-rollingband',
    # 'keplerdr25_g2001-l201_spline_gapped_glflux_norobovetterkois_shallue_starshuffle',
    # 'keplerdr25_g2001-l201_spline_gapped_glflux_norobovetterkois_fdl_starshuffle'
    # 'keplerdr25_g2001-l201_spline_gbal_nongapped_norobovetterkois_starshuffle_configE_glflux-glcentr-loe-lwks-6stellar-bfap-ghost-rollingband_prelu'
    # 'keplerdr25_g2001-l201_spline_gbal_nongapped_norobovetterkois_starshuffle_configE_glflux-glcentr-loe-lwks-6stellar-ghost-bfap-rollingband_prelu_nobugdur'
    # 'keplerdr25_g2001-l201_spline_nongapped_norobovetterkois_starshuffle_configD_glflux-glcentr_std_noclip-loe-lwks-6stellar-bfap-ghost-rollingband_prelu',
    # 'keplerdr25_g2001-l201_spline_nongapped_norobovetterkoisnopps_starshuffle_configD_glflux-glcentr_std_noclip-loe-lwks-6stellar-bfap-ghost-rollingband_prelu'
    # 'keplerdr25_g2001-l201_spline_nongapped_norobovetterkoisnopps_starshuffle_configD_glflux_prelu',
    # 'keplerdr25_g2001-l201_spline_nongapped_norobovetterkoisnopps_starshuffle_configD_glflux-glcentr_std_noclip_prelu',
    # 'keplerdr25_g2001-l201_spline_nongapped_norobovetterkoisnopps_starshuffle_configD_glflux-glcentr_std_noclip-loe-lwks-6stellar-bfap-ghost-rollingband_prelu',
    # 'keplerdr25_g2001-l201_spline_nongapped_norobovetterkoisnopps_starshuffle_astronet-300epochs-es20patience_glflux'
    # 'keplerdr25_g2001-l201_spline_nongapped_norobovetterkoisnopps_starshuffle_configE_glflux-glcentr_std_noclip-loe-lwks-6stellar-bfap-ghost-rollingband_prelu'
    # 'keplerdr25_g2001-l201_spline_nongapped_norobovetterkoisnopps_starshuffle_exonet-300epochs-es20patience_glflux-glcentr_fdl-6stellar'
    ##
    # 'keplerdr25_g2001-l201_spline_nongapped_norobovetterkois_starshuffle_exonet-300epochs-es20patience_glflux-glcentr_fdl-6stellar',
    # 'keplerdr25_g2001-l201_spline_nongapped_norobovetterkois_starshuffle_astronet-300epochs-es20patience_glflux',
    # 'keplerdr25_g2001-l201_spline_nongapped_norobovetterkois_starshuffle_configE_glflux-glcentr_std_noclip-loe-lwks-6stellar-bfap-ghost-rollingband_prelu',
    # 'keplerdr25_g2001-l201_spline_nongapped_norobovetterkois_starshuffle_configE_glflux-glcentr_std_noclip-loe-lwks-6stellar_prelu',
    'keplerdr25_g2001-l201_spline_nongapped_norobovetterkois_starshuffle_configD_glflux_prelu',
    # 'keplerdr25_g2001-l201_spline_nongapped_norobovetterkois_starshuffle_configD_glflux-loe_prelu',
    # 'keplerdr25_g2001-l201_spline_nongapped_norobovetterkois_starshuffle_configD_glflux-lwks_prelu',
    # 'keplerdr25_g2001-l201_spline_nongapped_norobovetterkois_starshuffle_configD_glflux-glcentr_std_noclip_prelu',
    # 'keplerdr25_g2001-l201_spline_nongapped_norobovetterkois_starshuffle_configD_glflux-6stellar_prelu',
    # 'keplerdr25_g2001-l201_spline_nongapped_norobovetterkois_starshuffle_configD_glflux-bfap_prelu',
    # 'keplerdr25_g2001-l201_spline_nongapped_norobovetterkois_starshuffle_configD_glflux-ghost_prelu',
    # 'keplerdr25_g2001-l201_spline_nongapped_norobovetterkois_starshuffle_configD_glflux-rollingband_prelu'
    'keplerdr25_g2001-l201_spline_nongapped_norobovetterkois_starshuffle_configE_glflux-glcentr_std_noclip-lwks_fluxnorm-loe-6stellar-bfap-ghost-rollingband-co_kic_oot-wksmaxmes_prelu',
    'keplerdr25_g2001-l201_spline_nongapped_norobovetterkois_starshuffle_configE_glflux-glcentr_std_noclip-loe-lwks-6stellar-bfap-ghost-rollingband_prelu',
    'keplerdr25_g2001-l201_spline_nongapped_norobovetterkois_starshuffle_configE_glflux-glcentr_std_noclip-loe-lwks-6stellar_prelu',
    'keplerdr25_g2001-l201_spline_nongapped_norobovetterkois_starshuffle_astronet-300epochs-es20patience_glflux',
    'keplerdr25_g2001-l201_spline_nongapped_norobovetterkois_starshuffle_exonet-300epochs-es20patience_glflux-glcentr_fdl-6stellar'
]

datasets = ['train', 'val', 'test']

k_arr = {'train': [100, 1000, 2084], 'val': [50, 150, 257], 'test': [50, 150, 283]}
# k_arr = {'train': [100, 1000, 1818], 'val': [50, 150, 222], 'test': [50, 150, 251]}  # PPs

k_curve_arr = {
    'train': np.linspace(25, 2000, 100, endpoint=True, dtype='int'),
    'val': np.linspace(25, 250, 10, endpoint=True, dtype='int'),
    'test': np.linspace(25, 250, 10, endpoint=True, dtype='int'),
    # 'train': np.linspace(25, 1800, 100, endpoint=True, dtype='int'),  # PPs
    # 'val': np.linspace(25, 200, 10, endpoint=True, dtype='int'),
    # 'test': np.linspace(25, 200, 10, endpoint=True, dtype='int'),
}
k_curve_arr_plot = {
    'train': np.linspace(200, 2000, 10, endpoint=True, dtype='int'),
    'val': np.linspace(25, 250, 8, endpoint=True, dtype='int'),
    'test': np.linspace(25, 250, 8, endpoint=True, dtype='int'),
    # 'train': np.linspace(200, 1800, 8, endpoint=True, dtype='int'),  # PPs
    # 'val': np.linspace(25, 200, 8, endpoint=True, dtype='int'),
    # 'test': np.linspace(25, 200, 8, endpoint=True, dtype='int')
}

rootDir = '/home/msaragoc/Projects/Kepler-TESS_exoplanet/Kepler_planet_finder/results_ensemble'

for study in studies:
    print('Running for study {}'.format(study))
    for dataset in datasets:
        compute_precision_at_k(study, dataset, rootDir, k_arr[dataset], k_curve_arr[dataset], k_curve_arr_plot[dataset],
                               plot=True, verbose=True)

#%% Get precision at k values for different studies

rootDir = '/home/msaragoc/Projects/Kepler-TESS_exoplanet/Kepler_planet_finder/results_ensemble'
datasets = ['train', 'val', 'test']

studies = [
    # 'keplerdr25_tps-tce1_g2001-l201_spline_gbal_nongapped_norobovetterkois_starshuffle_configE_glflux-glcentr-loe-lwks-6stellar_prelu',
    # 'keplerdr25_tce1_g2001-l201_spline_gbal_nongapped_norobovetterkois_starshuffle_configE_glflux-glcentr-loe-lwks-6stellar_prelu',
    # 'keplerdr25_g2001-l201_spline_gbal_nongapped_norobovetterkois_starshuffle_configD_glflux-glcentrFDL_prelu',
    # 'keplerdr25_g2001-l201_spline_gbal_nongapped_norobovetterkois_starshuffle_configD_glflux-lwks_prelu',
    # 'keplerdr25_g2001-l201_spline_gbal_nongapped_norobovetterkois_starshuffle_configE_glflux-glcentr-loe-lwks_prelu',
    # 'keplerdr25_g2001-l201_spline_gbal_nongapped_norobovetterkois_starshuffle_configE_glflux-6stellar-bfap-ghost-rollingband_prelu',
    # 'keplerdr25_g2001-l201_spline_gbal_nongapped_norobovetterkois_starshuffle_configE_glflux-glcentr-loe-6stellar_prelu',
    # 'keplerdr25_g2001-l201_spline_gbal_nongapped_norobovetterkois_starshuffle_configE_glflux-glcentr-loe-lwks-6stellar_prelu',
    # 'keplerdr25_g2001-l201_spline_gbal_nongapped_norobovetterkois_starshuffle_configE_glflux-glcentr-loe-lwks-6stellar-bfap-ghost-rollingband_prelu_augon',
    # 'keplerdr25_g2001-l201_spline_gbal_nongapped_norobovetterkois_starshuffle_configD_glflux_prelu',
    # 'keplerdr25_g2001-l201_spline_gbal_nongapped_norobovetterkois_starshuffle_configD_glflux-rollingband_prelu',
    # 'keplerdr25_g2001-l201_spline_gbal_nongapped_norobovetterkois_starshuffle_configD_glflux-loe_prelu',
    # 'keplerdr25_g2001-l201_spline_gbal_nongapped_norobovetterkois_starshuffle_configD_glflux-ghost_prelu',
    # 'keplerdr25_g2001-l201_spline_gbal_nongapped_norobovetterkois_starshuffle_configD_glflux-bfap_prelu',
    # 'keplerdr25_g2001-l201_spline_gbal_nongapped_norobovetterkois_starshuffle_configD_glflux-6stellar_prelu',
    # 'keplerdr25_g2001-l201_spline_gbal_nongapped_norobovetterkois_starshuffle_configE_glflux-glcentrmedcmaxn-loe-lwks-6stellar-bfap-ghost-rollingband_prelu',
    # 'keplerdr25_g2001-l201_spline_gbal_nongapped_norobovetterkois_starshuffle_configD_glflux-glcentrmedn_prelu',
    # 'keplerdr25_g2001-l201_spline_gbal_nongapped_norobovetterkois_starshuffle_configD_glflux-glcentrmedcmaxn_prelu',
    # 'keplerdr25_g2001-l201_spline_gbal_nongapped_norobovetterkois_starshuffle_configD_glflux-glcentrmedcmaxn_dir_prelu',
    # 'keplerdr25_g2001-l201_spline_gbal_nongapped_norobovetterkois_starshuffle_configD_glflux-glcentr_prelu',
    # 'keplerdr25_g2001-l201_spline_gbal_nongapped_norobovetterkois_starshuffle_configE_glflux-glcentr-loe-lwks-6stellar-bfap-ghost-rollingband',
    # 'keplerdr25_g2001-l201_spline_gbal_nongapped_norobovetterkois_starshuffle_configC_glflux-glcentr-loe-lwks-6stellar-bfap-ghost-rollingband',
    # 'keplerdr25_g2001-l201_spline_gbal_nongapped_norobovetterkois_starshuffle_configE_glflux-glcentr-loe-lwks-6stellar-ghost-bfap-rollingband_prelu_nobugdur'
    # 'keplerdr25_g2001-l201_spline_nongapped_norobovetterkoisnopps_starshuffle_exonet-300epochs-es20patience_glflux-glcentr_fdl-6stellar'
    ]
for study in studies:
    for dataset in datasets:
        res = np.load(os.path.join(rootDir, study, 'precision_at_k_{}.npy'.format(dataset)), allow_pickle=True).item()
        print('Study: {} | {} set: {}'.format(study, dataset, res))

#%% Plot ROC, PR curve and Precision at k curve for a set of studies

studies = {
    # 'keplerdr25_g2001-l201_spline_gbal_nongapped_norobovetterkois_starshuffle_configE_glflux-glcentr-loe-6stellar_prelu',
    # 'keplerdr25_g2001-l201_spline_gbal_nongapped_norobovetterkois_starshuffle_configE_glflux-glcentr-loe-lwks-6stellar_prelu': 'ExoMiner-TPS',
    # 'keplerdr25_g2001-l201_spline_gbal_nongapped_norobovetterkois_starshuffle_configE_glflux-glcentr-loe-lwks-6stellar-bfap-ghost-rollingband_prelu': 'ExoMiner-DV',
    ###
    # 'keplerdr25_g2001-l201_spline_nongapped_norobovetterkois_starshuffle_exonet-300epochs-es20patience_glflux-glcentr_fdl-6stellar': 'Exonet',
    # 'keplerdr25_g2001-l201_spline_nongapped_norobovetterkois_starshuffle_astronet-300epochs-es20patience_glflux': 'Astronet',
    # '/home/msaragoc/Projects/Kepler-TESS_exoplanet/Analysis/robovetter_analysis': 'Robovetter'
    # 'keplerdr25_g2001-l201_spline_nongapped_norobovetterkois_starshuffle_configE_glflux-glcentr_std_noclip-loe-lwks-6stellar-bfap-ghost-rollingband_prelu': 'ExoMiner-DV',
    # 'keplerdr25_g2001-l201_spline_nongapped_norobovetterkois_starshuffle_configE_glflux-glcentr_std_noclip-loe-lwks-6stellar_prelu': 'ExoMiner-TPS',
    # 'keplerdr25_g2001-l201_spline_nongapped_norobovetterkois_starshuffle_configD_glflux-glcentr_std_noclip-loe-lwks-6stellar-bfap-ghost-rollingband_prelu': 'ExoMiner-PPs',
    # 'keplerdr25_g2001-l201_spline_nongapped_norobovetterkoisnopps_starshuffle_configD_glflux-glcentr_std_noclip-loe-lwks-6stellar-bfap-ghost-rollingband_prelu': 'ExoMiner-No PPs'
    ###
    'keplerdr25_g2001-l201_spline_nongapped_norobovetterkois_starshuffle_configD_glflux_prelu': 'Baseline',
    # 'keplerdr25_g2001-l201_spline_nongapped_norobovetterkois_starshuffle_configD_glflux-loe_prelu': 'Baseline + L Odd-Even',
    # 'keplerdr25_g2001-l201_spline_nongapped_norobovetterkois_starshuffle_configD_glflux-lwks_prelu': 'Baseline + L Wk. Secondary',
    # 'keplerdr25_g2001-l201_spline_nongapped_norobovetterkois_starshuffle_configD_glflux-glcentr_std_noclip_prelu': 'Baseline + GL Centroid',
    # 'keplerdr25_g2001-l201_spline_nongapped_norobovetterkois_starshuffle_configD_glflux-6stellar_prelu': 'Baseline + Stellar',
    # 'keplerdr25_g2001-l201_spline_nongapped_norobovetterkois_starshuffle_configD_glflux-bfap_prelu': 'Baseline + Boostrap FA Prob.',
    # 'keplerdr25_g2001-l201_spline_nongapped_norobovetterkois_starshuffle_configD_glflux-ghost_prelu': 'Baseline + Ghost Diagnostic',
    # 'keplerdr25_g2001-l201_spline_nongapped_norobovetterkois_starshuffle_configD_glflux-rollingband_prelu': 'Baseline + Rolling Band Diagnostic',
    'keplerdr25_g2001-l201_spline_nongapped_norobovetterkois_starshuffle_configE_glflux-glcentr_std_noclip-lwks_fluxnorm-loe-6stellar-bfap-ghost-rollingband-co_kic_oot-wksmaxmes_prelu': 'Exominer_new',
    'keplerdr25_g2001-l201_spline_nongapped_norobovetterkois_starshuffle_configE_glflux-glcentr_std_noclip-loe-lwks-6stellar-bfap-ghost-rollingband_prelu': 'Exominer',
    'keplerdr25_g2001-l201_spline_nongapped_norobovetterkois_starshuffle_configE_glflux-glcentr_std_noclip-loe-lwks-6stellar_prelu': 'Exominer-TPS',
    'keplerdr25_g2001-l201_spline_nongapped_norobovetterkois_starshuffle_astronet-300epochs-es20patience_glflux': 'Astronet',
    'keplerdr25_g2001-l201_spline_nongapped_norobovetterkois_starshuffle_exonet-300epochs-es20patience_glflux-glcentr_fdl-6stellar': 'Exonet'
}

rootDir = '/home/msaragoc/Projects/Kepler-TESS_exoplanet/Kepler_planet_finder/results_ensemble'
datasets = ['train', 'val', 'test']

k_arr = {'train': [100, 1000, 2084], 'val': [50, 150, 257], 'test': [50, 150, 283]}
# k_arr = {'train': [100, 1000, 1818], 'val': [50, 150, 222], 'test': [50, 150, 251]}  # PPs

k_curve_arr = {
    'train': np.linspace(25, 2000, 100, endpoint=True, dtype='int'),
    'val': np.linspace(25, 250, 10, endpoint=True, dtype='int'),
    'test': np.linspace(25, 250, 10, endpoint=True, dtype='int'),
    # 'train': np.linspace(25, 1800, 100, endpoint=True, dtype='int'),  # PPs
    # 'val': np.linspace(25, 200, 10, endpoint=True, dtype='int'),
    # 'test': np.linspace(25, 200, 10, endpoint=True, dtype='int'),
}
k_curve_arr_plot = {
    'train': np.linspace(200, 2000, 10, endpoint=True, dtype='int'),
    'val': np.linspace(25, 250, 8, endpoint=True, dtype='int'),
    'test': np.linspace(25, 250, 8, endpoint=True, dtype='int'),
    # 'train': np.linspace(200, 1800, 8, endpoint=True, dtype='int'),  # PPs
    # 'val': np.linspace(25, 200, 8, endpoint=True, dtype='int'),
    # 'test': np.linspace(25, 200, 8, endpoint=True, dtype='int')
}

k_arr_limits = {
    'train': [0, k_arr['train'][-1]],
    'val': [0, k_arr['val'][-1]],
    'test': [0, k_arr['val'][-1]],
}

# plot precision at k curve
for dataset in datasets:

    f, ax = plt.subplots()

    for study, studyName in studies.items():

        if studyName == 'Robovetter':
            precision_at_k = np.load(os.path.join(study, 'precision_at_k_curve_{}.npy'.format(dataset)),
                                     allow_pickle=True).item()
        else:
            precision_at_k = np.load(os.path.join(rootDir, study, 'precision_at_k_curve_{}.npy'.format(dataset)),
                                     allow_pickle=True).item()

        ax.plot(list(precision_at_k.keys()), list(precision_at_k.values()), label=studyName)

    ax.set_ylabel('Precision')
    ax.set_xlabel('Top-K')
    ax.grid(True)
    # ax.set_xticks(np.linspace(k_arr[0], k_arr[-1], 11, endpoint=True))
    # ax.set_xticks(np.linspace(25, 250, 10, endpoint=True, dtype='int'))
    ax.set_xticks(k_curve_arr_plot[dataset])
    ax.set_yticks(np.linspace(0.9, 1, 11))
    ax.set_xlim([k_curve_arr[dataset][0], k_curve_arr[dataset][-1]])
    # ax.set_xlim(k_arr_limits[dataset])
    ax.set_ylim(top=1.001, bottom=0.9)
    ax.legend()

    f.savefig(os.path.join(rootDir, 'precision_at_k_{}.svg'.format(dataset)))
    plt.close()

    f, ax = plt.subplots()

    for study, studyName in studies.items():

        if studyName == 'Robovetter':
            precision_at_k = np.load(os.path.join(study, 'precision_at_k_curve_{}.npy'.format(dataset)),
                                     allow_pickle=True).item()
        else:
            precision_at_k = np.load(os.path.join(rootDir, study, 'precision_at_k_curve_{}.npy'.format(dataset)),
                                     allow_pickle=True).item()

        kvalues = np.array(list(precision_at_k.keys()))
        precvalues = np.array(list(precision_at_k.values()))
        num_misclf_examples = kvalues - kvalues * precvalues
        ax.plot(kvalues, num_misclf_examples, label=studyName)

    ax.set_ylabel('Number Misclassfied TCEs')
    ax.set_xlabel('Top-K')
    ax.grid(True)
    # ax.set_xticks(np.linspace(k_arr[0], k_arr[-1], 11, endpoint=True))
    # ax.set_xticks(k_arr[dataset])
    # ax.set_yticks(np.linspace(0.9, 1, 11))
    # ax.set_xlim([k_arr[dataset][0], k_arr[dataset][-1]])
    # ax.set_ylim(top=1.001, bottom=0.9)
    ax.set_xticks(k_curve_arr_plot[dataset])
    # ax.set_xlim(k_arr_limits[dataset])
    ax.set_xlim([k_curve_arr_plot[dataset][0], k_curve_arr_plot[dataset][-1]])
    ax.legend()

    f.savefig(os.path.join(rootDir, 'misclassified_at_k_{}.svg'.format(dataset)))
    plt.close()


# plot PR curve
for dataset in datasets:

    f, ax = plt.subplots()

    for study, studyName in studies.items():

        if studyName == 'Robovetter':
            recallThr = np.load(os.path.join(study, 'recall_thr_{}.npy'.format(dataset)), allow_pickle=True)
            precisionThr = np.load(os.path.join(study, 'precision_thr_{}.npy'.format(dataset)), allow_pickle=True)
        else:
            metricsStudy = np.load(os.path.join(rootDir, study, 'results_ensemble.npy'), allow_pickle=True).item()
            recallThr = metricsStudy['{}_rec_thr'.format(dataset)]
            precisionThr = metricsStudy['{}_prec_thr'.format(dataset)]

        ax.plot(recallThr, precisionThr, label=studyName)

    ax.set_ylabel('Precision')
    ax.set_xlabel('Recall')
    ax.grid(True)
    ax.set_xticks(np.linspace(0.5, 1, 11, endpoint=True))
    ax.set_yticks(np.linspace(0.5, 1, 11, endpoint=True))
    ax.set_xlim([0.5, 1])
    ax.set_ylim([0.5, 1])
    ax.legend()

    f.savefig(os.path.join(rootDir, 'precision_recall_curve_{}.svg'.format(dataset)))
    plt.close()

    f, ax = plt.subplots()

    for study, studyName in studies.items():

        if studyName == 'Robovetter':
            recallThr = np.load(os.path.join(study, 'recall_thr_{}.npy'.format(dataset)), allow_pickle=True)
            precisionThr = np.load(os.path.join(study, 'precision_thr_{}.npy'.format(dataset)), allow_pickle=True)
        else:
            metricsStudy = np.load(os.path.join(rootDir, study, 'results_ensemble.npy'), allow_pickle=True).item()
            recallThr = metricsStudy['{}_rec_thr'.format(dataset)]
            precisionThr = metricsStudy['{}_prec_thr'.format(dataset)]

        ax.plot(recallThr, precisionThr, label=studyName)

    ax.set_ylabel('Precision')
    ax.set_xlabel('Recall')
    ax.grid(True)
    ax.set_xticks(np.linspace(0.5, 1, 11, endpoint=True))
    ax.set_yticks(np.linspace(0.5, 1, 11, endpoint=True))
    ax.set_xlim([0.6, 1])
    ax.set_ylim([0.7, 1])
    ax.legend()

    f.savefig(os.path.join(rootDir, 'precision_recall_curve_{}_zoomin.svg'.format(dataset)))
    plt.close()

# plot ROC
for dataset in datasets:

    f, ax = plt.subplots()

    for study, studyName in studies.items():

        if studyName == 'Robovetter':
            recallThr = np.load(os.path.join(study, 'recall_thr_{}.npy'.format(dataset)), allow_pickle=True)
            fprThr = np.load(os.path.join(study, 'fpr_thr_{}.npy'.format(dataset)), allow_pickle=True)
        else:
            metricsStudy = np.load(os.path.join(rootDir, study, 'results_ensemble.npy'), allow_pickle=True).item()
            recallThr = metricsStudy['{}_rec_thr'.format(dataset)]
            fprThr = metricsStudy['{}_fp'.format(dataset)] / (metricsStudy['{}_tn'.format(dataset)] + metricsStudy['{}_fp'.format(dataset)])

        ax.plot(fprThr, recallThr, label=studyName)

    ax.set_ylabel('True Positive Rate')
    ax.set_xlabel('False Positive Rate')
    ax.grid(True)
    ax.set_xticks(np.linspace(0, 0.5, 11, endpoint=True))
    ax.set_yticks(np.linspace(0.75, 1, 11, endpoint=True))
    ax.set_xlim([0, 0.5])
    ax.set_ylim([0.75, 1])
    ax.legend()

    f.savefig(os.path.join(rootDir, 'roc_{}.svg'.format(dataset)))
    plt.close()

    f, ax = plt.subplots()

    for study, studyName in studies.items():

        if studyName == 'Robovetter':
            recallThr = np.load(os.path.join(study, 'recall_thr_{}.npy'.format(dataset)), allow_pickle=True)
            fprThr = np.load(os.path.join(study, 'fpr_thr_{}.npy'.format(dataset)), allow_pickle=True)
        else:
            metricsStudy = np.load(os.path.join(rootDir, study, 'results_ensemble.npy'), allow_pickle=True).item()
            recallThr = metricsStudy['{}_rec_thr'.format(dataset)]
            fprThr = metricsStudy['{}_fp'.format(dataset)] / (metricsStudy['{}_tn'.format(dataset)] + metricsStudy['{}_fp'.format(dataset)])

        ax.plot(fprThr, recallThr, label=studyName)

    ax.set_ylabel('True Positive Rate')
    ax.set_xlabel('False Positive Rate')
    ax.grid(True)
    ax.set_xticks(np.linspace(0, 0.5, 11, endpoint=True))
    ax.set_yticks(np.linspace(0.75, 1, 11, endpoint=True))
    ax.set_xlim([0, 0.5])
    ax.set_ylim([0.9, 1])
    ax.legend()

    f.savefig(os.path.join(rootDir, 'roc_{}_zoomin.svg'.format(dataset)))
    plt.close()
