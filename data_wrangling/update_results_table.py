"""
Automate the update of the table with the results of studies
"""

import pandas as pd
import numpy as np
import os

#%%


def add_study_to_restbl(studyName, studyDir, studyRootDir, mapDatasetMetric, resTblFp):
    """ Add study results to table containing results from previous studies.

    :param studyName: str, name to give the study in the results study table.
    :param studyDir: str, name of the study directory.
    :param studyRootDir: str, filepath to root directory containing the study.
    :param mapDatasetMetric: dict, mapping between study result keys and column names in the result study table.
    :param resTblFp: str, filepath to results study table
    :return:
    """

    if os.path.exists(resTblFp):
        resTbl = pd.read_csv(resTblFp)
    else:
        resTbl = pd.DataFrame(columns=['Study'] + list(mapDatasetMetric.values()) + ['Study folder'])

    resStudy = np.load(os.path.join(studyRootDir, studyDir, 'results_ensemble.npy'), allow_pickle=True).item()

    dataDict = {}
    for metric in mapDatasetMetric:
        dataDict[mapDatasetMetric[metric]] = [resStudy[metric]]

    studyTbl = pd.DataFrame(data=dataDict)

    studyTbl.insert(0, 'Study', studyName)
    studyTbl['Study folder'] = studyDir

    resTbl = pd.concat([resTbl, studyTbl], axis=0)

    resTbl.to_csv(resTblFp, index=False)

#%% Create mapping between metric key and column name


def create_metric_mapping(topk, saveFp=None):
    """

    :param topk:
    :param saveFp:
    :return:
        datasetsMetrics:
    """

    metrics = {
        'binary_accuracy': 'Acc',
        'precision': 'Prec',
        'recall': 'Rec',
        'auc_pr': 'AUC PR',
        'auc_roc': 'AUC ROC',
    }

    datasets = {
        'train': 'Train',
        'val': 'Val',
        'test': 'Test'
    }

    datasetsMetrics = {}
    for dataset in datasets:
        for metric in metrics:
            if dataset == '':
                datasetsMetrics['{}'.format(metric)] = 'Train{}'.format(metrics[metric])
            else:
                datasetsMetrics['{}_{}'.format(dataset, metric)] = '{}{}'.format(datasets[dataset], metrics[metric])

        if dataset == '':
            for k_i in range(len(topk['train'])):
                if k_i == len(topk['train']) - 1:
                    datasetsMetrics['{}_precision_at_{}'.format('train', topk['train'][k_i])] = \
                        'TrainPrec{}'.format('Max')
                else:
                    datasetsMetrics['{}_precision_at_{}'.format('train', topk['train'][k_i])] = \
                        'TrainPrec{}'.format(topk['train'][k_i])
        else:
            for k_i in range(len(topk[dataset])):
                if k_i == len(topk[dataset]) - 1:
                    datasetsMetrics['{}_precision_at_{}'.format(dataset, topk[dataset][k_i])] = \
                        '{}PrecMax'.format(datasets[dataset])
                else:
                    datasetsMetrics['{}_precision_at_{}'.format(dataset, topk[dataset][k_i])] = \
                        '{}Prec{}'.format(datasets[dataset], topk[dataset][k_i])

    if saveFp:
        np.save(saveFp, datasetsMetrics)

    return datasetsMetrics


#%%

# create new results table
# datasetsMetrics = np.load('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Kepler_planet_finder/results_ensemble/'
#                           'map_datasets_metrics.npy', allow_pickle=True).item()
# resTbl = pd.DataFrame(columns=['Study'] + list(datasetsMetrics.values()) + ['Study folder'])
# resTbl.to_csv('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Kepler_planet_finder/results_ensemble/'
#               'results_studies_7-30-2020.csv', index=False)

studyRootDir = '/home/msaragoc/Projects/Kepler-TESS_exoplanet/Kepler_planet_finder/results_ensemble/'
resTblFp = '/home/msaragoc/Projects/Kepler-TESS_exoplanet/Kepler_planet_finder/results_ensemble/' \
           'results_studies_8-26-2020.csv'

studiesToAdd = [
    # ('ConfigE_spline_gapped_gbal_glflux+glcentr+loediff+lwks+6stellar+bfap+ghost+rollingband_prelu_nobugdur',
    #  'keplerdr25_g2001-l201_spline_gbal_nongapped_norobovetterkois_starshuffle_configE_glflux-glcentr-loediff-lwks-6stellar-ghost-bfap-rollingband_prelu_nobugdur'),
    # ('ConfigE_spline_gapped_gbal_glflux+glcentr+loe+lwks+6stellar+bfap+ghost+rollingband_prelu',
    #  'keplerdr25_g2001-l201_spline_gbal_nongapped_norobovetterkois_starshuffle_configE_glflux-glcentr-loe-lwks-6stellar-bfap-ghost-rollingband_prelu'),
    # ('ConfigD_spline_gapped_gbal_glflux+glcentr+loe+lwks+6stellar_prelu_tps-tce1',
    #  'keplerdr25_tps-tce1_g2001-l201_spline_gbal_nongapped_norobovetterkois_starshuffle_configE_glflux-glcentr-loe-lwks-6stellar_prelu'),
    # ('ConfigD_spline_gapped_gbal_glflux+glcentr+loe+lwks+6stellar_prelu_dv-tce1',
    #  'keplerdr25_tce1_g2001-l201_spline_gbal_nongapped_norobovetterkois_starshuffle_configE_glflux-glcentr-loe-lwks-6stellar_prelu')
    # ('ConfigD_spline_gapped_gbal_glflux+bfap_prelu',
    #  'keplerdr25_g2001-l201_spline_gbal_nongapped_norobovetterkois_starshuffle_configD_glflux-bfap_prelu'),
    # ('ConfigD_spline_gapped_gbal_glflux+ghost_prelu',
    #  'keplerdr25_g2001-l201_spline_gbal_nongapped_norobovetterkois_starshuffle_configD_glflux-ghost_prelu'),
    # ('ConfigD_spline_gapped_gbal_glflux+rollingband_prelu',
    #  'keplerdr25_g2001-l201_spline_gbal_nongapped_norobovetterkois_starshuffle_configD_glflux-rollingband_prelu'),
    # ('ConfigD_spline_gapped_gbal_glflux+loe_prelu',
    #  'keplerdr25_g2001-l201_spline_gbal_nongapped_norobovetterkois_starshuffle_configD_glflux-loe_prelu'),
    # ('ConfigD_spline_gapped_glflux+loe_prelu',
    #  'keplerdr25_g2001-l201_spline_nongapped_norobovetterkois_starshuffle_configD_glflux-loe_prelu'),
    # ('ConfigD_spline_gapped_glflux_prelu',
    #  'keplerdr25_g2001-l201_spline_nongapped_norobovetterkois_starshuffle_configD_glflux_prelu'),
    # ('ConfigD_spline_nongapped_nopps_glflux_prelu',
    #  'keplerdr25_g2001-l201_spline_nongapped_norobovetterkoisnopps_starshuffle_configD_glflux_prelu'),
    # ('ConfigD_spline_nongapped_nopps_glflux+glcentr_std_noclip_prelu',
    #  'keplerdr25_g2001-l201_spline_nongapped_norobovetterkoisnopps_starshuffle_configD_glflux-glcentr_std_noclip_prelu'),
    # ('ConfigD_spline_nongapped_nopps_glflux+glcentr_std_noclip+loe+lwks+6stellar+bfap+ghost+rollingband_prelu',
    #  'keplerdr25_g2001-l201_spline_nongapped_norobovetterkoisnopps_starshuffle_configD_glflux-glcentr_std_noclip-loe-lwks-6stellar-bfap-ghost-rollingband_prelu'),
    # ('ConfigD_spline_nongapped_glflux+glcentr_std_noclip+loe+lwks+6stellar+bfap+ghost+rollingband_prelu',
    #  'keplerdr25_g2001-l201_spline_nongapped_norobovetterkois_starshuffle_configD_glflux-glcentr_std_noclip-loe-lwks-6stellar-bfap-ghost-rollingband_prelu'),
    # ('ConfigD_spline_nongapped_glflux+glcentr_std_noclip+loe+lwks+6stellar_prelu',
    #  'keplerdr25_g2001-l201_spline_nongapped_norobovetterkois_starshuffle_configD_glflux-glcentr_std_noclip-loe-lwks-6stellar_prelu'),
    # ('ConfigD_spline_nongapped_glflux+glcentr_std_noclip+loe+lwks+6stellar_prelu_dv-tce1',
    #  'keplerdr25_g2001-l201_spline_nongapped_norobovetterkois_starshuffle_configD_glflux-glcentr_std_noclip-loe-lwks-6stellar_prelu_tce1'),
    # ('ConfigD_spline_nongapped_glflux+glcentr_std_noclip+loe+lwks+6stellar_prelu_tps-tce1',
    #  'keplerdr25_g2001-l201_spline_nongapped_norobovetterkois_starshuffle_configD_glflux-glcentr_std_noclip-loe-lwks-6stellar_prelu_tps-tce1'),
    # ('ConfigD_spline_gapped_gbal_glflux_prelu',
    #  'keplerdr25_g2001-l201_spline_gbal_nongapped_norobovetterkois_starshuffle_configD_glflux_prelu'),
    # ('Astronet-50epochs_spline_nongapped_glflux_nopps',
    #  'keplerdr25_g2001-l201_spline_nongapped_norobovetterkoisnopps_starshuffle_astronet-50epochs_glflux'),
    # ('Astronet-300epochs-ES20patience_spline_nongapped_glflux_nopps',
    #  'keplerdr25_g2001-l201_spline_nongapped_norobovetterkoisnopps_starshuffle_astronet-300epochs-es20patience_glflux'),
    # ('Exonet-300epochs-ES20patience_spline_nongapped_glflux-glcentr_fdl-6stellar',
    #  'keplerdr25_g2001-l201_spline_nongapped_norobovetterkois_starshuffle_exonet-300epochs-es20patience_glflux-glcentr_fdl-6stellar'),
    # ('Exonet-50epochs_spline_nongapped_glflux-glcentr_fdl-6stellar',
    #  'keplerdr25_g2001-l201_spline_nongapped_norobovetterkois_starshuffle_exonet-50epochs_glflux-glcentr_fdl-6stellar'),
    # ('ConfigE_spline_nongapped_glflux+glcentr_std_noclip+loe+lwks+6stellar_prelu',
    #  'keplerdr25_g2001-l201_spline_nongapped_norobovetterkois_starshuffle_configE_glflux-glcentr_std_noclip-loe-lwks-6stellar_prelu'),
    # ('ConfigE_spline_nongapped_glflux+glcentr_std_noclip+loe+lwks+6stellar-bfap-ghost-rollingband_prelu_nopps',
    #  'keplerdr25_g2001-l201_spline_nongapped_norobovetterkoisnopps_starshuffle_configE_glflux-glcentr_std_noclip-loe-lwks-6stellar-bfap-ghost-rollingband_prelu')
    # ('Exonet-300epochs-ES20patience_spline_nongapped_glflux+glcentr_fdl+6stellar',
    #  'keplerdr25_g2001-l201_spline_nongapped_norobovetterkoisnopps_starshuffle_exonet-300epochs-es20patience_glflux-glcentr_fdl-6stellar'),
    # ('ConfigD_spline_gapped_gbal_glflux+glcentr_fdl_prelu',
    #  'keplerdr25_g2001-l201_spline_nongapped_norobovetterkois_starshuffle_configD_glflux-glcentr_fdl_prelu'),
    # ('ConfigD_spline_gapped_glflux+6stellar_prelu',
    #  'keplerdr25_g2001-l201_spline_nongapped_norobovetterkois_starshuffle_configD_glflux-6stellar_prelu'),
    # ('ConfigD_spline_gapped_glflux+bfap_prelu',
    #  'keplerdr25_g2001-l201_spline_nongapped_norobovetterkois_starshuffle_configD_glflux-bfap_prelu'),
    # ('ConfigD_spline_gapped_glflux+ghost_prelu',
    #  'keplerdr25_g2001-l201_spline_nongapped_norobovetterkois_starshuffle_configD_glflux-ghost_prelu'),
    # ('ConfigD_spline_gapped_glflux+rollingband_prelu',
    #  'keplerdr25_g2001-l201_spline_nongapped_norobovetterkois_starshuffle_configD_glflux-rollingband_prelu'),
    # ('ConfigD_spline_gapped_gbal_glflux+loe_prelu',
    #  'keplerdr25_g2001-l201_spline_gbal_nongapped_norobovetterkois_starshuffle_configD_glflux-loe_prelu'),
    # ('ConfigD_spline_gapped_gbal_glflux+glcentrstdclip_prelu',
    #  'keplerdr25_g2001-l201_spline_nongapped_norobovetterkois_starshuffle_configD_glflux-glcentr_std_clip_prelu'),
    # ('ConfigD_spline_gapped_gbal_glflux+glcentrstdnoclip_prelu',
    #  'keplerdr25_g2001-l201_spline_nongapped_norobovetterkois_starshuffle_configD_glflux-glcentr_std_noclip_prelu'),
    # ('ConfigD_spline_gapped_gbal_glflux+glcentrmedcmaxn_prelu',
    #  'keplerdr25_g2001-l201_spline_gbal_nongapped_norobovetterkois_starshuffle_configD_glflux-glcentrmedcmaxn_prelu'),
    # ('ConfigD_spline_gapped_glflux+glcentrmedcmaxn_dir_prelu',
    #  'keplerdr25_g2001-l201_spline_nongapped_norobovetterkois_starshuffle_configD_glflux-glcentrmedcmaxn_dir_prelu'),
    # ('ConfigD_spline_nongapped_glflux+glcentr_fdl+6stellar_prelu',
    #  'keplerdr25_g2001-l201_spline_nongapped_norobovetterkois_starshuffle_configD_glflux-glcentr_fdl-6stellar_prelu'),
    # ('ConfigE_spline_nongapped_glflux+glcentr_fdl+6stellar_prelu',
    #  'keplerdr25_g2001-l201_spline_nongapped_norobovetterkois_starshuffle_configE_glflux-glcentr_fdl-6stellar_prelu')
    # ('ConfigE_spline_gapped_glflux+glcentr_std_noclip+loe+lwks+6stellar-bfap-ghost-rollingband_prelu',
    #  'keplerdr25_g2001-l201_spline_gapped_norobovetterkois_starshuffle_configE_glflux-glcentr_std_noclip-loe-lwks-6stellar-bfap-ghost-rollingband_prelu'),
    # ('ConfigE_spline_nongapped_glflux+glcentr_std_noclip+loe+lwks+6stellar-bfap-ghost-rollingband_prelu_g301-l31',
    #  'keplerdr25_g301-l31_spline_gapped_norobovetterkois_starshuffle_configD_glflux-glcentr_std_noclip-loe-lwks-6stellar-bfap-ghost-rollingband_prelu'),
    # ('ConfigE_spline_nongapped_glflux+glcentr_medind_std+loe+lwks+6stellar-bfap-ghost-rollingband_prelu',
    #  'keplerdr25_g2001-l201_spline_nongapped_norobovetterkois_starshuffle_configE_glflux-glcentr_medind_std-loe-lwks-6stellar-bfap-ghost-rollingband_prelu'),
    # ('ConfigD_spline_nongapped_glflux+mes_prelu',
    #  '/home/msaragoc/Projects/Kepler-TESS_exoplanet/Kepler_planet_finder/results_ensemble/keplerdr25_g2001-l201_spline_nongapped_norobovetterkois_starshuffle_configD_glflux-mes_prelu'),
    # ('ConfigD_spline_nongapped_glflux+lwks_fluxnorm_wksmaxmes_prelu',
    #  '/home/msaragoc/Projects/Kepler-TESS_exoplanet/Kepler_planet_finder/results_ensemble/keplerdr25_g2001-l201_spline_nongapped_norobovetterkois_starshuffle_configD_glflux-lwks_fluxnorm-wksmaxmes_prelu')
    # ('ConfigD_spline_nongapped_glflux+co_kic_oot_prelu',
    #  'keplerdr25_g2001-l201_spline_nongapped_norobovetterkois_starshuffle_configD_glflux-co_kic_oot_prelu'),
    # ('ConfigD_spline_nongapped_glflux+glcentr_std_noclip+co_kic_oot_prelu',
    #  'keplerdr25_g2001-l201_spline_nongapped_norobovetterkois_starshuffle_configD_glflux-glcentr_std_noclip-co_kic_oot_prelu'),
    # ('ConfigE_spline_nongapped_glflux+glcentr_std_noclip+loe+lwks_fluxnorm+6stellar+bfap+ghost+rollingband+co_kic_oot+wksmaxmes_prelu',
    #  'keplerdr25_g2001-l201_spline_nongapped_norobovetterkois_starshuffle_configE_glflux-glcentr_std_noclip-lwks_fluxnorm-loe-6stellar-bfap-ghost-rollingband-co_kic_oot-wksmaxmes_prelu'),
    ('ConfigH_g301-l31_spline_nongapped_glflux+glcentr_std_noclip+loe+lwks_fluxnorm+6stellar+bfap+ghost+rollingband_prelu',
     'keplerdr25_g301-l31_spline_nongapped_norobovetterkois_starshuffle_configH_glflux-glcentr_std_noclip-lwks_fluxnorm-loe-6stellar-bfap-ghost-rollingband_prelu')
]

topk = {'train': [100, 1000, 2084], 'val': [50, 150, 257], 'test': [50, 150, 283]}
# topk = {'train': [100, 1000, 1818], 'val': [50, 150, 222], 'test': [50, 150, 251]}  # No PPs
saveFp = None
# saveFp = '/home/msaragoc/Projects/Kepler-TESS_exoplanet/Kepler_planet_finder/results_ensemble/map_datasets_metrics.npy'
# datasetsMetrics = np.load('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Kepler_planet_finder/results_ensemble/'
#                           'map_datasets_metrics.npy', allow_pickle=True).item()
for studyName, studyDir in studiesToAdd:

    datasetsMetrics = create_metric_mapping(topk, saveFp)

    add_study_to_restbl(studyName, studyDir, studyRootDir, datasetsMetrics, resTblFp)
