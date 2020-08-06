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


metrics = {
    'binary_accuracy': 'Acc',
    'precision': 'Prec',
    'recall': 'Rec',
    'auc_pr': 'AUC PR',
    'auc_roc': 'AUC ROC',
}

# '' means training set
datasets = {
    'train': 'Train',
    'val': 'Val',
    'test': 'Test'
}

topk = {'train': [100, 1000, 2000], 'val': [50, 150, 250], 'test': [50, 150, 250]}
# topk = {'train': [10, 100, 1000], 'val': [10, 100, 1000], 'test': [10, 100, 1000]}

datasetsMetrics = {}
for dataset in datasets:
    for metric in metrics:
        if dataset == '':
            datasetsMetrics['{}'.format(metric)] = 'Train{}'.format(metrics[metric])
        else:
            datasetsMetrics['{}_{}'.format(dataset, metric)] = '{}{}'.format(datasets[dataset], metrics[metric])

    if dataset == '':
        for k in topk['train']:
            datasetsMetrics['{}_precision_at_{}'.format('train', k)] = 'TrainPrec{}'.format(k)
    else:
        for k in topk[dataset]:
            datasetsMetrics['{}_precision_at_{}'.format(dataset, k)] = '{}Prec{}'.format(datasets[dataset], k)

np.save('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Kepler_planet_finder/results_ensemble/map_datasets_metrics.npy',
        datasetsMetrics)

# resTbl = pd.DataFrame(columns=['Study'] + list(datasetsMetrics.values()) + ['Study folder'])
# resTbl.to_csv('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Kepler_planet_finder/results_ensemble/'
#               'results_studies_7-30-2020.csv', index=False)

#%%

studyRootDir = '/home/msaragoc/Projects/Kepler-TESS_exoplanet/Kepler_planet_finder/results_ensemble/'
resTblFp = '/home/msaragoc/Projects/Kepler-TESS_exoplanet/Kepler_planet_finder/results_ensemble/' \
           'results_studies_7-30-2020.csv'
datasetsMetrics = np.load('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Kepler_planet_finder/results_ensemble/'
                          'map_datasets_metrics.npy', allow_pickle=True).item()

studiesToAdd = [
    ('ConfigE_spline_gapped_gbal_glflux+glcentroid+loe+lwks+6stellar_prelu',
     'keplerdr25_g2001-l201_spline_gbal_nongapped_norobovetterkois_starshuffle_configE_glflux-glcentr-loe-lwks-6stellar_prelu'),
    # ('ConfigD_spline_gapped_gbal_glflux+6stellar_prelu',
    #  'keplerdr25_g2001-l201_spline_gbal_nongapped_norobovetterkois_starshuffle_configD_glflux-6stellar_prelu'),
    # ('ConfigD_spline_gapped_gbal_glflux+bfap_prelu',
    #  'keplerdr25_g2001-l201_spline_gbal_nongapped_norobovetterkois_starshuffle_configD_glflux-bfap_prelu'),
    # ('ConfigD_spline_gapped_gbal_glflux+ghost_prelu',
    #  'keplerdr25_g2001-l201_spline_gbal_nongapped_norobovetterkois_starshuffle_configD_glflux-ghost_prelu'),
    # ('ConfigD_spline_gapped_gbal_glflux+rollingband_prelu',
    #  'keplerdr25_g2001-l201_spline_gbal_nongapped_norobovetterkois_starshuffle_configD_glflux-rollingband_prelu'),
    # ('ConfigD_spline_gapped_gbal_glflux+loe_prelu',
    #  'keplerdr25_g2001-l201_spline_gbal_nongapped_norobovetterkois_starshuffle_configD_glflux-loe_prelu'),
    # ('ConfigD_spline_gapped_gbal_glflux+lwks_prelu',
    #  'keplerdr25_g2001-l201_spline_gbal_nongapped_norobovetterkois_starshuffle_configD_glflux-lwks_prelu'),
    # ('ConfigD_spline_gapped_gbal_glflux_prelu',
    #  'keplerdr25_g2001-l201_spline_gbal_nongapped_norobovetterkois_starshuffle_configD_glflux_prelu')
]

for studyName, studyDir in studiesToAdd:
    add_study_to_restbl(studyName, studyDir, studyRootDir, datasetsMetrics, resTblFp)
