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
    ('ConfigK_g301-l31_6tr_spline_nongapped_gl-depth_flux+glcentr_std_noclip-ldiffimgunc+loe_subtract+lwks-ptemp-albedo-mes+stellar+bfap+rollingband+ghost_nopps_secsymphase_wksnorm_maxflux-wks_nokoiephemtest',
     'keplerdr25-dv_g301-l31_6tr_spline_nongapped_norobovetterkois_starshuffle_configK_prelu_secsymphase_wksnorm_maxflux-wks_koiephem_nopps_nokoiephemtest'),
    ]

# topk = {'train': [100, 1000, 2084], 'val': [50, 150, 257], 'test': [50, 150, 283]}
topk = {'train': [100, 1000, 1818], 'val': [50, 150, 222], 'test': [50, 150, 251]}  # No PPs
saveFp = None
# saveFp = '/home/msaragoc/Projects/Kepler-TESS_exoplanet/Kepler_planet_finder/results_ensemble/map_datasets_metrics.npy'
# datasetsMetrics = np.load('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Kepler_planet_finder/results_ensemble/'
#                           'map_datasets_metrics.npy', allow_pickle=True).item()
for studyName, studyDir in studiesToAdd:

    datasetsMetrics = create_metric_mapping(topk, saveFp)

    add_study_to_restbl(studyName, studyDir, studyRootDir, datasetsMetrics, resTblFp)
