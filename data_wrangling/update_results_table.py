""" Automate the update of the table with the results of new studies. """

import pandas as pd
import numpy as np
import os
from datetime import datetime

#%% Function used to add experiment to table with results for different experiments


def add_study_to_restbl(studyName, studyDir, studyRootDir, mapDatasetMetric, resTblFp):
    """ Add study results to table.

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

#%% Function used to create mapping between metric key and column name


def create_metric_mapping(topk, saveFp=None):
    """ Create mapping between metric key and column name.

    :param topk: dict, k values used to compute precision at k for different datasets
    :param saveFp: str, filepath used to save the metric mapping
    :return:
        datasetsMetrics: dict, mapping between metric key and column name in the table
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


#%% Create new results table

datasetsMetrics = np.load('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Kepler_planet_finder/results_ensemble/'
                          'map_datasets_metrics.npy', allow_pickle=True).item()

resTbl = pd.DataFrame(columns=['Study'] + list(datasetsMetrics.values()) + ['Study folder'])

resTbl.to_csv('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Kepler_planet_finder/results_ensemble/'
              'results_studies_{}.csv'.format(datetime.now().date()), index=False)

#%% Add new experiment result to table with results

# experiment directory
studyRootDir = '/home/msaragoc/Projects/Kepler-TESS_exoplanet/Kepler_planet_finder/results_ensemble/'

# filepath of table with results
resTblFp = '/home/msaragoc/Projects/Kepler-TESS_exoplanet/Kepler_planet_finder/results_ensemble/' \
           'results_studies_8-26-2020.csv'

# name of the experiement and name of the experiment's directory to be added
studiesToAdd = [
    # ('Astronet_g2001-l201_9tr_spline_gapped_glflux_correctprimarygapping_nopps_ckoiper_secparams',
    #  'keplerdr25-dv_g2001-l201_9tr_spline_gapped_norobovetterkois_starshuffle_astronet_secsymphase_nopps_ckoiper'),
    # ('AstronetES20_g2001-l201_9tr_spline_gapped_glflux_correctprimarygapping_nopps_ckoiper_secparams',
    #  'keplerdr25-dv_g2001-l201_9tr_spline_gapped_norobovetterkois_starshuffle_astronetes20_secsymphase_nopps_ckoiper'),
    # ('Exonet_g2001-l201_9tr_spline_gapped_glflux+glcentrfdl+stellar_correctprimarygapping_nopps_ckoiper_secparams',
    #  'keplerdr25-dv_g2001-l201_9tr_spline_gapped_norobovetterkois_starshuffle_exonet_secsymphase_nopps_ckoiper'),
    # ('ExonetES20_g2001-l201_9tr_spline_gapped_glflux+glcentrfdl+stellar_correctprimarygapping_nopps_ckoiper_secparams',
    #  'keplerdr25-dv_g2001-l201_9tr_spline_gapped_norobovetterkois_starshuffle_exonetes20_secsymphase_nopps_ckoiper'),
    # ('ConfigK_g301-l31_6tr_spline_nongapped_gl-depth_flux+glcentr_std_noclip+loe_subtract+lwks_normmaxflux_wks+stellar+correctprimarygapping_nopps_ckoiper_secparams_per',
    #  'keplerdr25-dv_g301-l31_6tr_spline_nongapped_norobovetterkois_starshuffle_configK_secsymphase_nopps_ckoiper_tpsfeatures'),
    # ('ConfigK_g301-l31_6tr_spline_nongapped_gl-depth_flux+glcentr_std_noclip+loe_subtract+lwks_normmaxflux_wks+stellar+correctprimarygapping_nopps_ckoiper_secparams_per_TCEs1',
    #  'keplerdr25-dv_g301-l31_6tr_spline_nongapped_norobovetterkois_starshuffle_configK_secsymphase_nopps_ckoiper_tpsfeatures_tces1'),
    # ('ConfigK_g301-l31_6tr_spline_nongapped_gl-depth_flux+glcentr_std_noclip+loe_subtract+lwks_normmaxflux_wks+stellar+correctprimarygapping_nopps_ckoiper_secparams_per_TPS',
    #  'keplerdr25-tps_g301-l31_6tr_spline_nongapped_norobovetterkois_starshuffle_configK_secsymphase_nopps_ckoiper')
    ('ConfigK_g301-l31_6tr_spline_nongapped_gl-depth_flux+glcentr_std_noclip+loe_subtract+lwks_normmaxflux_wks+stellar+correctprimarygapping_nopps_ckoiper_secparams_per_DVonTPS',
    'keplerdr25-tps_g301-l31_6tr_spline_nongapped_norobovetterkois_starshuffle_configK_secsymphase_nopps_ckoiper_dvmodel')
    ]

# select values of k for which precision at k was computed
# topk = {'train': [100, 1000, 2084], 'val': [50, 150, 257], 'test': [50, 150, 283]}
topk = {'train': [100, 1000, 1818], 'val': [50, 150, 222], 'test': [50, 150, 251]}  # No PPs

# if saveFP is not None, create a new metric mapping and save it
saveFp = None
# saveFp = '/home/msaragoc/Projects/Kepler-TESS_exoplanet/Kepler_planet_finder/results_ensemble/map_datasets_metrics.npy'
# datasetsMetrics = np.load('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Kepler_planet_finder/results_ensemble/'
#                           'map_datasets_metrics.npy', allow_pickle=True).item()

for studyName, studyDir in studiesToAdd:

    datasetsMetrics = create_metric_mapping(topk, saveFp)

    add_study_to_restbl(studyName, studyDir, studyRootDir, datasetsMetrics, resTblFp)
