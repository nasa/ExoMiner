import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report

def robovetter_objective(eff, complt, complt_target):
    """ Objective function used for threshold optimization.

    :param eff: float, effectiveness [0, 1]
    :param complt: float, completeness [0, 1]
    :param complt_target: float, completeness target [0, 1]
    :return:
    """

    return np.sqrt((1 - eff) ** 2 + (complt - complt_target) ** 2)


def _effectivess_score(pred_class, labels):

    tn = len(np.where((pred_class + labels) == 0)[0])
    fp = len(np.where((labels - pred_class) == -1)[0])

    return tn / (tn + fp)


def _completeness_score(pred_class, labels):

    tp = np.sum(pred_class * labels)
    fn = len(np.where((labels - pred_class) == 1)[0])

    return tp / (tp + fn)


num_thresholds = 1000
threshold_range = np.linspace(0, 1, num=num_thresholds, endpoint=True)
rankingDir = '/home/msaragoc/Projects/Kepler-TESS_exoplanet/Kepler_planet_finder/results_ensemble/' \
             'dr25_spline_gapped_glflux_cumkoidisp_glfluxhvbconfig/'
rankingTblTrain = pd.read_csv(os.path.join(rankingDir, 'ensemble_ranked_predictions_trainset'))
rankingTblVal = pd.read_csv(os.path.join(rankingDir, 'ensemble_ranked_predictions_valset'))
rankingTbl = pd.concat([rankingTblTrain, rankingTblVal])

pred_class = {thr: (rankingTbl['score'].values >= thr).astype('int') for thr in threshold_range}

complt_target = 0.8
eff_complt_thr = {thr: {'eff': _effectivess_score(pred_class[thr], rankingTbl['label'].values),
                        'complt': _completeness_score(pred_class[thr], rankingTbl['label'].values)}
                  for thr in threshold_range}

obj_values = {thr: robovetter_objective(complt_target=complt_target, **eff_complt_thr[thr]) for thr in threshold_range}

opt_thr = min(obj_values, key=obj_values.get)

f, ax = plt.subplots()
ax.plot(threshold_range, [obj_values[thr] for thr in threshold_range])
ax.scatter(threshold_range, [obj_values[thr] for thr in threshold_range])
ax.scatter(opt_thr, obj_values[opt_thr], c='r',
           label='optimal threshold (obj)= {:.4f} ({:.4f})'.format(opt_thr, obj_values[opt_thr]))
ax.set_title('E={:.4f}, C={:.4f}, C_0={:.4f}'.format(eff_complt_thr[opt_thr]['eff'],
                                                     eff_complt_thr[opt_thr]['complt'],
                                                     complt_target))
ax.legend()
ax.set_ylabel('Objective value')
ax.set_xlabel('Threshold')
ax.set_xlim([0, 1])
ax.set_ylim(bottom=0)
ax.grid(True)
ax.set_xticks(np.linspace(0, 1, 11, endpoint=True))
f.savefig(os.path.join(rankingDir, 'obj_function_grid{}.png'.format(num_thresholds)))

rankingTblTest = pd.read_csv(os.path.join(rankingDir, 'ensemble_ranked_predictions_testset'))

pred_class_test = (rankingTblTest['score'].values >= opt_thr).astype('int')

metrics_dict = classification_report(rankingTblTest['label'].values, pred_class_test, output_dict=True,
                                     target_names=['FA', 'PC'])

print(classification_report(rankingTblTest['label'].values, pred_class_test, output_dict=False,
                            target_names=['FA', 'PC']))
