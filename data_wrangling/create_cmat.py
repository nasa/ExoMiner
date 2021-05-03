""" Generate a confusion matrix based on a given set of dispositions for TESS data. """

# 3rd party
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import pandas as pd
from pathlib import Path

# filepath to ranking table produced by a model
ranking_tbl_fp = Path('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Kepler_planet_finder/results_ensemble/tess-dv_g301-l31_5tr_spline_nongapped_norobovetterkois_starshuffle_configK_allparameterstess_nocentroid_nodicco_nosigmaoe_nodepth_noprad_noperiod_4-28-2021/ensemble_ranked_predictions_predictset.csv')
ranking_tbl = pd.read_csv(ranking_tbl_fp)

# column used to get the dispositions
disp_col = 'original_label'

# how to deal with NaN or 'None' dispositions
ranking_tbl.loc[(ranking_tbl[disp_col].isna()) | (ranking_tbl[disp_col] == 'None')] = 'N/A'


def _map_tess_disp_to_label_id(label, label_map):
    """ Map TESS disposition to label id encoded in the label_map dictionary.

    :param label: ground truth label. Must be a key in label_map
    :param label_map: dict, maps from label to label id
    :return:
        int, ground truth label mapped to label id
    """
    return label_map[label]


# dictionary that maps ground truth label to label id (many to 3: PC (1), non-PC (0), missing label (-1))
label_map = {
    'PC': 1,
    'KP': 1,
    'EB': 0,
    'CP': 1,
    'O': 0,
    'N/A': -1,  # class for missing disposition
}

# only when using TESS disposition as ground truth;
# map dispositions to label ids (1 for PC, 0 for non-PC, -1 for missing label)
# ranking_tbl['label'] = ranking_tbl['TESS Disposition'].apply(_map_tess_disp_to_label_id, axis=1, label_map=label_map)


def _map_pred_class_to_label(row, disp_col):
    """ Map predicted label id to label.

    :param row: Pandas time series, contains 'label' field with ground truth label id, 'predicted class' with predicted
    label id, and `disp_col` with ground truth label
    :param disp_col: str, name of field with ground truth label
    :return:
        int, predicted label id mapped to label
    """

    if row['label'] == row['predicted class']:
        return row[disp_col]
    else:
        return 'Misclf'

# map predicted label id to predicted label
ranking_tbl['predicted_label'] = \
    ranking_tbl[['label', disp_col, 'predicted class']].apply(_map_pred_class_to_label, axis=1, args=(disp_col,))

# compute confusion matrix for each disposition
cmat = confusion_matrix(ranking_tbl[disp_col].values, ranking_tbl['predicted_label'].values)

# get list of classes
disp_classes = sorted(np.append(ranking_tbl[disp_col].unique(), 'Misclf'))

# plot confusion matrix and save it
cmat_disp = ConfusionMatrixDisplay(cmat, display_labels=disp_classes)
cmat_fig = cmat_disp.plot()
cmat_fig.figure_.savefig(ranking_tbl_fp.parent / f'cmat_{disp_col}.pdf')

#%% compute accuracy per class



