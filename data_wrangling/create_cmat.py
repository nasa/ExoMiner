import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import pandas as pd
from pathlib import Path


ranking_tbl_fp = Path('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Kepler_planet_finder/results_ensemble/tess-dv_g301-l31_5tr_spline_nongapped_norobovetterkois_starshuffle_configK_onlytimeseries_stellar_4-26-2021/ensemble_ranked_predictions_predictset.csv')
ranking_tbl = pd.read_csv(ranking_tbl_fp)

disp_col = 'original_label'

ranking_tbl.loc[(ranking_tbl[disp_col].isna()) | (ranking_tbl[disp_col] == 'None')] = 'N/A'


def _map_tess_disp_to_label_id(label, **kwargs):
    # print(kwargs)
    return kwargs['label_map'][label]


label_map = {
    'PC': 1,
    'KP': 1,
    'EB': 0,
    'CP': 1,
    'O': 0,
    'N/A': -1,
}

# only when using TESS disposition as ground truth
# ranking_tbl['label'] = ranking_tbl['TESS Disposition'].apply(_map_tess_disp_to_label_id, axis=1, label_map=label_map)


def _map_pred_class_to_label(row, disp_col='original_label'):

    if row['label'] == row['predicted class']:
        return row[disp_col]
    else:
        return 'Misclf'

ranking_tbl['predicted_label'] = ranking_tbl[['label', disp_col, 'predicted class']].apply(_map_pred_class_to_label, axis=1, args=(disp_col,))

cmat = confusion_matrix(ranking_tbl[disp_col].values, ranking_tbl['predicted_label'].values)

disp_classes = sorted(np.append(ranking_tbl[disp_col].unique(), 'Misclf'))
cmat_disp = ConfusionMatrixDisplay(cmat, display_labels=disp_classes)
cmat_fig = cmat_disp.plot()
cmat_fig.figure_.savefig(ranking_tbl_fp.parent / f'cmat_{disp_col}.pdf')

#%% compute accuracy per class



