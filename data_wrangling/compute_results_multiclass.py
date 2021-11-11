""" Perform some analysis on the results from multi-class task conducted by Hongbo. """

# 3rd party
from pathlib import Path
from sklearn.metrics import classification_report, multilabel_confusion_matrix, confusion_matrix, ConfusionMatrixDisplay
import pandas as pd

ranking_tbl_fp = Path('/Users/msaragoc/Downloads/ensemble_ranked_predictions_multiclass_PC_AFP_NTP_10-24-2021.csv')
ranking_tbl = pd.read_csv(ranking_tbl_fp)
target_names = ['NTP', 'AFP', 'PC']
a = classification_report(ranking_tbl['label'], ranking_tbl['predicted class'], target_names=target_names)

print(a)

# b = multilabel_confusion_matrix(ranking_tbl['label'], ranking_tbl['predicted class'])
cmat = confusion_matrix(ranking_tbl['label'], ranking_tbl['predicted class'])

# plot confusion matrix and save it
cmat_disp = ConfusionMatrixDisplay(cmat, display_labels=target_names)
cmat_fig = cmat_disp.plot()
