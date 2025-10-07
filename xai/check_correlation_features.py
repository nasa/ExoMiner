"""Check correlation between features."""

#%% 3rd party
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_regression
import yaml
from itertools import combinations
import numpy as np

#%% prepare paths

exp_dir = Path('/Users/msaragoc/Projects/exoplanet_transit_classification/experiments/xai/test_deep-shap_clf-head-stellar-dvtce/')

corr_dir = exp_dir / 'correlation_analysis'
corr_dir.mkdir(exist_ok=True)

config_fp = Path('/Users/msaragoc/Projects/exoplanet_transit_classification/experiments/tess_spoc_2min/tess_paper/cv_tess-spoc-2min_s1-s67_kepler_trainset_tcenumtransits_tcenumtransitsobs_1-12-2025_1036/cv_iter_0/models/model0/config_cv.yaml')

# load config filepath
with open(config_fp, 'r') as config_yaml:
    config = yaml.unsafe_load(config_yaml)

#%% load features dataset

feat_df = pd.read_csv(exp_dir / 'conv_features_dataset.csv', comment='#')

#%% check correlation between features

feat_cols = [feat_name for feat_name in feat_df.columns if feat_name.startswith('fc_prelu_diff_img') or feat_name.startswith('fc_prelu_local_centroid')]

# Pearson correlation matrix
pearson_corr = feat_df[feat_cols].corr(method='pearson')

# Spearman correlation matrix
spearman_corr = feat_df[feat_cols].corr(method='spearman')

# Kendall correlation matrix
kendall_corr = feat_df[feat_cols].corr(method='kendall')

#%% plot correlation matrices

for corr_matrix, method in zip([pearson_corr, spearman_corr, kendall_corr],
                                ['Pearson', 'Spearman', 'Kendall']):
    f = plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.title(f'Feature Correlation ({method})')
    plt.show()
    f.savefig(corr_dir / f'feature_correlation_diffimg-centroidmotion_{method.lower()}.png', bbox_inches='tight')
    

# %% compute mutual information between features

ordered_feat_pairs = list(combinations(feat_cols, 2))

mi_feats_dict = {ordered_pair: np.nan for ordered_pair in ordered_feat_pairs}
for ordered_pair in ordered_feat_pairs:
    mi_feats_dict[ordered_pair] = mutual_info_regression(np.expand_dims(feat_df[ordered_pair[0]], axis=-1), feat_df[ordered_pair[1]], discrete_features=False, n_neighbors=3, random_state=42)
    