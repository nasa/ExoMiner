""" Study distribution of labels in explanations dataset. """

# 3rd party
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

#%% read CSV with explanations labels

plot_dir = Path('/Users/msaragoc/Projects/exoplanet_transit_classification/experiments/xai/study_explanations_labels_9-3-2025_1018')
plot_dir.mkdir(parents=True, exist_ok=True)

xai_df = pd.read_csv('/Users/msaragoc/Projects/exoplanet_transit_classification/experiments/xai/tess-spoc-2min_tces_dv_s1-s88_3-27-2025_1316_explanations_labels_9-2-2025_1658.csv')

#%% plot count of TCEs for each explanation branch label

branch_cols = [col for col in xai_df.columns if col.endswith('_branch_label')]

f = plt.figure(figsize=(12, 8))
ax = xai_df[branch_cols].sum().plot(kind='bar')
ax.set_ylabel('TCE Count')
ax.set_xlabel('Explanation Branch')
ax.set_xticklabels(branch_cols, rotation=45, ha='right')    
# ax.set_yscale('log')
f.tight_layout()
f.savefig(plot_dir / 'hist_explanation_branch_labels.png')

# break it down per disposition
for label in xai_df['label'].unique():
    f = plt.figure(figsize=(12, 8))
    ax = xai_df.loc[xai_df['label'] == label, branch_cols].sum().plot(kind='bar')
    ax.set_ylabel('TCE Count')
    ax.set_xlabel('Explanation Branch')
    ax.set_title(f'TCEs with label: {label}')
    ax.set_xticklabels(branch_cols, rotation=45, ha='right')    
    # ax.set_yscale('log')
    f.tight_layout()
    f.savefig(plot_dir / f'hist_explanation_branch_labels_disp{label}.png')
    
# %%
