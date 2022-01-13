"""
Aggregate performance metrics from different experiments into the same csv file.
"""

# 3rd party
import pandas as pd
from pathlib import Path
import numpy as np

# directory that contains the experiments whose results are to be aggregated
exp_root_dir = Path('/Users/msaragoc/OneDrive - NASA/Projects/exoplanet_transit_classification/experiments/results_ensemble/')

exp_dirs = [fp for fp in exp_root_dir.iterdir() if fp.is_dir()]

exp_res_fps = []
for exp_dir in exp_dirs:
    exp_res_fp = [fp for fp in exp_dir.iterdir() if fp.name == 'results_ensemble.npy'][0]
    exp_res_fps.append(exp_res_fp)

data_to_tbl = {'experiment': [exp_res_fp.parent.name for exp_res_fp in exp_res_fps]}
for exp_res_fp in exp_res_fps:

    res = np.load(exp_res_fp, allow_pickle=True).item()
    res = {metric_name: metric_val for metric_name, metric_val in res.items() if isinstance(metric_val, float)}

    for metric_name, metric_val in res.items():
        if metric_name not in data_to_tbl:
            data_to_tbl[metric_name] = [metric_val]
        else:
            data_to_tbl[metric_name].append(metric_val)

data_df = pd.DataFrame(data_to_tbl)
data_df.to_csv(exp_root_dir / f'results_explainability_shap1_1-10-2022.csv', index=False)
