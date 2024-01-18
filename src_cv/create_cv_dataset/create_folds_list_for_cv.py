"""
Create yaml file with list of folds (tfrecord file paths) for each data set in each CV iteration
"""

# 3rd party
from pathlib import Path
import numpy as np
import yaml

#%% Set file paths

dest_dir = Path('/home6/msaragoc/work_dir/Kepler-TESS_exoplanet/data/tfrecords/Combined/cv_tess_s1-s67_kepler_q1q17dr25_12-06-2023_1700')
dest_dir.mkdir(exist_ok=True)

cv_folds_np_tess = np.load('/home6/msaragoc/work_dir/Kepler-TESS_exoplanet/data/tfrecords/TESS/cv_tess_s1-s67_12-06-2023_1147/cv_folds_runs_withval.npy', allow_pickle=True)

tess_folds_dir = Path('/home6/msaragoc/work_dir/Kepler-TESS_exoplanet/data/tfrecords/TESS/cv_tess_s1-s67_12-06-2023_1147/tfrecords/eval')
tess_folds_fps = [fp for fp in tess_folds_dir.iterdir()]

kepler_folds_dir = Path('/home6/msaragoc/work_dir/Kepler-TESS_exoplanet/data/tfrecords/Kepler/Q1-Q17_DR25/cv_kepler_q1q17dr25_obs_11-22-2023_2356/tfrecords/eval_kepler_suffix')
kepler_folds_fps = [fp for fp in kepler_folds_dir.iterdir()]

#%% Create CV folds with Kepler data as part of the training set

cv_folds_merged = []

for cv_fold in cv_folds_np_tess:

    curr_cv_fold = {dataset: [tess_folds_dir / fn for fn in dataset_folds]
                    for dataset, dataset_folds in cv_fold.items()}

    curr_cv_fold['train'] += kepler_folds_fps
    # curr_cv_fold['train'] = kepler_folds_fps

    cv_folds_merged.append(curr_cv_fold)

# np.save(dest_dir / 'cv_folds_runs_withval_trainkepler.yaml', cv_folds_merged)

with open(dest_dir / 'cv_folds_runs_withval_trainkepler.yaml', 'w') as yml_file:
# with open(dest_dir / 'cv_folds_runs_withval.yaml', 'w') as yml_file:

    yaml.dump(cv_folds_merged, yml_file)

#%% Create CV folds NumPy file for a given CV run

# cv_run_dir = Path('/home6/msaragoc/work_dir/Kepler-TESS_exoplanet/data/tfrecords/Combined/cv_tess_s1-s67_kepler_q1q17dr25_12-06-2023_1700/tfrecords/eval_normalized_trainonlykepler')
# # cv_iter_dir_fps = [fp for fp in cv_run_dir.iterdir() if fp.is_dir()]
# 
# kepler_cv_folds = [f'shard-000{fold_i}_kepler' for fold_i in range(10)]
# 
# cv_folds_np_tess = np.load('/home6/msaragoc/work_dir/Kepler-TESS_exoplanet/data/tfrecords/TESS/cv_tess_s1-s67_12-06-2023_1147/cv_folds_runs_withval.npy', allow_pickle=True)
# 
# cv_folds = []
# 
# for cv_i in range(10):
# 
#     cv_folds.append(cv_folds_np_tess[cv_i])
#     # print(cv_folds[-1])
#     # cv_folds[-1]['train'] = list(cv_folds[-1]['train']) + kepler_cv_folds
#     cv_folds[-1]['train'] = kepler_cv_folds
# 
# np.save(cv_run_dir / 'cv_folds_runs_withval_trainonlykepler_normalized.npy', cv_folds)
# 
# print(cv_folds)
