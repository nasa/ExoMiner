"""
Create yaml file with list of folds (tfrecord file paths) for each data set in each CV iteration
"""

# 3rd party
from pathlib import Path
import numpy as np
import yaml

#%% Set file paths

dest_dir = Path('/home6/msaragoc/work_dir/Kepler-TESS_exoplanet/data/tfrecords/Combined/cv_tess_s1-s67_kepler_q1q17dr25_targets_maxsectors_2-5-2023_1250')
dest_dir.mkdir(exist_ok=True)

cv_folds_np_tess = np.load('/home6/msaragoc/work_dir/Kepler-TESS_exoplanet/data/tfrecords/TESS/cv_tess_s1-s67_12-06-2023_1147/cv_folds_runs_withval.npy', allow_pickle=True)

tess_folds_dir = Path('/home6/msaragoc/work_dir/Kepler-TESS_exoplanet/data/tfrecords/TESS/cv_tess_s1-s67_updated_labels_02-02-2024_1210/tfrecords/eval_targets_maxsectors')
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


with open('/home6/msaragoc/work_dir/Kepler-TESS_exoplanet/data/tfrecords/Combined/cv_tess_s1-s67_kepler_q1q17dr25_2-2-2023_1308/cv_folds_runs_withval_trainkepler.yaml', 'r') as yml_file:
    cv_folds = yaml.unsafe_load(yml_file)

cv_folds_fns = []
for cv_fold in cv_folds:
    cv_fold_iter = {dataset: np.array([fp.name for fp in fps]) for dataset, fps in cv_fold.items()}
    cv_folds_fns.append(cv_fold_iter)

np.save('/home6/msaragoc/work_dir/Kepler-TESS_exoplanet/data/tfrecords/Combined/cv_tess_s1-s67_kepler_q1q17dr25_2-2-2023_1308/cv_folds_runs_withval_trainkepler.npy', cv_folds_fns)
