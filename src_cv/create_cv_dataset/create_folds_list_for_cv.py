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


# with open('/home6/msaragoc/work_dir/Kepler-TESS_exoplanet/data/tfrecords/Combined/cv_tess_s1-s67_kepler_q1q17dr25_2-2-2023_1308/cv_folds_runs_withval_trainkepler.yaml', 'r') as yml_file:
#     cv_folds = yaml.unsafe_load(yml_file)
#
# cv_folds_fns = []
# for cv_fold in cv_folds:
#     cv_fold_iter = {dataset: np.array([fp.name for fp in fps]) for dataset, fps in cv_fold.items()}
#     cv_folds_fns.append(cv_fold_iter)
#
# np.save('/home6/msaragoc/work_dir/Kepler-TESS_exoplanet/data/tfrecords/Combined/cv_tess_s1-s67_kepler_q1q17dr25_2-2-2023_1308/cv_folds_runs_withval_trainkepler.npy', cv_folds_fns)

#%% Create CV iterations yaml and numpy files

dest_dir = Path('/home6/msaragoc/work_dir/Kepler-TESS_exoplanet/data/tfrecords/Kepler/Q1-Q17_DR25/cv_kepler_q1q17dr25_obs_11-22-2023_2356/tfrecords/eval_normalized_with_tce_fwm_stat')
dest_dir.mkdir(exist_ok=True)
cv_tfrec_dir = Path('/home6/msaragoc/work_dir/Kepler-TESS_exoplanet/data/tfrecords/Kepler/Q1-Q17_DR25/cv_kepler_q1q17dr25_obs_11-22-2023_2356/tfrecords/eval')
cv_folds_fps = np.array(list(cv_tfrec_dir.iterdir()))  # file paths to CV folds
n_cv_iterations = len(cv_folds_fps)  # total number of CV iterations
datasets = ['train', 'test', 'val']
n_val_folds = 1  # number of folds in validation set for any given CV iteration
cv_iterations_lst = []
for cv_i in range(n_cv_iterations):

    cv_iteration_dict = {dataset: [] for dataset in datasets}

    # get test fold
    mask = np.ones(cv_folds_fps.shape, bool)
    cv_iteration_dict['test'] = [cv_folds_fps[cv_i]]

    # get train+val folds
    mask[cv_i] = False
    cv_train_folds_fps = cv_folds_fps[mask]
    assert len(cv_train_folds_fps) > n_val_folds

    # choose randomly validation folds and assign folds to training and validation sets
    mask = np.ones(cv_train_folds_fps.shape, bool)
    idx_fold_val = np.random.choice(np.arange(len(cv_train_folds_fps)), n_val_folds)
    mask[idx_fold_val] = False
    cv_iteration_dict['train'] = list(cv_train_folds_fps[mask])
    cv_iteration_dict['val'] = list(cv_train_folds_fps[~mask])

    cv_iterations_lst.append(cv_iteration_dict)

np.save(dest_dir / 'cv_folds_runs.npy', cv_iterations_lst)

with open(dest_dir / 'cv_folds_runs.yaml', 'w') as yml_file:

    yaml.dump(cv_iterations_lst, yml_file)

#%%

cv_folds_fns = [{k: np.array([cv_fold_fp.name for cv_fold_fp in v]) for k, v in cv_fold_dict.items()} for cv_fold_dict in cv_iterations_lst]
np.save(dest_dir / 'cv_folds_fns_runs.npy', cv_folds_fns)

#%% Create yaml file for normalized CV iterations

cv_folds_fns = np.load('/home6/msaragoc/work_dir/Kepler-TESS_exoplanet/data/tfrecords/Kepler/Q1-Q17_DR25/tfrecords_kepler_q1q17dr25_obsplanets_siminj1_2-22-2024_1115/cv_keplerq1q17dr25_obsplanets_siminj1_data_02-22-2024_1155/cv_folds_runs.npy', allow_pickle=True)
cv_normalized_run_dir = Path('/home6/msaragoc/work_dir/Kepler-TESS_exoplanet/data/tfrecords/Kepler/Q1-Q17_DR25/tfrecords_kepler_q1q17dr25_obsplanets_siminj1_2-22-2024_1115/cv_keplerq1q17dr25_obsplanets_siminj1_data_02-22-2024_1155/eval_normalized')
cv_iter_fps_lst = []
for cv_iter_i, cv_folds_iter_i in enumerate(cv_folds_fns):

    cv_folds_iter_i_fps = {dataset: [cv_normalized_run_dir / f'cv_iter_{cv_iter_i}' / 'norm_data' / dataset_fp.name for dataset_fp in dataset_fps] for dataset, dataset_fps in cv_folds_iter_i.items()}
    cv_iter_fps_lst.append(cv_folds_iter_i_fps)

with open(cv_normalized_run_dir / 'cv_iters_fps.yaml', 'w') as file:
    yaml.dump(cv_iter_fps_lst, file, sort_keys=False)
