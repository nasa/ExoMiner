"""
Create yaml CV iteration files for CV experiments.

1) Normalize CV dataset to prepare it to be used for a CV experiment.
2) Create CV iterations yaml for the CV experiment.
"""

# 3rd party
import yaml
from pathlib import Path

#%% Create yaml file to be used to create the normalized labeled dataset for the CV experiment from the nonnormalized dataset

data_dir = Path('/nobackupp19/msaragoc/work_dir/Kepler-TESS_exoplanet/data/tfrecords/TESS/tfrecords_tess_spoc_2min_s1-s67_9-24-2024_1159_data/cv_tfrecords_tess_spoc_2min_s1-s67_tcedikcorat_crowdsap_tcedikcocorr_11-23-2024_0047/tfrecords/eval')
datasets = ['train', 'test']

cv_folds_fps = [fp for fp in data_dir.iterdir() if fp.name.startswith('shard-')]

cv_iters = []  # aggregate CV iterations (each is a dictionary that maps to 'train', 'val', and 'test' sets)
for fp in cv_folds_fps:

    cv_iter = {dataset: None for dataset in datasets}  # 'val', 'test']}

    cv_iter['test'] = [fp]  # each CV fold shows up as test once; number of folds determines number of CV iterations

    no_test_fps = [fp_n for fp_n in cv_folds_fps if fp_n != fp]

    cv_iter['train'] = no_test_fps  # no_test_fps[:-1]
    # cv_iter['val'] = [no_test_fps[-1]]  # get one of the training folds as validation fold

    cv_iters.append(cv_iter)

with open(data_dir / 'cv_iterations.yaml', 'w') as file:
    yaml.dump(cv_iters, file, sort_keys=False)

#%% Create yaml file to be used to run the CV experiment with the normalized labeled dataset

data_dir = Path('/nobackupp19/msaragoc/work_dir/Kepler-TESS_exoplanet/data/tfrecords/TESS/tfrecords_tess_spoc_2min_s1-s67_9-24-2024_1159_data/cv_tfrecords_tess_spoc_2min_s1-s67_tcedikcorat_crowdsap_tcedikcocorr_11-23-2024_0047/tfrecords/eval_normalized_with_kepler_trainset')
# use yaml file for CV iterations created when normalizing the data
cv_iterations_fp = Path('/nobackupp19/msaragoc/work_dir/Kepler-TESS_exoplanet/data/tfrecords/TESS/tfrecords_tess_spoc_2min_s1-s67_9-24-2024_1159_data/cv_tfrecords_tess_spoc_2min_s1-s67_tcedikcorat_crowdsap_tcedikcocorr_11-23-2024_0047/tfrecords/eval_normalized_with_kepler_trainset/src_cv_iterations.yaml')

with open(cv_iterations_fp, 'r') as file:
    cv_iterations = yaml.unsafe_load(file)

cv_iters = []  # aggregate CV iterations (each is a dictionary that maps to 'train', 'val', and 'test' sets)
for cv_iter_i, cv_iter in enumerate(cv_iterations):

    cv_iter = {dataset: [data_dir / f'cv_iter_{cv_iter_i}/norm_data' / dataset_fp.name for dataset_fp in dataset_fps]
               for dataset, dataset_fps in cv_iter.items()}

    cv_iters.append(cv_iter)

with open(data_dir / 'cv_iterations.yaml', 'w') as file:
    yaml.dump(cv_iters, file, sort_keys=False)

#%% Create yaml file to be used to run the CV trained models on a predict dataset

data_dir = Path('/Users/msaragoc/Projects/exoplanet_transit_classification/data/tfrecords/tess/tess_spoc_ffi/cv_tfrecords_tess_spoc_ffi_s36-s69_7-24-2024_1610_predict/tfrecords_normalized')
n_cv_iterations = 5

data_fps = [fp for fp in data_dir.iterdir() if fp.name.startswith('shard-')]
cv_iters = [{'predict': data_fps} for cv_i in range(n_cv_iterations)]

with open(data_dir / 'cv_iterations.yaml', 'w') as file:
    yaml.dump(cv_iters, file, sort_keys=False)

#%% Create new yaml file from another one by including Kepler data in the training set

src_cv_iterations_fp = Path('/nobackupp19/msaragoc/work_dir/Kepler-TESS_exoplanet/data/tfrecords/TESS/tfrecords_tess_spoc_2min_s1-s67_9-24-2024_1159_data/cv_tfrecords_tess_spoc_2min_s1-s67_tcedikcorat_crowdsap_tcedikcocorr_11-23-2024_0047/tfrecords/eval/cv_iterations.yaml')
kepler_src_dir = Path('/nobackupp19/msaragoc/work_dir/Kepler-TESS_exoplanet/data/tfrecords/Kepler/Q1-Q17_DR25/tfrecords_kepler_q1q17dr25_9-30-2024_1730_data/cv_tfrecords_kepler_spoc_q1q17dr25_10-3-2024_1227_tcedikcorat/tfrecords/eval')
dest_dir = Path('/nobackupp19/msaragoc/work_dir/Kepler-TESS_exoplanet/data/tfrecords/TESS/tfrecords_tess_spoc_2min_s1-s67_9-24-2024_1159_data/cv_tfrecords_tess_spoc_2min_s1-s67_tcedikcorat_crowdsap_tcedikcocorr_11-23-2024_0047/tfrecords/eval_normalized_with_kepler_trainset')

dest_dir.mkdir(exist_ok=True)

kepler_src_fps = [fp for fp in kepler_src_dir.iterdir() if fp.name.startswith('shard-')]

with open(src_cv_iterations_fp, 'r') as file:
    cv_iterations = yaml.unsafe_load(file)

new_cv_iters = []  # aggregate CV iterations (each is a dictionary that maps to 'train', 'val', and 'test' sets)
for cv_iter_i, cv_iter in enumerate(cv_iterations):

    cv_iter['train'] += kepler_src_fps

    new_cv_iters.append(cv_iter)

with open(dest_dir / 'src_cv_iterations.yaml', 'w') as file:
    yaml.dump(new_cv_iters, file, sort_keys=False)

#%% Create new yaml file from another one by adding only Kepler data to the training set

src_cv_iterations_fp = Path('/home6/msaragoc/work_dir/Kepler-TESS_exoplanet/data/tfrecords/TESS/tfrecords_tess_spoc_2min_s1-s67_9-24-2024_1159_data/cv_tfrecords_tess_spoc_2min_s1-s67_9-24-2024_1159/tfrecords/eval/cv_iterations.yaml')
kepler_src_dir = Path('/nobackupp19/msaragoc/work_dir/Kepler-TESS_exoplanet/data/tfrecords/Kepler/Q1-Q17_DR25/tfrecords_kepler_q1q17dr25_9-30-2024_1730_data/cv_tfrecords_kepler_spoc_q1q17dr25_10-3-2024_1227/tfrecords/eval')
dest_dir = Path('/nobackupp19/msaragoc/work_dir/Kepler-TESS_exoplanet/data/tfrecords/TESS/tfrecords_tess_spoc_2min_s1-s67_9-24-2024_1159_data/cv_tfrecords_tess_spoc_2min_s1-s67_tcedikcorat_crowdsap_tcedikcocorr_11-23-2024_0047/tfrecords/eval_/eval_normalized_with_kepler_trainset')

dest_dir.mkdir(exist_ok=True)

kepler_src_fps = [fp for fp in kepler_src_dir.iterdir() if fp.name.startswith('shard-')]

with open(src_cv_iterations_fp, 'r') as file:
    cv_iterations = yaml.unsafe_load(file)

new_cv_iters = []  # aggregate CV iterations (each is a dictionary that maps to 'train', 'val', and 'test' sets)
for cv_iter_i, cv_iter in enumerate(cv_iterations):

    cv_iter['val'] = cv_iter['train']
    cv_iter['train'] = kepler_src_fps

    new_cv_iters.append(cv_iter)

with open(dest_dir / 'src_cv_iterations.yaml', 'w') as file:
    yaml.dump(new_cv_iters, file, sort_keys=False)
