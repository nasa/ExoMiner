"""
Create CV dataset folds yaml file based on a source CV dataset folds yaml file. This is useful when running a new CV
experiment using a CV dataset that was built in a different system/directory than the current copy.
"""

# 3rd party
from pathlib import Path
import yaml
import re

src_fp = Path('/Users/msaragoc/Projects/exoplanet_transit_classification/data/tfrecords/tess/tess_spoc_2min_paper/cv_tfrecords_tess_spoc_2min_s1-s67_tcedikcorat_crowdsap_tcedikcocorr_11-23-2024_0047/tfrecords/eval_normalized/cv_iterations.yaml')
dest_fp = Path('/Users/msaragoc/Projects/exoplanet_transit_classification/data/tfrecords/tess/tess_spoc_2min_paper/cv_tfrecords_tess_spoc_2min_s1-s67_tcedikcorat_crowdsap_tcedikcocorr_11-23-2024_0047/tfrecords/eval_normalized/cv_iterations_local.yaml')
dest_root_dir = Path('/Users/msaragoc/Projects/exoplanet_transit_classification/data/tfrecords/tess/tess_spoc_2min_paper/cv_tfrecords_tess_spoc_2min_s1-s67_tcedikcorat_crowdsap_tcedikcocorr_11-23-2024_0047/tfrecords/eval_normalized/')

with open(src_fp, 'r') as file:
    cv_config = yaml.unsafe_load(file)

new_cv_config = []
for cv_iter in cv_config:

    new_cv_iter = {dataset: [dest_root_dir / str(dataset_fp)[re.search(r'cv_iter_[0-9]',
                                                                                str(dataset_fp)).start():]
                             for dataset_fp in dataset_fps] for dataset, dataset_fps in cv_iter.items()}
    new_cv_config.append(new_cv_iter)

with open(dest_fp, 'w') as config_file:
    yaml.dump(new_cv_config, config_file)
