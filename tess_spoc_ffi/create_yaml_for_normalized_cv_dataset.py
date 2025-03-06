"""
Uses the CV iteration yaml file that was used to create the normalized CV dataset, and creates a new CV iteration yaml
file for the normalized CV dataset.
"""

# 3rd party
import yaml
from pathlib import Path

#%%

src_yaml_fp = Path('/nobackupp19/msaragoc/work_dir/Kepler-TESS_exoplanet/data/tfrecords/TESS/tfrecords_tess_spoc_ffi_s36-s72_multisector_s56-s69_1-3-2025_1157_data/cv_tfrecords_tess_spoc_ffi_s36-s72_multisector_s56-s69_1-6-2025_1132/tfrecords/eval_with_2mindata_transferlearning/cv_iterations_twomin.yaml')
dest_cv_dir = Path('/nobackupp19/msaragoc/work_dir/Kepler-TESS_exoplanet/data/tfrecords/TESS/tfrecords_tess_spoc_ffi_s36-s72_multisector_s56-s69_1-3-2025_1157_data/cv_tfrecords_tess_spoc_ffi_s36-s72_multisector_s56-s69_1-6-2025_1132/tfrecords/eval_with_2mindata_transferlearning/cv_2min')
