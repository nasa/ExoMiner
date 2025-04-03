"""
Analyze single predictions
"""

# 3rd party
import yaml
import ydf
import pandas as pd
from pathlib import Path

# local
from tess_spoc_ffi.decision_tree.train_model import create_dataset

#%% when ydf model uses pandas dataframe

model = ydf.load_model('/nobackup/msaragoc/work_dir/Kepler-TESS_exoplanet/experiments/tess_spoc_ffi/cv_tess-spoc-ffi_s36-s72_multisector_s56-s69_with2mindata_gbt_allscalarfeatures_ffi_vs_2min_3-25-2025_1134/cv_iter_5/models/model0/model')

with open('/nobackup/msaragoc/work_dir/Kepler-TESS_exoplanet/experiments/tess_spoc_ffi/cv_tess-spoc-ffi_s36-s72_multisector_s56-s69_with2mindata_gbt_allscalarfeatures_ffi_vs_2min_3-25-2025_1134/cv_iter_5/models/model0/config_cv.yaml', 'r') as config_file:
    config_dict = yaml.unsafe_load(config_file)

ds = pd.read_csv('/nobackup/msaragoc/work_dir/Kepler-TESS_exoplanet/experiments/tess_spoc_ffi/cv_tess-spoc-ffi_s36-s72_multisector_s56-s69_with2mindata_exominer_newarchitecture_allfeatures_globalmaxpoolingextractedfeatures_ffi_vs_2min_3-25-2025_1810/cv_iter_5/models_ffi/extracted_learned_features_endbranches_after_prelu_test.csv')

a = model.analyze_prediction(ds[config_dict['features_set']].iloc[:1])
a.to_file(f'/nobackup/msaragoc/work_dir/Kepler-TESS_exoplanet/experiments/tess_spoc_ffi/cv_tess-spoc-ffi_s36-s72_multisector_s56-s69_with2mindata_gbt_extractedfeaturesconvbranchesafterprelu_ffi_vs_2min_3-31-2025_1657/cv_iter_5/analysis_{ds.iloc[0].uid}_{ds.iloc[0].label}.html')


#%% when ydf model uses tfrecord dataset

model = ydf.load_model('/nobackup/msaragoc/work_dir/Kepler-TESS_exoplanet/experiments/tess_spoc_ffi/cv_tess-spoc-ffi_s36-s72_multisector_s56-s69_with2mindata_gbt_allscalarfeatures_ffi_vs_2min_3-25-2025_1134/cv_iter_5/models/model0/model')

with open('/nobackup/msaragoc/work_dir/Kepler-TESS_exoplanet/experiments/tess_spoc_ffi/cv_tess-spoc-ffi_s36-s72_multisector_s56-s69_with2mindata_gbt_allscalarfeatures_ffi_vs_2min_3-25-2025_1134/cv_iter_5/models/model0/config_cv.yaml', 'r') as config_file:
    config_dict = yaml.unsafe_load(config_file)

config_dict['dataset_type'] = 'tfrecord'

ds = create_dataset(config_dict)

ex = ds['test'].unbatch().take(1)
for example in ex:
    ex_np = {key: value.numpy() for key, value in example.items()}
ex_df = pd.DataFrame([ex_np])
a = model.analyze_prediction(ex_df)

a.to_file(f'/nobackup/msaragoc/work_dir/Kepler-TESS_exoplanet/experiments/tess_spoc_ffi/cv_tess-spoc-ffi_s36-s72_multisector_s56-s69_with2mindata_gbt_allscalarfeatures_ffi_vs_2min_3-25-2025_1134/cv_iter_5/analysis_example.html')
