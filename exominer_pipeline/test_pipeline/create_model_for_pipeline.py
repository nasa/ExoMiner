"""Create model for ExoMiner Pipeline from existing models."""

# 3rd party
from pathlib import Path
import  yaml

# local
from models.create_ensemble_avg_model import create_avg_ensemble_model
from src.utils.utils_dataio import set_tf_data_type_for_features

#%% set paths

cv_exp_dir = Path('/data3/exoplnt_dl/experiments/tess_2min_paper/cv_tess-spoc-2min_s1-s67_kepler_trainset_tcenumtransits_tcenumtransitsobs_1-12-2025_1036')
model_save_fp = '/data3/exoplnt_dl/codebase/exominer_pipeline/data/exominer-plusplus_cv-mean-ensemble_tess-spoc-2min-s1s67_tess-kepler.keras'

#%% get model fps

models_fps = list(cv_exp_dir.rglob('*ensemble_avg_model.keras'))
print(f'Found {len(models_fps)} models in {cv_exp_dir}')

#%% get features set

cv_config_fp = cv_exp_dir / 'cv_iter_0' / 'models' / 'model0' / 'config_cv.yaml'
with open(cv_config_fp, 'r') as file:
    cv_config = yaml.unsafe_load(file)

features_set = set_tf_data_type_for_features(cv_config['features_set'])

#%% create model from ensemble

print('Creating ensemble model...')
create_avg_ensemble_model(models_fps, features_set, model_save_fp)

print('Done.')
