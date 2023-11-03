"""
Create ensemble average model
"""

# 3rd party
from pathlib import Path
from keras.saving import load_model
import yaml
from tensorflow.keras import utils.cuscustom_object_scope

# local
from models.utils_models import create_ensemble
from models.models_keras import Time2Vec


# get models file paths
models_root_dir = Path('/Users/msaragoc/Projects/exoplanet_transit_classification/experiments/kepler_simulated_data_exominer/exominer_train_kepler_simulateddata_for_simulateddata_11-2-2023_1547/models')
models_fps = [fp / f'{fp.name}.keras' for fp in models_root_dir.iterdir() if fp.is_dir()]
# load models into a list
models = []
custom_objects = {"Time2Vec": Time2Vec}
with utils.custom_object_scope(custom_objects):
    for model_i, model_fp in enumerate(models_fps):
        model = load_model(filepath=model_fp, compile=False)
        model._name = f'model{model_i}'
        models.append(model)

# get features set
run_params_fp = Path('/Users/msaragoc/Projects/exoplanet_transit_classification/experiments/kepler_simulated_data_exominer/exominer_train_kepler_simulateddata_for_simulateddata_11-2-2023_1547/run_params.yaml')
with open(run_params_fp, 'r') as run_params_file:
    run_params = yaml.unsafe_load(run_params_file)
features = run_params['features_set']

# create ensemble average model
ensemble_avg_model_fp = Path('/Users/msaragoc/Projects/exoplanet_transit_classification/experiments/kepler_simulated_data_exominer/exominer_train_kepler_simulateddata_for_simulateddata_11-2-2023_1547/ensemble_avg_model.keras')
ensemble_avg_model = create_ensemble(features, models)
ensemble_avg_model.save(ensemble_avg_model_fp)

print('Created ensemble average model.')
