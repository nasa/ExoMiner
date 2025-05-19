# Module *models* 

### Goals

The code in `models` is used for storing and testing designed architectures.

### Code

- [`models_keras.py`](models_keras.py): main script. Contains the designed architectures of ExoMiner.
- [`test_model_architecture.py`](test_model_architecture.py): script used to load model instance of a chosen 
architecture in [`models_keras.py`](models_keras.py) and generate model summary and architecture image.
- [`utils_models.py`](utils_models.py): utility functions required for model creation.
- [`create_ensemble_avg_model.py`](create_ensemble_avg_model.py): script used to create an ensemble average model from 
a set of pre-existing models.
