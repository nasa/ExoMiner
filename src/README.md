# Module *src* 

### Goals

The code in `src` is used for training and evaluating models, and conducting inference with trained models on a given 
TFRecord dataset (e.g., a single train-val-test dataset split; run inference on predict set).

### Code

- [`train`](train): contains scripts related to training a model. The main scripts are 
[`train_model.py`](train/train_model.py) and the configuration file [`config_train.yaml`](train/config_train.yaml). The 
bash script [`run_train_sequential.py`](train/run_train_sequential.sh) can be used to train a set of models sequentially
   (e.g., train X models to create average ensemble to reduce bias due to random initialization). Finally, the script 
[`run_train_iter.sh`](train/run_train_iter.sh) perform the training and evaluation of a model, and then runs inference 
on the provided datasets.
- [`evaluate`](evaluate): contains scripts related to evaluation of a model on a dataset.
- [`predict`](predict): contains scripts relevant for applying a model on a dataset to generate predictions.
- [`postprocessing`](postprocessing): contains scripts to perform postprocessing steps after training a model or 
running inference.
- [`utils`](utils): utility functions that are shared by the other submodules.
- [`run_ensemble.sh`](run_ensemble.sh): script that evaluates an ensemble of models and runs inference on a dataset.
