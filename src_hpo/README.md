# Hyperparameter Optimization

### Goals

The code in `src_hpo` is used for running hyperparameter optimization (HPO) experiments to 
find optimized architectures for a given classification task, features, and data set.

### Methodology

The HPO module uses the Python module `hpbandster` [1], which consists of an implementation in 
Python of the **Bayesian-Hyperband (BOHB)**, **Bayesian-only**, and **Random Search** algorithms.

We wish to train multiple models (with different configurations) on a given number of epochs (budget per model). 
However, we have limited compute and time resources, so BOHB explores the configuration space in a 'smarter' way than 
random or grid search. The Bayesian optimizer takes into account past sampled configurations to inform the selection of 
candidate configurations to be evaluated. Furthermore, Hyperband evluates configurations in smaller budgets and chooses 
those that are in the top to be evaluated on larger budgets, since there's no point in evaluating configurations using
the maximum budget if at smaller budgets they already are worse than the rest of the sampled population - this of course 
assumes some smoothness in the performance from one budget to the other; The spearman correlation plot shows whether 
that assumption holds.

### Code

1. `run_hpo_worker.sh`: bash script used to start and control each worker.
2. `run_hpo.py`: main Python script. All workers call this Python application.
3. `worker_hpo.py`: worker implementation.
4. `utils_hpo.py`: utility functions.
5. `config_hpo.yaml`: configuration parameters for the HPO experiment.
6. `Submit_hpo_gpus.pbs`: PBS script used to run HPO on a multi-GPU and multi-node setup in the HECC cluster.
7. `hparamopt_res.py`: script used to analyze results from an HPO run in postprocessing.


### Results

The main results for each HPO experiment are two json files (`configs.json` and `results.json`). The `configs.json` file 
logs all the configurations that were sampled for the experiment. Invalid configurations, i.e., that cannot be built or 
had an error are assigned a loss of infinity. The `results.json` file logs the results and performance of the sampled 
configurations evaluated on the different budgets.

### References

[1] Falkner, Stefan, Aaron Klein, and Frank Hutter. "BOHB: Robust and efficient hyperparameter optimization at scale."
arXiv preprint arXiv:1807.01774 (2018).