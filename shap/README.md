# SHAP

### Goal

Conduct SHAP experiments for `ExoMiner`. SHAP is one of the standard and most popular methods in Machine Learning
explainability. The algorithm can be computionally expensive if the set of features becomes large, since it requires to
run all possible combinations of features. The original paper "A Unified Approach to Interpreting Model Predictions"
by Lundberg and Lee (2017) can be found [here](https://arxiv.org/abs/1705.07874).

### Method's Outline

1. Create configuration files for each run: in order to computer the SHAP values for each feature, one needs to train
   and evaluate `ExoMiner` on all the possible combinations of features (runs). Using `create_config_files.py` and
   default configuration files for training an ensemble of models and for evaluating and running inference using that
   same ensemble, the config files needed for each run are created. The set of features to be evaluated is defined in
   `create_config_files.py`.
2. Train and evaluate the ensemble for all those runs and run them on inference to obtain the scores for the examples in
   the dataset. This process can be set to run automatically and sequentially on the high-end computing cluster by using
   the scripts `run_shap_configs_seq.sh`, `Submitjob_shap_single_train.pbs` and `Submitjob_shap_single_pred.pbs` in
   `job_scripts`.
3. Create examples and runs tables using `utils_shap.py`. The examples table contains all examples in the dataset and
   informative details on them such as ID, ephemerides, label.
4. Compute the marginal contributions for each feature in each run using `compute_mcs.py`.
5. Add marginal contributions for each feature across the different runs to compute the SHAP values for each example.
   This uses `utils_shap.py`.

[//]: # (### Set of Features &#40;as of 1-27-2022&#41;)

[//]: # (The experiments were conducted with a set of 7 groups of features, which yields 2**7=128 combinations &#40;runs&#41;. **The )

[//]: # (scores for the examples in the null run &#40;i.e., no features&#41; is estimated as the mean score based on the label )

[//]: # (distribution in the training set**. The groups of features are the following:)

[//]: # (- Global flux view)

[//]: # (- Local flux view + transit depth)

[//]: # (- Local flux odd view + local flux even view + odd SE OOT + even SE OOT)

[//]: # (- Global centroid view + local centroid view + cat magnitude + FWM stat + diff. img. CO from OOT/target + diff. img. )

[//]: # (CO uncertainties)

[//]: # (- Local weak secondary flux view + albedo stat + secondary MES + planet eff. temp. stat + secondary transit depth)

[//]: # (- Stellar parameters)

[//]: # (- Ghost diagnostics + bootstrap FA prob. + rolling band lvl zero + period + planet radius)
