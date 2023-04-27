# Explainability

## Goal
Run different explainability methods on top of `ExoMiner` to come up a framework for 
explaining scores output by the model for subject matter experts. The methods currently implemented are:
- SHapley Additive exPlanations (SHAP) values
- Random Baseline
- Feature Occlusion (zeroing out)
- Model-based Positive Class Replacement

## Methods

### SHAP
Check `README.md` in `shap`.

### Feature Occlusion (zeroing out) 
In this method, `ExoMiner` is trained on the full set of features. In each run, a model is trained and evaluated with a 
feature (or group of features) zeroed-out. The process repeats until all groups of features were zeroed-out once. For a 
given group of features, if the score for the example decreases, then that group of features (in combination with 
the remaining features) contributed positively for the planet classification. The problem with this approach is that 
zeroing out a feature does not necessarily mean that it is not conveying any useful information for a given class, 
i.e., that it is neutral for the classification of the example.

Hyper-parameters
- Change in score threshold

Algorithm
1. For feature f_1, ..., f_L
      1. For example e_1, ..., e_K
         1. Replace feature f_l by zero array in example e_k.
         2. Use model to run inference on this modified example and produce an 'occlusion' score _occl_score_.
         3. Subtract from the new score the original one, i.e., delta_score = occl_score - original_score

Run script `run_exp_occlusion.py` to conduct a feature occlusion experiment. The configuration file 
`config_occlusion.yaml` defines the parameters for the run.

### Model-based Positive Class Replacement
This method is based on the concept of planet that the model learns from the training data. A set of examples highly 
scored by the model is chosen as representative prototypes of planets. For each one of these, groups of features are 
sequentially replaced by the corresponding features from the example which we are trying to explain. If such replacement
leads to a significant decrease in the score, then that group of features is determined as being in part responsible 
for the classification of that example as false positive. This method is based on the mindset of 'innocent until proven 
guilty', i.e., an example is considered a planet as long as it does not fail any diagnostic test (e.g., existence of 
significant centroid offset from the target star).

Since a set of representative examples needs to be chosen, a batch of trials is conducted. In each trial, a new set of 
examples is selected as prototypes. The results are then averaged over the trials.

Hyper-parameters
- Number of planet prototypes
- Score threshold for candidates to planet prototypes
- Number of trials
- Change in score threshold

Algorithm
1. For trial t_1, ..., t_N
   1. Choose M planet prototypes from examples with score > thr_M from the training set.
   2. For planet prototype p_1, ..., p_M 
      1. For feature f_1, ..., f_L
            1. For example e_1, ..., e_K
               1. Replace feature f_l in planet prototype p_m for corresponding feature in example e_k.
               2. Use model to run inference on this modified example, which generates a new score _replace_score_.
               3. Subtract from the new score the original one, i.e., delta_score = replace_score - original_score.

Run script `run_exp_replacing_pc.py` to conduct a model-based positive class replacement experiment. The configuration 
file `config_replacing_pc.yaml` defines the parameters for the run.

### Random Explanations
This method is a baseline method used to compare other explanability methods against random selection of explanations.

Hyper-parameters
- Number of trials
- Number of flags (also known as feature groups, branches) randomly chosen

Algorithm
1. For n_flags_active in n_branches:
   1. For trial t_1, ..., t_N
      1. Choose M flags prototypes to be set to 1.
      2. Compute precision, recall and other performance metrics.
   3. Compute mean of performance metrics across trials.

Run script `Random Analysis.ipynb` to conduct random explanation runs.

## Validation
