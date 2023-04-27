"""
Utility functions for running explainability tests for replacing model PCs method.
"""

# 3rd party
import numpy as np
import pandas as pd
import logging
from tensorflow.keras.models import load_model


def explain_branches_pcs(branches_to_block, all_data_dataset, all_data_train, model, best_example_ind):
    """ Run inference after replacing all features except those in the `branches_to_block` for each example in the
    dataset by the features from the representative PC example in the training set.

    :param branches_to_block: list, sets of features (branches) that are not replaced
    :param all_data_dataset: dict, features for all examples in the dataset. Key is a feature and the value is an array
    [n_examples x feature_dim]
    :param all_data_train: dict, features for all examples in the training set. Key is a feature and the value is an
    array [n_examples x feature_dim]
    :param model: Keras model, model used to run inference
    :param best_example_ind: int, index for representative PC in the training set
    :return:
        scores, NumPy array, scores for all examples in the dataset after replacing features
    """

    data_to_modify = all_data_dataset.copy()

    # choose which features that are not replaced based on their index
    indices_to_keep = []
    param_list = np.array(list(data_to_modify.keys()))
    for branch in branches_to_block:  # iterate over the branches that are blocked
        for parameter in branch:  # iterate over the features in a given branch
            index = np.where(param_list == parameter)[0][0]
            indices_to_keep.append(index)
    # get the indices of the features that are to be changed
    indices_to_replace = np.setdiff1d(np.arange(len(data_to_modify.keys())), indices_to_keep)
    params_to_perfect = param_list[indices_to_replace]
    # replace non-blocked features in all examples with the features from the representative PC
    for parameters in params_to_perfect:
        data_to_modify[parameters][:] = all_data_train[parameters][best_example_ind]

    # run inference for modified set of features
    scores = model.predict(data_to_modify)

    return scores


def form_info_table_pcs(full_scores_dataset, full_scores_train, feature_grouping_scores, best_example_ind,
                        examples_info):
    """ Aggregate results obtained from replacing different sets of features for a given representative PC. Compute the
    subtraction between the score obtained when replacing a set of features and the original score for the
    representative PC.

    :param full_scores_dataset: NumPy array, scores for all examples in the dataset without replacing any features
    :param full_scores_train: NumPy array, scores for all examples in the training dataset without replacing any
    features
    :param feature_grouping_scores: dict, each key maps to a NumPy array with scores for all examples in the dataset
    after replacing the given set of features described by the key
    :param best_example_ind: int, index for representative PC in the training set
    :param examples_info: pandas DataFrame, additional information for the examples in the dataset (e.g., TCE parameters
    such as period and epoch)
    :return:
        pandas DataFrame, table with aggregated results
    """

    tbl = examples_info.copy(deep=True)
    tbl['full'] = full_scores_dataset
    scores_to_tbl = {feature_grouping_name:
                         (feature_grouping_scores[feature_grouping_name] -
                          full_scores_train[best_example_ind]).squeeze()
                     for feature_grouping_name in feature_grouping_scores}
    scores_df = pd.DataFrame(data=scores_to_tbl)

    return pd.concat([tbl, scores_df], axis=1)


def generate_scores_replace_pc(trial_num, pc_ind_i, pc_ind, all_data, examples_info, full_dataset_scores,
                               dataset, run_config, logger):
    """ Run replace model PC method to generate scores for each feature grouping and example.

    Args:
        trial_num: int, trial run id
        pc_ind_i:
        pc_ind:
        all_data: dict, features for all examples for each data set (keys are the data sets)
        examples_info: dict, information on examples for each data set (keys are the data sets)
        full_dataset_scores: dict, full model scores for all examples for each data set (keys are the data sets)
        dataset: str, data set to be run
        run_config: dict, configuration dictionary for all runs
        logger: logger

    Returns:

    """

    logger.info(f'[Trial {trial_num + 1}/{run_config["num_trials"]}, Dataset {dataset}] - '
                        f'PC {pc_ind_i + 1}/{run_config["num_PCs"]}')

    # load model
    logger.info(f'Loading model from {run_config["model_filepath"]}')
    model = load_model(filepath=run_config['model_filepath'], compile=False)
    # model.summary()

    # iterate over the combinations of features to be replaced
    features_grouping_scores = {feature_grouping_name: []
                                for feature_grouping_name in run_config['features_grouping']}
    for feature_grouping_name, feature_grouping in run_config['features_grouping'].items():

        logger.info(f'[Trial {trial_num + 1}/{run_config["num_trials"]}, Dataset {dataset}, '
                    f'PC {pc_ind_i + 1}/{run_config["num_PCs"]}] - Feature grouping {feature_grouping_name}')

        explain_feature_grouping_scores = explain_branches_pcs(
            [feature_grouping],
            all_data[dataset],
            all_data['train'],
            model,
            pc_ind
        )
        features_grouping_scores[feature_grouping_name] = explain_feature_grouping_scores

    # create table with scores for the different branch occlusion + full model for the data set
    group = form_info_table_pcs(full_dataset_scores[dataset],
                                full_dataset_scores['train'],
                                features_grouping_scores,
                                pc_ind,
                                examples_info[dataset])

    return group


def run_trial(trial_num, run_config, exp_dir, best_PCs, examples_info, all_data, full_dataset_scores,
              logger=None):
    """ Run a single trial. This consists of sample `run_config['num_PCs']` model PCs, and then iterating through each
    data set. For each data set and sampled model PC, generate scores for each feature grouping and example.

    Args:
        trial_num: int, trial run id
        run_config: dict, configuration dictionary for all runs
        exp_dir: Path, experiment directory
        best_PCs: int, number of model PCs to be sampled from the representative set
        examples_info: dict, information on examples for each data set (keys are the data sets)
        all_data: dict, features for all examples for each data set (keys are the data sets)
        full_dataset_scores: dict, full model scores for all examples for each data set (keys are the data sets)
        logger: logger

    Returns:

    """

    trial_run = f'exp_{run_config["num_PCs"]}_PCs_trial_{trial_num}'
    trial_run_dir = exp_dir / trial_run
    trial_run_dir.mkdir(exist_ok=True)

    if logger is None:
        # set up logger
        logger = logging.getLogger(name=f'explainability run_trial_{trial_num}')
        logger_handler = logging.FileHandler(filename=trial_run_dir / f'run_trial_{trial_num}.log', mode='w')
        logger_formatter = logging.Formatter('%(asctime)s - %(message)s')
        logger.setLevel(logging.INFO)
        logger_handler.setFormatter(logger_formatter)
        logger.addHandler(logger_handler)

    logger.info(f'Starting trial {trial_num}...')

    logger.info(f'Trial {trial_num + 1}/{run_config["num_trials"]}')

    # randomly select num_PCs out of the candidates for representative PCs in the training set
    pc_inds = np.random.choice(best_PCs, run_config['num_PCs'])
    logger.info(f'[Trial {trial_num + 1}/{run_config["num_trials"]}] Selected {run_config["num_PCs"]} examples from '
                f'the training set as representative PCs')

    # write which PCs were used as the best templates
    with open(trial_run_dir / "PC_list.txt", "w") as text_file:
        for best_example_ind in pc_inds:
            text_file.write(f'KIC {examples_info["train"]["target_id"][best_example_ind]}.'
                            f'{examples_info["train"]["tce_plnt_num"][best_example_ind]}\n')

    for dataset in run_config['datasets']:  # iterate over the datasets

        logger.info(f'[Trial {trial_num + 1}/{run_config["num_trials"]}] - Dataset {dataset}')

        for pc_ind_i, pc_ind in enumerate(pc_inds):  # iterate over the PC templates

            group = generate_scores_replace_pc(trial_num, pc_ind_i, pc_ind, all_data, examples_info,
                                               full_dataset_scores, dataset, run_config, logger)

            # save results for each representative PC into a csv file
            group.to_csv(trial_run_dir / f'{dataset}_top_{pc_ind_i}.csv', index=False)
