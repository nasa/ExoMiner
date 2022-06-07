"""
Utility functions for running explainability tests.
"""

# 3rd party
import numpy as np
import pandas as pd


def explain_branches_pcs(branches_to_block, all_data_dataset, all_data_train, model, best_example_ind):
    """ Run inference after replacing all features except those in the `branches_to_block` for each example in the
    dataset by the features from the representative PC example in the training set.

    :param branches_to_block: list, sets of features (branches) that are not replaced
    :param all_data_dataset: dict, features for all examples in the dataset. Key is a feature and the value is an array
    n_examplesxfeature_dim
    :param all_data_train: dict, features for all examples in the training set. Key is a feature and the value is an
    array n_examplesxfeature_dim
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
        for parameter in branch:
            index = np.where(param_list == parameter)[0][0]
            indices_to_keep.append(index)
    indices_to_replace = np.setdiff1d(np.arange(len(data_to_modify.keys())), indices_to_keep)
    params_to_perfect = param_list[indices_to_replace]
    for parameters in params_to_perfect:
        data_to_modify[parameters][:] = all_data_train[parameters][best_example_ind]

    # run inference for modified set of features
    scores = model.predict(data_to_modify)

    return scores


def explain_branches_occlusion(branches_to_block, all_data_dataset, model):
    """ Run inference after zeroing-out features except those in the `branches_to_block` for each example in the
    dataset.

    :param branches_to_block: list, sets of features (branches) that are not replaced
    :param all_data_dataset: dict, features for all examples in the dataset. Key is a feature and the value is an array
    n_examplesxfeature_dim
    :param model: Keras model, model used to run inference
    :return:
        scores, NumPy array, scores for all examples in the dataset after replacing features
    """

    data_to_modify = all_data_dataset.copy()

    # choose which features that are not replaced based on their index
    indices_to_keep = []
    param_list = np.array(list(data_to_modify.keys()))
    for branch in branches_to_block:  # iterate over the branches that are blocked
        for parameter in branch:
            index = np.where(param_list == parameter)[0][0]
            indices_to_keep.append(index)
    indices_to_replace = np.setdiff1d(np.arange(len(data_to_modify.keys())), indices_to_keep)
    params_to_perfect = param_list[indices_to_replace]
    for parameters in params_to_perfect:
        data_to_modify[parameters][:] = np.zeros_like(data_to_modify[parameters][0])

    # run inference for modified set of features
    scores = model.predict(data_to_modify)

    return scores


def form_info_table_pcs(full_scores_dataset, full_scores_train, feature_grouping_scores, best_example_ind, examples_info):
    """ Aggregate results obtained from replacing different sets of features for a given representative PC. Compute the
    subtraction between the score obtained when replacing a set of features and the original score for the
    representative PC.

    :param full_scores_dataset: NumPy array, scores for all examples in the dataset without replacing any features
    :param full_scores_train: NumPy array, scores for all examples in the training dataset without replacing any
    features
    :param feature_grouping_scores: dict, each key maps to a NumPy array with scores for all examples in the dataset after
    replacing the given set of features described by the key
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


def form_info_table_occlusion(full_scores, feature_grouping_scores, examples_info):
    """ Aggregate results obtained from replacing different sets of features for a given representative PC. Compute the
    subtraction between the score obtained when replacing a set of features and the original score for the
    representative PC.

    :param full_scores: NumPy array, scores for all examples in the dataset without replacing any features
    :param feature_grouping_scores: dict, each key maps to a NumPy array with scores for all examples in the dataset after
    replacing the given set of features described by the key
    :param examples_info: pandas DataFrame, additional information for the examples in the dataset (e.g., TCE parameters
    such as period and epoch)
    :return:
        pandas DataFrame, table with aggregated results
    """

    tbl = examples_info.copy(deep=True)
    tbl['full'] = full_scores
    scores_to_tbl = {feature_grouping_name: feature_grouping_scores[feature_grouping_name].squeeze()
                     for feature_grouping_name in feature_grouping_scores}
    scores_df = pd.DataFrame(data=scores_to_tbl)

    return pd.concat([examples_info, scores_df], axis=1)
