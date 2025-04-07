"""
Utility functions for running occlusion explainability tests.
"""

# 3rd party
import pandas as pd
import numpy as np


def explain_branches_occlusion(branches_to_block, all_data_dataset, model):
    """ Run inference after zeroing-out features except those in the `branches_to_block` for each example in the
    dataset.

    :param branches_to_block: list, sets of features that are not replaced
    :param all_data_dataset: dict, features for all examples in the dataset. Key is a feature and the value is an array
    [n_examples x feature_dim]
    :param model: Keras model, model used to run inference
    :return:
        scores, NumPy array, scores for all examples in the dataset after replacing features
    """

    # choose which features that are not replaced based on their index
    indices_to_keep = []
    param_list = np.array(list(all_data_dataset.keys()))
    for branch in branches_to_block:  # iterate over the branches that are blocked
        for parameter in branch:  # iterate over the features in a given branch
            index = np.where(param_list == parameter)[0][0]
            indices_to_keep.append(index)

    modified_data = {param: np.array(values) for param, values in all_data_dataset.items()}

    # get the indices of the features that are to be changed
    # indices_to_replace = np.setdiff1d(np.arange(len(data_to_modify.keys())), indices_to_keep)
    # params_to_perfect = param_list[indices_to_replace]
    params_to_zero = param_list[indices_to_keep]

    # zero out features in examples
    for parameters in params_to_zero:
        modified_data[parameters][:] = np.zeros_like(modified_data[parameters][0])

    # run inference for modified set of features
    scores = model.predict(modified_data)

    return scores


def form_info_table_occlusion(full_scores, feature_grouping_scores, examples_info):
    """ Aggregate results obtained from occluding different sets of features for examples in a data set. Compute the
    subtraction between the score obtained when occluding the set of features and the original score for each example.

    :param full_scores: NumPy array, scores for all examples in the dataset without replacing any features
    :param feature_grouping_scores: dict, each key maps to a NumPy array with scores for all examples in the dataset
    after replacing the given set of features described by the key
    :param examples_info: pandas DataFrame, additional information for the examples in the dataset (e.g., TCE parameters
    such as period and epoch)
    :return:
        pandas DataFrame, table with aggregated results
    """

    tbl = examples_info.copy(deep=True)
    tbl['full'] = full_scores
    scores_to_tbl = {feature_grouping_name:
                         (feature_grouping_scores[feature_grouping_name] - full_scores).squeeze()
                     for feature_grouping_name in feature_grouping_scores}
    scores_df = pd.DataFrame(data=scores_to_tbl)

    return pd.concat([examples_info, scores_df], axis=1)
