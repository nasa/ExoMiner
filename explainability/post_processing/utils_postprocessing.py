"""

"""

# 3rd party
import numpy as np


def post_process_flags(branch_flags, branch_explanations, get_max, branch_idxs_to_keep):
    """ Removes branches not used in analysis and merges global and local flux branches explanation scores.

    Args:
        branch_flags: NumPy array, branch flags [n_examples x n_flags]
        branch_explanations: NumPy array, branch scores from a given explainability test [n_examples x n_flags]
        get_max: bool, if True chooses max score value between the two flux branches (local and global), otherwise it
        chooses the minimum
        branch_idxs_to_keep: list, indices of flags/branches to keep.

    Returns:
        new NumPy arrays branch_flags and branch_explanations after doing the corrections

    """

    branch_flags = np.moveaxis(branch_flags, 0, -1)[branch_idxs_to_keep]
    branch_flags = np.moveaxis(branch_flags, 0, -1)

    branch_explanations = np.moveaxis(branch_explanations, 0, -1)
    if get_max:  # for occlusion, SHAP methods
        greater_flux = np.amax(branch_explanations[0:2, :], axis=0)
    else:  # for model PC replacement method
        greater_flux = np.amin(branch_explanations[0:2, :], axis=0)

    branch_explanations = branch_explanations[branch_idxs_to_keep]
    # update flux branch
    branch_explanations[0] = greater_flux  # combined_flux
    branch_explanations = np.moveaxis(branch_explanations, 0, -1)

    return branch_flags, branch_explanations


def top_n(branch_explanations, n, get_max):
    """ Compute binary explanations based on the explanation scores for each flag and after applying top-n cut.

    Args:
        branch_explanations: NumPy array, branch scores from a given explainability test [n_examples x n_flags]
        get_max: bool, if True chooses max score value between the two flux branches (local and global), otherwise it
        chooses the minimum
        n: int, top branches to consider as flagged by the method
        get_max: bool, if True chooses the highest top-n; otherwise it chooses the lowest top-n

    Returns:
        branch_explanations_binary, Numpy array with binary explanations after applying top-n cut [n_examples x n_flags]
    """

    branch_explanations_binary = np.zeros(branch_explanations.shape, dtype='int')
    for index in range(len(branch_explanations)):
        # if branch_explanations[index, -1] == 1:  # no_flag is set to true
        #     branch_explanations_binary[index] = np.array(branch_explanations[index])
        #     continue

        # pick the indices of the top n branch occlusion tests
        if get_max:  # top refers to largest scores
            max_contrib = np.argsort(branch_explanations[index])[-1*n:]
        else:  # top refers to smallest scores
            max_contrib = np.argsort(branch_explanations[index])[:n]

        # set all branch occlusion tests to zero except the ones in the top n
        branch_explanations_binary[index][max_contrib] = 1

    return branch_explanations_binary


def threshold(branch_explanations, thr):
    """ Compute binary explanations based on the explanation scores for each flag and after applying thresholding.

    Args:
        branch_explanations: NumPy array, branch scores from a given explainability test [n_examples x n_flags]
        thr: float, threshold

    Returns:
        branch_explanations_binary, Numpy array with binary explanations after applying top-n cut [n_examples x n_flags]
    """

    branch_explanations_binary = np.zeros_like(branch_explanations)
    for index in range(len(branch_explanations)):

        branch_explanations_binary[index] = np.where(branch_explanations[index] > thr, 1, 0)

    return branch_explanations_binary
