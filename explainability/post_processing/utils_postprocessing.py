"""

"""

# 3rd party
import numpy as np


def post_process_flags(branch_flags, branch_explanations, get_max):
    """ Removes branches not used in analysis and merges global and local flux branches explanation scores.

    Args:
        branch_flags: NumPy array, branch flags [n_examples x n_flags]
        branch_explanations: NumPy array, branch scores from a given explainability test [n_examples x n_flags]
        get_max: bool, if True chooses max score value between the two flux branches (local and global), otherwise it
        chooses the minimum

    Returns:
        new NumPy arrays branch_flags and branch_explanations after doing the corrections

    """

    BRANCH_INDS_TO_KEEP =[1, 2, 3, 4, 6]  # do not keep global flux and stellar flags

    branch_flags = np.moveaxis(branch_flags ,0, -1)[BRANCH_INDS_TO_KEEP]
    branch_flags = np.moveaxis(branch_flags, 0, -1)

    branch_explanations = np.moveaxis(branch_explanations, 0, -1)
    if get_max:  # for occlusion, SHAP methods
        greater_flux = np.amax(branch_explanations[0:1], axis=1)
    else:  # for model PC replacement method
        greater_flux = np.amin(branch_explanations[:, 0:2], axis=1)

    branch_explanations = branch_explanations[BRANCH_INDS_TO_KEEP]
    # update flux branch
    branch_explanations[0] = greater_flux  # combined_flux
    branch_explanations = np.moveaxis(branch_explanations, 0, -1)

    return branch_flags, branch_explanations
