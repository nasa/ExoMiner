""" Utility functions used for multiplicity boost experiments. """

# 3rd party
import numpy as np


def _compute_expected_ntargets_fps(p_1, n_1, n_fm, n_t, n_fps, n_k=None):
    """ Compute expected number of targets for scenarios with FPs.

    :param p_1: float, fidelity of the sample (fraction of candidates that are planets)
    :param n_1: int, number of targets with exactly one candidate
    :param n_fm: int, number of FP planet candidates present among the multis
    :param n_t: int, total number of targets
    :param n_fps: int, number of FPs in the given scenario
    :param n_k: int, number of targets with one or more candidates
    :return:
        float, expected number of  targets for scenarios with FPs
    """

    if n_fps == 1:
        if n_k is not None:
            return (1 - p_1) * n_k
        else:
            return (1 - p_1) * n_1
    else:
        return ((1 - p_1) * n_1 + n_fm) ** n_fps / (np.math.factorial(n_fps) * n_t ** (n_fps - 1))


def _compute_expected_ntargets_planets(p_1, n_1, n_fm, n_m, n_t, n_plnts):
    """ Compute expected number of targets for scenarios with planets.

    :param p_1: float, fidelity of the sample (fraction of candidates that are planets)
    :param n_1: int, number of targets with exactly one candidate
    :param n_fm: int, number of FP planet candidates present among the multis
    :param n_m: int, number of targets with two or more candidates (multis)
    :param n_t: int, total number of targets
    :param n_plnts: int, number of candidates in the given scenario
    :return:
        float, expected number of targets for scenarios with planets
    """

    if n_plnts == 1:
        return (n_1 * p_1 + n_fm) / n_t
    else:
        return n_m / n_t


def _compute_expected_ntargets(n_plnts, n_fps, p_1, n_1, n_fm, n_t, n_k, n_m):
    """ Compute expected number of targets for scenarios with FPs and planets.

    :param n_plnts: int, number of candidates in the given scenario
    :param n_fps: int, number of FPs in the given scenario
    :param p_1: float, fidelity of the sample (fraction of candidates that are planets)
    :param n_1: int, number of targets with exactly one candidate
    :param n_fm: int, number of FP planet candidates present among the multis
    :param n_t: int, total number of targets
    :param n_k: int, number of targets with one or more candidates
    :param n_m: int, number of targets with two or more candidates (multis)
    :return:
        float, expected number of targets for a given scenario
    """

    if n_plnts == 0:
        return _compute_expected_ntargets_fps(p_1, n_1, n_fm, n_t, n_fps, n_k=None)
    else:
        return _compute_expected_ntargets_planets(p_1, n_1, n_fm, n_m, n_t, n_plnts) * \
               _compute_expected_ntargets_fps(p_1, n_1, n_fm, n_t, n_fps, n_k=n_k)


def _compute_expected_ntargets_for_obs(n_plnts_fps_inputs, fn_compute_expected_ntargets, quantities, observations,
                                       logger=None):
    """ Compute expected number of targets vs observations for different scenarios.

    :param n_plnts_fps_inputs: list of tuples, each tuple is a pair with (number_of_fps, number_of_pcs) for a given
    scenario. Expected number of targets is computed for each one of these scenarios
    :param fn_compute_expected_ntargets: function, used to compute expected number of targets
    :param quantities: dict, counts needed for the statistical framework
    :param observations: dict, contains number of observations in the data for each scenario
    :param logger: bool, if True logs information
    :return:
        dict, key is a tuple pair in n_plnts_fps_inputs where the value is the expected number of targets for that
        scenario
    """

    predicted_obs = {(n_fps, n_plnts): np.nan for n_fps, n_plnts in n_plnts_fps_inputs}
    for n_fps, n_plnts in n_plnts_fps_inputs:

        val = fn_compute_expected_ntargets(n_plnts,
                                           n_fps,
                                           quantities['p_1'],
                                           quantities['n_1'],
                                           quantities['n_fm'],
                                           quantities['n_t'],
                                           quantities['n_k'],
                                           quantities['n_m'])

        if logger is None:
            print(f'{n_plnts} planets + {n_fps} FPs: {val} | Observations {observations[(n_fps, n_plnts)]}')
        else:
            logger.info(f'{n_plnts} planets + {n_fps} FPs: {val} | Observations {observations[(n_fps, n_plnts)]}')

        predicted_obs[n_fps, n_plnts] = val

    return predicted_obs


def loss_expect_ntargets(x, quantities, observations, n_plnts_fps_inputs):
    """ Compute least squares loss.

    :param x: NumPy array, input
    :param quantities: dict, counts
    :param observations: dict, observations for each scenario
    :param n_plnts_fps_inputs: list of tuples, different scenarios that are taken into account
    :return:
        float, least squares loss
    """

    return 0.5 * np.sum([(_compute_expected_ntargets(n_plnts,
                                                     n_fps,
                                                     x[0],
                                                     quantities['n_1'],
                                                     x[1],
                                                     quantities['n_t'],
                                                     quantities['n_k'],
                                                     quantities['n_m']) -
                          observations[n_fps, n_plnts]) ** 2
                         for n_fps, n_plnts in n_plnts_fps_inputs]
                        )


def residual_expect_ntargets(x, opt_quantities, quantities, observations, n_plnts_fps_inputs):
    """ Compute residual for the expected number of targets.

    :param x: NumPy array, input with value for quantities to be optimized
    :param opt_quantities: list, names of quantities to be optimized; same order as input x
    :param quantities: dict, estimates of non-optimizable parameters
    :param observations: dict, observations for each scenario
    :param n_plnts_fps_inputs: list of tuples, different scenarios that are taken into account
    :return:
        float, least squares loss residual
    """

    for x_i, x_el in enumerate(x):
        quantities[opt_quantities[x_i]] = x_el

    # return [(_compute_expected_ntargets(n_plnts,
    #                                     n_fps,
    #                                     **quantities) * n_fps -
    #          observations[n_fps, n_plnts]) ** 2
    #         for n_fps, n_plnts in n_plnts_fps_inputs]
    return (np.sum([_compute_expected_ntargets(n_plnts, n_fps, **quantities) * n_fps
                    for n_fps, n_plnts in n_plnts_fps_inputs]) - quantities['n_fm']) ** 2
