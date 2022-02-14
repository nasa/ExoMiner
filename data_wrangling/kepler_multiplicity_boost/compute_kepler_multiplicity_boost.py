"""
Compute multiplicity boost for Kepler data.
"""

# 3rd party
import sys
import pandas as pd
from pathlib import Path
import numpy as np
import logging
from scipy import optimize
from datetime import datetime

#%% Experiment initial setup

res_root_dir = Path('/Users/msaragoc/OneDrive - NASA/Projects/exoplanet_transit_classification/experiments/'
                    'kepler_multiplicity_boost/')

res_dir = res_root_dir / f'cumulativekoi_linlstsqr_{datetime.now().strftime("%m-%d-%Y_%H%M")}'
res_dir.mkdir(exist_ok=True)

# set up logger
logger = logging.getLogger(name='kepler_multiplicity_boost_run')
logger_handler = logging.FileHandler(filename=res_dir / f'kepler_multiplicity_boost_run.log', mode='w')
logger_formatter = logging.Formatter('%(asctime)s - %(message)s')
logger.setLevel(logging.INFO)
logger_handler.setFormatter(logger_formatter)
logger.addHandler(logger_handler)
logger.info(f'Starting run...')

#%% Create stellar and KOI catalog to be used

stellar_cat_fp = Path(res_root_dir / 'dr25_stellar_berger2020_multiplicity.txt')
stellar_cat = pd.read_csv(stellar_cat_fp, header=0)
assert len(stellar_cat) == len(stellar_cat['kepid'].unique())
logger.info(f'Using stellar catalog: {str(stellar_cat_fp)}')
logger.info(f'Number of targets in the stellar catalog: {len(stellar_cat)}')

# koi_cat_fp = Path(res_root_dir / 'q1_q17_dr25_sup_koi_2021.11.16_11.10.03.csv')
koi_cat_fp = Path(res_root_dir / 'cumulative_2021.11.22_15.20.09.csv')
# koi_cat_fp = Path(res_root_dir / 'q1_q8_koi_2021.11.16_15.27.49.csv')
koi_cat = pd.read_csv(koi_cat_fp,
                      header=56  # 56  # 94
                      )
logger.info(f'Using KOI catalog: {str(koi_cat_fp)}')
logger.info(f'Number of KOIs in the KOI catalog: {len(koi_cat)}')

# filter KOI catalog for those KOIs associated with targets in the stellar catalog
koi_cat = koi_cat.loc[koi_cat['kepid'].isin(stellar_cat['kepid'])]
logger.info(f'Number of KOIs in the KOI catalog after keeping only KOIs associated with targets in the stellar '
            f'catalog: {len(koi_cat)}')

# binary flag true and greater than 2%
koi_cat = koi_cat[(koi_cat['koi_period'] > 1.6) & ~((koi_cat['koi_depth'] > 20000) & (koi_cat['koi_fpflag_ss'] == 1))]
logger.info(f'Number of KOIs in the KOI catalog after removing KOIs with periods smaller than 1.6 days '
            f'and KOIs with stellar eclipse flag set to 1 and that have a transit depth greater than 2%:'
            f' {len(koi_cat)}')

koi_cat.to_csv(res_dir / f'{koi_cat_fp.stem}_filtered_stellar.csv', index=False)

#%% count different quantities needed for estimates that are plugged into the statistical framework

# # add FPWG dispositions
cfp_cat = pd.read_csv('/Users/msaragoc/OneDrive - NASA/Projects/exoplanet_transit_classification/data/ephemeris_tables/'
                      'kepler/koi_catalogs/fpwg_2022.02.14_11.08.30.csv',
                      header=17)
koi_cat = koi_cat.merge(cfp_cat[['kepoi_name', 'fpwg_disp_status']], on=['kepoi_name'], how='left')
# koi_cat.to_csv(res_dir / f'{koi_cat_fp.stem}_fpwg.csv', index=False)

tbls = []

# number of KOIs in target
cnt_kois_target = koi_cat['kepid'].value_counts().to_frame(name='num_kois_target').reset_index().rename(
    columns={'index': 'kepid'})
tbls.append(cnt_kois_target)
# number of Candidate KOIs in target
cnt_candidates_target = koi_cat.loc[koi_cat['koi_pdisposition'] == 'CANDIDATE', 'kepid'].value_counts().to_frame(
    name='num_candidates_target').reset_index().rename(columns={'index': 'kepid'})
tbls.append(cnt_candidates_target)
# number of known FP KOIs in target
# cnt_fps_target = koi_cat.loc[koi_cat['koi_pdisposition'] == 'FALSE POSITIVE', 'kepid'].value_counts().to_frame(name='num_fps_target').reset_index().rename(columns={'index': 'kepid'})
# cnt_fps_target = koi_cat.loc[((koi_cat['koi_pdisposition'] == 'FALSE POSITIVE') & (koi_cat['koi_disposition'] != 'CONFIRMED')), 'kepid'].value_counts().to_frame(name='num_fps_target').reset_index().rename(columns={'index': 'kepid'})
# cnt_fps_target = koi_cat.loc[((koi_cat['fpwg_disp_status'] == 'CERTIFIED FP') & (koi_cat['koi_disposition'] != 'CONFIRMED')), 'kepid'].value_counts().to_frame(name='num_fps_target').reset_index().rename(columns={'index': 'kepid'})
cnt_fps_target = koi_cat.loc[((koi_cat['fpwg_disp_status'] == 'CERTIFIED FP')), 'kepid'].value_counts().to_frame(
    name='num_fps_target').reset_index().rename(columns={'index': 'kepid'})
# cnt_fps_target = koi_cat.loc[((koi_cat['fpwg_disp_status'] == 'CERTIFIED FP') &
#                               koi_cat['koi_pdisposition'] != 'CANDIDATE'), 'kepid'].value_counts().to_frame(
#     name='num_fps_target').reset_index().rename(columns={'index': 'kepid'})
tbls.append(cnt_fps_target)
# number of Confirmed KOIs in target
cnt_planets_target = koi_cat.loc[koi_cat['koi_disposition'] == 'CONFIRMED', 'kepid'].value_counts().to_frame(
    name='num_planets_target').reset_index().rename(columns={'index': 'kepid'})
tbls.append(cnt_planets_target)

for tbl in tbls:
    koi_cat = koi_cat.merge(tbl, on=['kepid'], how='left')

koi_cat.to_csv(res_dir / f'{koi_cat_fp.stem}_counts.csv', index=False)

# koi_stellar_cat = koi_cat[['kepid', 'koi_steff', 'koi_steff_err1', 'koi_steff_err2',
#        'koi_slogg', 'koi_slogg_err1', 'koi_slogg_err2', 'koi_smet',
#        'koi_smet_err1', 'koi_smet_err2', 'koi_srad', 'koi_srad_err1',
#        'koi_srad_err2', 'koi_smass', 'koi_smass_err1', 'koi_smass_err2', 'ra',
#        'dec', 'koi_kepmag', 'num_kois_target', 'num_candidates_target', 'num_fps_target']].copy(deep=True)
koi_stellar_cat = koi_cat[['kepid', 'ra',
                           'dec', 'koi_kepmag', 'num_kois_target', 'num_candidates_target', 'num_fps_target',
                           'num_planets_target']].copy(deep=True)
koi_stellar_cat.drop_duplicates(subset='kepid', inplace=True, ignore_index=True)

koi_stellar_cat_cols = ['kepid', 'num_kois_target', 'num_candidates_target', 'num_fps_target', 'num_planets_target']
stellar_cat_koi_cnt = stellar_cat.merge(koi_stellar_cat[koi_stellar_cat_cols], on=['kepid'], how='left')
stellar_cat_koi_cnt.to_csv(res_dir / f'{stellar_cat_fp.stem}_koi_counts.csv', index=False)

# %% compute estimates for quantities plugged into the statistical framework

quantities = {
    'n_targets': len(stellar_cat_koi_cnt),
    'n_targets_cands': (stellar_cat_koi_cnt['num_candidates_target'] >= 1).sum(),
    'n_targets_multi_cands': (stellar_cat_koi_cnt['num_candidates_target'] >= 2).sum(),
    # 'n_fp_cand_multi': a,
    'p_1': (koi_cat['num_kois_target'] == 1 & (koi_cat['koi_pdisposition'] == 'CANDIDATE')).sum() /
           (stellar_cat_koi_cnt['num_kois_target'] == 1).sum()
}

# for n_cand_in_target in range(1, stellar_cat_koi_cnt['num_candidates_target'].max() + 1):
#     quantities[f'n_{n_cand_in_target}_cands'] = (stellar_cat_koi_cnt['num_candidates_target'] == n_cand_in_target).sum()
for n_cand_in_target in range(1, int(stellar_cat_koi_cnt['num_candidates_target'].max()) + 1):
    quantities[f'n_{n_cand_in_target}_cands'] = (stellar_cat_koi_cnt['num_candidates_target'] == n_cand_in_target).sum()

# quantities = {
#     'n_targets': 140016,
#     'n_targets_cands': 2962,
#     'n_targets_multi_cands': 463,
#     'n_1_cands': 2499,
#     'p_1': 0.5922,
#     'n_fp_cand_multi': 24.9
# }

logger.info(f'Estimated quantities:\n {quantities}')


# %% compute expected values


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


def _compute_expected_ntargets_for_obs(n_plnts_fps_inputs, fn_compute_expected_ntargets, quantities, logger=None):
    """ Compute expected number of targets vs observations for different scenarios.

    :param n_plnts_fps_inputs: list of tuples, each tuple is a pair with (number_of_fps, number_of_pcs) for a given
    scenario. Expected number of targets is computed for each one of these scenarios
    :param fn_compute_expected_ntargets: function, used to compute expected number of targets
    :param quantities: dict, counts needed for the statistical framework
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
                                           quantities['n_1_cands'],
                                           quantities['n_fp_cand_multi'],
                                           quantities['n_targets'],
                                           quantities['n_targets_cands'],
                                           quantities['n_targets_multi_cands'])

        if logger is None:
            print(f'{n_plnts} planets + {n_fps} FPs: {val} | Observations {observations[(n_fps, n_plnts)]}')
        else:
            logger.info(f'{n_plnts} planets + {n_fps} FPs: {val} | Observations {observations[(n_fps, n_plnts)]}')

        predicted_obs[n_fps, n_plnts] = val

    return predicted_obs


# compute observations for different scenarios
observations = {
    (2, 0): ((stellar_cat_koi_cnt['num_fps_target'] >= 2)).sum(),
    (3, 0): (stellar_cat_koi_cnt['num_fps_target'] >= 3).sum(),
    (1, 1): ((stellar_cat_koi_cnt['num_fps_target'] >= 1) & (stellar_cat_koi_cnt['num_candidates_target'] >= 1)).sum(),
    (2, 1): ((stellar_cat_koi_cnt['num_fps_target'] >= 2) & (stellar_cat_koi_cnt['num_candidates_target'] >= 1)).sum(),
    (1, 2): ((stellar_cat_koi_cnt['num_fps_target'] >= 1) & (stellar_cat_koi_cnt['num_candidates_target'] >= 2)).sum(),
    (2, 2): ((stellar_cat_koi_cnt['num_fps_target'] >= 2) & (stellar_cat_koi_cnt['num_candidates_target'] >= 2)).sum(),

}
# observations = {
#     (2, 0): 6,
#     (3, 0): 0,
#     (1, 1): 12,
#     (2, 1): 0,
#     (1, 2): 2,
#     (2, 2): 0
# }
#
# quantities = {
#     'n_targets': 2*95000,
#     'n_targets_cands': 2962,
#     'n_targets_multi_cands': 463,
#     'n_1_cands': 2499,
#     'p_1': 0.5922,
#     'n_fp_cand_multi': 24.9
# }

quantities['n_fp_cand_multi'] = 24.9  # ((koi_cat['num_kois_target'] >= 2) & (koi_cat['num_fps_target'] >= 0)).sum()
quantities['p_1'] = 0.5922
logger.info(f'Quantities without conducting optimization:\n {quantities}')

n_plnts_fps_inputs = [(2, 0), (3, 0), (1, 1), (2, 1), (1, 2), (2, 2)]
logger.info(f'Results without optimization for inputs: {n_plnts_fps_inputs}')
# _compute_expected_ntargets_for_obs(n_plnts_fps_inputs, _compute_expected_ntargets, quantities, logger=logger)
predict_obs = _compute_expected_ntargets_for_obs(n_plnts_fps_inputs,
                                                 _compute_expected_ntargets,
                                                 quantities,
                                                 logger=None)

#%% optimization fns to find n_{fm} and p_1 iteratively using the statistical framework with the  observed data and
# estimated counts


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
                                                     quantities['n_1_cands'],
                                                     x[1],
                                                     quantities['n_targets'],
                                                     quantities['n_targets_cands'],
                                                     quantities['n_targets_multi_cands']) -
                          observations[n_fps, n_plnts]) ** 2
                         for n_fps, n_plnts in n_plnts_fps_inputs]
                        )


def residual_expect_ntargets(x, quantities, observations, n_plnts_fps_inputs):
    """ Compute residual for the expected number of targets.

    :param x: NumPy array, input
    :param quantities: dict, counts
    :param observations: dict, observations for each scenario
    :param n_plnts_fps_inputs: list of tuples, different scenarios that are taken into account
    :return:
        float, least squares loss residual
    """

    return [(_compute_expected_ntargets(n_plnts,
                                        n_fps,
                                        x[0],
                                        quantities['n_1_cands'],
                                        x[1],
                                        quantities['n_targets'],
                                        quantities['n_targets_cands'],
                                        quantities['n_targets_multi_cands']) * n_fps -
             observations[n_fps, n_plnts]) ** 2
            for n_fps, n_plnts in n_plnts_fps_inputs]


# loss_expect_ntargets(x_0, quantities, observations, n_plnts_fps_inputs)

quantities_opt = quantities.copy()
# quantities_opt['n_targets'] = 140016
logger.info(f'Quantities used for optimization:\n {quantities_opt}')

#%% Optimize parameters using observations and estimated counts

handler = logging.StreamHandler(sys.stdout)
logger.addHandler(handler)

# initial input values
x_0 = np.array([
    quantities_opt['p_1'],
    quantities_opt['n_fp_cand_multi']
])
# scenarios to take into account for the optimization
n_plnts_fps_inputs = [(2, 0), (3, 0), (1, 1), (2, 1), (1, 2), (2, 2)]

logger.info(f'Optimizing using non-linear least squares for inputs and initial parameter values: '
            f'{n_plnts_fps_inputs}, {x_0}')
# res = optimize.leastsq(residual_expect_ntargets, x_0, args=(quantities_opt, observations, n_plnts_fps_inputs))

# optimization parameters
optimization_params = {
    'loss': 'linear',
    'f_scale': 1.0,
    'bounds': (
        [0, 0],  # [-np.inf, -np.inf],
        [1, np.inf],  # [np.inf, np.inf],  # ([0, 1], [0, 100])
    )
}

logger.info(f'Optimization parameters:\n {optimization_params}')

res = optimize.least_squares(residual_expect_ntargets,
                             x_0,
                             args=(quantities_opt, observations, n_plnts_fps_inputs),
                             **optimization_params
                             )
logger.info(f'Optimization results:\n{res}')

logger.info(f'Results of optimization:\n p_1: {res.x[0]} (p_1_0={quantities_opt["p_1"]})\n '
            f'n_fp_cand_multi: {res.x[1]} (n_fp_cand_multi_0={quantities_opt["n_fp_cand_multi"]})')

quantities_new = quantities_opt.copy()
quantities_new['p_1'] = res.x[0]
quantities_new['n_fp_cand_multi'] = res.x[1]
logger.info(f'New quantities after optimization:\n {quantities_new}')
_compute_expected_ntargets_for_obs(n_plnts_fps_inputs, _compute_expected_ntargets, quantities_new, logger=logger)
