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

# local
from data_wrangling.kepler_multiplicity_boost.utils_multiplicity_boost import _compute_expected_ntargets_for_obs, \
    _compute_expected_ntargets, residual_expect_ntargets

#%% Experiment initial setup

res_root_dir = Path('/data5/tess_project/experiments/current_experiments/kepler_multiplicity_boost/')

res_dir = res_root_dir / f'cumulativekoi_linlstsqr_{datetime.now().strftime("%m-%d-%Y_%H%M")}'
res_dir.mkdir(exist_ok=True)

# set up logger
logger = logging.getLogger(name='kepler_multiplicity_boost_run')
logger_handler = logging.FileHandler(filename=res_dir / f'kepler_multiplicity_boost_run.log', mode='w')
logger_formatter = logging.Formatter('%(asctime)s - %(message)s')
logger.setLevel(logging.INFO)
logger_handler.setFormatter(logger_formatter)
logger.addHandler(logger_handler)
handler = logging.StreamHandler(sys.stdout)
logger.addHandler(handler)
logger.info(f'Starting run...')

#%% Create stellar and KOI catalog to be used

stellar_cat_fp = Path(res_root_dir /
                      'stellar_catalogs' / 'stellar_cut_2-23-2022' / 'dr25_stellar_berger2020_multis.txt')
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

# add FPWG dispositions
cfp_cat_fp = Path('/data5/tess_project/Data/Ephemeris_tables/Kepler/kois_tables/fpwg_2021.03.02_12.09.58.csv')
cfp_cat = pd.read_csv(cfp_cat_fp, header=75)
logger.info(f'Using Certified FP table from {cfp_cat_fp}. Adding FPWG dispositions to compute number of observed FPs')
koi_cat = koi_cat.merge(cfp_cat[['kepoi_name', 'fpwg_disp_status']], on=['kepoi_name'], how='left', validate='one_to_one')
koi_cat.to_csv(res_dir / f'{koi_cat_fp.stem}_fpwg.csv', index=False)

tbls = []

# number of KOIs in target
cnt_kois_target = \
    koi_cat['kepid'].value_counts().to_frame(name='num_kois_target').reset_index().rename(columns={'index': 'kepid'})
tbls.append(cnt_kois_target)
# number of Candidate KOIs in target
cnt_candidates_target = \
    koi_cat.loc[(koi_cat['koi_pdisposition'] == 'CANDIDATE'), 'kepid'].value_counts().to_frame(name='num_candidates_target').reset_index().rename(columns={'index': 'kepid'})
tbls.append(cnt_candidates_target)
# number of known FP KOIs in target
# cnt_fps_target = koi_cat.loc[koi_cat['koi_pdisposition'] == 'FALSE POSITIVE', 'kepid'].value_counts().to_frame(name='num_fps_target').reset_index().rename(columns={'index': 'kepid'})
# cnt_fps_target = koi_cat.loc[((koi_cat['koi_pdisposition'] == 'FALSE POSITIVE') & (koi_cat['koi_disposition'] != 'CONFIRMED')), 'kepid'].value_counts().to_frame(name='num_fps_target').reset_index().rename(columns={'index': 'kepid'})
# cnt_fps_target = koi_cat.loc[((koi_cat['fpwg_disp_status'] == 'CERTIFIED FP') & (koi_cat['koi_disposition'] != 'CONFIRMED')), 'kepid'].value_counts().to_frame(name='num_fps_target').reset_index().rename(columns={'index': 'kepid'})
cnt_fps_target = \
    koi_cat.loc[(koi_cat['fpwg_disp_status'] == 'CERTIFIED FP'), 'kepid'].value_counts().to_frame(name='num_fps_target').reset_index().rename(columns={'index': 'kepid'})
# cnt_fps_target = koi_cat.loc[((koi_cat['fpwg_disp_status'] == 'CERTIFIED FP') &
#                               koi_cat['koi_pdisposition'] != 'CANDIDATE'), 'kepid'].value_counts().to_frame(
#     name='num_fps_target').reset_index().rename(columns={'index': 'kepid'})
tbls.append(cnt_fps_target)
# number of Confirmed KOIs in target
cnt_planets_target = \
    koi_cat.loc[koi_cat['koi_disposition'] == 'CONFIRMED', 'kepid'].value_counts().to_frame(name='num_planets_target').reset_index().rename(columns={'index': 'kepid'})
tbls.append(cnt_planets_target)

for tbl in tbls:
    koi_cat = koi_cat.merge(tbl, on=['kepid'], how='left', validate='many_to_one')

koi_cat.to_csv(res_dir / f'{koi_cat_fp.stem}_counts.csv', index=False)

#%% update FP observations using Jon's table

logger.info('Updating number of FP observations according to Jon\'s analysis...')
fps_jon_tbl = pd.read_csv('/data5/tess_project/experiments/current_experiments/kepler_multiplicity_boost/Kepler FPs from Miguel JJ.csv')

koi_cat['jon_obs'] = 'no_comment'
for fp_i, fp in fps_jon_tbl.iterrows():
    if fp['Count as an FP'] == 'F':
        koi_cat.loc[koi_cat['kepid'] == int(fp['kepid']), 'num_fps_target'] -= 1
        koi_cat.loc[koi_cat['kepid'] == int(fp['kepid']), 'num_kois_target'] -= 1
        koi_cat.loc[koi_cat['kepid'] == int(fp['kepid']), 'jon_obs'] = fp['Comments']
koi_cat.to_csv(res_dir / f'{koi_cat_fp.stem}_counts_jon_fps.csv', index=False)

#%% solve disposition of KOIs with conflicting CFP and Kepler data KOI dispositions

logger.info('Updating KOI disposition conflicts between CANDIDATE and CERTIFIED FP dispositions...')
kois_disposition_conflict = {
    # 'K00950.01': {'disposition': 'CANDIDATE'},
    'K03936.01': {'disposition': 'CANDIDATE'},
}
logger.info(f'KOIs with disposition conflicts:\n{kois_disposition_conflict}')
for koi_id, koi_info in kois_disposition_conflict.items():
    if koi_info['disposition'] == 'CANDIDATE':
        koi_cat.loc[koi_cat['kepoi_name'] == koi_id, 'num_fps_target'] -= 1
        koi_cat.loc[koi_cat['kepoi_name'] == koi_id, 'num_candidates_target'] += 1

#%% get counts from KOI catalog to Kepid catalog

logger.info('Counting number of KOIs, Candidates, FPs, and Planets for each target...')
# koi_stellar_cat = koi_cat[['kepid', 'koi_steff', 'koi_steff_err1', 'koi_steff_err2',
#        'koi_slogg', 'koi_slogg_err1', 'koi_slogg_err2', 'koi_smet',
#        'koi_smet_err1', 'koi_smet_err2', 'koi_srad', 'koi_srad_err1',
#        'koi_srad_err2', 'koi_smass', 'koi_smass_err1', 'koi_smass_err2', 'ra',
#        'dec', 'koi_kepmag', 'num_kois_target', 'num_candidates_target', 'num_fps_target']].copy(deep=True)
koi_stellar_cat = koi_cat[['kepid', 'ra',
                           'dec', 'koi_kepmag', 'num_kois_target', 'num_candidates_target', 'num_fps_target',
                           'num_planets_target']].copy(deep=True)
koi_stellar_cat.drop_duplicates(subset='kepid', inplace=True, ignore_index=True)

# merge counts from Kepid catalog to stellar catalog
koi_stellar_cat_cols = ['kepid', 'num_kois_target', 'num_candidates_target', 'num_fps_target', 'num_planets_target']
stellar_cat_koi_cnt = stellar_cat.merge(koi_stellar_cat[koi_stellar_cat_cols],
                                        on=['kepid'],
                                        how='left',
                                        validate='one_to_one')
stellar_cat_koi_cnt.to_csv(res_dir / f'{stellar_cat_fp.stem}_koi_counts.csv', index=False)

# %% compute estimates for quantities plugged into the statistical framework

logger.info('Computing estimates for quantities needed for the statistical framework...')

quantities = {
    'n_t': len(stellar_cat_koi_cnt),
    'n_k': (stellar_cat_koi_cnt['num_candidates_target'] >= 1).sum(),
    'n_m': (stellar_cat_koi_cnt['num_candidates_target'] >= 2).sum(),
    'n_fm': np.nan,
    'p_1': np.nan,
}
p_1_estimates = {
    'p_1_sngle_cand_in_kois,': ((koi_cat['num_kois_target'] == 1) & (
            koi_cat['koi_pdisposition'] == 'CANDIDATE')).sum() / (koi_cat['num_kois_target'] == 1).sum(),
    'p_1_sngle_plnt_in_cands': (koi_cat['num_planets_target'] == 1 & (koi_cat['num_candidates_target'] == 1)).sum() / (
            koi_cat['num_candidates_target'] == 1).sum(),
    'p_1_no_sngle_fp_in_cands': (koi_cat['num_fps_target'] == 0 & (koi_cat['num_candidates_target'] == 1)).sum() / (
            koi_cat['num_candidates_target'] == 1).sum(),
}
# logger.info(f'Estimates for P_1:\n {p_1_estimates}')
# quantities['p_1'] = np.nan
#
# for n_cand_in_target in range(1, stellar_cat_koi_cnt['num_candidates_target'].max() + 1):
#     quantities[f'n_{n_cand_in_target}_cands'] = (stellar_cat_koi_cnt['num_candidates_target'] == n_cand_in_target).sum()
for n_cand_in_target in range(1, int(stellar_cat_koi_cnt['num_candidates_target'].max()) + 1):
    quantities[f'n_{n_cand_in_target}'] = (stellar_cat_koi_cnt['num_candidates_target'] == n_cand_in_target).sum()
#
# # quantities['n_targets'] = 140016
# # quantities['n_1_cands'] = 2499
# # quantities['n_targets_cands'] += quantities['n_1_cands'] - (stellar_cat_koi_cnt['num_candidates_target'] == 1).sum()
#
# quantities = {
#     'n_t': 140016,
#     'n_k': 2962,  # 1723 (validation), 2962 (cands+fps), 3100 (cands+fps+ebs)
#     'n_m': 463,  # 420 (validation), 463 (cands+fps) (cand+fp+ebs)
#     'n_1': 2499,  # 1303 (validation), 2499 (cands+fps), 2637 (cands+fps+ebs)
#     'p_1': 0.5922,  # 0.9 (validation), 0.5922 (cands+fps), 0.5612 (cands+fps+ebs)
#     'n_fm': 24.9  # 24.9 (cands+fps), 29.4 (cands+fps+ebs)
# }
logger.info(f'Estimated quantities:\n {quantities}')


# %% compute expected values

logger.info('Computing number of observations for each scenario...')

# compute observations for different scenarios
observations = {
    (2, 0): ((stellar_cat_koi_cnt['num_fps_target'] == 2) & (stellar_cat_koi_cnt['num_kois_target'] == 2)).sum(),
    (3, 0): ((stellar_cat_koi_cnt['num_fps_target'] == 3) & (stellar_cat_koi_cnt['num_kois_target'] == 3)).sum(),
    (1, 1): ((stellar_cat_koi_cnt['num_fps_target'] == 1) & (stellar_cat_koi_cnt['num_candidates_target'] == 1)).sum(),
    (2, 1): ((stellar_cat_koi_cnt['num_fps_target'] == 2) & (stellar_cat_koi_cnt['num_candidates_target'] == 1)).sum(),
    (1, 2): ((stellar_cat_koi_cnt['num_fps_target'] == 1) & (stellar_cat_koi_cnt['num_candidates_target'] == 2)).sum(),
    (2, 2): ((stellar_cat_koi_cnt['num_fps_target'] == 2) & (stellar_cat_koi_cnt['num_candidates_target'] == 2)).sum(),
}
# observations = {
#     (2, 0): 6,
#     (3, 0): 0,
#     (1, 1): 12,
#     (2, 1): 0,
#     (1, 2): 2,
#     (2, 2): 0
# }
logger.info(observations)

# quantities = {
#     'n_targets': 2*95000,
#     'n_targets_cands': 2962,
#     'n_targets_multi_cands': 463,
#     'n_1_cands': 2499,
#     'p_1': 0.5922,
#     'n_fp_cand_multi': 24.9
# }

quantities['n_fm'] = 24.9  # 2.09  # ((koi_cat['num_kois_target'] >= 2) & (koi_cat['num_fps_target'] >= 0)).sum()
quantities['p_1'] = 0.5922
# quantities['n_k'] = quantities['n_1']
logger.info(f'Quantities without conducting optimization:\n {quantities}')

n_plnts_fps_inputs = [(2, 0), (3, 0), (1, 1), (2, 1), (1, 2), (2, 2)]
logger.info(f'Results without optimization for inputs: {n_plnts_fps_inputs}')
# _compute_expected_ntargets_for_obs(n_plnts_fps_inputs, _compute_expected_ntargets, quantities, logger=logger)
predict_obs = _compute_expected_ntargets_for_obs(n_plnts_fps_inputs,
                                                 _compute_expected_ntargets,
                                                 quantities,
                                                 observations,
                                                 logger=logger)

#%% optimization fns to find n_{fm} and p_1 iteratively using the statistical framework with the  observed data and
# estimated counts

# loss_expect_ntargets(x_0, quantities, observations, n_plnts_fps_inputs)

quantities_opt = {k: v for k, v in quantities.items() if k in ['n_t', 'n_1', 'n_k', 'n_m', 'n_fm', 'p_1']}  # quantities.copy()
# quantities_opt['n_targets'] = 140016
logger.info(f'Quantities used for optimization:\n {quantities_opt}')

# initial input values
x_0 = np.array([
    quantities_opt['p_1'],
    quantities_opt['n_fm']
])
# x_0 = np.array([0.42, 250])
opt_quantities = [
    'p_1',
    'n_fm',
]
# scenarios to take into account for the optimization
n_plnts_fps_inputs = [(2, 0), (3, 0), (1, 1), (2, 1), (1, 2), (2, 2)]

logger.info(f'Optimizing using linear least squares for inputs and initial parameter values: '
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
                             args=(opt_quantities, quantities_opt.copy(), observations, n_plnts_fps_inputs),
                             **optimization_params
                             )
logger.info(f'Optimization results:\n{res}')

logger.info(f'Results of optimization:\n p_1: {res.x[0]} (p_1_0={quantities_opt["p_1"]})\n '
            f'n_fm: {res.x[1]} (n_fm_0={quantities_opt["n_fm"]})')

quantities_new = quantities_opt.copy()
quantities_new['p_1'] = res.x[0]
quantities_new['n_fm'] = res.x[1]
logger.info(f'New quantities after optimization:\n {quantities_new}')
estimated_observations = _compute_expected_ntargets_for_obs(n_plnts_fps_inputs,
                                   _compute_expected_ntargets,
                                   quantities_new,
                                   observations,
                                   logger=logger)
estimated_nfm = np.sum([scenario[0] * estimated_obs for scenario, estimated_obs in estimated_observations.items()])
logger.info(f'Estimated n_fm from the equations: {estimated_nfm:.3f} (optimized n_fm={quantities_new["n_fm"]:.3f})')
