"""
Compute marginal contribution values (mcs) and aggregate results for different features sets.
"""

# 3rd party
import pandas as pd
from pathlib import Path
from scipy.special._comb import _comb_int as comb
import multiprocessing
import numpy as np
from functools import reduce
import yaml
import argparse


# from mpi4py import MPI
# import sys


def compute_mc_for_feat_in_config_run(feat, config_run, feats_in_run, runs_tbl, examples_tbl, score_nofeats,
                                      n_features):
    """ Compute marginal contribution for a given feature and configuration for a set of examples.

    :param feat: str, feature for which to compute the marginal contribution
    :param config_run: str, configuration of features for which the marginal contribution of feature feat is being computed
    :param feats_in_run: list, features in configuration of features
    :param runs_tbl: pandas DataFrame, table that contains scores for all examples across all configuration runs relevant to compute the marginal contribution
    :param examples_tbl: pandas DataFrame, table that contains set of examples for which to compute the mcs
    :param score_nofeats: float, score of null configuration (no features)
    :param n_features: int, number of features in total
    :return:
        examples_tbl: pandas DataFrame, updated table with marginal contributions for feature feat and configuration
        config_run for all examples
    """

    print(f'Computing MC for feature {feat} in run {config_run}')

    # get score of run that does not use the feature
    config_run_minusfeat = '-'.join([el for el in feats_in_run if el != feat])

    for example_i, example in enumerate(runs_tbl.iterrows()):  # iterate through examples

        if example_i % 1000 == 0:
            print(f'Computing MC for feature {feat} in run {config_run} for example {example[0]} '
                  f'({example_i + 1} out of {len(runs_tbl)})')

        # get score of run that includes the feature
        score_config_run = example[1][
            'score']  # runs_tbl.xs(config_run, axis=0, level='run').loc[example[0][:2], 'score']

        if len(config_run_minusfeat) == 0:
            score_config_run_minusfeat = score_nofeats
        else:
            score_config_run_minusfeat = \
                runs_tbl.xs(config_run_minusfeat, axis=0, level='run').loc[example[0][:2], 'score']

        # compute marginal contribution for adding feat to set of features in this run
        examples_tbl.loc[example[0][:2], feat] = \
            examples_tbl.loc[example[0][:2], feat] + \
            (n_feats_in_run * comb(n_features, n_feats_in_run)) ** (-1) * \
            (score_config_run - score_config_run_minusfeat)

    return examples_tbl


if __name__ == "__main__":

    # used in job arrays
    parser = argparse.ArgumentParser()
    parser.add_argument('--job_idx', type=int, help='Job index', default=0)
    args = parser.parse_args()

    # get the configuration parameters
    path_to_yaml = Path('/home/msaragoc/Projects/Kepler-TESS_exoplanet/codebase/shap/config_agg_results.yaml')

    with(open(path_to_yaml, 'r')) as file:
        config = yaml.safe_load(file)

    # if config['run_parallel']:  # train models in parallel
    #     rank = MPI.COMM_WORLD.rank
    #     config['rank'] = config['ngpus_per_node'] * args.job_idx + rank
    #     config['size'] = MPI.COMM_WORLD.size
    #     print(f'Rank = {config["rank"]}/{config["size"] - 1}')
    #     sys.stdout.flush()
    # else:
    #     config['rank'] = 0
    rank = args.job_idx

    runs_tbl = pd.read_csv(config['runs_tbl_fp'], index_col=['target_id', 'tce_plnt_num', 'run'])
    examples_tbl = pd.read_csv(config['examples_tbl_fp'], index_col=['target_id', 'tce_plnt_num'])
    score_nofeats = examples_tbl.loc[examples_tbl['dataset'] == 'train']['label'].mean()
    n_features = len(config['features'])
    config_runs = np.array_split(np.load(config['config_runs_fp']), config['n_jobs'])[rank]

    for config_run_i, config_run in enumerate(config_runs):

        print(f'[{rank}] Computing MC for features in run {config_run} ({config_run_i + 1} ouf of {len(config_runs)})')

        feats_in_run = config_run.split('-')  # set of features used in the run
        n_feats_in_run = len(feats_in_run)
        print(f'[{rank}] Number of features in the run: {n_feats_in_run}')

        # get runs that are only related to this configuration
        runs_tbl_config_run = [runs_tbl.xs(config_run, axis=0, level='run', drop_level=False)]
        for feat in feats_in_run:
            config_run_minusfeat = '-'.join([el for el in feats_in_run if el != feat])
            if config_run_minusfeat != '':
                runs_tbl_config_run.append(runs_tbl.xs(config_run_minusfeat, axis=0, level='run', drop_level=False))
        runs_tbl_config_run = pd.concat(runs_tbl_config_run, axis=0)

        n_processes = min(config['n_processes'], n_feats_in_run)
        pool = multiprocessing.Pool(processes=n_processes)
        jobs = [(feat,) +
                (config_run, feats_in_run, runs_tbl_config_run, examples_tbl.copy(deep=True), score_nofeats, n_features)
                for feat_job_i, feat in enumerate(feats_in_run)]
        async_results = [pool.apply_async(compute_mc_for_feat_in_config_run, job) for job in jobs]
        pool.close()

        mc_tbls = [async_result.get()[config['features']] for async_result in async_results]
        mc_config_run_tbl = reduce(lambda df1, df2: df1.add(df2), mc_tbls)  # sum MCs across feats runs

        mc_config_run_tbl.to_csv(Path(config['results_dir']) / f'mc_values_configrun_{config_run}.csv')
