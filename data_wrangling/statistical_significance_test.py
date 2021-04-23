""" Exploring statistical signficance tests between the results obtained for ExoMiner and other reported models ."""

from pathlib import Path
from scipy.stats import normaltest, ttest_ind, t
import numpy as np

#%%

exps_dirs = [
    Path('/data5/tess_project/git_repo/trained_models/experiments_paper(9-14-2020_to_1-19-2021)/keplerdr25-dv_g301-l31_6tr_spline_nongapped_norobovetterkois_starshuffle_configK_secsymphase_wksnormmaxflux-wks_corrprimgap_ptempstat_albedostat_wstdepth_fwmstat_nopps_ckoiper_secparams_prad_per'),
    Path('/data5/tess_project/git_repo/trained_models/experiments_paper(9-14-2020_to_1-19-2021)/keplerdr25-dv_g2001-l201_9tr_spline_gapped_norobovetterkois_starshuffle_astronet_secsymphase_nopps_ckoiper'),
    Path('/data5/tess_project/git_repo/trained_models/experiments_paper(9-14-2020_to_1-19-2021)/keplerdr25-dv_g2001-l201_9tr_spline_gapped_norobovetterkois_starshuffle_exonet_secsymphase_nopps_ckoiper')
             ]

metrics = {
    exp_dir.stem: {'test_auc_pr': []
                   } for exp_dir in exps_dirs}

for exp_dir in exps_dirs:
    for model_dir in (exp_dir / 'models').iterdir():

        model_res = np.load(model_dir / 'results.npy', allow_pickle=True).item()
        for metric_name in metrics[exp_dir.stem]:
            if 'test' in metric_name:
                res_aux = model_res[metric_name]
            else:
                res_aux = model_res[metric_name][-1]
            metrics[exp_dir.stem][metric_name].append(res_aux)

#%% statistical test to confirm the metrics are normally distributed

test_metric = 'test_auc_pr'

alpha = 0.05
for exp_dir in exps_dirs:
    metrics[exp_dir.stem][f'{test_metric}_normal_stat'], metrics[exp_dir.stem][f'{test_metric}_normal_pval'] = \
        normaltest(metrics[exp_dir.stem][test_metric])

    print(f'[{exp_dir.stem}] p-value {metrics[exp_dir.stem][f"{test_metric}_normal_pval"]} vs alpha {alpha}')

#%% confidence interval

conf = 0.95
test_metric = 'test_auc_pr'

for exp_dir in exps_dirs:
    n_samples = len(metrics[exp_dir.stem][test_metric])
    t_val = t.ppf(1 - conf / 3, n_samples - 1)
    delta = t_val * np.std(metrics[exp_dir.stem][test_metric], ddof=1) / np.sqrt(n_samples)
    metrics[exp_dir.stem][f'{test_metric}_conf_interval'] = np.mean(metrics[exp_dir.stem][test_metric]) \
                                                            + delta * np.array([-1, 1])

    print(f'[{exp_dir.stem}] CI ({conf}) {metrics[exp_dir.stem][f"{test_metric}_conf_interval"]}')

#%% Welch's t-test from 2 samples with different variance

alpha = 0.05
test_metric = 'test_auc_pr'

exp_to_compare = exps_dirs[0]

for exp_dir in exps_dirs:
    t_val, p_val = ttest_ind(metrics[exps_dirs[0].stem][test_metric], metrics[exp_dir.stem][test_metric],
                             equal_var=False)
    metrics[exps_dirs[0].stem][f'{test_metric}_2sample_noequal_var_welchttest'] = {'t': t_val, 'p-value': p_val}

    print(f'[{exp_dir.stem}] Welch\'s t-test 2 samples not same var (alpha={alpha}) '
          f'{metrics[exps_dirs[0].stem][f"{test_metric}_2sample_noequal_var_welchttest"]}')
