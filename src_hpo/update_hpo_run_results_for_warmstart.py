"""
Update HPO results for warmstart which includes removing invalid configurations/runs.
"""

# 3rd party
from hpbandster.core.result import logged_results_to_HBS_result
from pathlib import Path
import json
import numpy as np
import shutil


def clean_hpo_run_results_for_warmstart(hpo_run_fp):
    """ Clean configs.json and results.json from invalid configurations and runs, respectively (i.e., with HPO loss
    set to infinity or `None`).

    :param hpo_run_fp: Path, HPO run directory

    :return:
    """

    res = logged_results_to_HBS_result(str(hpo_run_fp))

    config_fp = hpo_run_fp / 'configs.json'
    results_fp = hpo_run_fp / 'results.json'

    shutil.copyfile(config_fp, hpo_run_fp / 'configs_original.json')
    shutil.copyfile(results_fp, hpo_run_fp / 'results_original.json')

    # get list of failed sampled configurations
    failed_configs = []
    for config_id, config in res.data.items():

        for config_budget, config_budget_res in config.results.items():
            if config_budget_res:
                if not np.isfinite(config_budget_res['loss']):
                    failed_configs.append(config_id)
                    break
            else:
                failed_configs.append(config_id)

    print(f'Number of failed configs: {len(failed_configs)}')

    # load both results and config files
    results = []
    with open(hpo_run_fp / 'results_original.json') as f:
        for line in f:
            results.append(json.loads(line))

    configs = []
    with open(hpo_run_fp / 'configs_original.json', 'r') as f:
        for line in f:
            configs.append(json.loads(line))

    # keep only valid results
    clean_results, clean_configs_ids = [], []
    for result_list in results:
        if result_list[3]:
            if np.isfinite(result_list[3]['loss']):
                clean_results.append(result_list)
                config_id = result_list[0]
                if config_id not in clean_configs_ids:
                    clean_configs_ids.append(config_id)

    clean_configs = []
    for config_list in configs:
        if config_list[0] in clean_configs_ids:
            clean_configs.append(config_list)

    # Save cleaned files
    with open(results_fp, 'w') as f:
        for result_list in clean_results:
            json.dump(result_list, f)
            f.write('\n')

    with open(config_fp, 'w') as f:
        for clean_config in clean_configs:
            json.dump(clean_config, f)
            f.write('\n')

if __name__ == "__main__":

    hpo_run_dir = Path('/home6/msaragoc/work_dir/Kepler-TESS_exoplanet/experiments/tess_spoc_ffi/hpo_configs/hporun_newexominer_tessspoc_2mins1-s88_notsharedtargetsffi_5-22-2025_1544/')

    clean_hpo_run_results_for_warmstart(hpo_run_dir)
