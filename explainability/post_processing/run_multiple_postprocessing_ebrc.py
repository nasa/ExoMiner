"""
Run postprocessing analysis for EBRC experiments for different runs.
"""

# 3rd party
import json
import subprocess
from pathlib import Path


def run_notebook(notebook_file, **arguments):
    """ Pass arguments to a Jupyter notebook, run it and convert to html.

    Args:
        notebook_file: str, Jupyter notebook file path
        **arguments: dict, arguments for notebook

    Returns:

    """

    # Create the arguments file
    with open('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/experiments/explainability/model_pc_replacement/arguments_postprocessing.json', 'w') as fid:
        json.dump(arguments, fid)
    # Run the notebook
    subprocess.call([
        'jupyter-nbconvert',
        '--execute',
        '--to', 'html',
        # '--output', output_file,
        notebook_file
    ])

notebook_fp = '/Users/msaragoc/OneDrive - NASA/Projects/exoplanet_transit_classification/codebase/explainability/post_processing/modelpc_replacement_analysis.ipynb'
exp_root_dir = Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/experiments/explainability/model_pc_replacement/')
exp_dirs_paths = [fp for fp in exp_root_dir.iterdir() if fp.name.startswith('run_num_model_pcs_')]

# Run the notebook with different arguments
for exp_dir_path in exp_dirs_paths:
    print(f'Running postprocessing analysis for experiment {exp_dir_path.name}...')
    # n_pcs = int(exp_dir_path.name.split('_')[4])
    # if n_pcs not in [1, 2, 3, 4, 6, 7, 8]:
    #     continue
    run_notebook(notebook_fp, exp_dir_fp=str(exp_dir_path))

print('Done.')
