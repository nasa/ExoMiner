"""
Wrapper for preprocess_diff_img.py that iterates over NumPy files that contain the raw difference image data extracted
from the TESS DV xml files for different sector runs.
"""

# 3rd party
from pathlib import Path
import subprocess
from datetime import datetime
import yaml

# file path to script that preprocesses difference image data for a sector run (i.e, for a DV xml file)
script_fp = '/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/codebase/src_preprocessing/diff_img/preprocessing/preprocess_diff_img.py'

# path to directory with difference image data for the different sector runs
diff_img_data_dir = Path('/Users/msaragoc/Projects/exoplanet_transit_classification/data/fits_files/tess/2min_cadence_data/dv/preprocessing/11-17-2023_1400/data')
# list of file paths to DV NumPy files for the sector runs to be preprocessed
diff_img_data_fps = diff_img_data_dir.iterdir()

# destination root directory for all the preprocessed data
dest_root_dir = Path(
    f'/Users/msaragoc/Projects/exoplanet_transit_classification/data/fits_files/tess/2min_cadence_data/dv/preprocessing_step2/{datetime.now().strftime("%m-%d-%Y_%H%M")}')
dest_root_dir.mkdir(exist_ok=True)

qual_metrics_tbl_dir = Path('/Users/msaragoc/Projects/exoplanet_transit_classification/data/fits_files/tess/2min_cadence_data/dv/diff_img_quality_metric')

# set file path for default config file
default_config_fp = Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/codebase/src_preprocessing/diff_img/preprocessing/config_preprocessing.yaml')

for diff_img_data_fp in diff_img_data_fps:

    # load yaml file with run setup
    with(open(default_config_fp, 'r')) as file:
        config = yaml.safe_load(file)

    config['diff_img_data_fp'] = str(diff_img_data_fp)

    print(f'Started processing data from {diff_img_data_fp}...')

    # set sector run; find quality metric for that sector run
    sector_run = diff_img_data_fp.stem.split('_')[3]

    if 'multisector' in diff_img_data_fp.name:  # multi-sector run

        s_sector, e_sector = [int(s[1:]) for s in sector_run.split('-')]
        qual_metrics_tbl_fn = f'diff_img_quality_metric_tess_{s_sector}-{e_sector}.csv'

        # set destination directory for results of this sector run
        dest_dir = dest_root_dir / f'multisector_s{str(s_sector).zfill(4)}-s{str(e_sector).zfill(4)}'

    else:  # single-sector run
        qual_metrics_tbl_fn = f'diff_img_quality_metric_tess_{sector_run}.csv'
        s_sector, e_sector = sector_run, sector_run

        dest_dir = dest_root_dir / f'sector_{s_sector}'

    # set destination directory for sector run
    dest_dir.mkdir(exist_ok=True)
    config['dest_dir'] = str(dest_dir)

    # get file path to corresponding quality metric table
    qual_metrics_tbl_fp = qual_metrics_tbl_dir / qual_metrics_tbl_fn
    if not qual_metrics_tbl_fp.exists():
        raise FileNotFoundError(f'File not found for quality metric {qual_metrics_tbl_fp} ({qual_metrics_tbl_fn}).')
    print(f'Using quality metrics in table {qual_metrics_tbl_fp}')

    config['qual_metrics_tbl_fp'] = str(qual_metrics_tbl_fp)

    # save config file for sector run
    config_fp = dest_dir / 'run_params.yaml'
    with open(str(config_fp), 'w') as file:
        yaml.dump(config, file, sort_keys=False)

    args = {
        'config_fp': str(config_fp),
    }
    args_list = []
    for arg_name, arg_val in args.items():
        args_list.append(f'--{arg_name}')
        args_list.append(f'{arg_val}')
    python_command = ['python', script_fp] + args_list
    print(''.join(python_command))

    p = subprocess.run(python_command)
    while p.returncode != 0:  # only start new process once this one finishes
        continue
    print(f'Finished process used to preprocess data in {diff_img_data_fp}.')
