"""
Wrapper for preprocess_diff_img.py that iterates over NumPy files that contain the raw difference image data extracted
from the TESS DV xml files for different sector runs.
"""

# 3rd party
from pathlib import Path
import subprocess
from datetime import datetime

# file path to script that preprocesses difference image data for a sector run (i.e, for a DV xml file)
script_fp = '/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/codebase/diff_img/preprocessing/preprocess_diff_img.py'

# path to directory with difference image data for the different sector runs
diff_img_data_dir = Path('/Users/msaragoc/Projects/exoplanet_transit_classification/data/fits_files/tess/2min_cadence_data/dv/preprocessing/3-1-2023_1422/data')
# list of file paths to DV NumPy files for the sector runs to be preprocessed
diff_img_data_fps = diff_img_data_dir.iterdir()

# path to directory with quality metric tables for the different sector runs
# check to see if the quality metric table for each sector run is available
qual_metrics_tbl_dir = Path('/Users/msaragoc/Projects/exoplanet_transit_classification/data/fits_files/tess/2min_cadence_data/dv/diff_img_quality_metric')
# qual_metrics_tbl_fps = qual_metrics_tbl_dir.iterdir()

# destination root directory for all the preprocessed data
dest_root_dir = Path(f'/Users/msaragoc/Projects/exoplanet_transit_classification/data/fits_files/tess/2min_cadence_data/dv/preprocessing_step2/{datetime.now().strftime("%m-%d-%Y_%H%M")}')
dest_root_dir.mkdir()

args = {
    'mission': 'tess',   # either `kepler` or `tess`
    # destination file path to preprocessed data; set under `dest_root_dir` once the run starts
    'dest_dir': '',
    # file path to table with information on saturated stars
    'sat_tbl_fp': '/Users/msaragoc/Projects/exoplanet_transit_classification/data/ephemeris_tables/tess/DV_SPOC_mat_files/10-05-2022_1338/tess_tces_dv_s1-s55_10-05-2022_1338_ticstellar_ruwe_tec_tsoebs_ourmatch_preproc.csv',
    # saturated target threshold
    'sat_thr': 7,
    # number of quarters/sectors to get
    'num_sampled_imgs': 5,
    # padding original images with this number of pixels
    'pad_n_pxs': 20,
    # dimension of final image (final_dim, final_dim)
    'final_dim': 11,
    # number of processes used to parallelize work for a given sector run; number of jobs running in simultaneous
    'n_processes': 4,
    # number of jobs in total
    'n_jobs': 4,
}

for diff_img_data_fp in diff_img_data_fps:

    args['diff_img_tbl_fp'] = str(diff_img_data_fp)

    print(f'Started processing data from {diff_img_data_fp}...')

    # find quality metric for that sector run
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

    args['dest_dir'] = str(dest_dir)

    # get file path to corresponding quality metric table
    qual_metrics_tbl_fp = qual_metrics_tbl_dir / qual_metrics_tbl_fn
    if not qual_metrics_tbl_fp.exists():
        raise FileNotFoundError(f'File not found for quality metric {qual_metrics_tbl_fp} ({qual_metrics_tbl_fn}).')
    print(f'Using quality metrics in table {qual_metrics_tbl_fp}')

    args['qual_metrics_tbl_fp'] = str(qual_metrics_tbl_fp)

    args_list = []
    for arg_name, arg_val in args.items():
        args_list.append(f'--{arg_name}')
        args_list.append(f'{arg_val}')
    python_command = ['python', script_fp] + args_list
    print(python_command)

    p = subprocess.run(python_command)
    while p.returncode != 0:  # only start new process once this one finishes
        continue
    print(f'Finished process used to preprocess data in {diff_img_data_fp}.')
