"""

"""

# 3rd party
from pathlib import Path
import subprocess
from datetime import datetime

script_fp = '/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/codebase/diff_img/preprocessing/preprocess_diff_img.py'

# path to directory with difference image data for the different sector runs
diff_img_data_dir = Path('/Users/msaragoc/Projects/exoplanet_transit_classification/data/fits_files/tess/2min_cadence_data/dv/preprocessing/3-1-2023_1422/data')
diff_img_data_fps = diff_img_data_dir.iterdir()

# path to directory with quality metric tables for the different sector runs
qual_metrics_tbl_dir = Path('/Users/msaragoc/Projects/exoplanet_transit_classification/data/fits_files/tess/2min_cadence_data/dv/diff_img_quality_metric')
# qual_metrics_tbl_fps = qual_metrics_tbl_dir.iterdir()

args = {
    'mission': 'tess',   # either `kepler` or `tess`
    # destination file path to preprocessed data
    'dest_dir': f'/Users/msaragoc/Projects/exoplanet_transit_classification/data/fits_files/tess/2min_cadence_data/dv/preprocessing_step2/{datetime.now().strftime("%m-%d-%Y_%H%M")}',
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
}

for diff_img_data_fp in diff_img_data_fps:

    args['diff_img_tbl_fp'] = str(diff_img_data_fp)

    print(f'Started processing data from {diff_img_data_fp}...')

    # find quality metric for that sector run
    sector_run = diff_img_data_fp.stem.split('_')[3]
    if 'multisector' in diff_img_data_fp.name:  # multi-sector run
        s_sector, e_sector = [int(s[1:]) for s in sector_run.split('-')]
        qual_metrics_tbl_fn = f'diff_img_quality_metric_tess_{s_sector}-{e_sector}.csv'
        # args['n_max_imgs_avail'] = e_sector - s_sector + 1
    else:  # single-sector run
        qual_metrics_tbl_fn = f'diff_img_quality_metric_tess_{sector_run}.csv'
        # args['n_max_imgs_avail'] = 1

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
