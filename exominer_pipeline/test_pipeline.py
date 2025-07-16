"""
Test ExoMiner pipeline.
"""

# 3rd party
import pandas as pd
from pathlib import Path
from datetime import datetime
from time import perf_counter

# local
from exominer_pipeline.run_pipeline import run_exominer_pipeline_main

if __name__ == '__main__':

    import tensorflow as tf

    physical_devices = tf.config.list_physical_devices('CPU')
    tf.config.set_visible_devices(physical_devices, 'CPU')

    # exp_run_dir = Path(f'/Users/msaragoc/Downloads/exominer_pipeline_run_{datetime.now().strftime("%Y%m%d-%H%M%S")}')
    # exp_run_dir = Path('/Users/msaragoc/Downloads/exominer_pipeline_run_20250701-130322/')
    exp_run_dir = Path(f'/Users/msaragoc/Projects/exoplanet_transit_classification/experiments/exominer_pipeline/runs/exominer_pipeline_run_{datetime.now().strftime("%Y%m%d-%H%M%S")}')
    exp_run_dir.mkdir(parents=True, exist_ok=True)

    # # create CSV file with TICs
    # twomin = pd.read_csv('/Users/msaragoc/Projects/exoplanet_transit_classification/data/ephemeris_tables/tess/tess_spoc_2min/tess_2min_tces_dv_s1-s88_3-27-2025_1316_label.csv')
    # twomin = twomin.loc[twomin['uid'] == '273574141-1-S14']
    # # twomin = twomin.loc[twomin['sector_run'] == '36']

    # tics_tbl = twomin[['target_id', 'sector_run']]
    # tics_tbl['sector_run'] = tics_tbl['sector_run'].apply(lambda x: f'{x}-{x}' if '-' not in x else x)
    # tics_tbl = tics_tbl.drop_duplicates(subset=['target_id', 'sector_run'])
    # tics_tbl = tics_tbl.rename(columns={'target_id': 'tic_id'})

    tics_tbl = pd.DataFrame(
        data = {
            'tic_id': [
                # 167526485,
                # 167526485,
                # 167526485,
                # 184240683,  # non-existing ffi
                356473034,  # ffi
            ],
            'sector_run': [
                # '6-6',
                # '7-7',
                # '1-39',
                # '29-29',  # non-existing ffi
                '60-60',
            ]
        }
    )
    # pred_tbl_prev_exp = pd.read_csv('/Users/msaragoc/Downloads/exominer_pipeline_run_20250630-174917/predictions_exominer_pipeline_run_20250630-174917.csv')
    # tics_tbl = tics_tbl.loc[~tics_tbl['tic_id'].isin(pred_tbl_prev_exp['target_id'])]

    tics_tbl_fp = exp_run_dir / 'tics_tbl.csv'
    tics_tbl.to_csv(tics_tbl_fp, index=False)

    pipeline_config_fp = 'exominer_pipeline/pipeline_run_config.yaml'
    data_collection_mode = 'ffi'
    num_processes = 1
    num_jobs = 1
    start_t = perf_counter()
    print(f'OUTPUT DIRECTORY: {exp_run_dir}')
    run_exominer_pipeline_main(
        # config_fp=pipeline_config_fp,
        output_dir=str(exp_run_dir),
        tic_ids_fp=str(tics_tbl_fp),
        data_collection_mode=data_collection_mode,
        num_processes=num_processes,
        num_jobs=num_jobs,
    )
    end_t = perf_counter()
    print(f'Elapsed time: {end_t - start_t:.2f} seconds.')
