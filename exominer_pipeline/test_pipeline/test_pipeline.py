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
        
    exp_run_dir = Path(f'/data3/exoplnt_dl/experiments/exominer_pipeline/runs/exominer_pipeline_run_unseen_tois_8-20-2025') # {datetime.now().strftime("%Y%m%d-%H%M%S")}')
    # exp_run_dir = Path('/Users/msaragoc/Downloads/exominer_pipeline_run_20250701-130322/')
    # exp_run_dir = Path(f'/Users/msaragoc/Projects/exoplanet_transit_classification/experiments/exominer_pipeline/runs/exominer_pipeline_run_tic356473034.1_s60_2min_s14-78_nodetrending_{datetime.now().strftime("%Y%m%d-%H%M%S")}')
    exp_run_dir.mkdir(parents=True, exist_ok=True)

    # # # create CSV file with TICs
    # twomin = pd.read_csv('/Users/msaragoc/Projects/exoplanet_transit_classification/data/ephemeris_tables/tess/tess_spoc_2min/tess_2min_tces_dv_s1-s88_3-27-2025_1316_label.csv')
    # twomin = twomin.loc[((twomin['target_id'] == 235678745) & (twomin['sector_run'] == '14-78'))]
    # # # twomin = twomin.loc[twomin['sector_run'] == '36']

    # tics_tbl = twomin[['target_id', 'sector_run']]
    # tics_tbl['sector_run'] = tics_tbl['sector_run'].apply(lambda x: f'{x}-{x}' if '-' not in x else x)
    # tics_tbl = tics_tbl.drop_duplicates(subset=['target_id', 'sector_run'])
    # tics_tbl = tics_tbl.rename(columns={'target_id': 'tic_id'})

    tics_tbl_fp = Path('/data3/exoplnt_dl/experiments/exominer_pipeline/inputs/tois_in_unseen_tics_8-20-2025.csv')
    # tics_tbl = pd.read_csv('/Users/msaragoc/Projects/exoplanet_transit_classification/experiments/exominer_pipeline/inputs/tics_tbl_356473034_S60.csv')

    # tics_tbl = pd.DataFrame(
    #     data = {
    #         'tic_id': [
    #             # 167526485,
    #             # 167526485,
    #             # 167526485,
    #             # 184240683,  # non-existing ffi
    #             # 356473034,  # both ffi and 2min
    #             # 420114776,
    #             235678745
    #         ],
    #         'sector_run': [
    #             # '6-6',
    #             # '7-7',
    #             # '1-39',
    #             # '29-29',  # non-existing ffi
    #             # '60-60',
    #             # '24-24',
    #             '14-86',
    #         ]
    #     }
    # )
    # # pred_tbl_prev_exp = pd.read_csv('/Users/msaragoc/Downloads/exominer_pipeline_run_20250630-174917/predictions_exominer_pipeline_run_20250630-174917.csv')
    # # tics_tbl = tics_tbl.loc[~tics_tbl['tic_id'].isin(pred_tbl_prev_exp['target_id'])]

    # tics_tbl_fp = exp_run_dir / 'tics_tbl.csv'
    # tics_tbl.to_csv(tics_tbl_fp, index=False)

    # stellar_tbl = pd.DataFrame(
    #     data = {
    #         'target_id': [356473034],
    #         'tic_steff': [0],
    #          'tic_steff_err': [0],
    #          'tic_smass': [0],
    #          'tic_smass_err': [0],
    #          'tic_smet': [0],
    #          'tic_smet_err': [0],
    #          'tic_sradius': [0],
    #          'tic_sradius_err': [0],
    #          'tic_sdens': [0],
    #          'tic_sdens_err': [0],
    #          'tic_slogg': [0],
    #          'tic_slogg_err': [0],
    #          'tic_ra': [0],
    #          'tic_dec': [0],
    #          'kic_id': [0],
    #          'gaia_id': [0],
    #          'tic_tmag': [0],
    #          'tic_tmag_err': [0],
    #     }
    # )
    # stellar_tbl_fp = '/Users/msaragoc/Projects/exoplanet_transit_classification/experiments/exominer_pipeline/source_catalogs/tic_stellar_parameters.csv'
    # stellar_tbl.to_csv(stellar_tbl_fp, index=False)
    #
    # ruwe_tbl = pd.DataFrame(
    #     data={
    #         'target_id': [356473034],
    #         'ruwe': [1.0],
    #
    #           }
    #                         )
    # ruwe_tbl_fp = '/Users/msaragoc/Projects/exoplanet_transit_classification/experiments/exominer_pipeline/source_catalogs/tic_ruwe.csv'
    # ruwe_tbl.to_csv(ruwe_tbl_fp, index=False)

    # pipeline_config_fp = 'exominer_pipeline/pipeline_run_config.yaml'
    data_collection_mode = '2min'
    num_processes = 10
    num_jobs = 60
    download_spoc_data_products = 'false'
    external_data_repository = None   # '/Users/msaragoc/Projects/exoplanet_transit_classification/experiments/exominer_pipeline/runs/exominer_pipeline_run_7-18-2025_1213/job_0/mastDownload'
    stellar_parameters_source = 'ticv8'  # stellar_tbl_fp
    ruwe_source = 'gaiadr2'

    start_t = perf_counter()
    print(f'OUTPUT DIRECTORY: {exp_run_dir}')
    run_exominer_pipeline_main(
        output_dir=str(exp_run_dir),
        tic_ids_fp=str(tics_tbl_fp),
        data_collection_mode=data_collection_mode,
        num_processes=num_processes,
        num_jobs=num_jobs,
        download_spoc_data_products=download_spoc_data_products,
        external_data_repository=external_data_repository,
        stellar_parameters_source=stellar_parameters_source,
        ruwe_source=ruwe_source,
    )
    end_t = perf_counter()
    print(f'Elapsed time: {end_t - start_t:.2f} seconds.')
