"""
Test ExoMiner pipeline.
"""

# 3rd party
import pandas as pd
from pathlib import Path
from datetime import datetime

# local
from exominer_pipeline.run_pipeline import run_exominer_pipeline_main

if __name__ == '__main__':

    exp_run_dir = Path(f'/Users/msaragoc/Downloads/exominer_pipeline_run_{datetime.now().strftime("%Y%m%d-%H%M%S")}')
    exp_run_dir.mkdir(parents=True, exist_ok=True)

    # create CSV file with TICs
    tics_tbl = pd.DataFrame(
        data = {
            'tic_id': [
                13829713,
            ],
            'sector_run': [
                '14-60',
            ]
        }
    )
    tics_tbl_fp = exp_run_dir / 'tics_tbl.csv'
    tics_tbl.to_csv(tics_tbl_fp)

    exominer_pipeline_config_fp = '/Users/msaragoc/Projects/exoplanet_transit_classification/exoplanet_dl/exominer_pipeline/exominer_config.yaml'
    data_collection_mode = '2min'
    num_processes = 1
    run_exominer_pipeline_main(
        config_fp=exominer_pipeline_config_fp,
        output_dir=str(exp_run_dir),
        tic_ids_fp=str(tics_tbl_fp),
        data_collection_mode=data_collection_mode,
        num_processes=num_processes,
    )
