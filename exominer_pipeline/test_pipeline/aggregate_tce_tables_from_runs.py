"""Aggregate TCE tables from multiple ExoMiner runs into a single table. """

# 3rd party
import pandas as pd
from pathlib import Path

#%% setup

runs_root_dir = Path('/data3/exoplnt_dl/experiments/exominer_pipeline/runs')

runs_dirs = list(runs_root_dir.glob('exominer_pipeline_run_*'))
print(f"Found {len(runs_dirs)} ExoMiner Pipeline runs.")

#%% aggregate tce tables from all runs

tce_tbls_runs_fps = []
for run_dir in runs_dirs:
    
    tce_tbls_run_fps = list(run_dir.rglob('tess-spoc-dv_tces_*_processed.csv'))
    print(f"Run {run_dir.name} has {len(tce_tbls_run_fps)} TCE tables.")
    if len(tce_tbls_run_fps) > 0:
        tce_tbls_runs_fps += tce_tbls_run_fps
    else:
        print(f"Run {run_dir.name} does not have a TCE table. Skipping.")

tce_tbl_agg = pd.concat([pd.read_csv(fp) for fp in tce_tbls_runs_fps], ignore_index=True, axis=0)

# get duplicates TCEs
duplicates = tce_tbl_agg[tce_tbl_agg.duplicated(subset='uid', keep=False)]
print(f"Found {len(duplicates['uid'].unique())} duplicate TCEs across runs.")

# drop duplicates TCEs
tce_tbl_agg = tce_tbl_agg.drop_duplicates(subset='uid', keep='first')
print(f'Processed {len(tce_tbl_agg)} TCEs across runs.')

tce_tbl_agg.to_csv('/data3/exoplnt_dl/ephemeris_tables/tess/tess_spoc_2min/tess-spoc-2min-tces-dv_s68s94_aggregated_9-10-2025_1440.csv', index=False)
