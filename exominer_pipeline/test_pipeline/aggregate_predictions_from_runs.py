"""
Aggregate predictions from multiple runs of the ExoMiner pipeline.
"""

# 3rd party
import pandas as pd
from pathlib import Path

#%% setup

runs_root_dir = Path('/data3/exoplnt_dl/experiments/exominer_pipeline/runs')

#%% generate prediction table for run in case the run did not finish but some jobs still finished

runs_dirs = list(runs_root_dir.glob("exominer_pipeline_run_*"))

for run_dir in runs_dirs:

    pred_fps_cand = list(run_dir.glob('predictions*.csv'))
    if len(pred_fps_cand) == 1:
        print(f"Run {run_dir.name} already has a predictions table.")
        continue
    elif len(pred_fps_cand) > 1:
        raise ValueError(f"Run {run_dir.name} has multiple predictions tables.")
    else:
    
        pred_tbls_fps = list(run_dir.rglob("ranked_predictions_predictset.csv"))
        if len(pred_tbls_fps) > 0:
            print(f"Found {len(pred_tbls_fps)} prediction tables in {run_dir.name}. Aggregating...")
            pred_tbl = pd.concat([pd.read_csv(fp) for fp in pred_tbls_fps], ignore_index=True, axis=0)
            pred_tbl.to_csv(run_dir / f'predictions_{run_dir.stem}.csv', index=False)
            print(f"Aggregated predictions table saved to {run_dir / f'predictions_{run_dir.stem}.csv'}.")

#%% aggregate predictions from all runs

pred_runs_fps = []
for run_dir in runs_dirs:
    
    pred_fps_cand = list(run_dir.glob('predictions*.csv'))
    if len(pred_fps_cand) == 1:
        pred_runs_fps.append(pred_fps_cand[0])
    elif len(pred_fps_cand) > 1:
        raise ValueError(f"Run {run_dir.name} has multiple predictions tables.")
    else:
        print(f"Run {run_dir.name} does not have a predictions table. Skipping.")

pred_runs = pd.concat([pd.read_csv(fp) for fp in pred_runs_fps], ignore_index=True, axis=0)

# get duplicates TCEs
duplicates = pred_runs[pred_runs.duplicated(subset='uid', keep=False)]
print(f"Found {len(duplicates['uid'].unique())} duplicate TCEs across runs.")
duplicates_sorted = duplicates.sort_values(by='uid')
duplicates_sorted.to_csv(runs_root_dir / 'duplicate_tces_across_runs.csv', index=False)

# drop duplicates TCEs
pred_runs = pred_runs.drop_duplicates(subset='uid', keep='first')
print(f'Processed {len(pred_runs)} TCEs across runs.')

# get sector run ID
def create_sector_run_id(tce):
    
    sector_run = tce['uid'].split('-')[2:]
    if len(sector_run) == 2:
        sector_run_id = '-'.join(sector_run)[1:]
    elif len(sector_run) == 1:
        sector_run_id = f'{sector_run[0][1:]}-{sector_run[0][1:]}'
    else:
        raise ValueError(f"Invalid uid format: {tce['uid']}")
    
    return sector_run_id

pred_runs['sector_run'] = pred_runs.apply(create_sector_run_id, axis=1)

# get only target id/sector run pairs
target_sector_run_processed = pred_runs.drop_duplicates(subset=['target_id', 'sector_run'], keep='first')
print(f'Found {len(target_sector_run_processed)} unique target_id/sector_run pairs processed across runs.')
target_sector_run_processed.to_csv(runs_root_dir / 'target_sector_run_pairs_processed_8-25-2025_1559.csv', index=False)
