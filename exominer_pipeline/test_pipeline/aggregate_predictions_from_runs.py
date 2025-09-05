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
pred_runs.to_csv(runs_root_dir / 'predictions_exominer_pipeline_run_tics_aggregated_9-3-2025_1201.csv', index=False)

# get only target id/sector run pairs
target_sector_run_processed = pred_runs.drop_duplicates(subset=['target_id', 'sector_run'], keep='first')
print(f'Found {len(target_sector_run_processed)} unique target_id/sector_run pairs processed across runs.')
target_sector_run_processed.to_csv(runs_root_dir / 'target_sector_run_pairs_processed_8-25-2025_1559.csv', index=False)

#%% get number of processed TCEs per sector run vs number of TCEs

mast_dvr_tbls_dir = Path('/data3/exoplnt_dl/ephemeris_tables/tess/tess_spoc_2min/mast_dvr_tables')
mast_dvr_fps = list(mast_dvr_tbls_dir.glob('*.csv'))
print(f"Found {len(mast_dvr_fps)} MAST DVR tables.")

sector_run_tce_cnt = pred_runs['sector_run'].value_counts().reset_index(name='num_tces_processed')
mast_dvr_tce_cnt = {field: [] for field in ['sector_run', 'num_tces_total']}
for mast_dvr_fp in mast_dvr_fps:
    
    mast_dvr_tbl = pd.read_csv(mast_dvr_fp, header=6)
    mast_dvr_tbl_sector_run = '-'.join([str(int(el[1:])) for el in mast_dvr_fp.stem.split('_')[0].split('-')[1:3]])
    mast_dvr_tce_cnt['sector_run'].append(mast_dvr_tbl_sector_run)
    mast_dvr_tce_cnt['num_tces_total'].append(len(mast_dvr_tbl))

mast_dvr_tce_cnt = pd.DataFrame(mast_dvr_tce_cnt)
sector_run_comparison = mast_dvr_tce_cnt.merge(sector_run_tce_cnt, on='sector_run', how='left', validate='one_to_one')

sector_run_comparison['num_tces_not_processed'] = sector_run_comparison['num_tces_total'] - sector_run_comparison['num_tces_processed']

sector_run_comparison.to_csv(runs_root_dir / 'sector_runs_tce_processing_9-3-2025_1201.csv', index=False)