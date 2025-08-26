"""
Check which TCEs were processed by the ExoMiner pipeline.
"""

# 3rd party
import pandas as pd
from pathlib import Path

#%% setup

# mast DVR TCE tables directory
mast_dvr_tbls_dir = Path('/data3/exoplnt_dl/ephemeris_tables/tess/tess_spoc_2min/mast_dvr_tables')

# ExoMiner pipeline runs directory
exominer_pipeline_runs_dir = Path('/data3/exoplnt_dl/experiments/exominer_pipeline/runs')

#%% read DVR TCE tables

mast_dvr_tbls = {}
for fp in mast_dvr_tbls_dir.glob('*tcestats.csv'):
    
    sector_run = '-'.join([s_el[1:].lstrip('0') for s_el in fp.stem.split('-')[1:3]])[:-4]
    
    s_sector, e_sector = sector_run.split('-')
    if s_sector == e_sector:
        sector_tce_uid = f"S{s_sector}"
    else:
        sector_tce_uid = f"S{s_sector}-{e_sector}"
        
    mast_dvr_tbl = pd.read_csv(fp, usecols=['ticid', 'tce_plnt_num'], header=6)
    
    mast_dvr_tbl['uid'] = mast_dvr_tbl.apply(lambda tce: f"{tce['ticid']}-{tce['tce_plnt_num']}-{sector_tce_uid}", axis=1)
     
    mast_dvr_tbls[sector_run] = mast_dvr_tbl

#%% check how many TCEs were processed by comparing with the predictions files

for sector_run_id, mast_dvr_tbl in mast_dvr_tbls.items():
    
    print(f"Iterating on sector run {sector_run_id}...")
    
    # get the ExoMiner pipeline run directory for this sector run
    exominer_run_dir_lst = list(exominer_pipeline_runs_dir.glob(f'*tics_S{sector_run_id}*'))
    
    if len(exominer_run_dir_lst) == 1:
        print(f"Found ExoMiner pipeline run for sector run {sector_run_id}: {exominer_run_dir_lst[0]}")
    elif len(exominer_run_dir_lst) > 1:
        print(f"Multiple ExoMiner pipeline runs found for sector run {sector_run_id}: {exominer_run_dir_lst}")
        print(f'Using the first one: {exominer_run_dir_lst[0]}')
    else:
        print(f"No ExoMiner pipeline run found for sector run {sector_run_id}. Skipping")
        continue
    
    exominer_run_dir = exominer_run_dir_lst[0]
    
    # read predictions table
    pred_tbl_fp_lst = list(exominer_run_dir.glob('predictions*.csv'))
    
    if len(pred_tbl_fp_lst) == 0:
        print(f"No predictions files found in ExoMiner pipeline run directory for sector run {sector_run_id}. Skipping...")
        continue
    elif len(pred_tbl_fp_lst) > 1:
        raise ValueError(f"Multiple predictions files found in ExoMiner pipeline run directory for sector run {sector_run_id}: {pred_tbl_fp_lst}")
    
    pred_tbl_fp = pred_tbl_fp_lst[0]
    pred_tbl = pd.read_csv(pred_tbl_fp)
    
    # get all prediction files in the ExoMiner pipeline run directory
    pred_files = list(exominer_run_dir.glob('predictions*.csv'))
    
    num_tces_total = len(mast_dvr_tbl)
    tces_processed_idx = mast_dvr_tbl['uid'].isin(pred_tbl['uid'])
    num_tces_process = tces_processed_idx.sum()
    num_tces_notprocess = num_tces_total - num_tces_process
    
    print(f"Sector run {sector_run_id}: {num_tces_process} TCEs processed out of {num_tces_total} total TCEs. {num_tces_notprocess} TCEs not processed.")


# %%
