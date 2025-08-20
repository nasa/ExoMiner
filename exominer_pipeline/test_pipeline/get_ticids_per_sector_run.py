""" Grab TIC IDs for targets observed in a given sector run from the corresponding MAST DVR TCE table and store them in a new CSV file so they can be used as input tables to the ExoMiner pipeline. """

# 3rd party
import pandas as pd
from pathlib import Path

#%% setup

mast_dvr_tbls_dir = Path('/data3/exoplnt_dl/ephemeris_tables/tess/tess_spoc_2min/mast_dvr_tables')

mast_dvr_tbls_fps = list(mast_dvr_tbls_dir.glob('*tcestats.csv'))

print(f'Found {len(mast_dvr_tbls_fps)} MAST DVR TCE tables in {mast_dvr_tbls_dir}.')

#%% extract TIC IDs and store them in inputs tables for the ExoMiner pipeline

dest_dir = Path('/data3/exoplnt_dl/experiments/exominer_pipeline/inputs')

for fp_i, fp in enumerate(mast_dvr_tbls_fps):
    
    print(f'Iterating over table {fp.name} ({fp_i + 1}/{len(mast_dvr_tbls_fps)})...')
    mast_dvr_tbl = pd.read_csv(fp, usecols=['ticid'], header=6)
    
    mast_dvr_tbl.drop_duplicates(subset=['ticid'], inplace=True)
    
    mast_dvr_tbl.rename(columns={'ticid': 'tic_id'}, inplace=True)
    
    sector_run = '-'.join([str(int(sector_id[1:])) if sector_i == 0 else str(int(sector_id[1:-4])) for sector_i, sector_id in enumerate(fp.stem.split('-')[1:3])])
    
    mast_dvr_tbl['sector_run'] = sector_run
    
    print(f'Found {len(mast_dvr_tbl)} unique TIC IDs in {fp.name} for sector run {sector_run}.')
    
    mast_dvr_tbl.to_csv(dest_dir / f'tics_S{sector_run}.csv', index=False)
    