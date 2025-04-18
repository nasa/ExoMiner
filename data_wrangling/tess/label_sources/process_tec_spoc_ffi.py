"""
Script used to preprocess TEC results for TESS SPOC FFI
"""

# 3rd party
import pandas as pd
from pathlib import Path

#%%

tec_spoc_ffi_dir = Path('/nobackupp18/dacaldwe/FTL/TCEs/TEC_results')
save_fp = Path('/nobackup/msaragoc/work_dir/Kepler-TESS_exoplanet/data/Ephemeris_tables/TESS/tess_spoc_ffi/tec_fluxtriage_4-15-2025_1330.csv')

#%%

# get flux triage files
flux_triage_fps = list(tec_spoc_ffi_dir.glob('*fluxtriage*'))

# exclude redundant files
flux_triage_fps = [fp for fp in flux_triage_fps if 'orig' not in fp.name]

tec_tbls = []
for fp in flux_triage_fps:

    tec_tbl = pd.read_csv(fp, names=['target_id', 'tce_plnt_num', 'pass', 'comment'], sep=r'\s+')

    sector_run = [el for el in fp.stem.split('_') if 'sector' in el][0][6:8]
    print(f'Sector run: {sector_run}')
    tec_tbl['sector_run'] = sector_run
    tec_tbls.append(tec_tbl)

tec_tbl_full = pd.concat(tec_tbls, axis=0)

# setting tces uids
tec_tbl_full['uid'] = tec_tbl_full[['target_id', 'tce_plnt_num', 'sector_run']].apply(lambda x: f'{x["target_id"]}-{x["tce_plnt_num"]}-S{x["sector_run"]}', axis=1)

tec_tbl_full.rename(columns={'pass': 'tec_fluxtriage_pass', 'comment': 'tec_fluxtriage_comment'}, inplace=True)

tec_tbl_full.to_csv(save_fp, index=False)
