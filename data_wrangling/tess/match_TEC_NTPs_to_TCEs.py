""" Using TEC flux triage tables to assign TCEs as NTPs for TESS data. """

# 3rd party
import pandas as pd
from pathlib import Path

# %% Processing TEC flux triage tables

tec_dir = Path('/data5/tess_project/Data/Ephemeris_tables/TESS/TEC_SPOC')

tec_tbls = []
tec_tbls_dirs = [fp for fp in tec_dir.iterdir() if fp.is_dir()]
for run_dir in tec_tbls_dirs:
    print(f'Iterating over {run_dir}')
    triage0_fps = [fp for fp in run_dir.iterdir() if 'fluxtriage' in fp.stem]
    if len(triage0_fps) == 0:
        print(f'No triage file for {run_dir}')
        continue
    assert len(triage0_fps) == 1
    triage0_fp = triage0_fps[0]
    print(f'Reading table {triage0_fp}')

    tec_tbl = pd.read_csv(triage0_fp, names=['target_id', 'tce_plnt_num', 'pass', 'comment'], sep=r'\s+')

    sector_run = [el for el in triage0_fp.stem.split('_') if 'sector' in el][0][6:]
    print(f'Sector run: {sector_run}')
    tec_tbl['sector_run'] = sector_run
    tec_tbls.append(tec_tbl)

tec_tbl_full = pd.concat(tec_tbls)
tec_tbl_full.rename(columns={'pass': 'tec_fluxtriage_pass', 'comment': 'tec_fluxtriage_comment'}, inplace=True)
tec_tbl_full.to_csv(
    '/data5/tess_project/Data/Ephemeris_tables/TESS/DV_SPOC_mat_files/11-29-2021/tess_tces_s1-s40_11-23-2021_1409_stellarparams_updated.csv',
    index=False)

# tec_tbl_full['id'] = tec_tbl_full.apply(lambda x: f'{x["target_id"]}_{x["tce_plnt_num"]}_{x["sector_run"]}', axis=1)


