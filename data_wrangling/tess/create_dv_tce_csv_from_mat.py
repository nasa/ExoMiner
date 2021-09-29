""" Create TESS DV TCE csv files from the original mat files. """

# 3rd party
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

dv_root_dir = Path('/data5/tess_project/Data/Ephemeris_tables/TESS/DV_SPOC_mat_files')

ssector_runs_dir = dv_root_dir / 'single-sector'
msector_runs_dir = dv_root_dir / 'multi-sector'

ssector_runs_fps = sorted(list(ssector_runs_dir.iterdir()))
# FILTERING SINGLE-SECTOR RUNS
ssector_runs_fps = ssector_runs_fps[-6:]
msector_runs_fps = list(msector_runs_dir.iterdir())
# FILTERING MULTI-SECTOR RUNS
msector_runs_fps = [fp for fp in msector_runs_fps if fp.stem[13:] in ['0136', '0139']]

mat_runs_fps = ssector_runs_fps + msector_runs_fps

mat_runs_fps = [fp for fp in mat_runs_fps if fp.suffix == '.mat']

for mat_run_fp in mat_runs_fps:

    print(f'Creating csv table for mat file {mat_run_fp.name}')

    csv_save_dir = mat_run_fp.parent / 'csv_tables'
    csv_save_dir.mkdir(exist_ok=True)

    mat_struct = loadmat(mat_run_fp)

    cols = []
    cnt_no_name_col = 0
    for el_i, el in enumerate(mat_struct['dvOutputMatrixColumns']):

        try:
            cols.append(el[0][0])
        except:
            # print('EXCEPT', el[0])
            cols.append(f'no_name_{cnt_no_name_col}')
            cnt_no_name_col += 1

    num_tces = len(mat_struct['dvOutputMatrix'])

    tce_tbl = pd.DataFrame(columns=cols, data=np.nan * np.ones((num_tces, len(cols))))

    for tce_i, tce in enumerate(mat_struct['dvOutputMatrix']):
        tce_tbl.loc[tce_i] = tce

    tce_tbl.to_csv(csv_save_dir / f'{mat_run_fp.stem}.csv', index=False)
