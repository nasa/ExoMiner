from pathlib import Path
import numpy as np
import pandas as pd
from scipy.io import loadmat

dv_root_dir = Path('/data5/tess_project/Data/Ephemeris_tables/TESS/DV_SPOC_mat_files')

ssector_runs_dir = dv_root_dir / 'single-sector'
msector_runs_dir = dv_root_dir / 'multi-sector'

mat_runs_fps = list(ssector_runs_dir.iterdir()) + list(msector_runs_dir.iterdir())

for mat_run_fp in mat_runs_fps:

    print(f'Creating csv table for mat file {mat_run_fp.name}')

    csv_save_dir = mat_run_fp.parent / 'csv_tables'
    csv_save_dir.mkdir(exist_ok=True)

    mat_struct = loadmat(mat_run_fp)

    cols = []
    cnt_no_name_col = 0
    for el_i, el in enumerate(mat_struct['dvOutputMatrixColumns']):

        try:
            # print(f'{el[0][0]}')
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

#%%
