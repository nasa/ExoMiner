"""
Filter TIC lc files that have DV results.

For each sector run, checking if for a given target in the lc sh file there's a corresponding entry in the DV sh file
for the same sector run. If there isn't, then the target does not have any DV results (hence no TCEs were detected for
the target in that sector run).
"""

# 3rd party
from pathlib import Path
import numpy as np
import pandas as pd

#%%

dv_sh_dir = Path('/data5/tess_project/Data/tess_spoc_ffi_data/dv/target_sh')
src_lc_sh_dir = Path('/data5/tess_project/Data/tess_spoc_ffi_data/lc/complete_target_list_sh')
dest_lc_sh_dir = Path('/data5/tess_project/Data/tess_spoc_ffi_data/lc/dv_target_list_sh')
dest_lc_sh_dir.mkdir(exist_ok=True)

for src_sh_fp in [fp for fp in src_lc_sh_dir.iterdir() if fp.suffix == '.sh']:

    print(f'Iterating through {src_sh_fp.name}...')

    sector_run_id = src_sh_fp.name.split('_')[4]  # sector run id

    # get TIC IDs that have DV results
    dv_sh_fp = dv_sh_dir / f'hlsp_tess-spoc_tess_phot_{sector_run_id}_tess_v1_dl-dv.sh'
    tics_dv_res = []
    with open(dv_sh_fp, 'r') as dv_file:
        for line in dv_file:
            if 'curl' in line:
                tic_id = int(line.split(' ')[-2].split('/')[-1].split('_')[-4].split('-')[0])
                tics_dv_res.append(tic_id)

    dest_sh_fp = dest_lc_sh_dir / src_sh_fp.name

    lc_targets_found_in_dv_res, lc_targets_notfound_in_dv_res = [], []

    with open(src_sh_fp, 'r') as src_file, open(dest_sh_fp, 'w') as dest_file:

        for line in src_file:
            if 'curl' in line:
                # find TIC ID in curl statement
                tic_id = int(line.split(' ')[-2].split('/')[-1].split('_')[-4].split('-')[0])
                # if target found in DV results, write to destination file
                if tic_id in tics_dv_res:
                    dest_file.write(line)
                    lc_targets_found_in_dv_res.append(tic_id)
                else:
                    lc_targets_notfound_in_dv_res.append(tic_id)
            else:
                dest_file.write(line)

    dv_targets_not_in_lc_res = np.setdiff1d(tics_dv_res, lc_targets_notfound_in_dv_res)

    print(f'Number of LC targets found in DV results for sector run {sector_run_id}: '
          f'{len(lc_targets_found_in_dv_res)} (not found {len(lc_targets_notfound_in_dv_res)})')

    print(f'Number of DV targets not found in LC results for sector run {sector_run_id}: '
          f'{len(dv_targets_not_in_lc_res)}')
    if len(dv_targets_not_in_lc_res) > 0:
        dv_targets_not_in_lc_res_df = pd.DataFrame(dv_targets_not_in_lc_res, columns=['tic_id'])
        dv_targets_not_in_lc_res_df.to_csv(dest_lc_sh_dir / f'dv_targets_not_found_in_lc_res_{sector_run_id}.csv',
                                           index=False)

    src_sh_fp.rename(src_sh_fp.parent / 'completed' / src_sh_fp.name)
