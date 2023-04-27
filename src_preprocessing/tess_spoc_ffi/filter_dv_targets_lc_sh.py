"""
Filter TIC lc files that have DV results.
"""

# 3rd party
# import pandas as pd
from pathlib import Path

#%%

# dv_target_tbls_dir = Path('/Users/msaragoc/Downloads/tess_spoc_ffi_data/dv/target_tbls')
dv_sh_dir = Path('/Users/msaragoc/Downloads/tess_spoc_ffi_data/dv/target_sh')
src_lc_sh_dir = Path('/Users/msaragoc/Downloads/tess_spoc_ffi_data/lc/complete_target_list_sh')
dest_lc_sh_dir = Path('/Users/msaragoc/Downloads/tess_spoc_ffi_data/lc/dv_target_list_sh')
dest_lc_sh_dir.mkdir(exist_ok=True)

for src_sh_fp in src_lc_sh_dir.iterdir():

    print(f'Iterating through {src_sh_fp.name}...')

    targets_found_in_dv_res, targets_notfound_in_dv_res = 0, 0

    sector_run_id = src_sh_fp.name.split('_')[4]  # sector run id

    # # load DV TIC ID table
    # dv_sector_run_tbl = pd.read_csv(dv_target_tbls_dir / f'{sector_run_id}.csv')

    # get TIC IDs that have DV results
    dv_sh_fp = dv_sh_dir / f'hlsp_tess-spoc_tess_phot_{sector_run_id}_tess_v1_dl-dv.sh'
    tics_dv_res = []
    with open(dv_sh_fp, 'r') as dv_file:
        for line in dv_file:
            if 'curl' in line:
                tic_id = int(line.split(' ')[-2].split('/')[-1].split('_')[-4].split('-')[0])
                tics_dv_res.append(tic_id)

    dest_sh_fp = dest_lc_sh_dir / src_sh_fp.name

    with open(src_sh_fp, 'r') as src_file, open(dest_sh_fp, 'a') as dest_file:

        for line in src_file:
            if 'curl' in line:
                # find TIC ID in curl statement
                tic_id = int(line.split(' ')[-2].split('/')[-1].split('_')[-4].split('-')[0])
                # if target found in DV results, write to destination file
                if tic_id in tics_dv_res:
                    dest_file.write(line)
                    targets_found_in_dv_res += 1
                else:
                    targets_notfound_in_dv_res += 1
            else:
                dest_file.write(line)

    print(f'Number of targets found in DV results for sector run {sector_run_id}: '
          f'{targets_found_in_dv_res} (not found {targets_notfound_in_dv_res})')
