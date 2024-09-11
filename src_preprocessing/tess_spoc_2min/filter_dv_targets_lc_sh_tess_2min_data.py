"""
Filter TIC lc sh files for 2-min data that do not have DV results.

Using the sh files for the DV products of a set of sector runs, gets all the TIC IDs of the target stars that have DV
results. Then uses this list to  filter from the lc sh files the curl statements for the TICs that do not have DV
results.

TICs without DV results do not have any TCEs detected by the SPOC pipeline.
"""

# 3rd party
from pathlib import Path
import numpy as np
import re
import pandas as pd

#%% Set up paths

# directory with DV sh files
dv_sh_dir = Path('/data5/tess_project/Data/tess_spoc_2min_data/dv/target_sh')
# directory with lc sh files
src_lc_sh_dir = Path('/data5/tess_project/Data/tess_spoc_2min_data/lc/complete_target_list_sh')
# destination directory to save new lc sh files after removing curl statements for targets without DV results
dest_lc_sh_dir = Path(f'/data5/tess_project/Data/tess_spoc_2min_data/lc/dv_target_list_sh')
dest_lc_sh_dir.mkdir(exist_ok=True)

#%%

dv_sh_fps = dv_sh_dir.iterdir()  # get iterator for fps of DV sh files
n_dv_sh_files = len(list(dv_sh_fps))  # number of DV sh files

# get tic ids of all target stars with DV results; i.e., with a curl statement for DV products in the DV sh files
tics_dv_dict = {}
# iterate through DV sh files
for dv_sh_fp_i, dv_sh_fp in enumerate([fp for fp in dv_sh_dir.iterdir() if fp.suffix == '.sh']):

    print(f'Iterating through DV sh file {dv_sh_fp.name} ({dv_sh_fp_i + 1}/{n_dv_sh_files})...')

    if 'multisector' in dv_sh_fp.name:  # multi-sector run tesscurl_multisector_sXXXX-sXXXX_dv.sh
        s_sector, e_sector = dv_sh_fp.stem.split('_')[2].split('-')
        s_sector, e_sector = s_sector[1:], e_sector[1:]
    else:  # single-sector tesscurl_sector_X_dv.sh
        sectors = np.array([int(dv_sh_fp.stem.split('_')[2])])
        s_sector, e_sector = [dv_sh_fp.stem.split('_')[2].zfill(4)] * 2

    # find targets in sector run with DV results
    with open(dv_sh_fp, 'r') as dv_file:
        for line in dv_file:
            if 'curl' in line:
                tic_id = int(re.search(f's{s_sector}-s{e_sector}-[0-9]*', line).group().split('-')[2])
                if tic_id not in tics_dv_dict:  # tic was already observed in some other sector runs
                    tics_dv_dict[tic_id] = [f's{s_sector}-{e_sector}']
                else:
                    tics_dv_dict[tic_id].append(f's{s_sector}-{e_sector}')

print(f'Total number of TICs found in DV results: {len(tics_dv_dict)}.')

# tics_dv_df = pd.DataFrame({'ticid': list(tics_dv_dict.keys()), 'sector_runs': list(tics_dv_dict.values())})
# tics_dv_df.to_csv(dv_sh_dir.parent / 'tics_in_dv.csv', index=False)

# remove curl statements for lc sh files of targets that do not have DV results in any sector run
# tics_lc_dict = {}
# tics_not_in_lc_dict = {}
for src_sh_fp in [fp for fp in src_lc_sh_dir.iterdir() if fp.suffix == '.sh']:

    print(f'Iterating through {src_sh_fp.name}...')

    sector_run_id = src_sh_fp.name.split('_')[2].zfill(4)  # sector run id, tesscurl_sector_X_lc.sh

    dest_sh_fp = dest_lc_sh_dir / src_sh_fp.name

    lc_targets_found_in_dv_res, lc_targets_notfound_in_dv_res = [], []

    with open(src_sh_fp, 'r') as src_file, open(dest_sh_fp, 'w') as dest_file:

        for line in src_file:
            if 'curl' in line:
                # find TIC ID in curl statement
                tic_id = int(re.search(f's{sector_run_id}-[0-9]*', line).group().split('-')[1])

                # if target found in DV results, write to destination file
                if tic_id in tics_dv_dict:  # target got results in at least one sector run
                    dest_file.write(line)
                    # if tic_id not in tics_lc_dict:
                    #     tics_lc_dict[tic_id] = [sector_run_id]
                    # else:
                    #     tics_lc_dict[tic_id].append(sector_run_id)
                    lc_targets_found_in_dv_res.append(tic_id)
                else:
                    lc_targets_notfound_in_dv_res.append(tic_id)
            else:
                dest_file.write(line)

    dv_targets_not_in_lc_res = np.setdiff1d(list(tics_dv_dict.keys()), lc_targets_found_in_dv_res)

    print(f'Number of LC targets found in DV results for sector run {sector_run_id}: '
          f'{len(lc_targets_found_in_dv_res)} (not found {len(lc_targets_notfound_in_dv_res)})')

    # print(f'Number of DV targets not found in LC results for sector run {sector_run_id}: '
    #       f'{len(dv_targets_not_in_lc_res)}')
    # if len(dv_targets_not_in_lc_res) > 0:
    #     dv_targets_not_in_lc_res_df = pd.DataFrame(dv_targets_not_in_lc_res, columns=['tic_id'])
    #     dv_targets_not_in_lc_res_df.to_csv(dest_lc_sh_dir / f'dv_targets_not_found_in_lc_res_{sector_run_id}.csv',
    #                                        index=False)

    # src_sh_fp.rename(src_sh_fp.parent / 'completed' / src_sh_fp.name)

# print(f'Total number of TICs found in LC results: {len(tics_lc_dict)}.')

# tics_lc_df = pd.DataFrame({'ticid': list(tics_lc_dict.keys()), 'sector_runs': list(tics_lc_dict.values())})
# tics_lc_df.to_csv(dest_lc_sh_dir / 'tics_in_lc.csv', index=False)
