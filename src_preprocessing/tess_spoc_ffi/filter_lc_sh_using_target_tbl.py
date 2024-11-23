"""
Filter TIC lc files that have DV results.

For each TIC lc FITS file, check whether there are TCEs in that TIC in the TCE table. If there isn't, then the lc FITS
file is skipped and the data for the TIC in the corresponding sector is not downloaded.
"""

# 3rd party
from pathlib import Path
import numpy as np
import pandas as pd

#%%

tce_tbl = pd.read_csv('/home6/msaragoc/work_dir/Kepler-TESS_exoplanet/data/Ephemeris_tables/TESS/tess_spoc_ffi/tess_spoc_ffi_s36-s72_multisector_s56-s69_sfromdvxml_11-22-2024_0942/tess_spoc_ffi_s36-s72_multisector_s56-s69_sfromdvxml_11-22-2024_0942.csv')
src_lc_sh_dir = Path('/home6/msaragoc/work_dir/Kepler-TESS_exoplanet/data/FITS_files/TESS/spoc_ffi/lc_sh/src_lc_sh')
dest_lc_sh_dir = Path('/home6/msaragoc/work_dir/Kepler-TESS_exoplanet/data/FITS_files/TESS/spoc_ffi/lc_sh/filtered_src_lc_sh')

tics_in_tce_tbl = tce_tbl['catId'].unique()

dest_lc_sh_dir.mkdir(exist_ok=True)

for src_sh_fp in [fp for fp in src_lc_sh_dir.iterdir() if fp.suffix == '.sh']:

    print(f'Iterating through {src_sh_fp.name}...')

    sector_run_id = src_sh_fp.name.split('_')[4]  # sector run id

    dest_sh_fp = dest_lc_sh_dir / src_sh_fp.name
    lc_targets_found_in_dv_res, lc_targets_notfound_in_dv_res = [], []
    with open(src_sh_fp, 'r') as src_file, open(dest_sh_fp, 'w') as dest_file:

        for line in src_file:
            if 'curl' in line:
                # find TIC ID in curl statement
                tic_id = int(line.split(' ')[-2].split('/')[-1].split('_')[-4].split('-')[0])
                # if target found in DV results, write to destination file
                if tic_id in tics_in_tce_tbl:
                    dest_file.write(line)
                    lc_targets_found_in_dv_res.append(tic_id)
                else:
                    lc_targets_notfound_in_dv_res.append(tic_id)
            else:
                dest_file.write(line)

    dv_targets_not_in_lc_res = np.setdiff1d(tics_in_tce_tbl, lc_targets_notfound_in_dv_res)

    print(f'Number of LC targets found in DV results for sector run {sector_run_id}: '
          f'{len(lc_targets_found_in_dv_res)} (not found {len(lc_targets_notfound_in_dv_res)})')

    print(f'Number of DV targets not found in LC results for sector run {sector_run_id}: '
          f'{len(dv_targets_not_in_lc_res)}')
    if len(dv_targets_not_in_lc_res) > 0:
        dv_targets_not_in_lc_res_df = pd.DataFrame(dv_targets_not_in_lc_res, columns=['tic_id'])
        dv_targets_not_in_lc_res_df.to_csv(dest_lc_sh_dir / f'dv_targets_not_found_in_lc_res_{sector_run_id}.csv',
                                           index=False)

    # src_sh_fp.rename(src_sh_fp.parent / 'completed' / src_sh_fp.name)
