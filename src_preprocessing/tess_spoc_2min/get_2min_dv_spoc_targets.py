"""
Get SPOC targets with DV results on 2-min cadence data.
"""

# 3rd party
import pandas as pd
from pathlib import Path

#%%

dv_sector_runs_root_dir = Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/data/fits_files/tess/dv/sector_runs')
save_dir = Path('/Users/msaragoc/Downloads/test_get_tess_spoc-dv_targets_2mincadencedata')
save_dir.mkdir(exist_ok=True)

# single-sector runs
sector_run_dirs = [fp for fp in (dv_sector_runs_root_dir / 'single-sector').iterdir() if fp.is_dir()]
for sector_run in sector_run_dirs:

    data_to_tbl = {'target_id': [fits_fp.name.split('-')[3] for fits_fp in sector_run.iterdir()]}
    data_df = pd.DataFrame(data_to_tbl)
    data_df.to_csv(save_dir / f'tess_spoc-dv_targets_2min_S{sector_run.name.split("_")[1]}.csv', index=False)

# multi-sector runs
sector_run_dirs = [fp for fp in (dv_sector_runs_root_dir / 'multi-sector').iterdir() if fp.is_dir()]
for sector_run in sector_run_dirs:
    s_sector, e_sector = sector_run.name.split('_')[1].split('-')
    s_sector, e_sector = int(s_sector[1:]), int(e_sector[1:])
    data_to_tbl = {'target_id': [fits_fp.name.split('-')[3] for fits_fp in sector_run.iterdir()]}
    data_to_df = pd.DataFrame(data_to_tbl)
    for sector in range(s_sector, e_sector + 1):
        tbl_fp = Path(save_dir / f'tess_spoc-dv_targets_2min_S{sector}.csv')
        if tbl_fp.exists():
            data_df = pd.read_csv(tbl_fp)
            data_df = pd.concat([data_df, data_to_df.loc[~data_to_df['target_id'].isin(data_df['target_id'])]], axis=0)
            data_df.to_csv(save_dir / f'tess_spoc-dv_targets_2min_S{sector}.csv', index=False)
        else:
            data_to_df.to_csv(save_dir / f'tess_spoc-dv_targets_2min_S{sector}.csv', index=False)
