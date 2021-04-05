""" Count number of TECEs generated in the single- and multi-sector runs for TESS."""

import pandas as pd
from pathlib import Path

#%% single-sector runs

dv_tables_dir = Path('/data5/tess_project/Data/Ephemeris_tables/TESS/DV_ephemeris/single-sector runs')

count_dict = {'sector': [], 'num_tces': []}

dv_tbls_fps = [fp for fp in dv_tables_dir.iterdir() if fp.suffix == '.csv']

for dv_tbl in dv_tbls_fps:

    sector = dv_tbl.name.split('-')[1]
    num_tces_sector = len(pd.read_csv(dv_tbl, header=6))
    count_dict['sector'].append(sector)
    count_dict['num_tces'].append(num_tces_sector)

count_df = pd.DataFrame(data=count_dict)
count_df.sort_values('sector', inplace=True)
count_dict_stat = pd.DataFrame(data={'sector': ['total', 'mean', 'std'], 'num_tces': [count_df['num_tces'].sum(), count_df['num_tces'].mean(), count_df['num_tces'].std()]})
count_df = count_df.append(count_dict_stat, ignore_index=True)

count_df.to_csv(dv_tables_dir.parent / 'num_tces_single-sector_3-9-2021.csv', index=False)

#%% multi-sector runs

dv_tables_dir = Path('/data5/tess_project/Data/Ephemeris_tables/TESS/DV_ephemeris/multi-sector runs')

count_dict = {'sector': [], 'num_tces': []}

dv_tbls_fps = [fp for fp in dv_tables_dir.iterdir() if fp.suffix == '.csv']

for dv_tbl in dv_tbls_fps:

    sector = '-'.join(dv_tbl.name.split('_')[0].split('-')[1:])
    num_tces_sector = len(pd.read_csv(dv_tbl, header=6))
    count_dict['sector'].append(sector)
    count_dict['num_tces'].append(num_tces_sector)

count_df = pd.DataFrame(data=count_dict)
count_df.sort_values('sector', inplace=True)
count_dict_stat = pd.DataFrame(data={'sector': ['total', 'mean', 'std'], 'num_tces': [count_df['num_tces'].sum(), count_df['num_tces'].mean(), count_df['num_tces'].std()]})
count_df = count_df.append(count_dict_stat, ignore_index=True)

count_df.to_csv(dv_tables_dir.parent / 'num_tces_multi-sector_3-9-2021.csv', index=False)