"""
- Prepare TEC flux triage catalog to be used for:
1) labeling TCEs as NTPs to build an NTP set for TESS;
2) matching secondaries with primaries.
- Process TEC flux triage, and Tier 1, 2, and 3 results.
"""

# 3rd party
import pandas as pd
from pathlib import Path
import logging
import re
from functools import reduce

# %% Processing TEC flux triage tables

tec_dir = Path('/Users/msaragoc/Projects/exoplanet_transit_classification/data/ephemeris_tables/tess/TEC_SPOC/sector_run_results')
save_fp = Path('/Users/msaragoc/Projects/exoplanet_transit_classification/data/ephemeris_tables/tess/TEC_SPOC/')

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

tec_tbl_full = pd.concat(tec_tbls, axis=0)

# setting tces uids
tec_tbl_full['uid'] = tec_tbl_full[['target_id', 'tce_plnt_num', 'sector_run']].apply(lambda x: f'{x["target_id"]}-{x["tce_plnt_num"]}-S{x["sector_run"]}', axis=1)

tec_tbl_full.rename(columns={'pass': 'tec_fluxtriage_pass', 'comment': 'tec_fluxtriage_comment'}, inplace=True)

tec_tbl_full.to_csv(save_fp / 'tec_tbl_fluxtriage_s1-s41_10-4-2023.csv', index=False)

#%% Adding TEC results to TCE table

res_dir = Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/data/ephemeris_tables/tess/DV_SPOC_mat_files/10-05-2022_1338')

# set up logger
logger = logging.getLogger(name='add_tec_to_tce_tbl')
logger_handler = logging.FileHandler(filename=res_dir / f'add_tec_to_tce_tbl.log', mode='w')
logger_formatter = logging.Formatter('%(asctime)s - %(message)s')
logger.setLevel(logging.INFO)
logger_handler.setFormatter(logger_formatter)
logger.addHandler(logger_handler)
logger.info(f'Starting run...')

tec_tbl_fp = Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/data/ephemeris_tables/tess/TEC_SPOC/tec_tbl_fluxtriage_s1-s41_10-29-2021.csv')
tec_tbl = pd.read_csv(tec_tbl_fp)
logger.info(f'Loaded TEC table from {tec_tbl_fp}')
logger.info(f'Number of TCEs from TEC: {len(tec_tbl)}')

tce_tbl_fp = res_dir / 'tess_tces_dv_s1-s55_10-05-2022_1338_ticstellar_ruwe.csv'
tce_tbl = pd.read_csv(tce_tbl_fp)
logger.info(f'Loaded TCE table from {str(tce_tbl_fp)}')

tce_tbl = tce_tbl.merge(tec_tbl, on=['target_id', 'tce_plnt_num', 'sector_run'], how='left', validate='one_to_one')

logger.info(f'Number of TCEs with TEC results: {tce_tbl["tec_fluxtriage_pass"].isna().sum()} out of {len(tce_tbl)} TCEs.')
logger.info(f'Counts for flux triage:\n{tce_tbl["tec_fluxtriage_pass"].value_counts()}')
logger.info(f'Counts for flux triage comment:\n{tce_tbl["tec_fluxtriage_comment"].value_counts()}')

tce_tbl.to_csv(res_dir / f'{tce_tbl_fp.stem}_tec.csv', index=False)
logger.info('Saved TCE table with TEC results')

#%% Create table for tier 3

sector_runs_root_dir = Path('/Users/msaragoc/Projects/exoplanet_transit_classification/data/ephemeris_tables/tess/tec/tec_spoc_2min/sector_run_results/')

sector_runs_dirs = [fp for fp in sector_runs_root_dir.iterdir() if fp.is_dir()]

print(f'Found {len(sector_runs_dirs)} sector run results.')

sector_run_tbls_lst = []
for sector_i, sector_run_dir in enumerate(sector_runs_dirs):
    
    print(f'Processing {sector_run_dir.name} ({sector_i + 1}/{len(sector_runs_dirs)})...')

    # get sector id
    singlesector_re_search = re.search('Sector_[0-9]*', sector_run_dir.name)
    singlesector_re_search_nounderscore = re.search('Sector[0-9]*', sector_run_dir.name)

    # multisector_re_search = re.search('MultiSector_[0-9]*_[0-9]*', sector_run_dir.name)
    multisector_re_search2 = re.search('Sector_[0-9]*_[0-9]+', sector_run_dir.name)
    multisector_re_search3 = re.search(r"Sector(\d+)[-_](\d+)", sector_run_dir.name)

    if singlesector_re_search and not multisector_re_search2:  #  and not multisector_re_search2:
        sector_run_id = str(int(singlesector_re_search.group().split('_')[-1]))
    elif singlesector_re_search_nounderscore and not multisector_re_search2 and not multisector_re_search3:
        sector_run_id = str(int(singlesector_re_search_nounderscore.group()[6:]))
    elif multisector_re_search2:
        sector_run_id = '-'.join([str(int(el)) for el in multisector_re_search2.group().split('_')[1:]])
    elif multisector_re_search3:
        sector_run_id = f'{int(multisector_re_search3.group(1))}-{int(multisector_re_search3.group(2))}'
    else:
        raise ValueError(f'No matching pattern found for sector run in {sector_run_dir.name}')

    print(f'Sector run ID: {sector_run_id}')
    
    # get txt table
    sector_run_tbl_fp = list(sector_run_dir.glob('*Tier3*'))[0]
    print(f'Reading table {sector_run_tbl_fp.name} for sector run {sector_run_id}...')
    
    # read and parse table
    sector_run_tbl = pd.read_csv(sector_run_tbl_fp, names=['target_id', 'tce_plnt_num', 'priority_rank', 'known_flag', 'has_sec_flag', 'sweet_fail_flag'], sep=' ')

    # add sector id
    sector_run_tbl['sector_run'] = sector_run_id

    # set unique id
    sector_run_tbl['uid'] = sector_run_tbl[['target_id', 'tce_plnt_num', 'sector_run']].apply(lambda x: f'{x["target_id"]}-{x["tce_plnt_num"]}-S{x["sector_run"]}', axis=1)

    sector_run_tbls_lst.append(sector_run_tbl)
    
    print(f'Finished processing {sector_run_dir.name}.')

tier3_tbl = pd.concat(sector_run_tbls_lst, axis=0, ignore_index=True)
tier3_tbl.set_index('uid', inplace=True)
tier3_tbl.to_csv(sector_runs_root_dir.parent / 'tec_tier3_9-2-2025_1440.csv', index=True)

#%% Create table for tier 2

sector_runs_root_dir = Path('/Users/msaragoc/Projects/exoplanet_transit_classification/data/ephemeris_tables/tess/tec/tec_spoc_2min/sector_run_results/')

sector_runs_dirs = [fp for fp in sector_runs_root_dir.iterdir() if fp.is_dir()]

print(f'Found {len(sector_runs_dirs)} sector run results.')

sector_run_tbls_lst = []
for sector_i, sector_run_dir in enumerate(sector_runs_dirs):
    
    print(f'Processing {sector_run_dir.name} ({sector_i + 1}/{len(sector_runs_dirs)})...')

    # get sector id
    singlesector_re_search = re.search('Sector_[0-9]*', sector_run_dir.name)
    singlesector_re_search_nounderscore = re.search('Sector[0-9]*', sector_run_dir.name)

    # multisector_re_search = re.search('MultiSector_[0-9]*_[0-9]*', sector_run_dir.name)
    multisector_re_search2 = re.search('Sector_[0-9]*_[0-9]+', sector_run_dir.name)
    multisector_re_search3 = re.search(r"Sector(\d+)[-_](\d+)", sector_run_dir.name)

    if singlesector_re_search and not multisector_re_search2:  #  and not multisector_re_search2:
        sector_run_id = str(int(singlesector_re_search.group().split('_')[-1]))
    elif singlesector_re_search_nounderscore and not multisector_re_search2 and not multisector_re_search3:
        sector_run_id = str(int(singlesector_re_search_nounderscore.group()[6:]))
    elif multisector_re_search2:
        sector_run_id = '-'.join([str(int(el)) for el in multisector_re_search2.group().split('_')[1:]])
    elif multisector_re_search3:
        sector_run_id = f'{int(multisector_re_search3.group(1))}-{int(multisector_re_search3.group(2))}'
    else:
        raise ValueError(f'No matching pattern found for sector run in {sector_run_dir.name}')

    print(f'Sector run ID: {sector_run_id}')
    
    # get txt table
    sector_run_tbl_fp = list(sector_run_dir.glob('*Tier2*'))[0]
    print(f'Reading table {sector_run_tbl_fp.name} for sector run {sector_run_id}...')
    
    # read and parse table
    sector_run_tbl = pd.read_csv(sector_run_tbl_fp, names=['target_id', 'tce_plnt_num', 'priority_rank', 'known_flag', 'fail_flags_bits', 'fail_flags_descr'], sep=' ')

    # add sector id
    sector_run_tbl['sector_run'] = sector_run_id

    # set unique id
    sector_run_tbl['uid'] = sector_run_tbl[['target_id', 'tce_plnt_num', 'sector_run']].apply(lambda x: f'{x["target_id"]}-{x["tce_plnt_num"]}-S{x["sector_run"]}', axis=1)

    sector_run_tbls_lst.append(sector_run_tbl)
    
    print(f'Finished processing {sector_run_dir.name}.')

tier2_tbl = pd.concat(sector_run_tbls_lst, axis=0, ignore_index=True)
tier2_tbl.set_index('uid', inplace=True)
tier2_tbl.to_csv(sector_runs_root_dir.parent / 'tec_tier2_9-2-2025_1440.csv', index=True)

#%% Create table for tier 1

sector_runs_root_dir = Path('/Users/msaragoc/Projects/exoplanet_transit_classification/data/ephemeris_tables/tess/tec/tec_spoc_2min/sector_run_results/')

sector_runs_dirs = [fp for fp in sector_runs_root_dir.iterdir() if fp.is_dir()]

print(f'Found {len(sector_runs_dirs)} sector run results.')

sector_run_tbls_lst = []
for sector_i, sector_run_dir in enumerate(sector_runs_dirs):
    
    print(f'Processing {sector_run_dir.name} ({sector_i + 1}/{len(sector_runs_dirs)})...')

    # get sector id
    singlesector_re_search = re.search('Sector_[0-9]*', sector_run_dir.name)
    singlesector_re_search_nounderscore = re.search('Sector[0-9]*', sector_run_dir.name)

    # multisector_re_search = re.search('MultiSector_[0-9]*_[0-9]*', sector_run_dir.name)
    multisector_re_search2 = re.search('Sector_[0-9]*_[0-9]+', sector_run_dir.name)
    multisector_re_search3 = re.search(r"Sector(\d+)[-_](\d+)", sector_run_dir.name)

    if singlesector_re_search and not multisector_re_search2:  #  and not multisector_re_search2:
        sector_run_id = str(int(singlesector_re_search.group().split('_')[-1]))
    elif singlesector_re_search_nounderscore and not multisector_re_search2 and not multisector_re_search3:
        sector_run_id = str(int(singlesector_re_search_nounderscore.group()[6:]))
    elif multisector_re_search2:
        sector_run_id = '-'.join([str(int(el)) for el in multisector_re_search2.group().split('_')[1:]])
    elif multisector_re_search3:
        sector_run_id = f'{int(multisector_re_search3.group(1))}-{int(multisector_re_search3.group(2))}'
    else:
        raise ValueError(f'No matching pattern found for sector run in {sector_run_dir.name}')

    print(f'Sector run ID: {sector_run_id}')
    
    # get txt table
    sector_run_tbl_fp = list(sector_run_dir.glob('*Tier1*'))[0]
    print(f'Reading table {sector_run_tbl_fp.name} for sector run {sector_run_id}...')
    
    # read and parse table
    sector_run_tbl = pd.read_csv(sector_run_tbl_fp, names=['target_id', 'tce_plnt_num', 'priority_rank', 'known_flag'], sep=' ')

    # add sector id
    sector_run_tbl['sector_run'] = sector_run_id

    # set unique id
    sector_run_tbl['uid'] = sector_run_tbl[['target_id', 'tce_plnt_num', 'sector_run']].apply(lambda x: f'{x["target_id"]}-{x["tce_plnt_num"]}-S{x["sector_run"]}', axis=1)

    sector_run_tbls_lst.append(sector_run_tbl)
    
    print(f'Finished processing {sector_run_dir.name}.')

tier1_tbl = pd.concat(sector_run_tbls_lst, axis=0, ignore_index=True)
tier1_tbl.set_index('uid', inplace=True)
tier1_tbl.to_csv(sector_runs_root_dir.parent / 'tec_tier1_9-2-2025_1440.csv', index=True)

#%% Create table for flux triage

sector_runs_root_dir = Path('/Users/msaragoc/Projects/exoplanet_transit_classification/data/ephemeris_tables/tess/tec/tec_spoc_2min/sector_run_results/')

sector_runs_dirs = [fp for fp in sector_runs_root_dir.iterdir() if fp.is_dir()]

print(f'Found {len(sector_runs_dirs)} sector run results.')

sector_run_tbls_lst = []
for sector_i, sector_run_dir in enumerate(sector_runs_dirs):
    
    print(f'Processing {sector_run_dir.name} ({sector_i + 1}/{len(sector_runs_dirs)})...')

    # get sector id
    singlesector_re_search = re.search('Sector_[0-9]*', sector_run_dir.name)
    singlesector_re_search_nounderscore = re.search('Sector[0-9]*', sector_run_dir.name)

    # multisector_re_search = re.search('MultiSector_[0-9]*_[0-9]*', sector_run_dir.name)
    multisector_re_search2 = re.search('Sector_[0-9]*_[0-9]+', sector_run_dir.name)
    multisector_re_search3 = re.search(r"Sector(\d+)[-_](\d+)", sector_run_dir.name)

    if singlesector_re_search and not multisector_re_search2:  #  and not multisector_re_search2:
        sector_run_id = str(int(singlesector_re_search.group().split('_')[-1]))
    elif singlesector_re_search_nounderscore and not multisector_re_search2 and not multisector_re_search3:
        sector_run_id = str(int(singlesector_re_search_nounderscore.group()[6:]))
    elif multisector_re_search2:
        sector_run_id = '-'.join([str(int(el)) for el in multisector_re_search2.group().split('_')[1:]])
    elif multisector_re_search3:
        sector_run_id = f'{int(multisector_re_search3.group(1))}-{int(multisector_re_search3.group(2))}'
    else:
        raise ValueError(f'No matching pattern found for sector run in {sector_run_dir.name}')

    print(f'Sector run ID: {sector_run_id}')
    
    # get txt table
    sector_run_fps = list(sector_run_dir.glob('*fluxtriage*'))
    if len(sector_run_fps) == 0:
        print(f'No flux triage file for {sector_run_dir}')
        continue
    sector_run_tbl_fp = sector_run_fps[0]
    print(f'Reading table {sector_run_tbl_fp.name} for sector run {sector_run_id}...')
    
    # read and parse table
    sector_run_tbl = pd.read_csv(sector_run_tbl_fp, names=['target_id', 'tce_plnt_num', 'tec_fluxtriage_pass', 'tec_fluxtriage_comment'], sep=r'\s+')

    # add sector id
    sector_run_tbl['sector_run'] = sector_run_id

    # set unique id
    sector_run_tbl['uid'] = sector_run_tbl[['target_id', 'tce_plnt_num', 'sector_run']].apply(lambda x: f'{x["target_id"]}-{x["tce_plnt_num"]}-S{x["sector_run"]}', axis=1)

    sector_run_tbls_lst.append(sector_run_tbl)
    
    print(f'Finished processing {sector_run_dir.name}.')

flux_triage_tbl = pd.concat(sector_run_tbls_lst, axis=0, ignore_index=True)
flux_triage_tbl.set_index('uid', inplace=True)
flux_triage_tbl.to_csv(sector_runs_root_dir.parent / 'tec_fluxtriage_9-2-2025_1440.csv', index=True)

# %% Merge all TEC results into a single table

tec_tbls_lst = [
    pd.read_csv('/Users/msaragoc/Projects/exoplanet_transit_classification/data/ephemeris_tables/tess/tec/tec_spoc_2min/tec_fluxtriage_9-2-2025_1440.csv'),
    pd.read_csv('/Users/msaragoc/Projects/exoplanet_transit_classification/data/ephemeris_tables/tess/tec/tec_spoc_2min/tec_tier1_9-2-2025_1440.csv', usecols=['uid', 'priority_rank', 'known_flag']),
    pd.read_csv('/Users/msaragoc/Projects/exoplanet_transit_classification/data/ephemeris_tables/tess/tec/tec_spoc_2min/tec_tier2_9-2-2025_1440.csv', usecols=['uid', 'fail_flags_bits', 'fail_flags_descr']),
    pd.read_csv('/Users/msaragoc/Projects/exoplanet_transit_classification/data/ephemeris_tables/tess/tec/tec_spoc_2min/tec_tier3_9-2-2025_1440.csv', usecols=['uid', 'has_sec_flag', 'sweet_fail_flag']),
] 

# merge_cols = ['uid', 'target_id', 'tce_plnt_num', 'sector_run']
tec_tbl_all = reduce(lambda left, right: pd.merge(left, right, on='uid', how='outer', validate='one_to_one'), tec_tbls_lst)
tec_tbl_all.to_csv(sector_runs_root_dir.parent / 'tec_results_9-2-2025_1440.csv', index=False)
