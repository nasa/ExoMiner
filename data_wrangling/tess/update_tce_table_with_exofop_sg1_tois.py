"""Update TCE table with dispositions for TOIs in ExoFOP and SG1 TOI catalogs."""

# 3rd party
import pandas as pd
from pathlib import Path

#%% load tables

tce_tbl_fp = Path('/u/msaragoc/work_dir/Kepler-TESS_exoplanet/data/Ephemeris_tables/TESS/tess_spoc_ffi/tess-spoc-ffi-tces-dv_s36-s72_multisector-s56s69_10-8-2025.csv')
exofop_toi_tbl_fp = Path('/u/msaragoc/work_dir/Kepler-TESS_exoplanet/data/Ephemeris_tables/TESS/exofop_tois/exofop_tois_9-11-2025.csv')
sg1_toi_tbl_fp = Path('/u/msaragoc/work_dir/Kepler-TESS_exoplanet/data/Ephemeris_tables/TESS/sg1/sg1_tois_9-11-2025.csv')

tce_tbl = pd.read_csv(tce_tbl_fp, dtype={'matched_object': str})
exofop_toi_tbl = pd.read_csv(exofop_toi_tbl_fp, dtype={'TOI': str})
sg1_toi_tbl = pd.read_csv(sg1_toi_tbl_fp, dtype={'TOI': str})
sg1_toi_tbl.drop_duplicates('TOI', inplace=True)

print(tce_tbl['label'].value_counts())

#%% prepare tables

exofop_toi_tbl_cols = [
    'TOI',
    'Predicted Mass (M_Earth)',
    'Predicted Radial Velocity Semi-amplitude (m/s)',
    'TESS Disposition',
    'TFOPWG Disposition',
    'Epoch (BJD)',
    'Period (days)',
    'Duration (hours)',
    'Depth (ppm)',
    'Planet Radius (R_Earth)',
    'Planet Insolation (Earth Flux)',
    'Planet Equil Temp (K)',
    'Planet SNR',
    'Sectors',
    'Stellar Distance (pc)',
    'Comments',
    'Date TOI Alerted (UTC)',
    'Date TOI Updated (UTC)',
    'Date Modified',
]
exofop_toi_tbl = exofop_toi_tbl[exofop_toi_tbl_cols]
exofop_toi_tbl.rename(columns={'TOI': 'matched_object', 'Epoch (BJD)': 'epoch_exofop', 'Period (days)': 'period_exofop', 'Duration (hours)': 'duration_exofop'}, inplace=True, errors='raise')

sg1_toi_tbl_cols = [
    'TOI',
    # 'Sec-tor',
    'Master Disposition',
    'Phot Disposition',
    'Spec Disposition',
    'Gaia\nRU\nWE',
    'Comments',
    'SG2 Notes',
]
sg1_toi_tbl = sg1_toi_tbl[sg1_toi_tbl_cols]
sg1_toi_tbl.rename(columns={'TOI': 'matched_object', 'Comments': 'sg1_comments', 'Master Disposition': 'sg1_master_disp', 'Gaia\nRU\nWE': 'sg1_gaia_ruwe'}, inplace=True, errors='raise')

#%% update existent columns 

tce_tbl.set_index('matched_object', inplace=True)
exofop_toi_tbl.set_index('matched_object', inplace=True)
sg1_toi_tbl.set_index('matched_object', inplace=True)

existent_cols_exofop = [col for col in exofop_toi_tbl.columns if col in tce_tbl.columns]
existent_cols_sg1 = [col for col in sg1_toi_tbl.columns if col in tce_tbl.columns]
tce_tbl.update(exofop_toi_tbl[existent_cols_exofop])
tce_tbl.update(sg1_toi_tbl[existent_cols_sg1])

tce_tbl.reset_index(inplace=True)
exofop_toi_tbl.reset_index(inplace=True)
sg1_toi_tbl.reset_index(inplace=True)

#%% add nonexistent columns

nonexistent_cols_exofop = [col for col in exofop_toi_tbl.columns if col not in tce_tbl.columns]
nonexistent_cols_sg1 = [col for col in sg1_toi_tbl.columns if col not in tce_tbl.columns]

tce_tbl = tce_tbl.merge(exofop_toi_tbl[['matched_object'] + nonexistent_cols_exofop], on='matched_object', how='left', validate='many_to_one')
tce_tbl = tce_tbl.merge(sg1_toi_tbl[['matched_object'] + nonexistent_cols_sg1], on='matched_object', how='left', validate='many_to_one')

#%% update labels based on updated TOIs dispositions

print('Before', tce_tbl['label'].value_counts())

tfopwg_disps = ['CP', 'KP', 'FP']
for tfopwg_disp in tfopwg_disps:  # set TFOPWG labels first since they have higher priority
    tce_tbl.loc[tce_tbl['TFOPWG Disposition'] == tfopwg_disp, ['label', 'label_source']] = [tfopwg_disp, 'TFOPWG']

# set CP BDs to BDs
tce_tbl.loc[tce_tbl['sg1_master_disp'] == 'BD', ['label', 'label_source']] = ['BD', 'SG1']

print('After', tce_tbl['label'].value_counts())

#%% update number of TOIs per TIC and TOIs in TIC

# tce_tbl['n_tois_in_tic'] = 0
# tce_tbl['tois_in_tic'] = ''
# tces_in_tics = tce_tbl.groupby('target_id')
# for tic, tces_in_tic in tces_in_tics:
    
#     # get only TOI IDs
#     tois_in_tic = tces_in_tic.loc[((~tces_in_tic['matched_object'].isna()) & (tces_in_tic['matched_object'].str.contains('.'))), 'matched_object'].unique()
#     n_tois_in_tic = len(tois_in_tic)
#     tois_in_tic = '_'.join(tois_in_tic)
    
#     tce_tbl.loc[tce_tbl['target_id'] == tic, ['n_tois_in_tic', 'tois_in_tic']] = [n_tois_in_tic, tois_in_tic]

# Drop the old columns if they exist to avoid merge conflicts
tce_tbl = tce_tbl.drop(columns=['n_tois_in_tic', 'tois_in_tic', 'n_tois_in_tic_x', 'tois_in_tic_x', 'n_tois_in_tic_y', 'tois_in_tic_y'], errors='ignore')

# define the function to extract TOIs
def extract_tois(group):
    
    tois = group['matched_object'].dropna()
    tois = tois[tois.str.match(r'^\d{1,4}\.\d{2}$')]
    tois = tois.unique()

    return pd.Series({
        'n_tois_in_tic': len(tois),
        'tois_in_tic': '_'.join(tois)
    })

# apply the function to each group
tois_summary = tce_tbl.groupby('target_id', group_keys=False).apply(extract_tois)

# merge the result back into the original table
tce_tbl = tce_tbl.merge(tois_summary, on='target_id', how='left')

# fill NaNs if any (in case some target_ids had no TOIs)
tce_tbl['n_tois_in_tic'] = tce_tbl['n_tois_in_tic'].fillna(0).astype(int)
tce_tbl['tois_in_tic'] = tce_tbl['tois_in_tic'].fillna('')

#%% save TCE table

tce_tbl.to_csv(tce_tbl_fp.parent / f'{tce_tbl_fp.stem}_exofop-sg1-tois9-11-2025.csv', index=False)

# for col in exofop_toi_tbl.columns:
#     if col not in tce_tbl.columns:
#         print(col)

# for col in sg1_toi_tbl.columns:
#     if col not in tce_tbl.columns:
#         print(col)

#%% Count changes in dispositions, including SG1 Master dispositions

tce_tbl['sector_run'] = tce_tbl['sector_run'].astype('string')

new_sectors_tbl = tce_tbl.loc[tce_tbl['sector_run'].isin([str(s_sector) for s_sector in range(68, 94 + 1)] + ['14-86', '14-78', '1-69', '2-72'])]
prev_sectors_tbl = tce_tbl.loc[~tce_tbl['uid'].isin(new_sectors_tbl['uid'])]

print('New sectors TCEs', new_sectors_tbl['label'].value_counts())
print('Previous sectors TCEs', prev_sectors_tbl['label'].value_counts())

print('New sectors TCEs', new_sectors_tbl['sg1_master_disp'].value_counts())
print('Previous sectors TCEs', prev_sectors_tbl['sg1_master_disp'].value_counts())

# shared TOIs TCEs dispositioned as KP, CP, FP, BD
shared_tois_tces = new_sectors_tbl.loc[((new_sectors_tbl['matched_object'].isin(prev_sectors_tbl['matched_object']) & (new_sectors_tbl['label'].isin(['KP', 'CP', 'FP', 'BD']))))]
new_tois_tces = new_sectors_tbl.loc[((~new_sectors_tbl['matched_object'].isin(prev_sectors_tbl['matched_object']) & (new_sectors_tbl['label'].isin(['KP', 'CP', 'FP', 'BD']))))]

print('Shared TOI TCEs', shared_tois_tces['label'].value_counts())
print('New TOI TCEs', new_tois_tces['label'].value_counts())

print('Shared TOI TCEs', shared_tois_tces['sg1_master_disp'].value_counts())
print('New TOI TCEs', new_tois_tces['sg1_master_disp'].value_counts())
