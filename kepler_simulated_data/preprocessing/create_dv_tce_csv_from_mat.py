"""
Create DV TCE csv files from the source mat files for the Kepler simulated data and related Robovetter input tables.

- Load DV TCE mat files and convert them to csv files.
- Check which DV TCEs exist in the Robovetter input tables for the different simulated data runs.
- Create DV TCE tables with only the TCEs found in the Robovetter input tables.
- Aggregate DV TCEs for all simulated data runs into a single table.
"""

# 3rd party
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.io import loadmat
import mat73

#%% Create csv files from mat files for all DV TCE tables for simulated data

dv_mat_tbl_fp = Path('/Users/msaragoc/Projects/exoplanet_transit_classification/data/ephemeris_tables/kepler/q1-q17_dr25/simulated_data/transit_inversion/ksop-2543-transit-inversion-dv-rerun/dv/dvOutputMatrix.mat')
dv_mat_tbl = loadmat(dv_mat_tbl_fp)

print(f'Creating csv table for mat file {dv_mat_tbl_fp.name}')

csv_save_dir = Path('/Users/msaragoc/Projects/exoplanet_transit_classification/data/ephemeris_tables/kepler/q1-q17_dr25/simulated_data/transit_inversion/')

try:
    mat_struct = mat73.loadmat(dv_mat_tbl_fp)
    tce_tbl = pd.DataFrame(columns=[el[0] for el in mat_struct['dvOutputMatrixColumns']],
                           data=mat_struct['dvOutputMatrix'])
except:
    mat_struct = loadmat(dv_mat_tbl_fp)

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

print(f'TCE table with {len(tce_tbl)} TCEs.')
tce_tbl.to_csv(csv_save_dir / f'{dv_mat_tbl_fp.stem}.csv', index=False)

#%% Get only TCEs from the DV injection batches that are in the Robovetter input tables for the injection groups

columnNames = ['Data_Set', 'TCE_ID', 'KIC', 'pn', 'n_plan', 'ntrans', 'nrealt', 'ngoodt', 'iflag0', 'iflag1', 'iflag2',
              'iflag3', 'iflag4', 'iflag5', 'cflag0', 'cflag1', 'cflag2', 'cflag3', 'cflag4', 'cflag5', 'cflag6',
               'cflag7', 'cflag8', 'ephem_disp', 'cent_disp', 'cent_score', 'period', 'period_err', 'epoch',
               'epoch_err', 'duration', 'duration_err', 'impact', 'impact_err', 'depth', 'depth_err', 'depth_alt',
               'sma', 'rstar', 'mes', 'ses_to_mes', 'new_mes', 'lpp_dv', 'lpp_alt', 'all_tran_chases', 'sweet_snr',
               'sweet_amp', 'shape_metric', 'halo_ghost', 'mod_sig_pri_dv', 'mod_sig_sec_dv', 'mod_sig_ter_dv',
              'mod_sig_pos_dv', 'mod_fred_dv', 'mod_fa1_dv', 'mod_fa2_dv', 'mod_sig_pri_alt', 'mod_sig_sec_alt',
               'mod_sig_ter_alt', 'mod_sig_pos_alt', 'mod_fred_alt', 'mod_fa1_alt', 'mod_fa2_alt', 'modshiftval1_dv',
              'modshiftval2_dv', 'modshiftval3_dv', 'modshiftval4_dv', 'modshiftval5_dv', 'modshiftval6_dv',
               'modshiftval1_alt', 'modshiftval2_alt', 'modshiftval3_alt', 'modshiftval4_alt', 'modshiftval5_alt',
               'modshiftval6_alt', 'oesig_dv', 'oesig_alt', 'mod_oe_dv', 'mod_oe_alt', 'rp_dv', 'rp_dv_perr',
               'rp_dv_merr', 'alb_dv', 'alb_dv_perr', 'alb_dv_merr', 'mod_depth_pri_dv', 'mod_depth_sec_dv',
               'mod_ph_sec_dv', 'mod_ph_ter_dv', 'rp_alt', 'rp_alt_perr', 'rp_alt_merr', 'alb_alt', 'alb_alt_perr',
               'alb_alt_merr', 'mod_depth_pri_alt', 'mod_depth_sec_alt', 'mod_ph_sec_alt', 'mod_ph_ter_alt',
               'sdepthsig_dv', 'sdepthsig_alt']


# robovetter_tbl_fp = Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/data/ephemeris_tables/kepler/q1-q17_dr25/simulated_data/robovetter_input_tables/Injected_Q1-Q17_DR25/kplr_dr25_inj3_robovetter_input.txt')
# robovetter_tbl = pd.read_table(robovetter_tbl_fp, skiprows=223, names=columnNames, skipinitialspace=False, delim_whitespace=True)
# robovetter_tbl['uid'] = robovetter_tbl[['TCE_ID']].apply(lambda x: "-".join([str(int(el)) for el in x["TCE_ID"].split("-")]), axis=1)
#
# tce_tbl['uid'] = tce_tbl[['keplerId', 'planetIndexNumber']].apply(lambda x: f'{int(x["keplerId"])}-{int(x["planetIndexNumber"])}', axis=1)
#
# a = tce_tbl.loc[tce_tbl['uid'].isin(robovetter_tbl['uid'])]
# b = robovetter_tbl.loc[robovetter_tbl['uid'].isin(tce_tbl['uid'])]

# load Robovetter input tables for the injection groups
robovetter_tbls_dir = Path('/Users/msaragoc/Projects/exoplanet_transit_classification/data/ephemeris_tables/kepler/q1-q17_dr25/simulated_data/robovetter_input_tables/Injected_Q1-Q17_DR25')
robovetter_tbls = []
for tbl_fp in [fp for fp in robovetter_tbls_dir.iterdir() if fp.name.endswith('.txt')]:
    robovetter_tbl = pd.read_table(tbl_fp, skiprows=223, names=columnNames, skipinitialspace=False, delim_whitespace=True)
    grp = tbl_fp.stem.split('_')[2]
    print(f'Pixel-level Injection Group {grp}')
    print(f'Found {len(robovetter_tbl)} TCEs injected in {len(robovetter_tbl["KIC"].unique())} targets')
    robovetter_tbl['grp'] = grp
    robovetter_tbl['uid'] = robovetter_tbl[['TCE_ID']].apply(lambda x: "-".join([str(int(el)) for el in x["TCE_ID"].split("-")]), axis=1)
    robovetter_tbls.append(robovetter_tbl)

robovetter_tbl_all_grps = pd.concat(robovetter_tbls, axis=0)

# load DV TCE tables for the injection batches
tce_tbls_dir = Path('/Users/msaragoc/Projects/exoplanet_transit_classification/data/ephemeris_tables/kepler/q1-q17_dr25/simulated_data/pixel_level_transit_injections')
tce_tbls = []
for tbl_fp in [fp for fp in tce_tbls_dir.iterdir() if fp.name.endswith('.csv')]:
    tce_tbl = pd.read_csv(tbl_fp)
    batch_number = tbl_fp.stem.split('_')[1][-1]
    print(f'Pixel-level Injection Batch {batch_number}')
    print(f'Found {len(tce_tbl)} TCEs injected in {len(tce_tbl["keplerId"].unique())} targets')
    tce_tbl['batch'] = batch_number
    tce_tbl['uid'] = tce_tbl[['keplerId', 'planetIndexNumber']].apply(lambda x: f'{int(x["keplerId"])}-{int(x["planetIndexNumber"])}', axis=1)
    tce_tbls.append(tce_tbl)

# # check target overlap between batch tables
# for tce_tbl in tce_tbls:
#     print('#' * 50)
#     for tce_tbl2 in tce_tbls:
#         print('-' * 50)
#         print(f'Pairing TCE table for injection group {tce_tbl["batch"].values[0]} with TCE table for batch {tce_tbl2["batch"].values[0]}...')
#         n_targets_in_tce_tbl = len(tce_tbl.loc[tce_tbl['keplerId'].isin(tce_tbl2['keplerId']), 'keplerId'].unique())
#         print(f'Number of shared KICs: {n_targets_in_tce_tbl}')

tce_tbl_all_batches = pd.concat(tce_tbls, axis=0)
tce_tbl_all_batches['dataset'] = np.nan
for robovetter_tbl in robovetter_tbls:
    print('#' * 50)
    print(f'Pairing Robovetter input table for injection group {robovetter_tbl["Data_Set"].values[0]}...')
    print(f'Number of TCEs in the Robovetter input table: {len(robovetter_tbl)}')
    shared_tces = tce_tbl_all_batches['uid'].isin(robovetter_tbl['uid'])
    n_tces_in_tce_tbl = shared_tces.sum()
    n_targets_in_tce_tbl = len(tce_tbl_all_batches.loc[tce_tbl_all_batches['keplerId'].isin(robovetter_tbl['KIC']), 'keplerId'].unique())
    print(f'Found {n_tces_in_tce_tbl} TCEs (out of {len(robovetter_tbl)})')
    print(f'Found {n_targets_in_tce_tbl} KICs (out of {len(robovetter_tbl["KIC"].unique())})')
    tce_tbl_all_batches.loc[shared_tces, 'dataset'] = robovetter_tbl["Data_Set"].values[0]

print(tce_tbl_all_batches['dataset'].value_counts())
print(f'Number of TCEs not found in any injection group: {tce_tbl_all_batches["dataset"].isna().sum()}')

# exclude TCEs not found in the Robovetter input tables for the injection groups
tce_tbl_all_batches = tce_tbl_all_batches.loc[~tce_tbl_all_batches["dataset"].isna()]

tce_tbl_all_batches.to_csv('/Users/msaragoc/Projects/exoplanet_transit_classification/data/ephemeris_tables/kepler/q1-q17_dr25/simulated_data/dvOutputMatrix_tces_only_robovetter_input_inj_grps.csv', index=False)


#%% Get only TCEs from the DV inverted tbl that are in the Robovetter input table for the inverted run

columnNames = ['Data_Set', 'TCE_ID', 'KIC', 'pn', 'n_plan', 'ntrans', 'nrealt', 'ngoodt', 'iflag0', 'iflag1', 'iflag2',
              'iflag3', 'iflag4', 'iflag5', 'cflag0', 'cflag1', 'cflag2', 'cflag3', 'cflag4', 'cflag5', 'cflag6',
               'cflag7', 'cflag8', 'ephem_disp', 'cent_disp', 'cent_score', 'period', 'period_err', 'epoch',
               'epoch_err', 'duration', 'duration_err', 'impact', 'impact_err', 'depth', 'depth_err', 'depth_alt',
               'sma', 'rstar', 'mes', 'ses_to_mes', 'new_mes', 'lpp_dv', 'lpp_alt', 'all_tran_chases', 'sweet_snr',
               'sweet_amp', 'shape_metric', 'halo_ghost', 'mod_sig_pri_dv', 'mod_sig_sec_dv', 'mod_sig_ter_dv',
              'mod_sig_pos_dv', 'mod_fred_dv', 'mod_fa1_dv', 'mod_fa2_dv', 'mod_sig_pri_alt', 'mod_sig_sec_alt',
               'mod_sig_ter_alt', 'mod_sig_pos_alt', 'mod_fred_alt', 'mod_fa1_alt', 'mod_fa2_alt', 'modshiftval1_dv',
              'modshiftval2_dv', 'modshiftval3_dv', 'modshiftval4_dv', 'modshiftval5_dv', 'modshiftval6_dv',
               'modshiftval1_alt', 'modshiftval2_alt', 'modshiftval3_alt', 'modshiftval4_alt', 'modshiftval5_alt',
               'modshiftval6_alt', 'oesig_dv', 'oesig_alt', 'mod_oe_dv', 'mod_oe_alt', 'rp_dv', 'rp_dv_perr',
               'rp_dv_merr', 'alb_dv', 'alb_dv_perr', 'alb_dv_merr', 'mod_depth_pri_dv', 'mod_depth_sec_dv',
               'mod_ph_sec_dv', 'mod_ph_ter_dv', 'rp_alt', 'rp_alt_perr', 'rp_alt_merr', 'alb_alt', 'alb_alt_perr',
               'alb_alt_merr', 'mod_depth_pri_alt', 'mod_depth_sec_alt', 'mod_ph_sec_alt', 'mod_ph_ter_alt',
               'sdepthsig_dv', 'sdepthsig_alt']

# load Robovetter input table for the inverted run
robovetter_tbl_fp = Path('/Users/msaragoc/Projects/exoplanet_transit_classification/data/ephemeris_tables/kepler/q1-q17_dr25/simulated_data/robovetter_input_tables/kplr_dr25_inv_robovetter_input.txt')
robovetter_tbl = pd.read_table(robovetter_tbl_fp, skiprows=223, names=columnNames, skipinitialspace=False, delim_whitespace=True)
robovetter_tbl['uid'] = robovetter_tbl[['TCE_ID']].apply(lambda x: "-".join([str(int(el)) for el in x["TCE_ID"].split("-")]), axis=1)

# load DV TCE table for the inverted run
tce_tbl_fp = Path('/Users/msaragoc/Projects/exoplanet_transit_classification/data/ephemeris_tables/kepler/q1-q17_dr25/simulated_data/transit_inversion/dvOutputMatrix.csv')
tce_tbl = pd.read_csv(tce_tbl_fp)
tce_tbl['uid'] = tce_tbl[['keplerId', 'planetIndexNumber']].apply(lambda x: f'{int(x["keplerId"])}-{int(x["planetIndexNumber"])}', axis=1)
tce_tbl['dataset'] = np.nan

print(f'Pairing Robovetter input table for injection group {robovetter_tbl["Data_Set"].values[0]}...')
print(f'Number of TCEs in the Robovetter input table: {len(robovetter_tbl)}')
shared_tces = tce_tbl['uid'].isin(robovetter_tbl['uid'])
n_tces_in_tce_tbl = shared_tces.sum()
n_targets_in_tce_tbl = len(tce_tbl.loc[tce_tbl['keplerId'].isin(robovetter_tbl['KIC']), 'keplerId'].unique())
print(f'Found {n_tces_in_tce_tbl} TCEs (out of {len(robovetter_tbl)})')
print(f'Found {n_targets_in_tce_tbl} KICs (out of {len(robovetter_tbl["KIC"].unique())})')
tce_tbl.loc[shared_tces, 'dataset'] = robovetter_tbl["Data_Set"].values[0]

print(tce_tbl['dataset'].value_counts())
print(f'Number of TCEs not found in any injection group: {tce_tbl["dataset"].isna().sum()}')

# exclude TCEs not found in the Robovetter input table for the inverted run
tce_tbl = tce_tbl.loc[~tce_tbl["dataset"].isna()]

tce_tbl.to_csv('/Users/msaragoc/Projects/exoplanet_transit_classification/data/ephemeris_tables/kepler/q1-q17_dr25/simulated_data/dvOutputMatrix_tces_only_robovetter_input_inv.csv', index=False)

#%% Get only TCEs from the DV scrambled runs tables that are in the Robovetter input table for the scrambled groups

columnNames = ['Data_Set', 'TCE_ID', 'KIC', 'pn', 'n_plan', 'ntrans', 'nrealt', 'ngoodt', 'iflag0', 'iflag1', 'iflag2',
              'iflag3', 'iflag4', 'iflag5', 'cflag0', 'cflag1', 'cflag2', 'cflag3', 'cflag4', 'cflag5', 'cflag6',
               'cflag7', 'cflag8', 'ephem_disp', 'cent_disp', 'cent_score', 'period', 'period_err', 'epoch',
               'epoch_err', 'duration', 'duration_err', 'impact', 'impact_err', 'depth', 'depth_err', 'depth_alt',
               'sma', 'rstar', 'mes', 'ses_to_mes', 'new_mes', 'lpp_dv', 'lpp_alt', 'all_tran_chases', 'sweet_snr',
               'sweet_amp', 'shape_metric', 'halo_ghost', 'mod_sig_pri_dv', 'mod_sig_sec_dv', 'mod_sig_ter_dv',
              'mod_sig_pos_dv', 'mod_fred_dv', 'mod_fa1_dv', 'mod_fa2_dv', 'mod_sig_pri_alt', 'mod_sig_sec_alt',
               'mod_sig_ter_alt', 'mod_sig_pos_alt', 'mod_fred_alt', 'mod_fa1_alt', 'mod_fa2_alt', 'modshiftval1_dv',
              'modshiftval2_dv', 'modshiftval3_dv', 'modshiftval4_dv', 'modshiftval5_dv', 'modshiftval6_dv',
               'modshiftval1_alt', 'modshiftval2_alt', 'modshiftval3_alt', 'modshiftval4_alt', 'modshiftval5_alt',
               'modshiftval6_alt', 'oesig_dv', 'oesig_alt', 'mod_oe_dv', 'mod_oe_alt', 'rp_dv', 'rp_dv_perr',
               'rp_dv_merr', 'alb_dv', 'alb_dv_perr', 'alb_dv_merr', 'mod_depth_pri_dv', 'mod_depth_sec_dv',
               'mod_ph_sec_dv', 'mod_ph_ter_dv', 'rp_alt', 'rp_alt_perr', 'rp_alt_merr', 'alb_alt', 'alb_alt_perr',
               'alb_alt_merr', 'mod_depth_pri_alt', 'mod_depth_sec_alt', 'mod_ph_sec_alt', 'mod_ph_ter_alt',
               'sdepthsig_dv', 'sdepthsig_alt']

# load Robovetter input table for the scramble group
robovetter_tbl_fp = Path('/Users/msaragoc/Projects/exoplanet_transit_classification/data/ephemeris_tables/kepler/q1-q17_dr25/simulated_data/robovetter_input_tables/Scrambled_Q1-Q17_DR25/kplr_dr25_scr2_robovetter_input.txt')
robovetter_tbl = pd.read_table(robovetter_tbl_fp, skiprows=223, names=columnNames, skipinitialspace=False, delim_whitespace=True)
robovetter_tbl['uid'] = robovetter_tbl[['TCE_ID']].apply(lambda x: "-".join([str(int(el)) for el in x["TCE_ID"].split("-")]), axis=1)

# load DV TCE table for a scramble group
tce_tbl_fp = Path('/Users/msaragoc/Projects/exoplanet_transit_classification/data/ephemeris_tables/kepler/q1-q17_dr25/simulated_data/scrambling_quarters/dvOutputMatrix_scr2.csv')
tce_tbl = pd.read_csv(tce_tbl_fp)
tce_tbl['uid'] = tce_tbl[['keplerId', 'planetIndexNumber']].apply(lambda x: f'{int(x["keplerId"])}-{int(x["planetIndexNumber"])}', axis=1)
tce_tbl['dataset'] = np.nan

print(f'Pairing Robovetter input table for scrambled run {robovetter_tbl["Data_Set"].values[0]}...')
print(f'Number of TCEs in the Robovetter input table: {len(robovetter_tbl)}')
shared_tces = tce_tbl['uid'].isin(robovetter_tbl['uid'])
n_tces_in_tce_tbl = shared_tces.sum()
n_targets_in_tce_tbl = len(tce_tbl.loc[tce_tbl['keplerId'].isin(robovetter_tbl['KIC']), 'keplerId'].unique())
print(f'Found {n_tces_in_tce_tbl} TCEs (out of {len(robovetter_tbl)})')
print(f'Found {n_targets_in_tce_tbl} KICs (out of {len(robovetter_tbl["KIC"].unique())})')
tce_tbl.loc[shared_tces, 'dataset'] = robovetter_tbl["Data_Set"].values[0]

print(tce_tbl['dataset'].value_counts())
print(f'Number of TCEs not found in the scrambled run: {tce_tbl["dataset"].isna().sum()}')

# exclude TCEs not found in the Robovetter input tables for the injection groups
tce_tbl = tce_tbl.loc[~tce_tbl["dataset"].isna()]

tce_tbl.to_csv('/Users/msaragoc/Projects/exoplanet_transit_classification/data/ephemeris_tables/kepler/q1-q17_dr25/simulated_data/dvOutputMatrix_tces_only_robovetter_input_scr2.csv', index=False)

#%% Merge all DV TCE tables

sim_tbls_dir = Path('/Users/msaragoc/Projects/exoplanet_transit_classification/data/ephemeris_tables/kepler/q1-q17_dr25/simulated_data/')
sim_runs_fp_list = [fp for fp in sim_tbls_dir.iterdir() if fp.name.startswith('dvOutputMatrix_tces_only_robovetter_input_')]

sim_tbl_all = pd.concat([pd.read_csv(fp) for fp in sim_runs_fp_list], axis=0)

sim_tbl_all.to_csv(sim_tbls_dir / 'dvOutputMatrix_allruns.csv', index=False)
