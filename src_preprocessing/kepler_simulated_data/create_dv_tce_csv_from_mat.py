""" Create DV TCE csv files from the source mat files for the Kepler simulated data. """

# 3rd party
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.io import loadmat
import mat73

dv_mat_tbl_fp = Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/data/ephemeris_tables/kepler/q1-q17_dr25/simulated_data/pixel_level_transit_injections/ksop-2540-dv-batch4/dv/dvOutputMatrix.mat')
dv_mat_tbl = loadmat(dv_mat_tbl_fp)

print(f'Creating csv table for mat file {dv_mat_tbl_fp.name}')

csv_save_dir = Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/data/ephemeris_tables/kepler/q1-q17_dr25/simulated_data/pixel_level_transit_injections')

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

#%% Check against Robovetter input table

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


robovetter_tbl_fp = Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/data/ephemeris_tables/kepler/q1-q17_dr25/simulated_data/robovetter_input_tables/Injected_Q1-Q17_DR25/kplr_dr25_inj3_robovetter_input.txt')
robovetter_tbl = pd.read_table(robovetter_tbl_fp, skiprows=223, names=columnNames, skipinitialspace=False, delim_whitespace=True)
robovetter_tbl['uid'] = robovetter_tbl[['TCE_ID']].apply(lambda x: "-".join([str(int(el)) for el in x["TCE_ID"].split("-")]), axis=1)

tce_tbl['uid'] = tce_tbl[['keplerId', 'planetIndexNumber']].apply(lambda x: f'{int(x["keplerId"])}-{int(x["planetIndexNumber"])}', axis=1)

a = tce_tbl.loc[tce_tbl['uid'].isin(robovetter_tbl['uid'])]
b = robovetter_tbl.loc[robovetter_tbl['uid'].isin(tce_tbl['uid'])]

#%% Merge all DV TCE tables

sim_runs_dict = {
    'INV': Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/data/ephemeris_tables/kepler/q1-q17_dr25/simulated_data/transit_inversion/dvOutputMatrix.csv'),
    'SCR1': Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/data/ephemeris_tables/kepler/q1-q17_dr25/simulated_data/scrambling_quarters/dvOutputMatrix_scr1.csv'),
    'SCR2': Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/data/ephemeris_tables/kepler/q1-q17_dr25/simulated_data/scrambling_quarters/dvOutputMatrix_scr2.csv'),
    'INJ_B1': Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/data/ephemeris_tables/kepler/q1-q17_dr25/simulated_data/pixel_level_transit_injections/dvOutputMatrix_batch1.csv'),
    'INJ_B2': Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/data/ephemeris_tables/kepler/q1-q17_dr25/simulated_data/pixel_level_transit_injections/dvOutputMatrix_batch2.csv'),
    'INJ_B3': Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/data/ephemeris_tables/kepler/q1-q17_dr25/simulated_data/pixel_level_transit_injections/dvOutputMatrix_batch3.csv'),
    'INJ_B4': Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/data/ephemeris_tables/kepler/q1-q17_dr25/simulated_data/pixel_level_transit_injections/dvOutputMatrix_batch4.csv'),
}

sim_tbls = []
for run_name, run_tbl_fp in sim_runs_dict.items():
    run_tbl = pd.read_csv(run_tbl_fp)
    run_tbl['run'] = run_name
    sim_tbls.append(run_tbl)

sim_tbl_all = pd.concat(sim_tbls, axis=0)
sim_tbl_all.to_csv('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/data/ephemeris_tables/kepler/q1-q17_dr25/simulated_data/dvOutputMatrix_allruns.csv', index=False)
