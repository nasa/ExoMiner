import os
import pandas as pd
import numpy as np

#%% Convert ascii scrambled TCE tables to csv

scrTblDir = '/data5/tess_project/Data/Ephemeris_tables/Kepler/Scrambled DR25'
scrTblFps = [os.path.join(scrTblDir, scrTblFn) for scrTblFn in os.listdir(scrTblDir) if '.txt' in scrTblFn]

# columnNames = ['Data_Set', 'TCE_ID', 'KIC', 'pn', 'n_plan', 'ntrans', 'nrealt', 'ngoodt', 'iflag0', 'iflag1', 'iflag2',
#               'iflag3', 'iflag4', 'iflag5', 'cflag0', 'cflag1', 'cflag2', 'cflag3', 'cflag4', 'cflag5' 'cflag6',
#                'cflag7', 'cflag8', 'ephem_disp', 'cent_disp', 'cent_score', 'period', 'period_err', 'epoch',
#                'epoch_err', 'duration', 'duration_err', 'impact', 'impact_err', 'depth', 'depth_err', 'depth_alt',
#                'sma', 'rstar', 'mes', 'ses_to_mes', 'new_mes', 'lpp_dv', 'lpp_alt', 'all_tran_chases', 'sweet_snr',
#                'sweet_amp', 'shape_metric', 'halo_ghost', 'mod_sig_pri_dv', 'mod_sig_sec_dv', 'mod_sig_ter_dv',
#               'mod_sig_pos_dv|    mod_fred_dv|     mod_fa1_dv|     mod_fa2_dv|mod_sig_pri_alt|mod_sig_sec_alt|mod_sig_ter_alt|mod_sig_pos_alt|   mod_fred_alt|    mod_fa1_alt|    mod_fa2_alt| modshiftval1_dv| modshiftval2_dv| modshiftval3_dv| modshiftval4_dv| modshiftval5_dv| modshiftval6_dv|modshiftval1_alt|modshiftval2_alt|modshiftval3_alt|modshiftval4_alt|modshiftval5_alt|modshiftval6_alt|       oesig_dv|      oesig_alt|      mod_oe_dv|     mod_oe_alt|          rp_dv|     rp_dv_perr|     rp_dv_merr|         alb_dv|    alb_dv_perr|    alb_dv_merr|mod_depth_pri_dv|mod_depth_sec_dv|  mod_ph_sec_dv|  mod_ph_ter_dv|         rp_alt|    rp_alt_perr|    rp_alt_merr|        alb_alt|   alb_alt_perr|   alb_alt_merr|mod_depth_pri_alt|mod_depth_sec_alt| mod_ph_sec_alt| mod_ph_ter_alt|    sdepthsig_dv|   sdepthsig_alt]

columnNames = ['TCE_ID', 'KIC', 'Disp', 'Score', 'NTL', 'SS', 'CO', 'EM', 'period', 'epoch', 'Expected_MES', 'MES',
               'NTran', 'depth', 'duration', 'Rp', 'Rs', 'Ts', 'logg', 'a', 'Rp/Rs', 'a/Rs', 'impact', 'SNR_DV', 'Sp',
               'Fit_Prov']

for scrTblFp in scrTblFps:

    scrDf = pd.read_table(scrTblFp, skiprows=75, names=columnNames, skipinitialspace=False, delim_whitespace=True)

    print(scrDf.head())
    # aaa
    print(len(scrDf))

    scrDf.to_csv(scrTblFp.replace('.txt', '.csv'), index=False)


#%% Add stellar parameters to scrambled TCE tables

stellar_tbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Stellar parameters/Kepler/'
                          'q1_q17_dr25_stellar_gaiadr2_nanstosolar.csv')

stellar_fields_out = ['kepmag', 'tce_steff', 'tce_steff_err1', 'tce_steff_err2', 'tce_slogg', 'tce_slogg_err1',
                      'tce_slogg_err2', 'tce_smet', 'tce_smet_err1', 'tce_smet_err2', 'tce_sradius', 'tce_sradius_err1',
                      'tce_sradius_err2', 'tce_smass', 'tce_smass_err1', 'tce_smass_err2', 'tce_sdens',
                      'tce_sdens_err1', 'tce_dens_serr2', 'ra', 'dec']
stellar_fields_in = ['kepmag', 'teff', 'teff_err1', 'teff_err2', 'logg', 'logg_err1', 'logg_err2', 'feh', 'feh_err1',
                     'feh_err2', 'radius', 'radius_err1', 'radius_err2', 'mass', 'mass_err1', 'mass_err2', 'dens',
                     'dens_err1', 'dens_err2', 'ra', 'dec']

scrTblDir = '/data5/tess_project/Data/Ephemeris_tables/Kepler/Scrambled DR25'
scrTblFps = [os.path.join(scrTblDir, scrTblFn) for scrTblFn in os.listdir(scrTblDir)
             if scrTblFn.endswith('.csv')]

count_vec = []
for scrTblFp in scrTblFps:
    print(scrTblFp.split('/')[-1])

    scrDf = pd.read_csv(scrTblFp)

    for stellar_param in stellar_fields_out:
        scrDf[stellar_param] = np.nan

    count = 0
    for row_star_i, row_star in stellar_tbl.iterrows():

        if row_star_i % 100 == 0:
            print('Star {} out of {} ({} %)\n Number of scrambled TCEs updated: {}'.format(row_star_i,
                                                                                 len(stellar_tbl),
                                                                                 row_star_i / len(stellar_tbl) * 100,
                                                                                 count))

        target_cond = scrDf['KIC'] == row_star['kepid']

        count += target_cond.sum()

        scrDf.loc[target_cond, stellar_fields_out] = row_star[stellar_fields_in].values

    count_vec.append((scrTblFp, count))
    print('Number of scrambled TCEs updated: {}'.format(count))
    scrDf.to_csv(scrTblFp.replace('.csv', '_stellar.csv'), index=False)

print(count_vec)

#%% Check that stellar parameters are not NaN for any scrambled TCE

stellar_fields_out = ['kepmag', 'tce_steff', 'tce_steff_err1', 'tce_steff_err2', 'tce_slogg', 'tce_slogg_err1',
                      'tce_slogg_err2', 'tce_smet', 'tce_smet_err1', 'tce_smet_err2', 'tce_sradius', 'tce_sradius_err1',
                      'tce_sradius_err2', 'tce_smass', 'tce_smass_err1', 'tce_smass_err2', 'tce_sdens',
                      'tce_sdens_err1', 'tce_dens_serr2', 'ra', 'dec']

scrTblDir = '/data5/tess_project/Data/Ephemeris_tables/Kepler/Scrambled DR25'
scrTblFps = [os.path.join(scrTblDir, scrTblFn) for scrTblFn in os.listdir(scrTblDir)
             if scrTblFn.endswith('stellar.csv')]

for scrTblFp in scrTblFps:
    print(scrTblFp.split('/')[-1])

    scrDf = pd.read_csv(scrTblFp, usecols=stellar_fields_out)

    print(scrDf.isna().any(axis=0))

#%% Standardize fields

# changing the field name in rawFields
rawFields = ['KIC', 'period', 'epoch', 'duration', 'depth', 'TCE_ID']
newFields = ['target_id', 'tce_period', 'tce_time0bk', 'tce_duration', 'transit_depth', 'tce_plnt_num']

scrTblDir = '/data5/tess_project/Data/Ephemeris_tables/Kepler/Scrambled DR25'
scrTblFps = [os.path.join(scrTblDir, scrTblFn) for scrTblFn in os.listdir(scrTblDir)
             if scrTblFn.endswith('stellar.csv')]

for scrTblFp in scrTblFps:
    print(scrTblFp.split('/')[-1])

    scrDf = pd.read_csv(scrTblFp)
    print(len(scrDf))

    # # remove TCEs with any NaN in the required fields
    # injDf.dropna(axis=0, subset=np.array(rawFields)[[0, 1, 2, 3, 6, 8, 10, 12]], inplace=True)
    # print(len(injDf))

    # rename fields to standardize fieldnames
    renameDict = {}
    for i in range(len(rawFields)):
        renameDict[rawFields[i]] = newFields[i]
    scrDf.rename(columns=renameDict, inplace=True)

    # remove TCEs with zero period or transit duration
    scrDf = scrDf.loc[(scrDf['tce_period'] > 0) & (scrDf['tce_duration'] > 0)]
    print(len(scrDf))

    scrDf.to_csv(scrTblFp.replace('.csv', '_processed.csv'), index=False)

#%% Add label to the scrambled TCEs

scrTblDir = '/data5/tess_project/Data/Ephemeris_tables/Kepler/Scrambled DR25'
scrTblFps = [os.path.join(scrTblDir, scrTblFn) for scrTblFn in os.listdir(scrTblDir)
             if scrTblFn.endswith('processed.csv')]

for scrTblFp in scrTblFps:
    print(scrTblFp.split('/')[-1])

    scrDf = pd.read_csv(scrTblFp)

    scrDf['label'] = 'AFP'

    scrDf.to_csv(scrTblFp.replace('.csv', '_withlabels.csv'), index=False)
