import os
import pandas as pd
import numpy as np

#%% Convert ascii injected TCE tables to csv

injTblDir = '/data5/tess_project/Data/Ephemeris_tables/Kepler/Injected DR25'
injTblFps = [os.path.join(injTblDir, injTblFn) for injTblFn in os.listdir(injTblDir) if '.txt' in injTblFn]

columnNames = ['KIC_ID', 'Sky_Group', 'i_period', 'i_epoch', 'N_Transit', 'i_depth', 'i_dur', 'i_b', 'i_ror', 'i_dor',
               'EB_injection', 'Offset_from_source', 'Offset_distance', 'Expected_MES', 'Recovered', 'TCE_ID',
               'Measured_MES', 'r_period', 'r_epoch', 'r_depth', 'r_dur', 'r_b', 'r_ror', 'r_dor', 'Fit_Provenance']

for injTblFp in injTblFps:

    injDf = pd.read_table(injTblFp, skiprows=73, names=columnNames, skipinitialspace=False, delim_whitespace=True)

    print(injDf.head())
    # aaa
    print(len(injDf))

    injDf.to_csv(injTblFp.replace('.txt', '.csv'), index=False)


#%% Add stellar parameters to injected TCE tables

stellar_tbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Stellar parameters/Kepler/'
                          'q1_q17_dr25_stellar_gaiadr2_nanstosolar.csv')

stellar_fields_out = ['kepmag', 'tce_steff', 'tce_steff_err1', 'tce_steff_err2', 'tce_slogg', 'tce_slogg_err1',
                      'tce_slogg_err2', 'tce_smet', 'tce_smet_err1', 'tce_smet_err2', 'tce_sradius', 'tce_sradius_err1',
                      'tce_sradius_err2', 'tce_smass', 'tce_smass_err1', 'tce_smass_err2', 'tce_sdens',
                      'tce_sdens_err1', 'tce_dens_serr2', 'ra', 'dec']
stellar_fields_in = ['kepmag', 'teff', 'teff_err1', 'teff_err2', 'logg', 'logg_err1', 'logg_err2', 'feh', 'feh_err1',
                     'feh_err2', 'radius', 'radius_err1', 'radius_err2', 'mass', 'mass_err1', 'mass_err2', 'dens',
                     'dens_err1', 'dens_err2', 'ra', 'dec']

injTblDir = '/data5/tess_project/Data/Ephemeris_tables/Kepler/Injected DR25'
injTblFps = [os.path.join(injTblDir, injTblFn) for injTblFn in os.listdir(injTblDir)
             if injTblFn.endswith('.csv')]

count_vec = []
for injTblFp in injTblFps:
    print(injTblFp.split('/')[-1])

    injDf = pd.read_csv(injTblFp)

    for stellar_param in stellar_fields_out:
        injDf[stellar_param] = np.nan

    count = 0
    for row_star_i, row_star in stellar_tbl.iterrows():

        if row_star_i % 100 == 0:
            print('Star {} out of {} ({} %)\n Number of injected TCEs updated: {}'.format(row_star_i,
                                                                                 len(stellar_tbl),
                                                                                 row_star_i / len(stellar_tbl) * 100,
                                                                                 count))

        target_cond = injDf['KIC_ID'] == row_star['kepid']

        count += target_cond.sum()

        injDf.loc[target_cond, stellar_fields_out] = row_star[stellar_fields_in].values

    count_vec.append((injTblFp, count))
    print('Number of injected TCEs updated: {}'.format(count))
    injDf.to_csv(injTblFp.replace('.csv', '_stellar.csv'), index=False)

print(count_vec)

#%% Check that stellar parameters are not NaN for any injected TCE

stellar_fields_out = ['kepmag', 'tce_steff', 'tce_steff_err1', 'tce_steff_err2', 'tce_slogg', 'tce_slogg_err1',
                      'tce_slogg_err2', 'tce_smet', 'tce_smet_err1', 'tce_smet_err2', 'tce_sradius', 'tce_sradius_err1',
                      'tce_sradius_err2', 'tce_smass', 'tce_smass_err1', 'tce_smass_err2', 'tce_sdens',
                      'tce_sdens_err1', 'tce_dens_serr2', 'ra', 'dec']

injTblDir = '/data5/tess_project/Data/Ephemeris_tables/Kepler/Injected DR25'
injTblFps = [os.path.join(injTblDir, injTblFn) for injTblFn in os.listdir(injTblDir)
             if injTblFn.endswith('stellar.csv')]

for injTblFp in injTblFps:
    print(injTblFp.split('/')[-1])

    injDf = pd.read_csv(injTblFp, usecols=stellar_fields_out)

    print(injDf.isna().any(axis=0))

#%% Standardize fields

# changing the field name in rawFields
rawFields = ['KIC_ID', 'r_period', 'r_epoch', 'r_dur', 'r_depth', 'TCE_ID']
newFields = ['target_id', 'tce_period', 'tce_time0bk', 'tce_duration', 'transit_depth', 'tce_plnt_num']

injTblDir = '/data5/tess_project/Data/Ephemeris_tables/Kepler/Injected DR25'
injTblFps = [os.path.join(injTblDir, injTblFn) for injTblFn in os.listdir(injTblDir)
             if injTblFn.endswith('stellar.csv')]

for injTblFp in injTblFps:
    print(injTblFp.split('/')[-1])

    injDf = pd.read_csv(injTblFp)
    print(len(injDf))

    # # remove TCEs with any NaN in the required fields
    # injDf.dropna(axis=0, subset=np.array(rawFields)[[0, 1, 2, 3, 6, 8, 10, 12]], inplace=True)
    # print(len(injDf))

    # rename fields to standardize fieldnames
    renameDict = {}
    for i in range(len(rawFields)):
        renameDict[rawFields[i]] = newFields[i]
    injDf.rename(columns=renameDict, inplace=True)

    # remove TCEs with zero period or transit duration
    injDf = injDf.loc[(injDf['tce_period'] > 0) & (injDf['tce_duration'] > 0)]
    print(len(injDf))

    injDf.to_csv(injTblFp.replace('.csv', '_processed.csv'), index=False)

#%% Add label to the injected TCEs

injTblDir = '/data5/tess_project/Data/Ephemeris_tables/Kepler/Injected DR25'
injTblFps = [os.path.join(injTblDir, injTblFn) for injTblFn in os.listdir(injTblDir)
             if injTblFn.endswith('processed.csv')]

for injTblFp in injTblFps:
    print(injTblFp.split('/')[-1])

    injDf = pd.read_csv(injTblFp)

    if 'inj1' in injTblFp:
        injDf['label'] = 'PC'
    else:
        injDf['label'] = 'AFP'

    injDf.to_csv(injTblFp.replace('.csv', '_withlabels.csv'), index=False)

#%% Filter non-recovered TCEs by the Kepler pipeline and duplicate ones

injTblDir = '/data5/tess_project/Data/Ephemeris_tables/Kepler/Injected DR25'
injTblFps = [os.path.join(injTblDir, injTblFn) for injTblFn in os.listdir(injTblDir)
             if injTblFn.endswith('withlabels.csv')]

for injTblFp in injTblFps:
    print(injTblFp.split('/')[-1])

    injDf = pd.read_csv(injTblFp)

    injDf = injDf.loc[injDf['Recovered'] != 0]

    injDf.drop_duplicates(subset=['tce_plnt_num'], inplace=True)

    print(injDf.head())
    # aaa
    print(len(injDf))

    injDf.to_csv(injTblFp.replace('.csv', '_recovered.csv'), index=False)
