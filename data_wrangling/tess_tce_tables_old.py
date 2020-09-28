

#%% Preprocess TEV MIT TCE lists

rawTceTable = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/TESS/TEV_MIT_TOI_lists/raw/'
                          'toi-plus-tev.mit.edu_2020-04-15.csv', header=4)
print(len(rawTceTable))
rawTceTableGroupDisposition = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/TESS/TEV_MIT_TOI_lists/raw/'
                                          'toi-tev.mit.edu_2020-04-15.csv', header=4,
                                          usecols=['Full TOI ID', 'Group Disposition'])
print(len(rawTceTableGroupDisposition))

# add group disposition to the TCE plus list
rawTceTable['Group Disposition'] = np.nan
for tce_i, tce in rawTceTableGroupDisposition.iterrows():
    rawTceTable.loc[rawTceTable['Full TOI ID'] == tce['Full TOI ID'], 'Group Disposition'] = tce['Group Disposition']

rawTceTable.to_csv('/data5/tess_project/Data/Ephemeris_tables/TESS/TEV_MIT_TOI_lists/'
                   'toi-plus-tev.mit.edu_2020-04-15_addedgroupdisposition.csv', index=False)

# changing the field name in rawFields
rawFields = ['TIC', 'Full TOI ID', 'TIC Right Ascension', 'TIC Declination', 'TMag Value',
             'TMag Uncertainty', 'Orbital Epoch Value', 'Orbital Epoch Error', 'Orbital Period Value',
             'Orbital Period Error', 'Transit Duration Value', 'Transit Duration Error', 'Transit Depth Value',
             'Transit Depth Error', 'Sectors', 'Surface Gravity Value', 'Surface Gravity Uncertainty',
             'Star Radius Value', 'Star Radius Error', 'Effective Temperature Value',
             'Effective Temperature Uncertainty']
newFields = ['target_id', 'oi', 'ra', 'dec', 'mag', 'mag_err', 'tce_time0bk', 'tce_time0bk_err',
             'tce_period', 'tce_period_err', 'tce_duration', 'tce_duration_err', 'transit_depth', 'transit_depth_err',
             'sectors', 'tce_slogg', 'tce_slogg_err', 'tce_sradius', 'tce_sradius_err', 'tce_steff', 'tce_steff_err']

# remove TCEs with any NaN in the required fields
rawTceTable.dropna(axis=0, subset=np.array(rawFields)[[0, 1, 2, 3, 6, 8, 10, 12]], inplace=True)
print(len(rawTceTable))

# rename fields to standardize fieldnames
renameDict = {}
for i in range(len(rawFields)):
    renameDict[rawFields[i]] = newFields[i]
rawTceTable.rename(columns=renameDict, inplace=True)

# remove TCEs with zero period or transit duration
rawTceTable = rawTceTable.loc[(rawTceTable['tce_period'] > 0) & (rawTceTable['tce_duration'] > 0)]
print(len(rawTceTable))

# rawTceTable = rawTceTable.drop(['Edited', 'Alerted'], axis=1)

# 1732 TCEs
rawTceTable.to_csv('/data5/tess_project/Data/Ephemeris_tables/TESS/TEV_MIT_TOI_lists/'
                   'toi-plus-tev.mit.edu_2020-04-15_processed.csv', index=False)

# split TCE list into two TCE lists with TOI and Group disposition
rawTceTable = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/TESS/TEV_MIT_TOI_lists/'
                          'toi-plus-tev.mit.edu_2020-04-15_stellar.csv')

disposition_src = ['Group', 'TOI']
for disposition in disposition_src:
    dispTceTable = rawTceTable.rename(columns={'{} Disposition'.format(disposition): 'label'})
    dispTceTable.to_csv('/data5/tess_project/Data/Ephemeris_tables/TESS/TEV_MIT_TOI_lists/'
                        'toi-plus-tev.mit.edu_2020-04-15_{}disposition.csv'.format(disposition), index=False)

#%% Preprocess NASA Exoplanet Archive TESS TCE list to standardized fields

rawTceTable = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/TESS/NASA_Exoplanet_Archive_TOI_lists/raw/'
                          'TOI_2020.04.14_23.04.53.csv', header=69)

# 1766
print(len(rawTceTable))

# Create Group or TOI disposition table by changing the field name in rawFields
rawFields = ['toi', 'tid', 'tfopwg_disp', 'ra', 'dec', 'st_tmag', 'st_tmagerr1', 'st_tmagerr2', 'pl_tranmid',
             'pl_tranmiderr1', 'pl_tranmiderr2', 'pl_orbper',
             'pl_orbpererr1', 'pl_orbpererr2', 'pl_trandurh', 'pl_trandurherr1', 'pl_trandurherr2', 'pl_trandep',
             'pl_trandeperr1', 'pl_trandeperr2', 'st_teff', 'st_tefferr1', 'st_tefferr2', 'st_logg', 'st_loggerr1',
             'st_loggerr2', 'st_rad', 'st_raderr1', 'st_raderr2']
newFields = ['oi', 'target_id', 'label', 'ra', 'dec', 'mag', 'mag_uncert', 'mag_uncert2', 'tce_time0bk',
             'tce_time0bk_err', 'tce_time0bk_err2', 'tce_period', 'tce_period_err', 'tce_period_err2', 'tce_duration',
             'tce_duration_err', 'tce_duration_err2', 'transit_depth', 'transit_depth_err', 'transit_depth_err2',
             'tce_steff', 'tce_steff_err1', 'tce_steff_err2', 'tce_slogg', 'tce_slogg_err1', 'tce_slogg_err2',
             'tce_sradius', 'tce_sradius_err1', 'tce_sradius_err2']

# remove TCEs with any NaN in the required fields - except label field
rawTceTable.dropna(axis=0, subset=np.array(rawFields)[[0, 1, 3, 4, 8, 11, 14, 17]], inplace=True)
# 1730
print(len(rawTceTable))

# rename fields to standardize fieldnames
renameDict = {}
for i in range(len(rawFields)):
    renameDict[rawFields[i]] = newFields[i]
rawTceTable.rename(columns=renameDict, inplace=True)

# remove TCEs with zero period or transit duration
rawTceTable = rawTceTable.loc[(rawTceTable['tce_period'] > 0) & (rawTceTable['tce_duration'] > 0)]
# 1730
print(len(rawTceTable))

# convert epoch value from BJD to TBJD
rawTceTable['tce_time0bk'] -= 2457000
# 1730 TCEs
rawTceTable.to_csv('/data5/tess_project/Data/Ephemeris_tables/TESS/NASA_Exoplanet_Archive_TOI_lists/'
                   'TOI_2020.04.14_23.04.53_processed.csv', index=False)

rawTceTable = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/TESS/NASA_Exoplanet_Archive_TOI_lists/'
                          'TOI_2020.04.14_23.04.53_stellar.csv')
print(len(rawTceTable))

# drop unlabeled TCEs
rawTceTable.dropna(axis=0, subset=['label'], inplace=True)
print(len(rawTceTable))

# 645 TCEs
rawTceTable.to_csv('/data5/tess_project/Data/Ephemeris_tables/TESS/NASA_Exoplanet_Archive_TOI_lists/'
                   'TOI_2020.04.14_23.04.53_TFOPWG.csv', index=False)

#%% Preprocess fields in EXOFOP Community TESS disposition list

rawTceTable = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/TESS/EXOFOP_TOI_lists/exofop_ctoilists.csv',
                          header=0)
print(len(rawTceTable))

rawFields = ['CTOI', 'TIC ID', 'User Disposition', 'RA', 'Dec', 'TESS Mag', 'TESS Mag err', 'Midpoint (BJD)',
             'Midpoint err', 'Period (days)', 'Period (days) Error', 'Duration (hrs)', 'Duration (hrs) Error',
             'Depth ppm', 'Depth ppm Error']
newFields = ['oi', 'target_id', 'label', 'ra', 'dec', 'mag', 'mag_uncert', 'tce_time0bk', 'tce_time0bk_err',
             'tce_period', 'tce_period_err', 'tce_duration', 'tce_duration_err', 'transit_depth', 'transit_depth_err']

# remove TCEs with any NaN in the required fields
rawTceTable.dropna(axis=0, subset=np.array(rawFields)[[0, 1, 2, 3, 4, 7, 9, 11, 13]], inplace=True)
print(len(rawTceTable))

# rename fields to standardize fieldnames
renameDict = {}
for i in range(len(rawFields)):
    renameDict[rawFields[i]] = newFields[i]
rawTceTable.rename(columns=renameDict, inplace=True)

# remove TCEs with zero period or transit duration
rawTceTable = rawTceTable.loc[(rawTceTable['tce_period'] > 0) & (rawTceTable['tce_duration'] > 0)]
print(len(rawTceTable))

# convert epoch value from BJD to TBJD
rawTceTable['tce_time0bk'] -= 2457000

# 321 to 273 TCEs
rawTceTable.to_csv('/data5/tess_project/Data/Ephemeris_tables/TESS/EXOFOP_TOI_lists/final_tce_tables/'
                   'exofop_ctoilists_Community_processed.csv', index=False)

#%%

# rawTceTbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/TESS/NASA_Exoplanet_Archive_TOI_lists/'
#                         'TOI_2020.01.21_13.55.10.csv', header=72)
#
# print(len(rawTceTbl))
#
# rawTceTbl = rawTceTbl.dropna(axis=0, subset=['tfopwg_disp'])
#
# print(len(rawTceTbl))
#
# rawTceTbl = rawTceTbl.loc[(rawTceTbl['pl_orbper'] > 0) & (rawTceTbl['pl_trandurh'] > 0)]
# print(len(rawTceTbl))
#
#
# stellar_params = ['st_teff', 'st_logg', 'st_rad', 'ra', 'dec']
#
# print(rawTceTbl[stellar_params].isna().sum(axis=0))

tceTbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/TESS/TEV_MIT_TOI_lists/'
                     'toi-plus-tev.mit.edu_2020-01-15_TOI Disposition_processed_stellar.csv')

ticsFits = np.load('/data5/tess_project/Data/Ephemeris_tables/Stellar parameters/TESS/final_target_list_s1-s20.npy')

print(len(tceTbl))

tceTbl = tceTbl.loc[tceTbl['target_id'].isin(ticsFits)]

print(len(tceTbl))

stellarColumns = ['ra', 'dec', 'tce_steff', 'tce_slogg', 'tce_sradius', 'tce_smass', 'tce_smet', 'tce_sdens']

# count all nan stellar rows
print(tceTbl.loc[tceTbl[stellarColumns].isna().all(axis=1)])

print(tceTbl[stellarColumns].isna().sum())

targetsTbl = tceTbl.drop_duplicates(subset=['target_id'])
print(targetsTbl[stellarColumns].isna().sum())

#%% Updated stellar parameters in TESS TCE using the values found in the TESS FITS files

fitsTbl = pd.read_csv('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Kepler_planet_finder/stellar_parameters_analysis/'
                      'tess_stellar_parameters_fits/fitsStellarTbl_s1-s20_unique.csv')

tceTbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/TESS/TEV_MIT_TOI_lists/'
                     'toi-plus-tev.mit.edu_2020-04-15_TOIdisposition.csv')

count = 0
stellarColumnsTceTbl = np.array(['mag', 'tce_steff', 'tce_slogg', 'tce_smet', 'tce_sradius'])
stellarColumnsFitsTbl = np.array(['TESSMAG', 'TEFF', 'LOGG', 'MH', 'RADIUS'])
print(tceTbl[stellarColumnsTceTbl].isna().sum())
for tce_i, tce in tceTbl.iterrows():

    targetInFitsTbl = fitsTbl.loc[fitsTbl['TICID'] == tce.target_id]

    if len(targetInFitsTbl) == 0:
        # print('TIC ID {} not found in FITS files'.format(tce.target_id))
        count += 1
        continue

    targetInTceTbl = tceTbl.loc[tce_i, stellarColumnsTceTbl].values

    targetInTceTbl = targetInTceTbl.astype('float')
    idxsParamsNan = np.where(np.isnan(targetInTceTbl))[0]

    if len(idxsParamsNan) == 0:
        continue  # no missing values

    tceTbl.loc[tce_i, stellarColumnsTceTbl[idxsParamsNan]] = \
        targetInFitsTbl[stellarColumnsFitsTbl[idxsParamsNan]].values[0]

    # aaaa
print(tceTbl[stellarColumnsTceTbl].isna().sum())

#%% Updated stellar parameters in TESS TCE lists using the values found in the EXOFOP TOI list

fitsTbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/TESS/EXOFOP_TOI_lists/TOI/raw/'
                      'exofop_toilists_4-17-2020.csv')

fitsTbl.drop_duplicates(subset=['TIC ID'], inplace=True)

tceTbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/TESS/TEV_MIT_TOI_lists/'
                     'toi-plus-tev.mit.edu_2020-04-15_TOIdisposition.csv')

count = 0
stellarColumnsTceTbl = np.array(['mag', 'tce_steff', 'tce_slogg', 'tce_smet', 'tce_sradius'])
stellarColumnsFitsTbl = np.array(['TESS Mag', 'Stellar Eff Temp (K)', 'Stellar log(g) (cm/s^2)', 'Stellar Metallicity',
                                  'Stellar Radius (R_Sun)'])
print(tceTbl[stellarColumnsTceTbl].isna().sum())
print(fitsTbl[stellarColumnsFitsTbl].isna().sum())
for tce_i, tce in tceTbl.iterrows():

    targetInFitsTbl = fitsTbl.loc[fitsTbl['TIC ID'] == tce.target_id]

    if len(targetInFitsTbl) == 0:
        # print('TIC ID {} not found in FITS files'.format(tce.target_id))
        count += 1
        continue

    targetInTceTbl = tceTbl.loc[tce_i, stellarColumnsTceTbl].values

    targetInTceTbl = targetInTceTbl.astype('float')
    idxsParamsNan = np.where(np.isnan(targetInTceTbl))[0]

    if len(idxsParamsNan) == 0:
        continue  # no missing values

    tceTbl.loc[tce_i, stellarColumnsTceTbl[idxsParamsNan]] = \
        targetInFitsTbl[stellarColumnsFitsTbl[idxsParamsNan]].values[0]

    # aaaa
print(tceTbl[stellarColumnsTceTbl].isna().sum())