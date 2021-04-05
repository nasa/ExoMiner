
#%% Standardize columns in TCE lists

tceTbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17_DR24/'
                     'q1_q17_dr24_tce_2020.03.02_17.51.43_stellar_shuffled.csv')

rawFields = ['kepid', 'av_training_set']
newFields = ['target_id', 'label']
# rawFields = ['kepid', 'av_training_set', 'tce_plnt_num', 'ra', 'dec', 'kepmag', 'tce_time0bk', 'tce_time0bk_err',
#              'tce_period', 'tce_period_err', 'tce_duration', 'tce_duration_err',
#              'tce_depth', 'tce_depth_err']
# newFields = ['target_id', 'label', 'tce_plnt_num', 'ra', 'dec', 'mag', 'tce_time0bk', 'tce_time0bk_err',
#              'tce_period', 'tce_period_err', 'tce_duration', 'tce_duration_err', 'transit_depth', 'transit_depth_err']

print(len(tceTbl))

# remove TCEs with any NaN in the required fields
tceTbl.dropna(axis=0, subset=['tce_period', 'tce_duration', 'tce_time0bk'], inplace=True)
print(len(tceTbl))

# rename fields to standardize fieldnames
renameDict = {}
for i in range(len(rawFields)):
    renameDict[rawFields[i]] = newFields[i]
tceTbl.rename(columns=renameDict, inplace=True)

# tceTbl.drop(columns=['rowid'], inplace=True)

tceTbl.to_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17_DR24/'
              'q1_q17_dr24_tce_2020.03.02_17.51.43_stellar_shuffled.csv', index=False)

#%% Normalize stellar parameters in the DR24 Q1-Q17 Kepler TCE table

tce_tbl_fp = '/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17_DR24/' \
             'q1_q17_dr24_tce_2020.03.02_17.51.43_nounks_shuffled.csv'
tce_tbl = pd.read_csv(tce_tbl_fp)

stellar_params = ['tce_sradius', 'tce_steff', 'tce_slogg', 'tce_smet', 'tce_smass', 'tce_sdens']

# # fill in NaNs using Solar parameters
# tce_tbl['tce_smass'] = tce_tbl['tce_smass'].fillna(value=1.0)
# tce_tbl['tce_sdens'] = tce_tbl['tce_sdens'].fillna(value=1.408)

# # remove TCEs with 'UNK' label
# tce_tbl = tce_tbl.loc[tce_tbl['av_training_set'] != 'UNK']

# # Randomly shuffle the TCE table using the same seed as Shallue and FDL
# np.random.seed(123)
# tce_table = tce_tbl.iloc[np.random.permutation(len(tce_tbl))]

# normalize using statistics computed in the training set
trainingset_idx = int(len(tce_tbl) * 0.8)

stellar_params_med = tce_tbl[stellar_params][:trainingset_idx].median(axis=0, skipna=False)
stellar_params_std = tce_tbl[stellar_params][:trainingset_idx].std(axis=0, skipna=False)
stats_norm = {'med': stellar_params_med, 'std': stellar_params_std}

np.save('{}_stats_norm.npy'.format(tce_tbl_fp.replace('.csv', '')), stats_norm)

tce_tbl[stellar_params] = (tce_tbl[stellar_params] - stellar_params_med) / stellar_params_std

tce_tbl[stellar_params].median(axis=0, skipna=False)
tce_tbl[stellar_params].std(axis=0, skipna=False)

# rawFields = ['kepid', 'av_training_set', 'tce_plnt_num', 'ra', 'dec', 'kepmag', 'tce_time0bk', 'tce_time0bk_err',
#              'tce_period', 'tce_period_err', 'tce_duration', 'tce_duration_err',
#              'tce_depth', 'tce_depth_err']
# newFields = ['target_id', 'label', 'tce_plnt_num', 'ra', 'dec', 'mag', 'tce_time0bk', 'tce_time0bk_err',
#              'tce_period', 'tce_period_err', 'tce_duration', 'tce_duration_err', 'transit_depth', 'transit_depth_err']

print(len(tce_tbl))

tce_tbl.rename(columns={'tce_depth': 'transit_depth'}, inplace=True)

# remove TCEs with any NaN in the required fields
# tce_tbl.dropna(axis=0, subset=np.array(rawFields)[[0, 1, 2, 3, 4, 6, 8, 10, 12]], inplace=True)
tce_tbl.dropna(axis=0, subset=['target_id', 'tce_plnt_num', 'tce_period', 'tce_time0bk', 'tce_duration',
                               'transit_depth', 'ra', 'dec'], inplace=True)
print(len(tce_tbl))

# # rename fields to standardize fieldnames
# renameDict = {}
# for i in range(len(rawFields)):
#     renameDict[rawFields[i]] = newFields[i]
# tce_tbl.rename(columns=renameDict, inplace=True)

# remove TCEs with zero period or transit duration
tce_tbl = tce_tbl.loc[(tce_tbl['tce_period'] > 0) & (tce_tbl['tce_duration'] > 0)]
print(len(tce_tbl))

# 15737 to 15737 TCEs
tce_tbl.to_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17_DR24/'
               'q1_q17_dr24_tce_2020.03.02_17.51.43_nounks_processed_stellarnorm.csv',
               index=False)

#%% Filter 'UNK' TCEs in Q1-Q17 DR24

tce_tbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17_DR24/'
                      'q1_q17_dr24_tce_2020.03.02_17.51.43_stellar.csv')

print(len(tce_tbl))

stellar_params = ['tce_sradius', 'tce_steff', 'tce_slogg', 'tce_smet', 'tce_smass', 'tce_sdens']

# remove TCEs with 'UNK' label
tce_tbl = tce_tbl.loc[tce_tbl['label'] != 'UNK']

print(len(tce_tbl))

# 20367 to 15737 TCEs
tce_tbl.to_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17_DR24/'
               'q1_q17_dr24_tce_2020.03.02_17.51.43_nounks.csv',
               index=False)

#%% Shuffle TCEs in Q1-Q17 DR24 TCE table

np.random.seed(123)  # same seed as Shallue & Vanderburg and Ansdell et al

tceTbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17_DR24/'
                     'q1_q17_dr24_tce_2020.03.02_17.51.43_nounks.csv')

tceTbl = tceTbl.iloc[np.random.permutation(len(tceTbl))]

tceTbl.to_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17_DR24/'
              'q1_q17_dr24_tce_2020.03.02_17.51.43_nounks_shuffled.csv', index=False)