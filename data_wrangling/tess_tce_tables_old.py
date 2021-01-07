# 3rd party
import pandas as pd
from pathlib import Path
import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt

# local
from data_wrangling.utils_ephemeris_matching import create_binary_time_series, find_nearest_epoch_to_this_time

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


#%% Prepare TOI catalog; get stellar parameters

toi_dir = Path('/data5/tess_project/Data/Ephemeris_tables/TESS/TOI_catalogs/12-4-2020')
toi_tbl = pd.read_csv(toi_dir / 'tois.csv', header=4)
print(f'Number of TOIs in total: {len(toi_tbl)}')

# remove QLP TOIs
toi_tbl = toi_tbl.loc[toi_tbl['Source Pipeline'] == 'spoc']
print(f'Number of SPOC TOIs: {len(toi_tbl)}')

# remove TOIs that do not have (or have, but invalid) orbital period, transit duration, transit depth
for col in ['Epoch Value', 'Orbital Period Value', 'Transit Duration Value', 'Transit Depth Value']:
    invalid_tois = (toi_tbl[col].isna() | (toi_tbl[col] < 0))
    print(f'Number of invalid TOIs for {col}: {invalid_tois.sum()}')
    toi_tbl = toi_tbl.loc[~invalid_tois]

print(f'Number of TOIs after removing invalid ones: {len(toi_tbl)}')
print(f'Number of TOIs per disposition:\n{toi_tbl["TOI Disposition"].value_counts()}')

# add stellar parameter values from the stellar table
tic_tbl = pd.concat([pd.read_csv(file)
                     for file in toi_dir.parent.iterdir() if 'stellartic8_tois_12' in file.stem] +
                    [pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/TESS/TOI_catalogs/8-14-2020/'
                                 'stellartic8_tois_8-14-2020.csv')], axis=0)

assert len(tic_tbl) == len(np.unique(tic_tbl['[ID] [bigint]']))
tic_tbl_fp = Path(toi_dir / f'tic8_tois.csv')
tic_tbl.to_csv(tic_tbl_fp, index=False)
# tic_tbl = pd.read_csv(tic_tbl_fp)

tics_in_toi_tbl = np.unique(toi_tbl['TIC'])
assert len(np.intersect1d(tics_in_toi_tbl, tic_tbl['[ID] [bigint]'])) == len(tics_in_toi_tbl)
print(f'Number of TICs: {tics_in_toi_tbl}')

# add stellar parameters from TIC table to the TOI table
stellar_fields_out = ['mag', 'mag_err', 'tce_steff', 'tce_steff_err', 'tce_slogg', 'tce_slogg_err',
                      'tce_smet', 'tce_smet_err', 'tce_sradius', 'tce_sradius_err',
                      'tce_smass', 'tce_smass_err', 'tce_sdens', 'tce_sdens_err', 'ra', 'dec']
stellar_fields_in = ['[Tmag] [real]', '[e_Tmag] [real]', '[Teff] [real]', '[e_Teff] [real]', '[logg] [real]',
                     '[e_logg] [real]', '[MH] [real]', '[e_MH] [real]', '[rad] [real]', '[e_rad] [real]',
                     '[mass] [real]', '[e_mass] [real]', '[rho] [real]', '[e_rho] [real]', '[ra] [float]',
                     '[dec] [float]']
for stellar_param in stellar_fields_out:
    toi_tbl[stellar_param] = np.nan
stellarColsTceTbl = ['ra', 'dec', 'tce_steff', 'tce_slogg', 'tce_sradius', 'tce_smass', 'tce_smet', 'tce_sdens']
for toi_i, toi in toi_tbl.iterrows():
    # find target stellar parameters in stellar table
    target_params = tic_tbl.loc[tic_tbl['[ID] [bigint]'] == toi.TIC][stellar_fields_in].values
    if len(target_params) > 0:  # if it found, update TOI table
        # only update parameters which are not NaN - do not overwrite the others with NaNs
        idxs_nnan = np.where(~np.isnan(target_params[0]))[0]
        if len(idxs_nnan) > 0:  # if there are not NaN parameters, update TCE table
            stellar_fields_out_aux = np.array(stellar_fields_out)[idxs_nnan]
            toi_tbl.loc[toi_i, stellar_fields_out_aux] = target_params[0][idxs_nnan]

print(toi_tbl[stellarColsTceTbl].isna().sum())

# check if there are missing values coming from the stellar table, and if there are, replace by the value already
# present in the TOI table
stellar_fields_in_out = {'TMag Value': 'mag',
                         'TMag Uncertainty': 'mag_err',
                         'Surface Gravity Value': 'tce_slogg',
                         'Surface Gravity Uncertainty': 'tce_slogg_err',
                         'Star Radius Value': 'tce_sradius',
                         'Star Radius Error': 'tce_sradius_err',
                         'Effective Temperature Value': 'tce_steff',
                         'Effective Temperature Uncertainty': 'tce_steff_err'
                         }
toi_count = 0
for toi_i, toi in toi_tbl.iterrows():
    params_out = np.array(toi[list(stellar_fields_in_out.values())].to_numpy(), dtype='float')
    idxs_nan = np.where(np.isnan(params_out))[0]
    if len(idxs_nan) > 0:
        stellar_params_in_aux = np.array(list(stellar_fields_in_out.keys()))[idxs_nan]
        stellar_params_out_aux = np.array(list(stellar_fields_in_out.values()))[idxs_nan]
        toi_tbl.loc[toi_i, stellar_params_out_aux] = toi[stellar_params_in_aux].values
        toi_count += 1

print(f'Number of TOIs updated: {toi_count} (out of {len(toi_tbl)})')
print(toi_tbl[stellarColsTceTbl].isna().sum())

# replace missing stellar parameters by solar parameters
solar_params = {'tce_steff': 5777.0, 'tce_slogg': 2.992, 'tce_smet': 0, 'tce_sradius': 1.0, 'tce_sdens': 1.408}
print(f'Before replacing missing values by solar parameters:\n{toi_tbl[list(solar_params.keys())].isna().sum()}')
for solar_param in solar_params:
    miss_stellarparam_tois = (toi_tbl[solar_param].isna())
    toi_tbl.loc[toi_tbl[solar_param].isna(), solar_param] = solar_params[solar_param]

print(f'After replacing missing values by solar parameters:\n{toi_tbl[list(solar_params.keys())].isna().sum()}')
toi_tbl.to_csv(toi_dir / f'tois_stellar.csv', index=False)

# remove TOIs without any sector
toi_tbl = toi_tbl.loc[~toi_tbl['Sectors'].isna()]
toi_tbl.to_csv(toi_dir / f'tois_stellar_nosectornan.csv', index=False)

#%% Matching between TOIs and TCEs

toi_dir = Path('/data5/tess_project/Data/Ephemeris_tables/TESS/TOI_catalogs/12-4-2020')

toi_tbl = pd.read_csv(toi_dir / f'tois_stellar_nosectornan.csv')

tce_root_dir = Path('/data5/tess_project/Data/Ephemeris_tables/TESS/DV_ephemeris')

multisector_tce_dir = tce_root_dir / 'multi-sector runs'
singlesector_tce_dir = tce_root_dir / 'single-sector runs'

multisector_tce_tbls = {(int(file.stem.split('-')[1][1:]), int(file.stem.split('-')[2][1:5])): pd.read_csv(file,
                                                                                                           header=6)
                        for file in multisector_tce_dir.iterdir() if 'tcestats' in file.name}
singlesector_tce_tbls = {int(file.stem.split('-')[1][1:]): pd.read_csv(file, header=6)
                         for file in singlesector_tce_dir.iterdir() if 'tcestats' in file.name}
singlesector_tce_tbls[21].drop_duplicates(subset='tceid', inplace=True, ignore_index=True)

match_thr = np.inf  # 0.25
sampling_interval = 0.00001  # approximately 1 min

# toi_tce_match = {}
# cols = [
    # 'orbitalPeriodDays',
    # 'transitDepthPpm',
    # 'transitDurationHours',
    # 'transitEpochBtjd',
    # 'ws_mes',
    # 'ws_mesphase',
    # 'mes'
# ]
max_num_tces = 30
matching_tbl = pd.DataFrame(columns=['Full TOI ID', 'TIC', 'Matched TCEs'] + [f'matching_dist_{i}'
                                                                              for i in range(max_num_tces)])
for toi_i, toi in toi_tbl.iterrows():

    if toi_i % 50 == 0:
        print(f'Matched {toi_i} ouf of {len(toi_tbl)}...')

    # if toi['Full TOI ID'] == 2411.01:
    #     aaaa

    # toi_tce_match[toi['Full TOI ID']] = {}

    toi_tceid = '{}-{}'.format(f'{toi["TIC"]}'.zfill(11), f'{toi["Signal ID"]}'.zfill(2))

    toi_sectors = [int(sector) for sector in toi['Sectors'].split(' ')]

    toi_bin_ts = create_binary_time_series(epoch=toi['Epoch Value'],
                                           duration=toi['Transit Duration Value'] / 24,
                                           period=toi['Orbital Period Value'],
                                           tStart=toi['Epoch Value'],
                                           tEnd=toi['Epoch Value'] + toi['Orbital Period Value'],
                                           samplingInterval=sampling_interval)

    matching_dist_dict = {}

    for toi_sector in toi_sectors:

        # check the single sector run table
        tce_tbl_aux = singlesector_tce_tbls[toi_sector]
        tce_found = tce_tbl_aux.loc[tce_tbl_aux['tceid'] == toi_tceid]

        if len(tce_found) > 0:
            epoch_aux = find_nearest_epoch_to_this_time(tce_found['transitEpochBtjd'].values[0],
                                                        tce_found['orbitalPeriodDays'].values[0],
                                                        toi['Epoch Value'])
            # epoch_shift = np.abs(p1[2] - p2[2]) / p1[1]
            # e2 = np.abs(round(epoch_shift) - epoch_shift) * p1[1]
            tce_bin_ts = create_binary_time_series(epoch=epoch_aux,
                                                   duration=tce_found['transitDurationHours'].values[0] / 24,
                                                   period=tce_found['orbitalPeriodDays'].values[0],
                                                   tStart=toi['Epoch Value'],
                                                   tEnd=toi['Epoch Value'] + toi['Orbital Period Value'],
                                                   samplingInterval=sampling_interval)

            match_distance = distance.cosine(toi_bin_ts, tce_bin_ts)

            if match_distance < match_thr:
                # toi_tce_match[toi['Full TOI ID']][toi_sector] = tce_found
                matching_dist_dict[f'{toi_sector}'] = match_distance

        # check the multi sector runs tables
        for multisector_tce_tbl in multisector_tce_tbls:
            if toi_sector >= multisector_tce_tbl[0] and toi_sector <= multisector_tce_tbl[1]:
                tce_tbl_aux = multisector_tce_tbls[multisector_tce_tbl]
                tce_found = tce_tbl_aux.loc[tce_tbl_aux['tceid'] == toi_tceid]

                if len(tce_found) > 0:

                    epoch_aux = find_nearest_epoch_to_this_time(tce_found['transitEpochBtjd'].values[0],
                                                                tce_found['orbitalPeriodDays'].values[0],
                                                                toi['Epoch Value'])
                    # epoch_shift = np.abs(p1[2] - p2[2]) / p1[1]
                    # e2 = np.abs(round(epoch_shift) - epoch_shift) * p1[1]
                    tce_bin_ts = create_binary_time_series(epoch=epoch_aux,
                                                           duration=tce_found['transitDurationHours'].values[0] / 24,
                                                           period=tce_found['orbitalPeriodDays'].values[0],
                                                           tStart=toi['Epoch Value'],
                                                           tEnd=toi['Epoch Value'] + toi['Orbital Period Value'],
                                                           samplingInterval=sampling_interval)

                    match_distance = distance.cosine(toi_bin_ts, tce_bin_ts)

                    if match_distance < match_thr:
                        # toi_tce_match[toi['Full TOI ID']][(multisector_tce_tbl[0], multisector_tce_tbl[1])] = \
                        #     tce_found[cols]
                        matching_dist_dict[f'{multisector_tce_tbl[0]}-{multisector_tce_tbl[1]}'] = match_distance

    matching_dist_dict = {k: v for k, v in sorted(matching_dist_dict.items(), key=lambda x: x[1])}
    data_to_tbl = {'Full TOI ID': [toi['Full TOI ID']], 'TIC': [toi['TIC']],
                   'Matched TCEs': ' '.join(list(matching_dist_dict.keys()))}
    matching_dist_arr = list(matching_dist_dict.values())
    for i in range(max_num_tces):
        if i < len(matching_dist_arr):
            data_to_tbl[f'matching_dist_{i}'] = [matching_dist_arr[i]]
        else:
            data_to_tbl[f'matching_dist_{i}'] = [np.nan]
    matching_tbl = pd.concat([matching_tbl, pd.DataFrame(data=data_to_tbl)], axis=0)

matching_tbl.to_csv(f'/home/msaragoc/Projects/Kepler-TESS_exoplanet/Analysis/toi_tce_matching/'
                    f'tois_matchedtces_ephmerismatching_thr{match_thr}_samplint{sampling_interval}.csv', index=False)

#%% check maximum number of TCEs per TOI

num_tces_per_toi_tbl = pd.DataFrame(columns=['Full TOI ID', 'Number of TCEs'])
for toi in toi_tce_match:
    num_tces_per_toi_tbl = pd.concat([num_tces_per_toi_tbl,
                                      pd.DataFrame(data={'Full TOI ID': toi,
                                                         'Number of TCEs': [len(toi_tce_match[toi])]})])

bins_num_tces = np.arange(0, 31)
f, ax = plt.subplots(figsize=(10, 6))
ax.hist(num_tces_per_toi_tbl['Number of TCEs'], bins=bins_num_tces, edgecolor='k')
ax.set_ylabel('Count (Number of TOIs)')
ax.set_yscale('log')
ax.set_xlabel('Number of TCEs per TOI')
ax.set_xticks(bins_num_tces)
ax.set_xlim([bins_num_tces[0], bins_num_tces[-1]])
f.savefig('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Analysis/toi_tce_matching/hist_tces_per_toi.png')

#%% create table with secondary phase for matched TCEs to each TOI

max_num_tces = 21
ws_mesphase_tbl = pd.DataFrame(columns=['Full TOI ID'] + [f'ws_mesphase_{i}' for i in range(max_num_tces)])
for toi in toi_tce_match:
    data_to_tbl = {'Full TOI ID': toi}
    tces_for_toi_arr = list(toi_tce_match[toi].keys())
    num_tces_for_toi = len(toi_tce_match[toi])
    for i in range(max_num_tces):
        if i < num_tces_for_toi:
            data_to_tbl[f'ws_mesphase_{i}'] = toi_tce_match[toi][tces_for_toi_arr[i]]['ws_mesphase'].to_list()
        else:
            data_to_tbl[f'ws_mesphase_{i}'] = [np.nan]

    ws_mesphase_tbl = pd.concat([ws_mesphase_tbl, pd.DataFrame(data=data_to_tbl)], axis=0)

ws_mesphase_tbl.to_csv(f'/home/msaragoc/Projects/Kepler-TESS_exoplanet/Analysis/toi_tce_matching/'
                       f'tois_matchedtces_wsphase_ephmerismatching_thr{match_thr}_samplint{sampling_interval}.csv',
                       index=False)

#%% add secondary phase to the TOIs using the TCE that matched the TOI

tce_root_dir = Path('/data5/tess_project/Data/Ephemeris_tables/TESS/DV_ephemeris')

multisector_tce_dir = tce_root_dir / 'multi-sector runs'
singlesector_tce_dir = tce_root_dir / 'single-sector runs'

multisector_tce_tbls = {(int(file.stem.split('-')[1][1:]), int(file.stem.split('-')[2][1:5])): pd.read_csv(file,
                                                                                                           header=6)
                        for file in multisector_tce_dir.iterdir() if 'tcestats' in file.name}
singlesector_tce_tbls = {int(file.stem.split('-')[1][1:]): pd.read_csv(file, header=6)
                         for file in singlesector_tce_dir.iterdir() if 'tcestats' in file.name}
singlesector_tce_tbls[21].drop_duplicates(subset='tceid', inplace=True, ignore_index=True)

matching_tbl = pd.read_csv('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Analysis/toi_tce_matching/'
                           'tois_matchedtces_ephmerismatching_thrinf_samplint1e-05.csv')

toi_tbl_fp = Path('/data5/tess_project/Data/Ephemeris_tables/TESS/TOI_catalogs/12-4-2020/tois_stellar_nosectornan.csv')
toi_tbl = pd.read_csv(toi_tbl_fp)
print(f'Total number of TOIs: {len(toi_tbl)}')

print(toi_tbl['TOI Disposition'].value_counts())

# filter TOIs that were not matched with any TCE
toi_tbl = toi_tbl.loc[toi_tbl['Full TOI ID'].isin(matching_tbl.loc[~matching_tbl['Matched TCEs'].isna(),
                                                                   'Full TOI ID'].to_list())]
print(f'Number of TOIs after removing those not matched to any TCE: {len(toi_tbl)}')

toi_tbl['tce_maxmesd'] = np.nan

for toi_i, toi in toi_tbl.iterrows():

    toi_found = matching_tbl.loc[matching_tbl['Full TOI ID'] == toi['Full TOI ID']]

    if len(toi_found) > 0:

        toi_tceid = '{}-{}'.format(f'{toi["TIC"]}'.zfill(11), f'{toi["Signal ID"]}'.zfill(2))
        sector_str = toi_found['Matched TCEs'].values[0].split(' ')[0]

        if '-' in sector_str:
            sector_start, sector_end = sector_str.split('-')
            tce_tbl = multisector_tce_tbls[(int(sector_start), int(sector_end))]
            toi_tbl.loc[toi_i, 'tce_maxmesd'] = tce_tbl.loc[(tce_tbl['tceid'] == toi_tceid), 'ws_mesphase'].values
        else:
            tce_tbl = singlesector_tce_tbls[int(sector_str)]
            toi_tbl.loc[toi_i, 'tce_maxmesd'] = tce_tbl.loc[(tce_tbl['tceid'] == toi_tceid), 'ws_mesphase'].values

toi_tbl.to_csv(toi_tbl_fp.parent / f'{toi_tbl_fp.stem}_wsphase.csv', index=False)

print(toi_tbl['TOI Disposition'].value_counts())

#%% rename columns

toi_tbl_fp = Path('/data5/tess_project/Data/Ephemeris_tables/TESS/TOI_catalogs/12-4-2020/tois_stellar_nosectornan.csv')
toi_tbl = pd.read_csv(toi_tbl_fp)

rename_dict = {
    'TIC': 'target_id',
    'Full TOI ID': 'oi',
    'TOI Disposition': 'label',
    # 'TIC Right Ascension': 'ra',
    # 'TIC Declination': 'dec',
    # 'TMag Value': 'mag',
    # 'TMag Uncertainty': 'mag_err',
    'Epoch Value': 'tce_time0bk',
    'Epoch Error': 'tce_time0bk_err',
    'Orbital Period Value': 'tce_period',
    'Orbital Period Error': 'tce_period_err',
    'Transit Duration Value': 'tce_duration',
    'Transit Duration Error': 'tce_duration_err',
    'Transit Depth Value': 'transit_depth',
    'Transit Depth Error': 'tce_depth_err',
    'Sectors': 'sectors',
}

toi_tbl.rename(columns=rename_dict, inplace=True)
toi_tbl.to_csv(toi_tbl_fp.parent / f'{toi_tbl_fp.stem}_renamedcols.csv', index=False)
