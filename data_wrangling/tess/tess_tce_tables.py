import os
import pandas as pd
import numpy as np
from pathlib import Path

#%% Merge TCEs from the single-sector DV lists

tceDvDir = '/data5/tess_project/Data/Ephemeris_tables/TESS/DV_ephemeris/single-sector runs/'

tceDvlistFilenames = [os.path.join(tceDvDir, filename) for filename in os.listdir(tceDvDir)
                      if filename.endswith('tcestats.csv')]

tceDf = None

for filename in tceDvlistFilenames:

    tceSectorDf = pd.read_csv(filename, header=6)
    tceSectorDf['sector'] = filename.split('-')[-3][1:].lstrip('0')

    if tceDf is None:
        tceDf = tceSectorDf
    else:
        tceDf = pd.concat([tceDf, tceSectorDf])

print('Total number of TCEs: {}'.format(len(tceDf)))
print('Number of TCEs per sector:')
print(tceDf['sector'].value_counts())

sectorSeries = tceDf['sector'].astype('int')
firstSector, lastSector = sectorSeries.min(), sectorSeries
# 28173 TCEs Sectors 1-23, 5-7-2020 12.04 pm
tceDf.to_csv(os.path.join(tceDvDir, 'tess-s{}-s{}_dvr-tcestats.csv'.format(str(sectorSeries.min()).zfill(4),
                                                                           str(sectorSeries.max()).zfill(4))),
             index=False)

#%% Merge TCEs from multi-sector analysis



#%% Filter out QLP TOIs from the EXOFOP list

exofopDf = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/TESS/EXOFOP_TOI_lists/TOI/raw/'
                       'exofop_toilists_5-7-2020.csv')

exofopDf = exofopDf.loc[exofopDf['Source'] != 'qlp']
print('Number of TOIs left after removing the QLP TOIs: {}'.format(len(exofopDf)))

# 925 TOIs, 5-7-2020 12.26 pm
exofopDf.to_csv('/data5/tess_project/Data/Ephemeris_tables/TESS/EXOFOP_TOI_lists/TOI/raw/'
                'exofop_toilists_5-7-2020_spoc.csv', index=False)

#%% Ephemeris matching between TCEs and EXOFOP SPOC TOIs; add TOI disposition to matched TCE

tceDf = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/TESS/DV_ephemeris/single-sector runs/'
                    'tess-s0001-s0023_dvr-tcestats.csv')
exofopDf = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/TESS/EXOFOP_TOI_lists/TOI/raw/'
                       'exofop_toilists_5-7-2020_spoc.csv')

#%% Merge TCEs from the single- and multi-sector DV lists

tceDvDir = '/data5/tess_project/Data/Ephemeris_tables/TESS/DV_ephemeris/'

tceDvDirSingleSector = os.path.join(tceDvDir, 'single-sector runs')
tceDvDirMultiSector = os.path.join(tceDvDir, 'multi-sector runs')

tceDvlistFilenamesSingleSector = [os.path.join(tceDvDirSingleSector, filename)
                                  for filename in os.listdir(tceDvDirSingleSector)
                                  if filename.endswith('tcestats.csv')]
tceDvlistFilenamesMultiSector = [os.path.join(tceDvDirMultiSector, filename)
                                 for filename in os.listdir(tceDvDirMultiSector)
                                 if filename.endswith('tcestats.csv')]

tceDvlistFilenames = tceDvlistFilenamesSingleSector + tceDvlistFilenamesMultiSector

tceDf = None
for filename in tceDvlistFilenames:

    tceSectorDf = pd.read_csv(filename, header=6)
    if 'single' in filename:
        tceSectorDf['sector'] = filename.split('-')[-3][1:].lstrip('0')
    else:
        tceSectorDf['sector'] = filename.split('-')[-3][1:].lstrip('0') + ',' + \
                                filename.split('-')[-2].split('_')[0][1:].lstrip('0')

    if tceDf is None:
        tceDf = tceSectorDf
    else:
        tceDf = pd.concat([tceDf, tceSectorDf], sort=False)

print('Total number of TCEs: {}'.format(len(tceDf)))
print('Number of TCEs per sector:')
print(tceDf['sector'].value_counts())

# 49412 TCEs Sectors 1-24 multi and single-sector runs, 6-25-2020 12.00 pm
tceDf.to_csv(os.path.join(tceDvDir, 'tess-s1-s24_m-sruns_dvr-tcestats.csv'), index=False)

#%% Update TOI catalog with stellar parameters

toiTbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/TESS/TOI_catalogs/8-14-2020/tois.csv', header=4)
toiTblTargets = toiTbl.drop_duplicates(subset=['TIC'])
columnNames = ['TMag Value', 'Surface Gravity Value', 'Star Radius Value', 'Effective Temperature Value']
for columnName in columnNames:
    print(toiTblTargets[columnName].isna().value_counts())

stellarTbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/TESS/TOI_catalogs/8-14-2020/'
                         'stellartic8_tois_8-14-2020.csv')

columnNamesStellar = ['[Tmag] [real]', '[Teff] [real]', '[logg] [real]', '[MH] [real]', '[rad] [real]', '[mass] [real]',
                      '[rho] [real]', '[ra] [float]', '[dec] [float]']
# columnNamesToi = ['mag', 'tce_steff', 'tce_slogg', 'tce_smet', 'tce_sradius', 'tce_smass', 'tce_sdens', 'ra', 'dec']
columnNamesToi = ['TMag Value', 'Effective Temperature Value', 'Surface Gravity Value', '[MH] [real]',
                  'Star Radius Value', '[mass] [real]', '[rho] [real]', 'TIC Right Ascension', 'TIC Declination']

# for columnName in columnNamesToi:
#     toiTbl[columnName] = np.nan

# create columns for stellar parameters not present in the TOI catalog
for columnName in ['[MH] [real]', '[mass] [real]', '[rho] [real]']:
    toiTbl[columnName] = np.nan

# toisUpdated = 0
# for target_i, target in stellarTbl.iterrows():
#     matchedTois = toiTbl['TIC'] == target['[ID] [bigint]']
#     toiTbl.loc[matchedTois, columnNamesToi] = target[columnNamesStellar].values
#     toisUpdated += matchedTois.sum()
#
# assert toisUpdated == len(toiTbl)

for toi_i, toi in toiTbl.iterrows():

    ticMatched = stellarTbl.loc[stellarTbl['[ID] [bigint]'] == toi['TIC'], columnNamesStellar]
    for param_i in range(len(columnNamesToi)):
        # only update missing values
        if np.isnan(toi[columnNamesToi[param_i]]) and not np.isnan(ticMatched[columnNamesStellar[param_i]].values[0]):
            toiTbl.loc[toi_i, [columnNamesToi[param_i]]] = ticMatched[columnNamesStellar[param_i]].values[0]

toiTbl.to_csv('/data5/tess_project/Data/Ephemeris_tables/TESS/TOI_catalogs/8-14-2020/tois_stellar.csv', index=False)
toiTblTargets = toiTbl.drop_duplicates(subset=['TIC'])
for columnName in columnNamesToi:
    print(toiTblTargets[columnName].isna().value_counts())

#%% Replace missing values by solar parameters

# solar parameters
solarParams = {'Effective Temperature Value': 5777.0, 'Surface Gravity Value': 4.438, '[MH] [real]': 0,
               'Star Radius Value': 1.0, '[rho] [real]': 1.408, '[mass] [real]': 1.0}

toiTbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/TESS/TOI_catalogs/8-14-2020/tois_stellar.csv')

paramsUpdated = 0
for toi_i, toi in toiTbl.iterrows():
    for param in solarParams:
        if np.isnan(toi[param]):
            toiTbl.loc[toi_i, [param]] = solarParams[param]
            paramsUpdated += 1

toiTbl.to_csv('/data5/tess_project/Data/Ephemeris_tables/TESS/TOI_catalogs/8-14-2020/tois_stellar_missvaltosolar.csv',
              index=False)
toiTblTargets = toiTbl.drop_duplicates(subset=['TIC'])
for columnName in columnNamesToi:
    print(toiTblTargets[columnName].isna().value_counts())

#%% Standardize TCE table columns

toiTbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/TESS/TOI_catalogs/8-14-2020/'
                     'tois_stellar_missvaltosolar.csv')
print('Number of TOIs: {}'.format(len(toiTbl)))

rawFields = ['TIC', 'Full TOI ID', 'TOI Disposition', 'TIC Right Ascension', 'TIC Declination', 'TMag Value',
             'Orbital Epoch Value', 'Orbital Epoch Error', 'Orbital Period Value', 'Orbital Period Error',
             'Transit Duration Value', 'Transit Duration Error', 'Transit Depth Value', 'Transit Depth Error',
             'Sectors', 'Surface Gravity Value', 'Star Radius Value', 'Effective Temperature Value', '[MH] [real]',
             '[mass] [real]', '[rho] [real]']
newFields = ['target_id', 'oi', 'label', 'ra', 'dec', 'mag', 'tce_time0bk', 'tce_time0bk_err',
             'tce_period', 'tce_period_err', 'tce_duration', 'tce_duration_err', 'transit_depth', 'transit_depth_err',
             'sectors', 'tce_slogg', 'tce_sradius', 'tce_steff', 'tce_smet', 'tce_smass', 'tce_sdens']

# remove TCEs with any NaN in the required fields
toiTbl.dropna(axis=0, subset=np.array(rawFields)[[0, 1, 3, 4, 6, 8, 10, 12, 14]], inplace=True)
print('Number of TOIs after dropping TOIs with NaNs: {}'.format(len(toiTbl)))

# rename fields to standardize fieldnames
renameDict = {}
for i in range(len(rawFields)):
    print(rawFields[i], newFields[i])
    renameDict[rawFields[i]] = newFields[i]
toiTbl.rename(columns=renameDict, inplace=True)

# remove TCEs with zero period or transit duration
toiTbl = toiTbl.loc[(toiTbl['tce_period'] > 0) & (toiTbl['tce_duration'] > 0) & (toiTbl['transit_depth'])]
print('Number of TOIs after dropping TOIs with nonpositive period, transit duration or '
      'transit depth: {}'.format(len(toiTbl)))

toiTbl.to_csv('/data5/tess_project/Data/Ephemeris_tables/TESS/TOI_catalogs/8-14-2020/'
              'tois_stellar_missvaltosolar_stdcols.csv', index=False)

#%% Filter out QLP TOIs

toiTbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/TESS/TOI_catalogs/8-14-2020/'
                     'tois_stellar_missvaltosolar_stdcols.csv')

toiTbl = toiTbl.loc[toiTbl['Source Pipeline'] == 'spoc']
print('Number of SPOC TOIs: {}'.format(len(toiTbl)))

toiTbl.to_csv('/data5/tess_project/Data/Ephemeris_tables/TESS/TOI_catalogs/8-14-2020/'
              'tois_stellar_missvaltosolar_stdcols_spoc.csv',
              index=False)

#%%

tceTbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/TESS/DV_ephemeris/tess-s1-s24_m-sruns_dvr-tcestats.csv')

columnNamesStellar = ['starTeffKelvin', 'starLoggCgs', 'starRadiusSolarRadii']

tceTblTargets = tceTbl.drop_duplicates(subset=['ticid'])
for columnName in columnNamesStellar:
    print(tceTblTargets[columnName].isna().value_counts())

#%% Get number of TCEs per run

tce_root_dir = Path('/data5/tess_project/Data/Ephemeris_tables/TESS/DV_ephemeris')

multisector_tce_dir = tce_root_dir / 'multi-sector runs'
singlesector_tce_dir = tce_root_dir / 'single-sector runs'

multisector_tce_tbls = {(int(file.stem.split('-')[1][1:]), int(file.stem.split('-')[2][1:5])): pd.read_csv(file,
                                                                                                           header=6)
                        for file in multisector_tce_dir.iterdir() if 'tcestats' in file.name}
singlesector_tce_tbls = {int(file.stem.split('-')[1][1:]): pd.read_csv(file, header=6)
                         for file in singlesector_tce_dir.iterdir() if 'tcestats' in file.name}
singlesector_tce_tbls[21].drop_duplicates(subset='tceid', inplace=True, ignore_index=True)

# order keys by sector
singlesector_tce_tbls = {key: singlesector_tce_tbls[key] for key in sorted(singlesector_tce_tbls.keys())}
multisector_tce_tbls = {key: multisector_tce_tbls[key] for key in sorted(multisector_tce_tbls.keys())}

sector_tce_tbls = dict(singlesector_tce_tbls)
sector_tce_tbls.update(multisector_tce_tbls)

num_tces = {'sector(s)': [], 'number of tces': []}
for run in sector_tce_tbls:
    num_tces['sector(s)'].append(run)
    num_tces['number of tces'].append(len(sector_tce_tbls[run]))

num_tces_tbl = pd.DataFrame(data=num_tces)
num_tces_tbl.to_csv(tce_root_dir / 'num_tces_1-13-2021.csv', index=False)
