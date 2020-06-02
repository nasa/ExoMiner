import os
import pandas as pd
import numpy as np

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


