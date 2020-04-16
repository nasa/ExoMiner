"""
Analysis of the distribution of stellar parameters in Kepler and TESS.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

#%% Utility functions

colsKeplerStellarTbl = ['kepmag', 'teff', 'logg', 'feh', 'radius', 'mass', 'dens', 'ra', 'dec']
colsTessStellarTbl = ['[Tmag] [real]', '[Teff] [real]', '[logg] [real]', '[MH] [real]', '[rad] [real]', '[mass] [real]',
                      '[rho] [real]', '[ra] [float]', '[dec] [float]']


def get_stellar_params(targetList, stellarTbl, satellite):

    if satellite == 'kepler':
        columnsStellarTbl = colsKeplerStellarTbl
        target_identifier = 'kepid'
    else:
        columnsStellarTbl = colsTessStellarTbl
        target_identifier = '[ID] [bigint]'

    targetDf = pd.DataFrame(columns=columnsStellarTbl)

    targetDf['target_id'] = targetList

    for target_i, target in targetDf.iterrows():
        targetDf.loc[target_i, columnsStellarTbl] = stellarTbl.loc[stellarTbl[target_identifier] ==
                                                                   target.target_id][columnsStellarTbl].values[0]

    return targetDf


#%% Define results directory

resDir = '/home/msaragoc/Projects/Kepler-TESS_exoplanet/Kepler_planet_finder/stellar_parameters_analysis'

#%% load stellar tables for Kepler and TESS

keplerStellarTbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Stellar parameters/Kepler/'
                               'q1_q17_dr25_stellar_gaiadr2_nanstosolar.csv')
tessStellarTbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Stellar parameters/TESS/'
                             'final_target_list_s1-s20-tic8.csv')

#%% Get targets from Kepler light curve dataset used to come up with TCEs

keplerFitsRootDir = '/data5/tess_project/Data/Kepler-Q1-Q17-DR25/pdc-tce-time-series-fits/'

keplerFitsFiles = list(Path(keplerFitsRootDir).rglob('*.fits'))

# get Kepler IDs
keplerFits = [int(filePath.name.split('-')[0][4:]) for filePath in keplerFitsFiles]

keplerFitsTargets = np.unique(keplerFits)

print('Number of Kepler IDs found in the light curve dataset used to create TCEs: {}'.format(len(keplerFitsTargets)))

np.save(os.path.join(resDir, 'keplerFitsLc.npy'), keplerFitsTargets)

keplerDf = get_stellar_params(keplerFitsTargets, keplerStellarTbl, 'kepler')

keplerDf.to_csv(os.path.join(resDir, 'keplerFitsLc_stellarparams.csv'))

#%% Get targets from Kepler light curve dataset used to come up with TCEs+non-TCEs

# targets from Kepler light curve dataset used to come up with TCEs
keplerFitsRootDir = '/data5/tess_project/Data/Kepler-Q1-Q17-DR25/dr_25_all_final/'

keplerFitsFiles = list(Path(keplerFitsRootDir).rglob('*.fits'))

# get Kepler IDs
keplerFits = [int(filePath.name.split('-')[0][4:]) for filePath in keplerFitsFiles]

keplerFitsTargets = np.unique(keplerFits)

print('Number of Kepler IDs found in the light curve dataset used to create TCEs + non-TCEs: '
      '{}'.format(len(keplerFitsTargets)))

np.save(os.path.join(resDir, 'keplerFitsLcAll.npy'), keplerFitsTargets)

keplerDf = get_stellar_params(keplerFitsTargets, keplerStellarTbl, 'kepler')

keplerDf.to_csv(os.path.join(resDir, 'keplerFitsLcAll_stellarparams.csv'))

#%% Get targets from TESS light curve dataset used to come up with TCEs+non-TCEs

tessFitsTargets = np.load('')

print('Number of TESS IDs found in the light curve dataset used to create TCEs + non-TCEs: '
      '{}'.format(len(tessFitsTargets)))

tessDf = get_stellar_params(tessFitsTargets, keplerStellarTbl, 'tess')

tessDf.to_csv(os.path.join(resDir, 'tessFitsLcAll_stellarparams.csv'))

#%% Kepler targets from Q1-Q17 DR25 TCE list

rogues = '_noroguetces'  # either '' or '_noroguetces'

keplerTblTargets = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/'
                               'q1_q17_dr25_tce_cumkoi2020.02.21_stellar{}_shuffled.csv'.format(rogues),
                               usecols=['target_id'])

# keplerTblTargets = np.unique(keplerTblTargets.values.ravel())
keplerTblTargets = keplerTblTargets.values.ravel()

print('Number of Kepler IDs found in Q1-Q17 DR25 TCE list: {}'.format(len(keplerTblTargets)))

keplerDf = get_stellar_params(keplerTblTargets, keplerStellarTbl, 'kepler')

keplerDf.to_csv(os.path.join(resDir, 'keplerQ1-Q17DR25{}_stellarparams_alltcespertarget.csv'.format(rogues)))

#%% Kepler targets from Q1-Q17 DR24 TCE list

keplerTblTargets = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17_DR24/'
                               'q1_q17_dr24_tce_2020.03.02_17.51.43_stellar_shuffled.csv',
                               usecols=['target_id'])

# keplerTblTargets = np.unique(keplerTblTargets.values.ravel())
keplerTblTargets = keplerTblTargets.values.ravel()

print('Number of Kepler IDs found in Q1-Q17 DR24 TCE list: {}'.format(len(keplerTblTargets)))

keplerDf = get_stellar_params(keplerTblTargets, keplerStellarTbl, 'kepler')

keplerDf.to_csv(os.path.join(resDir, 'keplerQ1-Q17DR24_stellarparams_alltcespertarget.csv'))

#%% Kepler 180k non-TCEs TCE list

keplerTblTargets = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/180k non-TCEs/'
                               '180k_nontce_stellar_shuffled.csv', usecols=['target_id'])

# keplerTblTargets = np.unique(keplerTblTargets.values.ravel())
keplerTblTargets = keplerTblTargets.values.ravel()

print('Number of Kepler IDs found in 180k non-TCE list: {}'.format(len(keplerTblTargets)))

keplerDf = get_stellar_params(keplerTblTargets, keplerStellarTbl, 'kepler')

keplerDf.to_csv(os.path.join(resDir, 'kepler180nonTces_stellarparams_alltcespertarget.csv'))

#%% Load stellar parameters dataframes

dfStellarParamsFp = {'Kepler FITS': os.path.join(resDir, 'keplerFitsLc_stellarparams.csv'),
                     'Kepler FITS all': os.path.join(resDir, 'keplerFitsLcAll_stellarparams.csv'),
                     'Kepler Q1-Q17 DR25': os.path.join(resDir, 'keplerQ1-Q17DR25_stellarparams_alltcespertarget.csv'),
                     'Kepler Q1-Q17 DR24': os.path.join(resDir, 'keplerQ1-Q17DR24_stellarparams_alltcespertarget.csv'),
                     'Kepler 180k non-TCEs': os.path.join(resDir, 'kepler180nonTces_stellarparams_alltcespertarget.csv'),
                     # 'TESS FITS': '',
                     }

dfStellarParams = {dfStellarName: pd.read_csv('{}'.format(dfStellarParamsFp[dfStellarName])) for dfStellarName in
                   dfStellarParamsFp}

# standardize column names
stellar_fields_out = ['mag', 'tce_steff', 'tce_slogg', 'tce_smet', 'tce_sradius', 'tce_smass', 'tce_sdens', 'ra', 'dec']
stellar_fields_in = colsKeplerStellarTbl

# rename fields to standardize fieldnames
renameDict = {stellar_fields_in[i]: stellar_fields_out[i] for i in range(len(stellar_fields_in))}

for df in dfStellarParams:
    dfStellarParams[df].rename(columns=renameDict, inplace=True)

# Boxplots

stellar_params = {'tce_steff': 'Effective Temperature (K)',
                  'tce_slogg': 'Surface Gravity (log10[cm.s^-2])',
                  'tce_smet': 'Metallicity (dex)',
                  'tce_sradius': 'Radius (Solar radii)',
                  'tce_smass': 'Mass (Solar mass)',
                   'tce_sdens': 'Density (g/cm^3)'}
for stellar_param in stellar_params:

    x = [dfStellarParams[dfStellarName][stellar_param].values for dfStellarName in dfStellarParams]

    f, ax = plt.subplots(figsize=(12, 6))
    ax.boxplot(x, notch=True)
    ax.set_ylabel('Value')
    ax.set_title(stellar_params[stellar_param])
    ax.set_xticklabels(list(dfStellarParams.keys()))
    if stellar_param in ['tce_sradius', 'tce_sdens']:
        ax.set_yscale('log')
    f.savefig(os.path.join(resDir, 'boxplot_kepler_{}.png'.format(stellar_param)))
    plt.close()
