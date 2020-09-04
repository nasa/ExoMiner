"""
Extract stellar parameters for TOIs from multiple sources. Sources: CTL, TIC-8
"""

import pandas as pd
import os
import numpy as np
import multiprocessing
from datetime import date

#%% Get TICs to be extracted from the major stellar catalogs (CTL, TIC8, ...)

toiTbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/TESS/TOI_catalogs/8-14-2020/tois.csv', header=4)
toiTargetsCurr = toiTbl['TIC'].unique()

toiTargetsPrev = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/TESS/TOI_catalogs/6-19-2020/tois.csv',
                             header=4)['TIC'].unique()

toiTargets = np.setdiff1d(toiTargetsCurr, toiTargetsPrev)

print('Number of new TICs to extract from the TESS stellar catalogs: {}'.format(len(toiTargets)))

#%% Read from CTL catalog to update sub CTL table


def get_tics_from_ctl(tics, tbli, ctlFp, headerListFp, chunksize, saveDir):
    """ Get stellar parameters from the CTL catalog for a set of TICs.

    :param tics: NumPy array, TICs to get stellar parameters
    :param tbli: int, ID of the process
    :param ctlFp: str, filepath to CTL catalog
    :param headerListFp: str, filepath for the file with the header for the CTL catalog
    :param chunksize: int, number of rows to read each time from the CTL catalog
    :param saveDir: str, saving directory
    :return:
    """

    # load the CTL list
    # 9488282 targets in the CTL
    headerList = pd.read_csv(headerListFp, skiprows=1, header=None, sep='\t', usecols=[1]).values.ravel()
    dfCtlReader = pd.read_csv(ctlFp, chunksize=chunksize, header=None, names=headerList)

    dfTicList = pd.DataFrame(columns=headerList)
    i = 1
    for chunk in dfCtlReader:
        print('Reading TICs {}-{}'.format(i, i + len(chunk) - 1))

        # check which TICs in this chunk are in the list
        validTics = chunk.loc[chunk['[ID] [bigint]'].isin(tics)]

        dfTicList = pd.concat([dfTicList, validTics])

        i += len(chunk)
        print('Number of valid TICs added: {}\nTotal added (Total): {} ({})'.format(len(validTics), len(dfTicList),
                                                                                    len(tics)))

        if len(dfTicList) == len(tics):
            break

    dfTicList.to_csv(os.path.join(saveDir, 'stellarctl_tois_{}.csv'.format(tbli)), index=False)

    # for columnName in columnNames:
    #     print(dfTicList[columnName].isna().value_counts())


# columnNames = ['[Tmag] [real]', '[Teff] [real]', '[logg] [real]', '[MH] [real]', '[rad] [real]', '[mass] [real]',
#                '[rho] [real]', '[ra] [float]', '[dec] [float]']

saveDir = '/data5/tess_project/Data/Ephemeris_tables/TESS/TOI_catalogs/'

ctlFp = '/data5/tess_project/Data/Ephemeris_tables/TESS/TIC_tables/CTL/exo_CTL_08.01xTIC_v8.csv'
headerListFp = '/data5/tess_project/Data/Ephemeris_tables/TESS/TIC_tables/tic_column_description.txt'

# define number of rows to be read from the dataframe at each step
chunksize = 10000

nProcesses = 15
pool = multiprocessing.Pool(processes=nProcesses)
ticsProcs = np.array_split(toiTargets, nProcesses)
jobs = [(tics, tics_i, ctlFp, headerListFp, chunksize, saveDir) for tics_i, tics in enumerate(ticsProcs)]
async_results = [pool.apply_async(get_tics_from_ctl, job) for job in jobs]
pool.close()

# Instead of pool.join(), async_result.get() to ensure any exceptions raised by the worker processes are raised here
for async_result in async_results:
    async_result.get()

#%% Merge previous stellar table with the new targets

saveDir = '/data5/tess_project/Data/Ephemeris_tables/TESS/TOI_catalogs/'

ctlTbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/TESS/TOI_catalogs/stellarctl_tois.csv')
print('Number of targets before update: {}'.format(len(ctlTbl)))

newTICsTblFps = [os.path.join(saveDir, file) for file in os.listdir(saveDir) if 'stellarctl_tois_' in file]

for newTICsTblFp in newTICsTblFps:

    newTICsTbl = pd.read_csv(newTICsTblFp)

    ctlTbl = pd.concat([ctlTbl, newTICsTbl])

ctlTbl.to_csv(os.path.join(saveDir, 'stellarctl_tois_{}.csv'.format(date.today().strftime("%m-%d-%Y"))), index=False)
print('Number of targets after update: {}'.format(len(ctlTbl)))

#%% Update sub TIC8 table


def get_tics_from_tic(tics, ticTbls, tbli, headerListFp, chunksize, saveDir):
    """ Get stellar parameters from TIC-8 catalog for a set of TICs.

    :param tics: NumPy array, TICs to get stellar parameters
    :param ticTbls: list, containing filepaths to the TIC-8 tables to search on
    :param tbli: int, ID of the process
    :param headerListFp: str, filepath for the file with the header for the CTL catalog
    :param chunksize: int, number of rows to read each time from the CTL catalog
    :param saveDir: str, saving directory
    :return:
    """
    print('Getting {} TICs from {} TIC-8 subtables and saving them to table {}'.format(len(tics), len(ticTbls), tbli))

    headerList = pd.read_csv(headerListFp, skiprows=1, header=None, sep='\t', usecols=[1]).values.ravel()
    dataTypes = {col: None for col in headerList}
    for col in headerList:
        if '[int]' in col or '[bigint]' in col:
            dataTypes[col] = np.float64
        elif 'real' in col:
            dataTypes[col] = np.float64
        else:
            dataTypes[col] = np.str

    dataTypes = {col: np.str if 'varchar' in col else np.float for col in headerList}
    # print(dataTypes)
    dfTicList = pd.DataFrame(columns=headerList)

    # iterate through TIC-8 csv files
    for ticQuadTbl_i, ticQuadTbl in enumerate(ticTbls):

        print('Reading TIC table {}({})'.format(ticQuadTbl_i, len(ticTbls)))
        print('Number of TICs added: {}({})'.format(len(dfTicList), len(tics)))

        dfTicReader = pd.read_csv(ticQuadTbl, chunksize=chunksize, header=None, names=headerList, dtype=dataTypes)

        i = 1
        for chunk in dfTicReader:

            # check which TICs in this chunk are in the list
            validTics = chunk.loc[chunk['[ID] [bigint]'].isin(tics)]

            dfTicList = pd.concat([dfTicList, validTics])

            i += len(chunk)

            if len(dfTicList) == len(tics):
                break

        if len(dfTicList) == len(tics):
            break

    print('Added {} TICs'.format(len(dfTicList)))

    dfTicList.to_csv(os.path.join(saveDir, 'stellartic8_tois_{}.csv'.format(tbli)), index=False)

    # for columnName in columnNames:
    #     print(dfTicList[columnName].isna().value_counts())


# columnNames = ['[Tmag] [real]', '[Teff] [real]', '[logg] [real]', '[MH] [real]', '[rad] [real]', '[mass] [real]',
#                '[rho] [real]', '[ra] [float]', '[dec] [float]']

saveDir = '/data5/tess_project/Data/Ephemeris_tables/TESS/TOI_catalogs/'

# root directory with TIC-8 csv files
ticDir = '/data5/tess_project/Data/Ephemeris_tables/TESS/TIC_tables/TIC-8'
ticQuadTbls = [os.path.join(ticDir, ticQuadTbl) for ticQuadTbl in os.listdir(ticDir)]

headerListFp = '/data5/tess_project/Data/Ephemeris_tables/TESS/TIC_tables/tic_column_description.txt'

# define number of rows to be read from the dataframe at each step
chunksize = 1000

nProcesses = 4
pool = multiprocessing.Pool(processes=nProcesses)
ticTblsProcs = np.array_split(ticQuadTbls, nProcesses)
jobs = [(toiTargets, ticTblsSub, tbl_i, headerListFp, chunksize, saveDir) for tbl_i, ticTblsSub in
        enumerate(ticTblsProcs)]
async_results = [pool.apply_async(get_tics_from_tic, job) for job in jobs]
pool.close()

# Instead of pool.join(), async_result.get() to ensure any exceptions raised by the worker processes are raised here
for async_result in async_results:
    async_result.get()

#%% Merge previous stellar table with the new targets

saveDir = '/data5/tess_project/Data/Ephemeris_tables/TESS/TOI_catalogs/'

tic8Tbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/TESS/TOI_catalogs/stellartic8_tois.csv')
print('Number of targets before update: {}'.format(len(tic8Tbl)))

newTICsTblFps = [os.path.join(saveDir, file) for file in os.listdir(saveDir) if 'stellartic8_tois_' in file]

for newTICsTblFp in newTICsTblFps:

    newTICsTbl = pd.read_csv(newTICsTblFp)

    tic8Tbl = pd.concat([tic8Tbl, newTICsTbl])

# 119 targets added
tic8Tbl.to_csv(os.path.join(saveDir, 'stellartic8_tois_{}.csv'.format(date.today().strftime("%m-%d-%Y"))), index=False)
print('Number of targets after update: {}'.format(len(tic8Tbl)))

#%% Analyze CTL TOI and TIC-8 TOI catalogs

tic8Tbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/TESS/TOI_catalogs/8-14-2020/'
                      'stellartic8_tois_8-14-2020.csv')
ctlTbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/TESS/TOI_catalogs/8-14-2020/'
                     'stellarctl_tois_8-14-2020.csv')

columnNames = ['[Tmag] [real]', '[Teff] [real]', '[logg] [real]', '[MH] [real]', '[rad] [real]', '[mass] [real]',
               '[rho] [real]', '[ra] [float]', '[dec] [float]']

# print(tic8Tbl[columnNames].isna().sum())
# the number of missing values is the same, which confirms that CTL is a subset of TIC-8
print(tic8Tbl.loc[tic8Tbl['[ID] [bigint]'].isin(ctlTbl['[ID] [bigint]'])][columnNames].isna().sum())
print(ctlTbl[columnNames].isna().sum())

# print(len(np.intersect1d(tic8Tbl['[ID] [bigint]'], ctlTbl['[ID] [bigint]'])))
# print(len(np.setdiff1d(tic8Tbl['[ID] [bigint]'], ctlTbl['[ID] [bigint]'])))
# print(len(np.setdiff1d(ctlTbl['[ID] [bigint]'], tic8Tbl['[ID] [bigint]'])))
