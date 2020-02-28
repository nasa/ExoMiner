import pandas as pd
import msgpack
import numpy as np
from astropy.io import fits
from tensorflow import gfile
import os
import logging

# TODO: do the same for the targets of the 180k Kepler non-TCEs and for TESS
#%% Add Kepler Stellar parameters in the TCE table in the NASA Exoplanet Archive
# last time updated? Provenance all except star radius: KIC, J-K, Pinsonneault, Spectroscopy, Photometry,
# Transits or EBs, Asteroseismology, Solar (when unknown parameters)
# Provenance for stellar radius: DSEP, MULT

tce_tbl = pd.read_csv('/home/msaragoc/Downloads/q1_q17_dr25_tce_2019.03.12_updt_tcert.csv')
stellar_tbl = pd.read_csv('/home/msaragoc/Downloads/q1_q17_dr25_tce_2019.10.08_14.19.27.csv', header=17)

stellar_fields_out = ['sTEff', 'sLogG', 'sMet', 'sRad']
stellar_fields_in = ['tce_steff', 'tce_slogg', 'tce_smet', 'tce_sradius']

# drop Kepler ID duplicates
stellar_tbl.drop_duplicates('kepid', inplace=True)

for field in stellar_fields_out:
    tce_tbl[field] = np.nan

for i_tce in range(len(tce_tbl)):

    if i_tce % 500 == 0:
        print('Row {}/{}'.format(i_tce + 1, len(tce_tbl)))

    tce_tbl.loc[i_tce, stellar_fields_out] = \
        stellar_tbl.loc[stellar_tbl['kepid'] == tce_tbl.iloc[i_tce]['kepid']][stellar_fields_in].values[0]

tce_tbl.to_csv('/home/msaragoc/Downloads/keplerq1-q17_dr25_NEAstellar.csv', index=False)

kepid_tbl = tce_tbl['kepid']
kepid_tbl.drop_duplicates(inplace=True)

kepid_tbl.to_csv('/home/msaragoc/Downloads/keplerq1-q17_dr25_targets.csv', index=False)

#%% Add Kepler Stellar parameters in the KIC10 from http://archive.stsci.edu/kepler/kic10/search.php
# KIC last time updated: 2008

tce_tbl = pd.read_csv('/home/msaragoc/Downloads/q1_q17_dr25_tce_2019.03.12_updt_tcert.csv')
stellar_tbl = pd.read_csv('/home/msaragoc/Downloads/kic10_search10k.txt', header=0)
stellar_tbl2 = pd.read_csv('/home/msaragoc/Downloads/kic10_search10k-17k.txt', header=0)

stellar_fields_out = ['kMag', 'sTEff', 'sLogG', 'sMet', 'sRad']
stellar_fields_in = ['Kepler Mag', 'Teff (deg K)', 'Log G (cm/s/s)', 'Metallicity (solar=0.0)', 'Radius (solar=1.0)']

for field in stellar_fields_out:
    tce_tbl[field] = np.nan

for i_tce in range(len(tce_tbl)):

    if i_tce % 500 == 0:
        print('Row {}/{}'.format(i_tce + 1, len(tce_tbl)))

    match_kepid = stellar_tbl.loc[stellar_tbl['Kepler ID'] == tce_tbl.iloc[i_tce]['kepid']]
    if len(match_kepid) == 0:
        match_kepid = \
            stellar_tbl2.loc[stellar_tbl2['Kepler ID'] == tce_tbl.iloc[i_tce]['kepid']]

    if len(match_kepid) == 0:
        print('Kepler ID {} not found.'.format(tce_tbl.iloc[i_tce]['kepid']))
        continue

    tce_tbl.loc[i_tce, stellar_fields_out] = match_kepid[stellar_fields_in].values[0]

tce_tbl.to_csv('/home/msaragoc/Downloads/keplerq1-q17_dr25_KICstellar.csv', index=False)

#%% Add Kepler Stellar parameters in the Kepler Stellar Data in
# https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblSearch/nph-tblSearchInit?app=ExoTbls&config=keplerstellar
# Link to custom search used to generate the table
# https://exoplanetarchive.ipac.caltech.edu/workspace/2019.08.07_16.11.31_007922/TblSearch/2019.10.08_14.30.03_013902/bigResult.html

tce_tbl = pd.read_csv('/home/msaragoc/Downloads/q1_q17_dr25_tce_2019.03.12_updt_tcert.csv')
stellar_tbl = pd.read_csv('/home/msaragoc/Downloads/keplerstellar.csv')

stellar_fields_out = ['kMag', 'sTEff', 'sLogG', 'sMet', 'sRad', 'sMass', 'sDens']
stellar_fields_in = ['kepmag', 'teff', 'logg', 'feh', 'radius', 'mass', 'dens']

for field in stellar_fields_out:
    tce_tbl[field] = np.nan

for i_tce in range(len(tce_tbl)):

    if i_tce % 500 == 0:
        print('Row {}/{}'.format(i_tce + 1, len(tce_tbl)))

    tce_tbl.loc[i_tce, stellar_fields_out] = \
        stellar_tbl.loc[stellar_tbl['kepid'] == tce_tbl.iloc[i_tce]['kepid']][stellar_fields_in].values[0]

tce_tbl.to_csv('/home/msaragoc/Downloads/keplerq1-q17_dr25_Keplerstellar.csv', index=False)

#%% Add Kepler Q1-Q17 DR25 Stellar parameters in http://archive.stsci.edu/kepler/stellar17/search.php
# Provenance of stellar effective temperature, surface gravity and metallicity: KIC, J-K, Pinsonneault, Spectroscopy,
# Photometry, Transits or EBs, Asteroseismology, Solar (when unknown parameters)
# Secondary provenances (radius, mass, density): DSEP, MULT
# Dec 2016

tce_tbl = pd.read_csv('/home/msaragoc/Downloads/q1_q17_dr25_tce_2019.03.12_updt_tcert.csv')
stellar_tbl = pd.read_csv('/home/msaragoc/Downloads/kepler_stellar17_search10k.txt')
stellar_tbl2 = pd.read_csv('/home/msaragoc/Downloads/kepler_stellar17_search10k-17k.txt')

stellar_fields_out = ['kMag', 'sTEff', 'sLogG', 'sMet', 'sRad', 'sMass', 'sDens']
stellar_fields_in = ['kepmag', 'teff', 'logg', 'feh', 'radius', 'mass', 'dens']

for field in stellar_fields_out:
    tce_tbl[field] = np.nan

for i_tce in range(len(tce_tbl)):

    if i_tce % 500 == 0:
        print('Row {}/{}'.format(i_tce + 1, len(tce_tbl)))

    match_kepid = stellar_tbl.loc[stellar_tbl['kepid'] == tce_tbl.iloc[i_tce]['kepid']][stellar_fields_in]

    if len(match_kepid) == 0:
        match_kepid = stellar_tbl2.loc[stellar_tbl2['kepid'] == tce_tbl.iloc[i_tce]['kepid']][stellar_fields_in]

    if len(match_kepid) == 0:
        print('Kepler ID {} is missing.'.format(tce_tbl.iloc[i_tce]['kepid']))
        continue

    tce_tbl.loc[i_tce, stellar_fields_out] = match_kepid.values[0]

tce_tbl.to_csv('/home/msaragoc/Downloads/keplerq1-q17_dr25_Keplerstellar17.csv', index=False)

#%% Add Kepler Stellar Table Q1-Q17 DR25 and Supplemental Stellar Table Q1-Q17 DR25 in
# https://exoplanetarchive.ipac.caltech.edu/docs/program_interfaces.html#koi
# Q1-17 DR25 Supplemental Stellar, includes values provided by the Kepler Stellar Properties Working Group (SPWG)
# independent of any pipeline processing in order to report their most current stellar values
# It contains corrected metallicities and the resulting derived stellar parameters for 779 stars, which were incorrect
# in the original DR25 table

# 347 targets had to be extracted from the original DR25 table

tce_tbl = pd.read_csv('/home/msaragoc/Downloads/q1_q17_dr25_tce_2019.03.12_updt_tcert.csv')
stellar_tbl = pd.read_csv('/home/msaragoc/Downloads/q1_q17_dr25_stellar.csv')
stellar_tbl_supp = pd.read_csv('/home/msaragoc/Downloads/q1_q17_dr25_supp_stellar.csv')

stellar_fields_out = ['kMag', 'sTEff', 'sLogG', 'sMet', 'sRad', 'sMass', 'sDens']
stellar_fields_in = ['kepmag', 'teff', 'logg', 'feh', 'radius', 'mass', 'dens']

for field in stellar_fields_out:
    tce_tbl[field] = np.nan

count = 0
for i_tce in range(len(tce_tbl)):

    if i_tce % 500 == 0:
        print('Row {}/{}'.format(i_tce + 1, len(tce_tbl)))

    # first check the supplemental table
    match_kepid = stellar_tbl_supp.loc[stellar_tbl_supp['kepid'] == tce_tbl.iloc[i_tce]['kepid']][stellar_fields_in]

    if len(match_kepid) == 0:
        count += 1
        match_kepid = stellar_tbl.loc[stellar_tbl['kepid'] == tce_tbl.iloc[i_tce]['kepid']][stellar_fields_in]

    if len(match_kepid) == 0:
        print('Kepler ID {} is missing.'.format(tce_tbl.iloc[i_tce]['kepid']))
        continue

    tce_tbl.loc[i_tce, stellar_fields_out] = match_kepid.values[0]

tce_tbl.to_csv('/home/msaragoc/Downloads/keplerq1-q17_dr25_q1q17dr25supp.csv', index=False)

#%% Update effective temperature and stellar radius using Gaia DR2
# 15100 targets were updated

# tce_tbl = pd.read_csv('/home/msaragoc/Downloads/keplerq1-q17_dr25_q1q17dr25supp.csv')
kepid_tbl = pd.read_csv('/home/msaragoc/Downloads/q1_q17_dr25_supp_stellar.csv')
kepid_tbl2 = pd.read_csv('/home/msaragoc/Downloads/q1_q17_dr25_stellar.csv')

kepid_tbl = pd.concat([kepid_tbl, kepid_tbl2.loc[~kepid_tbl2['kepid'].isin(kepid_tbl['kepid'])]])

gaia_fp = '/data5/tess_project/Data/Ephemeris_tables/Stellar parameters/Kepler/gaia_kepler_crossref_dict_huber.msgp'  # 177911 targets

with open(gaia_fp, 'rb') as data_file:
    crossref_dict = msgpack.unpack(data_file)

count = 0
for i_kepid, kepid in enumerate(crossref_dict):

    if i_kepid % 500 == 0:
        print('Row {}/{}'.format(i_kepid + 1, len(crossref_dict)))

    # match_kepid = tce_tbl['kepid'] == kepid
    match_kepid = kepid_tbl['kepid'] == kepid

    if np.any(match_kepid):
        count += 1

    # tce_tbl.loc[match_kepid, ['sTEff', 'sRad']] = [int(crossref_dict[kepid][b'teff']),
    #                                                float(crossref_dict[kepid][b'radius'])]
    kepid_tbl.loc[match_kepid, ['teff', 'radius']] = [int(crossref_dict[kepid][b'teff']),
                                                   float(crossref_dict[kepid][b'radius'])]

# tce_tbl.to_csv('/home/msaragoc/Downloads/keplerq1-q17_dr25_q1q17dr25supp_gaiadr2.csv', index=False)
kepid_tbl.to_csv('/home/msaragoc/Downloads/q1_q17_dr25_stellar_gaiadr2.csv')

#%% Update Kepler TCE table (34k) with the latest stellar parameters

stellar_tbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Stellar parameters/Kepler/'
                          'q1_q17_dr25_stellar_gaiadr2.csv')
tce_tbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/'
                      'q1_q17_dr25_tce_2019.03.12_updt_tcert_extended.csv')

stellar_fields_out = ['kepmag', 'tce_steff', 'tce_steff_err1', 'tce_steff_err2', 'tce_slogg', 'tce_slogg_err1',
                      'tce_slogg_err2', 'tce_smet', 'tce_smet_err1', 'tce_smet_err2', 'tce_sradius', 'tce_sradius_err1',
                      'tce_sradius_err2', 'tce_smass', 'tce_smass_err1', 'tce_smass_err2', 'tce_sdens',
                      'tce_sdens_err1', 'tce_dens_serr2', 'ra', 'dec']
stellar_fields_in = ['kepmag', 'teff', 'teff_err1', 'teff_err2', 'logg', 'logg_err1', 'logg_err2', 'feh', 'feh_err1',
                     'feh_err2', 'radius', 'radius_err1', 'radius_err2', 'mass', 'mass_err1', 'mass_err2', 'dens',
                     'dens_err1', 'dens_err2', 'ra', 'dec']

tce_tbl_cols = list(tce_tbl.columns)

for stellar_param in stellar_fields_out:
    if stellar_param not in tce_tbl_cols:
        tce_tbl[stellar_param] = np.nan

count = 0
for row_star_i, row_star in stellar_tbl.iterrows():

    if row_star_i % 100 == 0:
        print('Star {} out of {} ({} %)\n Number of TCEs updated: {}'.format(row_star_i,
                                                                             len(stellar_tbl),
                                                                             row_star_i / len(stellar_tbl) * 100,
                                                                             count))
    target_cond = tce_tbl['kepid'] == row_star['kepid']

    count += target_cond.sum()

    tce_tbl.loc[target_cond, stellar_fields_out] = row_star[stellar_fields_in].values

print('Number of TCEs updated: {}'.format(count))
tce_tbl.to_csv('/home/msaragoc/Downloads/newtcetbl_test.csv', index=False)

#%% Update Kepler non-TCE table (~180k) with the latest stellar parameters

tce_tbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/180k_tce.csv')

stellar_tbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Stellar parameters/Kepler/'
                          'q1_q17_dr25_stellar_gaiadr2.csv')

stellar_fields_out = ['kepmag', 'tce_steff', 'tce_steff_err1', 'tce_steff_err2', 'tce_slogg', 'tce_slogg_err1',
                      'tce_slogg_err2', 'tce_smet', 'tce_smet_err1', 'tce_smet_err2', 'tce_sradius', 'tce_sradius_err1',
                      'tce_sradius_err2', 'tce_smass', 'tce_smass_err1', 'tce_smass_err2', 'tce_sdens',
                      'tce_sdens_err1', 'tce_dens_serr2', 'ra', 'dec']
stellar_fields_in = ['kepmag', 'teff', 'teff_err1', 'teff_err2', 'logg', 'logg_err1', 'logg_err2', 'feh', 'feh_err1',
                     'feh_err2', 'radius', 'radius_err1', 'radius_err2', 'mass', 'mass_err1', 'mass_err2', 'dens',
                     'dens_err1', 'dens_err2', 'ra', 'dec']

tce_tbl_cols = list(tce_tbl.columns)

for stellar_param in stellar_fields_out:
    if stellar_param not in tce_tbl_cols:
        tce_tbl[stellar_param] = np.nan

count = 0
for row_star_i, row_star in stellar_tbl.iterrows():

    if row_star_i % 100 == 0:
        print('Star {} out of {} ({} %)\n Number of TCEs updated: {}'.format(row_star_i,
                                                                             len(stellar_tbl),
                                                                             row_star_i / len(stellar_tbl) * 100,
                                                                             count))
    target_cond = tce_tbl['kepid'] == row_star['kepid']

    count += target_cond.sum()

    tce_tbl.loc[target_cond, stellar_fields_out] = row_star[stellar_fields_in].values

print('Number of TCEs updated: {}'.format(count))
tce_tbl.to_csv('/home/msaragoc/Downloads/newtcetbl180k_test.csv', index=False)

#%% Create TESS target list based on which targets were observed in sectors 1-19

# list of targets
ticIdsVisited2 = []

# observation sectors
sectors = np.arange(1, 20)

# FITS base directory
fitsDir = '/data5/tess_project/Data/TESS_TOI_fits(MAST)'

# in_fields = ['RADIUS', '']
# out_fields = ['ticid', 'tce_sradius', 'tce_steff', 'tce_slogg', 'tce_smet']
# stellar_dict = {field: [] for field in out_fields}
for sector in sectors:

    # sectorTicIds = []
    # fitsFilepaths = []
    fitsFilenames = os.listdir(os.path.join(fitsDir, 'sector_{}'.format(sector)))
    for i, fitsFilename in enumerate(fitsFilenames):
        print('Sector {} - {} % complete | Number of TIC IDs: {}'.format(sector, i / len(fitsFilenames) * 100,
                                                                         len(ticIdsVisited2)))
        if 'tesscurl' in fitsFilename:
            continue
        sectorTicId = int(fitsFilename.split('-')[2])
        if sectorTicId not in ticIdsVisited2:
            ticIdsVisited2.append(sectorTicId)
        # sectorTicIds.append(int(fitsFilename.split('-')[2]))
        # fitsFilepaths.append(os.path.join(fitsDir, os.path.join(sector, fitsFilename)))

    # for i in range(len(sectorTicIds)):
    #     if sectorTicIds[i] not in ticIdsVisited:
            with fits.open(gfile.Open(os.path.join(fitsDir, os.path.join('sector_{}'.format(sector), fitsFilename)), "rb")) \
                    as hdu_list:
                ticIdsVisited2.append(hdu_list['PRIMARY'].header['TICID'])
    #         ticIdsVisited.append(ticId)

# 182939 targets
np.save('/data5/tess_project/Data/Ephemeris_tables/Stellar_parameters/TESS/final_target_list_s1-s19.npy', ticIdsVisited)

#%% Create TESS stellar parameters table based on info from the FITS files and using the CTL list for sectors 1-19

headerList = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Stellar parameters/TESS/tic_column_description.txt',
                         skiprows=1, header=None, sep='\t', usecols=[1]).values.ravel()

chunksize = 1000
# 9488282 targets in the CTL
dfCtlReader = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Stellar_parameters/TESS/CTL/'
                          'exo_CTL_08.01xTIC_v8.csv', chunksize=chunksize, header=None, names=headerList)

dfTicList = None
i = 1
for chunk in dfCtlReader:
    print('Reading TICs {}-{}'.format(i, i + len(chunk) - 1))

    validTics = chunk.loc[chunk['[ID] [bigint]'].isin(ticIdsVisited)]

    if dfTicList is None:
        dfTicList = validTics
    else:
        dfTicList = pd.concat([dfTicList, validTics])

    i += len(chunk)
    print('Number of valid TICs added: {}\nTotal: {}'.format(len(validTics), len(dfTicList)))

# 144318 targets
dfTicList.to_csv('/data5/tess_project/Data/Ephemeris_tables/Stellar_parameters/TESS/aaaa.csv')

# ####################
# for ticId in ticIdsVisited:
#
#     numMatches = len(dfTicList.loc[dfTicList['[ID] [bigint]'] == ticId])
#     print(numMatches)
#     if numMatches == 0:
#         aaaa
#
# for sector in sectors:
#     fitsFilepaths = []
#     for fitsFilename in os.listdir(os.path.join(fitsDir, 'sector_{}'.format(sector))):
#         if 'tesscurl' in fitsFilename:
#             continue
#         sectorTicId = int(fitsFilename.split('-')[2])
#         if sectorTicId == 1078:  # 234302530:
#             aaaaaa
#
# dfCtlReader2 = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Stellar parameters/TESS/CTL/'
#                            'exo_CTL_08.01xTIC_v8.csv', chunksize=15210, header=None, names=headerList)
#
# j = 1
# for chunk2 in dfCtlReader2:
#     print('Reading TICs {}-{}'.format(j, j + len(chunk2) - 1))
#
#     numMatches = len(chunk2.loc[chunk2['[ID] [bigint]'] == 234302530])
#     if numMatches > 0:
#         aaaaa
#
#     j += len(chunk2)

#%% Create TESS stellar parameters table based on info from the FITS files and using the TIC-8 list for sectors 1-19

logging.basicConfig(filename='/data5/tess_project/Data/Ephemeris_tables/Stellar parameters/TESS/tic8.log',
                    filemode='a', format='%(message)s', level=logging.INFO)

# column descriptions
headerList = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Stellar parameters/TESS/tic_column_description.txt',
                         skiprows=1, header=None, sep='\t', usecols=[1]).values.ravel()

# root directory with TIC-8 csv files
ticDir = '/data5/tess_project/Data/Ephemeris_tables/Stellar parameters/TESS/TIC-8'
ticQuadTbls = [os.path.join(ticDir, ticQuadTbl) for ticQuadTbl in os.listdir(ticDir)]

# targets observed in sectors 1-19; look for parameters for these
ticsFits = np.load('/data5/tess_project/Data/Ephemeris_tables/Stellar parameters/TESS/final_target_list_s1-s19.npy')

# define number of rows to read from a table in one iteration
chunksize = 10000

# initialize target table variable
dfTicList = None

# iterate through TIC-8 csv files
for tbl_i, ticQuadTbl in enumerate(ticQuadTbls):

    # create an iterator that reads chunksize rows each time
    dfCtlReader = pd.read_csv(ticQuadTbl, chunksize=chunksize, header=None, names=headerList)

    i = 1
    for chunk in dfCtlReader:

        # print('Reading TICs {}-{} from table {} ({} %)'.format(i, i + len(chunk) - 1, ticQuadTbl.split('/')[-1],
        #                                                        tbl_i / len(ticQuadTbls) * 100))
        logging.info('Reading TICs {}-{} from table {} ({} %)'.format(i, i + len(chunk) - 1, ticQuadTbl.split('/')[-1],
                                                                      tbl_i / len(ticQuadTbls) * 100))

        validTics = chunk.loc[chunk['[ID] [bigint]'].isin(ticsFits)]

        if dfTicList is None:
            dfTicList = validTics
        else:
            dfTicList = pd.concat([dfTicList, validTics])

        i += len(chunk)
        # print('Number of valid TICs added: {}\nTotal: {}'.format(len(validTics), len(dfTicList)))
        logging.info('Number of valid TICs added: {}\nTotal: {}'.format(len(validTics), len(dfTicList)))

# 182939 targets
dfTicList.to_csv('/data5/tess_project/Data/Ephemeris_tables/Stellar parameters/TESS/final_target_list_s1-s19-tic8.csv',
                 index=False)

#%% Update TESS TCE table with stellar parameters with standardized column names

tce_tbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/TESS/....csv')

stellar_tbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Stellar parameters/TESS/'
                          'final_target_list_s1-s19-tic8.csv')

stellar_fields_out = ['tessmag', 'tce_steff', 'tce_steff_err', 'tce_slogg', 'tce_slogg_err',
                      'tce_smet', 'tce_smet_err', 'tce_sradius', 'tce_sradius_err',
                      'tce_smass', 'tce_smass_err', 'tce_sdens', 'tce_sdens_err', 'ra', 'dec']
stellar_fields_in = ['[Tmag] [real]', '[e_Tmag] [real]', '[Teff][real]', '[e_Teff][real]', '[logg][real]',
                     '[e_logg][real]', '[MH][real]', '[e_MH][real]', '[rad][real]', '[e_rad][real]', '[mass][real]',
                     '[e_mass][real]', '[rho][real]', '[e_rho][real]', '[ra] [float]', '[dec] [float]']

tce_tbl_cols = list(tce_tbl.columns)

for stellar_param in stellar_fields_out:
    if stellar_param not in tce_tbl_cols:
        tce_tbl[stellar_param] = np.nan

count = 0
for row_star_i, row_star in stellar_tbl.iterrows():

    if row_star_i % 100 == 0:
        print('Star {} out of {} ({} %)\n Number of TCEs updated: {}'.format(row_star_i,
                                                                             len(stellar_tbl),
                                                                             row_star_i / len(stellar_tbl) * 100,
                                                                             count))
    target_cond = tce_tbl['kepid'] == row_star['kepid']

    count += target_cond.sum()

    tce_tbl.loc[target_cond, stellar_fields_out] = row_star[stellar_fields_in].values

print('Number of TCEs updated: {}'.format(count))
tce_tbl.to_csv('/data5/tess_project/Data/Ephemeris_tables/TESS/final_tce_tables/....csv', index=False)
