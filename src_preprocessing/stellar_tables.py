import pandas as pd
import msgpack
import numpy as np

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
