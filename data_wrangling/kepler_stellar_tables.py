import pandas as pd
import msgpack
import numpy as np
import os

#%%

headerStr = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/KIC_tables/q1_q17_dr25_stellar.csv',
                        header=204, sep=',', skiprows=[205, 206, 207], lineterminator='\n').columns.to_list()[0]
header = [el.strip() for el in headerStr.split('|')[1:-1]]
# header.insert(1, '2MASS')
stellarTbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/KIC_tables/q1_q17_dr25_stellar.csv',
                         header=None,
                         names=header,
                         sep=r'[ ]{2,}',  # '\s+',
                         # delim_whitespace=True,
                         skiprows=np.arange(208),
                         lineterminator='\n')

print('Number of Kepler IDs in Q1-Q17 DR25 Stellar: {}'.format(len(stellarTbl)))

stellarTbl.to_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/KIC_tables/q1_q17_dr25_stellarproc.csv',
                  index=False)

suppStellarTbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/KIC_tables/q1_q17_dr25_supp_stellar.csv',
                             header=None,
                             names=header,
                             sep=r'[ ]{2,}',  # '\s+',
                             # delim_whitespace=True,
                             skiprows=np.arange(208),
                             lineterminator='\n')

print('Number of Kepler IDs in Q1-Q17 DR25 Supplemental Stellar: {}'.format(len(suppStellarTbl)))

suppStellarTbl.to_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/KIC_tables/q1_q17_dr25_supp_stellarproc.csv',
                  index=False)

print('Number of Kepler IDs in Q1-Q17 DR25 Supplemental Stellar that are in '
      'Q1-Q17 DR25 Stellar: {}'.format(suppStellarTbl['kepid'].isin(stellarTbl['kepid']).sum()))

updtTargets = 0
for row_i, row in suppStellarTbl.iterrows():

    if row_i % 5000 == 0:
        print('Row {}/{}'.format(row_i + 1, len(suppStellarTbl)))

    if not (stellarTbl.loc[stellarTbl['kepid'] == row['kepid'], ['feh', 'feh_err1', 'feh_err2']] !=
            row[['feh', 'feh_err1', 'feh_err2']]).all(axis=1).values[0]:
        updtTargets += 1

    stellarTbl.loc[stellarTbl['kepid'] == row['kepid'], ['feh', 'feh_err1', 'feh_err2']] = \
        row[['feh', 'feh_err1', 'feh_err2']].values

print('Number of KeplerIDs updated: {}'.format(updtTargets))

stellarTbl.to_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/KIC_tables/q1_q17_dr25_stellar_plus_supp.csv',
                  index=False)

#%% Update effective temperature and stellar radius using Gaia DR2; Check "Revised Radii of Kepler Stars and Planets
# Using Gaia Data Release 2" by Berger et al.
# 15100 targets were updated (when?)

# tce_tbl = pd.read_csv('/home/msaragoc/Downloads/keplerq1-q17_dr25_q1q17dr25supp.csv')
# kepid_tbl = pd.read_csv('/home/msaragoc/Downloads/q1_q17_dr25_supp_stellar.csv')
# kepid_tbl2 = pd.read_csv('/home/msaragoc/Downloads/q1_q17_dr25_stellar.csv')

# kepid_tbl = pd.concat([kepid_tbl, kepid_tbl2.loc[~kepid_tbl2['kepid'].isin(kepid_tbl['kepid'])]])
kepid_tbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/KIC_tables/q1_q17_dr25_stellar_plus_supp.csv')

gaia_fp = '/data5/tess_project/Data/Ephemeris_tables/Kepler/KIC_tables/gaia_kepler_crossref_dict_huber.msgp'  # 177911 targets

with open(gaia_fp, 'rb') as data_file:
    crossref_dict = msgpack.unpack(data_file, strict_map_key=False)

print('Number of Kepler IDs in Gaia DR25: {}'.format(len(crossref_dict)))

count = 0
for i_kepid, kepid in enumerate(crossref_dict):

    if i_kepid % 5000 == 0:
        print('Row {}/{}'.format(i_kepid + 1, len(crossref_dict)))

    match_kepid = kepid_tbl['kepid'] == kepid

    assert match_kepid.sum() <= 1

    if match_kepid.sum() == 1:
        count += 1

    # kepid_tbl.loc[match_kepid, ['teff', 'radius']] = [int(crossref_dict[kepid][b'teff']),
    #                                                   float(crossref_dict[kepid][b'radius'])]
    kepid_tbl.loc[match_kepid, ['teff', 'radius']] = [int(crossref_dict[kepid]['teff']),
                                                      float(crossref_dict[kepid]['radius'])]

print('Number of Kepler IDs updated: {}'.format(count))

# tce_tbl.to_csv('/home/msaragoc/Downloads/keplerq1-q17_dr25_q1q17dr25supp_gaiadr2.csv', index=False)
# kepid_tbl.to_csv('/home/msaragoc/Downloads/q1_q17_dr25_stellar_gaiadr2.csv')
kepid_tbl.to_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/KIC_tables/'
                 'q1_q17_dr25_stellar_plus_supp_gaiadr2.csv', index=False)

#%% Deal with NaNs in Kepler stellar table

# keplerStellarTbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Stellar parameters/Kepler/'
#                                'q1_q17_dr25_stellar_gaiadr2.csv')
keplerStellarTbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/KIC_tables/'
                               'q1_q17_dr25_stellar_plus_supp_gaiadr2.csv')

colsKeplerStellarTbl = ['teff', 'logg', 'feh', 'radius', 'mass', 'dens', 'ra', 'dec']

# solar parameters
solarParamsKepler = {'mass': 1.0, 'dens': 1.408, 'logg': 2.992, 'feh': 0, 'radius': 1, 'teff': 5777}

# count number of NaNs for each stellar parameter
print('Number of missing values')
print(keplerStellarTbl[colsKeplerStellarTbl].isna().sum(axis=0))

# update NaNs using Solar parameters
for solarParam in solarParamsKepler:
    keplerStellarTbl.loc[keplerStellarTbl[solarParam].isna(), solarParam] = solarParamsKepler[solarParam]

# confirm after updating with Solar parameters
print(keplerStellarTbl[colsKeplerStellarTbl].isna().sum(axis=0))

keplerStellarTbl.to_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/KIC_tables/'
                        'q1_q17_dr25_stellar_plus_supp_gaiadr2_missingvaluestosolar.csv', index=False)

#%% Update Kepler Q1-Q!7 DR25 TCE table (~34k TCEs) with the latest stellar parameters

stellar_tbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/KIC_tables/'
                          'q1_q17_dr25_stellar_plus_supp_gaiadr2_missingvaluestosolar.csv')

tce_tbl_dir = '/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17_DR25/raw'
# tce_tbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/'
#                       'q1_q17_dr25_tce_2019.03.12_updt_tcert_extended.csv')
# tce_tbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/'
#                       'q1_q17_dr25_tce_cumkoi2020.02.21_numtcespertarget.csv')
# tce_tbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17 DR25/raw/'
#                       'q1_q17_dr25_tce_2020.04.15_23.19.10.csv', header=57)
tce_tbl_filename = 'q1_q17_dr25_tce_2020.09.28_10.36.22.csv'
tce_tbl_filepath = os.path.join(tce_tbl_dir, tce_tbl_filename)
tce_tbl = pd.read_csv(tce_tbl_filepath, header=83)

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

    if row_star_i % 5000 == 0:
        print('Star {} out of {} ({} %)\n Number of TCEs updated: {}'.format(row_star_i,
                                                                             len(stellar_tbl),
                                                                             row_star_i / len(stellar_tbl) * 100,
                                                                             count))

    target_cond = tce_tbl['kepid'] == row_star['kepid']

    tce_tbl.loc[target_cond, stellar_fields_out] = row_star[stellar_fields_in].values

    count += target_cond.sum()

    if count > len(tce_tbl):
        print('All TCEs were updated.')
        break

print('Number of TCEs updated: {}'.format(count))
# tce_tbl.to_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17 DR25/'
#                'q1_q17_dr25_tce_2020.04.15_23.19.10_stellar.csv', index=False)
tce_tbl.to_csv(os.path.join(tce_tbl_dir[:-4], tce_tbl_filename)[:-4] + '_stellar.csv', index=False)

#%% Update Kepler Q1-Q17 DR25 TPS table (~200k TCEs) with the latest stellar parameters

# tce_tbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/180k non-TCEs/180k_nontce.csv')
tce_tbl_filepath = '/data5/tess_project/Data/Ephemeris_tables/Kepler/TPS_tables/Q1-Q17_DR25/keplerTPS_KSOP2536.csv'
tce_tbl = pd.read_csv(tce_tbl_filepath)

# stellar_tbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Stellar parameters/Kepler/'
#                           'q1_q17_dr25_stellar_gaiadr2.csv')
stellar_tbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/KIC_tables/'
                          'q1_q17_dr25_stellar_gaiadr2_nanstosolar.csv')

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

    if row_star_i % 5000 == 0:
        print('Star {} out of {} ({} %)\n Number of TCEs updated: {}'.format(row_star_i,
                                                                             len(stellar_tbl),
                                                                             row_star_i / len(stellar_tbl) * 100,
                                                                             count))
    target_cond = tce_tbl['kepid'] == row_star['kepid']

    tce_tbl.loc[target_cond, stellar_fields_out] = row_star[stellar_fields_in].values

    count += target_cond.sum()

    if count >= len(tce_tbl):
        print('All TCEs were updated')
        break

print('Number of TCEs updated: {}'.format(count))
# tce_tbl.to_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/180k non-TCEs/180k_nontce_stellar.csv', index=False)
tce_tbl.to_csv(os.path.join(tce_tbl_filepath)[:-4] + '_stellar.csv', index=False)
