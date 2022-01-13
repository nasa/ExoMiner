"""
- Get most up-to-date stellar parameters for KIC targets.
- Update TCE table stellar parameters.
"""

# 3rd party
import multiprocessing
import os
from pathlib import Path
import msgpack
import numpy as np
import pandas as pd

# %% Combine data from Q1-Q17 DR25 stellar and supplemental stellar tables


headerStr = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/KIC_tables/q1_q17_dr25_stellar.csv',
                        header=204, sep=',', skiprows=[205, 206, 207], lineterminator='\n').columns.to_list()[0]
header = [el.strip() for el in headerStr.split('|')[1:-1]]
# header.insert(1, '2MASS')
# load Q1-Q17 DR25 Stellar catalog which was used in the final Q1-Q17 DR25 TCE run
# 200,038 KIC stars
# check "Revised Stellar Properties of Kepler Targets for the Q1-17 (DR25) Transit Detection Run", Mathur et al.
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

# load Q1-Q17 DR25 Supplemental Stellar catalog
# updated eff. temp., logg, feh and derived parameters (mass, radius and density) for 197,096 KIC stars
# check "ERRATUM: Revised Stellar Properties of Kepler Targets for the Q1-17 (DR25) Transit Detection Run",
# Mathur et al.
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


def _update_q1q17dr25_stellar_with_suppl(sub_tbl_i, stellar_tbl, supp_stellar_tbl, cols_to_update):
    """ Update Kepler Q1-Q17 DR25 Stellar table with values from the Kepler Q1-Q17 DR25 Supplemental Stellar table.

    :param sub_tbl_i: int, Kepler Q1-Q17 DR25 Stellar sub-table id
    :param stellar_tbl: pandas DataFrame, Kepler Q1-Q17 DR25 Stellar sub-table
    :param supp_stellar_tbl: pandas DataFrame, Kepler Q1-Q17 DR25 Supplemental Stellar table
    :param cols_to_update: list, columns to update in the Kepler Q1-Q17 DR25 Stellar table with values from the Kepler
    Q1-Q17 DR25 Supplemental Stellar table
    :return:
        pandas DataFrame, updated Kepler Q1-Q17 DR25 Stellar sub-table
    """

    for row_i, row in supp_stellar_tbl.iterrows():

        if row_i % 5000 == 0:
            print(f'[Subtable {sub_tbl_i}] KIC {row_i + 1}/{len(supp_stellar_tbl)}')

        kic_found = stellar_tbl['kepid'] == row['kepid']

        if kic_found.sum() == 1:
            stellar_tbl.loc[kic_found, cols_to_update] = row[cols_to_update].values

    return stellar_tbl


# update Q1-Q17 DR25 Stellar with updated parameters from Q1-Q17 DR25 Supplemental Stellar
# updtTargets = 0
cols_to_update = ['kepid', 'tm_designation', 'teff', 'teff_err1', 'teff_err2', 'logg',
                  'logg_err1', 'logg_err2', 'feh', 'feh_err1', 'feh_err2', 'mass',
                  'mass_err1', 'mass_err2', 'radius', 'radius_err1', 'radius_err2',
                  'dens', 'dens_err1', 'dens_err2', 'prov_sec', 'kepmag',
                  'dist', 'dist_err1', 'dist_err2', 'nconfp', 'nkoi',
                  'ntce', 'st_delivname', 'st_vet_date_str', 'ra', 'dec',
                  'teff_prov', 'logg_prov', 'feh_prov', 'jmag', 'jmag_err',
                  'hmag', 'hmag_err', 'kmag', 'kmag_err', 'av',
                  'av_err1', 'av_err2']

n_processes = 15
stellar_tbl_split = np.array_split(stellarTbl, n_processes, axis=0)
pool = multiprocessing.Pool(processes=n_processes)
jobs = [(stellar_tbl_i, stellar_sub_tbl, suppStellarTbl, cols_to_update)
        for stellar_tbl_i, stellar_sub_tbl in enumerate(stellar_tbl_split)]
async_results = [pool.apply_async(_update_q1q17dr25_stellar_with_suppl, job) for job in jobs]
pool.close()

stellarTbl = pd.concat([async_result.get() for async_result in async_results], axis=0, ignore_index=True)

# for row_i, row in suppStellarTbl.iterrows():
#
#     if row_i % 5000 == 0:
#         print('Row {}/{}'.format(row_i + 1, len(suppStellarTbl)))
#
#     # if not (stellarTbl.loc[stellarTbl['kepid'] == row['kepid'], ['feh', 'feh_err1', 'feh_err2']] !=
#     #         row[['feh', 'feh_err1', 'feh_err2']]).all(axis=1).values[0]:
#     #     updtTargets += 1
#
#     stellarTbl.loc[stellarTbl['kepid'] == row['kepid'], cols_to_update] = row[cols_to_update].values
#
# # print('Number of KeplerIDs updated: {}'.format(updtTargets))

stellarTbl.to_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/KIC_tables/q1_q17_dr25_stellar_plus_supp.csv',
                  index=False)


# %% Update Kepler stellar table using data from Gaia DR2 2018 and 2020


def _update_stellar_tbl(sub_tbl_i, stellar_tbl, updt_stellar_tbl, cols_to_update, cols_to_get, col_ids, col_updt_tbl):
    """ Update a stellar table with values from another table.

    :param sub_tbl_i: int, stellar sub-table id
    :param stellar_tbl: pandas DataFrame, stellar table
    :param updt_stellar_tbl: pandas DataFrame, table to get the new values from
    :param cols_to_update: list, columns to update in the stellar table
    :param cols_to_get: list, columns to get from the update table
    :param col_ids: tuple, ids of the columns in the stellar and update table used to match KICs
    :param col_updt_tbl: str, column to add to the stellar table to identify a match
    :return:
        pandas DataFrame, updated stellar table
        int, number of KICs updated
    """

    count = 0

    for row_i, row in stellar_tbl.iterrows():

        if row_i % 5000 == 0:
            print(f'[Subtable {sub_tbl_i}] KIC {row_i + 1}/{len(stellar_tbl)}')

        kic_found = updt_stellar_tbl.loc[updt_stellar_tbl[col_ids[1]] == row[col_ids[0]]]

        if len(kic_found) == 1:
            count += 1
            stellar_tbl.loc[row_i, cols_to_update] = kic_found[cols_to_get].values[0]
            stellar_tbl.loc[row_i, col_updt_tbl] = 'yes'

    return stellar_tbl, count


kepid_tbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/KIC_tables/q1_q17_dr25_stellar_plus_supp.csv')

gaia_dr2_2018_tbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/KIC_tables/gaia_dr_tbls/'
                                'gaia_dr2_2018/DR2PapTable1.txt',
                                sep='&', lineterminator='\\', skip_blank_lines=True)
for kic_i, kic in gaia_dr2_2018_tbl.iterrows():
    if isinstance(kic['KIC'], str):
        try:
            gaia_dr2_2018_tbl.loc[kic_i, 'KIC'] = int(kic['KIC'][1:])
        except:
            print(kic)
gaia_dr2_2018_tbl = gaia_dr2_2018_tbl[:-1]
gaia_dr2_2018_tbl.to_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/KIC_tables/gaia_dr_tbls/'
                         'gaia_dr2_2018/DR2PapTable1.csv', index=False)

print(f'Number of Kepler IDs in Gaia DR25: {len(gaia_dr2_2018_tbl)}')

kepid_tbl['gaia_dr2_2018'] = 'no'

cols_to_update = ['teff', 'teff_err1', 'teff_err2', 'radius', 'radius_err1', 'radius_err2', 'dist', 'dist_err1',
                  'dist_err2']
cols_to_get = ['teff', 'teffe', 'teffe', 'rad', 'radep', 'radem', 'dis', 'disep', 'disem']

# count = 0
# for i_kepid, kepid in gaia_dr2_2018_tbl.iterrows():
#
#     if i_kepid % 5000 == 0:
#         print(f'KIC {i_kepid + 1}/{len(gaia_dr2_2018_tbl)}')
#
#     match_kepid = kepid_tbl['kepid'] == kepid['KIC']
#
#     assert match_kepid.sum() <= 1
#
#     if match_kepid.sum() == 1:
#         count += 1
#
#         kepid_tbl.loc[match_kepid, cols_to_update] = kepid[cols_to_get].values
#         # kepid_tbl.loc[match_kepid, 'teff_prov'] = 'gaia_dr2_2018'
#         kepid_tbl.loc[match_kepid, 'gaia_dr2_2018'] = 'yes'

n_processes = 15
stellar_tbl_split = np.array_split(kepid_tbl, n_processes, axis=0)
pool = multiprocessing.Pool(processes=n_processes)
jobs = [(stellar_tbl_i, stellar_sub_tbl, gaia_dr2_2018_tbl, cols_to_update, cols_to_get, ('kepid', 'KIC'),
         'gaia_dr2_2018')
        for stellar_tbl_i, stellar_sub_tbl in enumerate(stellar_tbl_split)]
async_results = [pool.apply_async(_update_stellar_tbl, job) for job in jobs]
pool.close()

mp_res = [async_result.get() for async_result in async_results]
kepid_tbl = pd.concat([el[0] for el in mp_res], axis=0, ignore_index=True)
count = sum([el[1] for el in mp_res])

print('Number of Kepler IDs updated: {}'.format(count))

kepid_tbl.to_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/KIC_tables/'
                 'q1_q17_dr25_stellar_plus_supp_gaiadr2_2018.csv', index=False)

gaia_dr2_2020_tbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/KIC_tables/gaia_dr_tbls/'
                                'gaia_dr2_2020/GKSPCPapTable2_Final.txt',
                                sep='&', lineterminator='\\')
for kic_i, kic in gaia_dr2_2020_tbl.iterrows():
    if isinstance(kic['KIC'], str):
        try:
            gaia_dr2_2020_tbl.loc[kic_i, 'KIC'] = int(kic['KIC'][1:])
        except:
            print(kic)
gaia_dr2_2020_tbl = gaia_dr2_2020_tbl[:-1]
gaia_dr2_2020_tbl.to_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/KIC_tables/gaia_dr_tbls/'
                         'gaia_dr2_2020/GKSPCPapTable2_Final.csv', index=False)

print(f'Number of Kepler IDs in Gaia DR25: {len(gaia_dr2_2020_tbl)}')

kepid_tbl['gaia_dr2_2020'] = 'no'

cols_to_update = ['teff', 'teff_err1', 'teff_err2', 'radius', 'radius_err1', 'radius_err2', 'dist', 'dist_err1',
                  'dist_err2', 'mass', 'mass_err1', 'mass_err2', 'logg', 'logg_err1', 'logg_err2',
                  'feh', 'feh_err1', 'feh_err2', 'dens', 'dens_err1', 'dens_err2']
cols_to_get = ['iso_teff', 'iso_teff_err1', 'iso_teff_err2', 'iso_rad', 'iso_rad_err1', 'iso_rad_err2', 'iso_dis',
               'iso_dis_err1', 'iso_dis_err2', 'iso_mass', 'iso_mass_err1', 'iso_mass_err2', 'iso_logg',
               'iso_logg_err1', 'iso_logg_err2', 'iso_feh', 'iso_feh_err1', 'iso_feh_err2', 'iso_rho', 'iso_rho_err1',
               'iso_rho_err2']

# count = 0
# for i_kepid, kepid in gaia_dr2_2018_tbl.iterrows():
#
#     if i_kepid % 5000 == 0:
#         print(f'KIC {i_kepid + 1}/{len(gaia_dr2_2018_tbl)}')
#
#     match_kepid = kepid_tbl['kepid'] == kepid['KIC']
#
#     assert match_kepid.sum() <= 1
#
#     if match_kepid.sum() == 1:
#         count += 1
#
#         kepid_tbl.loc[match_kepid, cols_to_update] = kepid[cols_to_get].values
#         # kepid_tbl.loc[match_kepid, 'teff_prov'] = 'gaia_dr2_2018'
#         kepid_tbl.loc[match_kepid, 'gaia_dr2_2020'] = 'yes'

n_processes = 15
stellar_tbl_split = np.array_split(kepid_tbl, n_processes, axis=0)
pool = multiprocessing.Pool(processes=n_processes)
jobs = [(stellar_tbl_i, stellar_sub_tbl, gaia_dr2_2020_tbl, cols_to_update, cols_to_get, ('kepid', 'KIC'),
         'gaia_dr2_2020')
        for stellar_tbl_i, stellar_sub_tbl in enumerate(stellar_tbl_split)]
async_results = [pool.apply_async(_update_stellar_tbl, job) for job in jobs]
pool.close()

mp_res = [async_result.get() for async_result in async_results]
kepid_tbl = pd.concat([el[0] for el in mp_res], axis=0, ignore_index=True)
count = sum([el[1] for el in mp_res])

print('Number of Kepler IDs updated: {}'.format(count))

kepid_tbl.to_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/KIC_tables/'
                 'q1_q17_dr25_stellar_plus_supp_gaiadr2_2018_gaiadr2_2020.csv', index=False)


# %% Replace parameters set to Solar parameters by NaN

def _update_solar_to_nan(stellar_tbl, sub_tbl_i=None):
    """ Change parameters that were set to Solar parameters to NaN.

    :param stellar_tbl: pandas DataFrame, original stellar table
    :param sub_tbl_i: int, table id
    :return:
        pandas DataFrame, stellar table with missing values set to NaN
        dict, count missing parameters changed
    """

    count_missing = {stellar_param: 0 for stellar_param in ['teff', 'logg', 'feh', 'radius', 'mass', 'dens']}

    for kic_i, kic in stellar_tbl.iterrows():

        if kic_i % 5000 == 0:
            print(f'[Sub-table {sub_tbl_i}] KIC {kic_i + 1}/{len(stellar_tbl)}')

        if kic['teff_prov'] == 'Solar' and kic['gaia_dr2_2018'] == 'no' and kic['gaia_dr2_2020'] == 'no':
            stellar_tbl.loc[kic_i, 'teff'] = np.nan
            count_missing['teff'] += 1

        if kic['logg_prov'] == 'Solar' and kic['gaia_dr2_2020'] == 'no':
            stellar_tbl.loc[kic_i, 'logg'] = np.nan
            count_missing['logg'] += 1

        if kic['feh_prov'] == 'Solar' and kic['gaia_dr2_2020'] == 'no':
            stellar_tbl.loc[kic_i, 'feh'] = np.nan
            count_missing['feh'] += 1

        if kic['prov_sec'] == 'Solar' and kic['gaia_dr2_2018'] == 'no' and kic['gaia_dr2_2020'] == 'no':
            stellar_tbl.loc[kic_i, 'radius'] = np.nan
            count_missing['radius'] += 1

        if kic['prov_sec'] == 'Solar' and kic['gaia_dr2_2020'] == 'no':
            stellar_tbl.loc[kic_i, 'mass'] = np.nan
            count_missing['mass'] += 1

        if kic['prov_sec'] == 'Solar' and kic['gaia_dr2_2020'] == 'no':
            stellar_tbl.loc[kic_i, 'dens'] = np.nan
            count_missing['dens'] += 1

    return stellar_tbl, count_missing


kepid_tbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/KIC_tables/'
                        'q1_q17_dr25_stellar_plus_supp_gaiadr2_2018_gaiadr2_2020.csv')

# count_missing = {stellar_param: 0 for stellar_param in ['teff', 'logg', 'feh', 'radius', 'mass', 'dens']}
# for kic_i, kic in kepid_tbl.iterrows():
#
#     if kic_i % 5000 == 0:
#         print(f'KIC {kic_i + 1}/{len(kepid_tbl)}')
#
#     if kic['teff_prov'] == 'Solar' and kic['gaia_dr2_2018'] == 'no' and kic['gaia_dr2_2020'] == 'no':
#         kic['teff'] = np.nan
#         count_missing['teff'] += 1
#
#     if kic['logg_prov'] == 'Solar' and kic['gaia_dr2_2020'] == 'no':
#         kic['logg'] = np.nan
#         count_missing['logg'] += 1
#
#     if kic['feh_prov'] == 'Solar' and kic['gaia_dr2_2020'] == 'no':
#         kic['feh'] = np.nan
#         count_missing['feh'] += 1
#
#     if kic['prov_sec'] == 'Solar' and kic['gaia_dr2_2018'] == 'no' and kic['gaia_dr2_2020'] == 'no':
#         kic['radius'] = np.nan
#         count_missing['radius'] += 1
#
#     if kic['prov_sec'] == 'Solar' and kic['gaia_dr2_2020'] == 'no':
#         kic['mass'] = np.nan
#         count_missing['mass'] += 1
#
#     if kic['prov_sec'] == 'Solar' and kic['gaia_dr2_2020'] == 'no':
#         kic['dens'] = np.nan
#         count_missing['dens'] += 1

n_processes = 15
stellar_tbl_split = np.array_split(kepid_tbl, n_processes, axis=0)
pool = multiprocessing.Pool(processes=n_processes)
jobs = [(stellar_sub_tbl.reset_index(), stellar_tbl_i)
        for stellar_tbl_i, stellar_sub_tbl in enumerate(stellar_tbl_split)]
async_results = [pool.apply_async(_update_solar_to_nan, job) for job in jobs]
pool.close()

mp_res = [async_result.get() for async_result in async_results]

kepid_tbl = pd.concat([el[0] for el in mp_res], axis=0, ignore_index=True)

count_missing = {stellar_param: 0 for stellar_param in ['teff', 'logg', 'feh', 'radius', 'mass', 'dens']}
for el in mp_res:
    count_missing = {key: val + el[1][key] for key, val in count_missing.items()}

print(f'Missing values set to Solar parameters that were set to NaN: {count_missing}')

kepid_tbl.to_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/KIC_tables/'
                 'q1_q17_dr25_stellar_plus_supp_gaiadr2_2018_gaiadr2_2020_missing_solar_to_nan.csv', index=False)

# %% Update effective temperature and radius using Gaia DR2

kepid_tbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/KIC_tables/q1_q17_dr25_stellar_plus_supp.csv')

# load Gaia-Kepler cross-reference
# 177,911 targets
# check "Revised Radii of Kepler Stars and Planets using Gaia Data Release 2", Berger et al.
gaia_fp = '/data5/tess_project/Data/Ephemeris_tables/Kepler/KIC_tables/gaia_kepler_crossref_dict_huber.msgp'
with open(gaia_fp, 'rb') as data_file:
    crossref_dict = msgpack.unpack(data_file, strict_map_key=False)

print('Number of Kepler IDs in Gaia DR25: {}'.format(len(crossref_dict)))

kepid_tbl['gaia_dr2_update'] = 'no'

count = 0
for i_kepid, kepid in enumerate(crossref_dict):

    if i_kepid % 5000 == 0:
        print('Row {}/{}'.format(i_kepid + 1, len(crossref_dict)))

    match_kepid = kepid_tbl['kepid'] == kepid

    assert match_kepid.sum() <= 1

    if match_kepid.sum() == 1:
        count += 1

    kepid_tbl.loc[match_kepid, ['teff', 'teff_err1', 'teff_err2', 'radius', 'radius_err1', 'radius_err2',
                                'teff_prov', 'gaia_dr2_update']] = [int(crossref_dict[kepid]['teff']),
                                                                    np.nan,
                                                                    np.nan,
                                                                    float(crossref_dict[kepid]['radius']),
                                                                    np.nan,
                                                                    np.nan,
                                                                    'gaia_dr2',
                                                                    'yes'
                                                                    ]

print('Number of Kepler IDs updated: {}'.format(count))

kepid_tbl.to_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/KIC_tables/'
                 'q1_q17_dr25_stellar_plus_supp_gaiadr2.csv', index=False)

#%% Deal with NaNs in Kepler stellar table by replacing for Solar parameters

keplerStellarTbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/KIC_tables/'
                               'q1_q17_dr25_stellar_plus_supp_gaiadr2.csv')

colsKeplerStellarTbl = ['teff', 'logg', 'feh', 'radius', 'mass', 'dens', 'ra', 'dec']

# solar parameters
solarParamsKepler = {'mass': 1.0, 'dens': 1.408, 'logg': 4.438, 'feh': 0.0, 'radius': 1, 'teff': 5777.0}

# count number of NaNs for each stellar parameter
print('Number of missing values')
print(keplerStellarTbl[colsKeplerStellarTbl].isna().sum(axis=0))

# update NaNs using Solar parameters
for solarParam in solarParamsKepler:
    keplerStellarTbl.loc[keplerStellarTbl[solarParam].isna(),
                         [solarParam, f'{solarParam}_err1', f'{solarParam}_err2']] = \
        [solarParamsKepler[solarParam], np.nan, np.nan]

# confirm after updating with Solar parameters
print(keplerStellarTbl[colsKeplerStellarTbl].isna().sum(axis=0))

keplerStellarTbl.to_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/KIC_tables/'
                        'q1_q17_dr25_stellar_plus_supp_gaiadr2_missingvaluestosolar.csv', index=False)

# %%

stellar_tbl = pd.read_csv(
    '/data5/tess_project/Data/Ephemeris_tables/Kepler/KIC_tables/q1_q17_dr25_stellar_plus_supp_gaiadr2_2018_gaiadr2_2020_missing_solar_to_nan.csv')

tce_tbl_fp = Path(
    '/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17_DR25/q1_q17_dr25_tce_2020.09.28_10.36.22_stellar_koi_cfp_norobovetterlabels_renamedcols_nomissingval_symsecphase_confirmedkoiperiod_sec_rba_cnt0n_koiperiodonlydiff.csv')

tce_tbl = pd.read_csv(tce_tbl_fp)
print(f'Number of TCEs: {len(tce_tbl)}')

stellar_fields_out = ['kepmag', 'tce_steff', 'tce_steff_err1', 'tce_steff_err2', 'tce_slogg', 'tce_slogg_err1',
                      'tce_slogg_err2', 'tce_smet', 'tce_smet_err1', 'tce_smet_err2', 'tce_sradius', 'tce_sradius_err1',
                      'tce_sradius_err2', 'tce_smass', 'tce_smass_err1', 'tce_smass_err2', 'tce_sdens',
                      'tce_sdens_err1', 'tce_sdens_err2', 'ra', 'dec']
stellar_fields_in = ['kepmag', 'teff', 'teff_err1', 'teff_err2', 'logg', 'logg_err1', 'logg_err2', 'feh', 'feh_err1',
                     'feh_err2', 'radius', 'radius_err1', 'radius_err2', 'mass', 'mass_err1', 'mass_err2', 'dens',
                     'dens_err1', 'dens_err2', 'ra', 'dec']

tce_tbl_cols = list(tce_tbl.columns)

for stellar_param in stellar_fields_out:
    if stellar_param not in tce_tbl_cols:
        tce_tbl[stellar_param] = np.nan

tce_tbl['updated_stellar_derived'] = 0
count = 0
for tce_i, tce in tce_tbl.iterrows():

    kic_found = stellar_tbl.loc[stellar_tbl['kepid'] == tce['target_id']]

    assert len(kic_found) == 1

    if (kic_found[['radius', 'teff', 'logg']] != tce[['tce_sradius', 'tce_steff', 'tce_slogg']].values).values.any():
        tce_tbl.loc[tce_i, 'updated_stellar_derived'] = 1

    tce_tbl.loc[tce_i, stellar_fields_out] = kic_found[stellar_fields_in].values[0]

    count += 1

    if tce_i % 500 == 0:
        print(f'Number of TCEs updated: {tce_i + 1}({len(tce_tbl)})')

print(f'Number of TCEs updated: {count}')

# get maximum uncertainty for each stellar parameters
for col in ['tce_steff', 'tce_slogg', 'tce_smass', 'tce_sradius', 'tce_sdens', 'tce_smet']:
    tce_tbl[f'{col}_err'] = np.abs(tce_tbl[[f'{col}_err1', f'{col}_err2']]).max(axis=1, skipna=True)

tce_tbl.to_csv(tce_tbl_fp.parent / f'{tce_tbl_fp.stem}_5-26-2020.csv', index=False)

# %% Update Kepler Q1-Q17 DR25 TCE table (~34k TCEs) with the latest stellar parameters

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
