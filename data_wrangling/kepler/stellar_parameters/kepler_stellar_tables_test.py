
# 3rd party
import multiprocessing
import os
import matplotlib.pyplot as plt
import msgpack
import numpy as np
import pandas as pd
from astropy.io import fits

# local
from src_preprocessing.lc_preprocessing.kepler_io import kepler_filenames

# %% Add Kepler Stellar parameters in the TCE table in the NASA Exoplanet Archive
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

#%% Update Kepler Q1-Q17 DR24 TCE table with the latest stellar parameters

tce_tbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17_DR24/'
                      'q1_q17_dr24_tce_2020.03.02_17.51.43.csv', header=53)

# stellar_tbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Stellar parameters/Kepler/'
#                           'q1_q17_dr25_stellar_gaiadr2.csv')
stellar_tbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Stellar parameters/Kepler/'
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

    if row_star_i % 100 == 0:
        print('Star {} out of {} ({} %)\n Number of TCEs updated: {}'.format(row_star_i,
                                                                             len(stellar_tbl),
                                                                             row_star_i / len(stellar_tbl) * 100,
                                                                             count))
    target_cond = tce_tbl['kepid'] == row_star['kepid']

    count += target_cond.sum()

    tce_tbl.loc[target_cond, stellar_fields_out] = row_star[stellar_fields_in].values

print('Number of TCEs updated: {}'.format(count))
tce_tbl.to_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17_DR24/'
               'q1_q17_dr24_tce_2020.03.02_17.51.43_stellar.csv', index=False)

#%% Compare stellar radius and effective temperature from Q1-Q17 DR25 stellar and Gaia DR2

save_dir = '/home/msaragoc/Projects/Kepler-TESS_exoplanet/Analysis/gaiadr2_vs_q1q17dr25stellar'

kepid_tbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/KIC_tables/q1_q17_dr25_stellar_plus_supp.csv')

gaia_fp = '/data5/tess_project/Data/Ephemeris_tables/Kepler/KIC_tables/gaia_kepler_crossref_dict_huber.msgp'  # 177911 targets

with open(gaia_fp, 'rb') as data_file:
    crossref_dict = msgpack.unpack(data_file, strict_map_key=False)

gaia_tbl = pd.DataFrame(data={'kepid': np.nan * np.ones(len(crossref_dict)),
                              'teff': np.nan * np.ones(len(crossref_dict)),
                              'radius': np.nan * np.ones(len(crossref_dict))})

for i_kepid, kepid in enumerate(crossref_dict):
    gaia_tbl.loc[i_kepid, ['kepid', 'teff', 'radius']] = [kepid, int(crossref_dict[kepid]['teff']),
                                                          float(crossref_dict[kepid]['radius'])]

# remove Kepler IDs not in Gaia DR2
kepid_tbl = kepid_tbl.loc[kepid_tbl['kepid'].isin(gaia_tbl['kepid'])]

kepid_tbl.sort_values('kepid', axis=0, ascending=True, inplace=True)
gaia_tbl.sort_values('kepid', axis=0, ascending=True, inplace=True)

f, ax = plt.subplots()
ax.scatter(kepid_tbl['teff'], gaia_tbl['teff'], s=5)
ax.plot([2e3, 2e4], [2e3, 2e4], 'k--')
ax.set_xlim([2e3, 2e4])
ax.set_ylim([2e3, 2e4])
ax.set_ylabel('Gaia DR2 Stellar Effective Temperature (K)')
ax.set_xlabel('Q1-Q17 DR25 Stellar Effective Temperature (K)')
f.savefig(os.path.join(save_dir, 'stellar_effective_temp.png'))

f, ax = plt.subplots()
ax.scatter(kepid_tbl['radius'], gaia_tbl['radius'], s=5)
ax.plot([1e-2, 1e3], [1e-2, 1e3], 'k--')
ax.set_ylabel('Gaia DR2 Stellar Radius (R{})'.format(r'$_\bigodot$'))
ax.set_xlabel('Q1-Q17 DR25 Stellar Radius (R{})'.format(r'$_\bigodot$'))
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_xlim([1e-2, 1e3])
ax.set_ylim([1e-2, 1e3])
f.savefig(os.path.join(save_dir, 'stellar_radius.png'))

#%% Plot histograms of stellar parameters for Q1-Q17 DR25

stellar_parameters = ['teff', 'logg', 'feh', 'radius', 'mass', 'dens']
stellar_tbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/KIC_tables/'
                          'q1_q17_dr25_stellar_plus_supp_gaiadr2_missingvaluestosolar.csv', usecols=stellar_parameters)

name_dict = {'teff': 'Effective Temperature (K)',
             'logg': 'Surface Gravity ({})'.format(r'$\log_{10}(cm.s^{-2})$'),
             'feh': 'Metallicity (dex)',
             'radius': 'Radius (R{})'.format(r'$_\bigodot$'),
             'mass': 'Mass (M{})'.format(r'$_\bigodot$'),
             'dens': 'Density ({})'.format(r'$\rho$')}
bins_dict= {'teff': 50,
             'logg': 50,
             'feh': 50,
             'radius': np.logspace(-2, 2, 50),
             'mass': 50,
             'dens': np.logspace(-2, 2, 50)}
f, ax = plt.subplots(2, 3, figsize=(12, 8))
for i in range(6):
    row = int(i / 3)
    col = i % 3
    ax[row, col].hist(stellar_tbl[stellar_parameters[i]], bins=bins_dict[stellar_parameters[i]], edgecolor='k')
    ax[row, col].set_xlabel(name_dict[stellar_parameters[i]])
    if col == 0:
        ax[row, col].set_ylabel('Counts')
    if stellar_parameters[i] in ['radius']:
        # ax[row, col].set_xscale('log')
        ax[row, col].set_xlim([0, 25])
    if stellar_parameters[i] in ['dens']:
        ax[row, col].set_xlim([0, 15])
f.savefig('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Analysis/stellar_parameters_analysis/kepler/'
          'q1_q17_dr25_stellar_plus_supp_gaiadr2_missingvaluestosolar.svg')
f.tight_layout()

#%% Compare Kepler IDs stellar parameters in LC FITS files to Stellar table

# stellar parameters
tableToFITSField = {'tce_steff_fits': 'TEFF', 'tce_slogg_fits': 'LOGG', 'tce_smet_fits': 'FEH',
                    'tce_sradius_fits': 'RADIUS', 'ra_fits': 'RA_OBJ', 'dec_fits': 'DEC_OBJ'}
stellarParams = ['tce_slogg', 'tce_smet', 'tce_sradius', 'tce_steff', 'ra', 'dec']

# root directory for the LC FITS files
fitsRootDir = '/data5/tess_project/Data/Kepler-Q1-Q17-DR25/dr_25_all_final'

# directory in which results are saved
saveDir = '/home/msaragoc/Projects/Kepler-TESS_exoplanet/Analysis/kepler_stellartablevsfits_params'

tceTbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17_DR25/q1_q17_dr25_tce_2020.09.28_10.36.22_'
                     'stellar_koi_cfp_norobovetterlabels_renamedcols_nomissingval_rmcandandfpkois_norogues.csv')
# get only Kepler IDs and stellar parameters from the TCE table
targetTbl = tceTbl.groupby('target_id').first()[stellarParams].reset_index(level=0, inplace=False)
# add columns for the FITS stellar parameters
targetTbl = pd.concat([targetTbl, pd.DataFrame(columns=['{}_fits'.format(col) for col in stellarParams])])

numProcess = 12  # number of processes

subTargetTblList = np.array_split(targetTbl, numProcess)  # split table into subtables processed by each process


def get_stellar_params_from_fits(targetTbl, targetTblIdx, fitsRootDir, saveDir):
    """ Add light curve FITS stellar parameters to the Kepler IDs in the target table.

    :param targetTbl: pandas DataFrame, subtable with stellar parameters for Kepler IDs
    :param targetTblIdx: int, subtable index
    :param fitsRootDir: str, FITS root directory
    :param saveDir: str, directory in which the processed subtables are saved
    :return:
        targetTbl: pandas DataFrame, subtable with also stellar parameters extracted from the light curve FITS files for
        the Kepler IDs
    """

    print('Iterating over sub table {} ({} targets)...'.format(targetTblIdx, len(targetTbl)))
    # iterate over the Kepler IDs in the subtable
    for target_i, target in targetTbl.iterrows():

        # get filenames for the lc FITS files associated with the Kepler ID
        filenames = kepler_filenames(fitsRootDir, target['target_id'])

        # get primary header that contains stellar parameters data
        fitsPrimaryHeader = fits.getheader(filenames[0], 'PRIMARY')

        # add FITS stellar parameters to the table for the given Kepler ID
        for param in tableToFITSField:
            targetTbl.loc[target_i, param] = fitsPrimaryHeader[tableToFITSField[param]]

    print('Finished iterating over sub table {} ({} targets).'.format(targetTblIdx, len(targetTbl)))

    # save the processed subtable
    targetTblFp = os.path.join(saveDir, 'keplerdr25stellar_with_stellarFITS_{}.csv'.format(targetTblIdx))
    targetTbl.to_csv(targetTblFp, index=False)
    print('Saved sub table {} to {}.'.format(targetTblIdx, targetTblFp))

    return targetTbl


# create mp pool that spans n processes each one processing with a subtable
pool = multiprocessing.Pool(processes=numProcess)
jobs = [(subTargetTbl, subTargetTbl_i, fitsRootDir, saveDir)
        for subTargetTbl_i, subTargetTbl in enumerate(subTargetTblList)]
async_results = [pool.apply_async(get_stellar_params_from_fits, job) for job in jobs]
pool.close()

# instead of pool.join(), async_result.get() to ensure any exceptions raised by the worker processes are raised here
subTblsProcessed = [async_result.get() for async_result in async_results]
stellarTblWithFits = pd.concat(subTblsProcessed, axis=0)
stellarTblWithFits.to_csv(os.path.join(saveDir, 'keplerdr25stellar_with_stellarFITS.csv'), index=False)

# scatter plot of stellar parameters in table against LC FITS data
for param in stellarParams:

    f, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(stellarTblWithFits[param], stellarTblWithFits['{}_fits'.format(param)], s=5)
    ax.set_xlabel('Stellar Table\n{}'.format(param))
    ax.set_ylabel('LC FITS\n{}'.format(param))
    ax.grid(True)
    f.savefig(os.path.join(saveDir, 'stellartblvslcfits_{}.png'.format(param)))
