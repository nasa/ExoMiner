""" Create TESS TCE tables. """

from pathlib import Path
import pandas as pd
from datetime import datetime
import numpy as np
import copy
from tqdm import tqdm
from astroquery.mast import Catalogs
import matplotlib.pyplot as plt

#%% Create TCE table with the TCEs matched to the TOIs

res_dir = Path(f'/data5/tess_project/Data/Ephemeris_tables/TESS/tce_table_{datetime.now().strftime("%m-%d-%Y_%H%M")}')
res_dir.mkdir(exist_ok=True)

match_thr = 0.25

toi_tbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/TESS/EXOFOP_TOI_lists/TOI/3-11-2021/exofop_toilists_spoc_nomissingpephem.csv')
matching_tbl = pd.read_csv('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Analysis/toi_tce_matching/03-12-2021_1308/tois_matchedtces_ephmerismatching_thrinf_samplint1e-05.csv')
matching_tbl = matching_tbl.loc[~matching_tbl['Matched TCEs'].isna()]

tce_root_dir = Path('/data5/tess_project/Data/Ephemeris_tables/TESS/DV_ephemeris')

multisector_tce_dir = tce_root_dir / 'multi-sector runs'
singlesector_tce_dir = tce_root_dir / 'single-sector runs'

sector_tce_tbls = {f'{int(file.stem.split("-")[1][1:])}-{int(file.stem.split("-")[2][1:5])}': pd.read_csv(file,
                                                                                                           header=6)
                        for file in multisector_tce_dir.iterdir() if 'tcestats' in file.name and file.suffix == '.csv'}
sector_tce_tbls.update({f'{int(file.stem.split("-")[1][1:])}': pd.read_csv(file, header=6)
                         for file in singlesector_tce_dir.iterdir() if 'tcestats' in file.name and file.suffix == '.csv'})
sector_tce_tbls['21'].drop_duplicates(subset='tceid', inplace=True, ignore_index=True)

map_tce_tbl_names = {
    'planetNumber': 'tce_plnt_num',
    'orbitalPeriodDays': 'tce_period',
    'orbitalPeriodDays_err': 'tce_period_err',
    'transitEpochBtjd': 'tce_time0bt',
    'transitEpochBtjd_err': 'tce_time0bt_err',
    'ratioPlanetRadiusToStarRadius': 'tce_ror',
    'ratioSemiMajorAxisToStarRadius': 'tce_dor',
    'minImpactParameter': 'tce_impact',
    'transitDurationHours': 'tce_duration',
    'transitIngressTimeHours': 'tce_ingress',
    'transitDepthPpm': 'tce_depth',
    'transitDepthPpm_err': 'tce_depth_err',
    'expectedtransitcount': 'tce_num_transits',
    'planetRadiusEarthRadii': 'tce_prad',
    'planetRadiusEarthRadii_err': 'tce_prad_err',
    'semiMajorAxisAu': 'tce_sma',
    'equilibriumTempKelvin': 'tce_eqt',
    'InsolationFlux': 'tce_insol',
    'planetCandidateCount': 'tce_ntoi',
    'starTeffKelvin': 'tce_steff',
    'starLoggCgs': 'tce_slogg',
    'chiSquare2': 'tce_chisq2',
    'mes': 'tce_max_mult_ev',
    'ses': 'tce_max_sngle_ev',
    'ws_mes': 'tce_ws_maxmes',
    'ws_mesphase': 'tce_ws_maxmesd',
    'starRadiusSolarRadii': 'tce_sradius'
}

for sector_tce_tbl in sector_tce_tbls:

    if sector_tce_tbl not in [str(el) for el in range(31, 35)]:
        sector_tce_tbls[sector_tce_tbl].rename(columns=map_tce_tbl_names, inplace=True)
        for col in sector_tce_tbls['31'].columns:
            if col not in sector_tce_tbls[sector_tce_tbl]:
                sector_tce_tbls[sector_tce_tbl][col] = np.nan

tce_tbl = pd.DataFrame(columns=list(sector_tce_tbls['31'].columns) + toi_cols + ['match_dist'])

toi_tbl.rename(columns={'Sectors': 'TOI Sectors'}, inplace=True)

for toi_i, toi in tqdm(matching_tbl.iterrows()):

    if toi_i % 50 == 0:
        print(f'Iterated through {toi_i}/{len(matching_tbl)} TOIs...\n{len(tce_tbl)} TCEs added to the table.')

    toi_disp = toi_tbl.loc[toi_tbl['TOI'] == toi['Full TOI ID'], toi_cols]

    tces_seen = []

    matched_tces = toi['Matched TCEs'].split(' ')
    target_id = toi['TIC']

    # do not match TCE if the matching distance is aboce the matching threshold
    for matched_tce_i in range(len(matched_tces)):
        if toi[f'matching_dist_{matched_tce_i}'] > match_thr:
            break

        sector_str, tce_plnt_num = matched_tces[matched_tce_i].split('_')

        tce = copy.deepcopy(sector_tce_tbls[sector_str].loc[(sector_tce_tbls[sector_str]['ticid'] == target_id) &
                                                            (sector_tce_tbls[sector_str]['tce_plnt_num'] ==
                                                             int(tce_plnt_num))])

        if '-' not in sector_str:
            tce['sectors'] = sector_str
        else:
            s_sector, e_sector = sector_str.split('-')
            multi_sector = [str(el) for el in list(range(int(s_sector), int(e_sector) + 1))]
            tce['sectors'] = ' '.join(multi_sector)

        # do not match TCE if a TCE from the same run was previously matched
        if tce['sectors'].values[0] in tces_seen:
            continue
        else:
            tces_seen.append(tce['sectors'].values[0])

        tce['match_dist'] = toi[f'matching_dist_{matched_tce_i}']

        for col in toi_disp:
            tce[col] = toi_disp[col].item()

        tce_tbl = pd.concat([tce_tbl, tce[tce_tbl.columns]], axis=0, ignore_index=True)

        # aaaa

tce_tbl_fp = Path(res_dir / f'tess_tce_s1-s34_thr{match_thr}.csv')
tce_tbl.to_csv(tce_tbl_fp, index=False)

# %% Get TIC parameters from TIC

tce_tbl_fp = Path('/data5/tess_project/Data/Ephemeris_tables/TESS/DV_SPOC_mat_files/5-10-2021/tess_tces_s1-s35.csv')
tce_tbl = pd.read_csv(tce_tbl_fp)

tic_fields = {
    'tic_teff': 'Teff',
    'tic_teff_err': 'e_Teff',
    'tic_mass': 'mass',
    'tic_mass_err': 'e_mass',
    'tic_met': 'MH',
    'tic_met_err': 'e_MH',
    'tic_rad': 'rad',
    'tic_rad_err': 'e_rad',
    'tic_rho': 'rho',
    'tic_rho_err': 'e_rho',
    'tic_logg': 'logg',
    'tic_logg_err': 'e_logg',
    'tic_ra': 'ra',
    'tic_dec': 'dec',
    'kic_id': 'KIC',
    'gaia_id': 'GAIA',
    'tic_tmag': 'Tmag',
    'tic_tmag_err': 'e_Tmag',
    'tic_version': 'version'
}

# catalog_data = Catalogs.query_criteria(catalog='TIC', ID=toi_tbl['TIC ID'].unique().tolist()).to_pandas()
catalog_data = Catalogs.query_criteria(catalog='TIC', ID=tce_tbl['target_id'].unique().tolist()).to_pandas()

for tic_i, tic in catalog_data.iterrows():
    # tce_tbl.loc[tce_tbl['ticid'] == int(tic['ID']), tic_fields.keys()] = tic[tic_fields.values()].values
    tce_tbl.loc[tce_tbl['target_id'] == int(tic['ID']), tic_fields.keys()] = tic[tic_fields.values()].values

tce_tbl.to_csv(tce_tbl_fp.parent / f'{tce_tbl_fp.stem}_ticparams.csv', index=False)

# %% Update TIC parameters for the TCEs

tce_tbl = pd.read_csv(tce_tbl_fp.parent / f'{tce_tbl_fp.stem}_ticparams.csv')

tic_map = {
    'tce_steff': 'tic_teff',
    # 'tce_steff_err': 'tic_teff_err',
    'tce_slogg': 'tic_logg',
    # 'tce_slogg_err': 'tic_logg_err',
    'tce_sradius': 'tic_rad',
    # 'tce_sradius_err': 'tic_rad_err',
    'tce_smet': 'tic_met',
    # 'tce_smet_err': 'tic_met_err',
    # 'tce_smass': 'tic_mass',
    'tce_sdens': 'tic_rho',
    # 'tce_sdensity_err': 'tic_rho_err',
    'ra': 'tic_ra',
    'dec': 'tic_dec',
    'mag': 'tic_tmag'
}

# check how many TCEs have Solar parameters
solar_val = {'tic_teff': 5780.0, 'tic_logg': 4.438, 'tic_rad': 1.0, 'tic_met': 0.0, 'tic_rho': 1.0}
for tce_param, tic_param in tic_map.items():
    print(
        f'{tic_param} TCE (TIC) missing values: {tce_tbl[tce_param].isna().sum()} ({tce_tbl[tic_param].isna().sum()})')
    if tic_param not in ['tic_ra', 'tic_dec', 'tic_tmag']:
        a = tce_tbl.loc[tce_tbl[tic_param] != solar_val[tic_param]]
    else:
        a = tce_tbl.copy(deep=True)
    print(f'TCE {a[tce_param].value_counts().idxmax()}: {a[tce_param].value_counts().max()}')
    print(f'TIC {tce_tbl[tic_param].value_counts().idxmax()}: {tce_tbl[tic_param].value_counts().max()}')

tce_stellar_names, tic_stellar_names = [], []
for param in tic_map.keys():
    tce_stellar_names.append(f'{param}')
    tic_stellar_names.append(f'{tic_map[param]}')
    if param not in ['ra', 'dec']:
        tce_stellar_names.append(f'{param}_err')
        tic_stellar_names.append(f'{tic_map[param]}_err')

# DV populates missing values for stellar parameters with Solar parameters; TIC does not
# tce_params_cnt = []
for tce_i, tce in tce_tbl.iterrows():
    # tce_params_cnt_aux = 0
    tce_tbl.loc[tce_i, tce_stellar_names] = tce[tic_stellar_names].values
    # for param in tic_map.keys():
    #     # if np.isnan(tce[param]):
    #     tce_tbl.loc[tce_i, [param]] = tce[tic_map[param]]
    #     if param not in ['ra', 'dec']:
    #         tce_tbl.loc[tce_i, [f'{param}_err']] = tce[f'{tic_map[param]}_err']
    # tce_params_cnt_aux += 1
    # tce_params_cnt.append(tce_params_cnt_aux)

# print(f'Number of TCEs with stellar parameters changed from TIC: {len(np.where(np.array(tce_params_cnt) > 0)[0])}')

tce_tbl.rename(columns={'tic_mass': 'tce_smass', 'tic_mass_err': 'tce_smass_err'}, inplace=True)

tce_tbl.to_csv(tce_tbl_fp.parent / f'{tce_tbl_fp.stem}_stellarparams_updated.csv', index=False)

# %% Rename columns

# choose one disposition as the label column
tce_tbl['label'] = tce_tbl['TFOPWG Disposition']

# rename columns
rename_dict = {
    # 'ticid': 'target_id',
    # 'TOI': 'oi',
    # 'TOI Disposition': 'label',
    # 'TFOPWG Disposition': 'label',
    'tce_depth': 'transit_depth',
    # 'tic_ra': 'ra',
    # 'tic_dec': 'dec',
    # 'tic_tmag': 'mag',
    # 'tic_tmag_err': 'mag_err',
    # 'tce_time0bt': 'tce_time0bk',
    # 'tce_time0bt_err': 'tce_time0bk_err',
    # 'tce_sdensity': 'tce_sdens',
    # 'tce_sdensity_err': 'tce_dens_err',
    # 'tce_ws_maxmesd': 'tce_maxmesd',
    # 'tce_ws_maxmes': 'tce_maxmes'

}
tce_tbl.rename(columns=rename_dict, inplace=True)

# change data type in columns
type_dict = {
    # 'tce_steff': int,
    'sectors': str,
    # 'oi': str
}
tce_tbl = tce_tbl.astype(dtype=type_dict)

tce_tbl.to_csv(tce_tbl_fp.parent / f'{tce_tbl_fp.stem}_renamedcols.csv', index=False)

#%% Check if fields used in preprocessing the time series are valid

fields_to_check = [
    'tce_period',
    'tce_time0bk',
    'tce_duration',
    'transit_depth',
    'tce_maxmesd',
    'ra',
    'dec'
    # 'wst_depth',
    # 'tce_maxmes',
    # 'tce_prad',
]

for field in fields_to_check:
    print(f'Field {field}: {len(tce_tbl.loc[tce_tbl[field].isna()])} missing values')
    if field not in ['tce_maxmesd', 'ra', 'dec']:
        print(f'Field {field}: {len(tce_tbl.loc[tce_tbl[field] <= 0])} non-positive values')

#%% Check if fields used as scalar parameters are valid

fields_to_check = [
    'tce_period',
    'transit_depth',
    'wst_depth',
    'tce_maxmes',
    'tce_prad',
    'tce_steff',
    'tce_smet',
    'tce_slogg',
    'tce_sradius',
    'tce_smass',
    'tce_sdens',
    'mag'
]

for field in fields_to_check:
    print(f'Field {field}: {len(tce_tbl.loc[tce_tbl[field].isna()])} missing values')
    if field not in ['tce_maxmesd']:
        print(f'Field {field}: {len(tce_tbl.loc[tce_tbl[field] <= 0])} non-positive values')

#%% Check number of TCEs per sector

tce_sector_counts = {sector: 0 for sector in range(1, 35)}
for tce_i, tce in tce_tbl.iterrows():
    tce_sectors = tce['sectors'].split(' ')
    for tce_sector in tce_sectors:
        tce_sector_counts[int(tce_sector)] += 1

f, ax = plt.subplots(figsize=(8, 6))
ax.bar(tce_sector_counts.keys(), tce_sector_counts.values(), align='center', edgecolor='k')
ax.set_xlim([0.5, len(tce_sector_counts) + 0.5])
ax.set_ylabel('Counts')
ax.set_xlabel('Sector')
ax.set_xticks(list(tce_sector_counts.keys()))
f.savefig(tce_tbl_fp.parent / 'hist_sector_tces.png')

# just single-sector runs
tce_sector_counts = {sector: 0 for sector in range(1, 35)}
for tce_i, tce in tce_tbl.iterrows():
    tce_sectors = tce['sectors'].split(' ')
    if len(tce_sectors) > 1:
        continue
    for tce_sector in tce_sectors:
        tce_sector_counts[int(tce_sector)] += 1

f, ax = plt.subplots(figsize=(8, 6))
ax.bar(tce_sector_counts.keys(), tce_sector_counts.values(), align='center', edgecolor='k')
ax.set_xlim([0.5, len(tce_sector_counts) + 0.5])
ax.set_ylabel('Counts')
ax.set_xlabel('Sector')
ax.set_xticks(list(tce_sector_counts.keys()))
f.savefig(tce_tbl_fp.parent / 'hist_single-sector_tces.png')

bins = np.linspace(0, 40, 41, endpoint=True)
f, ax = plt.subplots()
ax.hist(tce_tbl['TOI'].value_counts().values, bins=bins, edgecolor='k')
ax.set_ylabel('Number of TOIs')
ax.set_xlabel('Number of TCEs per TOI')
ax.set_yscale('log')
ax.set_xlim([bins[0], bins[-1]])
f.savefig(tce_tbl_fp.parent / 'hist_num_tces_per_toi.png')

# %% Check if there is more than one TCE per target star in the same sector that was matched to the same TOI

unique_tois = tce_tbl['oi'].unique()

tois_more_tces = {}
for toi in unique_tois:
    tces_to_toi = tce_tbl.loc[tce_tbl['oi'] == toi, ['sectors']].value_counts()
    if len(tces_to_toi[tces_to_toi > 1]) > 0:
        tois_more_tces[toi] = tces_to_toi[tces_to_toi > 1]
