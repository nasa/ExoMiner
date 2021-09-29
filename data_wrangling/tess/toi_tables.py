from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astroquery.mast import Catalogs

# %% MAST TOI catalog

toi_dir = Path('/data5/tess_project/Data/Ephemeris_tables/TESS/TOI_catalogs/2-18-2021')

toi_tbl = pd.read_csv(toi_dir / f'tois.csv', header=4)
num_tois = len(toi_tbl)
print(f'Total number of TOIs: {num_tois}')

# remove QLP TOIs
toi_tbl = toi_tbl.loc[toi_tbl['Source Pipeline'] == 'spoc']
print(f'Total number of TOIs after removing QLP TOIs: {len(toi_tbl)} ({num_tois - len(toi_tbl)})')
num_tois = len(toi_tbl)

for param in ['Epoch Value', 'Orbital Period Value', 'Transit Duration Value', 'Transit Depth Value', 'Sectors']:
    toi_tbl = toi_tbl.loc[~toi_tbl[param].isna()]
    print(f'Total number of TOIs after removing TOIs without {param}: {len(toi_tbl)} ({num_tois - len(toi_tbl)})')
    num_tois = len(toi_tbl)

toi_tbl.to_csv(toi_dir / 'tois_spoc_nomissingephemerides.csv', index=False)

# %% EXOFOP TOI catalog

toi_dir = Path('/data5/tess_project/Data/Ephemeris_tables/TESS/EXOFOP_TOI_lists/TOI/9-10-2021/')

toi_tbl = pd.read_csv(toi_dir / f'exofop_tess_tois.csv', header=1)

# compute TESS BJD
# toi_tbl['Epoch (TBJD)'] = toi_tbl['Epoch (BJD)'] - 2457000
toi_tbl['Epoch (TBJD)'] = toi_tbl['Transit Epoch (BJD)'] - 2457000
toi_tbl.to_csv(toi_dir / 'exofop_toilists_tbjd.csv', index=False)

num_tois = len(toi_tbl)
print(f'Total number of TOIs: {num_tois}')

# # remove QLP TOIs
# toi_tbl = toi_tbl.loc[toi_tbl['Source'] == 'spoc']
# print(f'Total number of TOIs after removing QLP TOIs: {len(toi_tbl)} ({num_tois - len(toi_tbl)})')
# num_tois = len(toi_tbl)

for param in ['Epoch (TBJD)', 'Period (days)', 'Duration (hours)', 'Depth (ppm)', 'Sectors']:
    if param in ['Period (days)']:
        toi_tbl = toi_tbl.loc[toi_tbl[param] != 0]

    toi_tbl = toi_tbl.loc[~toi_tbl[param].isna()]
    print(f'Total number of TOIs after removing TOIs without {param}: {len(toi_tbl)} ({num_tois - len(toi_tbl)})')
    num_tois = len(toi_tbl)

toi_tbl.to_csv(toi_dir / 'exofop_toilists_nomissingpephem.csv', index=False)

#%% add parameters from the TCE tables to the TOI table using the matched TCEs to each TOI

tce_root_dir = Path('/data5/tess_project/Data/Ephemeris_tables/TESS/DV_ephemeris')

multisector_tce_dir = tce_root_dir / 'multi-sector runs'
singlesector_tce_dir = tce_root_dir / 'single-sector runs'

multisector_tce_tbls = {(int(file.stem.split('-')[1][1:]), int(file.stem.split('-')[2][1:5])): pd.read_csv(file,
                                                                                                           header=6)
                        for file in multisector_tce_dir.iterdir() if 'tcestats' in file.name}
singlesector_tce_tbls = {int(file.stem.split('-')[1][1:]): pd.read_csv(file, header=6)
                         for file in singlesector_tce_dir.iterdir() if 'tcestats' in file.name and file.suffix == '.csv'}
singlesector_tce_tbls[21].drop_duplicates(subset='tceid', inplace=True, ignore_index=True)

matching_tbl = pd.read_csv('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Analysis/toi_tce_matching/04-12-2021_1316/tois_matchedtces_ephmerismatching_thrinf_samplint1e-05.csv')
# remove TOIs whose closest TCE matching distance is above the matching threshold
matching_thr = np.inf
matching_tbl = matching_tbl.loc[matching_tbl['matching_dist_0'] <= matching_thr]


toi_tbl_fp = Path('/data5/tess_project/Data/Ephemeris_tables/TESS/EXOFOP_TOI_lists/TOI/4-12-2021/'
                  'exofop_toilists_nomissingpephem.csv')
toi_tbl = pd.read_csv(toi_tbl_fp)
print(f'Total number of TOIs: {len(toi_tbl)}')

print(toi_tbl['TESS Disposition'].value_counts())
print(toi_tbl['TFOPWG Disposition'].value_counts())

# filter TOIs that were not matched with any TCE
# toi_tbl = toi_tbl.loc[toi_tbl['Full TOI ID'].isin(matching_tbl.loc[~matching_tbl['Matched TCEs'].isna(),
#                                                                    'Full TOI ID'].to_list())]
toi_tbl = toi_tbl.loc[toi_tbl['TOI'].isin(matching_tbl.loc[~matching_tbl['Matched TCEs'].isna(),
                                                                   'Full TOI ID'].to_list())]
print(f'Number of TOIs after removing those not matched to any TCE: {len(toi_tbl)}')

fields_to_add = {
    'src_old':
        {
            'tce_plnt_num': 'planetNumber',
            'tce_period_dv': 'orbitalPeriodDays',
            'tce_duration_dv': 'transitDurationHours',
            'tce_time0bk_dv': 'transitEpochBtjd',
            'transit_depth_dv': 'transitDepthPpm',
            'tce_prad_dv': 'planetRadiusEarthRadii',
            'tce_ror': 'ratioPlanetRadiusToStarRadius',
            'tce_dor': 'ratioSemiMajorAxisToStarRadius',
            'tce_sma': 'semiMajorAxisAu',
            'tce_ingress': 'transitIngressTimeHours',
            'tce_impact': 'minImpactParameter',
            # 'tce_incl': np.nan,
            # 'wst_depth': np.nan,
            'tce_maxmes': 'ws_mes',
            'tce_maxmesd': 'ws_mesphase',
            'tce_max_mult_ev': 'mes',
            'tce_max_sngle_ev': 'maxses',
            'tce_eqt_dv': 'equilibriumTempKelvin',
            'tce_insol_dv': 'InsolationFlux',
            'num_transits': 'expectedtransitcount',
            # 'tce_albedo': np.nan,
            # 'tce_albedo_stat': np.nan,
            # 'tce_ptemp': np.nan,
            # 'tce_ptemp_stat': np.nan,
            # 'boot_fap': np.nan,
            # 'tce_cap_stat': np.nan,
            # 'tce_hap_stat': np.nan,
            # 'tce_dicco_msky': np.nan,
            # 'tce_dicco_msky_err': np.nan,
            # 'tce_dikco_msky': np.nan,
            # 'tce_dikco_msky_err': np.nan,
            'tce_steff_dv': 'starTeffKelvin',
            'tce_slogg_dv': 'starLoggCgs',
            # 'tce_smet': np.nan,
            # 'tce_sradius': np.nan,
            # 'tce_densitiy': np.nan
        },

    'new':
        {
            'tce_plnt_num': 'tce_plnt_num',
            'tce_period_dv': 'tce_period',
            'tce_duration_dv': 'tce_duration',
            'tce_time0bk_dv': 'tce_time0bt',
            'transit_depth_dv': 'tce_depth',
            'tce_prad_dv': 'tce_prad',
            'tce_ror': 'tce_ror',
            'tce_dor': 'tce_dor',
            'tce_sma': 'tce_sma',
            'tce_ingress': 'tce_ingress',
            'tce_impact': 'tce_impact',
            'tce_incl': 'tce_incl',
            'wst_depth': 'wst_depth',
            'tce_maxmes': 'tce_ws_maxmes',
            'tce_maxmesd': 'tce_ws_maxmesd',
            'tce_max_mult_ev': 'tce_max_mult_ev',
            'tce_max_sngle_ev': 'tce_max_sngle_ev',
            'tce_eqt_dv': 'tce_eqt',
            'tce_insol_dv': 'tce_insol',
            'num_transits': 'tce_num_transits',
            'tce_albedo': 'tce_albedo',
            'tce_albedo_stat': 'tce_albedo_stat',
            'tce_ptemp': 'tce_ptemp',
            'tce_ptemp_stat': 'tce_ptemp_stat',
            'boot_fap': 'boot_fap',
            'tce_cap_stat': 'tce_cap_stat',
            'tce_hap_stat': 'tce_hap_stat',
            'tce_dicco_msky': 'tce_dicco_msky',
            'tce_dicco_msky_err': 'tce_dicco_msky_err',
            'tce_dikco_msky': 'tce_ditco_msky',
            'tce_dikco_msky_err': 'tce_ditco_msky_err',
            'tce_steff_dv': 'tce_steff',
            'tce_slogg_dv': 'tce_slogg',
            'tce_smet': 'tce_smet',
            'tce_sradius': 'tce_sradius',
            'tce_densitiy': 'tce_sdensity',
        }
}

for field in fields_to_add['new'].keys():
    toi_tbl[field] = np.nan

for toi_i, toi in toi_tbl.iterrows():

    if toi_i % 100 == 0:
        print(f'Iterated over {toi_i + 1} out of {len(toi_tbl)}')

    # toi_found = matching_tbl.loc[matching_tbl['Full TOI ID'] == toi['Full TOI ID']]
    toi_found = matching_tbl.loc[matching_tbl['Full TOI ID'] == toi['TOI']]

    if len(toi_found) > 0:

        matched_tce_sector_str, matched_tceid = toi_found['Matched TCEs'].values[0].split(' ')[0].split('_')
        toi_tceid = '{}-{}'.format(f'{toi_found["TIC"].values[0]}'.zfill(11), f'{matched_tceid}'.zfill(2))

        if '-' in matched_tce_sector_str:
            sector_start, sector_end = matched_tce_sector_str.split('-')
            tce_tbl = multisector_tce_tbls[(int(sector_start), int(sector_end))]
            toi_tbl.loc[toi_i, list(fields_to_add['src_old'].keys())] = tce_tbl.loc[(tce_tbl['tceid'] == toi_tceid),
                                                                                    list(fields_to_add[
                                                                                             'src_old'].values())].values[
                0]
        else:
            tce_tbl = singlesector_tce_tbls[int(matched_tce_sector_str)]
            if int(matched_tce_sector_str) <= 30:
                toi_tbl.loc[toi_i, list(fields_to_add['src_old'].keys())] = tce_tbl.loc[(tce_tbl['tceid'] == toi_tceid),
                                                                                        list(fields_to_add[
                                                                                                 'src_old'].values())].values[
                    0]
            else:
                toi_tbl.loc[toi_i, list(fields_to_add['new'].keys())] = tce_tbl.loc[(tce_tbl['tceid'] == toi_tceid),
                                                                           list(fields_to_add['new'].values())].values[0]

toi_tbl.to_csv(toi_tbl_fp.parent / f'{toi_tbl_fp.stem}_tcesparams_thr{matching_thr}.csv', index=False)

print(toi_tbl['TESS Disposition'].value_counts())
print(toi_tbl['TFOPWG Disposition'].value_counts())

#%% add parameters from the TCE tables to the TOI table using the matched TCEs to each TOI from the DV mat tables

tce_root_dir = Path('/data5/tess_project/Data/Ephemeris_tables/TESS/DV_SPOC_mat_files')

multisector_tce_dir = tce_root_dir / 'multi-sector' / 'csv_tables'
singlesector_tce_dir = tce_root_dir / 'single-sector' / 'csv_tables'

multisector_tce_tbls = {(int(file.stem[14:16]), int(file.stem[16:18])): pd.read_csv(file)
                        for file in multisector_tce_dir.iterdir()}
singlesector_tce_tbls = {int(file.stem[14:16]): pd.read_csv(file)
                         for file in singlesector_tce_dir.iterdir()}

matching_tbl = pd.read_csv('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Analysis/toi_tce_matching/04-16-2021_1625/'
                           'tois_matchedtces_ephmerismatching_thrinf_samplint1e-05.csv')
# remove TOIs not matched to any TCE
matching_tbl = matching_tbl.loc[~matching_tbl['Matched TCEs'].isna()]
print(f'Number of TOIs before thresholding the matching distance to the closest TCE: {len(matching_tbl)}')
# remove TOIs whose closest TCE matching distance is above the matching threshold
matching_thr = 0.25
matching_tbl = matching_tbl.loc[matching_tbl['matching_dist_0'] <= matching_thr]
print(f'Number of TOIs after thresholding (thr={matching_thr}) the matching distance to the closest TCE: '
      f'{len(matching_tbl)}')

toi_tbl_fp = Path('/data5/tess_project/Data/Ephemeris_tables/TESS/EXOFOP_TOI_lists/TOI/4-22-2021/exofop_toilists_nomissingpephem.csv')
toi_tbl = pd.read_csv(toi_tbl_fp)
print(f'Total number of TOIs: {len(toi_tbl)}')

print(toi_tbl['TESS Disposition'].value_counts())
print(toi_tbl['TFOPWG Disposition'].value_counts())

toi_tbl = toi_tbl.loc[toi_tbl['TOI'].isin(matching_tbl['TOI ID'].to_list())]
# print(f'Number of TOIs after removing those not matched to any TCE: {len(toi_tbl)}')

# choose columns to add from the TCE tables
fields_to_add = {
    'planetIndexNumber': 'tce_plnt_num',
    'planetCandidate.maxSingleEventSigma': 'tce_max_sngle_ev',
    'planetCandidate.maxMultipleEventSigma': 'tce_max_mult_ev',
    'allTransitsFit_transitDepthPpm_value': 'tce_depth_dv',
    'allTransitsFit_transitDepthPpm_uncertainty': 'tce_depth_err_dv',
    'allTransitsFit_orbitalPeriodDays_value': 'tce_period_dv',
    'allTransitsFit_orbitalPeriodDays_uncertainty': 'tce_period_err_dv',
    'allTransitsFit_transitEpochBtjd_value': 'tce_time0bk_dv',
    'allTransitsFit_transitEpochBtjd_uncertainty': 'tce_time0bk_err_dv',
    'allTransitsFit_transitDurationHours_value': 'tce_duration_dv',
    'allTransitsFit_transitDurationHours_uncertainty': 'tce_duration_err_dv',
    'allTransitsFit_planetRadiusEarthRadii_value': 'tce_prad_dv',
    'allTransitsFit_planetRadiusEarthRadii_uncertainty': 'tce_prad_err_dv',
    'allTransitsFit_starRadiusSolarRadii_value': 'tce_sradius',
    'allTransitsFit_starRadiusSolarRadii_uncertainty': 'tce_sradius_err',
    'allTransitsFit_semiMajorAxisAu_value': 'tce_sma',
    'allTransitsFit_semiMajorAxisAu_uncertainty': 'tce_sma_err',
    'allTransitsFit_minImpactParameter_value': 'tce_impact',
    'allTransitsFit_minImpactParameter_uncertainty': 'tce_impact_err',
    'oddTransitsFit_transitDepthPpm_value': 'tce_depth_odd',
    'oddTransitsFit_transitDepthPpm_uncertainty': 'tce_depth_odd_err',
    'oddTransitsFit_orbitalPeriodDays_value': 'tce_period_odd',
    'oddTransitsFit_orbitalPeriodDays_uncertainty': 'tce_period_odd_err',
    'oddTransitsFit_transitEpochBtjd_value': 'tce_time0bk_odd',
    'oddTransitsFit_transitEpochBtjd_uncertainty': 'tce_time0bk_odd_err',
    'oddTransitsFit_transitDurationHours_value': 'tce_duration_odd',
    'oddTransitsFit_transitDurationHours_uncertainty': 'tce_duration_odd_err',
    'evenTransitsFit_transitDepthPpm_value': 'tce_depth_even',
    'evenTransitsFit_transitDepthPpm_uncertainty': 'tce_depth_even_err',
    'evenTransitsFit_orbitalPeriodDays_value': 'tce_period_even',
    'evenTransitsFit_orbitalPeriodDays_uncertainty': 'tce_period_even_err',
    'evenTransitsFit_transitEpochBtjd_value': 'tce_time0bk_even',
    'evenTransitsFit_transitEpochBtjd_uncertainty': 'tce_time0bk_even_err',
    'evenTransitsFit_transitDurationHours_value': 'tce_duration_even',
    'evenTransitsFit_transitDurationHours_uncertainty': 'tce_duration_even_err',
    # 'planetCandidate.epochTjd': 'tce_time0bk',
    # 'planetCandidate.orbitalPeriod': 'tce_period',
    'centroidResults.differenceImageMotionResults.msTicCentroidOffsets.meanSkyOffset_value': 'tce_dikco_msky',
    'centroidResults.differenceImageMotionResults.msTicCentroidOffsets.meanSkyOffset_uncertainty': 'tce_dikco_msky_err',
    'centroidResults.differenceImageMotionResults.msControlCentroidOffsets.meanSkyOffset_value': 'tce_dicco_msky',
    'centroidResults.differenceImageMotionResults.msControlCentroidOffsets.meanSkyOffset_uncertainty': 'tce_dicco_msky_err',
    'tessMag_value': 'mag_dv',
    'tessMag_uncertainty': 'mag_err_dv',
    'radius_value': 'tce_sradius_dv',
    'radius_uncertainty': 'tce_sradius_err_dv',
    'stellarDensity_value': 'tce_sdens_dv',
    'stellarDensity_uncertainty': 'tce_dens_err_dv',
    'effectiveTemp_value': 'tce_steff_dv',
    'effectiveTemp_uncertainty': 'tce_steff_err_dv',
    'log10SurfaceGravity_value': 'tce_slogg_dv',
    'log10SurfaceGravity_uncertainty': 'tce_slogg_err_dv',
    'log10Metallicity_value': 'tce_smet_dv',
    'log10Metallicity_uncertainty': 'tce_met_err_dv',
    'allTransitsFit_equilibriumTempKelvin_value': 'tce_eqt_dv',
    'allTransitsFit_equilibriumTempKelvin_uncertainty': 'tce_eqt_err_dv',
    'planetCandidate.weakSecondaryStruct.maxMes': 'tce_maxmes',
    'planetCandidate.weakSecondaryStruct.maxMesPhaseInDays': 'tce_maxmesd',
    'binaryDiscriminationResults.oddEvenTransitDepthComparisonStatistic_value': 'tce_bin_oedp_stat',
    'binaryDiscriminationResults.shorterPeriodComparisonStatistic_value': 'tce_shortper_stat',
    'binaryDiscriminationResults.longerPeriodComparisonStatistic_value': 'tce_longper_stat',
    'bootstrapResults.falseAlarmRate': 'boot_fap',
    'allTransitsFit_ratioPlanetRadiusToStarRadius_value': 'tce_ror',
    'allTransitsFit_ratioPlanetRadiusToStarRadius_uncertainty': 'tce_ror_err',
    'allTransitsFit_ratioSemiMajorAxisToStarRadius_value': 'tce_dor',
    'allTransitsFit_ratioSemiMajorAxisToStarRadius_uncertainty': 'tce_dor_err',
    'allTransitsFit.modelFitSnr': 'tce_model_snr_dv',
    'allTransitsFit_transitIngressTimeHours_value': 'tce_ingress',
    'allTransitsFit_transitIngressTimeHours_uncertainty': 'tce_ingress_err',
    'allTransitsFit_inclinationDegrees_value': 'tce_incl',
    'allTransitsFit_inclinationDegrees_uncertainty': 'tce_incl_err',
    'allTransitsFit_eccentricity_value': 'tce_eccen',
    'allTransitsFit_eccentricity_uncertainty': 'tce_eccen_err',
    'allTransitsFit_longitudeOfPeriDegrees_value': 'tce_longp',
    'allTransitsFit_longitudeOfPeriDegrees_uncertainty': 'tce_longp_err',
    'raDegrees_value': 'ra',
    'decDegrees_value': 'dec',
    'secondaryEventResults.planetParameters.geometricAlbedo_value': 'tce_albedo',
    'secondaryEventResults.planetParameters.geometricAlbedo_uncertainty': 'tce_albedo_err',
    'secondaryEventResults.planetParameters.planetEffectiveTemp_value': 'tce_ptemp',
    'secondaryEventResults.planetParameters.planetEffectiveTemp_uncertainty': 'tce_ptemp_err',
    'secondaryEventResults.comparisonTests.albedoComparisonStatistic_value': 'tce_albedo_stat',
    'secondaryEventResults.comparisonTests.tempComparisonStatistic_value': 'tce_ptemp_stat',
    'allTransitsFit_effectiveStellarFlux_value': 'tce_insol_dv',
    'allTransitsFit_effectiveStellarFlux_uncertainty': 'tce_insol_err_dv',
    'ghostDiagnosticResults.coreApertureCorrelationStatistic_value': 'tce_cap_stat',
    'ghostDiagnosticResults.haloApertureCorrelationStatistic_value': 'tce_hap_stat',
    'planetCandidate.observedTransitCount': 'tce_num_transits_obs',
    'planetCandidate.expectedTransitCount': 'tce_num_transits',
    'planetCandidate.weakSecondaryStruct.depthPpm_value': 'wst_depth',
    'planetCandidate.weakSecondaryStruct.depthPpm_uncertainty': 'wst_depth_err',
    # 'allTransitsFit_starDensitySolarDensity_value': 'tce_sdens',
    # 'allTransitsFit_starDensitySolarDensity_uncertainty': 'tce_sdens_err'
}

for field in fields_to_add.values():
    toi_tbl[field] = np.nan

toi_tbl.reset_index(inplace=True)
for toi_i, toi in toi_tbl.iterrows():

    if toi_i % 100 == 0:
        print(f'Iterated over {toi_i + 1} out of {len(toi_tbl)}')

    toi_found = matching_tbl.loc[matching_tbl['TOI ID'] == toi['TOI']]

    if len(toi_found) > 0:

        matched_tce_sector_str, matched_tceid = toi_found['Matched TCEs'].values[0].split(' ')[0].split('_')

        if '-' in matched_tce_sector_str:
            sector_start, sector_end = matched_tce_sector_str.split('-')
            tce_tbl = multisector_tce_tbls[(int(sector_start), int(sector_end))]
        else:
            tce_tbl = singlesector_tce_tbls[int(matched_tce_sector_str)]

        toi_tbl.loc[toi_i, list(fields_to_add.values())] = \
            tce_tbl.loc[(tce_tbl['catId'] == toi_found['TIC'].values[0]) &
                        (tce_tbl['planetIndexNumber'] == float(matched_tceid)), list(fields_to_add.keys())].values[0]

toi_tbl.to_csv(toi_tbl_fp.parent / f'{toi_tbl_fp.stem}_tcesparams_thr{matching_thr}.csv', index=False)

print(toi_tbl['TESS Disposition'].value_counts())
print(toi_tbl['TFOPWG Disposition'].value_counts())

#%% Convert TIC RA and Dec from sexagesimal to deg

ra_dec_deg = SkyCoord(ra=toi_tbl['RA'].to_numpy(), dec=toi_tbl['Dec'].to_numpy(), unit=(u.hourangle, u.degree),
                                                 frame='icrs')
toi_tbl['ra_deg'], toi_tbl['dec_deg'] = ra_dec_deg.ra.value, ra_dec_deg.dec.value

toi_tbl.to_csv(toi_tbl_fp.parent / f'{toi_tbl_fp.stem}_ra_dec_deg.csv', index=False)

#%% Get TIC parameters from TIC

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
    'tic_tmag_err': 'e_Tmag'
}

catalog_data = Catalogs.query_criteria(catalog='TIC', ID=toi_tbl['TIC ID'].unique().tolist()).to_pandas()

for tic_i, tic in catalog_data.iterrows():
    toi_tbl.loc[toi_tbl['TIC ID'] == int(tic['ID']), tic_fields.keys()] = tic[tic_fields.values()].values

toi_tbl.to_csv(toi_tbl_fp.parent / f'{toi_tbl_fp.stem}_ticparams.csv', index=False)

#%% Update TIC parameters for the TOIs

tic_map = {
    'Stellar Eff Temp (K)': 'tic_teff',
    'Stellar Eff Temp (K) err': 'tic_teff_err',
    'Stellar log(g) (cm/s^2)': 'tic_logg',
    'Stellar log(g) (cm/s^2) err': 'tic_logg_err',
    'Stellar Radius (R_Sun)': 'tic_rad',
    'Stellar Radius (R_Sun) err': 'tic_rad_err',
    'Stellar Metallicity': 'tic_met',
    'Stellar Metallicity err': 'tic_met_err',
    'TESS Mag': 'tic_tmag',
    'TESS Mag err': 'tic_tmag_err',
}

toi_params_cnt = []
for toi_i, toi in toi_tbl.iterrows():
    toi_params_cnt_aux = 0
    for param in tic_map.keys():
        if np.isnan(toi[param]):
            toi[param] = toi[tic_map[param]]
            toi_params_cnt_aux += 1
    toi_params_cnt.append(toi_params_cnt_aux)

print(f'Number of TOIs with stellar parameters changed from TIC: {len(np.where(np.array(toi_params_cnt) > 0)[0])}')

toi_tbl.to_csv(toi_tbl_fp.parent / f'{toi_tbl_fp.stem}_stellarparams_updated.csv', index=False)

#%% Replace stellar missing values by solar parameters

# solar parameters
solar_params = {
    'Stellar Eff Temp (K)': 5777,
    'Stellar log(g) (cm/s^2)': 4.438,
    'Stellar Radius (R_Sun)': 1.0,
    'Stellar Metallicity': 0.0,
    'tic_mass': 1.0,
    'tic_rho': 1.408
}

for solar_param in solar_params:
    print(f'Number of missing values for {solar_param}: {len(toi_tbl.loc[toi_tbl[solar_param].isna()])}')

for toi_i, toi in toi_tbl.iterrows():
    for param in solar_params:
        if np.isnan(toi[param]):
            toi_tbl.loc[toi_i, [param]] = solar_params[param]

toi_tbl.to_csv(toi_tbl_fp.parent / f'{toi_tbl_fp.stem}_stellarparams_missing_to_solar.csv', index=False)

#%% Rename columns

# rename columns
rename_dict = {
    # 'TIC': 'target_id',
    'TIC ID': 'target_id',
    # 'Full TOI ID': 'oi',
    'TOI': 'oi',
    # 'TOI Disposition': 'label',
    'TFOPWG Disposition': 'label',
    # 'TIC Right Ascension': 'ra',
    # 'TIC Declination': 'dec',
    'TESS Mag': 'mag',
    'TESS Mag err': 'mag_err',
    # 'Epoch Value': 'tce_time0bk',
    # 'Epoch Error': 'tce_time0bk_err',
    # 'Orbital Period Value': 'tce_period',
    # 'Orbital Period Error': 'tce_period_err',
    # 'Transit Duration Value': 'tce_duration',
    # 'Transit Duration Error': 'tce_duration_err',
    # 'Transit Depth Value': 'transit_depth',
    # 'Transit Depth Error': 'tce_depth_err',

    'Sectors': 'sectors',
    'Period (days)': 'tce_period',
    'Period (days) err': 'tce_period_err',
    'Epoch (TBJD)': 'tce_time0bk',
    'Epoch (BJD) err': 'tce_time0bk_err',
    'Duration (hours)': 'tce_duration',
    'Duration (hours) err': 'tce_duration_err',
    'Depth (ppm)': 'transit_depth',
    'Depth (ppm) err': 'tce_depth_err',
    'Planet Radius (R_Earth)': 'tce_prad',
    'Planet Radius (R_Earth) err': 'tce_prad_err',
    'Planet Insolation (Earth Flux)': 'tce_insol',
    'Planet Equil Temp (K)': 'tce_eqt',
    'Planet SNR': 'tce_model_snr',
    'Stellar Eff Temp (K)': 'tce_steff',
    'Stellar Eff Temp (K) err': 'tce_steff_err',
    'Stellar log(g) (cm/s^2)': 'tce_slogg',
    'Stellar log(g) (cm/s^2) err': 'tce_slogg_err',
    'Stellar Radius (R_Sun)': 'tce_srad',
    'Stellar Radius (R_Sun) err': 'tce_srad_err',
    'Stellar Metallicity': 'tce_smet',
    'Stellar Metallicity err': 'tce_smet_err',
    'tic_mass': 'tce_smass',
    'tic_mass_err': 'tce_smass_err',
    'tic_rho': 'tce_sdens',
    'tic_rho_err': 'tce_sdens_err'
}
toi_tbl.rename(columns=rename_dict, inplace=True)

# change data type in columns
type_dict = {
    # 'tce_steff': int,
    'sectors': str,
    'oi': str
}
toi_tbl = toi_tbl.astype(dtype=type_dict)

toi_tbl.to_csv(toi_tbl_fp.parent / f'{toi_tbl_fp.stem}_renamedcols.csv', index=False)

#%% Change sector column format from comma separation to space

toi_tbl['sectors'] = toi_tbl['sectors'].apply(lambda x: ' '.join(x.split(',')))
toi_tbl.to_csv(toi_tbl_fp.parent / f'{toi_tbl_fp.stem}_sectors.csv', index=False)

#%% Check if fields used in preprocessing the time series are valid

fields_to_check = [
    'tce_period',
    'tce_time0bk',
    'tce_duration',
    'transit_depth',
    'tce_maxmesd',
    # 'wst_depth',
    # 'tce_maxmes',
    # 'tce_prad',
]

for field in fields_to_check:
    print(f'Field {field}: {len(toi_tbl.loc[toi_tbl[field].isna()])} missing values')
    if field not in ['tce_maxmesd']:
        print(f'Field {field}: {len(toi_tbl.loc[toi_tbl[field] <= 0])} non-positive values')

#%% Check relationship between TOI and DV TCE parameters

params = [
    ('tce_period', 'tce_period_dv'),
    ('tce_duration', 'tce_duration_dv'),
    ('tce_time0bk', 'tce_time0bk_dv'),
    ('transit_depth', 'tce_depth_dv'),
    ('tce_prad', 'tce_prad_dv'),
    ('tce_eqt', 'tce_eqt_dv'),
    ('tce_insol', 'tce_insol_dv'),
    ('tce_steff', 'tce_steff_dv'),
    ('tce_slogg', 'tce_slogg_dv')
]

params_zoom_range =\
    {
    # 'tce_period': [0, 50],
    # 'tce_duration': [0, 10],
    # 'tce_insol': [0, 20000],
        'tce_steff': [3000, 10000],
        'tce_slogg': [3, 6],
}

params_log_scale = ['tce_period', 'tce_duration', 'tce_insol', 'tce_prad', 'transit_depth']

for param in params:

    toi_tbl_param = toi_tbl.loc[(~toi_tbl[param[0]].isna()) & ((~toi_tbl[param[1]].isna()))]
    print(f'{param[0]}: {len(toi_tbl_param)} out of {len(toi_tbl)}')
    toi_tbl_param_min, toi_tbl_param_max = min(toi_tbl_param[param[0]].min(), toi_tbl_param[param[1]].min()), \
                                           max(toi_tbl_param[param[0]].max(), toi_tbl_param[param[1]].max())

    f, ax = plt.subplots()
    ax.scatter(toi_tbl_param[param[0]], toi_tbl_param[param[1]], c='b', s=8)
    ax.plot(np.linspace(toi_tbl_param_min, toi_tbl_param_max, 2, endpoint=True),
            np.linspace(toi_tbl_param_min, toi_tbl_param_max, 2, endpoint=True), 'r--')
    ax.set_ylabel(param[0])
    ax.set_xlabel(param[1])
    ax.grid(True)
    if param in params_log_scale:
        ax.set_yscale('log')
        ax.set_xscale('log')
        f.savefig(toi_tbl_fp.parent / f'scatter_{param[0]}_log.png')
    else:
        f.savefig(toi_tbl_fp.parent / f'scatter_{param[0]}.png')
    plt.close()

    if param[0] in params_zoom_range:
        f, ax = plt.subplots()
        ax.scatter(toi_tbl_param[param[0]], toi_tbl_param[param[1]], c='b', s=8)
        ax.plot(np.linspace(toi_tbl_param_min, toi_tbl_param_max, 2, endpoint=True),
                np.linspace(toi_tbl_param_min, toi_tbl_param_max, 2, endpoint=True), 'r--')
        ax.set_ylabel(param[0])
        ax.set_xlabel(param[1])
        ax.set_xlim(params_zoom_range[param[0]])
        ax.set_ylim(params_zoom_range[param[0]])
        ax.grid(True)
        # if param in params_log_scale:
        #     ax.set_yscale('log')
        #     ax.set_xscale('log')
        #     f.savefig(toi_tbl_fp.parent / f'scatter_{param[0]}_log.png')
        # else:
        f.savefig(toi_tbl_fp.parent / f'scatter_{param[0]}_zoomin.png')
        plt.close()

#%% Check TOI vs TCE epoch with a sector grid

fits_root_dir = Path('/data5/tess_project/Data/TESS_TOI_fits(MAST)')

sectors_dir = [fp for fp in fits_root_dir.iterdir() if fp.name.startswith('sector')]

sectors_se = {}
for sector_fp in sectors_dir:
    print(f'{sector_fp.stem}')
    fits_fp = list(sector_fp.iterdir())[0]
    with fits.open(fits_fp, ignoring_missing_end=True) as hdu_list:
        time = hdu_list['LIGHTCURVE'].data.TIME
        sectors_se[sector_fp.stem] = (time[0], time[-1])
# order by sector
sectors_se = {key: sectors_se[key] for key in sorted(sectors_se, key=lambda x: int(x.split('_')[1]))}

param = ('tce_time0bk', 'tce_time0bk_dv')
toi_tbl_param = toi_tbl.loc[(~toi_tbl[param[0]].isna()) & ((~toi_tbl[param[1]].isna()))]
print(f'{param[0]}: {len(toi_tbl_param)} out of {len(toi_tbl)}')
toi_tbl_param_min, toi_tbl_param_max = min(toi_tbl_param[param[0]].min(), toi_tbl_param[param[1]].min()), \
                                       max(toi_tbl_param[param[0]].max(), toi_tbl_param[param[1]].max())

f, ax = plt.subplots(figsize=(14, 14))
ax.scatter(toi_tbl_param[param[0]], toi_tbl_param[param[1]], c='b', s=8)
ax.plot(np.linspace(toi_tbl_param_min, toi_tbl_param_max, 2, endpoint=True),
        np.linspace(toi_tbl_param_min, toi_tbl_param_max, 2, endpoint=True), 'k--')
# ax.set_xlim([sectors_se['sector_1'][0], sectors_se['sector_33'][-1]])
# ax.set_ylim([sectors_se['sector_1'][0], sectors_se['sector_33'][-1]])
ax2, ax3 = ax.twinx(), ax.twiny()
xlines, ylines = [], []
for sector in sectors_se:
    xlines.extend(list(sectors_se[sector]))
    ylines.extend(list(sectors_se[sector]))
sectors_ticklabels = [f's{el.split("_")[1]}' for el in list(sectors_se.keys())]
sectors_ticks = [(xlines[i] + xlines[i + 1]) / 2 for i in range(0, len(xlines), 2)]
# sectors_ticklabels = [el for el in sectors_ticklabels for i in range(2)]
ax2.vlines(x=xlines, ymin=len(xlines) * [sectors_se['sector_1'][0]], ymax=len(xlines) * [sectors_se['sector_33'][1]], colors=len(sectors_se) * ['g', 'r'])
ax2.set_ylabel('Sectors')
ax2.set_yticks(sectors_ticks)
ax2.set_yticklabels(sectors_ticklabels)
ax3.hlines(y=ylines, xmin=len(ylines) * [sectors_se['sector_1'][0]], xmax=len(ylines) * [sectors_se['sector_33'][1]], color=len(sectors_se) * ['g', 'r'])
ax3.set_xlabel('Sectors')
ax3.set_xticks(sectors_ticks)
ax3.set_xticklabels(sectors_ticklabels)
ax.set_ylabel(param[0])
ax.set_xlabel(param[1])
f.savefig(toi_tbl_fp.parent / f'scatter_{param[0]}_sectorgrid.png')
plt.close()

#%%

def _get_frac_offset_epoch(toi):

    k_mult = (toi['tce_time0bk'] - toi['tce_time0bk_dv']) / toi['tce_period']

    return np.abs(k_mult - np.round(k_mult))

toi_tbl['epoch_offset_frac'] = toi_tbl[['tce_time0bk', 'tce_time0bk_dv', 'tce_period']].apply(_get_frac_offset_epoch, axis=1)

bins_frac = np.logspace(-10, 2, 50)  # np.linspace(0, 100, 11, endpoint=True)
bins_per = np.logspace(-10, 1, 50)
f, ax = plt.subplots(1, 2, figsize=(8, 6))
ax[0].hist(toi_tbl['epoch_offset_frac'] * 100, bins=bins_frac, edgecolor='k')
ax[0].set_yscale('log')
ax[0].set_xscale('log')
ax[0].set_xlabel('Epoch offset relative to TOI period (%)')
ax[0].set_ylabel('Number of TOIs')
ax[1].hist(toi_tbl['epoch_offset_frac'] * toi_tbl['tce_period'], bins_per, edgecolor='k')
ax[1].set_yscale('log')
ax[1].set_xlabel('Epoch offset relative to TOI period (day)')
ax[1].set_xscale('log')
f.savefig(toi_tbl_fp.parent / f'hist_epoch_offset_toi_tce.png')
plt.close()

toi_tbl['epoch_offset_frac_dur'] = toi_tbl['epoch_offset_frac'] * toi_tbl['tce_period'] / (toi_tbl['tce_duration'] / 24)
bins_frac = np.logspace(-6, 3, 50)
f, ax = plt.subplots()
ax.set_ylabel('Number of TOIs')
ax.hist(toi_tbl['epoch_offset_frac_dur'] * 100, bins_frac, edgecolor='k')
ax.set_yscale('log')
ax.set_xlabel('Epoch offset relative to TOI duration (%)')
ax.set_xscale('log')
f.savefig(toi_tbl_fp.parent / f'hist_epoch_offset_dur_toi_tce.png')
plt.close()

def local_view_binshift(ed, nbins, ntransits):
    return ed*nbins/ntransits

for err in np.array([700, 500, 250, 100, 50, 25, 10, 5]) / 100:
    print(f'Error {err}')
    print(f'Number of TOIs below threshold {(toi_tbl["epoch_offset_frac_dur"] < err).sum()} approx. local bin shift {local_view_binshift(err, 31, 5)}')
