from pathlib import Path
import numpy as np
import pandas as pd
from scipy.io import loadmat
from tqdm import tqdm
import copy

#%% Create csv DV TCE files from the original mat files

dv_root_dir = Path('/data5/tess_project/Data/Ephemeris_tables/TESS/DV_SPOC_mat_files')

ssector_runs_dir = dv_root_dir / 'single-sector'
msector_runs_dir = dv_root_dir / 'multi-sector'

mat_runs_fps = list(ssector_runs_dir.iterdir()) + list(msector_runs_dir.iterdir())

for mat_run_fp in mat_runs_fps:

    print(f'Creating csv table for mat file {mat_run_fp.name}')

    csv_save_dir = mat_run_fp.parent / 'csv_tables'
    csv_save_dir.mkdir(exist_ok=True)

    mat_struct = loadmat(mat_run_fp)

    cols = []
    cnt_no_name_col = 0
    for el_i, el in enumerate(mat_struct['dvOutputMatrixColumns']):

        try:
            cols.append(el[0][0])
        except:
            # print('EXCEPT', el[0])
            cols.append(f'no_name_{cnt_no_name_col}')
            cnt_no_name_col += 1

    num_tces = len(mat_struct['dvOutputMatrix'])

    tce_tbl = pd.DataFrame(columns=cols, data=np.nan * np.ones((num_tces, len(cols))))

    for tce_i, tce in enumerate(mat_struct['dvOutputMatrix']):
        tce_tbl.loc[tce_i] = tce

    tce_tbl.to_csv(csv_save_dir / f'{mat_run_fp.stem}.csv', index=False)

#%% match TCEs to TOIs

# results directory
res_dir = Path('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Analysis/toi_tce_matching/tce_match_toi_5-5-2021')
res_dir.mkdir(exist_ok=True)

# get TCE tables
tce_root_dir = Path('/data5/tess_project/Data/Ephemeris_tables/TESS/DV_SPOC_mat_files')

multisector_tce_dir = tce_root_dir / 'multi-sector' / 'csv_tables'
singlesector_tce_dir = tce_root_dir / 'single-sector' / 'csv_tables'

sector_tce_tbls = {str(int(file.stem[14:16])): pd.read_csv(file) for file in singlesector_tce_dir.iterdir()}
sector_tce_tbls.update({f'{int(file.stem[14:16])}-{int(file.stem[16:18])}': pd.read_csv(file)
                        for file in multisector_tce_dir.iterdir()})

# get TOI-TCE matching table
matching_tbl = pd.read_csv('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Analysis/toi_tce_matching/04-16-2021_1625/'
                           'tois_matchedtces_ephmerismatching_thrinf_samplint1e-05.csv')

# remove TOIs not matched to any TCE
matching_tbl = matching_tbl.loc[~matching_tbl['Matched TCEs'].isna()]
print(f'Number of TOIs before thresholding the matching distance to the closest TCE: {len(matching_tbl)}')

# remove TOIs whose closest TCE matching distance is above the matching threshold
# matching_thr = 0.25
# matching_tbl = matching_tbl.loc[matching_tbl['matching_dist_0'] <= matching_thr]
# print(f'Number of TOIs after thresholding (thr={matching_thr}) the matching distance to the closest TCE: '
#       f'{len(matching_tbl)}')

# get TOI table
toi_tbl_fp = Path('/data5/tess_project/Data/Ephemeris_tables/TESS/EXOFOP_TOI_lists/TOI/4-22-2021/exofop_toilists_nomissingpephem.csv')
toi_tbl = pd.read_csv(toi_tbl_fp)
print(f'Total number of TOIs: {len(toi_tbl)}')

print(toi_tbl['TESS Disposition'].value_counts())
print(toi_tbl['TFOPWG Disposition'].value_counts())

# choose columns to rename in the TCE tables
fields_to_add = {
    'catId': 'target_id',
    'planetIndexNumber': 'tce_plnt_num',
    'planetCandidate.maxSingleEventSigma': 'tce_max_sngle_ev',
    'planetCandidate.maxMultipleEventSigma': 'tce_max_mult_ev',
    'allTransitsFit_transitDepthPpm_value': 'tce_depth',
    'allTransitsFit_transitDepthPpm_uncertainty': 'tce_depth_err',
    'allTransitsFit_orbitalPeriodDays_value': 'tce_period',
    'allTransitsFit_orbitalPeriodDays_uncertainty': 'tce_period_err',
    'allTransitsFit_transitEpochBtjd_value': 'tce_time0bk',
    'allTransitsFit_transitEpochBtjd_uncertainty': 'tce_time0bk_err',
    'allTransitsFit_transitDurationHours_value': 'tce_duration',
    'allTransitsFit_transitDurationHours_uncertainty': 'tce_duration_err',
    'allTransitsFit_planetRadiusEarthRadii_value': 'tce_prad',
    'allTransitsFit_planetRadiusEarthRadii_uncertainty': 'tce_prad_err',
    # 'allTransitsFit_starRadiusSolarRadii_value': 'tce_sradius',
    # 'allTransitsFit_starRadiusSolarRadii_uncertainty': 'tce_sradius_err',
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
    'tessMag_value': 'mag',
    'tessMag_uncertainty': 'mag_err',
    'radius_value': 'tce_sradius',
    'radius_uncertainty': 'tce_sradius_err',
    'stellarDensity_value': 'tce_sdens',
    'stellarDensity_uncertainty': 'tce_dens_err',
    'effectiveTemp_value': 'tce_steff',
    'effectiveTemp_uncertainty': 'tce_steff_err',
    'log10SurfaceGravity_value': 'tce_slogg',
    'log10SurfaceGravity_uncertainty': 'tce_slogg_err',
    'log10Metallicity_value': 'tce_smet',
    'log10Metallicity_uncertainty': 'tce_met_err',
    'allTransitsFit_equilibriumTempKelvin_value': 'tce_eqt',
    'allTransitsFit_equilibriumTempKelvin_uncertainty': 'tce_eqt_err',
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
    'allTransitsFit.modelFitSnr': 'tce_model_snr',
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
    'allTransitsFit_effectiveStellarFlux_value': 'tce_insol',
    'allTransitsFit_effectiveStellarFlux_uncertainty': 'tce_insol_err',
    'ghostDiagnosticResults.coreApertureCorrelationStatistic_value': 'tce_cap_stat',
    'ghostDiagnosticResults.haloApertureCorrelationStatistic_value': 'tce_hap_stat',
    'planetCandidate.observedTransitCount': 'tce_num_transits_obs',
    'planetCandidate.expectedTransitCount': 'tce_num_transits',
    'planetCandidate.weakSecondaryStruct.depthPpm_value': 'wst_depth',
    'planetCandidate.weakSecondaryStruct.depthPpm_uncertainty': 'wst_depth_err',
    # 'allTransitsFit_starDensitySolarDensity_value': 'tce_sdens',
    # 'allTransitsFit_starDensitySolarDensity_uncertainty': 'tce_sdens_err'
}

for sector_tce_tbl in sector_tce_tbls:
    sector_tce_tbls[sector_tce_tbl].rename(columns=fields_to_add, inplace=True)

# initialize TCE table
tce_tbl_cols = list(fields_to_add.values())
toi_cols = ['TOI', 'TESS Disposition', 'TFOPWG Disposition', 'Period (days)', 'Duration (hours)', 'Depth (ppm)',
            'Epoch (TBJD)', 'Epoch (BJD)', 'Sectors', 'Comments']
tce_tbl = pd.DataFrame(columns=list(fields_to_add.values()) + toi_cols + ['tce_sectors', 'match_dist'])

# match_thr = np.inf  # inf does not work if there are NaN distances

toi_tbl.reset_index(inplace=True)
tces_not_matched = []
for toi_i, toi in tqdm(matching_tbl.iterrows()):

    if toi_i % 50 == 0:
        print(f'Iterated through {toi_i}/{len(matching_tbl)} TOIs...\n{len(tce_tbl)} TCEs added to the table.')

    toi_disp = toi_tbl.loc[toi_tbl['TOI'] == toi['TOI ID'], toi_cols]

    # list of TCEs matched to the TOI; force only one TCE match per run
    tces_seen = []

    matched_tces = toi['Matched TCEs'].split(' ')
    target_id = toi['TIC']

    # do not match TCE if the matching distance is aboce the matching threshold
    for matched_tce_i in range(len(matched_tces)):
        # if toi[f'matching_dist_{matched_tce_i}'] > match_thr:
        #     break

        sector_str, tce_plnt_num = matched_tces[matched_tce_i].split('_')
        tce_plnt_num = int(tce_plnt_num.split('.')[0])

        tce = copy.deepcopy(sector_tce_tbls[sector_str].loc[(sector_tce_tbls[sector_str]['target_id'] == target_id) &
                                                            (sector_tce_tbls[sector_str]['tce_plnt_num'] ==
                                                             tce_plnt_num)])

        # if '-' not in sector_str:
        tce['tce_sectors'] = sector_str
        # else:
        #     s_sector, e_sector = sector_str.split('-')
        #     multi_sector = [str(el) for el in list(range(int(s_sector), int(e_sector) + 1))]
        #     tce['tce_sectors'] = ' '.join(multi_sector)

        # do not match TCE if a TCE from the same run was previously matched
        if tce['tce_sectors'].values[0] in tces_seen:
            tces_not_matched.append((f'{tce["target_id"]}.{tce["tce_plnt_num"]}', sector_str))
            continue
        else:
            tces_seen.append(tce['tce_sectors'].values[0])

        tce['match_dist'] = toi[f'matching_dist_{matched_tce_i}']

        # add data from matched TOI
        for col in toi_disp:
            tce[col] = toi_disp[col].item()

        # add TCE to TCE table
        tce_tbl = pd.concat([tce_tbl, tce[tce_tbl.columns]], axis=0, ignore_index=True)

        # aaaa

print(toi_tbl['TESS Disposition'].value_counts())
print(toi_tbl['TFOPWG Disposition'].value_counts())

tce_tbl_fp = Path(res_dir / f'tess_tces_s1-s35.csv')
tce_tbl.to_csv(tce_tbl_fp, index=False)
