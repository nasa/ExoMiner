""" Create TESS TCE table from DV SPOC TCE mat tables. """

# 3rd party
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime

# results directory
root_dir = Path('/data5/tess_project/Data/Ephemeris_tables/TESS/dv_spoc_ffi/')
res_dir = Path('/data5/tess_project/Data/Ephemeris_tables/TESS/dv_spoc_ffi/preprocessed_tce_tables/07-08-2024_1616_s47-s69_from_mat_files')
tce_tbl_name = 'tess_spoc_ffi_tces_dv_s47-s69.csv'

res_dir.mkdir(exist_ok=True)

# set up logger
logger = logging.getLogger(name='create_tess_tce_tbl_from_dv')
logger_handler = logging.FileHandler(filename=res_dir / 'create_tess_tce_tbl_from_dv.log', mode='w')
logger_formatter = logging.Formatter('%(asctime)s - %(message)s')
logger.setLevel(logging.INFO)
logger_handler.setFormatter(logger_formatter)
logger.addHandler(logger_handler)
logger.info(f'Starting run...')

# get TCE tables
# multisector_tce_dir = root_dir / 'multi-sector' / 'csv_tables'
singlesector_tce_dir = root_dir / 'single_sector_runs' / 'csv_tables'

logger.info('Loading DV SPOC TCE tables for the multiple single- and multi-sector runs...')
sector_tce_tbls = {file.stem.split('_')[2][1:]: pd.read_csv(file) for file in singlesector_tce_dir.iterdir() if file.suffix == '.csv' and not file.match('dvOutputMatrix_FFI_S7*.csv')}
# sector_tce_tbls.update({f'{int(file.stem[14:16])}-{int(file.stem[16:18])}': pd.read_csv(file)
#                         for file in multisector_tce_dir.iterdir()})
logger.info(f'{len(sector_tce_tbls)} DV SPOC TCE tables loaded:')
for sector_tce_tbls_name in sector_tce_tbls:
    logger.info(f'{sector_tce_tbls_name}: {len(sector_tce_tbls[sector_tce_tbls_name])} TCEs.')

for sector_tce_tbls_name in sector_tce_tbls:
    sector_tce_tbls[sector_tce_tbls_name]['sector_run'] = sector_tce_tbls_name

tce_tbl = pd.concat([sector_tce_tbls[sector_tce_tbl_name] for sector_tce_tbl_name in sector_tce_tbls], axis=0,
                    ignore_index=True)

logger.info(f'Total number of TCEs: {len(tce_tbl)}')

# choose columns to rename in the TCE tables
logger.info('Renaming columns...')
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
    'stellarDensity_uncertainty': 'tce_sdens_err',
    'effectiveTemp_value': 'tce_steff',
    'effectiveTemp_uncertainty': 'tce_steff_err',
    'log10SurfaceGravity_value': 'tce_slogg',
    'log10SurfaceGravity_uncertainty': 'tce_slogg_err',
    'log10Metallicity_value': 'tce_smet',
    'log10Metallicity_uncertainty': 'tce_smet_err',
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
    'secondaryEventResults.comparisonTests.albedoComparisonStatistic_significance': 'tce_albedo_stat_err',
    'secondaryEventResults.comparisonTests.tempComparisonStatistic_value': 'tce_ptemp_stat',
    'secondaryEventResults.comparisonTests.tempComparisonStatistic_significance': 'tce_ptemp_stat_err',
    'allTransitsFit_effectiveStellarFlux_value': 'tce_insol',
    'allTransitsFit_effectiveStellarFlux_uncertainty': 'tce_insol_err',
    'ghostDiagnosticResults.coreApertureCorrelationStatistic_value': 'tce_cap_stat',
    'ghostDiagnosticResults.coreApertureCorrelationStatistic_significance': 'tce_cap_stat_err',
    'ghostDiagnosticResults.haloApertureCorrelationStatistic_value': 'tce_hap_stat',
    'ghostDiagnosticResults.haloApertureCorrelationStatistic_significance': 'tce_hap_stat_err',
    'planetCandidate.observedTransitCount': 'tce_num_transits_obs',
    'planetCandidate.expectedTransitCount': 'tce_num_transits',
    'planetCandidate.weakSecondaryStruct.depthPpm_value': 'wst_depth',
    'planetCandidate.weakSecondaryStruct.depthPpm_uncertainty': 'wst_depth_err',
    # 'allTransitsFit_starDensitySolarDensity_value': 'tce_sdens',
    # 'allTransitsFit_starDensitySolarDensity_uncertainty': 'tce_sdens_err',
    'allTransitsFit.modelChiSquare': 'tce_model_chisq',
    'planetCandidate.robustStatistic': 'tce_robstat',

}
logger.info(f'Fields renamed:')
for old_field_name, new_field_name in fields_to_add.items():
    logger.info(f'{old_field_name} -> {new_field_name}')

tce_tbl.rename(columns=fields_to_add, inplace=True)

tce_tbl = tce_tbl.astype({'target_id': int, 'tce_plnt_num': int})
tce_tbl['uid'] = tce_tbl.apply(lambda x: '{}-{}-S{}'.format(x['target_id'], x['tce_plnt_num'], x['sector_run']), axis=1)

tce_tbl.to_csv(res_dir / tce_tbl_name, index=False)
logger.info(f'Saved TCE table {tce_tbl_name} to {res_dir}.')
