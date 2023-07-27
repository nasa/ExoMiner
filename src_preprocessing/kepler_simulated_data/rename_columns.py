"""
- Rename columns in DV TCE table.
- Cast target id and tce_plnt_num to integer.
- Set labels based on simulated groups.
"""

# 3rd party
import pandas as pd

#%% load TCE table

tce_tbl = pd.read_csv('/Users/msaragoc/Projects/exoplanet_transit_classification/data/ephemeris_tables/kepler/q1-q17_dr25/simulated_data/dvOutputMatrix_allruns_updtstellar.csv')

#%% choose columns to rename in the TCE table

fields_to_add = {
    'keplerId': 'target_id',
    'planetIndexNumber': 'tce_plnt_num',

    'planetCandidate.maxSingleEventSigma': 'tce_max_sngle_ev',
    'planetCandidate.maxMultipleEventSigma': 'tce_max_mult_ev',

    'allTransitsFit_transitDepthPpm_value': 'tce_depth',
    'allTransitsFit_transitDepthPpm_uncertainty': 'tce_depth_err',

    # ephemerides
    'allTransitsFit_orbitalPeriodDays_value': 'tce_period',
    'allTransitsFit_orbitalPeriodDays_uncertainty': 'tce_period_err',
    'allTransitsFit_transitEpochBkjd_value': 'tce_time0bk',
    'allTransitsFit_transitEpochBkjd_uncertainty': 'tce_time0bk_err',
    'allTransitsFit_transitDurationHours_value': 'tce_duration',
    'allTransitsFit_transitDurationHours_uncertainty': 'tce_duration_err',

    # transit fit
    'allTransitsFit_planetRadiusEarthRadii_value': 'tce_prad',
    'allTransitsFit_planetRadiusEarthRadii_uncertainty': 'tce_prad_err',
    # 'allTransitsFit_starRadiusSolarRadii_value': 'tce_sradius',
    # 'allTransitsFit_starRadiusSolarRadii_uncertainty': 'tce_sradius_err',
    'allTransitsFit_semiMajorAxisAu_value': 'tce_sma',
    'allTransitsFit_semiMajorAxisAu_uncertainty': 'tce_sma_err',
    'allTransitsFit_minImpactParameter_value': 'tce_impact',
    'allTransitsFit_minImpactParameter_uncertainty': 'tce_impact_err',

    # odd and even transit fit
    'oddTransitsFit_transitDepthPpm_value': 'tce_depth_odd',
    'oddTransitsFit_transitDepthPpm_uncertainty': 'tce_depth_odd_err',
    'oddTransitsFit_orbitalPeriodDays_value': 'tce_period_odd',
    'oddTransitsFit_orbitalPeriodDays_uncertainty': 'tce_period_odd_err',
    'oddTransitsFit_transitEpochBkjd_value': 'tce_time0bk_odd',
    'oddTransitsFit_transitEpochBkjd_uncertainty': 'tce_time0bk_odd_err',
    'oddTransitsFit_transitDurationHours_value': 'tce_duration_odd',
    'oddTransitsFit_transitDurationHours_uncertainty': 'tce_duration_odd_err',
    'evenTransitsFit_transitDepthPpm_value': 'tce_depth_even',
    'evenTransitsFit_transitDepthPpm_uncertainty': 'tce_depth_even_err',
    'evenTransitsFit_orbitalPeriodDays_value': 'tce_period_even',
    'evenTransitsFit_orbitalPeriodDays_uncertainty': 'tce_period_even_err',
    'evenTransitsFit_transitEpochBkjd_value': 'tce_time0bk_even',
    'evenTransitsFit_transitEpochBkjd_uncertainty': 'tce_time0bk_even_err',
    'evenTransitsFit_transitDurationHours_value': 'tce_duration_even',
    'evenTransitsFit_transitDurationHours_uncertainty': 'tce_duration_even_err',

    # 'planetCandidate.epochTjd': 'tce_time0bk',
    # 'planetCandidate.orbitalPeriod': 'tce_period',

    'allTransitsFit_effectiveStellarFlux_value': 'tce_insol',
    'allTransitsFit_effectiveStellarFlux_uncertainty': 'tce_insol_err',
    'allTransitsFit_equilibriumTempKelvin_value': 'tce_eqt',
    'allTransitsFit_equilibriumTempKelvin_uncertainty': 'tce_eqt_err',
    'allTransitsFit.modelChiSquare': 'tce_model_chisq',
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

    # Kepler magnitude was not found in DV TCE table
    # 'tessMag_value': 'mag',
    # 'tessMag_uncertainty': 'mag_err',
    # we have already a better set of stellar parameters
    # 'radius.value': 'tce_sradius',
    # 'radius_uncertainty': 'tce_sradius_err',
    # 'stellarDensity_value': 'tce_sdens',
    # 'stellarDensity_uncertainty': 'tce_sdens_err',
    # 'effectiveTemp.value': 'tce_steff',
    # 'effectiveTemp_uncertainty': 'tce_steff_err',
    # 'log10SurfaceGravity.value': 'tce_slogg',
    # 'log10SurfaceGravity_uncertainty': 'tce_slogg_err',
    # 'log10Metallicity.value': 'tce_smet',
    # 'log10Metallicity_uncertainty': 'tce_smet_err',
    # 'allTransitsFit_starDensitySolarDensity_value': 'tce_sdens',
    # 'allTransitsFit_starDensitySolarDensity_uncertainty': 'tce_sdens_err'

    # odd and even statistics
    'binaryDiscriminationResults.oddEvenTransitDepthComparisonStatistic.value': 'tce_bin_oedp_stat',

    'binaryDiscriminationResults.shorterPeriodComparisonStatistic.value': 'tce_shortper_stat',
    'binaryDiscriminationResults.longerPeriodComparisonStatistic.value': 'tce_longper_stat',

    'bootstrap_falseAlarmRate': 'boot_fap',

    # no RA and Dec found in DV TCE table
    # 'raDegrees_value': 'ra',
    # 'decDegrees_value': 'dec',

    # secondary results
    'planetCandidate.weakSecondaryStruct.maxMes': 'tce_maxmes',
    'planetCandidate.weakSecondaryStruct.maxMesPhaseInDays': 'tce_maxmesd',
    'planetCandidate.weakSecondaryStruct.depthPpm.value': 'wst_depth',
    'planetCandidate.weakSecondaryStruct.depthPpm.uncertainty': 'wst_depth_err',
    'secondaryEventResults.planetParameters.geometricAlbedo.value': 'tce_albedo',
    'secondaryEventResults.planetParameters.geometricAlbedo.uncertainty': 'tce_albedo_err',
    'secondaryEventResults.planetParameters.planetEffectiveTemp.value': 'tce_ptemp',
    'secondaryEventResults.planetParameters.planetEffectiveTemp.uncertainty': 'tce_ptemp_err',
    'secondaryEventResults.comparisonTests.albedoComparisonStatistic.value': 'tce_albedo_stat',
    'secondaryEventResults.comparisonTests.albedoComparisonStatistic.significance': 'tce_albedo_stat_err',
    'secondaryEventResults.comparisonTests.tempComparisonStatistic.value': 'tce_ptemp_stat',
    'secondaryEventResults.comparisonTests.tempComparisonStatistic.significance': 'tce_ptemp_stat_err',

    # centroid-related results
    # 'centroidResults.prfMotionResults.motionDetectionStatistic.significance': '',
    'centroidResults.fluxWeightedMotionResults.motionDetectionStatistic.significance': 'tce_fwm_stat',
    'centroidResults.differenceImageMotionResults.mqKicCentroidOffsets.meanSkyOffset.value': 'tce_dikco_msky',
    'centroidResults.differenceImageMotionResults.mqKicCentroidOffsets.meanSkyOffset.uncertainty': 'tce_dikco_msky_err',
    'centroidResults.differenceImageMotionResults.mqControlCentroidOffsets.meanSkyOffset.value': 'tce_dicco_msky',
    'centroidResults.differenceImageMotionResults.mqControlCentroidOffsets.meanSkyOffset.uncertainty': 'tce_dicco_msky_err',

    # ghost diagnostic
    'ghostDiagnosticResults.coreApertureCorrelationStatistic_value': 'tce_cap_stat',
    'ghostDiagnosticResults.coreApertureCorrelationStatistic_significance': 'tce_cap_stat_err',
    'ghostDiagnosticResults.haloApertureCorrelationStatistic_value': 'tce_hap_stat',
    'ghostDiagnosticResults.haloApertureCorrelationStatistic_significance': 'tce_hap_stat_err',
    'planetCandidate.observedTransitCount': 'tce_num_transits_obs',
    'planetCandidate.expectedTransitCount': 'tce_num_transits',

}

# check which columns were not found
for col in fields_to_add:
    if col not in tce_tbl:
        print(f'{col} not found in TCE table.')

tce_tbl = tce_tbl.rename(columns=fields_to_add)

#%% change data type for columns
tce_tbl = tce_tbl.astype({'target_id': int, 'tce_plnt_num': int})

#%% set labels to dataset name

tce_tbl['label'] = tce_tbl['dataset']

#%% save updated table

tce_tbl.to_csv('/Users/msaragoc/Projects/exoplanet_transit_classification/data/ephemeris_tables/kepler/q1-q17_dr25/simulated_data/dvOutputMatrix_allruns_updtstellar_final_7-24-2023_1548.csv', index=False)
