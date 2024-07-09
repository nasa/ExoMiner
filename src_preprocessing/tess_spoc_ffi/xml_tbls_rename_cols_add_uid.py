"""
Script that processes the TCE tables with data extracted from the DV xml files:
1. Rename columns to required naming conventions.
2. Set unique ids (uids) for each TCE.
"""

# 3rd party
import pandas as pd
from pathlib import Path


MAP_DV_XML_FIELDS = {
    'catId': 'target_id',
    'planetIndexNumber': 'tce_plnt_num',

    'planetCandidate.maxSingleEventSigma': 'tce_max_sngle_ev',
    'planetCandidate.maxMultipleEventSigma': 'tce_max_mult_ev',

    'allTransitsFit.transitDepthPpm.value': 'tce_depth',
    'allTransitsFit.transitDepthPpm.uncertainty': 'tce_depth_err',
    'allTransitsFit.orbitalPeriodDays.value': 'tce_period',
    'allTransitsFit.orbitalPeriodDays.uncertainty': 'tce_period_err',
    'allTransitsFit.transitEpochBtjd.value': 'tce_time0bk',
    'allTransitsFit.transitEpochBtjd.uncertainty': 'tce_time0bk_err',
    'allTransitsFit.transitDurationHours.value': 'tce_duration',
    'allTransitsFit.transitDurationHours.uncertainty': 'tce_duration_err',
    'allTransitsFit.planetRadiusEarthRadii.value': 'tce_prad',
    'allTransitsFit.planetRadiusEarthRadii.uncertainty': 'tce_prad_err',
    # 'allTransitsFit_starRadiusSolarRadii_value': 'tce_sradius',
    # 'allTransitsFit_starRadiusSolarRadii_uncertainty': 'tce_sradius_err',
    'allTransitsFit.semiMajorAxisAu.value': 'tce_sma',
    'allTransitsFit.semiMajorAxisAu.uncertainty': 'tce_sma_err',
    'allTransitsFit.minImpactParameter.value': 'tce_impact',
    'allTransitsFit.minImpactParameter.uncertainty': 'tce_impact_err',
    'allTransitsFit.equilibriumTempKelvin.value': 'tce_eqt',
    'allTransitsFit.equilibriumTempKelvin.uncertainty': 'tce_eqt_err',
    'allTransitsFit.effectiveStellarFlux.value': 'tce_insol',
    'allTransitsFit.effectiveStellarFlux.uncertainty': 'tce_insol_err',

    # 'oddTransitsFit.transitDepthPpm.value': 'tce_depth_odd',
    # 'oddTransitsFit.transitDepthPpm.uncertainty': 'tce_depth_odd_err',
    # 'oddTransitsFit.orbitalPeriodDays.value': 'tce_period_odd',
    # 'oddTransitsFit.orbitalPeriodDays.uncertainty': 'tce_period_odd_err',
    # 'oddTransitsFit.transitEpochBtjd.value': 'tce_time0bk_odd',
    # 'oddTransitsFit.transitEpochBtjd.uncertainty': 'tce_time0bk_odd_err',
    # 'oddTransitsFit.transitDurationHours.value': 'tce_duration_odd',
    # 'oddTransitsFit.transitDurationHours.uncertainty': 'tce_duration_odd_err',
    # 'evenTransitsFit.transitDepthPpm.value': 'tce_depth_even',
    # 'evenTransitsFit.transitDepthPpm.uncertainty': 'tce_depth_even_err',
    # 'evenTransitsFit.orbitalPeriodDays.value': 'tce_period_even',
    # 'evenTransitsFit.orbitalPeriodDays.uncertainty': 'tce_period_even_err',
    # 'evenTransitsFit.transitEpochBtjd.value': 'tce_time0bk_even',
    # 'evenTransitsFit.transitEpochBtjd.uncertainty': 'tce_time0bk_even_err',
    # 'evenTransitsFit.transitDurationHours.value': 'tce_duration_even',
    # 'evenTransitsFit.transitDurationHours.uncertainty': 'tce_duration_even_err',

    # 'planetCandidate.epochTjd': 'tce_time0bk',
    # 'planetCandidate.orbitalPeriod': 'tce_period',

    'centroidResults.differenceImageMotionResults.msTicCentroidOffsets.meanSkyOffset.value': 'tce_dikco_msky',
    'centroidResults.differenceImageMotionResults.msTicCentroidOffsets.meanSkyOffset.uncertainty': 'tce_dikco_msky_err',
    'centroidResults.differenceImageMotionResults.msControlCentroidOffsets.meanSkyOffset.value': 'tce_dicco_msky',
    'centroidResults.differenceImageMotionResults.msControlCentroidOffsets.meanSkyOffset.uncertainty': 'tce_dicco_msky_err',

    'tessMag.value': 'mag',
    'tessMag.uncertainty': 'mag_err',

    'radius.value': 'tce_sradius',
    'radius.uncertainty': 'tce_sradius_err',
    'stellarDensity.value': 'tce_sdens',
    'stellarDensity.uncertainty': 'tce_sdens_err',
    'effectiveTemp.value': 'tce_steff',
    'effectiveTemp.uncertainty': 'tce_steff_err',
    'log10SurfaceGravity.value': 'tce_slogg',
    'log10SurfaceGravity.uncertainty': 'tce_slogg_err',
    'log10Metallicity.value': 'tce_smet',
    'log10Metallicity.uncertainty': 'tce_smet_err',

    'planetCandidate.weakSecondary.maxMes': 'tce_maxmes',
    'planetCandidate.weakSecondary.maxMesPhaseInDays': 'tce_maxmesd',
    'planetCandidate.weakSecondary.depthPpm.value': 'wst_depth',
    'planetCandidate.weakSecondary.depthPpm.uncertainty': 'wst_depth_err',

    'binaryDiscriminationResults.oddEvenTransitDepthComparisonStatistic.value': 'tce_bin_oedp_stat',
    'binaryDiscriminationResults.shorterPeriodComparisonStatistic.value': 'tce_shortper_stat',
    'binaryDiscriminationResults.longerPeriodComparisonStatistic.value': 'tce_longper_stat',

    'bootstrapResults.significance': 'boot_fap',

    'allTransitsFit.ratioPlanetRadiusToStarRadius.value': 'tce_ror',
    'allTransitsFit.ratioPlanetRadiusToStarRadius.uncertainty': 'tce_ror_err',
    'allTransitsFit.ratioSemiMajorAxisToStarRadius.value': 'tce_dor',
    'allTransitsFit.ratioSemiMajorAxisToStarRadius.uncertainty': 'tce_dor_err',
    'allTransitsFit.modelFitSnr': 'tce_model_snr',
    'allTransitsFit.transitIngressTimeHours.value': 'tce_ingress',
    'allTransitsFit.transitIngressTimeHours.uncertainty': 'tce_ingress_err',
    'allTransitsFit.inclinationDegrees.value': 'tce_incl',
    'allTransitsFit.inclinationDegrees.uncertainty': 'tce_incl_err',
    'allTransitsFit.eccentricity.value': 'tce_eccen',
    'allTransitsFit.eccentricity.uncertainty': 'tce_eccen_err',
    'allTransitsFit.longitudeOfPeriDegrees.value': 'tce_longp',
    'allTransitsFit.longitudeOfPeriDegrees.uncertainty': 'tce_longp_err',

    'raDegrees.value': 'ra',
    'decDegrees.value': 'dec',

    'secondaryEventResults.planetParameters.geometricAlbedo.value': 'tce_albedo',
    'secondaryEventResults.planetParameters.geometricAlbedo.uncertainty': 'tce_albedo_err',
    'secondaryEventResults.planetParameters.planetEffectiveTemp.value': 'tce_ptemp',
    'secondaryEventResults.planetParameters.planetEffectiveTemp.uncertainty': 'tce_ptemp_err',
    'secondaryEventResults.comparisonTests.albedoComparisonStatistic.value': 'tce_albedo_stat',
    'secondaryEventResults.comparisonTests.albedoComparisonStatistic.significance': 'tce_albedo_stat_err',
    'secondaryEventResults.comparisonTests.tempComparisonStatistic.value': 'tce_ptemp_stat',
    'secondaryEventResults.comparisonTests.tempComparisonStatistic.significance': 'tce_ptemp_stat_err',

    'ghostDiagnosticResults.coreApertureCorrelationStatistic.value': 'tce_cap_stat',
    'ghostDiagnosticResults.coreApertureCorrelationStatistic.significance': 'tce_cap_stat_err',
    'ghostDiagnosticResults.haloApertureCorrelationStatistic.value': 'tce_hap_stat',
    'ghostDiagnosticResults.haloApertureCorrelationStatistic.significance': 'tce_hap_stat_err',

    'planetCandidate.observedTransitCount': 'tce_num_transits_obs',
    'planetCandidate.expectedTransitCount': 'tce_num_transits',

    # 'allTransitsFit_starDensitySolarDensity_value': 'tce_sdens',
    # 'allTransitsFit_starDensitySolarDensity_uncertainty': 'tce_sdens_err',

    'allTransitsFit.modelChiSquare': 'tce_model_chisq',
    'planetCandidate.robustStatistic': 'tce_robstat',

}

# directory with TCE tables with extracted data from the DV xml files
src_tbls_dir = Path('/data5/tess_project/Data/Ephemeris_tables/TESS/dv_spoc_ffi/preprocessed_tce_tables/07-03-2024_1311_from_dv_xml/')
dest_dir = src_tbls_dir.parent / f'{src_tbls_dir.name}_renamed_cols_added_uid'
dest_dir.mkdir(exist_ok=True)

src_tbls_fps = [fp for fp in src_tbls_dir.iterdir() if fp.suffix == '.csv']
print(f'Found {len(src_tbls_fps)} TCE tables.')

for src_tbl_fp in src_tbls_fps:
    print(f'Processing table {src_tbl_fp.name}...')
    src_tbl = pd.read_csv(src_tbl_fp)
    # rename columns
    src_tbl.rename(columns=MAP_DV_XML_FIELDS, inplace=True, errors='raise')
    # set uid
    src_tbl['uid'] = src_tbl.apply(lambda x: '{}-{}-S{}'.format(x['target_id'], x['tce_plnt_num'],
                                                                x['sector_run']), axis=1)

    src_tbl.set_index('uid', inplace=True)  # set uid as index
    src_tbl.to_csv(dest_dir / f'{src_tbl_fp.name}', index=True)

#%%

# sector_runs_lst = []
# for src_tbl_fp in src_tbls_fps:
#     sector_runs_lst.append(int(src_tbl_fp.stem[-2:]))
#
# import numpy as np
#
# a = np.setdiff1d(np.arange(36, 70), sector_runs_lst)
# print(a)