""" Create TESS TCE table based on TESS DV TCE tables, TOI catalog and TOI-TCE matching table. """

# 3rd party
import logging
import multiprocessing
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# local
from data_wrangling.tess.tess_dvmat_tables_utils import match_tces_to_tois, _map_to_sector_string

if __name__ == '__main__':

    # results directory
    res_dir = Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/data/ephemeris_matching/toi_tce_matching/07-18-2022_1205')
    res_dir.mkdir(exist_ok=True)

    # CHECK INPUT TABLES
    toi_tbl_fp = Path(
        '/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/data/ephemeris_tables/tess/EXOFOP_TOI_lists/TOI/7-11-2022/exofop_toilists_nomissingpephem.csv')
    # TOI data to be added to the TCE table
    toi_cols = ['TOI', 'TESS Disposition', 'TFOPWG Disposition', 'Period (days)', 'Duration (hours)', 'Depth (ppm)',
                'Epoch (TBJD)', 'Epoch (BJD)', 'Sectors', 'Comments']
    matching_tbl_fp = '/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/data/ephemeris_matching/toi_tce_matching/07-18-2022_1205/tois_matchedtces_ephmerismatching_thrinf_samplint0.00.csv'

    # set up logger
    logger = logging.getLogger(name='match_tces_to_tois')
    logger_handler = logging.FileHandler(filename=res_dir / 'match_tces_to_tois.log', mode='w')
    logger_formatter = logging.Formatter('%(asctime)s - %(message)s')
    logger.setLevel(logging.INFO)
    logger_handler.setFormatter(logger_formatter)
    logger.addHandler(logger_handler)
    logger.info(f'Starting run...')

    # get TCE tables
    tce_root_dir = Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/data/ephemeris_tables/tess/DV_SPOC_mat_files')
    logger.info(f'TCE tables directory: {tce_root_dir}')

    multisector_tce_dir = tce_root_dir / 'multi-sector' / 'csv_tables'
    singlesector_tce_dir = tce_root_dir / 'single-sector' / 'csv_tables'

    sector_tce_tbls = {str(int(file.stem[14:16])): pd.read_csv(file) for file in singlesector_tce_dir.iterdir()}
    sector_tce_tbls.update({f'{int(file.stem[14:16])}-{int(file.stem[16:18])}': pd.read_csv(file)
                            for file in multisector_tce_dir.iterdir()})

    # get TOI-TCE matching table
    matching_tbl = pd.read_csv(matching_tbl_fp)
    logger.info(f'Matching TOI-TCE table: {matching_tbl_fp}')

    # remove TOIs not matched to any TCE
    logger.info(f'Removing TOIs not matched to any TCE: {matching_tbl["Matched TCEs"].isna().sum()}')
    matching_tbl = matching_tbl.loc[~matching_tbl['Matched TCEs'].isna()]
    logger.info(f'Number of TOIs before thresholding the matching distance to the closest TCE: {len(matching_tbl)}')

    # remove TOIs whose closest TCE matching distance is above the matching threshold
    # matching_thr = 0.25
    # matching_tbl = matching_tbl.loc[matching_tbl['matching_dist_0'] <= matching_thr]
    # print(f'Number of TOIs after thresholding (thr={matching_thr}) the matching distance to the closest TCE: '
    #       f'{len(matching_tbl)}')

    # get TOI table
    toi_tbl = pd.read_csv(toi_tbl_fp)
    logger.info(f'TOI table: {toi_tbl_fp}')
    logger.info(f'Total number of TOIs: {len(toi_tbl)}')

    logger.info(toi_tbl['TESS Disposition'].value_counts())
    logger.info(toi_tbl['TFOPWG Disposition'].value_counts())

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
        # 'allTransitsFit_starDensitySolarDensity_uncertainty': 'tce_sdens_err'
    }

    for sector_tce_tbl in sector_tce_tbls:
        sector_tce_tbls[sector_tce_tbl].rename(columns=fields_to_add, inplace=True)

    # initialize TCE table
    toi_tbl.reset_index(inplace=True)

    tce_tbl_cols = list(fields_to_add.values()) + toi_cols + ['tce_sectors', 'match_dist']

    n_processes = 4
    pool = multiprocessing.Pool(processes=n_processes)
    jobs = [(tce_tbl_cols, sector, sector_tce_tbls[sector], toi_tbl[toi_cols], matching_tbl)
            for sector_tce_tbl_i, sector in enumerate(sector_tce_tbls)]
    async_results = [pool.apply_async(match_tces_to_tois, job) for job in jobs]
    pool.close()

    tce_tbl = pd.concat([async_result.get() for async_result in async_results], axis=0, ignore_index=True)

    logger.info(f'Number of TCEs matched to TOIs: {(~tce_tbl["match_dist"].isna()).sum()}')

    logger.info(tce_tbl['TESS Disposition'].value_counts())
    logger.info(tce_tbl['TFOPWG Disposition'].value_counts())

    # rename columns
    tce_tbl.rename(columns={'Sectors': 'toi_sectors', 'tce_sectors': 'sectors'}, inplace=True)

    # change data type for columns
    tce_tbl = tce_tbl.astype({'target_id': np.int, 'tce_plnt_num': np.int, 'sectors': np.str})

    tce_tbl['sector_run'] = tce_tbl['sectors']

    # map sectors to expected format
    tce_tbl['sectors'] = tce_tbl[['sectors']].apply(_map_to_sector_string, axis=1)

    tce_tbl_fp = Path(res_dir / f'tess_tces_s1-s52_{res_dir.name}.csv')
    tce_tbl.to_csv(tce_tbl_fp, index=False)

    # plot histogram of matching distances
    bins = np.linspace(0, 1, 21, endpoint=True)
    f, ax = plt.subplots(1, 2, figsize=(12, 8))
    ax[0].hist(tce_tbl['match_dist'], bins=bins, edgecolor='k')
    ax[0].set_ylabel('Counts')
    ax[0].set_yscale('log')
    ax[0].set_xlabel('Matching distance to closest TOI')
    ax[0].set_xlim([bins[0], bins[-1]])
    ax[1].hist(tce_tbl['match_dist'], bins=bins, edgecolor='k', cumulative=True)
    ax[1].set_ylabel('Cumulative Counts')
    # ax[1].set_yscale('log')
    ax[1].set_xlabel('Matching distance to closest TOI')
    ax[1].set_xlim([bins[0], bins[-1]])
    f.savefig(res_dir / 'hist_after_match_dist_0.png')
