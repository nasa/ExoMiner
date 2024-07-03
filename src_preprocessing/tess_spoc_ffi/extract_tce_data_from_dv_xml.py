"""
Extracts TCE data from DV xml files from TESS SPOC FFI DV XML files into csv files. Extracts data per target and sector
run.
"""

# 3rd party
import xml.etree.cElementTree as et
import pandas as pd
import numpy as np
from pathlib import Path
import multiprocessing

DV_XML_HEADER = '{http://www.nasa.gov/2018/TESS/DV}'

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


def append_attributes(descendants, tce_i, output_csv, keywords):
    """ Appends attributes of a given parameter. Helper to process_xml().

    Args:
        descendants: elem, given location within a processed .xml file. Will include the key value we are currently
        sitting at
        tce_i: int, current tce index that we are on to log which position we should output the attribute to within the
        csv file
        output_csv: pandas DataFrame, contains extracted data so far
        keywords: list, keywords that have been logged when traversing the tree of the .xml file. Necessary for naming
        a designated column in the final table

    Returns:

    """

    # get topmost values from given descendant
    attributes = descendants.attrib

    for cur_att in attributes:

        if "modelParameter" not in cur_att:
            keywords.append(cur_att)

        out_col_name = '.'.join(keywords)

        if cur_att == 'suspectedEclipsingBinary':
            output_csv.at[tce_i, out_col_name] = int(attributes[cur_att] == 'true')
        else:
            output_csv.at[tce_i, out_col_name] = attributes[cur_att]

        # we checked the attribute, backtrack from it
        keywords.pop()


def get_allTransitsFit(planet_res, tce_i, output_csv):
    """ Function designed specifically for getting the 'allTransitsFit' parameters.

    Hardcoded variables are designated to ensure that they are outputted correctly with desired labels to the final csv.

    Args:
        planet_res: elem, current location within the given XML file along the tree
        tce_i: int, current TCE index to properly output the current attributes to a proper
        output_csv: pandas DataFrame, contains extracted data so far

    Returns:

    """

    # loop through sub values within modelParameters
    planet_res_params = planet_res[0].findall(".//")[1]

    # loop over each attribute (longitude, etc.)
    for model_param in planet_res_params:
        # obtain the uncertainty and the value of each parameter
        unc_col = model_param.attrib['uncertainty']
        val_col = model_param.attrib['value']
        fitted_col = model_param.attrib['fitted']

        param_name = model_param.attrib['name']
        output_csv.at[tce_i, f'allTransitsFit.{param_name}.uncertainty'] = unc_col

        output_csv.at[tce_i, f'allTransitsFit.{param_name}.value'] = val_col
        output_csv.at[tce_i, f'allTransitsFit.{param_name}.fitted'] = fitted_col

    # meanskyoffset
    dikco_msky_val = planet_res[3][0][0][2].attrib['value']
    dikco_msky_unc = planet_res[3][0][0][2].attrib['uncertainty']
    output_csv.at[
        tce_i, 'centroidResults.differenceImageMotionResults.msTicCentroidOffsets.meanSkyOffset.value'] = dikco_msky_val
    output_csv.at[
        tce_i, 'centroidResults.differenceImageMotionResults.msTicCentroidOffsets.meanSkyOffset.uncertainty'] = dikco_msky_unc

    dicco_msky_val = planet_res[3][0][1][2].attrib['value']
    dicco_msky_unc = planet_res[3][0][1][2].attrib['uncertainty']
    output_csv.at[
        tce_i, 'centroidResults.differenceImageMotionResults.msControlCentroidOffsets.meanSkyOffset.value'] = dicco_msky_val
    output_csv.at[
        tce_i, 'centroidResults.differenceImageMotionResults.msControlCentroidOffsets.meanSkyOffset.uncertainty'] = dicco_msky_unc


def get_outside_params(root, output_csv):
    """ Hard-coded method for grabbing the attributes not nested within "descendants" or sub-levels within the tree of
    the xml file that need to grab.

    Args:
        root: root elem, the topmost level of the given xml file
        output_csv: pandas DataFrame, contains extracted data so far

    Returns:

    """

    # decDegrees
    decDegrees_val = root.find(f'{DV_XML_HEADER}decDegrees').attrib['value']
    decDegrees_unc = root.find(f'{DV_XML_HEADER}decDegrees').attrib['uncertainty']
    output_csv['decDegrees.value'] = decDegrees_val
    output_csv['decDegrees.uncertainty'] = decDegrees_unc

    # retrieve effective temp
    tce_steff = root.find(f'{DV_XML_HEADER}effectiveTemp').attrib['value']
    tce_steff_unc = root.find(f'{DV_XML_HEADER}effectiveTemp').attrib['uncertainty']
    output_csv['effectiveTemp.value'] = tce_steff
    output_csv['effectiveTemp.uncertainty'] = tce_steff_unc

    # limbDarkeningModel
    limbDark_val = root.find(f'{DV_XML_HEADER}limbDarkeningModel').attrib['modelName']
    output_csv['limbDarkeningModel.modelName'] = limbDark_val

    # obtain log10Metallicity
    tce_smet_val = root.find(f'{DV_XML_HEADER}log10Metallicity').attrib['value']
    tce_smet_unc = root.find(f'{DV_XML_HEADER}log10Metallicity').attrib['uncertainty']
    output_csv['log10Metallicity.value'] = tce_smet_val
    output_csv['log10Metallicity.uncertainty'] = tce_smet_unc

    # obtain log10SurfaceGravity
    tce_slogg_val = root.find(f'{DV_XML_HEADER}log10SurfaceGravity').attrib['value']
    tce_slogg_unc = root.find(f'{DV_XML_HEADER}log10SurfaceGravity').attrib['uncertainty']
    output_csv['log10SurfaceGravity.value'] = tce_slogg_val
    output_csv['log10SurfaceGravity.uncertainty'] = tce_slogg_unc

    # tessMag
    mag_val = root.find(f'{DV_XML_HEADER}tessMag').attrib['value']
    mag_unc = root.find(f'{DV_XML_HEADER}tessMag').attrib['uncertainty']
    output_csv['tessMag.value'] = mag_val
    output_csv['tessMag.uncertainty'] = mag_unc

    # stellar density
    stellar_density_val = root.find(f'{DV_XML_HEADER}stellarDensity').attrib['value']
    stellar_density_unc = root.find(f'{DV_XML_HEADER}stellarDensity').attrib['uncertainty']
    output_csv['stellarDensity.value'] = stellar_density_val
    output_csv['stellarDensity.uncertainty'] = stellar_density_unc

    # radius
    radius_val = root.find(f'{DV_XML_HEADER}radius').attrib['value']
    radius_unc = root.find(f'{DV_XML_HEADER}radius').attrib['uncertainty']
    output_csv['radius.value'] = radius_val
    output_csv['radius.uncertainty'] = radius_unc

    # raDegrees
    raDegrees_val = root.find(f'{DV_XML_HEADER}raDegrees').attrib['value']
    raDegrees_unc = root.find(f'{DV_XML_HEADER}raDegrees').attrib['uncertainty']
    output_csv['raDegrees.value'] = raDegrees_val
    output_csv['raDegrees.uncertainty'] = raDegrees_unc

    # pmRa
    pmRa_val = root.find(f'{DV_XML_HEADER}pmRa').attrib['value']
    pmRa_unc = root.find(f'{DV_XML_HEADER}pmRa').attrib['uncertainty']
    output_csv['ra.value'] = pmRa_val
    output_csv['ra.uncertainty'] = pmRa_unc

    # pmDec
    pmDec_val = root.find(f'{DV_XML_HEADER}pmDec').attrib['value']
    pmDec_unc = root.find(f'{DV_XML_HEADER}pmDec').attrib['uncertainty']
    output_csv['pm_dec.value'] = pmDec_val
    output_csv['pm_dec.uncertainty'] = pmDec_unc

    # pipelineTaskId
    pipelineTaskId_val = root.attrib['pipelineTaskId']
    output_csv['taskFieldId'] = pipelineTaskId_val

    s_sector, e_sector = (root.attrib['sectorsObserved'].find('1') + 1, len(root.attrib['sectorsObserved']) -
                          root.attrib['sectorsObserved'][::-1].find('1') - 1)
    if s_sector == e_sector:
        output_csv['sector_run'] = s_sector
    else:
        output_csv['sector_run'] = f'{s_sector}-{e_sector}'


def find_descendants(output_csv, descendants, tce_i, keywords):
    """ Helper function which iteratively finds descendants based on the current parameter. Leverages the helper of
    append_attributes() to then append the given attribute that we have navigated to via find_descendants().

    Args:
        output_csv: pandas DataFrame, contains extracted data so far
        descendants: list, descendants that we must iterate through and append their given attributes for (within the
        XML file)
        tce_i:
        keywords: list, keywords that should be passed to append_attributes to properly name the column for a given
        attribute we are processing

    Returns:

    """

    # get attributed from current element
    append_attributes(descendants, tce_i, output_csv, keywords)

    if len(list(descendants.findall('.//*'))) != 0:

        # iterate through children elements of `topmost_param`
        for descendant in descendants:

            next_keyword = descendant.tag.split(DV_XML_HEADER, 1)[1]

            # exclude modelParameter, not used in labeling
            if "modelParameter" not in next_keyword and "differenceImagePixelData" not in next_keyword:
                keywords.append(next_keyword)

            append_attributes(descendant, tce_i, output_csv, keywords)

            next_set_descendants = descendant.findall('.//*')

            # find new set of descendants
            if len(next_set_descendants) != 0:
                find_descendants(output_csv, descendant, tce_i, keywords)

            # exclude current node if search for descendants was completed
            if len(keywords) != 0:
                keywords.pop()


def process_xml(dv_xml_fp):
    """ Main call for processing xml files. Will take a desired XML directory and output it to a designate output
    directory. Will output a set of individual files, with each target being restricted to one CSV file. The next
    python files you will need to run will end up compiling these files together to form a cohesive TCE table.

    Args:
        dv_xml_fp: Path, path to the DV xml file

    Returns:
        output_csv: pandas DataFrame, contains extracted data from the DV xml file
    """

    tree = et.parse(dv_xml_fp)
    root = tree.getroot()

    tic_id = root.attrib['ticId']
    planet_res_lst = [el for el in root if 'planetResults' in el.tag]

    # add number of TCEs
    n_tces = len(planet_res_lst)

    output_csv = pd.DataFrame(index=np.arange(n_tces))

    # insert same number of planets for all the tces within this xml file
    output_csv['numberOfPlanets'] = n_tces

    # set any parameters on the topmost level
    output_csv['catId'] = tic_id

    for tce_i, planet_res in enumerate(planet_res_lst):

        output_csv.at[tce_i, 'planetIndexNumber'] = planet_res.attrib['planetNumber']

        # cur_eclipbin = planet_res[8].attrib['suspectedEclipsingBinary']
        # bool_cur_eclipbin = int(cur_eclipbin == 'true')
        # output_csv.at[tce_i, f'planetCandidate.suspectedEclipsingBinary'] = bool_cur_eclipbin

        # allTransitsFit_numberOfModelParameters
        output_csv.at[tce_i, f'allTransitsFit_numberOfModelParameters'] = len(planet_res[0][1])

        # iterate through elements in planet results and their children
        for topmost_param in planet_res:

            topmost_param_str = topmost_param.tag.split(DV_XML_HEADER, 1)[1]

            # handle specific cases
            if "binaryDiscriminationResults" in topmost_param_str:  # binary discrimination test
                longer_period_val = topmost_param[0].attrib['value']
                longer_period_sig = topmost_param[0].attrib['significance']
                output_csv.at[
                    tce_i, 'binaryDiscriminationResults.longerPeriodComparisonStatistic_val'] = longer_period_val
                output_csv.at[
                    tce_i, 'binaryDiscriminationResults.longerPeriodComparisonStatistic_significance'] = (
                    longer_period_sig)

                trans_depth_val = topmost_param[1].attrib['value']
                trans_depth_sig = topmost_param[1].attrib['significance']
                output_csv.at[
                    tce_i, 'binaryDiscriminationResults.oddEvenTransitDepthComparisonStatistic_val'] = trans_depth_val
                output_csv.at[
                    tce_i, 'binaryDiscriminationResults.oddEvenTransitDepthComparisonStatistic_significance'] = (
                    trans_depth_sig)

                short_period_val = topmost_param[2].attrib['value']
                short_period_sig = topmost_param[2].attrib['significance']
                output_csv.at[
                    tce_i, 'binaryDiscriminationResults.shorterPeriodComparisonStatistic_val'] = short_period_val
                output_csv.at[
                    tce_i, 'binaryDiscriminationResults.shorterPeriodComparisonStatistic_sig'] = short_period_sig

            if "centroidResults" in topmost_param_str:  # centroid results
                centroidOffsets = topmost_param[0][0]
                skyoffset_val = centroidOffsets[2].attrib['value']
                skyoffset_unc = centroidOffsets[2].attrib['uncertainty']
                output_csv.at[
                    tce_i,
                    'centroidResults.differenceImageMotionResults.msControlCentroidOffsets.meanSkyOffset_value'] = (
                    skyoffset_val)
                output_csv.at[
                    tce_i,
                    'centroidResults.differenceImageMotionResults.msControlCentroidOffsets.meanSkyOffset_uncertainty'] \
                    = skyoffset_unc

            # iteratively find descendants
            find_descendants(output_csv, topmost_param, tce_i, [topmost_param_str])

        # get the allTransitsFit results
        get_allTransitsFit(planet_res, tce_i, output_csv)

    # get all outside params from the planet candidates
    get_outside_params(root, output_csv)

    return output_csv


if __name__ == "__main__":

    # output file path to csv with extracted data
    new_tce_tbl_fp = Path('/data5/tess_project/Data/Ephemeris_tables/TESS/dv_spoc_ffi/preprocessed_tce_tables/tess_spoc_ffi_s36-s69_tces_7-3-2024_1242.csv')

    # dv_xml_fps = [
    #     Path('/data5/tess_project/Data/tess_spoc_ffi_data/dv/xml_files/single-sector/s0066/target/0000/0002/7947/8852/hlsp_tess-spoc_tess_phot_0000000279478852-s0066-s0066_tess_v1_dvr.xml'),
    # ]

    # get file paths to DV xml files
    # dv_xml_sector_runs_dirs = [
    #     Path('/data5/tess_project/Data/tess_spoc_ffi_data/dv/xml_files/single-sector/s0066'),
    #     Path('/data5/tess_project/Data/tess_spoc_ffi_data/dv/xml_files/single-sector/s0069'),
    # ]
    dv_xml_sector_runs_dirs = list(Path('/data5/tess_project/Data/tess_spoc_ffi_data/dv/xml_files/single-sector/').iterdir())
    print(f'Choosing sectors in {dv_xml_sector_runs_dirs}.')
    dv_xml_fps = []
    for dv_xml_sector_run in dv_xml_sector_runs_dirs:
        dv_xml_fps += [fp for fp in dv_xml_sector_run.rglob('*.xml')]

    # dv_xml_fps = dv_xml_fps[:2]
    print(f'Extracting TCEs from {len(dv_xml_fps)} xml files.')

    # parallel extraction of data from multiple DV xml files
    n_processes = 12
    n_jobs = len(dv_xml_fps)
    pool = multiprocessing.Pool(processes=n_processes)
    async_results = [pool.apply_async(process_xml, (dv_xml_fp,)) for dv_xml_fp in dv_xml_fps]
    pool.close()
    pool.join()
    new_tce_tbl = pd.concat([el.get() for el in async_results])

    print(f'Renaming fields...')
    new_tce_tbl = new_tce_tbl.rename(columns=MAP_DV_XML_FIELDS, inplace=False, errors='raise')

    new_tce_tbl = new_tce_tbl.astype({'target_id': int, 'tce_plnt_num': int})
    # set unique id for each TCE
    new_tce_tbl['uid'] = new_tce_tbl.apply(lambda x: '{}-{}-S{}'.format(x['target_id'], x['tce_plnt_num'],
                                                                        x['sector_run']), axis=1)
    new_tce_tbl.set_index('uid', inplace=True)  # set uid as index

    new_tce_tbl.to_csv(new_tce_tbl_fp, index=True)

    print(f'Saved TCE table to {str(new_tce_tbl_fp)}.')
