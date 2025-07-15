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
import re
import logging

DV_XML_HEADER = '{http://www.nasa.gov/2018/TESS/DV}'


def append_attributes(descendants, tce_dict, keywords):
    """ Appends attributes of a given parameter. Helper to process_xml().

    Args:
        descendants: elem, given location within a processed .xml file. Will include the key value we are currently
        sitting at
        tce_dict: dict, contains extracted data so far for the TCE
        keywords: list, keywords that have been logged when traversing the tree of the .xml file. Necessary for naming
        a designated column in the final table

    Returns: tce_dict with added attributes

    """

    # get topmost values from given descendant
    attributes = descendants.attrib

    for cur_att in attributes:

        if "modelParameter" not in cur_att:
            keywords.append(cur_att)

        out_col_name = '.'.join(keywords)

        if cur_att == 'suspectedEclipsingBinary':
            tce_dict[ out_col_name] = int(attributes[cur_att] == 'true')
        else:
            tce_dict[out_col_name] = attributes[cur_att]

        # we checked the attribute, backtrack from it
        keywords.pop()

    return tce_dict


def get_allTransitsFit(planet_res, tce_dict):
    """ Function designed specifically for getting the 'allTransitsFit' parameters.

    Hardcoded variables are designated to ensure that they are outputted correctly with desired labels to the final csv.

    Args:
        planet_res: elem, current location within the given XML file along the tree
        tce_i: int, current TCE index to properly output the current attributes to a proper
        tce_dict: dict, contains extracted data so far for the TCE

    Returns: tce_dict updated with allTransitsFit parameters

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
        tce_dict[f'allTransitsFit.{param_name}.uncertainty'] = unc_col

        tce_dict[f'allTransitsFit.{param_name}.value'] = val_col
        tce_dict[f'allTransitsFit.{param_name}.fitted'] = fitted_col

    # meanskyoffset
    dikco_msky_val = planet_res[3][0][0][2].attrib['value']
    dikco_msky_unc = planet_res[3][0][0][2].attrib['uncertainty']
    tce_dict['centroidResults.differenceImageMotionResults.msTicCentroidOffsets.meanSkyOffset.value'] = (
        dikco_msky_val)
    tce_dict['centroidResults.differenceImageMotionResults.msTicCentroidOffsets.meanSkyOffset.uncertainty'] = (
        dikco_msky_unc)

    dicco_msky_val = planet_res[3][0][1][2].attrib['value']
    dicco_msky_unc = planet_res[3][0][1][2].attrib['uncertainty']
    tce_dict['centroidResults.differenceImageMotionResults.msControlCentroidOffsets.meanSkyOffset.value'] = (
        dicco_msky_val)
    tce_dict['centroidResults.differenceImageMotionResults.msControlCentroidOffsets.meanSkyOffset.uncertainty'] = (
        dicco_msky_unc)

    return  tce_dict


def get_outside_params(root):
    """ Hard-coded method for grabbing the attributes not nested within "descendants" or sub-levels within the tree of
    the xml file that need to grab.

    Args:
        root: root elem, the topmost level of the given xml file

    Returns: target_params, dict with parameters shared by all TCEs in a given target

    """

    target_params = {}

    # decDegrees
    decDegrees_val = root.find(f'{DV_XML_HEADER}decDegrees').attrib['value']
    decDegrees_unc = root.find(f'{DV_XML_HEADER}decDegrees').attrib['uncertainty']
    target_params['decDegrees.value'] = decDegrees_val
    target_params['decDegrees.uncertainty'] = decDegrees_unc

    # retrieve effective temp
    tce_steff = root.find(f'{DV_XML_HEADER}effectiveTemp').attrib['value']
    tce_steff_unc = root.find(f'{DV_XML_HEADER}effectiveTemp').attrib['uncertainty']
    target_params['effectiveTemp.value'] = tce_steff
    target_params['effectiveTemp.uncertainty'] = tce_steff_unc

    # limbDarkeningModel
    limbDark_val = root.find(f'{DV_XML_HEADER}limbDarkeningModel').attrib['modelName']
    target_params['limbDarkeningModel.modelName'] = limbDark_val

    # obtain log10Metallicity
    tce_smet_val = root.find(f'{DV_XML_HEADER}log10Metallicity').attrib['value']
    tce_smet_unc = root.find(f'{DV_XML_HEADER}log10Metallicity').attrib['uncertainty']
    target_params['log10Metallicity.value'] = tce_smet_val
    target_params['log10Metallicity.uncertainty'] = tce_smet_unc

    # obtain log10SurfaceGravity
    tce_slogg_val = root.find(f'{DV_XML_HEADER}log10SurfaceGravity').attrib['value']
    tce_slogg_unc = root.find(f'{DV_XML_HEADER}log10SurfaceGravity').attrib['uncertainty']
    target_params['log10SurfaceGravity.value'] = tce_slogg_val
    target_params['log10SurfaceGravity.uncertainty'] = tce_slogg_unc

    # tessMag
    mag_val = root.find(f'{DV_XML_HEADER}tessMag').attrib['value']
    mag_unc = root.find(f'{DV_XML_HEADER}tessMag').attrib['uncertainty']
    target_params['tessMag.value'] = mag_val
    target_params['tessMag.uncertainty'] = mag_unc

    # stellar density
    stellar_density_val = root.find(f'{DV_XML_HEADER}stellarDensity').attrib['value']
    stellar_density_unc = root.find(f'{DV_XML_HEADER}stellarDensity').attrib['uncertainty']
    target_params['stellarDensity.value'] = stellar_density_val
    target_params['stellarDensity.uncertainty'] = stellar_density_unc

    # radius
    radius_val = root.find(f'{DV_XML_HEADER}radius').attrib['value']
    radius_unc = root.find(f'{DV_XML_HEADER}radius').attrib['uncertainty']
    target_params['radius.value'] = radius_val
    target_params['radius.uncertainty'] = radius_unc

    # raDegrees
    raDegrees_val = root.find(f'{DV_XML_HEADER}raDegrees').attrib['value']
    raDegrees_unc = root.find(f'{DV_XML_HEADER}raDegrees').attrib['uncertainty']
    target_params['raDegrees.value'] = raDegrees_val
    target_params['raDegrees.uncertainty'] = raDegrees_unc

    # pmRa
    pmRa_val = root.find(f'{DV_XML_HEADER}pmRa').attrib['value']
    pmRa_unc = root.find(f'{DV_XML_HEADER}pmRa').attrib['uncertainty']
    target_params['ra.value'] = pmRa_val
    target_params['ra.uncertainty'] = pmRa_unc

    # pmDec
    pmDec_val = root.find(f'{DV_XML_HEADER}pmDec').attrib['value']
    pmDec_unc = root.find(f'{DV_XML_HEADER}pmDec').attrib['uncertainty']
    target_params['pm_dec.value'] = pmDec_val
    target_params['pm_dec.uncertainty'] = pmDec_unc

    # pipelineTaskId
    pipelineTaskId_val = root.attrib['pipelineTaskId']
    target_params['taskFieldId'] = pipelineTaskId_val

    # get observed sectors
    obs_sectors = np.where(np.array([*root.attrib['sectorsObserved']]) == '1')[0]
    # get start and end sector
    target_params['sectors_observed'] = '_'.join([str(obs_sector) for obs_sector in obs_sectors])

    return  target_params


def find_descendants(tce_dict, descendants, keywords):
    """ Helper function which iteratively finds descendants based on the current parameter. Leverages the helper of
    append_attributes() to then append the given attribute that we have navigated to via find_descendants().

    Args:
        tce_dict: dict, contains extracted data so far for the TCE
        descendants: list, descendants that we must iterate through and append their given attributes for (within the
        XML file)
        keywords: list, keywords that should be passed to append_attributes to properly name the column for a given
        attribute we are processing

    Returns: tce_dict, dict updated with new attributes

    """

    # get attributed from current element
    append_attributes(descendants, tce_dict, keywords)

    if len(list(descendants.findall('.//*'))) != 0:

        # iterate through children elements of `topmost_param`
        for descendant in descendants:

            next_keyword = descendant.tag.split(DV_XML_HEADER, 1)[1]

            # exclude modelParameter, not used in labeling
            if "modelParameter" not in next_keyword and "differenceImagePixelData" not in next_keyword:
                keywords.append(next_keyword)

            tce_dict = append_attributes(descendant, tce_dict, keywords)

            next_set_descendants = descendant.findall('.//*')

            # find new set of descendants
            if len(next_set_descendants) != 0:
                tce_dict = find_descendants(tce_dict, descendant, keywords)

            # exclude current node if search for descendants was completed
            if len(keywords) != 0:
                keywords.pop()

    return tce_dict


def process_xml(dv_xml_fp, logger):
    """ Main call for processing xml files. Will take a desired XML directory and output it to a designate output
    directory. Will output a set of individual files, with each target being restricted to one CSV file. The next
    python files you will need to run will end up compiling these files together to form a cohesive TCE table.

    Args:
        dv_xml_fp: Path, path to the DV xml file
        logger: logger

    Returns:
        output_csv: pandas DataFrame, contains extracted data from the DV xml file
    """

    tree = et.parse(dv_xml_fp)
    root = tree.getroot()

    tic_id = root.attrib['ticId']
    planet_res_lst = [el for el in root if 'planetResults' in el.tag]

    # add number of TCEs
    n_tces = len(planet_res_lst)
    logger.info(f'Found {n_tces} TCEs results in {dv_xml_fp.name}')

    tces_in_target = []
    for tce_i, planet_res in enumerate(planet_res_lst):

        tce_dict = {}

        tce_dict['planetIndexNumber'] = planet_res.attrib['planetNumber']

        # cur_eclipbin = planet_res[8].attrib['suspectedEclipsingBinary']
        # bool_cur_eclipbin = int(cur_eclipbin == 'true')
        # output_csv.at[tce_i, f'planetCandidate.suspectedEclipsingBinary'] = bool_cur_eclipbin

        # allTransitsFit_numberOfModelParameters
        tce_dict[f'allTransitsFit_numberOfModelParameters'] = len(planet_res[0][1])

        # iterate through elements in planet results and their children
        for topmost_param in planet_res:

            topmost_param_str = topmost_param.tag.split(DV_XML_HEADER, 1)[1]

            # handle specific cases
            if "binaryDiscriminationResults" in topmost_param_str:  # binary discrimination test
                longer_period_val = topmost_param[0].attrib['value']
                longer_period_sig = topmost_param[0].attrib['significance']
                tce_dict['binaryDiscriminationResults.longerPeriodComparisonStatistic_val'] = longer_period_val
                tce_dict['binaryDiscriminationResults.longerPeriodComparisonStatistic_significance'] = (
                    longer_period_sig)

                trans_depth_val = topmost_param[1].attrib['value']
                trans_depth_sig = topmost_param[1].attrib['significance']
                tce_dict['binaryDiscriminationResults.oddEvenTransitDepthComparisonStatistic_val'] = trans_depth_val
                tce_dict['binaryDiscriminationResults.oddEvenTransitDepthComparisonStatistic_significance'] = (
                    trans_depth_sig)

                short_period_val = topmost_param[2].attrib['value']
                short_period_sig = topmost_param[2].attrib['significance']
                tce_dict['binaryDiscriminationResults.shorterPeriodComparisonStatistic_val'] = short_period_val
                tce_dict['binaryDiscriminationResults.shorterPeriodComparisonStatistic_sig'] = short_period_sig

            if "centroidResults" in topmost_param_str:  # centroid results
                centroidOffsets = topmost_param[0][0]
                skyoffset_val = centroidOffsets[2].attrib['value']
                skyoffset_unc = centroidOffsets[2].attrib['uncertainty']
                tce_dict[
                    'centroidResults.differenceImageMotionResults.msControlCentroidOffsets.meanSkyOffset_value'] = (
                    skyoffset_val)
                tce_dict[
                    'centroidResults.differenceImageMotionResults.msControlCentroidOffsets.meanSkyOffset_uncertainty'] \
                    = skyoffset_unc

            # iteratively find descendants
            tce_dict = find_descendants(tce_dict, topmost_param, [topmost_param_str])

        # get the allTransitsFit results
        tce_dict = get_allTransitsFit(planet_res, tce_dict)

        tces_in_target.append(tce_dict)

    tces_df = pd.DataFrame(tces_in_target)

    # get sector run from filename
    sector_run_str = re.search(r's\d{4}-s\d{4}', dv_xml_fp.name).group()
    s_sector, e_sector = sector_run_str.split('-')
    if s_sector == e_sector:  # single-sector run
        tces_df['sector_run'] = str(int(s_sector[1:]))
    else:   # multi-sector
        # tces_df['sector_run'] = f'{str(s_sector[1:])}-{str(e_sector[1:])}'
        tces_df['sector_run'] = f'{int(s_sector[1:])}-{int(e_sector[1:])}'

    # insert same number of planets for all the tces within this xml file
    tces_df['numberOfPlanets'] = n_tces

    # set any parameters on the topmost level
    tces_df['catId'] = tic_id

    # get all outside params from the planet candidates
    target_params = get_outside_params(root)

    for param_name, param_val in target_params.items():
        tces_df[param_name] = param_val

    if len(tces_df) != n_tces:
        logger.info(f'Number of TCEs extracted {len(tces_df)} is different than the number of TCEs with DV results '
                    f'({n_tces}) in {dv_xml_fp.name}.')

    return tces_df


def process_sector_run_of_dv_xmls(dv_xml_sector_run_dir, dv_xml_tbl_fp):
    """ Extracts TCE data from a set of DV xml files in a directory `dv_xml_sector_run_dir` into a table and returns
    the table as a pandas DataFrame.

    Args:
        dv_xml_sector_run_dir: Path, path to the sector run directory
        dv_xml_tbl_fp: Path, filepath used to save table with DV xml results

    Returns:
        dv_xml_tbl: pandas DataFrame, contains extracted data from the DV xml files
    """

    # set up logger
    logger = logging.getLogger(name='create_tess_tce_tbl_from_dv')
    logger_handler = logging.FileHandler(filename=dv_xml_tbl_fp.parent / 'logs' /
                                                  f'create_tce_tbl_from_dv_{dv_xml_sector_run_dir.name}.log', mode='w')
    logger_formatter = logging.Formatter('%(asctime)s - %(message)s')
    logger.setLevel(logging.INFO)
    logger_handler.setFormatter(logger_formatter)
    logger.addHandler(logger_handler)
    logger.info(f'Starting run...')

    dv_xml_fps = list(dv_xml_sector_run_dir.rglob('*.xml'))
    n_dv_xmls = len(dv_xml_fps)
    logger.info(f'Extracting TCEs from {n_dv_xmls} xml files for {dv_xml_sector_run_dir.name}...')

    dv_xmls_tbls = []
    for dv_xml_fp_i, dv_xml_fp in enumerate(dv_xml_fps):
        if dv_xml_fp_i % 100 == 0:
            logger.info(f'Target DV XML file {dv_xml_fp.name} ({dv_xml_fp_i + 1}/{n_dv_xmls})')

        try:
            dv_xml_tbl_target = process_xml(dv_xml_fp, logger)
            dv_xmls_tbls.append(dv_xml_tbl_target)
        except Exception as e:
            logger.info(f'Error when adding TCE table from {dv_xml_fp}.')
            logger.info(f'Error: {e}')

    dv_xml_tbl = pd.concat(dv_xmls_tbls, axis=0, ignore_index=True)
    dv_xml_tbl.to_csv(dv_xml_tbl_fp, index=False)

    logger.info(f'Finished extracting {len(dv_xml_tbl)} TCEs from {len(dv_xml_fps)} xml files for {dv_xml_tbl_fp.name}.')


if __name__ == "__main__":

    # output file path to csv with extracted data
    new_tce_tbls_dir = Path('/nobackupp19/msaragoc/work_dir/Kepler-TESS_exoplanet/data/Ephemeris_tables/TESS/tess_spoc_ffi/tess_spoc_ffi_tces_dv_s36-s72_s56s69_3-21-2025_1010/')
    new_tce_tbls_dir.mkdir(exist_ok=True)

    logs_dir = new_tce_tbls_dir / 'logs'
    logs_dir.mkdir(exist_ok=True)

    dv_xml_tbl_fp = new_tce_tbls_dir / f'{new_tce_tbls_dir.name}.csv'

    # get file paths to DV xml files
    dv_xml_sector_runs_root_dir = Path('/nobackupp19/msaragoc/work_dir/Kepler-TESS_exoplanet/data/FITS_files/TESS/spoc_ffi/dv/xml_files/')
    dv_xml_sector_runs_dirs_lst = [sector_run_dir for sector_run_dir in (dv_xml_sector_runs_root_dir / 'single-sector').iterdir()]  #  if sector_run_dir.name in [f'sector_{sector_run}' for sector_run in range(69, 88 + 1)]]
    dv_xml_sector_runs_dirs_lst += [sector_run_dir for sector_run_dir in (dv_xml_sector_runs_root_dir / 'multi-sector').iterdir()]  #  if sector_run_dir.name in [f'multisector_{sector_run}' for sector_run in ['s0014-s0078', 's0002-s0072', 's0001-s0069', 's0014-s0078']]]

    print(f'Choosing sectors {dv_xml_sector_runs_dirs_lst}.')

    # dv_xml_sector_runs_fps = list(dv_xml_sector_runs_dir.rglob('*dvr.xml'))
    # print(f'Extracting TCE data from {len(dv_xml_sector_runs_fps)} DV xml files.')

    # parallel extraction of data from multiple DV xml files
    n_processes = len(dv_xml_sector_runs_dirs_lst)  # 36
    n_jobs = len(dv_xml_sector_runs_dirs_lst)
    # dv_xml_arr = np.array_split(dv_xml_sector_runs_fps, n_jobs)
    print(f'Starting {n_processes} processes to deal with {n_jobs} jobs.')
    pool = multiprocessing.Pool(processes=n_processes)
    async_results = [pool.apply_async(process_sector_run_of_dv_xmls,
                                      (dv_xml_sector_run_dir,
                                       new_tce_tbls_dir / f'dv_xml_{dv_xml_sector_run_dir.name}.csv'))
                     for _, dv_xml_sector_run_dir in enumerate(dv_xml_sector_runs_dirs_lst)]
    pool.close()
    pool.join()

    # for dv_xml_i, dv_xml_sector_run_dir in enumerate(dv_xml_sector_runs_dirs_lst):
    #     process_sector_run_of_dv_xmls(dv_xml_sector_run_dir,
    #                                    new_tce_tbls_dir / f'dv_xml_{dv_xml_sector_run_dir.name}.csv')

    # concatenated extracted TCE tables for each sector run
    dv_xml_tbls = [pd.read_csv(fp) for fp in new_tce_tbls_dir.glob('*.csv') if fp != dv_xml_tbl_fp]

    dv_xml_tbl = pd.concat(dv_xml_tbls, axis=0, ignore_index=True)
    dv_xml_tbl.to_csv(dv_xml_tbl_fp, index=False)
    print(f'Saved TCE table to {str(dv_xml_tbl_fp)}.')
