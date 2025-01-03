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

DV_XML_HEADER = '{http://www.nasa.gov/2018/TESS/DV}'


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
        tce_i, 'centroidResults.differenceImageMotionResults.msTicCentroidOffsets.meanSkyOffset.value'] = (
        dikco_msky_val)
    output_csv.at[
        tce_i, 'centroidResults.differenceImageMotionResults.msTicCentroidOffsets.meanSkyOffset.uncertainty'] = (
        dikco_msky_unc)

    dicco_msky_val = planet_res[3][0][1][2].attrib['value']
    dicco_msky_unc = planet_res[3][0][1][2].attrib['uncertainty']
    output_csv.at[
        tce_i, 'centroidResults.differenceImageMotionResults.msControlCentroidOffsets.meanSkyOffset.value'] = (
        dicco_msky_val)
    output_csv.at[
        tce_i, 'centroidResults.differenceImageMotionResults.msControlCentroidOffsets.meanSkyOffset.uncertainty'] = (
        dicco_msky_unc)


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

    # get observed sectors
    obs_sectors_idxs = np.where(np.array([*root.attrib['sectorsObserved']]) == '1')[0]
    # get start and end sector
    output_csv['sectors_observed'] = obs_sectors_idxs
    # if len(obs_sectors_idxs) == 1:  # single-sector run
    #     output_csv['sector_run'] = str(obs_sectors_idxs[0])
    # else:  # multi-sector run
    #     output_csv['sector_run'] = f'{obs_sectors_idxs[0]}-{obs_sectors_idxs[-1]}'


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

    # get sector run from filename
    sector_run_str = re.search('s\d{4}-s\d{4}', dv_xml_fp.name).group()
    s_sector, e_sector = sector_run_str.split('-')
    if s_sector == e_sector:  # single-sector run
        output_csv['sector_run'] = str(s_sector[1:])
    else:   # multi-sector
        output_csv['sector_run'] = f'{str(s_sector[1:])}-{str(e_sector[1:])}'

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


def process_sector_run_of_dv_xmls(dv_xml_sector_run_dir, dv_xml_tbl_fp):
    """ Extracts TCE data from a set of DV xml files in a directory `dv_xml_sector_run_dir` into a table and returns
    the table as a pandas DataFrame.

    Args:
        # dv_xml_fps: list of Paths, DV xml filepaths
        dv_xml_sector_run_dir: Path, path to the sector run directory
        dv_xml_tbl_fp: Path, filepath used to save table with DV xml results

    Returns:
        dv_xml_tbl: pandas DataFrame, contains extracted data from the DV xml files
    """

    dv_xml_fps = list(dv_xml_sector_run_dir.rglob('*.xml'))
    n_dv_xmls = len(dv_xml_fps)
    print(f'Extracting TCEs from {n_dv_xmls} xml files for {dv_xml_sector_run_dir.name}...')
    # print(f'Extracting TCEs from {n_dv_xmls} xml files for...')

    dv_xmls_tbls = []
    for dv_xml_fp_i, dv_xml_fp in enumerate(dv_xml_fps):
        if dv_xml_fp_i % 500 == 0:
            print(f'Target DV XML file {dv_xml_fp.name} ({dv_xml_fp_i + 1}/{n_dv_xmls})')

        try:
            dv_xml_tbl_target = process_xml(dv_xml_fp)
            dv_xmls_tbls.append(dv_xml_tbl_target)
        except Exception as e:
            print(f'Error when adding TCE table from {dv_xml_fp}. Shape of table: {dv_xml_tbl_target.shape}')
            print(f'Data type: {type(dv_xml_tbl_target)}')
            print(f'Table:\n{dv_xml_tbl_target}')
            print(f'Error: {e}')

    dv_xml_tbl = pd.concat(dv_xmls_tbls, axis=0, ignore_index=True)
    dv_xml_tbl.to_csv(dv_xml_tbl_fp, index=False)

    # print(f'Finished extracting {len(dv_xml_tbl)} TCEs from {len(dv_xml_fps)} xml files for '
    #       f'{dv_xml_sector_run_dir.name}.')
    print(f'Finished extracting {len(dv_xml_tbl)} TCEs from {len(dv_xml_fps)} xml files.')

    return dv_xml_tbl


if __name__ == "__main__":

    # output file path to csv with extracted data
    new_tce_tbls_dir = Path('/home6/msaragoc/work_dir/Kepler-TESS_exoplanet/data/Ephemeris_tables/TESS/tess_spoc_ffi/tess_spoc_ffi_s36-s72_multisector_s56-s69_sfromdvxml_11-22-2024_0942/')
    new_tce_tbls_dir.mkdir(exist_ok=True)

    dv_xml_tbl_fp = new_tce_tbls_dir / f'{new_tce_tbls_dir.name}.csv'

    # get file paths to DV xml files
    dv_xml_sector_runs_root_dir = Path('/home6/msaragoc/work_dir/Kepler-TESS_exoplanet/data/FITS_files/TESS/spoc_ffi/dv/xml_files/')
    dv_xml_sector_runs_dirs_lst = [sector_run_dir
                                   for sector_run_dir in (dv_xml_sector_runs_root_dir / 'single-sector').iterdir()]
    dv_xml_sector_runs_dirs_lst += [sector_run_dir
                                    for sector_run_dir in (dv_xml_sector_runs_root_dir / 'multi-sector').iterdir()]

    print(f'Choosing sectors {dv_xml_sector_runs_dirs_lst}.')

    # dv_xml_sector_runs_fps = list(dv_xml_sector_runs_dir.rglob('*dvr.xml'))
    # print(f'Extracting TCE data from {len(dv_xml_sector_runs_fps)} DV xml files.')

    # parallel extraction of data from multiple DV xml files
    n_processes = 36
    n_jobs = len(dv_xml_sector_runs_dirs_lst)
    # dv_xml_arr = np.array_split(dv_xml_sector_runs_fps, n_jobs)
    print(f'Starting {n_processes} processes to deal with {n_jobs} jobs.')
    pool = multiprocessing.Pool(processes=n_processes)
    async_results = [pool.apply_async(process_sector_run_of_dv_xmls,
                                      (dv_xml_sector_run_dir,
                                       new_tce_tbls_dir / f'dv_xml_{dv_xml_sector_run_dir.name}.csv'))
                     for dv_xml_i, dv_xml_sector_run_dir in enumerate(dv_xml_sector_runs_dirs_lst)]
    pool.close()
    pool.join()

    dv_xml_tbls = []
    for proc_output_i, proc_output in enumerate(async_results):
        dv_xml_tbls.append(proc_output.get())
        # tce_tbl_sector = proc_output.get()
        # tce_tbl_fp = new_tce_tbls_dir / f'tess_spoc_ffi_tces_s{tce_tbl_sector["sector_run"][0]}.csv'
        # tce_tbl_fp = new_tce_tbls_dir / f'tess_spoc_ffi_tces_s{tce_tbl_sector["sector_run"][0]}.csv'
        # tce_tbl_sector.to_csv(tce_tbl_fp, index=False)
        # print(f'Saved TCE table for sector run {tce_tbl_sector} to {str(tce_tbl_fp)}.')

    dv_xml_tbl = pd.concat(dv_xml_tbls, axis=0, ignore_index=True)
    dv_xml_tbl.to_csv(dv_xml_tbl_fp, index=False)
    print(f'Saved TCE table to {str(dv_xml_tbl_fp)}.')
