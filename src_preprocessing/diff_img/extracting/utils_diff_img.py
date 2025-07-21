""" Utility functions for extracting difference imaging. """

# 3rd party
import xml.etree.cElementTree as et
import os
import matplotlib.pyplot as plt
import numpy as np
import logging
import re
from matplotlib.colors import LogNorm
import pandas as pd

plt.switch_backend('agg')

N_QUARTERS_KEPLER = 17
N_IMGS_IN_DIFF = 4  # diff, oot, it, snr
MAX_MAG = 25
MIN_MAG = 1
MAG_RANGE = MIN_MAG - MAX_MAG
TESS_MAG_SAT = 7
KEPLER_MAG_SAT = 12
MIN_IMG_VALUE = 1e-12


def plot_diff_img_data(diff_imgs, plot_fp, target_coords=None, neighbors_coords=None, target_mag=None, neighbors_mag=None,
                       mag_sat=1, min_mag=None, logscale=True):
    """ Plot difference image data for TCE in a given quarter/sector.

    Args:
        diff_imgs: NumPy array, difference image data [cols, rows, it|oot|diff|snr, value| uncertainty]
        plot_fp: Path, plot file path
        target_coords: tuple, target col and row coordinates
        neighbors_coords: list of tuples, neighbors col and row coordinates
        target_mag: float, target magnitude
        neighbors_mag: list of floats, neighbors magnitudes
        mag_sat: float, magnitude saturation threshold
        min_mag: float, minimum magnitude at which the colormap is clipped
        logscale: bool, if True images color is set to logscale

    Returns:

    """

    def _create_subplot(ax_img, img, img_title, target_coords=None, neighbors_coords=None, target_mag=None,
                        neighbors_mag=None, logscale=True, mask_invalid_pixels=False):

        if mask_invalid_pixels:
            img = np.ma.masked_less(img, 0)
            cmap_img = plt.cm.viridis
            cmap_img.set_bad(color='gray')  # Set the color for non-positive values
        else:
            cmap_img = plt.cm.viridis

        if logscale:  # handle zero-valued pixels
            img[img == 0] = MIN_IMG_VALUE

        # plot image data
        im = ax_img.imshow(img, cmap=cmap_img, norm=LogNorm() if logscale else None, origin='lower')
        # set target location and magnitude
        if target_coords:
            sc = ax_img.scatter(target_coords[0], target_coords[1],
                                marker='x', c=[target_mag], label='Target', zorder=2, cmap='plasma_r',
                                vmin=mag_sat, vmax=MAX_MAG if min_mag is None else min_mag)
            cbar_sc = plt.colorbar(sc, ax=ax_img, orientation='horizontal', location='top', fraction=0.046, pad=0.01)
            cbar_sc.set_label('Magnitude')

        # set neighbors location and magnitude
        if neighbors_coords:
            for neighbor_i, neighbor_coords in enumerate(neighbors_coords):
                ax_img.scatter(neighbor_coords[0], neighbor_coords[1],
                               marker='*', c=neighbors_mag[neighbor_i], label=None, zorder=1, cmap='plasma_r',
                               vmin=mag_sat, vmax=MAX_MAG if min_mag is None else min_mag)
        # set color bars
        cbar_im = plt.colorbar(im, ax=ax_img, orientation='vertical', fraction=0.046, pad=0.04)
        # Set colorbar labels
        cbar_im.set_label(r'Flux [$e^-/cadence$]')
        cbar_im.ax.set_position([cbar_im.ax.get_position().x1 - 0.02,
                                 cbar_im.ax.get_position().y0,
                                 cbar_im.ax.get_position().width,
                                 cbar_im.ax.get_position().height])

        ax_img.set_ylabel('Row')
        ax_img.set_xlabel('Col', labelpad=10)
        # ax_img.invert_yaxis()
        # ax[0, 0].legend()
        ax_img.set_title(img_title, pad=50)

    f, ax = plt.subplots(2, 2, figsize=(14, 14))

    # diff img
    _create_subplot(ax[0, 0], diff_imgs[:, :, 2, 0], 'Difference Flux', target_coords,
                    neighbors_coords=neighbors_coords, target_mag=target_mag, neighbors_mag=neighbors_mag,
                    logscale=False)

    # oot img
    _create_subplot(ax[0, 1], diff_imgs[:, :, 1, 0], 'Out-of-transit Flux', target_coords,
                    neighbors_coords=neighbors_coords, target_mag=target_mag, neighbors_mag=neighbors_mag,
                    logscale=logscale, mask_invalid_pixels=True)

    # it img
    _create_subplot(ax[1, 0], diff_imgs[:, :, 0, 0], 'In-transit Flux', target_coords,
                    neighbors_coords=neighbors_coords, target_mag=target_mag, neighbors_mag=neighbors_mag,
                    logscale=logscale, mask_invalid_pixels=True)

    # snr img
    _create_subplot(ax[1, 1], diff_imgs[:, :, 3, 0], 'SNR Flux', target_coords,
                    neighbors_coords=neighbors_coords, target_mag=target_mag, neighbors_mag=neighbors_mag,
                    logscale=logscale, mask_invalid_pixels=True)

    f.subplots_adjust(hspace=0.4, wspace=0.4)
    f.savefig(plot_fp)
    plt.close()


def get_data_from_kepler_dv_xml(dv_xml_fp, tces, plot_dir, plot_prob, logger):
    """ Extract difference image data from the DV XML file for a set of Kepler Q1-Q17 DR25 TCEs.

    :param dv_xml_fp: Path, file path to DV XML file
    :param tces: pandas DataFrame, TCEs for which to extract difference image data. Must contain two columns: 'uid' and
    'label'. 'uid' must be of the pattern '{tic_id}-{tce_plnt_num}'
    :param plot_dir: Path, plot directory
    :param plot_prob: float, probability to plot difference image for a given example ([0, 1])
    :param logger: logger
    :return: dict, each item is the difference image data for a given TCE. The TCE is identified by the string key
    '{kic_id}-{tce_plnt_num}'. The value is a dictionary that contains the following items: 'target_ref_centroid'
    is a dictionary that contains the value and uncertainty for the reference coordinates of the target star in the
    pixel domain; 'image_data' is a NumPy array (n_rows, n_cols, n_imgs, 2) that contains the in-transit,
    out-of-transit, difference, and SNR flux images in this order (pixel values and uncertainties are addressed by the
    last dimension of the array, in this order); 'image_number' is a list that contains the integer quarter numbers of
    the corresponding sequence of difference image data extracted for the TCE.
    """

    proc_id = os.getpid()

    # get an iterable
    context = et.iterparse(dv_xml_fp, events=("start", "end"))

    # get the root element
    event, root = next(context)

    n_tces = len(tces)
    tce_i = 0  # counter for TCEs in the DV XML
    data = {}
    for event, elem in context:

        n_tces_added = len(data)

        if event == "end" and elem.tag == "planetResults":  # iterate through each planet results container

            tce_i += 1

            uid = f'{elem.attrib["keplerId"]}-{elem.attrib["planetNumber"]}'

            if tce_i % 500 == 0:
                print(f'[{proc_id}] Iterating over TCE {tce_i} in {dv_xml_fp.name}', flush=True)

            if n_tces_added == n_tces:  # stop reading XML file once all TCEs were iterated through
                break

            if uid not in tces.index:
                continue

            logger.info(f'[{proc_id}] Getting difference image data for TCE KIC {uid}... ({n_tces_added}/{n_tces} '
                        f'TCEs)')

            kmag = float([el for el in root if 'keplerMag' in el.tag][0].attrib['value'])

            data[uid] = {
                'target_ref_centroid': [],
                'image_data': [],
                'image_number': [],
                'mag': kmag,
                'neighbor_data': None,
                'quality_metric': [],
            }

            # get difference image results
            diff_img_res = [el for el in elem if el.tag == 'differenceImageResults']

            n_quarters = len(diff_img_res)

            if n_quarters < N_QUARTERS_KEPLER:
                logger.info(f'[{proc_id}] TCE KIC {uid} has less than {N_QUARTERS_KEPLER} quarters ({n_quarters})')

            # iterate over quarters
            for quarter_i in range(n_quarters):

                img_res_q = diff_img_res[quarter_i]

                # get quality metric data
                q_metric_q = [el.attrib for el in img_res_q if 'qualityMetric' in el.tag][0]
                data[uid]['quality_metric'].append(q_metric_q)

                # get quarter information
                data[uid]['image_number'].append(int(img_res_q.attrib['quarter']))

                img_px_data = img_res_q.findall('differenceImagePixelData')

                # n_pxs = len(img_px_data)

                px_dict = {(int(el.attrib['ccdRow']), int(el.attrib['ccdColumn'])): list(el) for el in img_px_data}

                # get max and min row and col
                px_row_lst, px_col_lst = [], []
                for px_row, px_col in px_dict.keys():
                    px_row_lst.append(px_row)
                    px_col_lst.append(px_col)

                min_row, max_row = min(px_row_lst), max(px_row_lst)
                min_col, max_col = min(px_col_lst), max(px_col_lst)

                # determine size of images
                row_size = max_row - min_row + 1
                col_size = max_col - min_col + 1

                # populate array with pixel values
                diff_imgs = np.nan * np.ones((row_size, col_size, N_IMGS_IN_DIFF, 2), dtype='float')

                for px_coord, diff_imgs_q in px_dict.items():
                    diff_imgs[px_coord[0] - min_row, px_coord[1] - min_col, :, 0] = [float(el.attrib['value'])
                                                                                     for el in diff_imgs_q]
                    diff_imgs[px_coord[0] - min_row, px_coord[1] - min_col, :, 1] = [float(el.attrib['uncertainty'])
                                                                                     for el in diff_imgs_q]

                # get target position in pixel frame
                kic_centroid_ref = img_res_q.findall('kicReferenceCentroid')[0]
                # check for missing value
                if float(kic_centroid_ref.find('column').attrib['uncertainty']) == -1 or \
                        float(kic_centroid_ref.find('row').attrib['uncertainty']) == -1:
                    kic_centroid_ref_dict = {
                        'col': {k: float(v) for k, v in kic_centroid_ref.find('column').attrib.items()},
                        'row': {k: float(v) for k, v in kic_centroid_ref.find('row').attrib.items()}
                    }
                    logger.info(f'[{proc_id}] TCE KIC {uid} has missing reference centroid for target in quarter '
                                f'{img_res_q.attrib["quarter"]}.')
                    # continue
                else:
                    kic_centroid_ref_dict = {
                        'col': {k: float(v) - min_col if k == 'value' else float(v)
                                for k, v in kic_centroid_ref.find('column').attrib.items()},
                        'row': {k: float(v) - min_row if k == 'value' else float(v)
                                for k, v in kic_centroid_ref.find('row').attrib.items()}
                    }

                # plot difference image
                if np.random.uniform() <= plot_prob:
                    plot_diff_img_data(diff_imgs,
                                       target_coords=(kic_centroid_ref_dict['col']['value'],
                                                      kic_centroid_ref_dict['row']['value']),
                                       plot_fp=plot_dir / f'kic_{uid}.png',
                                       neighbors_coords=None,
                                       logscale=True,
                                       target_mag=data[uid]['mag'],
                                       neighbors_mag=None,
                                       mag_sat=KEPLER_MAG_SAT,
                                       )

                data[uid]['target_ref_centroid'].append(kic_centroid_ref_dict)
                data[uid]['image_data'].append(diff_imgs)

        root.clear()

    return data


def get_data_from_kepler_dv_xml_multiproc(dv_xml_fp, tces, save_dir, plot_dir, plot_prob, log_dir, job_i):
    """ Wrapper for `get_data_from_kepler_dv_xml()`. Extract difference image data from the DV XML file for a set of
    Kepler Q1-Q17 DR25 TCEs.

    :param dv_xml_fp: Path, file path to DV XML file
    :param tces: pandas DataFrame, TCEs for which to extract difference image data. Must contain two columns: 'uid' and
    'label'. 'uid' must be of the pattern '{tic_id}-{tce_plnt_num}'
    :param save_dir: Path, save directory
    :param plot_dir: Path, plot directory
    :param plot_prob: float, probability to plot difference image for a given example ([0, 1])
    :param log_dir: Path, log directory
    :param job_i: int, job id
    :return:
    """

    # set up logger
    logger = logging.getLogger(name=f'extract_img_data_kepler_dv_xml-{job_i}')
    logger_handler = logging.FileHandler(filename=log_dir / f'extract_img_data_from_kepler_dv_xml-{job_i}.log',
                                         mode='w')
    logger_formatter = logging.Formatter('%(asctime)s - %(message)s')
    logger.setLevel(logging.INFO)
    logger_handler.setFormatter(logger_formatter)
    logger.addHandler(logger_handler)
    logger.info(f'[{job_i}] Starting run {dv_xml_fp.name} ({len(tces)} TCEs)...')

    data = get_data_from_kepler_dv_xml(dv_xml_fp, tces, plot_dir, plot_prob, logger)

    np.save(save_dir / f'keplerq1q17_dr25_diffimg_{job_i}.npy', data)


def get_neighbors_for_target_in_sector(sectors_obs, neighbors_dir, tic_id, sector_run_id, proc_id, logger):
    """ Get data on neighbors for target `tic_id` in sectors in `sectors_obs`.

    :param sectors_obs: list,
    :param neighbors_dir: Path, path to directory containing target neighbors data
    :param tic_id: str, target ID
    :param sector_run_id: str, sector run ID
    :param proc_id: int, process ID
    :param logger: logger

    :return: neighbors_lst, list of data for neighbors of target `tic_id`.
    """

    neighbors_lst = []  # list of neighbors table for target across observed sectors
    for sector_obs in sectors_obs:

        # load neighbors table for this sector
        neighbors_tbl = pd.read_csv(neighbors_dir / f'S{sector_obs}' / 'mapping_results' /
                                    f'neighbors_pxcoords_S{sector_obs}.csv')

        # filter neighbors for this target
        neighbors_target = neighbors_tbl.loc[neighbors_tbl['target_id'] == int(tic_id)]

        if len(neighbors_target) == 0:

            raise ValueError(f'[{proc_id}] [Sector run {sector_run_id}] Target {tic_id}'
                             f' not found in the neighbors table for sector {sector_obs}')

        neighbors_target = neighbors_target.set_index('ID')

        neighbors_lst.append(neighbors_target)

        logger.info(f'[{proc_id}] [Sector run {sector_run_id}] Found {len(neighbors_target)} neighbors for '
                    f'target {tic_id} in sector {sector_obs}.')

    return neighbors_lst


def get_data_from_tess_dv_xml(dv_xml_fp, neighbors_dir, sector_run_id, plot_dir, plot_prob, logger, proc_id=-1):
    """ Extract difference image data from the TESS target DV XML file for the set of TCEs detected in that star for
    that TESS SPOC sector run.

    :param dv_xml_fp: Path, filepath to DV XML file.
    :param neighbors_dir: Path, path to directory containing target neighbors data
    :param sector_run_id: str, sector run ID
    :param plot_dir: Path, plot directory
    :param plot_prob: float, probability to plot difference image for a given example ([0, 1])
    :param logger: logger
    :param proc_id: int, process ID

    :return: dict, each item is the difference image data for a given TCE. The TCE is identified by the string key
    '{tic_id}-{tce_plnt_num}-S{sector_run}'. The value is a dictionary that contains six items:
        - 'target_ref_centroid' is a list of dictionaries that contain the value and uncertainty for the reference
        coordinates of the target star in the pixel domain in each observed sector;
        - 'image_data' is a list of NumPy array (n_rows, n_cols, n_imgs, 2) that contains the in-transit,
        out-of-transit, difference, and SNR flux images in this order (pixel values and uncertainties are addressed by
        the last dimension of the array, in this order) for each observe sector;
        - 'image_number' is a list that contains the integer sector number of the corresponding sequence of difference
        image data extracted for the TCE.
        - 'mag' is the target's magnitude.
        - 'neighbor_data' is a list that, for each sector, contains a dictionary where each key is the TIC ID of
        neighboring objects that maps to a dictionary with the column 'col_px' and row 'row_px' coordinates of these
        objects in the CCD pixel frame of the target star along with the corresponding magnitude 'TMag' and distance to
        the target in arcseconds 'dst_arcsec'.
    """

    data = {}

    try:
        tree = et.parse(dv_xml_fp)
    except Exception as e:
        raise Exception(f'{proc_id}] [Sector run {sector_run_id}] Exception found when reading {dv_xml_fp}: {e}.')

    root = tree.getroot()

    tic_id = root.attrib['ticId']

    tmag = float([el for el in root if 'tessMag' in el.tag][0].attrib['value'])

    planet_res_lst = [el for el in root if 'planetResults' in el.tag]

    n_sectors_expected = root.attrib['sectorsObserved'].count('1')
    sectors_obs = [i for i, char in enumerate(root.attrib['sectorsObserved']) if char == '1']

    # get neighboring stars
    if neighbors_dir:
        logger.info(f'{proc_id}] [Sector run {sector_run_id}] Getting neighbors information for target {tic_id} in '
                    f'sectors {sectors_obs}...')
        neighbors_lst = get_neighbors_for_target_in_sector(sectors_obs, neighbors_dir, tic_id, sector_run_id, proc_id,
                                                           logger)

    n_tces = len(planet_res_lst)
    tce_i = 0
    for planet_res in planet_res_lst:

        tce_i += 1

        uid = f'{root.attrib["ticId"]}-{planet_res.attrib["planetNumber"]}-S{sector_run_id}'

        if neighbors_dir:
            # filter neighbors based on transit depth - could these objects cause the observed transit depth?
            tce_depth = max(0,
                            float(planet_res.find(
                                './/dv:modelParameter[@name="transitDepthPpm"]',
                                {'dv': 'http://www.nasa.gov/2018/TESS/DV'}).attrib['value'])) + 1
            tce_depth /= 1e6
            delta_mag = tmag - np.log10(tce_depth) * 2.5
            tce_neighbors_lst = []
            for neighbors_sector in neighbors_lst:
                tce_neighbors_sector = {}
                for neighbor_id, neighbor_data in neighbors_sector.iterrows():
                    if np.isnan(neighbor_data['row_px']) or np.isnan(neighbor_data['col_px']):
                        continue
                    if neighbor_data['Tmag'] <= delta_mag:
                        tce_neighbors_sector.update({neighbor_id: dict(neighbor_data)})
                tce_neighbors_lst.append(tce_neighbors_sector)

        data[uid] = {
            'target_ref_centroid': [],
            'image_data': [],
            'mag': tmag,
            'image_number': [],
            'quality_metric': [],
        }
        if neighbors_dir:
            data[uid]['neighbor_data'] = tce_neighbors_lst

        logger.info(f'[{proc_id}] [Sector run {sector_run_id}] Getting difference image data for TCE TIC '
                    f'{uid} ({tce_i}/{n_tces} TCEs)...')

        # get difference image results
        diff_img_res = [el for el in planet_res if 'differenceImageResults' in el.tag]

        n_sectors = len(diff_img_res)

        if n_sectors < n_sectors_expected:
            logger.info(f'[{proc_id}] [Sector run {sector_run_id}] TCE TIC {uid} has less than '
                        f'{n_sectors_expected} '
                        f'sectors ({n_sectors})')

        # iterate over sectors
        for sector_i in range(n_sectors):

            img_res_s = diff_img_res[sector_i]

            # get quality metric data
            q_metric_s = [el.attrib for el in img_res_s if 'qualityMetric' in el.tag][0]
            q_metric_s['value'] = float(q_metric_s['value'])
            q_metric_s['attempted'] = True if q_metric_s['attempted'] == 'true' else False
            q_metric_s['valid'] = True if q_metric_s['valid'] == 'true' else False
            data[uid]['quality_metric'].append(q_metric_s)

            # get sector id
            sector = int(img_res_s.attrib['sector'])
            data[uid]['image_number'].append(sector)

            # get difference image pixel data
            img_px_data = [el for el in img_res_s if 'differenceImagePixelData' in el.tag]

            # n_pxs = len(img_px_data)

            px_dict = {(int(el.attrib['ccdRow']), int(el.attrib['ccdColumn'])): list(el) for el in img_px_data}

            # get max and min row and col
            px_row_lst, px_col_lst = [], []
            for px_row, px_col in px_dict.keys():
                px_row_lst.append(px_row)
                px_col_lst.append(px_col)

            min_row, max_row = min(px_row_lst), max(px_row_lst)
            min_col, max_col = min(px_col_lst), max(px_col_lst)

            # determine size of images
            row_size = max_row - min_row + 1
            col_size = max_col - min_col + 1

            # populate array with pixel values
            diff_imgs = np.nan * np.ones((row_size, col_size, N_IMGS_IN_DIFF, 2), dtype='float')

            for px_coord, diff_imgs_q in px_dict.items():
                diff_imgs[px_coord[0] - min_row, px_coord[1] - min_col, :, 0] = [float(el.attrib['value'])
                                                                                 for el in diff_imgs_q]
                diff_imgs[px_coord[0] - min_row, px_coord[1] - min_col, :, 1] = [float(el.attrib['uncertainty'])
                                                                                 for el in diff_imgs_q]

            # get target position in pixel frame
            tic_centroid_ref = [el for el in img_res_s if 'ticReferenceCentroid' in el.tag][0]
            tic_centroid_ref_col = [el for el in tic_centroid_ref if 'col' in el.tag][0].attrib
            tic_centroid_ref_row = [el for el in tic_centroid_ref if 'row' in el.tag][0].attrib

            # check for missing value
            if float(tic_centroid_ref_col['uncertainty']) == -1 or \
                    float(tic_centroid_ref_row['uncertainty']) == -1:
                tic_centroid_ref_dict = {
                    'col': {k: float(v) if k == 'value' else float(v)
                            for k, v in tic_centroid_ref_col.items()},
                    'row': {k: float(v) if k == 'value' else float(v)
                            for k, v in tic_centroid_ref_row.items()}
                }
                logger.info(f'[{proc_id}] [Sector run {sector_run_id}] TCE TIC {uid} has missing reference '
                            f'centroid for target in sector {img_res_s.attrib["sector"]}.')
                # continue
            else:
                tic_centroid_ref_dict = {
                    'col': {k: float(v) - min_col if k == 'value' else float(v)
                            for k, v in tic_centroid_ref_col.items()},
                    'row': {k: float(v) - min_row if k == 'value' else float(v)
                            for k, v in tic_centroid_ref_row.items()}
                }

            # plot difference image
            if np.random.uniform() <= plot_prob:

                if neighbors_dir:
                    neighbors_coords = [(neighbor_data['col_px'], neighbor_data['row_px'])
                                        for _, neighbor_data in data[uid]['neighbor_data'][sector_i].items()]
                    neighbors_mags = [neighbor_data['Tmag']
                                      for _, neighbor_data in tce_neighbors_lst[sector_i].items()]
                else:
                    neighbors_coords, neighbors_mags, delta_mag = None, None, None

                plot_diff_img_data(diff_imgs,
                                   plot_dir / f'tic_{uid}_sector_{sector}.png',
                                   target_coords=(tic_centroid_ref_dict['col']['value'],
                                                  tic_centroid_ref_dict['row']['value'])
                                   if tic_centroid_ref_dict['col']['uncertainty'] != -1 else None,
                                   neighbors_coords=neighbors_coords,
                                   logscale=True,
                                   target_mag=data[uid]['mag'],
                                   neighbors_mag=neighbors_mags,
                                   mag_sat=TESS_MAG_SAT,
                                   min_mag=delta_mag,
                                   )

            data[uid]['target_ref_centroid'].append(tic_centroid_ref_dict)
            data[uid]['image_data'].append(diff_imgs)

    return data


def get_data_from_tess_dv_xml_multiproc(dv_xml_run, save_dir, neighbors_dir, plot_dir, plot_prob, log_dir, job_i,
                                        check_existence_multiple_versions=False):
    """ Wrapper for `get_data_from_tess_dv_xml()`. Extract difference image data from the DV XML files for a TESS sector
    run.

    :param dv_xml_run: Path, path to sector run with DV XML files.
    :param save_dir: Path, save directory
    :param neighbors_dir: Path, path to directory containing target neighbors data
    :param plot_dir: Path, plot directory
    :param plot_prob: float, probability to plot difference image for a given example ([0, 1])
    :param log_dir: Path, log directory
    :param job_i: int, job id
    :param check_existence_multiple_versions: bool whether to check existence of multiple versions (different runs) of
        DV

    :return:
    """

    # set up logger
    logger = logging.getLogger(name=f'extract_img_data_tess_dv_xml_{dv_xml_run.name}')
    logger_handler = logging.FileHandler(filename=log_dir / f'extract_img_data_from_tess_dv_xml-{dv_xml_run.name}.log',
                                         mode='w')
    logger_formatter = logging.Formatter('%(asctime)s - %(message)s')
    logger.setLevel(logging.INFO)
    logger_handler.setFormatter(logger_formatter)
    logger.addHandler(logger_handler)
    logger.info(f'[{job_i}] Starting run {dv_xml_run.name}...')

    proc_id = os.getpid()

    data = {}

    # get filepaths to xml files
    dv_xml_run_fps = list(dv_xml_run.rglob(f"*.xml"))

    n_targets = len(dv_xml_run_fps)
    logger.info(f'[{proc_id}] Found {n_targets} targets DV xml files in {dv_xml_run}.')

    for target_i, dv_xml_fp in enumerate(dv_xml_run_fps):

        # get sector run ID from filename
        s_sector, e_sector = re.findall('-s[0-9]+', dv_xml_fp.stem)
        s_sector, e_sector = int(s_sector[2:]), int(e_sector[2:])
        if s_sector != e_sector:  # multisector run
            sector_run_id = f'{s_sector}-{e_sector}'
        else:
            sector_run_id = f'{s_sector}'

        if target_i % 1000 == 0:
            logger.info(f'[{proc_id}] [Sector run {sector_run_id}] Iterating over TIC {target_i}/{n_targets} in '
                        f'{dv_xml_fp.name}.')
        try:
            # check if there are results for more than one processing run for this TIC and sector run
            if check_existence_multiple_versions:
                tic_id = re.findall('\d{16}', dv_xml_fp.name)[0]  # get tic id from filename

                tic_drs = [fp for fp in dv_xml_run.glob(f'*{tic_id}*')]
                if len(tic_drs) > 1:
                    curr_dr = int(dv_xml_fp.stem.split('-')[-1][:-4])
                    latest_dr = sorted([int(fp.stem.split('-')[-1][:-4])
                                        for fp in dv_xml_run.glob(f'*{tic_id}*')])[-1]
                    if curr_dr != latest_dr:
                        logger.info(f'[{proc_id}] [Sector run {sector_run_id}] '
                                    f'Skipping {dv_xml_fp.name} for TIC {int(tic_id)} since there is '
                                    f'more recent processed results (current release {curr_dr}, latest release {latest_dr})'
                                    f'... ({target_i}/{n_targets} targets)')
                        continue

            data_dv_xml = get_data_from_tess_dv_xml(dv_xml_fp, neighbors_dir, sector_run_id, plot_dir, plot_prob,
                                                    logger, proc_id)
            data.update(data_dv_xml)
        except Exception as e:
            logger.info(f'[{job_i}] Exception occurred when getting data from {dv_xml_fp.name}:\n {e}')
            continue

    np.save(save_dir / f'tess_diffimg_{dv_xml_run.name}.npy', data)

    logger.info(f'[{job_i}] Finished run {dv_xml_run.name}.')
