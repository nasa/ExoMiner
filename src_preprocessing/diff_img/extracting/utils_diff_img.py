""" Utility functions for extracting difference imaging. """

# 3rd party
import xml.etree.cElementTree as et
import os
import matplotlib.pyplot as plt
import numpy as np
import logging
from mpl_toolkits.axes_grid1 import make_axes_locatable
import re

plt.switch_backend('agg')

N_QUARTERS_KEPLER = 17
N_IMGS_IN_DIFF = 4  # diff, oot, it, snr


def plot_diff_img_data(diff_imgs, target_col, target_row, uid, plot_dir, tce_lbl=None):
    """ Plot difference image data for TCE in a given quarter/sector.

    Args:
        diff_imgs: NumPy array, difference image data [cols, rows, it|oot|diff|snr, value| uncertainty]
        target_col: float, target column location
        target_row: float, target row location
        uid: str, TCE ID
        plot_dir: Path, plot directory
        tce_lbl: str, TCE label

    Returns:

    """

    f, ax = plt.subplots(2, 2, figsize=(12, 8))
    # diff img
    im = ax[0, 0].imshow(diff_imgs[:, :, 2, 0])
    divider = make_axes_locatable(ax[0, 0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    ax[0, 0].scatter(target_col, target_row,
                     marker='x',
                     color='r', label='Target')
    ax[0, 0].set_ylabel('Row')
    ax[0, 0].set_xlabel('Col')
    ax[0, 0].legend()
    ax[0, 0].set_title('Difference Flux (e-/cadence)')
    # oot img
    im = ax[0, 1].imshow(diff_imgs[:, :, 1, 0])
    divider = make_axes_locatable(ax[0, 1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    ax[0, 1].scatter(target_col, target_row,
                     marker='x',
                     color='r', label='Target')
    ax[0, 1].set_ylabel('Row')
    ax[0, 1].set_xlabel('Col')
    ax[0, 1].legend()
    ax[0, 1].set_title('Out of Transit Flux (e-/cadence)')
    # it img
    im = ax[1, 0].imshow(diff_imgs[:, :, 0, 0])
    divider = make_axes_locatable(ax[1, 0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    ax[1, 0].scatter(target_col, target_row,
                     marker='x',
                     color='r', label='Target')
    ax[1, 0].set_ylabel('Row')
    ax[1, 0].set_xlabel('Col')
    ax[1, 0].legend()
    ax[1, 0].set_title('In Transit Flux (e-/cadence)')
    # snr img
    im = ax[1, 1].imshow(diff_imgs[:, :, 3, 0])
    divider = make_axes_locatable(ax[1, 1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    ax[1, 1].scatter(target_col, target_row,
                     marker='x',
                     color='r', label='Target')
    ax[1, 1].set_ylabel('Row')
    ax[1, 1].set_xlabel('Col')
    ax[1, 1].legend()
    ax[1, 1].set_title('Difference SNR')

    f.suptitle(f'TCE {uid} {tce_lbl if not None else ""}')
    f.tight_layout()
    f.savefig(plot_dir / f'{uid}_diff_img.png')
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

            data[uid] = {
                'target_ref_centroid': [],
                'image_data': [],
                'image_number': [],
            }

            # get difference image results
            diff_img_res = [el for el in elem if el.tag == 'differenceImageResults']

            n_quarters = len(diff_img_res)

            if n_quarters < N_QUARTERS_KEPLER:
                logger.info(f'[{proc_id}] TCE KIC {uid} has less than {N_QUARTERS_KEPLER} quarters ({n_quarters})')

            # iterate over quarters
            for quarter_i in range(n_quarters):

                img_res_q = diff_img_res[quarter_i]

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

                    # plot difference image with target location
                    if np.random.uniform() <= plot_prob:
                        plot_diff_img_data(diff_imgs,
                                           kic_centroid_ref_dict['col']['value'],
                                           kic_centroid_ref_dict['row']['value'],
                                           uid,
                                           plot_dir)

                data[uid]['target_ref_centroid'].append(kic_centroid_ref_dict)
                data[uid]['image_data'].append(diff_imgs)

        root.clear()

    # np.save(save_dir / f'keplerq1q17_dr25_diffimg_pid{proc_id}.npy', data)

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


def get_data_from_tess_dv_xml(dv_xml_run, plot_dir, plot_prob, logger):
    """ Extract difference image data from the DV XML files for a TESS sector run.

    :param dv_xml_run: Path, path to sector run with DV XML files.
    :param plot_dir: Path, plot directory
    :param plot_prob: float, probability to plot difference image for a given example ([0, 1])
    :param logger: logger

    :return: dict, each item is the difference image data for a given TCE. The TCE is identified by the string key
    '{tic_id}-{tce_plnt_num}-S{sector_run}. The value is a dictionary that contains two items: 'target_ref_centroid' is
    a dictionary that contains the value and uncertainty for the reference coordinates of the target star in the pixel
    domain; 'image_data' is a NumPy array (n_rows, n_cols, n_imgs, 2) that contains the in-transit, out-of-transit,
    difference, and SNR flux images in this order (pixel values and uncertainties are addressed by the last dimension
    of the array, in this order); 'image_number' is a list that contains the integer sector number of the corresponding
    sequence of difference image data extracted for the TCE.
    """

    proc_id = os.getpid()

    data = {}

    # get filepaths to xml files
    dv_xml_run_fps = list(dv_xml_run.rglob("*.xml"))

    s_sector, e_sector = re.findall('-s[0-9]+', dv_xml_run_fps[0].stem)
    s_sector, e_sector = int(s_sector[2:]), int(e_sector[2:])
    if s_sector != e_sector:  # multisector run
        sector_run_id = f'{s_sector}-{e_sector}'
    else:
        sector_run_id = f'{s_sector}'

    n_targets = len(dv_xml_run_fps)
    logger.info(f'[{proc_id}] [Sector run {sector_run_id}] Found {n_targets} targets DV xml files in {dv_xml_run}.')

    for target_i, dv_xml_fp in enumerate(dv_xml_run_fps):

        if target_i % 1000 == 0:
            logger.info(f'[{proc_id}] [Sector run {sector_run_id}] Iterating over TIC {target_i}/{n_targets} in '
                        f'{dv_xml_fp.name}.')

        try:
            tree = et.parse(dv_xml_fp)
        except Exception as e:
            logger.info(f'Exception found when reading {dv_xml_fp}: {e}.')
            continue
        root = tree.getroot()

        # check if there are results for more than one processing run for this TIC and sector run
        tic_id = root.attrib['ticId']
        # tic_drs = [int(fp.stem.split('-')[-1][:-4]) for fp in dv_xml_run.glob(f'*{tic_id.zfill(16)}*')]
        tic_drs = [fp for fp in dv_xml_run.glob(f'*{tic_id.zfill(16)}*')]
        if len(tic_drs) > 1:
            curr_dr = int(dv_xml_fp.stem.split('-')[-1][:-4])
            latest_dr = sorted([int(fp.stem.split('-')[-1][:-4])
                                for fp in dv_xml_run.glob(f'*{tic_id.zfill(16)}*')])[-1]
            if curr_dr != latest_dr:
                logger.info(f'[{proc_id}] [Sector run {sector_run_id}] '
                            f'Skipping {dv_xml_fp.name} for TIC {tic_id} since there is '
                            f'more recent processed results (current release {curr_dr}, latest release {latest_dr})'
                            f'... ({target_i}/{n_targets} targets)')
                continue

        planet_res_lst = [el for el in root if 'planetResults' in el.tag]

        # sectors_obs = root.attrib['sectorsObserved']
        # first_sector_obs, last_sector_obs = sectors_obs.find('1'), sectors_obs.rfind('1')
        n_sectors_expected = root.attrib['sectorsObserved'].count('1')

        n_tces = len(planet_res_lst)
        tce_i = 0

        for planet_res in planet_res_lst:

            tce_i += 1

            uid = f'{root.attrib["ticId"]}-{planet_res.attrib["planetNumber"]}-S{sector_run_id}'

            data[uid] = {
                'target_ref_centroid': [],
                'image_data': [],
                # 'tic_tess_mag': tmag,
                'image_number': []
            }

            logger.info(f'[{proc_id}] [Sector run {sector_run_id}] Getting difference image data for TCE TIC '
                        f'{uid} ({tce_i}/{n_tces} TCEs)... ({target_i}/{n_targets} targets)')

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

                data[uid]['image_number'].append(int(img_res_s.attrib['sector']))

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

                    # plot difference image with target location
                    if np.random.uniform() <= plot_prob:
                        plot_diff_img_data(diff_imgs,
                                           tic_centroid_ref_dict['col']['value'],
                                           tic_centroid_ref_dict['row']['value'],
                                           uid,
                                           plot_dir,
                                           )

                data[uid]['target_ref_centroid'].append(tic_centroid_ref_dict)
                data[uid]['image_data'].append(diff_imgs)

    # np.save(save_dir / f'tess_diffimg_pid{proc_id}.npy', data)

    return data


def get_data_from_tess_dv_xml_multiproc(dv_xml_run, save_dir, plot_dir, plot_prob, log_dir, job_i):
    """ Wrapper for `get_data_from_tess_dv_xml()`. Extract difference image data from the DV XML files for a TESS sector
    run.

     :param dv_xml_run: Path, path to sector run with DV XML files.
    :param save_dir: Path, save directory
    :param plot_dir: Path, plot directory
    :param plot_prob: float, probability to plot difference image for a given example ([0, 1])
    :param log_dir: Path, log directory
    :param job_i: int, job id

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

    data = get_data_from_tess_dv_xml(dv_xml_run, plot_dir, plot_prob, logger)

    logger.info(f'[{job_i}] Finished run {dv_xml_run.name}.')

    np.save(save_dir / f'tess_diffimg_{dv_xml_run.name}.npy', data)
