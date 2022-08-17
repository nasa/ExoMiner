""" Utility functions for processing difference imaging. """

# 3rd party
import xml.etree.cElementTree as et
import os
import matplotlib.pyplot as plt
import numpy as np
import logging

plt.switch_backend('agg')


def get_data_from_kepler_dv_xml(dv_xml_fp, tces, save_dir, plot_prob, logger):
    """ Extract difference image data from the DV XML file for a set of Kepler Q1-Q17 DR25 TCEs.

    :param dv_xml_fp: Path, file path to DV XML file
    :param tces: pandas DataFrame, TCEs for which to extract difference image data. Must contain two columns: 'uid' and
    'label'. 'uid' must be of the pattern '{tic_id}-{tce_plnt_num}'
    :param save_dir: Path, save directory
    :param plot_prob: float, probability to plot difference image for a given example ([0, 1])
    :param logger: logger
    :return: dict, each item is the difference image data for a given TCE. The TCE is identified by the string key
    '{tic_id}-{tce_plnt_num}'. The value is a dictionary that contains two items: 'target_ref_centroid'
    is a dictionary that contains the value and uncertainty for the reference coordinates of the target star in the
    pixel domain; 'image_data' is a NumPy array (n_rows, n_cols, n_imgs, 2) that contains the in-transit,
    out-of-transit, difference, and SNR flux images in this order (pixel values and uncertainties are addressed by the
    last dimension of the array, in this order).
    """

    N_QUARTERS_KEPLER = 17
    N_IMGS_IN_DIFF = 4  # diff, oot, it, snr

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
                print(f'[{proc_id}] Iterating over TCE {tce_i}', flush=True)

            if n_tces_added == n_tces:  # stop reading XML file once all TCEs were iterated through
                break

            if uid not in tces.index:
                continue

            logger.info(f'[{proc_id}] Getting difference image data for TCE KIC {uid}... ({n_tces_added}/{n_tces} '
                        f'TCEs)')

            data[uid] = {
                'target_ref_centroid': [],
                'image_data': []
            }

            # get difference image results
            diff_img_res = [el for el in elem if el.tag == 'differenceImageResults']

            n_quarters = len(diff_img_res)

            if n_quarters < N_QUARTERS_KEPLER:
                logger.info(f'[{proc_id}] TCE KIC {uid} has less than {N_QUARTERS_KEPLER} quarters ({n_quarters})')

            # iterate over quarters
            for quarter_i in range(n_quarters):

                img_res_q = diff_img_res[quarter_i]

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
                        f, ax = plt.subplots()
                        ax.imshow(diff_imgs[:, :, 2, 0])
                        ax.scatter(kic_centroid_ref_dict['col']['value'], kic_centroid_ref_dict['row']['value'],
                                   marker='x',
                                   color='r', label='Target')
                        ax.set_ylabel('Row')
                        ax.set_xlabel('Col')
                        ax.set_title(f'KIC {uid} {tces.loc[uid]["label"]}')
                        ax.legend()
                        f.savefig(save_dir / 'plots' / f'{uid}_diff_img.png')
                        plt.close()

                data[uid]['target_ref_centroid'].append(kic_centroid_ref_dict)
                data[uid]['image_data'].append(diff_imgs)

        root.clear()

    # np.save(save_dir / f'keplerq1q17_dr25_diffimg_pid{proc_id}.npy', data)

    return data


def get_data_from_kepler_dv_xml_multiproc(dv_xml_fp, tces, save_dir, plot_prob, job_i):
    """ Wrapper for `get_data_from_kepler_dv_xml()`. Extract difference image data from the DV XML file for a set of
    Kepler Q1-Q17 DR25 TCEs.

    :param dv_xml_fp: Path, file path to DV XML file
    :param tces: pandas DataFrame, TCEs for which to extract difference image data. Must contain two columns: 'uid' and
    'label'. 'uid' must be of the pattern '{tic_id}-{tce_plnt_num}'
    :param save_dir: Path, save directory
    :param plot_prob: float, probability to plot difference image for a given example ([0, 1])\
    :param job_i: int, job id
    :return:
    """

    # set up logger
    log_dir = save_dir / 'logs'
    log_dir.mkdir(exist_ok=True)
    logger = logging.getLogger(name=f'extract_img_data_kepler_dv_xml-{job_i}')
    logger_handler = logging.FileHandler(filename=log_dir / f'extract_img_data_from_kepler_dv_xml-{job_i}.log',
                                         mode='w')
    logger_formatter = logging.Formatter('%(asctime)s - %(message)s')
    logger.setLevel(logging.INFO)
    logger_handler.setFormatter(logger_formatter)
    logger.addHandler(logger_handler)
    logger.info(f'[{job_i}] Starting run...')

    data = get_data_from_kepler_dv_xml(dv_xml_fp, tces, save_dir, plot_prob, logger)

    np.save(save_dir / f'keplerq1q17_dr25_diffimg_{job_i}.npy', data)


def get_data_from_tess_dv_xml(dv_xml_run, save_dir, plot_prob, tce_tbl, logger):
    """ Extract difference image data from the DV XML files for a TESS sector run.

    :param dv_xml_run: Path, path to sector run with DV XML files.
    :param save_dir: Path, save directory
    :param plot_prob: float, probability to plot difference image for a given example ([0, 1])
    :param tce_tbl: pandas DataFrame, TCE table
    :param logger: logger
    :return: dict, each item is the difference image data for a given TCE. The TCE is identified by the string key
    '{tic_id}-{tce_plnt_num}-S{sector_run}. The value is a dictionary that contains two items: 'target_ref_centroid' is
    a dictionary that contains the value and uncertainty for the reference coordinates of the target star in the pixel
    domain; 'image_data' is a NumPy array (n_rows, n_cols, n_imgs, 2) that contains the in-transit, out-of-transit,
    difference, and SNR flux images in this order (pixel values and uncertainties are addressed by the last dimension
    of the array, in this order).
    """

    N_IMGS_IN_DIFF = 4  # diff, oot, it, snr

    proc_id = os.getpid()

    data = {}
    target_i = 0
    dv_xml_run_fps = list(dv_xml_run.iterdir())
    n_targets = len(dv_xml_run_fps)
    for dv_xml_fp in dv_xml_run_fps:

        target_i += 1

        if target_i % 1000 == 0:
            print(f'[{proc_id}] Iterating over TIC {target_i}/{n_targets}', flush=True)

        tree = et.parse(dv_xml_fp)
        root = tree.getroot()
        planet_res_lst = [el for el in root if 'planetResults' in el.tag]

        sector_run = root.attrib['sectorsObserved']
        first_sector, last_sector = sector_run.find('1'), sector_run.rfind('1')
        n_sectors_expected = int(last_sector) - int(first_sector) + 1
        if first_sector == last_sector:  # single-sector run
            sector_run_id = first_sector
        else:  # multi-sector run
            sector_run_id = f'{first_sector}-{last_sector}'

        n_tces = len(planet_res_lst)
        tce_i = 0

        for planet_res in planet_res_lst:

            tce_i += 1

            uid = f'{root.attrib["ticId"]}-{planet_res.attrib["planetNumber"]}-S{sector_run_id}'

            data[uid] = {
                'target_ref_centroid': [],
                'image_data': []
            }

            logger.info(f'[{proc_id}] [{sector_run_id}] Getting difference image data for TCE TIC '
                        f'{uid} ({tce_i}/{n_tces} TCEs)... ({target_i}/{n_targets} targets)')

            # get difference image results
            diff_img_res = [el for el in planet_res if 'differenceImageResults' in el.tag]

            n_sectors = len(diff_img_res)

            if n_sectors < n_sectors_expected:
                logger.info(f'[{proc_id}] TCE TIC {uid} has less than {n_sectors_expected} sectors ({n_sectors})')

            # iterate over sectors
            for sector_i in range(n_sectors):

                img_res_s = diff_img_res[sector_i]

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
                        'col': tic_centroid_ref_col,
                        'row': tic_centroid_ref_row
                    }
                    logger.info(f'[{proc_id}] [{sector_run_id}] TCE TIC {uid} has missing reference centroid for '
                                f'target in sector {img_res_s.attrib["sector"]}.')
                    # continue
                else:
                    tic_centroid_ref_dict = {
                        'col': {k: float(v) - min_col if k == 'value' else float(v)
                                for k, v in tic_centroid_ref_col.items()},
                        'row': {k: float(v) - min_row if k == 'value' else float(v)
                                for k, v in tic_centroid_ref_row.items()}
                    }

                    # plot difference image with target location
                    if np.random.uniform() <= plot_prob and uid in tce_tbl.index:
                        f, ax = plt.subplots()
                        ax.imshow(diff_imgs[:, :, 2, 0])
                        ax.scatter(tic_centroid_ref_dict['col']['value'], tic_centroid_ref_dict['row']['value'],
                                   marker='x',
                                   color='r', label='Target')
                        ax.set_ylabel('Row')
                        ax.set_xlabel('Col')
                        ax.set_title(f'TIC {uid} {tce_tbl.loc[uid]["label"]}')
                        ax.legend()
                        f.savefig(save_dir / 'plots' / f'{uid}_diff_img.png')
                        plt.close()

                data[uid]['target_ref_centroid'].append(tic_centroid_ref_dict)
                data[uid]['image_data'].append(diff_imgs)

    # np.save(save_dir / f'tess_diffimg_pid{proc_id}.npy', data)

    return data


def get_data_from_tess_dv_xml_multiproc(dv_xml_run, save_dir, plot_prob, tce_tbl, job_i):
    """ Wrapper for `get_data_from_tess_dv_xml()`. Extract difference image data from the DV XML files for a TESS sector
    run.

     :param dv_xml_run: Path, path to sector run with DV XML files.
    :param save_dir: Path, save directory
    :param plot_prob: float, probability to plot difference image for a given example ([0, 1])
    :param tce_tbl: pandas DataFrame, TCE table
    :param job_i: int, job id
    :return:
    """

    # set up logger
    log_dir = save_dir / 'logs'
    log_dir.mkdir(exist_ok=True)
    logger = logging.getLogger(name=f'extract_img_data_tess_dv_xml_{job_i}')
    logger_handler = logging.FileHandler(filename=log_dir / f'extract_img_data_from_tess_dv_xml-{job_i}.log',
                                         mode='w')
    logger_formatter = logging.Formatter('%(asctime)s - %(message)s')
    logger.setLevel(logging.INFO)
    logger_handler.setFormatter(logger_formatter)
    logger.addHandler(logger_handler)
    logger.info(f'[{job_i}] Starting run...')

    data = get_data_from_tess_dv_xml(dv_xml_run, save_dir, plot_prob, tce_tbl, logger)

    np.save(save_dir / f'tess_diffimg_{dv_xml_run.name}.npy', data)
