"""
Extracting difference image data from the Kepler Q1-Q17 DR25 DV XML file.
"""

# 3rd party
import xml.etree.cElementTree as et
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import multiprocessing

plt.switch_backend('agg')


def get_data_from_dv_xml(dv_xml_fp, tces, save_dir, plot_prob):
    """ Extract difference image data from the DV XML file for a set of Kepler Q1-Q17 DR25 TCEs.

    :param dv_xml_fp: Path, file path to DV XML file
    :param tces: pandas DataFrame, TCEs for which to extract difference image data. Must contain two columns: 'uid' and
    'label'. 'uid' must be of the pattern '{tic_id}-{tce_plnt_num}-S{sector_run}'
    :param save_dir: Path, save directory
    :param plot_prob: float, probability to plot difference image for a given example ([0, 1])
    :return: dict, each item is the difference image data for a given TCE. The TCE is identified by the string key
    '{tic_id}-{tce_plnt_num}-S{sector_run}'. The value is a dictionary that contains two items: 'target_ref_centroid'
    is a dictionary that contains the value and uncertainty for the reference coordinates of the target star in the
    pixel domain; 'image_data' is a NumPy array (n_rows, n_cols, n_imgs, 2) that contains the in-transit,
    out-of-transit, difference, and SNR flux images in this order (pixel values and uncertainties are addressed by the
    last dimension of the array, in this order).
    """

    N_QUARTERS_KEPLER = 17
    N_IMGS_IN_DIFF = 4  # diff, oot, it, snr

    proc_id = os.getpid()

    # set up logger
    logger = logging.getLogger(name=f'extract_img_data_kepler_dv_xml_pid-{proc_id}')
    logger_handler = logging.FileHandler(filename=save_dir / f'extract_img_data_from_kepler_dv_xml-{proc_id}.log',
                                         mode='w')
    logger_formatter = logging.Formatter('%(asctime)s - %(message)s')
    logger.setLevel(logging.INFO)
    logger_handler.setFormatter(logger_formatter)
    logger.addHandler(logger_handler)
    logger.info(f'[{proc_id}] Starting run...')

    # get an iterable
    context = et.iterparse(dv_xml_fp, events=("start", "end"))

    # turn it into an iterator
    context = iter(context)

    n_tces = len(tces)
    tce_i = 0
    data = {}
    for event, elem in context:

        if event == "end" and elem.tag == "planetResults":  # iterate through each planet results container

            tce_i += 1

            uid = f'{elem.attrib["keplerId"]}-{elem.attrib["planetNumber"]}'

            if tce_i % 500 == 0:
                print(f'[{proc_id}] TCE {tce_i}', flush=True)

            if uid not in tces.index:
                continue

            logger.info(f'[{proc_id}] Getting difference image data for TCE KIC {uid}... ({tce_i}/{n_tces} TCEs)')

            data[uid] = {
                'target_ref_centroid': [],
                'image_data': []
            }

            # get difference image results
            diff_img_res = [el for el in elem if el.tag == 'differenceImageResults']

            n_quarters = len(diff_img_res)

            if n_quarters < N_QUARTERS_KEPLER:
                logger.info(f'TCE KIC {uid} has less than {N_QUARTERS_KEPLER} quarters ({n_quarters})')

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

    np.save(save_dir / f'keplerq1q17_dr25_diffimg_pid{proc_id}.npy', data)

    return data


if __name__ == '__main__':

    # DV XML file path
    dv_xml_fp = Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/data/fits_files/kepler/q1_q17_dr25/dv/kplr20160128150956_dv.xml')
    # TCE table file path
    tce_tbl_fp = Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/data/ephemeris_tables/kepler/q1-q17_dr25/11-17-2021_1243/q1_q17_dr25_tce_2020.09.28_10.36.22_stellar_koi_cfp_norobovetterlabels_renamedcols_nomissingval_symsecphase_cpkoiperiod_rba_cnt0n_valpc_modelchisqr_ruwe_magcat_uid.csv')
    # run directory
    run_dir = Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/data/fits_files/kepler/q1_q17_dr25/dv/preprocessing/8-16-2022_1147')

    # create run directory
    run_dir.mkdir(exist_ok=True, parents=True)

    # load TCE table to get examples for which the image data are going to be extracted
    tce_tbl = pd.read_csv(tce_tbl_fp, usecols=['uid', 'label'])
    tce_tbl.set_index('uid', inplace=True)
    tces = tce_tbl[:250]

    # creating plotting directory
    plot_dir = run_dir / 'plots'
    plot_dir.mkdir(exist_ok=True)
    plot_prob = 0.01

    n_processes = 4
    pool = multiprocessing.Pool(processes=n_processes)
    n_jobs = 4
    tces_jobs = np.array_split(tces, n_jobs)
    jobs = [(dv_xml_fp, tces_job, run_dir, plot_prob) for tces_job in tces_jobs]
    async_results = [pool.apply_async(get_data_from_dv_xml, job) for job in jobs]
    pool.close()

    data = {}
    for async_res in async_results:
        data.update(async_res.get())

    # data = get_data_from_dv_xml(dv_xml_fp, tces, plot_prob, plot_dir)

    np.save(run_dir / 'keplerq1q17_dr25_diffimg.npy', data)
