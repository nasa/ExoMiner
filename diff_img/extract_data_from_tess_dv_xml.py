"""
Extracting difference image data from the TESS DV XML files.
"""

# 3rd party
import xml.etree.cElementTree as et
import os
import matplotlib.pyplot as plt
# import pandas as pd
import numpy as np
from pathlib import Path
import logging
import multiprocessing
import pandas as pd

plt.switch_backend('agg')


def get_data_from_dv_xml(dv_xml_run, save_dir, plot_prob, tce_tbl):
    """ Extract difference image data from the DV XML files for a TESS sector run.

    :param dv_xml_run: Path, path to sector run with DV XML files.
    :param save_dir: Path, save directory
    :param plot_prob: float, probability to plot difference image for a given example ([0, 1])
    :param tce_tbl: pandas DataFrame, TCE table
    :return: dict, each item is the difference image data for a given TCE. The TCE is identified by the string key
    '{tic_id}-{tce_plnt_num}-S{sector_run}. The value is a dictionary that contains two items: 'target_ref_centroid' is
    a dictionary that contains the value and uncertainty for the reference coordinates of the target star in the pixel
    domain; 'image_data' is a NumPy array (n_rows, n_cols, n_imgs, 2) that contains the in-transit, out-of-transit,
    difference, and SNR flux images in this order (pixel values and uncertainties are addressed by the last dimension
    of the array, in this order).
    """

    # plt.switch_backend('agg')

    N_IMGS_IN_DIFF = 4  # diff, oot, it, snr

    proc_id = os.getpid()

    # set up logger
    logger = logging.getLogger(name=f'extract_img_data_tess_dv_xml_pid-{proc_id}')
    logger_handler = logging.FileHandler(filename=save_dir / f'extract_img_data_from_tess_dv_xml-{proc_id}.log',
                                         mode='w')
    logger_formatter = logging.Formatter('%(asctime)s - %(message)s')
    logger.setLevel(logging.INFO)
    logger_handler.setFormatter(logger_formatter)
    logger.addHandler(logger_handler)
    logger.info(f'[{proc_id}] Starting run...')

    logger.info(f'[{proc_id}] Extracting data from sector run {dv_xml_run.name}...')

    data = {}
    target_i = 0
    dv_xml_run_fps = list(dv_xml_run.iterdir())
    n_targets = len(dv_xml_run_fps)
    for dv_xml_fp in dv_xml_run_fps:

        target_i += 1

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

            logger.info(f'[{proc_id}] [{sector_run}] Getting difference image data for TCE TIC '
                        f'{uid} ({tce_i}/{n_tces} TCEs)... ({target_i}/{n_targets} targets)')

            # get difference image results
            diff_img_res = [el for el in planet_res if 'differenceImageResults' in el.tag]

            n_sectors = len(diff_img_res)

            if n_sectors < n_sectors_expected:
                logger.info(f'TCE TIC {uid} has less than {n_sectors_expected} sectors ({n_sectors})')

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
                                f'target in sector run {sector_run_id}.')
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

    np.save(save_dir / f'tess_diffimg_pid{proc_id}.npy', data)

    return data


if __name__ == '__main__':

    # DV XML file path
    dv_xml_root_fp = Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/data/fits_files/tess/dv/sector_runs')
    single_sector_runs = [fp for fp in (dv_xml_root_fp / 'single-sector').iterdir() if fp.is_dir()]
    multi_sector_runs = [fp for fp in (dv_xml_root_fp / 'multi-sector').iterdir() if fp.is_dir()]
    dv_xml_runs = list(single_sector_runs) + list(multi_sector_runs)

    # # TCE table file path
    tce_tbl_fp = Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/data/ephemeris_tables/tess/DV_SPOC_mat_files/11-29-2021/tess_tces_s1-s40_11-23-2021_1409_stellarparams_updated_eb_tso_tec_label_modelchisqr_astronet_ruwe_magcat_uid_corrtsoebs_corraltdetfail.csv')
    tce_tbl = pd.read_csv(tce_tbl_fp, usecols=['uid', 'label'])
    tce_tbl.set_index('uid', inplace=True)

    # run directory
    run_dir = Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/data/fits_files/tess/dv/preprocessing/8-16-2022_1205')

    # create run directory
    run_dir.mkdir(exist_ok=True, parents=True)

    # creating plotting directory
    plot_dir = run_dir / 'plots'
    plot_dir.mkdir(exist_ok=True)
    plot_prob = 0.01

    n_processes = 4
    pool = multiprocessing.Pool(processes=n_processes)
    # n_jobs = len(dv_xml_runs)
    # dv_xml_runs_jobs = np.array_split(dv_xml_runs, n_jobs)
    jobs = [(dv_xml_run, run_dir, plot_prob, tce_tbl) for dv_xml_run in dv_xml_runs]
    async_results = [pool.apply_async(get_data_from_dv_xml, job) for job in jobs]
    pool.close()

    data = {}
    for async_res in async_results:
        data.update(async_res.get())

    # data = get_data_from_dv_xml(dv_xml_run, run_dir, plot_prob, tce_tbl)

    np.save(run_dir / 'tess_diffimg.npy', data)
