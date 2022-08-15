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

plt.switch_backend('agg')


def get_data_from_dv_xml(dv_xml_run, save_dir, plot_prob):

    plt.switch_backend('agg')

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

    tce_i = 0
    data = {}
    for dv_xml_fp in dv_xml_run.iterdir():

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

        for planet_res in planet_res_lst:

                tce_i += 1

                uid = f'{root.attrib["ticId"]}-{planet_res.attrib["planetNumber"]}-S{sector_run_id}'

                logger.info(f'[{proc_id}] [{sector_run}] Getting difference image data for TCE KIC {uid}... (TCE {tce_i})')

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
                        logger.info(f'[{proc_id}] [{sector_run}] TCE TIC {uid} has missing reference centroid for target.')
                    else:
                        tic_centroid_ref_dict = {
                            'col': {k: float(v) - min_col for k, v in tic_centroid_ref_col.items()},
                            'row': {k: float(v) - min_row for k, v in tic_centroid_ref_row.items()}
                        }

                        # plot difference image with target location
                        if np.random.uniform() <= plot_prob:
                            f, ax = plt.subplots()
                            ax.imshow(diff_imgs[:, :, 2, 0])
                            ax.scatter(tic_centroid_ref_dict['col']['value'], tic_centroid_ref_dict['row']['value'],
                                       marker='x',
                                       color='r', label='Target')
                            ax.set_ylabel('Row')
                            ax.set_xlabel('Col')
                            ax.legend()
                            f.savefig(save_dir / 'plots' / f'{uid}_diff_img.png')
                            plt.close()

                    # root.clear()

                    data[uid] = {
                        'target_ref_centroid': tic_centroid_ref_dict,
                        'image_data': diff_imgs
                    }

    np.save(save_dir / f'tess_diffimg_pid{proc_id}.npy', data)

    return data


if __name__ == '__main__':

    plt.switch_backend('agg')

    # DV XML file path
    dv_xml_root_fp = Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/data/fits_files/tess/dv/sector_runs')
    single_sector_runs = [fp for fp in (dv_xml_root_fp / 'single-sector').iterdir() if fp.is_dir()]
    multi_sector_runs = [fp for fp in (dv_xml_root_fp / 'multi-sector').iterdir() if fp.is_dir()]
    dv_xml_runs = list(single_sector_runs) + list(multi_sector_runs)

    # # TCE table file path
    # tce_tbl_fp = Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/data/ephemeris_tables/kepler/q1-q17_dr25/11-17-2021_1243/q1_q17_dr25_tce_2020.09.28_10.36.22_stellar_koi_cfp_norobovetterlabels_renamedcols_nomissingval_symsecphase_cpkoiperiod_rba_cnt0n_valpc_modelchisqr_ruwe_magcat_uid.csv')
    # run directory
    run_dir = Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/data/fits_files/tess/dv/preprocessing/8-15-2022_1525')

    # create run directory
    run_dir.mkdir(exist_ok=True, parents=True)

    # # load TCE table to get examples for which the image data are going to be extracted
    # tce_tbl = pd.read_csv(tce_tbl_fp, usecols=['uid'])
    # tces_lst = tce_tbl['uid'].to_list()

    # creating plotting directory
    plot_dir = run_dir / 'plots'
    plot_dir.mkdir(exist_ok=True)
    plot_prob = 0.01

    n_processes = 4
    pool = multiprocessing.Pool(processes=n_processes)
    # n_jobs = len(dv_xml_runs)
    # dv_xml_runs_jobs = np.array_split(dv_xml_runs, n_jobs)
    jobs = [(dv_xml_run, run_dir, plot_prob) for dv_xml_run in dv_xml_runs]
    async_results = [pool.apply_async(get_data_from_dv_xml, job) for job in jobs]
    pool.close()

    data = {}
    for async_res in async_results:
        data.update(async_res.get())

    # data = get_data_from_dv_xml(dv_xml_fp, tces_lst, plot_prob, plot_dir)

    np.save(run_dir / 'tess_diffimg.npy', data)
