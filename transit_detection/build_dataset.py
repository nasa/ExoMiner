"""
Script used to extract data for a set of TCEs from light curve and targe pixel file data to build a dataset of examples
in TFRecord format.
"""

# 3rd party
import logging.handlers
from pathlib import Path
import pandas as pd
import numpy as np
import lightkurve as lk
import warnings
import tensorflow as tf
import multiprocessing
from functools import partial
from copy import deepcopy
import logging

# local
from transit_detection.utils_flux_processing import extract_flux_windows_for_tce, build_transit_mask_for_lightcurve, plot_detrended_flux_time_series_sg
from transit_detection.utils_difference_img_processing import extract_diff_img_data_from_window
from transit_detection.utils_build_dataset import serialize_set_examples_for_tce, write_data_to_auxiliary_tbl
from transit_detection.utils_local_fits_processing import search_and_read_tess_lightcurve, search_and_read_tess_targetpixelfile


from src_preprocessing.lc_preprocessing.detrend_timeseries import detrend_flux_using_sg_filter

#%%
def process_target_sector_run(target, sector_run, sector_run_data,
                                data_dir,
                                lc_dir,
                                tpf_dir,
                                n_durations_window,
                                frac_valid_cadences_in_window_thr,
                                frac_valid_cadences_it_thr,
                                buffer_time,
                                gap_width,
                                resampled_num_points,
                                rnd_seed,
                                size_img,
                                f_size,
                                center_target,
                                plot_dir,
                                log_queue
                               ):
    logger = logging.getLogger(f"worker-{target}-{sector_run}")
    logger.addHandler(logging.handlers.QueueHandler(log_queue)) #log through queue
    logger.setLevel(logging.INFO)

    try:
        sector_data_to_tfrec = []

        rng = np.random.default_rng(seed=rnd_seed)

        logger.info(f'Extracting and processing data for target: {type(target)}, sector_run: {sector_run}')

        plot_dir_target_sector_run = plot_dir / f'target_{target}_sector_run_{sector_run}'
        plot_dir_target_sector_run.mkdir(exist_ok=True, parents=True)

        target_sector_run_tce_data = []

        # get individual sector runs (needed for multisector runs)
        if '-' in sector_run:
            start_sector, end_sector = [int(sector) for sector in sector_run.split('-')]
            sector_run_arr = np.arange(start_sector, end_sector + 1)
        else:
            sector_run_arr = [int(sector_run)]

        # find light curve data for target, sector_run pair
        found_sectors, lcfs = search_and_read_tess_lightcurve(target=target,
                                                                        sectors=sector_run_arr,
                                                                        data_dir=lc_dir) #update to local data dir
        
        if len(found_sectors) == 0:
            print(f'No light curve data for sector run {sector_run} and TIC {tic_id}. Skipping this sector run.')
            return

        logger.info(f'Found {len(found_sectors)} sectors with light curve data for target, sector_run: {target}, {sector_run}.')

        for tce_i, tce_data in sector_run_data.iterrows(): # get all tces in target, sector_run pair
            target_sector_run_tce_data.append({
            "tce_time0bk" : tce_data['tce_time0bk'],
            "tce_period" : tce_data['tce_period'],
            "tce_duration" : tce_data['tce_duration']
            })

        # for sector_i, sector in enumerate(found_sectors):  # for given sector_run
        for sector, lcf in zip(found_sectors, lcfs):
        
            in_transit_mask = []

            data_for_tce_sector = {
                'sector': sector,
                'transit_examples': {'flux': [], 't': [], 'it_img': [], 'oot_img': [], 'diff_img': [], 'snr_img': [],
                                    'target_img': [], 'target_pos': []},
                'not_transit_examples': {'flux': [], 't': [], 'it_img': [], 'oot_img': [], 'diff_img': [], 'snr_img': [],
                                        'target_img': [], 'target_pos': []},
            }
            
            plot_dir_target_sector_run_sector = plot_dir_target_sector_run/ f'sector_{sector}' #individual sectors for multi sector runs
            plot_dir_target_sector_run_sector.mkdir(exist_ok=True, parents=True)

            # # download light curve fits file(s)
            # lcf = search_lc_res[sector_i].download(download_dir=str(lc_dir), quality_bitmask='default',
            #                                         flux_column='pdcsap_flux')

            lcf = lk.LightCurve({'time': lcf.time.value, 'flux': np.array(lcf.flux.value)})

            raw_time, raw_flux = lcf.time.value, lcf.flux.value

            in_transit_mask = build_transit_mask_for_lightcurve(time=raw_time, tce_list=target_sector_run_tce_data)

            time, flux, trend = detrend_flux_using_sg_filter(lc=lcf, mask_in_transit=in_transit_mask,
                                                                            win_len=int(1.2*24*30), sigma=5,
                                                                            max_poly_order=6, penalty_weight=1, 
                                                                            break_tolerance=5)


            plot_detrended_flux_time_series_sg(time=raw_time, flux=raw_flux, detrend_time=time,
                                            detrend_flux=flux, trend=trend, sector=sector, 
                                            plot_dir=plot_dir_target_sector_run_sector)
            
            for tce_i, tce_data in sector_run_data.iterrows():

                data_for_tce = {
                'tce_uid': tce_data['uid'],
                'tce_info': tce_data,
                'sectors': [],
                'disposition': tce_data['disposition']
                }

                # create directory for TCE plots
                plot_dir_target_sector_run_sector_tce = plot_dir_target_sector_run_sector / f'tce_{tce_data["uid"]}'
                plot_dir_target_sector_run_sector_tce.mkdir(exist_ok=True, parents=True)

                # get TCE unique id
                tic_id = tce_data['target_id']
                sector_run = tce_data['sector_run']
                tce_plnt_num = tce_data['tce_plnt_num']

                # ephemerides for TCE
                tce_time0bk = tce_data['tce_time0bk']
                period_days = tce_data['tce_period']
                tce_duration = tce_data['tce_duration']

                # extract flux time series windows
                (resampled_flux_it_windows_arr, resampled_flux_oot_windows_arr, midtransit_points_windows_arr,
                midoot_points_windows_arr) = extract_flux_windows_for_tce(time, flux, in_transit_mask, tce_time0bk, period_days, 
                                                                        tce_duration, n_durations_window, gap_width,
                                                                        buffer_time, frac_valid_cadences_in_window_thr,
                                                                        frac_valid_cadences_it_thr,
                                                                        resampled_num_points, tce_data['uid'], rng,
                                                                        plot_dir_target_sector_run_sector_tce)
                plot_dir_tce_diff_img_data = plot_dir_target_sector_run_sector_tce / 'diff_img_data'
                plot_dir_tce_diff_img_data.mkdir(exist_ok=True, parents=True)

                # find target pixel data for target
                found_sector, tpf = search_and_read_tess_targetpixelfile(target=target, 
                                                        sectors=sector, 
                                                        data_dir=tpf_dir)
                # if no target pixel
                if len(found_sector) == 0:
                    logger.info(f'No target pixel data for sector {sector} and TIC {tic_id} in sector run {sector_run}. Skipping this sector.')
                    continue

                # compute difference image data for each window for target pixel file data
                # compute transit difference image data
                it_diff_img_data = {feature_name: []
                                    for feature_name in ['diff_img', 'oot_img', 'snr_img', 'target_img', 'target_pos']}
                plot_dir_tce_transit_diff_img_data = plot_dir_tce_diff_img_data / 'transit_imgs'
                plot_dir_tce_transit_diff_img_data.mkdir(exist_ok=True, parents=True)
                excl_idxs_it_ts = []
                for window_i, midtransit_point_window in enumerate(midtransit_points_windows_arr):  # iterate on midtransit points

                    try:
                        diff_img_processed, oot_img_processed, snr_img_processed, target_img, target_pos_col, target_pos_row = (
                            extract_diff_img_data_from_window(tpf.path, midtransit_point_window, tce_duration, buffer_time,
                                                            tce_data['ra'], tce_data['dec'], size_img, f_size,
                                                            sector, center_target, tce_data['uid'],
                                                            plot_dir=plot_dir_tce_transit_diff_img_data))
                    except ValueError as error:
                        warnings.warn(error.args[0])
                        warnings.warn(f'no difference image data computed for midtransit point {midtransit_point_window}, '
                                    f'skipping this window')
                        excl_idxs_it_ts.append(window_i)
                        continue

                    it_diff_img_data['diff_img'].append(diff_img_processed)
                    it_diff_img_data['oot_img'].append(oot_img_processed)
                    it_diff_img_data['snr_img'].append(snr_img_processed)
                    it_diff_img_data['target_img'].append(target_img)
                    it_diff_img_data['target_pos'].append(np.array([target_pos_col, target_pos_row]))

                # compute oot transit difference image data
                oot_diff_img_data = {feature_name: []
                                    for feature_name in ['diff_img', 'oot_img', 'snr_img', 'target_img', 'target_pos']}
                plot_dir_tce_oottransit_diff_img_data = plot_dir_tce_diff_img_data / 'oot_transit_imgs'
                plot_dir_tce_oottransit_diff_img_data.mkdir(exist_ok=True, parents=True)
                excl_idxs_oot_ts = []
                for window_i, midoot_point_window in enumerate(midoot_points_windows_arr):  # iterate on oot points

                    try:
                        diff_img_processed, oot_img_processed, snr_img_processed, target_img, target_pos_col, target_pos_row = (
                            extract_diff_img_data_from_window(tpf.path, midoot_point_window, tce_duration, buffer_time,
                                                            tce_data['ra'], tce_data['dec'], size_img, f_size,
                                                            sector, center_target, tce_data['uid'],
                                                            plot_dir=plot_dir_tce_oottransit_diff_img_data))
                    except ValueError as error:
                        warnings.warn(error.args[0])
                        warnings.warn(f'no difference image data computed for out-of-transit point {midoot_point_window}, '
                                    f'skipping this window')
                        excl_idxs_oot_ts.append(window_i)
                        continue

                    oot_diff_img_data['diff_img'].append(diff_img_processed)
                    oot_diff_img_data['oot_img'].append(oot_img_processed)
                    oot_diff_img_data['snr_img'].append(snr_img_processed)
                    oot_diff_img_data['target_img'].append(target_img)
                    oot_diff_img_data['target_pos'].append(np.array([target_pos_col, target_pos_row]))

                # exclude examples for timestamps with no difference image data
                resampled_flux_it_windows_arr = np.delete(np.array(resampled_flux_it_windows_arr), excl_idxs_it_ts, axis=0)
                midtransit_points_windows_arr = np.delete(np.array(midtransit_points_windows_arr), excl_idxs_it_ts, axis=0)
                resampled_flux_oot_windows_arr = np.delete(np.array(resampled_flux_oot_windows_arr), excl_idxs_oot_ts, axis=0)
                midoot_points_windows_arr = np.delete(np.array(midoot_points_windows_arr), excl_idxs_oot_ts, axis=0)

                # add data for TCE for a given sector run
                data_for_tce_sector['transit_examples']['flux'] = resampled_flux_it_windows_arr
                data_for_tce_sector['transit_examples']['t'] = midtransit_points_windows_arr
                data_for_tce_sector['transit_examples'].update(it_diff_img_data)
                data_for_tce_sector['not_transit_examples']['flux'] = resampled_flux_oot_windows_arr
                data_for_tce_sector['not_transit_examples']['t'] = midoot_points_windows_arr
                data_for_tce_sector['not_transit_examples'].update(oot_diff_img_data)

                data_for_tce['sectors'].append(data_for_tce_sector)

            sector_data_to_tfrec.append(data_for_tce)
        return sector_data_to_tfrec
    except Exception as e:
        logger.error(f"Error processing target: {target}, sector run: {sector_run}: {e}")
        return None
    
def listener_configurer(log_dir):
    """Configure the listener to write logs to file or the console"""
    log_file = log_dir / 'multiprocessing_log.log'
    root = logging.getLogger()
    handler = logging.FileHandler(log_file)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    root.addHandler(handler)
    root.setLevel(logging.INFO)

def listener_process(log_queue, log_dir):
    """Listen for logs in the log queue and write them using the listener config"""
    listener_configurer(log_dir=log_dir)
    while True:
        try:
            record = log_queue.get()
            if record is None:
                break
            logger = logging.getLogger(record.name)
            logger.handle(record)
        except Exception as e:
            print(f"Error in listener process: {e}")

if __name__ == "__main__":
    #initialize logger
    manager = multiprocessing.Manager()
    log_queue = manager.Queue()

    log_dir = Path('/nobackup/jochoa4/work_dir/data/logging')
    log_dir.mkdir(parents=True, exist_ok=True)

    listener = multiprocessing.Process(target=listener_process, args=(log_queue,log_dir))
    listener.start()

    data_dir = Path('/nobackup/jochoa4/work_dir/data/datasets/TESS_exoplanet_dataset_10-19-2024_v1')
    # set light curve data directory
    lc_dir = Path('/nobackup/msaragoc/work_dir/Kepler-TESS_exoplanet/data/FITS_files/TESS/spoc_2min/lc')
    # set target pixel file data directory
    tpf_dir = Path('/nobackup/jochoa4/TESS/fits_files/spoc_2min/tp')
    # TCE table
    tce_tbl = pd.read_csv('/nobackup/jochoa4/work_dir/data/tables/tess_2min_tces_dv_s1-s68_all_msectors_11-29-2023_2157_newlabels_nebs_npcs_bds_ebsntps_to_unks.csv')
    tce_tbl = tce_tbl.loc[tce_tbl['label'].isin(['EB','KP','CP','NTP','NEB','NPC'])] #filter for relevant labels
    tce_tbl.rename(columns={'label': 'disposition', 'label_source': 'disposition_source'}, inplace=True)

    n_durations_window = 5  # number of transit durations in each extracted window
    frac_valid_cadences_in_window_thr = 0.85
    frac_valid_cadences_it_thr = 0.85
    # sampling_rate = 2  # min/cadence
    buffer_time = 30  # in minutes, between in-transit cadences and out-of-transit cadences
    # oot_buffer_ncadences = int(buffer_time / sampling_rate)
    # days used to split timeseries into smaller segments if interval between cadences is larger than gap_width
    gap_width = 0.75
    resampled_num_points = 100  # number of points in the window after resampling
    rnd_seed = 42
    # difference image data parameters
    size_img = [11, 11]  # resize images to this size
    f_size = [3, 3]  # enlarge `size_img` by these factors; final dimensions are f_size * size_img
    center_target = False  # center target in images

    # create dataset directory
    data_dir.mkdir(exist_ok=True, parents=True)

    # create plot dir
    plot_dir = data_dir / 'plots'
    plot_dir.mkdir(exist_ok=True, parents=True)

    tces_lst = [
        '68577662-1-S43',  # KP
        '394112898-1-S40',  # UNK, EB?
        '261136679-1-S13',  # CP
        '336767770-1-S17',  # NTP
        '352289014-1-S13',  # NTP
        '298734307-1-S14-60',  # multisector example
    ]

    tce_tbl = tce_tbl.loc[tce_tbl['uid'].isin(tces_lst)]

    # rng = np.random.default_rng(seed=rnd_seed)

    partial_func = partial(process_target_sector_run,
                                    data_dir=data_dir,
                                    lc_dir=lc_dir,
                                    tpf_dir=tpf_dir,
                                    n_durations_window=n_durations_window,
                                    frac_valid_cadences_in_window_thr=frac_valid_cadences_in_window_thr,
                                    frac_valid_cadences_it_thr=frac_valid_cadences_it_thr,
                                    buffer_time=buffer_time,
                                    gap_width=gap_width,
                                    resampled_num_points=resampled_num_points,
                                    rnd_seed=rnd_seed,
                                    size_img=deepcopy(size_img),
                                    f_size=deepcopy(f_size),
                                    center_target=center_target,
                                    plot_dir=plot_dir,
                                    log_queue=log_queue
                                    )

    pool = multiprocessing.Pool(processes=None) # None processes defaults to number of cpu cores
    # grouped sector_run_data from tce_tbl is unique, so we can use sector_run_data views for parallel ops
    jobs = [(target, sector_run, sector_run_data) for (target, sector_run), sector_run_data in tce_tbl.groupby(['target_id','sector_run'])]

    async_results = [pool.apply_async(partial_func, job) for job in jobs]
    pool.close()

    data_to_tfrec = []
    for async_result in async_results:
        for data in async_result.get():
            data_to_tfrec.append(data)
    
    log_queue.put(None) #stop listener
    listener.join() #wait for listener to finish
    #%% write data to TFRecord dataset
    print(f'Write data to TFRecords.')

    tfrec_dir = data_dir / 'tfrecords'
    tfrec_dir.mkdir(exist_ok=True, parents=True)
    tfrec_fp = tfrec_dir / 'test_shard_0001-0001'

    with tf.io.TFRecordWriter(str(tfrec_fp)) as writer:

        for data_for_tce in data_to_tfrec:
            print(f'Adding data for TCE {data_for_tce["tce_uid"]} to the TFRecord file {tfrec_fp}...')

            examples_for_tce = serialize_set_examples_for_tce(data_for_tce)

            for example_for_tce in examples_for_tce:
                writer.write(example_for_tce)

    # create auxiliary table
    print('Creating auxiliary table to TFRecord dataset.')
    data_tbl = write_data_to_auxiliary_tbl(data_to_tfrec, tfrec_fp)
    data_tbl.to_csv(tfrec_dir / 'data_tbl.csv', index=False)