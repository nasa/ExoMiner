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
from transit_detection.utils_flux_processing import extract_flux_windows_for_tce, build_transit_mask_for_lightcurve, split_timeseries_on_time_gap, find_first_epoch_after_this_time, plot_detrended_flux_time_series_sg
from transit_detection.utils_difference_img_processing import extract_diff_img_data_from_window
from transit_detection.utils_build_dataset import serialize_set_examples_for_tce, write_data_to_auxiliary_tbl
from transit_detection.utils_local_fits_processing import search_and_read_tess_lightcurve, search_and_read_tess_targetpixelfile


from src_preprocessing.lc_preprocessing.detrend_timeseries import detrend_flux_using_sg_filter


def extract_flux_windows_resampled_it_points_data_for_tce(time, flux, transit_mask, tce_time0bk, period_days, tce_duration, n_durations_window, 
                                 gap_width, buffer_time, frac_valid_cadences_in_window_thr, frac_valid_cadences_it_thr,
                                 resampled_num_points, tce_uid, rng, plot_dir=None):
    """ Determine expected resampled in transit points for extracted flux windows for a TCE based on its orbital period,
    epoch, and transit duration. These windows are set to `n_durations_window` * `tce_duration` and are resampled to 
    `resampled_num_points`. Transit windows are first  built centered around the midtransit points available in the `time`
    timestamps array. Out-of-transit windows are  chosen in equal number to the transit windows by randomly choosing a set
    of out-of-transit cadences that do not  overlap with in-transit regions (which are defined as midtransit +- tce_duration).

    Args:
        time: NumPy array, timestamps
        flux: NumPy array, flux values
        transit_mask: NumPy array with boolean elements for in-transit (True) and out-of-transit (False) timestamps.
        tce_time0bk: float,
        period_days: float, orbital period in days
        tce_duration: float, transit duration in hours
        n_durations_window: int, number of transit durations in the extracted window.
        gap_width: float, gap width in days between consecutive cadences to split arrays into sub-arrays
        buffer_time: int, buffer in minutes between out-of-transit and in-transit cadences
        frac_valid_cadences_in_window_thr: float, fraction of valid cadences in window
        frac_valid_cadences_it_thr: float, fraction of valid in-transit cadences
        resampled_num_points: int, number of points to resample
        tce_uid: str, TCE uid
        rng: NumPy rng generator
        plot_dir: Path, plot directory; set to None to disable plotting

    Returns:
        n_it_windows: Number of it windows extracted
        n_oot_windows: Number of oot windows extracted
        num_true_resampled_it_window_points: List of true of total it points per it_window
    """
    it_window_samples = []

    buffer_time /= 1440  # convert from minutes to days
    tce_duration /= 24  # convert from hours to days

    valid_idxs_data = np.logical_and(np.isfinite(time), np.isfinite(flux))
    time, flux, transit_mask = time[valid_idxs_data], flux[valid_idxs_data], transit_mask[valid_idxs_data]

    # split time series on gaps
    time_arrs, _ = split_timeseries_on_time_gap(time, flux, gap_width)
    print(f'Time series split into {len(time_arrs)} arrays due to gap(s) in time.')

    # find first midtransit point in the time array
    first_transit_time = find_first_epoch_after_this_time(tce_time0bk, period_days, time[0])
    # compute all midtransit points in the time array
    midtransit_points_arr = np.array([first_transit_time + phase_k * period_days
                                      for phase_k in range(int(np.ceil((time[-1] - time[0]) / period_days)))])

    print(f'Found {len(midtransit_points_arr)} midtransit points.')

    # get start and end timestamps for the windows based on the available midtransit points
    start_time_windows, end_time_windows = (midtransit_points_arr - n_durations_window * tce_duration / 2,
                                            midtransit_points_arr + n_durations_window * tce_duration / 2)

    # choose only those midtransit points whose windows fit completely inside the time array
    valid_midtransit_points_arr = midtransit_points_arr[np.logical_and(start_time_windows >= time[0],
                                                                      end_time_windows <= time[-1])]

    print(f'Found {len(valid_midtransit_points_arr)} midtransit points whose windows fit completely inside the time '
          f'array.')

    valid_idxs_data = np.logical_and(np.isfinite(time), np.isfinite(flux))

    # extract transit windows
    time_it_windows_arr, midtransit_points_windows_arr = [], []

    # extract transit window points samples
    n_resampled_it_points_arr = []

    # n_resampled_it_points_per_rr = {} # per resampling rate
    # for resampled_num_points in resampled_num_points_to_test:
        # n_resampled_it_points_per_rr[f'n_resampled_it_points_{resampled_num_points}'] = []

    n_original_it_points = []

    for start_time_window, end_time_window, midtransit_point_window in zip(start_time_windows, end_time_windows,
                                                                           valid_midtransit_points_arr):

        # find indices in window
        idxs_window = np.logical_and(time >= start_time_window, time <= end_time_window)
        if idxs_window.sum() == 0:
            print(f'No valid indices in window [{start_time_window}, {end_time_window}.')
            continue

        # find in-transit windows
        idxs_it_window = np.logical_and(time >= midtransit_point_window - tce_duration / 2,
                                        time <= midtransit_point_window + tce_duration / 2)

        if idxs_it_window.sum() == 0:
            print(f'No valid in-transit indices in window [{start_time_window}, {end_time_window}.')
            continue

        # check for valid window and in-transit region
        valid_window_flag = (((idxs_window & valid_idxs_data).sum() / idxs_window.sum()) >
                             frac_valid_cadences_in_window_thr)
        valid_it_window_flag = (((idxs_it_window & valid_idxs_data).sum() / idxs_it_window.sum()) >
                                frac_valid_cadences_it_thr)

        # if window & it window valid
        if valid_window_flag and valid_it_window_flag:
            time_window = time[idxs_window]
            # flux_window = flux[idxs_window]

            time_it_windows_arr.append(time_window)
            # flux_it_windows_arr.append(flux_window)
            midtransit_points_windows_arr.append(midtransit_point_window)

            # n_original_it_points = idxs_it_window.sum()

            window_resample_scale = resampled_num_points / len(time_window)
            n_resampled_it_points = idxs_it_window.sum() * window_resample_scale
            n_resampled_it_points_arr.append(n_resampled_it_points)

        for n_resampled_it_points, it_window in zip(n_resampled_it_points_arr, time_it_windows_arr):
            it_window_sample = {
                # "tce_uid" : tce_uid,
                f"n_resampled_it_points_{resampled_num_points}" : n_resampled_it_points,
                "window_start" : it_window[0],
                "window_end" : it_window[-1]
            }
            it_window_samples.append(it_window_sample)

    return it_window_samples

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
                                log_dir,
                               ):
    logger = logging.getLogger(f"worker_{target}-{sector_run}")

    logger.setLevel(logging.INFO)
    log_path = Path(log_dir) / f"process_log_target_{target}_sector_run_{sector_run}.log"
    file_handler = logging.FileHandler(log_path)
    logger_formatter = logging.Formatter('%(asctime)s - %(levelname)s- %(message)s')
    file_handler.setFormatter(logger_formatter)

    if logger.hasHandlers():
        logger.handlers.clear()
    
    logger.addHandler(file_handler)

    try:
        target_sector_run_sector_tce_samples = []

        rng = np.random.default_rng(seed=rnd_seed)

        logger.info(f'Extracting and processing data for target: {type(target)}, sector_run: {sector_run}')

        if plot_dir:
            plot_dir_target_sector_run = plot_dir / f'target_{target}_sector_run_{sector_run}'
            plot_dir_target_sector_run.mkdir(exist_ok=True, parents=True)
        else:
            plot_dir_target_sector_run = None


        

        # get individual sector runs (needed for multisector runs)
        if '-' in sector_run:
            start_sector, end_sector = [int(sector) for sector in sector_run.split('-')]
            sector_run_arr = [sector for sector in range(start_sector, end_sector + 1)]
        else:
            sector_run_arr = [int(sector_run)]

        # find light curve data for target, sector_run pair
        found_sectors, lcfs = search_and_read_tess_lightcurve(target=target,
                                                                        sectors=sector_run_arr,
                                                                        data_dir=lc_dir) #update to local data dir
        

        if len(found_sectors) == 0:
            logger.error(f'No light curve data for sector run {sector_run} and target {target}. Skipping this sector run.')
            return
        
        logger.info(f'Found {len(found_sectors)} sectors with light curve data for target, sector_run: {target}, {sector_run}.')

        target_sector_run_tce_data = []
        for tce_i, tce_data in sector_run_data.iterrows(): # get all tces in target, sector_run pair
            target_sector_run_tce_data.append({
            "tce_time0bk" : tce_data['tce_time0bk'],
            "tce_period" : tce_data['tce_period'],
            "tce_duration" : tce_data['tce_duration']
            })
        

        # for sector_i, sector in enumerate(found_sectors):  # for given sector_run
        for sector, lcf in zip(found_sectors, lcfs):
            sector_data_tce_samples = []
            in_transit_mask = []

            # data_for_tce_sector = {
            #     'sector': sector,
            #     'transit_examples': {'flux': [], 't': [], 'it_img': [], 'oot_img': [], 'diff_img': [], 'snr_img': [],
            #                         'target_img': [], 'target_pos': []},
            #     'not_transit_examples': {'flux': [], 't': [], 'it_img': [], 'oot_img': [], 'diff_img': [], 'snr_img': [],
            #                             'target_img': [], 'target_pos': []},
            # }
            
            if plot_dir_target_sector_run:
                plot_dir_target_sector_run_sector = plot_dir_target_sector_run/ f'sector_{sector}' #individual sectors for multi sector runs
                plot_dir_target_sector_run_sector.mkdir(exist_ok=True, parents=True)
            else:
                plot_dir_target_sector_run_sector = None

            lcf = lk.LightCurve({'time': lcf.time.value, 'flux': np.array(lcf.flux.value)})

            raw_time, raw_flux = lcf.time.value, lcf.flux.value

            in_transit_mask = build_transit_mask_for_lightcurve(time=raw_time, tce_list=target_sector_run_tce_data)

            time, flux, trend = detrend_flux_using_sg_filter(lc=lcf, mask_in_transit=in_transit_mask,
                                                                            win_len=int(1.2*24*30), sigma=5,
                                                                            max_poly_order=6, penalty_weight=1, 
                                                                            break_tolerance=5)

            if plot_dir_target_sector_run_sector:
                plot_detrended_flux_time_series_sg(time=raw_time, flux=raw_flux, detrend_time=time,
                                                detrend_flux=flux, trend=trend, sector=sector, 
                                                plot_dir=plot_dir_target_sector_run_sector)

            for tce_i, tce_data in sector_run_data.iterrows():

                # create directory for TCE plots
                if plot_dir_target_sector_run_sector:
                    plot_dir_target_sector_run_sector_tce = plot_dir_target_sector_run_sector / f'tce_{tce_data["uid"]}'
                    plot_dir_target_sector_run_sector_tce.mkdir(exist_ok=True, parents=True)
                else:
                    plot_dir_target_sector_run_sector_tce = None

                # get TCE unique id
                tic_id = tce_data['target_id']
                sector_run = tce_data['sector_run']
                tce_plnt_num = tce_data['tce_plnt_num']

                # ephemerides for TCE
                tce_time0bk = tce_data['tce_time0bk']
                period_days = tce_data['tce_period']
                tce_duration = tce_data['tce_duration']

                # extract flux time series windows
                # n_extracted_it_windows, n_extracted_oot_windows, n_original_it_points, n_resampled_it_points_per_rr = extract_flux_windows_resampled_it_points_data_for_tce(
                tce_it_window_samples = extract_flux_windows_resampled_it_points_data_for_tce(
                                                                        time, flux, in_transit_mask, tce_time0bk, period_days, 
                                                                        tce_duration, n_durations_window, gap_width,
                                                                        buffer_time, frac_valid_cadences_in_window_thr,
                                                                        frac_valid_cadences_it_thr,
                                                                        resampled_num_points, tce_data['uid'], rng,
                                                                        plot_dir_target_sector_run_sector_tce)
                if plot_dir_target_sector_run_sector_tce:
                    plot_dir_tce_diff_img_data = plot_dir_target_sector_run_sector_tce / 'diff_img_data'
                    plot_dir_tce_diff_img_data.mkdir(exist_ok=True, parents=True)
                else:
                    plot_dir_tce_diff_img_data = None
                # store each window as an example
                tce_it_window_samples = []
                # example id: sector_examplei. ie) for msr 1-4, may find sectors 2,3. for each assume 4 windows found. s2_ex1, s-2_ex-2, s
                for sample_i, sample_data in enumerate(tce_it_window_samples):
                    tce_it_window_sample = {
                    'tce_uid' : tce_data['uid'],
                    'tce_label' : tce_data['disposition'],
                    'tce_duration' : tce_data['tce_duration'],
                    'tce_time0bk' : tce_data['tce_time0bk'],
                    'tce_period' : tce_data['tce_period'],
                    'tce_plnt_num' : tce_data['tce_plnt_num'],
                    'target_id' : target,
                    'sector_run' : sector_run,
                    'sample_sector' : sector,
                    'sample_id' : f's-{sector}_e-{sample_i}',
                    'n_original_it_points'
                    f'resampled_it_points_{resampled_num_points}' : sample_data[f"n_resampled_it_points_{resampled_num_points}"],
                    'window_start' : sample_data['window_start'],
                    'window_end' : sample_data['window_end']
                    }
                    tce_it_window_samples.append(tce_it_window_sample)

                sector_data_tce_samples.extend(tce_it_window_samples)

            target_sector_run_sector_tce_samples.extend(sector_data_tce_samples)

        logger.info(f"Processing for sector_run {sector_run} complete.")
        print(f"Processing for sector_run {sector_run} complete.")
        return target_sector_run_sector_tce_samples
    except Exception as e:
        print(f"Error processing target: {target}, sector run: {sector_run} - {e}")
        logger.error(f"Error processing target: {target}, sector run: {sector_run} - {e}")
        return None
    finally:
        logger.removeHandler(file_handler)
        file_handler.close()

if __name__ == "__main__":

    # set light curve data directory
    lc_dir = Path('/nobackup/msaragoc/work_dir/Kepler-TESS_exoplanet/data/FITS_files/TESS/spoc_2min/lc')
    # set target pixel file data directory
    tpf_dir = Path('/nobackup/jochoa4/TESS/fits_files/spoc_2min/tp/')
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
    resampled_num_points_to_test = 100  # number of points in the window after resampling

    rnd_seed = 42
    # difference image data parameters
    size_img = [11, 11]  # resize images to this size
    f_size = [3, 3]  # enlarge `size_img` by these factors; final dimensions are f_size * size_img
    center_target = False  # center target in images

    plot_dir = None

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

    #create log directory
    log_dir = Path(f'/nobackup/jochoa4/work_dir/data/logging/tce_pipeline_analysis')
    log_dir.mkdir(parents=True, exist_ok=True)
    #create data directory
    data_dir = Path(f'/nobackup/jochoa4/work_dir/data/datasets/tce_pipeline_analysis')
    data_dir.mkdir(exist_ok=True, parents=True)

    partial_func = partial(process_target_sector_run,
                                    data_dir=data_dir,
                                    lc_dir=lc_dir,
                                    tpf_dir=tpf_dir,
                                    n_durations_window=n_durations_window,
                                    frac_valid_cadences_in_window_thr=frac_valid_cadences_in_window_thr,
                                    frac_valid_cadences_it_thr=frac_valid_cadences_it_thr,
                                    buffer_time=buffer_time,
                                    gap_width=gap_width,
                                    resampled_num_points=resampled_num_points_to_test,
                                    rnd_seed=rnd_seed,
                                    size_img=deepcopy(size_img),
                                    f_size=deepcopy(f_size),
                                    center_target=center_target,
                                    plot_dir=plot_dir,
                                    log_dir=log_dir,
                                    )

    pool = multiprocessing.Pool(processes=None) # None processes defaults to number of cpu cores
    # grouped sector_run_data from tce_tbl is unique, so we can use sector_run_data views for parallel ops
    jobs = [(target, sector_run, sector_run_data) for (target, sector_run), sector_run_data in tce_tbl.groupby(['target_id','sector_run'])]

    async_results = [pool.apply_async(partial_func, job) for job in jobs]
    pool.close()
    pool.join()

    # data_to_tfrec = []
    # for async_result in async_results:
    #     if async_result.get():
    #         for data in async_result.get():
    #             data_to_tfrec.append(data)
    data_to_df = []
    for async_result in async_results:
        if async_result.get():
            for data in async_result.get():
                data_to_df.append(data)

    df_columns = ['tce_uid',
                'tce_label',
                'tce_duration',
                'tce_time0bk',
                'tce_period',
                'tce_plnt_num',
                'target_id',
                'sector_run',
                'sample_sector',
                'sample_id',
                f'n_resampled_it_points_{resampled_num_points_to_test}',
                'window_start',
                'window_end',
                ]
    for target_sector_run_data in data_to_df:
        df = pd.DataFrame(columns=df_columns)
        for tce_sample in target_sector_run_data:
            tce_sample_entry = pd.DataFrame({
                        'tce_uid' : tce_sample['tce_uid'],
                        'tce_label' : tce_sample['disposition'],
                        'tce_duration' : tce_sample['tce_duration'],
                        'tce_time0bk' : tce_sample['tce_time0bk'],
                        'tce_period' : tce_sample['tce_period'],
                        'tce_plnt_num' : tce_sample['tce_plnt_num'],
                        'target_id' : tce_sample['target_id'],
                        'sector_run' : tce_sample['sector_run'],
                        'sample_sector' : tce_sample['sample_sector'],
                        'sample_id' : tce_sample['sample_id'],
                        f'n_resampled_it_points_{resampled_num_points_to_test}' : tce_sample[f'n_resampled_it_points_{resampled_num_points_to_test}'],
                        'window_start' : tce_sample['window_start'],
                        'window_end' : tce_sample['window_end'],
                        })
            df = pd.concat([df, tce_sample_entry], ignore_index=True)

    # resample_suffix = '_'.join([str(rr) for rr in resampled_num_points])
    csv_name = "tce_resampling_analysis_" + resampled_num_points_to_test + '.csv' 
    df_save_path = data_dir / csv_name
    df.to_csv(df_save_path, index=False)