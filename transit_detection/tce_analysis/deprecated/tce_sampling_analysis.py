"""
Script used to extract data for a set of TCEs from light curve and targe pixel file data to build a dataset of examples
in TFRecord format.
"""

# 3rd party
from pathlib import Path
import pandas as pd
import numpy as np
import lightkurve as lk
import warnings
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
import math
# local
from transit_detection.utils_flux_processing import extract_flux_windows_for_tce, build_transit_mask_for_lightcurve, split_timeseries_on_time_gap, find_first_epoch_after_this_time
from transit_detection.utils_difference_img_processing import extract_diff_img_data_from_window
from transit_detection.utils_build_dataset import serialize_set_examples_for_tce, write_data_to_auxiliary_tbl

from src_preprocessing.lc_preprocessing.detrend_timeseries import detrend_flux_using_sg_filter
#%%

# set dataset directory
data_dir = Path('/Users/jochoa4/Downloads/transit_detection/data/test_create_dataset_5-31-2024_1017')
# set light curve data directory
lc_dir = Path('/Users/jochoa4/Downloads/transit_detection/data/lc_data/')
# set target pixel file data directory
tpf_dir = Path('/Users/jochoa4/Downloads/transit_detection/data/tpf_data/')
# TCE table
tce_tbl = pd.read_csv('/Users/jochoa4/Projects/exoplanet_transit_classification/ephemeris_tables/preprocessing_tce_tables/tess_2min_tces_dv_s1-s68_all_msectors_11-29-2023_2157_newlabels_nebs_npcs_bds_ebsntps_to_unks.csv')
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


# create dataset directory
data_dir.mkdir(exist_ok=True, parents=True)

# tce_tbl.set_index('uid', inplace=True)
tces_lst = [
    '68577662-1-S43',  # KP
    '394112898-1-S40',  # UNK, EB?
    '261136679-1-S13',  # CP
    '336767770-1-S17',  # NTP
    '352289014-1-S13',  # NTP
    '260041957-1-S1-36',  # multisector example
]
tce_tbl = tce_tbl.loc[tce_tbl['uid'].isin(tces_lst)]

print(tce_tbl[['uid', 'disposition']])

rng = np.random.default_rng(seed=rnd_seed)

data_to_tfrec = []

transit_window_data = []

def determine_resampled_it_points(time, flux, transit_mask, tce_time0bk, period_days, tce_duration, n_durations_window, 
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
        resampled_flux_it_windows_arr, NumPy array with resampled flux for transit windows
        resampled_flux_oot_windows_arr, NumPy array with resampled flux for out-of-transit windows
        midtransit_points_windows_arr, list with midtransit points for transit windows
        midoot_points_windows_arr, list with midtransit points for out-of-transit windows
    """

    buffer_time /= 1440  # convert from minutes to days
    tce_duration /= 24  # convert from hours to days

    valid_idxs_data = np.logical_and(np.isfinite(time), np.isfinite(flux))


    time, flux = time[valid_idxs_data], flux[valid_idxs_data]

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
    time_it_windows_arr, flux_it_windows_arr, midtransit_points_windows_arr = [], [], []
    all_valid_resampled_it_points = []
    for start_time_window, end_time_window, midtransit_point_window in zip(start_time_windows, end_time_windows,
                                                                           valid_midtransit_points_arr):
        valid_resampled_it_points = []

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

        if valid_window_flag and valid_it_window_flag:
            time_window = time[idxs_window]
            flux_window = flux[idxs_window]

            time_it_windows_arr.append(time_window)
            flux_it_windows_arr.append(flux_window)
            midtransit_points_windows_arr.append(midtransit_point_window)
        
            window_resample_scale = resampled_num_points / len(time_window)
            rescaled_it_window_points = idxs_it_window.sum() * window_resample_scale
            valid_resampled_it_points.append(rescaled_it_window_points)

        if valid_resampled_it_points:
            all_valid_resampled_it_points.extend(valid_resampled_it_points)

    return all_valid_resampled_it_points

def main():
    grouped = tce_tbl.groupby(['target_id','sector_run'])
    # grouped = grouped.sort_values(by='tce_duration').iloc[::10]

    # subset_df = pd.concat([group for i, group in enumerate(grouped) if i % 100 == 0]).reset_index(drop=True)

    unique_tces_dict = {
        uid: {
            "uid" : uid,
            "disposition": tce_tbl[tce_tbl['uid'] == uid]['disposition'].iloc[0],
            'in_transit_counts' : [],
            'average_in_transit' : None
        }
        for uid in tce_tbl['uid'].unique()
    }
    count = 0
    if count < 30:
        for (target, sector_run), sector_run_data in grouped:
            target_tce_data = []

            for tce_i, tce_data in tce_tbl[tce_tbl['target_id'] == target].iterrows(): # for all tces associated with target
                target_tce_data.append({
                    "tce_time0bk" : tce_data['tce_time0bk'],
                    "tce_period" : tce_data['tce_period'],
                    "tce_duration" : tce_data['tce_duration']
                })

            # get TCE unique id
            tic_id = tce_data['target_id']
            sector_run = tce_data['sector_run']
            tce_plnt_num = tce_data['tce_plnt_num']

            # get sector run (needed for multisector TCEs)
            if '-' in sector_run:
                start_sector, end_sector = [int(sector) for sector in sector_run.split('-')]
                sector_run_arr = np.arange(start_sector, end_sector + 1)
            else:
                sector_run_arr = [int(sector_run)]

            # ephemerides for TCE
            tce_time0bk = tce_data['tce_time0bk']
            period_days = tce_data['tce_period']
            tce_duration = tce_data['tce_duration']

            # find light curve data for target
            search_lc_res = lk.search_lightcurve(target=f"tic{target}", mission='TESS', author=('TESS-SPOC', 'SPOC'),
                                                exptime=120, cadence='long', sector=sector_run_arr) #get ALL sectors

            found_sectors = [int(el.split(' ')[-1]) for el in search_lc_res.mission]
            print(f'Found {len(found_sectors)} sectors with light curve data for TIC {tic_id}.')
            
            for sector_i, sector in enumerate(found_sectors):  # for given sector_run
                if len(search_lc_res) == 0:
                    print(f'No light curve data for sector {sector_run} and TIC {tic_id}. Skipping this sector.')
                    continue

                # download light curve fits file(s)
                lcf = search_lc_res[sector_i].download(download_dir=str(lc_dir), quality_bitmask='default',
                                                        flux_column='pdcsap_flux')


                lcf = lk.LightCurve({'time': lcf.time.value, 'flux': np.array(lcf.flux.value)})
                time, flux = lcf.time.value, lcf.flux.value
                
                in_transit_mask = build_transit_mask_for_lightcurve(time=time, tce_list=target_tce_data)

                time, flux, trend = detrend_flux_using_sg_filter(lc=lcf, mask_in_transit=in_transit_mask,
                                                                                win_len=int(1.2*24*30), sigma=5,
                                                                                max_poly_order=6, penalty_weight=1, 
                                                                                break_tolerance=5)


                for tce_i, tce_data in sector_run_data.iterrows(): # iterate over each TCE in target, sector_run pair
                    #extract flux time series for each tce in tce_target_sector_run

                    # extract flux time series windows
                    resampled_it_points = determine_resampled_it_points(time, flux, in_transit_mask, tce_time0bk, period_days, tce_duration,
                                                                            n_durations_window, gap_width,
                                                                            buffer_time, frac_valid_cadences_in_window_thr,
                                                                            frac_valid_cadences_it_thr,
                                                                            resampled_num_points, tce_data['uid'], rng,
                                                                            )
                    if resampled_it_points:
                        unique_tces_dict[tce_data['uid']]['in_transit_counts'].extend(resampled_it_points)

            count = count + 1

    for uid, data in unique_tces_dict.items():
        in_transit_counts = data['in_transit_counts']
        average_it_count = sum(in_transit_counts) / len(in_transit_counts)
        data['average_in_transit'] = average_it_count

    save_path = Path('/Users/jochoa4/Downloads/transit_detection/data/')
    df_save_path = save_path / 'distribution_avg_it_counts.csv'

    data = {
        'uid' : [],
        'disposition' : [],
        'average_in_transit' : []
    }

    for uid, values in unique_tces_dict.items():
        data['uid'].append(values['uid'])
        data['disposition'].append(values['disposition'])
        data['average_in_transit'].append(values['average_in_transit'])

    df = pd.DataFrame.from_dict(data, orient='index').reset_index(drop=True)
    df = pd.DataFrame(data)
    df.to_csv(df_save_path, index=False)

    # fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    # 1. Violin Plot for Disposition and Save
    plt.figure(figsize=(10,6))
    sns.violinplot(x='disposition', y='average_in_transit', data=df, inner=None, color='darkblue', linewidth=1)


if __name__ == "__main__":
    main()
# %%
