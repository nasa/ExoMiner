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
from astropy.utils.masked import Masked

# local
from transit_detection.utils_flux_processing import extract_flux_windows_for_tce, build_transit_mask_for_lightcurve, detrend_flux_using_spline
from transit_detection.utils_difference_img_processing import extract_diff_img_data_from_window
from transit_detection.utils_build_dataset import serialize_set_examples_for_tce, write_data_to_auxiliary_tbl

from src_preprocessing.lc_preprocessing.detrend_timeseries import detrend_flux_using_sg_filter
#%%

# set dataset directory
data_dir = Path('/Users/jochoa4/Downloads/transit_detection/data/test_create_dataset_09-16-2024_1017')
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
# difference image data parameters
size_img = [11, 11]  # resize images to this size
f_size = [3, 3]  # enlarge `size_img` by these factors; final dimensions are f_size * size_img
center_target = False  # center target in images

# create dataset directory
data_dir.mkdir(exist_ok=True, parents=True)

# create plot dir
plot_dir = data_dir / 'plots'
plot_dir.mkdir(exist_ok=True, parents=True)

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

# for target in tce_tbl['target_id'].unique(): # for each target in tce_tbl
#     print(f'Processing target: {target}')

#     target_tces = []
#     for tce_i, tce_data in tce_tbl[tce_tbl['target_id'] == target].iterrows(): # for all tces in target
#         tce_uid = tce_data['uid']

#         target_tces.append({
#             "tce_time0bk" : tce_data['tce_time0bk'],
#             "tce_period" : tce_data['tce_period'],
#             "tce_duration" : tce_data['tce_duration']
#         })
#     print(target_tces)

#     for sector_run, sector_run_df in tce_tbl[tce_tbl['target_id'] == target].groupby('sector_run'): # for each target, sector pair
    
#         for tce_i, tce_data in sector_run_df.iterrows(): # for each tce in target, sector pair
#             #regular pipeline
            

for (target, sector_run), sector_run_data in tce_tbl.groupby(['target_id','sector_run']):
    print(f'Extracting and processing data for target: {target}, sector_run: {sector_run}')

    plot_dir_target = plot_dir / f'target_{target}'
    plot_dir_target.mkdir(exist_ok=True, parents=True)

    target_tce_data = []
    
    for tce_i, tce_data in tce_tbl[tce_tbl['target_id'] == target].iterrows(): # for all tces associated with target
        target_tce_data.append({
            "tce_time0bk" : tce_data['tce_time0bk'],
            "tce_period" : tce_data['tce_period'],
            "tce_duration" : tce_data['tce_duration']
        })


    # get individual sector runs (needed for multisector runs)
    if '-' in sector_run:
        start_sector, end_sector = [int(sector) for sector in sector_run.split('-')]
        sector_run_arr = np.arange(start_sector, end_sector + 1)
    else:
        sector_run_arr = [int(sector_run)]

    # find light curve data for target, sector_run pair
    search_lc_res = lk.search_lightcurve(target=f"tic{target}", mission='TESS', author=('TESS-SPOC', 'SPOC'),
                                        exptime=120, cadence='long', sector=sector_run_arr)

    found_sectors = [int(el.split(' ')[-1]) for el in search_lc_res.mission]
    print(f'Found {len(found_sectors)} sectors with light curve data for target, sector_run: {target},{sector_run}.')

    for sector_i, sector in enumerate(found_sectors):  # for given sector_run
        in_transit_mask = []

        data_for_tce_sector = {
            'sector': sector,
            'transit_examples': {'flux': [], 't': [], 'it_img': [], 'oot_img': [], 'diff_img': [], 'snr_img': [],
                                'target_img': [], 'target_pos': []},
            'not_transit_examples': {'flux': [], 't': [], 'it_img': [], 'oot_img': [], 'diff_img': [], 'snr_img': [],
                                    'target_img': [], 'target_pos': []},
        }

        # if len(search_lc_res) == 0:
        #     print(f'No light curve data for sector {sector_run} and TIC {tic_id}. Skipping this sector.')
        #     continue

        
        plot_dir_target_sector = plot_dir_target / f'sector_{sector}'
        plot_dir_target_sector.mkdir(exist_ok=True, parents=True)

        # download light curve fits file(s)
        lcf = search_lc_res[sector_i].download(download_dir=str(lc_dir), quality_bitmask='default',
                                                flux_column='pdcsap_flux')

        lcf = lk.LightCurve({'time': lcf.time.value, 'flux': np.array(lcf.flux.value)})

        # 1)get all in transit cadence for all TCEs in target, sector pair
        # 2)create mask of intransit cadences
        # 3)detrend flux using mask
        time,flux = lcf.time.value, lcf.flux.value
        in_transit_mask = build_transit_mask_for_lightcurve(time=lcf.time.value, tce_list=target_tce_data)

        print(f"LEN: {len(in_transit_mask)}  SUM: {np.sum(in_transit_mask)}")

        
        
        # print(f"TYPE FLUX: {type(flux)}")

        time, flux = detrend_flux_using_sg_filter(lc=lcf, mask_in_transit=in_transit_mask, win_len=1.2, sigma=5,
                                                    max_poly_order=6, penalty_weight=1, break_tolerance=5)


        for tce_i, tce_data in sector_run_data.iterrows(): # iterate over each TCE in target, sector_run pair
            #extract flux time series for each tce in tce_target_sector_run

            data_for_tce = {
            'tce_uid': sector_run_data['uid'],
            'tce_info': tce_data,
            'sectors': [],
            }

            # create directory for TCE plots
            plot_dir_target_sector_tce = plot_dir / f'tce_{tce_data["uid"]}'
            plot_dir_target_sector_tce.mkdir(exist_ok=True, parents=True)

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
                                                                    plot_dir_target_sector_tce)

            plot_dir_tce_diff_img_data = plot_dir_target_sector_tce / 'diff_img_data'
            plot_dir_tce_diff_img_data.mkdir(exist_ok=True, parents=True)

            # find target pixel data for target
            search_result = lk.search_targetpixelfile(target=f"tic{tic_id}", mission='TESS', author=('TESS-SPOC', 'SPOC'),
                                                    exptime=120, cadence="long", sector=sector)
            # download target pixel file
            tpf = search_result[0].download(download_dir=str(tpf_dir), quality_bitmask='default')

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

        data_to_tfrec.append(data_for_tce)

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