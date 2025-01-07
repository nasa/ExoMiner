
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
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator

from src_preprocessing.lc_preprocessing.utils_ephemeris import find_first_epoch_after_this_time

tce_tbl = pd.read_csv('/Users/jochoa4/Projects/exoplanet_transit_classification/ephemeris_tables/preprocessing_tce_tables/tess_2min_tces_dv_s1-s68_all_msectors_11-29-2023_2157_newlabels_nebs_npcs_bds_ebsntps_to_unks.csv')
tce_tbl = tce_tbl.loc[tce_tbl['label'].isin(['EB','KP','CP','NTP','NEB','NPC'])] #filter for relevant labels

# grouping by sorted value count to force labels with the lowest count instance to be displayed in front when plotted
counts = tce_tbl['label'].value_counts()
tce_tbl['label_counts'] = tce_tbl['label'].map(counts)
tce_tbl = tce_tbl.sort_values(by=['label_counts','label'], ascending=[True,True])
tce_tbl = tce_tbl.drop(columns=['label_counts'])

# tce_tbl = tce_tbl[::1000] #used for subsampling

def analyze_tce_window_to_period(tce_time0bk, tce_period, tce_duration, n_durations_window=5):#, num_transits_in_window = 3):
    buffer_time = 30 / 1440 # 30 mins in days
    # tce_duration /= 24 # hours to days

    # calculate window
    true_it_window_start_time = tce_time0bk - (tce_duration * n_durations_window / 2)
    true_it_window_end_time = tce_time0bk + (tce_duration * n_durations_window / 2)
    true_it_window_length = true_it_window_end_time - true_it_window_start_time 

    buffered_it_window_start_time = tce_time0bk - (tce_duration * ((n_durations_window + 1) / 2)) - buffer_time
    buffered_it_window_end_time = tce_time0bk + (tce_duration * ((n_durations_window + 1) / 2)) + buffer_time
    buffered_it_window_length = buffered_it_window_end_time - buffered_it_window_start_time

    buffered_it_window_period_ratio = buffered_it_window_length / tce_period

    # time cadence array sampled @ 2 mins
    # time = np.arange(buffered_it_window_start_time, buffered_it_window_end_time, tess_sampling_rate / 1440)

    # # find first midtransit point in the time array
    # first_transit_time = find_first_epoch_after_this_time(tce_time0bk, tce_period, time[0])
    # print(f'tce_uid: {tce_uid}')
    # print(f'time[0]: {time[0]}')
    # print(f'tce_time0bk: {tce_time0bk}')
    # print(f'tce_period: {tce_period}')
    # print(f'tce_duration: {tce_duration}')
    # print(f'first_transit_time: {first_transit_time}')
    # # compute all midtransit points in the time array
    # midtransit_points_arr = np.array([first_transit_time + phase_k * tce_period
    #                                   for phase_k in range(int(np.ceil((time[-1] - time[0]) / tce_period)))])
    
    # print(f'midtransit_points_arr:{midtransit_points_arr}')

    # # windows to consider in transit
    # start_time_windows, end_time_windows = (midtransit_points_arr - (n_durations_window + 1) * tce_duration / 2 -
    #                                     buffer_time,
    #                                     midtransit_points_arr + (n_durations_window + 1) * tce_duration / 2 +
    #                                     buffer_time)
    
    # it_window_mask = np.zeros(len(time), dtype=bool)
    # for start_time, end_time in zip(start_time_windows, end_time_windows):
    #     it_window_mask |= np.logical_and(time >= start_time, time <= end_time)
    
    # oot_points_arr = time[~it_window_mask] # points considered for oot beyond window
    # #size of windows in days
    # #size of window / period ratio
    return true_it_window_length, buffered_it_window_length, buffered_it_window_period_ratio

    # time_it_cadence_arr = tce_time0bk - tce_duration / 2 <= time_arr <= tce_time0bk + tce_duration / 2 

    # num_periods_set = set()
    # # determine which cadences are in transit
    # for cadence_i, cadence in enumerate(time_arr):
    #     # num periods from cadence to tce_time0bk, rounded to fall on actual period. 1.5 period lengths -> 2 | -1.5 -> -2
    #     num_periods_from_reference_time = round((cadence - tce_time0bk) / tce_period)
    #     closest_transit_center = tce_time0bk + num_periods_from_reference_time * tce_period

    #     transit_start = closest_transit_center - tce_duration / 2
    #     transit_end = closest_transit_center + tce_duration / 2

    #     if transit_start <= cadence <= transit_end:
    #         # mark time_it_cadence_arr as transiting and add period used to set
    #         time_it_cadence_arr[cadence_i] = 1
    #         num_periods_set.add(num_periods_from_reference_time)
    
    
    # in_transit_mask = np.zeros(len(time_arr), dtype=bool)
    # in_transit_mask |= np.abs((time_arr - tce_time0bk) % tce_period ) < tce_duration 
    # print(time_it_cadence_arr.sum(), in_transit_mask.sum())
    # #number of transits in window
    # n_transits_in_window = len(num_periods_set)

    # if len(num_periods_set) < 1:
    #     print(f"ERROR: {num_periods_set}")


def build_and_save_plots(df, plot_dir):
    # Joint Plots & Save
    palette = "colorblind"
    max_value = max(df['buffered_it_window_period_ratio'])
    discrete_bins = np.arange(0, max_value + 0.25 + 0.001, 0.25) if max_value >= 1 else [0, 0.25, 0.5, 0.75, 1.0]
    print(f"MAX_ALL: {max_value}")

    # 1. Violin Plot for Disposition
    plt.figure(figsize=(10,6))
    sns.set_palette("flare")
    sns.violinplot(data=df, x='tce_label', y=f'buffered_it_window_period_ratio', hue='tce_label', linewidth=1)
    plt.ylim(0,None)
    plt.title(f'Violin Plot of Buffered In-Transit Window Length to Period Ratio per TCE observation by Disposition')
    plt.xlabel('TCE Label')
    plt.ylabel('Buffered In-Transit Window Length to Period Ratio per TCE observation')
    plt.tight_layout()
    plt.savefig(plot_dir / f"violin_plot_all_dispositions.png", format="png", dpi=300, bbox_inches="tight")
    plt.close()

    # 2. Hist Plot for all Dispositions
    plt.figure(figsize=(10,6))
    sns.set_palette(palette)
    sns.displot(data=df, x=f'buffered_it_window_period_ratio', hue='tce_label', element="step", bins=discrete_bins)
    plt.title(f'Distribution of Buffered In-Transit Window Length to Period Ratio per TCE observation (Full Range)')
    plt.xlabel('Buffered In-Transit Window Length to Period Ratio per TCE observation')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(plot_dir / f"hist_plot_all_dispositions.png", format="png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # 3. Normalized Hist Plot for all Dispositions
    plt.figure(figsize=(10,6))
    sns.set_palette(palette)
    sns.displot(data=df, x=f'buffered_it_window_period_ratio', hue='tce_label', stat="density", common_norm=False, element="step", bins=discrete_bins) #linewidth=1)
    plt.title(f'Normalized Distribution of Buffered In-Transit Window Length to Period Ratio per TCE observation (Full Range)')
    plt.xlabel('Buffered In-Transit Window Length to Period Ratio per TCE observation')
    plt.ylabel('Probability Density of TCEs')
    plt.tight_layout()
    plt.savefig(plot_dir / f"hist_plot_all_dispositions_normalized.png", format="png", dpi=300, bbox_inches="tight")
    plt.close()

    # 4. Log Scale Hist Plot for all Dispositions
    plt.figure(figsize=(10,6))
    sns.set_palette(palette)
    sns.displot(data=df, x=f'buffered_it_window_period_ratio', hue='tce_label', log_scale=(True,False), stat="count", element="step")#linewidth=1)
    plt.title(f'Log-Scaled Distribution of Buffered In-Transit Window Length to Period Ratio per TCE observation (Full Range)')
    plt.xlabel('Log(Buffered In-Transit Window Length to Period Ratio per TCE observation)')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(plot_dir / f"hist_plot_all_dispositions_log_scale.png", format="png", dpi=300, bbox_inches="tight")
    plt.close()

    # Individual Plots:
    for label_i, label in enumerate(df['tce_label'].unique()):
        #1. Hist Plot of IT Counts by Disposition
        plt.figure(figsize=(10,6))
        label_filtered_df = df.loc[df['tce_label'] == label]
        print(f"MAX_LABEL_{label}: {max(label_filtered_df['buffered_it_window_period_ratio'])}")
        max_value = max(label_filtered_df['buffered_it_window_period_ratio'])
        discrete_bins = np.arange(0, max_value + 0.25 + 0.001, 0.25) if max_value >= 1 else [0, 0.25, 0.5, 0.75, 1.0]
        sns.displot(data=label_filtered_df, x=f'buffered_it_window_period_ratio', color=sns.color_palette(palette,6)[label_i], bins=discrete_bins)
        plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True)) #set y axis to display as int for small frequencies
        plt.title(f'Distribution of Buffered In-Transit Window Length to Period Ratio per TCE observation for {label}\'s')
        plt.xlabel('Buffered In-Transit Window Length to Period Ratio per TCE observation')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(plot_dir / f"hist_plot_disposition_{label}.png", format="png", dpi=300, bbox_inches="tight")
        plt.close()
    

"""
plots/
    resample_100/

                hist_plot_all_dispositions_100.png
                hist_plot_all_dispositions_normalized_100.png
                hist_plot_all_dispositions_log_scale_100.png
                hist_plot_all_dispositions_0-10_100.png
                hist_plot_all_dispositions_normalized_0-10_100.png
                hist_plot_disposition_EB_100.png
                hist_plot_disposition_CP_100.png
                hist_plot_disposition_KP_100.png
                hist_plot_disposition_NTP_100.png
                hist_plot_disposition_NEB_100.png
                hist_plot_disposition_NPC_100.png
                violin_plot_all_dispositions_100.png
"""

if __name__ == '__main__':
    
    save_path = Path('/Users/jochoa4/Downloads/transit_detection/data/tce_analysis/tce_window_period_analysis')
    save_path.mkdir(exist_ok=True, parents=True)

    # tce_tbl = tce_tbl[:]
    n_durations_window = 5
    resampling_rates = (100, 200, 500)

    df_columns = ['tce_uid','tce_label','target_id', 'sector_run','tce_time0bk','tce_period','tce_duration', 'true_it_window_length', 'buffered_it_window_length', 'buffered_it_window_period_ratio']#,'transits_per_window']

    df = pd.DataFrame(columns=df_columns)

    for tce_i, tce_data in tce_tbl.iterrows(): # for all tces associated with target

        tce_uid = tce_data['uid']
        tce_label = tce_data['label']
        target_id = tce_data['target_id']
        sector_run = tce_data['sector_run']
        tce_time0bk = tce_data['tce_time0bk'] # center of first transit
        tce_period = tce_data['tce_period']
        tce_duration = tce_data['tce_duration'] / 24

        tce_data = pd.DataFrame({
            'tce_uid' : [tce_uid],
            'tce_label' : [tce_label],
            'target_id' : [target_id],
            'sector_run' : [sector_run],
            'tce_time0bk' : [tce_time0bk],
            'tce_period' : [tce_period],
            'tce_duration' : [tce_duration],
           # 'transits_per_window' : num_transits_in_window
        })
        true_it_window_length, buffered_it_window_length, buffered_it_window_period_ratio= analyze_tce_window_to_period(tce_time0bk, tce_period, tce_duration, n_durations_window)

        tce_data['true_it_window_length'] = true_it_window_length
        tce_data['buffered_it_window_length'] = buffered_it_window_length
        tce_data['buffered_it_window_period_ratio'] = buffered_it_window_period_ratio

        df = pd.concat([df, tce_data], ignore_index=True)
    
    csv_name = "tce_window_period_analysis" + '.csv' 
    df_save_path = save_path / csv_name

    ss_save_path = save_path / 'summary_statistics'
    ss_save_path.mkdir(exist_ok=True, parents=True)

    summary_stats = df.groupby('tce_label')[f'buffered_it_window_period_ratio'].agg(['min','max','std','mean','median','count'])
    cumulative_stats = pd.DataFrame({
        'min' : [df[f'buffered_it_window_period_ratio'].min()],
        'max' : [df[f'buffered_it_window_period_ratio'].max()],
        'std' : [df[f'buffered_it_window_period_ratio'].std()],
        'mean' : [df[f'buffered_it_window_period_ratio'].mean()],
        'median' : [df[f'buffered_it_window_period_ratio'].median()],
        'count' : [df[f'buffered_it_window_period_ratio'].count()]
    }, index=['Overall'])
    # summary_stats = pd.concat([summary_stats,cumulative_stats]).round(3) 

    summary_stats.to_csv( ss_save_path / f'summary_statistics.csv' )

    #export csv
    df.to_csv(df_save_path, index=False)

    #create and export plots
    plot_dir = save_path / 'plots'
    plot_dir.mkdir(exist_ok=True, parents=True)

    build_and_save_plots(df, plot_dir)

