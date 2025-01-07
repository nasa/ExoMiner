
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

def analyze_tce_window_points(tce_time0bk, tce_duration, n_durations_window=5):#, num_transits_in_window = 3):
    tess_sampling_rate = 2 # mins

    # # calculate window
    window_start_time = tce_time0bk - tce_duration * n_durations_window / 2
    window_end_time = tce_time0bk + tce_duration * n_durations_window / 2

    # # time cadence array based on tess_sampling_rate
    time_arr = np.arange(window_start_time, window_end_time, tess_sampling_rate / 1440)
    
    expected_points_in_window = len(time_arr)

    return expected_points_in_window

# tce_tbl = tce_tbl[::1000] #used for subsampling

def build_and_save_plots(df, base_plot_dir):
    # Joint Plots & Save
    palette = "colorblind"
    y_axis_arr = [f'tce_duration', f'expected_points_in_window']

    plot_name_arr = [f'Duration', 'Num Expected Points in Window']

    for y_axis, plot_name in zip(y_axis_arr, plot_name_arr):
        plot_dir = base_plot_dir / y_axis
        plot_dir.mkdir(exist_ok=True, parents=True)

        max_value = max(df[y_axis])
        discrete_bins = np.arange(0, max_value + 0.25 + 0.001, 0.25) if max_value >= 1 else [0, 0.25, 0.5, 0.75, 1.0]

        # 1. Violin Plot for Disposition
        plt.figure(figsize=(10,6))
        sns.set_palette("flare")
        sns.violinplot(data=df, x='tce_label', y=y_axis, hue='tce_label', linewidth=1)
        plt.ylim(0,None)
        plt.title(f'Violin Plot of {plot_name} per TCE observation by Disposition')
        plt.xlabel('TCE Label')
        plt.ylabel(f'{plot_name} per TCE observation')
        plt.tight_layout()
        plt.savefig(plot_dir / f"violin_plot_all_dispositions.png", format="png", dpi=300, bbox_inches="tight")
        plt.close()

        # 2. Hist Plot for all Dispositions
        plt.figure(figsize=(10,6))
        sns.set_palette(palette)
        sns.displot(data=df, x=y_axis, hue='tce_label', element="step", bins=discrete_bins)
        plt.title(f'Distribution of {plot_name} per TCE observation (Full Range)')
        plt.xlabel(f'{plot_name} per TCE observation')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(plot_dir / f"hist_plot_all_dispositions.png", format="png", dpi=300, bbox_inches="tight")
        plt.close()
        
        # 3. Normalized Hist Plot for all Dispositions
        plt.figure(figsize=(10,6))
        sns.set_palette(palette)
        sns.displot(data=df, x=y_axis, hue='tce_label', stat="density", common_norm=False, element="step", bins=discrete_bins) #linewidth=1)
        plt.title(f'Normalized Distribution of {plot_name} per TCE observation (Full Range)')
        plt.xlabel(f'{plot_name} per TCE observation')
        plt.ylabel('Probability Density of TCEs')
        plt.tight_layout()
        plt.savefig(plot_dir / f"hist_plot_all_dispositions_normalized.png", format="png", dpi=300, bbox_inches="tight")
        plt.close()

        # 4. Log Scale Hist Plot for all Dispositions
        plt.figure(figsize=(10,6))
        sns.set_palette(palette)
        sns.displot(data=df, x=y_axis, hue='tce_label', log_scale=(True,False), stat="count", element="step")#linewidth=1)
        plt.title(f'Log-Scaled Distribution of {plot_name} per TCE observation (Full Range)')
        plt.xlabel(f'Log({plot_name} per TCE observation)')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(plot_dir / f"hist_plot_all_dispositions_log_scale.png", format="png", dpi=300, bbox_inches="tight")
        plt.close()

        # Individual Plots:
        for label_i, label in enumerate(df['tce_label'].unique()):
            #1. Hist Plot of IT Counts by Disposition
            plt.figure(figsize=(10,6))
            label_filtered_df = df.loc[df['tce_label'] == label]

            max_value = max(label_filtered_df[y_axis])
            discrete_bins = np.arange(0, max_value + 0.25 + 0.001, 0.25) if max_value >= 1 else [0, 0.25, 0.5, 0.75, 1.0]
            sns.displot(data=label_filtered_df, x=y_axis, color=sns.color_palette(palette,6)[label_i], bins=discrete_bins)
            plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True)) #set y axis to display as int for small frequencies
            plt.title(f'Distribution of {plot_name} per TCE observation for {label}\'s')
            plt.xlabel(f'{plot_name} per TCE observation')
            plt.ylabel('Frequency')
            plt.tight_layout()
            plt.savefig(plot_dir / f"hist_plot_disposition_{label}.png", format="png", dpi=300, bbox_inches="tight")
            plt.close()

            plt.figure(figsize=(10,6))
            sns.set_palette(palette)
            sns.displot(data=label_filtered_df, x=y_axis, hue='tce_label', log_scale=(True,False), stat="count", element="step")#linewidth=1)
            plt.title(f'Log-Scaled Distribution of {plot_name} per TCE observation for {label}\'s)')
            plt.xlabel(f'Log({plot_name} per TCE observation)')
            plt.ylabel('Frequency')
            plt.tight_layout()
            plt.savefig(plot_dir / f"hist_plot_disposition_{label}_log_scale.png", format="png", dpi=300, bbox_inches="tight")
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

    save_path = Path('/Users/jochoa4/Downloads/transit_detection/data/tce_analysis/tce_duration_analysis')
    save_path.mkdir(exist_ok=True, parents=True)

    # tce_tbl = tce_tbl[:]
    n_durations_window = 5

    df_columns = ['tce_uid','tce_label','target_id', 'sector_run','tce_time0bk','tce_period','tce_duration', 'expected_points_in_window']#,'transits_per_window']

    df = pd.DataFrame(columns=df_columns)

    for tce_i, tce_data in tce_tbl.iterrows(): # for all tces associated with target

        tce_uid = tce_data['uid']
        tce_label = tce_data['label']
        target_id = tce_data['target_id']
        sector_run = tce_data['sector_run']
        tce_time0bk = tce_data['tce_time0bk'] # center of first transit
        tce_period = tce_data['tce_period']
        tce_duration = tce_data['tce_duration'] / 24
        expected_points_in_window = analyze_tce_window_points(tce_time0bk=tce_time0bk, tce_duration=tce_duration, n_durations_window=5)

        tce_data = pd.DataFrame({
            'tce_uid' : [tce_uid],
            'tce_label' : [tce_label],
            'target_id' : [target_id],
            'sector_run' : [sector_run],
            'tce_time0bk' : [tce_time0bk],
            'tce_period' : [tce_period],
            'tce_duration' : [tce_duration],
            'expected_points_in_window' : expected_points_in_window
           # 'transits_per_window' : num_transits_in_window
        })

        df = pd.concat([df, tce_data], ignore_index=True)
    
    csv_name = "tce_duration_analysis" + '.csv' 
    df_save_path = save_path / csv_name

    ss_save_path = save_path / 'summary_statistics'
    ss_save_path.mkdir(exist_ok=True, parents=True)
    
    fields = ['tce_duration', 'expected_points_in_window']

    for field in fields:
        summary_stats = df.groupby('tce_label')[field].agg(['min','max','std','mean','median','count'])

        # cumulative_stats = pd.DataFrame({
        #     'min' : [df[field].min()],
        #     'max' : [df[field].max()],
        #     'std' : [df[field].std()],
        #     'mean' : [df[field].mean()],
        #     'median' : [df[field].median()],
        #     'count' : [df[field].count()]
        # }, index=['Overall'])
        
        # summary_stats = pd.concat([summary_stats,cumulative_stats]).round(3) 
        if field == 'expected_points_in_window':
            frac = df.groupby('tce_label')[field].apply(
                lambda x: (x > 100).sum() / len(x)
            )
            summary_stats['frac_above_100'] = frac
        summary_stats.to_csv( ss_save_path / f'summary_statistics_{field}.csv' )

    #export csv
    df.to_csv(df_save_path, index=False)

    #create and export plots
    plot_dir = save_path / 'plots'
    plot_dir.mkdir(exist_ok=True, parents=True)

    build_and_save_plots(df, plot_dir)

