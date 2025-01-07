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

tce_tbl = pd.read_csv('/Users/jochoa4/Projects/exoplanet_transit_classification/ephemeris_tables/preprocessing_tce_tables/tess_2min_tces_dv_s1-s68_all_msectors_11-29-2023_2157_newlabels_nebs_npcs_bds_ebsntps_to_unks.csv')
tce_tbl = tce_tbl.loc[tce_tbl['label'].isin(['EB','KP','CP','NTP','NEB','NPC'])] #filter for relevant labels

# grouping by sorted value count to force labels with the lowest count instance to be displayed in front when plotted
counts = tce_tbl['label'].value_counts()
tce_tbl['label_counts'] = tce_tbl['label'].map(counts)
tce_tbl = tce_tbl.sort_values(by=['label_counts','label'], ascending=[True,True])
tce_tbl = tce_tbl.drop(columns=['label_counts'])

# tce_tbl = tce_tbl[::100] #used for subsampling

def analyze_tce_sampling(tce_time0bk, tce_period, tce_duration, resampling_rates, num_transits_in_window = 3):
    # tess sampling rate (2 mins) not included in calculation as it cancels out assuming interpolation

    days_to_mins_scalar = (60*24)

    estimated_it_points_per_observation = []
    for resampling_rate in resampling_rates:

        window_start_time = tce_time0bk - 0.5 * tce_duration
        window_end_time = tce_time0bk + tce_period * (num_transits_in_window - 1) + 0.5 * tce_duration
        
        window_duration_mins = (window_end_time - window_start_time) * days_to_mins_scalar

        total_it_duration_mins = tce_duration * num_transits_in_window * days_to_mins_scalar

        resample_scale = resampling_rate / window_duration_mins
        resampled_it_points = total_it_duration_mins * resample_scale

        estimated_it_points_per_observation.append(resampled_it_points / num_transits_in_window)
    
    return estimated_it_points_per_observation

def build_and_save_plots(df, plot_dir, resampling_rate):
    # Joint Plots & Save
    palette = "colorblind"
    discrete_bins = list(range(resampling_rate + 1))
    # 1. Violin Plot for Disposition
    plt.figure(figsize=(10,6))
    sns.set_palette("flare")
    sns.violinplot(data=df, x='tce_label', y=f'it_points_per_observation_{resampling_rate}', hue='tce_label', linewidth=1)
    plt.ylim(0,None)
    plt.title(f'Violin Plot of Estimated In-Transit Point Count per TCE observation by Disposition')
    plt.xlabel('TCE Label')
    plt.ylabel('Estimated In-Transit Points per TCE observation')
    plt.tight_layout()
    plt.savefig(plot_dir / f"violin_plot_all_dispositions_{resampling_rate}.png", format="png", dpi=300, bbox_inches="tight")
    plt.close()

    # 2. Hist Plot for all Dispositions
    plt.figure(figsize=(10,6))
    sns.set_palette(palette)
    sns.displot(data=df, x=f'it_points_per_observation_{resampling_rate}', hue='tce_label', bins=discrete_bins, element="step")
    plt.title(f'Distribution of Estimated In-Transit Points per TCE observation (Full Range)')
    plt.xlabel('Estimated In-Transit Points per TCE observation')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(plot_dir / f"hist_plot_all_dispositions_{resampling_rate}.png", format="png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # 3. Normalized Hist Plot for all Dispositions
    plt.figure(figsize=(10,6))
    sns.set_palette(palette)
    sns.displot(data=df, x=f'it_points_per_observation_{resampling_rate}', hue='tce_label', stat="density", common_norm=False, bins=discrete_bins, element="step") #linewidth=1)
    plt.title(f'Normalized Distribution of Estimated In-Transit Points per TCE observation (Full Range)')
    plt.xlabel('Estimated In-Transit Points per TCE observation')
    plt.ylabel('Probability Density of TCEs')
    plt.tight_layout()
    plt.savefig(plot_dir / f"hist_plot_all_dispositions_normalized_{resampling_rate}.png", format="png", dpi=300, bbox_inches="tight")
    plt.close()

    # 4. Log Scale Hist Plot for all Dispositions
    plt.figure(figsize=(10,6))
    sns.set_palette(palette)
    sns.displot(data=df, x=f'it_points_per_observation_{resampling_rate}', hue='tce_label', log_scale=(True,False), stat="count", element="step")#linewidth=1)
    plt.title(f'Log-Scaled Distribution of Estimated In-Transit Points per TCE observation (Full Range)')
    plt.xlabel('Log(Estimated In-Transit Points per TCE observation)')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(plot_dir / f"hist_plot_all_dispositions_log_scale_{resampling_rate}.png", format="png", dpi=300, bbox_inches="tight")
    plt.close()

    # 5. Hist Plot for 0-10 for all Dispositions
    plt.figure(figsize=(10,6))
    sns.set_palette(palette)
    sns.displot(data=df, x=f'it_points_per_observation_{resampling_rate}', hue='tce_label', bins=list(range(11)), binrange=(0,10), element="step",)
    plt.title(f'Distribution of Estimated In-Transit Points per TCE observation (0-10 Range)')
    plt.xlabel('Estimated In-Transit Points per TCE observation')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(plot_dir / f"hist_plot_all_dispositions_0-10_{resampling_rate}.png", format="png", dpi=300, bbox_inches="tight")
    plt.close()

    # 6. Normalized Hist Plot for 0-10 for all Dispositions
    plt.figure(figsize=(10,6))
    sns.set_palette(palette)
    sns.displot(data=df, x=f'it_points_per_observation_{resampling_rate}', hue='tce_label', stat="density", common_norm=False, bins=list(range(11)), binrange=(0,10), element='step')
    plt.title(f'Normalized Distribution of Estimated In-Transit Points per TCE observation (0-10 Range)')
    plt.xlabel('Estimated In-Transit Points per TCE observation')
    plt.ylabel('Probability Density of TCEs')
    plt.tight_layout()
    plt.savefig(plot_dir / f"hist_plot_all_dispositions_0-10_normalized_{resampling_rate}.png", format="png", dpi=300, bbox_inches="tight")
    plt.close()

    # Individual Plots:
    for label_i, label in enumerate(df['tce_label'].unique()):
        #1. Hist Plot of IT Counts by Disposition
        plt.figure(figsize=(10,6))
        sns.displot(data=df.loc[df['tce_label'] == label], x=f'it_points_per_observation_{resampling_rate}', color=sns.color_palette(palette,6)[label_i], bins=discrete_bins)
        plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True)) #set y axis to display as int for small frequencies
        plt.title(f'Distribution of Estimated In-Transit Points per TCE observation for {label}\'s')
        plt.xlabel('Estimated In-Transit Points per TCE observation')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(plot_dir / f"hist_plot_disposition_{label}_{resampling_rate}.png", format="png", dpi=300, bbox_inches="tight")
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
    
    save_path = Path('/Users/jochoa4/Downloads/transit_detection/data/tce_analysis/tce_ephemerides_analysis')
    save_path.mkdir(exist_ok=True, parents=True)

    # tce_tbl = tce_tbl[:]
    num_transits_in_window = 3
    resampling_rates = (100, 200, 500)

    df_columns = ['tce_uid','tce_label','target_id', 'sector_run','tce_time0bk','tce_period','tce_duration']#,'transits_per_window']
    rr_columns = [f"it_points_per_observation_{rr}" for rr in resampling_rates]
    df_columns.extend(rr_columns)

    df = pd.DataFrame(columns=df_columns)

    for tce_i, tce_data in tce_tbl.iterrows(): # for all tces associated with target

        tce_uid = tce_data['uid']
        tce_label = tce_data['label']
        target_id = tce_data['target_id']
        sector_run = tce_data['sector_run']
        tce_time0bk = tce_data['tce_time0bk'] # center of first transit
        tce_period = tce_data['tce_period']
        tce_duration = tce_data['tce_duration']

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

        estimated_it_points_per_rr = analyze_tce_sampling(tce_time0bk, tce_period, tce_duration, resampling_rates, num_transits_in_window)

        for resampling_rate, estimated_it_points in zip(resampling_rates, estimated_it_points_per_rr):
            tce_data[f'it_points_per_observation_{resampling_rate}'] = estimated_it_points

        df = pd.concat([df, tce_data], ignore_index=True)

    resample_suffix = '_'.join([str(rate) for rate in resampling_rates]) # (100,200) -> 100_200
    csv_name = "tce_resampling_analysis_" + resample_suffix + '.csv' 
    df_save_path = save_path / csv_name

    ss_save_path = save_path / 'summary_statistics'
    ss_save_path.mkdir(exist_ok=True, parents=True)

    for resampling_rate in resampling_rates:

        summary_stats = df.groupby('tce_label')[f'it_points_per_observation_{resampling_rate}'].agg(['min','max','std','mean','median','count'])
        cumulative_stats = pd.DataFrame({
            'min' : [df[f'it_points_per_observation_{resampling_rate}'].min()],
            'max' : [df[f'it_points_per_observation_{resampling_rate}'].max()],
            'std' : [df[f'it_points_per_observation_{resampling_rate}'].std()],
            'mean' : [df[f'it_points_per_observation_{resampling_rate}'].mean()],
            'median' : [df[f'it_points_per_observation_{resampling_rate}'].median()],
            'count' : [df[f'it_points_per_observation_{resampling_rate}'].count()]
        }, index=['Overall'])
        # summary_stats = pd.concat([summary_stats,cumulative_stats]).round(3) 

        summary_stats.to_csv( ss_save_path / f'summary_statistics_{resampling_rate}.csv' )

    #export csv
    df.to_csv(df_save_path, index=False)

    #create and export plots
    for resampling_rate in resampling_rates:
        plot_dir = save_path / 'plots' / f'resample_{resampling_rate}'
        plot_dir.mkdir(exist_ok=True, parents=True)

        build_and_save_plots(df, plot_dir, resampling_rate)

