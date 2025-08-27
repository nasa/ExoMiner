# TESS TCE Ephemerides Data Analysis Format

## Data Overview

The data was constructed using TESS TCE Ephemerides data from the SPOC pipeline, with the goal of estimating the number of in-transit points
for each TCE across different resampling rates. The process involved defining an observation window for each 
TCE and calculating an estimated number of in-transit points, per observation, based on the window.

### Window Calculation

For each TCE:

**Window**: 
Defined using three TCE observations
- **Starts**: at time tce_time0bk - 0.5 * tce_duration
- **Ends**: at time tce_time0bk + 2 * tce_period + 0.5 * tce_duration

### Resampling and In-Transit Point Calculation
*NOTEs: 1. All calculations involving time are performed using a recalculation of relative BJD to minutes (time/60*24) 2. TESS sampling_rate of 2 mins was ommitted in calculation as it cancels out when resampling with interpolation*

**Resampling Rates**:
Data is resampled at various scales: (100, 200, 500), assuming that number of points after resampling per window.

**Resampling Calculation**:
- **num_observations_per_window** = 3
- **window_duration** = window_end_time - window_start_time
- **total_it_duration** = tce_duration * num_transits_in_window
- **resample_scale** = resampling_rate / window_duration
- **total_resampled_it_points** = total_it_duration_mins * resample_scale
- **estimated_it_points_per_observation** = total_it_duration / num_observations_per_window

## Data

### Exported Resampling Analysis

*Provided 1 csv containing all ephemerides and estimated in-transit points per tce observation in form ```tce_resampling_analysis_100_200_500.csv```*

**Ephemerides fields**: 'tce_uid', 'tce_label', 'target_id', 'sector_run', 'tce_time0bk', 'tce_period', 'tce_duration' 

**Calculated fields**: for each TCE observation at each resampling rate: 'it_points_per_observation_100', 'it_points_per_observation_200', 'it_points_per_observation_500'.

### Exported Summary Statistics
*Provided 3 csvs containing summary statistics at each resampling rate eg. ```summary_statistics_100.csv```*

**Summary Statistics**: Min, max, std, mean, median, and count of estimated in-transit points per TCE observation by label and across all data.

### Combined Plots:
Provided per resampling rate, eg. with resampling rate 100: ```hist_plot_all_dispositions_100.png```
*Contains estimated in-transit data for all dispositions, grouped by disposition*
***Full Range***
1. **Violin Plot** : Kernel Density Estimate and Box & Whisker
2. **Histogram** : Estimated In-Transit Points per observation vs. Frequency 
3. **Normalized Histogram** : Estimated In-Transit Points per observation vs. Probability Density
4. **Log-Scale (x-axis) Histogram**: Log(Estimated In-Transit Points per observation) vs. Frequency

***0-10 Range***

1. **Histogram 0-10 range** : Histogram with x-axis limited to estimates between 0-10
2. **Normalized Histogram 0-10 range** : Normalized Histogram with x-axis limited to estimates between 0-10

### By Label Plots:
*Contains estimated in-transit data only for each unique disposition*
1. **Histogram** : Estimated In-Transit Points per observation vs. Frequency

### Directory Format:
```
tce_analysis/ 
            tce_resampling_analysis_100_200_500.csv
            summary_statistics/
                                summary_statistics_100.csv
                                summary_statistics_200.csv
                                summary_statistics_500.csv
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
                resample_200/
                            ...
                resample_500/
                            ...
```
