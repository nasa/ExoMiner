# TESS TCE Ephemerides Data Analysis Format

## Data Overview

The data was constructed using TESS TCE Ephemerides data from the SPOC pipeline, with the goal of estimating the 
number of in-transit points for each TCE, per window and per transit, across different resampling rates. The process
involved defining an observation window for each TCE and calculating an estimated number of in-transit points, per
observation, based on the window.

### Window Calculation

For each TCE:

**Window**: 
Defined using five transit durations
- **window_start**: at time tce_time0bk - tce_duration * 2.5
- **window_end**: at time tce_time0bk + tce_duration * 2.5

### Resampling and In-Transit Point Calculation 

**Resampling Rates**:
Data is resampled at various scales: (100, 200, 500), assuming that number of points after resampling per window.

**Resampling Calculation**:
- **tess_sampling_rate** = 2 (in mins)
- **n_resampled_it_points** = A list containing the resampled_it_points per resampling rate
- **time_cadence_arr** = a list of time cadences (BJD) from window_start to window_end, spaced by tess_sampling_rate / 1440
- **time_it_cadence_arr** = a list full of 0's of len(time_arr)
- **num_periods_set** = a set of unique period lengths used in determining the n_transits_in_window
- **n_transits_in_window** : number of unique transits in window

***Process***:
- Iterate over each cadence in time_cadence_arr, and mark a cadence in time_it_cadence_arr as 1 if it falls on a time stamp of transit phase for the TCE.

- For each cadence define:

        num_periods_from_reference_time = round((cadence - tce_time0bk) / tce_period)
        closest_transit_center = tce_time0bk + num_periods_from_reference_time * tce_period
        
        transit_start = closest_transit_center - tce_duration / 2
        transit_end = closest_transit_center + tce_duration / 2

        if transit_start <= cadence <= transit_end:
            time_it_cadence_arr[cadence_i] = 1
            num_periods_set.add(num_periods_from_reference_time)

- Iterate over each resampling rate and resample the time_it_cadence_arr, and add resampled_num_it_points for that resampling rate to n_resampled_it_points
- For each resampling rate:

        resampled_it_indices = np.linspace(0, len(time_it_cadence_arr) - 1, resampled_num_points, dtype=int)
        resampled_time_it_cadence_arr = time_it_cadence_arr[resampled_it_indices]

        resampled_num_it_points = np.sum(resampled_time_it_cadence_arr)
        n_resampled_it_points.append(resampled_num_it_points)

## Data

### Exported Resampling Analysis

*Provided 1 csv containing all ephemerides and estimated in-transit points per tce observation in form ```tce_resampling_analysis_100_200_500.csv```*

**Ephemerides fields**: 'tce_uid', 'tce_label', 'target_id', 'sector_run', 'tce_time0bk', 'tce_period', 'tce_duration' 

**Calculated fields**: for each TCE observation at each resampling rate: 'it_points_per_window_100', 'it_points_per_observation_100', 'it_points_per_window_200', 'it_points_per_observation_200', 'it_points_per_window_500', 'it_points_per_observation_500'.

### Exported Summary Statistics
*Provided 3 csvs containing summary statistics at each resampling rate per window eg. ```summary_statistics_per_window_100.csv```*
*Provided 3 csvs containing summary statistics at each resampling rate per observation eg. ```summary_statistics_per_observation_100.csv```*

**Summary Statistics**: Min, max, std, mean, median, and count of estimated in-transit points per TCE observation by label and across all data.

### Combined Plots:
Provided per resampling rate, eg. with resampling rate 100: ```hist_plot_all_dispositions_100.png```
Provided per window and per observation.

*Contains estimated in-transit data for all dispositions, grouped by disposition*
***Full Range***
1. **Violin Plot** : Kernel Density Estimate and Box & Whisker
2. **Histogram** : Estimated In-Transit Points per observation/window vs. Frequency 
3. **Normalized Histogram** : Estimated In-Transit Points per observation/window vs. Probability Density
4. **Log-Scale (x-axis) Histogram**: Log(Estimated In-Transit Points per observation/window) vs. Frequency

***0-10 Range***
1. **Histogram 0-10 range** : Histogram with x-axis limited to estimates between 0-10
2. **Normalized Histogram 0-10 range** : Normalized Histogram with x-axis limited to estimates between 0-10

### By Label Plots:
*Contains estimated in-transit data only for each unique disposition*
1. **Histogram** : Estimated In-Transit Points per observation/window vs. Frequency

### Directory Format:
```
tce_analysis/ 
            tce_resampling_analysis_100_200_500.csv
            summary_statistics/
                                summary_statistics_per_window_100.csv
                                summary_statistics_per_window_200.csv
                                summary_statistics_per_window_500.csv
                                summary_statistics_per_observation_100.csv
                                summary_statistics_per_observation_200.csv
                                summary_statistics_per_observation_500.csv
            plots/
                resample_100/
                            per_observation/
                                hist_plot_all_dispositions_per_observation_100.png
                                hist_plot_all_dispositions_normalized_per_observation_100.png
                                hist_plot_all_dispositions_log_scale_per_observation_100.png
                                hist_plot_all_dispositions_0-10_per_observation_100.png
                                hist_plot_all_dispositions_normalized_0-10_per_observation_100.png
                                hist_plot_disposition_EB_per_observation_100.png
                                hist_plot_disposition_CP_per_observation_100.png
                                hist_plot_disposition_KP_per_observation_100.png
                                hist_plot_disposition_NTP_per_observation_100.png
                                hist_plot_disposition_NEB_per_observation_100.png
                                hist_plot_disposition_NPC_per_observation_100.png
                                violin_plot_all_dispositions_per_observation_100.png
                            per_window/
                                hist_plot_all_dispositions_per_window_100.png
                                hist_plot_all_dispositions_normalized_per_window_100.png
                                hist_plot_all_dispositions_log_scale_per_window_100.png
                                hist_plot_all_dispositions_0-10_per_window_100.png
                                hist_plot_all_dispositions_normalized_0-10_per_window_100.png
                                hist_plot_disposition_EB_per_window_100.png
                                hist_plot_disposition_CP_per_window_100.png
                                hist_plot_disposition_KP_per_window_100.png
                                hist_plot_disposition_NTP_per_window_100.png
                                hist_plot_disposition_NEB_per_window_100.png
                                hist_plot_disposition_NPC_per_window_100.png
                                violin_plot_all_dispositions_per_window_100.png
                resample_200/
                            ...
                resample_500/
                            ...
```





