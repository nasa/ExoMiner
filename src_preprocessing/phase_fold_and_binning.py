"""
Functions for phase folding time series and bin them.
"""

# 3rd party
import numpy as np
from astropy.stats import mad_std

# local
from src_preprocessing.light_curve import median_filter, util
from src_preprocessing.utils_imputing import impute_binned_ts
from src_preprocessing.utils_preprocessing_io import report_exclusion


def phase_fold_and_sort_light_curve(time, timeseries, period, t0, augmentation=False):
    """ Phase folds a light curve and sorts by ascending time.

    Args:
      time: 1D NumPy array of time values.
      timeseries: 1D NumPy array of time series values.
      period: A positive real scalar; the period to fold over.
      t0: The center of the resulting folded vector; this value is mapped to 0.
      augmentation: bool, set to True to augment phases through sample with replacement

    Returns:
      time: 1D NumPy array of phase folded time values in
          [-period / 2, period / 2), where 0 corresponds to t0 in the original
          time array. Values are sorted in ascending order.
      timeseries: 1D NumPy array. Values are the same as the original input
          array, but sorted by folded_time.
      num_transits: int, number of transits in the time series.
    """

    # Phase fold time.
    if not augmentation:
        time = util.phase_fold_time(time, period, t0)
    else:
        time, sampled_idxs, num_transits = util.phase_fold_time_aug(time, period, t0)
        timeseries = timeseries[sampled_idxs]

    # count number of transits in the phase domain
    num_transits = np.sum(np.diff(time) < 0) + 1

    # Sort by ascending time.
    sorted_i = np.argsort(time)
    time = time[sorted_i]
    timeseries = timeseries[sorted_i]

    return time, timeseries, num_transits


def phase_split_light_curve(time, timeseries, period, t0, duration, n_max_phases, keep_odd_even_order,
                            it_cadences_per_thr, num_cadences_per_h, extend_method, quarter_timestamps=None):
    """ Splits a 1D time series phases using the detected orbital period and epoch. Extracts `n_max_phases` from the
    time series.

    Args:
      time: 1D NumPy array of time values.
      timeseries: 1D NumPy array of time series values.
      period: A positive real scalar; the period to fold over (day).
      t0: The center of the resulting folded vector; this value is mapped to 0.
      duration: positive float; transit duration (day).
      n_max_phases: positive integer; maximum number of phases
      keep_odd_even_order: bool; if True, phases are extracted preserving odd/even alternating structure.
      it_cadences_per_thr: float, threshold for number of in-transit cadences per phase for classifying a phase as
      valid.
      num_cadences_per_h: float, number of cadences sampled per hour in the time series.
      extend_method: str, method used to deal with examples with less than `n_max_phases`. If 'zero_padding', baseline
      phases (i.e., filled with ones) are added; if 'copy_phases', phases are copied starting from the beginning.
      quarter_timestamps: dict, each value is a list. The key is the quarter id and the value is a list with the
      start and end timestamps for the given quarter obtained from the FITS file. If this variable is not `None`, then
      sampling takes into account transits from different quarters.

    Returns:
      phase_split: 2D NumPy array (n_phases x n_points) of phase folded time values in
          [-period / 2, period / 2), where 0 corresponds to t0 in the original
          time array. Values are split by phase.
      timeseries_split: 2D NumPy array. Values are the same as the original input
          array, but split by phases.
      n_obs_phases: int, number of phases extracted.
      odd_even_obs: list, each phase has an indicator for odd/even (0, 1) and not assigned (-1).
    """

    # phase interval for classifying cadences as in-transit cadences
    tmin_it, tmax_it = max(-period / 2, -1.5 * duration), min(period / 2, 1.5 * duration)

    expected_num_it_cadences = int(duration * 24 * num_cadences_per_h)

    # find mid-transit points in [time[0], t0] and [t0, time[-1]]
    k_min, k_max = int(np.ceil((t0 - time[0]) / period)), int(np.ceil((time[-1] - t0) / period))
    epochs = [t0 - k * period for k in range(k_min)]
    epochs += [t0 + k * period for k in range(1, k_max)]

    # estimate odd and even phases that should have been observed
    n_phases = len(epochs)
    odd_even_id_arr = -1 * np.ones(n_phases, dtype='int')
    odd_even_id_arr[::2] = 0
    odd_even_id_arr[1::2] = 1

    # fold time array over the estimated period to create phase array
    phase = util.phase_fold_time(time, period, t0)

    # split time series over phases
    diff_phase = np.diff(phase, prepend=np.nan)
    # get indices between phases based on going from positive to negative phase
    idx_splits = np.where(diff_phase < 0)[0]
    # get indices to split phases based on time
    diff_time = np.diff(time, prepend=np.nan)
    # time between negative and positive phase is longer than one period
    idx_splits_per = np.where(diff_time > period)[0]
    # time between same sign phases is longer than half period
    idx_splits_same_phase_sign = np.where(np.logical_and(diff_time > period / 2, diff_time < period))[0]
    idx_splits_same_phase_sign_valid = [idx for idx in idx_splits_same_phase_sign if phase[idx - 1] * phase[idx] == 1]
    # concatenate indices and sort them
    idx_splits = np.concatenate([idx_splits, idx_splits_per, idx_splits_same_phase_sign_valid]).astype('uint')
    idx_splits.sort()
    # use indices to split time series arrays
    time_split = np.array_split(time, idx_splits)
    phase_split = np.array_split(phase, idx_splits)
    timeseries_split = np.array_split(timeseries, idx_splits)
    n_obs_phases = len(phase_split)  # number of actual observed phases

    # assign observed phases as odd or even
    odd_even_obs = -1 * np.ones(n_obs_phases, dtype='int')  # initialize odd and even observed phases

    epoch_idx = 0
    for phase_i in range(n_obs_phases):  # iterate over observed phases
        for epoch_i in range(epoch_idx, n_phases):  # iterate over expected epochs
            # check if epoch matches the observed phase
            if np.logical_and(np.all(epochs[epoch_i] - period / 2 < time_split[phase_i]),
                              np.all(epochs[epoch_i] + period / 2 > time_split[phase_i])):
                odd_even_obs[phase_i] = odd_even_id_arr[epoch_i]
                epoch_idx = epoch_i + 1  # update the first epoch to be checked as the one after the matched epoch
                break

    # remove phases that were not matched with any epoch; this shouldn't happen often (?) though
    idx_valid_phases = odd_even_obs != -1
    time_split = [time_split[phase_i] for phase_i in range(n_obs_phases) if idx_valid_phases[phase_i]]
    timeseries_split = [timeseries_split[phase_i] for phase_i in range(n_obs_phases) if idx_valid_phases[phase_i]]
    phase_split = [phase_split[phase_i] for phase_i in range(n_obs_phases) if idx_valid_phases[phase_i]]
    odd_even_obs = [odd_even_obs[phase_i] for phase_i in range(n_obs_phases) if idx_valid_phases[phase_i]]
    n_obs_phases = len(time_split)
    if n_obs_phases == 0:
        return None, None, 0, None

    # remove phases that contain less than some number/fraction of in-transit cadences; they are not reliable
    # representations of the potential transit
    num_it_cadences_phases = np.array([np.sum(np.logical_and(phase_split[phase_i] > tmin_it,
                                                             phase_split[phase_i] < tmax_it))
                                       for phase_i in range(n_obs_phases)])
    # idx_valid_phases = num_it_cadences_phases > min_num_it_cadences
    idx_valid_phases = num_it_cadences_phases / expected_num_it_cadences > it_cadences_per_thr
    time_split = [time_split[phase_i] for phase_i in range(n_obs_phases) if idx_valid_phases[phase_i]]
    timeseries_split = [timeseries_split[phase_i] for phase_i in range(n_obs_phases) if idx_valid_phases[phase_i]]
    phase_split = [phase_split[phase_i] for phase_i in range(n_obs_phases) if idx_valid_phases[phase_i]]
    odd_even_obs = [odd_even_obs[phase_i] for phase_i in range(n_obs_phases) if idx_valid_phases[phase_i]]
    n_obs_phases = len(time_split)
    if n_obs_phases == 0:
        return None, None, 0, None

    # have alternating odd and even phases
    if keep_odd_even_order:
        keep_idxs = [0]
        curr_phase_type = odd_even_obs[0]
        for phase_i in range(1, n_obs_phases):
            if odd_even_obs[phase_i] != curr_phase_type:
                keep_idxs.append(phase_i)
                curr_phase_type = odd_even_obs[phase_i]
        time_split = [time_split[keep_idx] for keep_idx in keep_idxs]
        timeseries_split = [timeseries_split[keep_idx] for keep_idx in keep_idxs]
        phase_split = [phase_split[keep_idx] for keep_idx in keep_idxs]
        odd_even_obs = [odd_even_obs[keep_idx] for keep_idx in keep_idxs]
        n_obs_phases = len(time_split)
        if n_obs_phases == 0:
            return None, None, 0, None

    # choosing a random consecutive set of phases if there are more than the requested maximum number of phases
    if quarter_timestamps is None:
            if n_obs_phases > n_max_phases:

                chosen_phases_st = np.random.randint(n_obs_phases - n_max_phases)
                chosen_phases = np.arange(chosen_phases_st, chosen_phases_st + n_max_phases)

                time_split = [time_split[chosen_phase] for chosen_phase in chosen_phases]
                timeseries_split = [timeseries_split[chosen_phase] for chosen_phase in chosen_phases]
                phase_split = [phase_split[chosen_phase] for chosen_phase in chosen_phases]
                odd_even_obs = [odd_even_obs[chosen_phase] for chosen_phase in chosen_phases]

            elif n_obs_phases < n_max_phases:

                if extend_method == 'zero_padding':  # new phases are assigned baseline value
                    n_miss_phases = n_max_phases - n_obs_phases

                    # zero phases have median number of cadences of observed phases
                    med_n_cadences = np.median([time_split[phase_i] for phase_i in range(n_obs_phases)])
                    # zero phases have median value of observed phases
                    med_timeseries_val = np.median([timeseries_split[phase_i] for phase_i in range(n_obs_phases)])
                    # linear phase space
                    phase_split_miss = np.linspace(-period / 2, period / 2, med_n_cadences, endpoint=True)

                    phase_split += [phase_split_miss] * n_miss_phases
                    timeseries_split += [med_timeseries_val * np.ones(med_n_cadences, dtype='float')] * n_miss_phases
                    time_split += [np.nan * np.ones(med_n_cadences, dtype='float')] * n_miss_phases
                    odd_even_obs += [np.nan] * n_miss_phases

                elif extend_method == 'copy_phases':  # phases are copied from the start

                    n_full_group_phases = n_max_phases // n_obs_phases
                    n_partial_group_phases = n_max_phases % n_obs_phases
                    # phase_split = np.concatenate((np.tile(phase_split, n_full_group_phases),
                    #                               phase_split[:n_partial_group_phases]), axis=1)
                    phase_split = phase_split * n_full_group_phases + phase_split[:n_partial_group_phases]

                    time_split = list(time_split) * n_full_group_phases + \
                                 list(time_split[:n_partial_group_phases])
                    timeseries_split = list(timeseries_split) * n_full_group_phases + \
                                       list(timeseries_split[:n_partial_group_phases])
                    odd_even_obs = odd_even_obs * n_full_group_phases + odd_even_obs[:n_partial_group_phases]

                else:
                    raise ValueError(f'Extend method for phases `{extend_method}` not implemented.')

    else:
        # assign each phase to a season
        n_seasons = 4
        n_phases_per_season = n_max_phases // n_seasons
        # n_phases_left = n_max_phases % n_seasons
        t0_time_split = [time_phase[0] for time_phase in time_split]
        season_phases = np.nan * np.ones(len(time_split))
        for t_i, t in enumerate(t0_time_split):
            for q, q_t in quarter_timestamps.items():
                if t >= q_t[0] and t <= q_t[1]:
                    season_phases[t_i] = q % n_seasons
                    break

        chosen_idx_phases_all_seasons = []
        not_chosen_idxs_phases_all_seasons = []
        # if extend_method == 'zero_padding':
        #     phase_split_zero_padding = []
        for season_i in range(n_seasons):
            idxs_phases_in_season = np.where(season_phases == season_i)[0]  # get phases for this season
            n_phases_in_season = len(idxs_phases_in_season)  # number of seasons available for this season
            # more phases available than the requested number of phases per season
            if n_phases_in_season >= n_phases_per_season:
                # choose set of consecutive phases with random starting point
                chosen_phases_st = np.random.randint(n_phases_in_season - n_phases_per_season + 1)
                chosen_idx_phases_season = \
                    idxs_phases_in_season[chosen_phases_st: chosen_phases_st + n_phases_per_season]
                # bookkeeping for phases not chosen
                not_chosen_idxs_phases_all_seasons.append(np.setdiff1d(idxs_phases_in_season, chosen_idx_phases_season))
            else:  # fewer phases available than the requested number of phases per season
                chosen_idx_phases_season = idxs_phases_in_season
                # not_chosen_idxs_phases_all_seasons.append([])  # no phases left unchosen

            chosen_idx_phases_all_seasons.append(chosen_idx_phases_season)  # add chosen phases indices for this season

        chosen_idx_phases_all_seasons = np.concatenate(chosen_idx_phases_all_seasons)
        if len(not_chosen_idxs_phases_all_seasons) != 0:
            not_chosen_idxs_phases_all_seasons = np.concatenate(not_chosen_idxs_phases_all_seasons)

        # count number of chosen phases per season
        # n_chosen_phases_per_season = [len(season_phases) for season_phases in chosen_idx_phases_all_seasons]
        # total number of chosen phases
        n_chosen_phases_all_seasons = len(chosen_idx_phases_all_seasons)

        # # same for unchosen phases
        # n_notchosen_phases_per_season = [len(season_phases) for season_phases in not_chosen_idxs_phases_all_seasons]
        # n_notchosen_phases_all_seasons = len(not_chosen_idxs_phases_all_seasons)

        n_phases_left = n_max_phases - n_chosen_phases_all_seasons

        # add unchosen phases
        # np.random.shuffle(not_chosen_idxs_phases_all_seasons)  # shuffle phases indices
        if len(not_chosen_idxs_phases_all_seasons) != 0:
            chosen_idx_phases_all_seasons = np.concatenate([chosen_idx_phases_all_seasons,
                                                            not_chosen_idxs_phases_all_seasons[:n_phases_left]],
                                                           dtype='int')
        # not_chosen_idxs_phases_all_seasons = not_chosen_idxs_phases_all_seasons[n_phases_left:]

        n_chosen_phases_all_seasons = len(chosen_idx_phases_all_seasons)
        n_phases_left = n_max_phases - n_chosen_phases_all_seasons
        if n_phases_left > 0:
            if extend_method == 'copy_phases':
                n_full_group_phases = n_max_phases // n_chosen_phases_all_seasons
                n_partial_group_phases = n_max_phases % n_chosen_phases_all_seasons
                chosen_idx_phases_all_seasons = \
                    np.concatenate([np.tile(chosen_idx_phases_all_seasons, n_full_group_phases),
                                    chosen_idx_phases_all_seasons[:n_partial_group_phases]],
                                   axis=0)
            elif extend_method == 'zero_padding':
                # zero phases are flagged by index -1
                chosen_idx_phases_all_seasons = \
                    np.concatenate((chosen_idx_phases_all_seasons, -1 * np.ones(n_phases_left, dtype='int')), axis=1)
                # zero phases have median number of cadences of observed phases
                med_timeseries_val = np.median([time_split[phase_i] for phase_i in chosen_idx_phases_all_seasons])
                # zero phases have median value of observed phases
                med_n_cadences = np.median([time_split[phase_i] for phase_i in chosen_idx_phases_all_seasons])

                # linear phase space
                phase_split_zeropadding = np.linspace(-period / 2, period / 2, med_n_cadences, endpoint=True)
                time_split_zeroppading = np.nan * np.ones(med_n_cadences, dtype='float')
                timeseries_split_zeropadding = med_timeseries_val * np.ones(med_n_cadences, dtype='float')
            else:
                raise ValueError(f'Extend method for phases `{extend_method}` not implemented.')

        time_split_n, timeseries_split_n, phase_split_n, odd_even_obs_n = [], [], [], []
        for chosen_phase in chosen_idx_phases_all_seasons:
            if chosen_phase == -1:
                phase_split_n.append(phase_split_zeropadding)
                time_split_n.append(time_split_zeroppading)
                timeseries_split_n.append(timeseries_split_zeropadding)
                odd_even_obs_n.append(np.nan)
            else:
                time_split_n.append(time_split[chosen_phase])
                timeseries_split_n.append(timeseries_split[chosen_phase])
                phase_split_n.append(phase_split[chosen_phase])
                odd_even_obs_n.append(odd_even_obs[chosen_phase])

        time_split, timeseries_split, phase_split, odd_even_obs = time_split_n, timeseries_split_n, phase_split_n, \
            odd_even_obs_n

    n_obs_phases = len(time_split)

    if n_obs_phases == 0:
        return None, None, 0, None

    return phase_split, timeseries_split, n_obs_phases, odd_even_obs


def generate_view(time, flux, num_bins, bin_width, t_min, t_max, tce, centering=True, normalize=True, centroid=False,
                  **kwargs):
    """ Generates a view of a phase-folded light curve using a median filter.

    Args:
      time: 1D array of time values, sorted in ascending order.
      flux: 1D array of flux/centroid values.
      num_bins: The number of intervals to divide the time axis into.
      bin_width: The width of each bin on the time axis.
      t_min: The inclusive leftmost value to consider on the time axis.
      t_max: The exclusive rightmost value to consider on the time axis.
      tce: Pandas Series, TCE parameters
      centering: bool, whether to center the view by subtracting the median
      normalize: Whether to perform normalization
      centroid: bool, if True considers these view a centroid time series

    Returns:
      view: 1D NumPy array of size num_bins containing the median values of uniformly spaced bins on the phase-folded
      time axis.
      time_bins: NumPy array, phase for the binned time series
      view_var: 1D NumPy array of size num_bins containing the variability values of uniformly spaced bins on the
      phase-folded time axis.
      inds_nan: dict, boolean indices for the in-transit and out-of-transit imputed bins
      bin_counts, list, number of points in each bin
    """

    # binning using median
    view, time_bins, view_var, bin_counts = median_filter.median_filter(time, flux, num_bins, bin_width, t_min, t_max)

    # impute missing bin values
    view, view_var, inds_nan = impute_binned_ts(time_bins, view, tce['tce_period'], tce['tce_duration'], view_var)

    if centering:
        view -= np.median(view)

    if normalize:
        view = normalize_view(view, val=None, centroid=centroid, **kwargs)

    return view, time_bins, view_var, inds_nan, bin_counts


def generate_view_momentum_dump(time, momentum_dumps, num_bins, bin_width, t_min, t_max):
    """ Generates a view of a phase-folded momentum dump array curve using a max filter.

        Args:
          time: 1D array of time values, sorted in ascending order.
          momentum_dumps: 1D uint array of momentum dump flags.
          num_bins: The number of intervals to divide the time axis into.
          bin_width: The width of each bin on the time axis.
          t_min: The inclusive leftmost value to consider on the time axis.
          t_max: The exclusive rightmost value to consider on the time axis.

        Returns:
          1D NumPy array of size num_bins containing the mean momentum dump flag values uniformly spaced bins on the
          phase-folded time axis.
          1D NumPy array of size num_bins containing the mad std momentum dump flag values uniformly spaced bins on the
          phase-folded time axis.
          1D NumPy array of size num_bins containing the number of points per bin for the phase folded momentum dump
          flag time series uniformly spaced bins on the phase-folded time axis.
        """

    # binning using max function
    view, time_bins, view_var, _ = median_filter.median_filter(time, momentum_dumps,
                                                               num_bins,
                                                               bin_width,
                                                               t_min, t_max,
                                                               bin_fn=np.mean,
                                                               bin_var_fn=np.std)

    # impute missing bin values with zero; although there shouldn't be any NaNs by construction
    inds_nan = np.isnan(view)
    view[inds_nan] = 0
    view_var[inds_nan] = 0

    return view, view_var, time_bins


def normalize_view(view, val=None, centroid=False, **kwargs):
    """ Normalize the phase-folded time series.

    :param view: array, phase-folded time series
    :param val: float, value used to normalize the time series
    :param centroid: bool, True for centroid time series
    :param kwargs: dict, extra keyword parameters
    :return:
        array, normalized phase-folded time series
    """

    # for the centroid time series
    # range [new_min_val, 1], assuming max is positive, which should be since we are removing the median from a
    # non-negative time series
    # for the flux time series
    # range [-1, new_max_val], assuming min is negative, if not [1, new_max_val]
    if val is None:
        val = np.abs(np.max(view)) if centroid else np.abs(np.min(view))

    if val == 0:
        print(f'Dividing view by 0. Returning the non-normalized view {kwargs["report"]["view"]}.')
        report_exclusion(kwargs['report']['config'], kwargs['report']['tce'],
                         f'Dividing view by 0. Returning the non-normalized view {kwargs["report"]["view"]}.')

        return view

    return view / val


def centering_and_normalization(view, val_centr, val_norm, **kwargs):
    """ Center and normalize a 1D time series.

    :param view: array, 1D time series
    :param val_centr: float, value used to center the time series
    :param val_norm: float, value used normalize the time series
    :param kwargs: dict, extra keyword parameters
    :return:
        array, centered and normalized time series
    """

    if val_norm == 0:
        print(f'Dividing view by 0. Returning the non-normalized view {kwargs["report"]["view"]}.')
        report_exclusion(kwargs['report']['config'], kwargs['report']['tce'],
                         f'Dividing view by 0. Returning the non-normalized view {kwargs["report"]["view"]}.')

        return view - val_centr

    return (view - val_centr) / val_norm


def global_view(time, flux, period, tce, num_bins=2001, bin_width_factor=1/2001, centroid=False, normalize=True,
                centering=True, **kwargs):
    """Generates a 'global view' of a phase folded light curve.

    See Section 3.3 of Shallue & Vanderburg, 2018, The Astronomical Journal.
    http://iopscience.iop.org/article/10.3847/1538-3881/aa9e09/meta

    Args:
      time: 1D array of time values, sorted in ascending order.
      flux: 1D array of flux values.
      period: The period of the event (in days).
      tce: Pandas Series, TCE parameters
      num_bins: The number of intervals to divide the time axis into.
      bin_width_factor: Width of the bins, as a fraction of period.
      centering: bool, whether to center the view by subtracting the median
      normalize: Whether to perform normalization
      centroid: bool, if True considers these view a centroid time series


    Returns:
      view: 1D NumPy array of size num_bins containing the median values of uniformly spaced bins on the phase-folded
      time axis.
      time_bins: NumPy array, phase for the binned time series
      view_var: 1D NumPy array of size num_bins containing the variability values of uniformly spaced bins on the
      phase-folded time axis.
      inds_nan: dict, boolean indices for the in-transit and out-of-transit imputed bins
      bin_counts, list, number of points in each bin
    """

    view, time_bins, view_var, inds_nan, bin_counts = generate_view(
        time,
        flux,
        num_bins=num_bins,
        bin_width=max(period * bin_width_factor, kwargs['tce_duration'] * 0.16),
        t_min=-period / 2,
        t_max=period / 2,
        tce=tce,
        centroid=centroid,
        normalize=normalize,
        centering=centering,
        **kwargs)

    return view, time_bins, view_var, inds_nan, bin_counts


def local_view(time,
               flux,
               period,
               duration,
               tce,
               num_bins=201,
               bin_width_factor=0.16,
               num_durations=4,
               centroid=False,
               normalize=True,
               centering=True,
               **kwargs):
    """Generates a 'local view' of a phase folded light curve.

    See Section 3.3 of Shallue & Vanderburg, 2018, The Astronomical Journal.
    http://iopscience.iop.org/article/10.3847/1538-3881/aa9e09/meta

    Args:
      time: 1D array of time values, sorted in ascending order.
      flux: 1D array of flux values.
      period: The period of the event (in days).
      duration: The duration of the event (in days).
      tce: Pandas Series, TCE parameters
      num_bins: The number of intervals to divide the time axis into.
      bin_width_factor: Width of the bins, as a fraction of duration.
      num_durations: The number of durations to consider on either side of 0 (the event is assumed to be centered at 0).
      centering: bool, whether to center the view by subtracting the median
      normalize: Whether to perform normalization
      centroid: bool, if True considers these view a centroid time series


    Returns:
       view: 1D NumPy array of size num_bins containing the median values of uniformly spaced bins on the phase-folded
      time axis.
      time_bins: NumPy array, phase for the binned time series
      view_var: 1D NumPy array of size num_bins containing the variability values of uniformly spaced bins on the
      phase-folded time axis.
      inds_nan: dict, boolean indices for the in-transit and out-of-transit imputed bins
      bin_counts, list, number of points in each bin
    """

    t_min = max(-period / 2, -duration * num_durations)
    t_max = min(period / 2, duration * num_durations)

    if t_min > time[-1] or t_max < time[0]:
        report_exclusion(kwargs['report']['config'],
                         tce,
                         f'No in-transit cadences in view {kwargs["report"]["view"]}.')
        time_bins = np.linspace(t_min, t_max, num_bins, endpoint=True)
        med = np.median(flux)
        std_rob_estm = mad_std(flux)  # robust std estimator of the time series
        view_var = std_rob_estm * np.ones(num_bins, dtype='float')
        view = med + np.random.normal(0, std_rob_estm, num_bins)
        inds_nan = {'oot': False * np.ones(num_bins, dtype='bool'), 'it': False * np.ones(num_bins, dtype='bool')}
        bin_counts = np.ones(num_bins, dtype='float')
    else:
        view, time_bins, view_var, inds_nan, bin_counts = generate_view(
            time,
            flux,
            tce=tce,
            num_bins=num_bins,
            bin_width=duration * bin_width_factor,
            t_min=t_min,
            t_max=t_max,
            centroid=centroid,
            normalize=normalize,
            centering=centering,
            **kwargs)

    return view, time_bins, view_var, inds_nan, bin_counts