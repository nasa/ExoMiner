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
import tensorflow as tf
import multiprocessing
from functools import partial
from copy import deepcopy
import logging
import os
import psutil
import sys
import traceback
import yaml

# local
from transit_detection.utils_flux_processing import (
    extract_flux_windows_for_tce,
    build_transit_mask_for_lightcurve,
    # plot_detrended_flux_time_series_sg,
    plot_interpolated_flux_mask,
    plot_flux_diagnostics_sg,
    interpolate_missing_flux,
)
from transit_detection.utils_difference_img_processing import (
    extract_diff_img_data_from_window,
)
from transit_detection.utils_build_dataset import (
    serialize_set_examples_for_tce,
    write_data_to_auxiliary_tbl,
)
from transit_detection.utils_local_fits_processing import (
    # search_and_read_tess_lcfs_with_astropy_table,
    # search_and_read_tess_tpfs_with_lk,
    search_and_read_lcfs_and_tpfs,
)
from src_preprocessing.lc_preprocessing.detrend_timeseries import (
    detrend_flux_using_sg_filter,
)
from transit_detection.utils_chunk_dataset import build_chunk_mask


def process_target(
    chunked_target_data,
    chunk_num,
    num_chunks,
    lcf_dir,
    tpf_dir,
    tce_example_disps,
    n_durations_window,
    detrend_lc_flag,
    ex_it_mask_n_durations_window,
    sg_it_mask_n_durations_window,
    weak_sec_mask_maxmes_thr,
    frac_valid_cadences_in_window_thr,
    frac_valid_cadences_it_thr,
    buffer_time,
    # gap_width,
    resampled_num_points,
    rnd_seed,
    size_img,
    f_size,
    center_target,
    plot_dir,
    log_dir,
    data_dir,
):

    try:
        chunk_title = f"{str(chunk_num).zfill(4)}-{str(num_chunks).zfill(4)}"
        print(f"starting processing for chunk: {chunk_title}")

        # set logger for each process
        log_name = f"process_{chunk_title}"
        log_fp = Path(log_dir) / f"{log_name}.log"

        logger = logging.getLogger(log_name)
        logger.setLevel(logging.INFO)

        file_handler = logging.FileHandler(log_fp)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )

        if logger.hasHandlers():
            logger.handlers.clear()  # handle lingering handlers
        logger.addHandler(file_handler)

        logger.info(f"Beginning dataset build for chunk {chunk_num}.")
        logger.info(
            f"BEGIN_{log_name} using {psutil.Process().memory_info().rss / 1024 ** 2:.2f} MB mem"
        )

        logger.info(
            f"With chunked_target_data {type(chunked_target_data)} of len: {len(chunked_target_data)}"
        )

        rng = np.random.default_rng(seed=rnd_seed)

        # store example data per tce in chunk
        chunk_tce_example_data = []

        for target, t_tce_data in chunked_target_data:
            # process target
            logger.info(f"Starting processing for target: {target}")
            logger.info(
                f"{target} contains {len(t_tce_data['sector_run'].unique())} unique sector_runs: {t_tce_data['sector_run'].unique()}"
            )
            logger.info(
                f"{target} contains {len(t_tce_data['tce_uid'].unique())} tces: {t_tce_data['tce_uid'].unique()}"
            )
            logger.info(f"{t_tce_data[['tce_uid', 'disposition']]}")

            target_tce_info = []
            target_tce_example_data = {}

            for tce_i, tce_data in t_tce_data.iterrows():
                # add tce ephemerides info
                target_tce_info.append(
                    {
                        "disposition": tce_data["disposition"],
                        "tce_time0bk": tce_data["tce_time0bk"],
                        "tce_period": tce_data["tce_period"],
                        "tce_duration": tce_data["tce_duration"],
                        "tce_maxmes": tce_data["tce_maxmes"],
                        "tce_maxmesd": tce_data["tce_maxmesd"],
                    }
                )
                logger.info(
                    f"Added tce_data for {tce_data['tce_uid']} to target_tce_info arr."
                )

                # add tce example data placeholder
                target_tce_example_data[tce_data["tce_uid"]] = {
                    "tce_uid": tce_data["tce_uid"],
                    "tce_info": tce_data,
                    "sectors": [],
                    "disposition": tce_data["disposition"],
                }
                logger.info(
                    f"Added tce example data for {tce_data['tce_uid']} to target_tce_example_data map."
                )

            for sector_run, t_sr_tce_data in t_tce_data.groupby("sector_run"):
                logger.info(f"Starting processing for sector_run: {sector_run}")

                try:
                    target_sector_run_tce_info = []
                    target_sector_run_tce_example_data = {}

                    for tce_i, tce_data in t_sr_tce_data.iterrows():
                        # add tce ephemerides info
                        target_sector_run_tce_info.append(
                            {
                                "disposition": tce_data["disposition"],
                                "tce_time0bk": tce_data["tce_time0bk"],
                                "tce_period": tce_data["tce_period"],
                                "tce_duration": tce_data["tce_duration"],
                                "tce_maxmes": tce_data["tce_maxmes"],
                                "tce_maxmesd": tce_data["tce_maxmesd"],
                            }
                        )
                        logger.info(
                            f"Adding tce_data for {tce_data['tce_uid']} to target_sector_run_tce_info"
                        )

                        # add tce example data placeholder
                        target_sector_run_tce_example_data[tce_data["tce_uid"]] = {
                            "tce_uid": tce_data["tce_uid"],
                            "tce_info": tce_data,
                            "sectors": [],
                            "disposition": tce_data["disposition"],
                        }
                        logger.info(
                            f"Adding tce example data for {tce_data['tce_uid']} to target_sector_run_tce_example_data"
                        )

                    if plot_dir:
                        t_sr_plot_dir = plot_dir / f"t_{target}_sr_{sector_run}"
                        t_sr_plot_dir.mkdir(exist_ok=True, parents=True)
                    else:
                        t_sr_plot_dir = None

                    # get individual sector runs (needed for multisector runs)
                    if "-" in sector_run:
                        start_sector, end_sector = [
                            int(sector) for sector in sector_run.split("-")
                        ]
                        sector_run_arr = [
                            sector for sector in range(start_sector, end_sector + 1)
                        ]
                    else:
                        sector_run_arr = [int(sector_run)]

                    # find light curve data for target, sector_run pair
                    found_sectors, found_lcfs, found_tpfs = (
                        search_and_read_lcfs_and_tpfs(
                            target=target,
                            sectors=sector_run_arr,
                            lcf_dir=lcf_dir,
                            tpf_dir=tpf_dir,
                            # quality_bitmask="default",
                        )
                    )

                    if len(found_sectors) == 0:
                        logger.warning(
                            f"Skipping processing for t, sr: ({target}, {sector_run}). Found no lcf/tpf data "
                        )
                        continue

                    logger.info(
                        f"Found {len(found_sectors)} sectors with light curve data for t, sr: ({target}, {sector_run})."
                    )

                    # for given sector in sector_run
                    for sector, lcf, tpf in zip(found_sectors, found_lcfs, found_tpfs):
                        logger.info(
                            f"Beginning processing for s {sector} in sr {sector_run}"
                        )

                        if t_sr_plot_dir:
                            t_sr_s_plot_dir = t_sr_plot_dir / f"S{sector}"
                            t_sr_s_plot_dir.mkdir(exist_ok=True, parents=True)
                        else:
                            t_sr_s_plot_dir = None

                        # locally read lcfs w/ astropy table loading, have all timestamps included.
                        time, flux = lcf.time.value, lcf.flux.value

                        # masks for interpolated cadences
                        (
                            flux,
                            cadence_mask,
                            flux_quality,
                        ) = interpolate_missing_flux(time, flux)

                        # purely informational
                        true_it_mask = build_transit_mask_for_lightcurve(
                            time=time,
                            tce_list=[
                                tce_info
                                for tce_info in target_sector_run_tce_info
                                if tce_info["disposition"]
                                in ["EB", "CP", "KP", "NPC", "NEB"]
                            ],
                            n_durations_window=1,  # one td of true transits
                            maxmes_threshold=np.inf,  # no secondary transits
                        )

                        # used for ensuring neg disposition it_windows do not overlap w/ pos disposition it_windows
                        # because all neg disp it_windows receive a 0 label
                        ex_pos_disp_it_mask = build_transit_mask_for_lightcurve(
                            time=time,
                            tce_list=[
                                tce_info
                                for tce_info in target_tce_info
                                if tce_info["disposition"] in ["EB", "CP", "KP"]
                            ],
                            n_durations_window=ex_it_mask_n_durations_window,
                            maxmes_threshold=weak_sec_mask_maxmes_thr,
                        )

                        # used for ensuring any disp oot_windows do not overlap w/ any disp it_windows
                        ex_all_disp_it_mask = build_transit_mask_for_lightcurve(
                            time=time,
                            tce_list=target_tce_info,
                            n_durations_window=ex_it_mask_n_durations_window,
                            maxmes_threshold=weak_sec_mask_maxmes_thr,
                        )

                        sg_all_disp_it_mask = np.ones_like(time)

                        if detrend_lc_flag:
                            sg_all_disp_it_mask = build_transit_mask_for_lightcurve(
                                time=time,
                                tce_list=target_sector_run_tce_info,
                                n_durations_window=sg_it_mask_n_durations_window,
                                maxmes_threshold=weak_sec_mask_maxmes_thr,
                            )

                        # used for detrending
                        lcf = lk.LightCurve({"time": time, "flux": flux})
                        time, flux, trend, _ = detrend_flux_using_sg_filter(
                            lc=lcf,
                            mask_in_transit=sg_all_disp_it_mask,
                            win_len=int(1.2 * 24 * 30),  # 1.2 days,
                            sigma=5,
                            max_poly_order=6,
                            penalty_weight=1,
                            break_tolerance=5,
                        )

                        # temp_plot_dir = (
                        #     Path(data_dir)
                        #     / "plots"
                        #     / str(target)
                        #     / str(sector_run)
                        #     / str(sector)
                        # )
                        # temp_plot_dir.mkdir(parents=True, exist_ok=True)
                        # # plot_detrended_flux_time_series_sg(
                        # #     time=lcf.time.value,
                        # #     flux=lcf.flux.value,
                        # #     detrend_time=time,
                        # #     detrend_flux=flux,
                        # #     trend=trend,
                        # #     sector=sector,
                        # #     plot_dir=temp_plot_dir,
                        # # )
                        # try:
                        #     plot_flux_diagnostics_sg(
                        #         time=lcf.time.value,
                        #         flux=lcf.flux.value,
                        #         detrend_time=time,
                        #         detrended_flux=flux,
                        #         detrend_mask=sg_all_disp_it_mask,
                        #         quality_mask=flux_quality,
                        #         trend=trend,
                        #         sector=sector,
                        #         plot_dir=temp_plot_dir,
                        #     )
                        #     plot_interpolated_flux_mask(
                        #         time=lcf.time.value,
                        #         flux=lcf.flux.value,
                        #         cadence_mask=cadence_mask,
                        #         flux_quality=flux_quality,
                        #         sector=sector,
                        #         plot_dir=temp_plot_dir,
                        #     )
                        # except Exception as e:
                        #     logger.info(f"PLOT_ERROR: {e}")

                        for tce_i, tce_data in t_sr_tce_data.iterrows():

                            sector_tce_example_data = {
                                "sector": sector,
                                "transit_examples": {
                                    "flux": [],
                                    "flux_quality": [],
                                    "t": [],
                                    "true_it_mask": [],
                                    "it_img": [],
                                    "oot_img": [],
                                    "diff_img": [],
                                    "snr_img": [],
                                    "target_img": [],
                                    "target_pos": [],
                                },
                                "not_transit_examples": {
                                    "flux": [],
                                    "flux_quality": [],
                                    "t": [],
                                    "true_it_mask": [],
                                    "it_img": [],
                                    "oot_img": [],
                                    "diff_img": [],
                                    "snr_img": [],
                                    "target_img": [],
                                    "target_pos": [],
                                },
                            }

                            # get TCE unique id
                            tce_uid = tce_data["tce_uid"]
                            target_id = tce_data["target_id"]
                            sector_run = tce_data["sector_run"]

                            # ephemerides for TCE
                            tce_time0bk = tce_data["tce_time0bk"]
                            tce_period = tce_data["tce_period"]
                            tce_duration = tce_data["tce_duration"]

                            disposition = tce_data["disposition"]

                            # create directory for TCE plots
                            if t_sr_s_plot_dir:
                                tce_ex_plot_dir = t_sr_s_plot_dir / f"tce_{tce_uid}"
                                tce_ex_plot_dir.mkdir(exist_ok=True, parents=True)
                            else:
                                tce_ex_plot_dir = None

                            # extract flux time series windows
                            try:

                                logger.info(
                                    f"Beginning LC window extraction in s {sector} for tce: {tce_uid}"
                                )

                                if disposition not in tce_example_disps:
                                    logger.info(
                                        f"Skipping LC window extraction for {tce_uid} due to disposition: {disposition}"
                                    )
                                    continue

                                (
                                    resampled_flux_it_windows_arr,
                                    resampled_flux_oot_windows_arr,
                                    resampled_flux_quality_it_windows_arr,
                                    resampled_flux_quality_oot_windows_arr,
                                    resampled_true_it_mask_it_windows_arr,
                                    resampled_true_it_mask_oot_windows_arr,
                                    midtransit_points_windows_arr,
                                    midoot_points_windows_arr,
                                ) = extract_flux_windows_for_tce(
                                    time,
                                    flux,
                                    flux_quality,
                                    true_it_mask,
                                    ex_pos_disp_it_mask,
                                    ex_all_disp_it_mask,
                                    cadence_mask,
                                    tce_uid,
                                    tce_time0bk,
                                    tce_duration,
                                    tce_period,
                                    disposition,
                                    n_durations_window,
                                    frac_valid_cadences_in_window_thr,
                                    frac_valid_cadences_it_thr,
                                    resampled_num_points,
                                    rng,
                                    tce_ex_plot_dir,
                                    logger=logger,
                                )
                                logger.info(
                                    f"Finished extracting flux windows in s {sector} for tce: {tce_uid} "
                                )
                            except ValueError as e:
                                logger.warning(
                                    f"Could not extract flux windows in s: {sector} for tce: {tce_uid} - {e} "
                                )
                                continue
                            except Exception as e:
                                logger.error(
                                    f"Unexpected exception of type {type(e)} while extracting flux windows in s {sector} for tce: {tce_uid} - {e}"
                                )
                                continue

                            # if tce_ex_plot_dir:
                            #     tce_diff_img_data_plot_dir = (
                            #         tce_ex_plot_dir / "diff_img_data"
                            #     )
                            #     plot_dir_tce_diff_img_data.mkdir(exist_ok=True, parents=True)
                            # else:
                            #     plot_dir_tce_diff_img_data = None

                            # if plot_dir_tce_diff_img_data:
                            #     plot_dir_tce_transit_diff_img_data = (
                            #         plot_dir_tce_diff_img_data / "transit_imgs"
                            #     )
                            #     plot_dir_tce_transit_diff_img_data.mkdir(
                            #         exist_ok=True, parents=True
                            #     )
                            # else:
                            #     plot_dir_tce_transit_diff_img_data = None

                            # compute transit difference image data
                            it_diff_img_data = {
                                feature_name: []
                                for feature_name in [
                                    "diff_img",
                                    "oot_img",
                                    "snr_img",
                                    "target_img",
                                    "target_pos",
                                ]
                            }

                            excl_idxs_it_ts = []
                            for window_i, midtransit_point_window in enumerate(
                                midtransit_points_windows_arr
                            ):  # iterate on midtransit points in window
                                try:
                                    (
                                        diff_img_processed,
                                        oot_img_processed,
                                        snr_img_processed,
                                        target_img,
                                        target_pos_col,
                                        target_pos_row,
                                    ) = extract_diff_img_data_from_window(
                                        tpf.path,
                                        midtransit_point_window,
                                        tce_duration,
                                        buffer_time,
                                        tce_data["ra"],
                                        tce_data["dec"],
                                        size_img,
                                        f_size,
                                        sector,
                                        center_target,
                                        tce_uid,
                                        plot_dir=None,
                                    )
                                except ValueError as e:
                                    logger.warning(
                                        f"Skipping {tce_uid} it_window for time {midtransit_point_window}, no difference image data computed - {e}"
                                    )
                                    excl_idxs_it_ts.append(window_i)
                                    continue
                                except Exception as e:
                                    logger.error(
                                        f"Skipping {tce_uid} it_window for time {midtransit_point_window} due to unexpected exception of type: {type(e)} -  {e}"
                                    )
                                    excl_idxs_it_ts.append(window_i)
                                    continue

                                it_diff_img_data["diff_img"].append(diff_img_processed)
                                it_diff_img_data["oot_img"].append(oot_img_processed)
                                it_diff_img_data["snr_img"].append(snr_img_processed)
                                it_diff_img_data["target_img"].append(target_img)
                                it_diff_img_data["target_pos"].append(
                                    np.array([target_pos_col, target_pos_row])
                                )
                            # if plot_dir_tce_diff_img_data:
                            #     plot_dir_tce_oottransit_diff_img_data = (
                            #         plot_dir_tce_diff_img_data / "oot_transit_imgs"
                            #     )
                            #     plot_dir_tce_oottransit_diff_img_data.mkdir(
                            #         exist_ok=True, parents=True
                            #     )
                            # else:
                            #     plot_dir_tce_oottransit_diff_img_data = None

                            # compute oot transit difference image data
                            oot_diff_img_data = {
                                feature_name: []
                                for feature_name in [
                                    "diff_img",
                                    "oot_img",
                                    "snr_img",
                                    "target_img",
                                    "target_pos",
                                ]
                            }

                            excl_idxs_oot_ts = []
                            for window_i, midoot_point_window in enumerate(
                                midoot_points_windows_arr
                            ):  # iterate on oot points in window
                                try:
                                    (
                                        diff_img_processed,
                                        oot_img_processed,
                                        snr_img_processed,
                                        target_img,
                                        target_pos_col,
                                        target_pos_row,
                                    ) = extract_diff_img_data_from_window(
                                        tpf.path,
                                        midoot_point_window,
                                        tce_duration,
                                        buffer_time,
                                        tce_data["ra"],
                                        tce_data["dec"],
                                        size_img,
                                        f_size,
                                        sector,
                                        center_target,
                                        tce_uid,
                                        plot_dir=None,
                                    )
                                except ValueError as e:
                                    logger.warning(
                                        f"Skipping {tce_uid} oot_window for time {midoot_point_window}, no difference image data computed - {e}"
                                    )
                                    excl_idxs_oot_ts.append(window_i)
                                    continue
                                except Exception as e:
                                    logger.error(
                                        f"Skipping {tce_uid} oot_window for time {midoot_point_window} due to unexpected exception of type: {type(e)} - {e}"
                                    )
                                    excl_idxs_oot_ts.append(window_i)
                                    continue

                                oot_diff_img_data["diff_img"].append(diff_img_processed)
                                oot_diff_img_data["oot_img"].append(oot_img_processed)
                                oot_diff_img_data["snr_img"].append(snr_img_processed)
                                oot_diff_img_data["target_img"].append(target_img)
                                oot_diff_img_data["target_pos"].append(
                                    np.array([target_pos_col, target_pos_row])
                                )

                            logger.info(
                                f"Excluding {len(excl_idxs_it_ts)}/{len(midtransit_points_windows_arr)} it_window examples w/ no diff_img data."
                            )
                            logger.info(
                                f"Excluding {len(excl_idxs_oot_ts)}/{len(midoot_points_windows_arr)} oot_window examples w/ no diff_img data."
                            )

                            # exclude examples for timestamps with no difference image data
                            # it windows
                            resampled_flux_it_windows_arr = np.delete(
                                np.array(resampled_flux_it_windows_arr),
                                excl_idxs_it_ts,
                                axis=0,
                            )
                            resampled_flux_quality_it_windows_arr = np.delete(
                                np.array(resampled_flux_quality_it_windows_arr),
                                excl_idxs_it_ts,
                                axis=0,
                            )
                            resampled_true_it_mask_it_windows_arr = np.delete(
                                np.array(resampled_true_it_mask_it_windows_arr),
                                excl_idxs_it_ts,
                                axis=0,
                            )
                            midtransit_points_windows_arr = np.delete(
                                np.array(midtransit_points_windows_arr),
                                excl_idxs_it_ts,
                                axis=0,
                            )
                            # oot windows
                            resampled_flux_oot_windows_arr = np.delete(
                                np.array(resampled_flux_oot_windows_arr),
                                excl_idxs_oot_ts,
                                axis=0,
                            )
                            resampled_flux_quality_oot_windows_arr = np.delete(
                                np.array(resampled_flux_quality_oot_windows_arr),
                                excl_idxs_oot_ts,
                                axis=0,
                            )
                            resampled_true_it_mask_oot_windows_arr = np.delete(
                                np.array(resampled_true_it_mask_oot_windows_arr),
                                excl_idxs_oot_ts,
                                axis=0,
                            )
                            midoot_points_windows_arr = np.delete(
                                np.array(midoot_points_windows_arr),
                                excl_idxs_oot_ts,
                                axis=0,
                            )

                            # add data for TCE for a given sector
                            # it_window examples
                            sector_tce_example_data["transit_examples"][
                                "flux"
                            ] = resampled_flux_it_windows_arr
                            sector_tce_example_data["transit_examples"][
                                "flux_quality"
                            ] = resampled_flux_quality_it_windows_arr
                            sector_tce_example_data["transit_examples"][
                                "true_it_mask"
                            ] = resampled_true_it_mask_it_windows_arr
                            sector_tce_example_data["transit_examples"][
                                "t"
                            ] = midtransit_points_windows_arr
                            sector_tce_example_data["transit_examples"].update(
                                it_diff_img_data
                            )

                            # oot_window examples
                            sector_tce_example_data["not_transit_examples"][
                                "flux"
                            ] = resampled_flux_oot_windows_arr
                            sector_tce_example_data["not_transit_examples"][
                                "flux_quality"
                            ] = resampled_flux_quality_oot_windows_arr
                            sector_tce_example_data["not_transit_examples"][
                                "true_it_mask"
                            ] = resampled_true_it_mask_oot_windows_arr
                            sector_tce_example_data["not_transit_examples"][
                                "t"
                            ] = midoot_points_windows_arr
                            sector_tce_example_data["not_transit_examples"].update(
                                oot_diff_img_data
                            )

                            # add example data
                            target_sector_run_tce_example_data[tce_uid][
                                "sectors"
                            ].append(sector_tce_example_data)

                    logger.info(f"{chunk_num}) Processing for tce: {tce_uid} complete.")

                    # add data for chunk of target, sector_runs
                    chunk_tce_example_data.extend(
                        target_sector_run_tce_example_data.values()
                    )

                except Exception as e:
                    logger.error(
                        f"Skipping (t, sr): ({target}, {sector_run}) due to unexpected exception of type {type(e)}: {e}"
                    )
                    continue

        logger.info(
            f"PID_END {os.getpid()} using {psutil.Process().memory_info().rss / 1024 ** 2:.2f} MB"
        )
        logger.info(
            f"Finished extracting examples for all (t, sr) pairs in {chunk_title} successfully."
        )

        logger.info(f"Beginning dataset construction from extracted examples.")

        # write data to TFRecord dataset
        dataset_log_name = f"dataset_build_{chunk_title}"
        dataset_log_fp = Path(log_dir) / f"{dataset_log_name}.log"

        dataset_logger = logging.getLogger(dataset_log_name)
        dataset_logger.setLevel(logging.INFO)

        file_handler = logging.FileHandler(dataset_log_fp)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )

        dataset_logger.addHandler(file_handler)

        tfrec_dir = data_dir / "tfrecords"
        tfrec_dir.mkdir(exist_ok=True, parents=True)

        tfrec_fp = tfrec_dir / f"raw_shard_{chunk_title}.tfrecord"

        # for tce_example_data in chunk_tce_example_data:
        #     print(f"tce_ex_keys: {tce_example_data.keys()}")
        dataset_logger.info(f"Starting processing dataset chunk: {chunk_num}.")

        with tf.io.TFRecordWriter(str(tfrec_fp)) as writer:
            dataset_logger.info(f"Starting writing data to TFRecord: {tfrec_fp.name}.")
            # logger.info(
            #     f"PID_WRITE {os.getpid()} using {psutil.Process().memory_info().rss / 1024 ** 2:.2f} MB"
            # )
            for tce_example_data in chunk_tce_example_data:
                dataset_logger.info(
                    f"Adding data for TCE {tce_example_data['tce_uid']} to TFRecord"
                )
                examples_for_tce = serialize_set_examples_for_tce(tce_example_data)

                for example_for_tce in examples_for_tce:
                    writer.write(example_for_tce)

        # create auxiliary table
        dataset_logger.info(
            f"Creating auxillary data table for TFRecord: {tfrec_fp.name}."
        )

        data_tbl = write_data_to_auxiliary_tbl(chunk_tce_example_data, tfrec_fp)
        data_tbl.to_csv(tfrec_dir / f"data_tbl_{chunk_title}.csv", index=False)

        dataset_logger.info(
            f"Finished processing dataset chunk: {chunk_title} successfully."
        )
    except Exception as e:
        print(f"ERROR: {e}")


if __name__ == "__main__":
    # Set TF visible devices to CPU only for writing to TFRecords
    tf.config.set_visible_devices([], "GPU")

    # YAML configuration
    config_fp = Path(
        "/nobackupp27/jochoa4/work_dir/exoplanet_dl/transit_detection/config_build_dataset.yaml"
    )

    with open(config_fp, "r") as file:
        config = yaml.unsafe_load(file)

    # SETUP
    mode = config["setup"]["mode"]

    # Paths
    path_config = config["setup"]["paths"]

    log_dir = Path(path_config["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)  # create log dir

    data_dir = Path(path_config["data_dir"])
    data_dir.mkdir(exist_ok=True, parents=True)  # create dataset dir

    lcf_dir = Path(path_config["lcf_dir"])  # set lcf data dir
    tpf_dir = Path(path_config["tpf_dir"])  # set tpf data dir

    plot_dir = None
    if mode in config["modes"]:
        if config["modes"][mode]["plot"]:
            plot_dir = data_dir / "plots"
            plot_dir.mkdir(exist_ok=True, parents=True)

    # Setup Logging
    log_name = "build_dataset_setup"
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.INFO)
    log_path = log_dir / f"{log_name}.log"

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(file_handler)

    logger.info(f"Building dataset using configuration: {config}")

    # 2) tce table
    logger.info(f"Reading TCE table")
    tce_tbl = pd.read_csv(path_config["tce_tbl_fp"])

    tce_tbl_config = config["setup"]["tce_tbl"]

    # test on specific target, sector_runs
    # if mode == "test":
    #     test_config = config["modes"]["test"]
    #     t_sr_groups = tce_tbl.groupby(["target_id", "sector_run"])
    #     filtered_tce_tbl = pd.DataFrame([])
    #     for t, sr in test_config["target_sector_runs"]:
    #         filtered_tce_tbl = pd.concat(
    #             [filtered_tce_tbl, t_sr_groups.get_group((t, sr))]
    #         )
    #     tce_tbl = filtered_tce_tbl

    tce_tbl.rename(
        columns={
            "label": "disposition",
            "label_source": "disposition_source",
            "uid": "tce_uid",
        },
        inplace=True,
    )  # rename tce_tbl columns

    # filter for relevant dispositions
    logger.info(
        f"Filtering TCE table for TCEs that are of disposition {tce_tbl_config['keep_tce_disps']}"
    )
    tce_tbl = tce_tbl.loc[tce_tbl["disposition"].isin(tce_tbl_config["keep_tce_disps"])]

    logger.info(
        f"Filtering TCE table for targets that have at least one {tce_tbl_config['keep_target_disps']} TCE"
    )
    tce_tbl = tce_tbl.groupby("target_id").filter(
        lambda g: g["disposition"].isin(tce_tbl_config["keep_target_disps"]).any()
    )

    # 3) dataset preprocessing parameters
    dataset_config = config["setup"]["dataset_preproc_params"]
    logger.info(f"Setting up data configuration parameters using: {dataset_config}")

    tce_example_disps = dataset_config["tce_example_disps"]

    detrend_lc_flag = dataset_config["detrend_lc_flag"]
    n_durations_window = dataset_config["n_durations_window"]
    ex_it_mask_n_durations_window = dataset_config["ex_it_mask_n_durations_window"]
    sg_it_mask_n_durations_window = dataset_config["sg_it_mask_n_durations_window"]
    weak_sec_mask_maxmes_thr = dataset_config["weak_sec_mask_maxmes_thr"]
    frac_valid_cadences_in_window_thr = dataset_config[
        "frac_valid_cadences_in_window_thr"
    ]
    frac_valid_cadences_it_thr = dataset_config["frac_valid_cadences_it_thr"]
    buffer_time = dataset_config["buffer_time"]
    resampled_num_points = dataset_config[
        "resampled_num_points"
    ]  # number of points in the window after resampling
    rnd_seed = dataset_config["rnd_seed"]
    size_img = dataset_config["size_img"]  # resize images to this size
    f_size = dataset_config[
        "f_size"
    ]  # enlarge `size_img` by these factors; final dimensions are f_size * size_img
    center_target = dataset_config["center_target"]  # center target in images

    # 4) multiprocessing - job run
    mp_config = config["setup"]["multiprocessing"]
    logger.info(f"Setting up multiprocessing parameters using: {mp_config}")

    # grouped target data is unique, so can use views for parallel operations

    target_jobs = [
        (target, t_tce_data) for (target,), t_tce_data in tce_tbl.groupby(["target_id"])
    ]
    logger.info(f"Found {len(target_jobs)} confirming targets.")
    logger.info(
        f"of form {target_jobs[:10]} and type ({type(target_jobs)} w/ first {type(target_jobs[0])})"
    )

    job_chunk_size = mp_config["job_chunk_size"]

    chunked_jobs = [
        target_jobs[i : i + job_chunk_size]
        for i in range(0, len(target_jobs), job_chunk_size)
    ]
    logger.info(
        f"Chunked {len(target_jobs)} targets into groups of {job_chunk_size} for multiprocessing."
    )

    # chunk validation
    validate_chunks = mp_config["validate_chunks"]

    processed_chunk_mask = [0] * len(chunked_jobs)
    logger.info(
        f"Processing a total of {len(chunked_jobs)} chunks of size {job_chunk_size}"
    )
    if validate_chunks:
        # build chunk mask
        tfrec_dir = data_dir / "tfrecords"
        tfrec_dir.mkdir(exist_ok=True, parents=True)

        logger.info("Building chunk mask for processed chunks")
        processed_chunk_mask = build_chunk_mask(chunked_jobs, tfrec_dir)

    logger.info(
        f"Skipping processing for {sum(processed_chunk_mask)} chunks that have already been processed."
    )

    job_log_dir = log_dir / "jobs"
    job_log_dir.mkdir(parents=True, exist_ok=True)

    # defines constant func args

    partial_func = partial(
        process_target,
        lcf_dir=lcf_dir,
        tpf_dir=tpf_dir,
        tce_example_disps=deepcopy(tce_example_disps),
        n_durations_window=n_durations_window,
        detrend_lc_flag=detrend_lc_flag,
        ex_it_mask_n_durations_window=ex_it_mask_n_durations_window,
        sg_it_mask_n_durations_window=sg_it_mask_n_durations_window,
        weak_sec_mask_maxmes_thr=weak_sec_mask_maxmes_thr,
        frac_valid_cadences_in_window_thr=frac_valid_cadences_in_window_thr,
        frac_valid_cadences_it_thr=frac_valid_cadences_it_thr,
        buffer_time=buffer_time,
        resampled_num_points=resampled_num_points,
        rnd_seed=rnd_seed,
        size_img=deepcopy(size_img),
        f_size=deepcopy(f_size),
        center_target=center_target,
        plot_dir=plot_dir,
        log_dir=job_log_dir,
        data_dir=data_dir,
    )

    num_processes = min(mp_config["num_processes"], multiprocessing.cpu_count())
    # max tasks per child to manage memory leaks -> each worker restart after 1 task (100 pairs processed)
    logger.info(f"Beginning multiprocessing with {num_processes} processes.")
    with multiprocessing.Pool(processes=num_processes, maxtasksperchild=1) as pool:
        for chunk_i, job_chunk_data in enumerate(chunked_jobs):
            if not processed_chunk_mask[chunk_i]:
                try:
                    # print(f"Processing chunk {chunk_i + 1} with multiprocessing pool.")
                    logger.info(
                        f"Processing chunk {chunk_i + 1} with multiprocessing pool."
                    )
                    print(f"Processing chunk {chunk_i + 1} with multiprocessing pool.")
                    pool.apply_async(
                        partial_func,
                        args=(job_chunk_data, chunk_i + 1, len(chunked_jobs)),
                    )
                except Exception as e:
                    logger.info(
                        f"ERROR processing chunk {chunk_i + 1} with multiprocessing pool: {e}."
                    )
                    print(
                        f"ERROR processing chunk {chunk_i + 1} with multiprocessing pool: {e}."
                    )
            else:
                logger.info(f"Skipping processing for chunk {chunk_i + 1}.")
        (
            f"PID_MAIN {os.getpid()} using {psutil.Process().memory_info().rss / 1024 ** 2:.2f} MB"
        )
        pool.close()
        pool.join()
