"""
Script used to perform inference on a given lightcurve
"""

# 3rd Party
from keras.saving import load_model
from pathlib import Path
import numpy as np
import lightkurve as lk
from collections import defaultdict
import tensorflow as tf

import yaml
import logging

from typing import Dict, List, Tuple

# local
from src_preprocessing.lc_preprocessing.detrend_timeseries import detrend_flux_using_sg_filter
from transit_detection.inference.utils_inference import InferenceInputFn as InputFn, serialize_set_examples_for_tce
from transit_detection.utils_difference_img_processing import extract_diff_img_data_from_window


def write_example_to_tfrecord(tfrec_fp, flux, diff_img, oot_img, snr_img, target_img, target_pos_col, target_pos_row) -> None:
    pass
    # with tf.io.TFRecordWriter(str(tfrec_fp)) as writer:
    #     for data_for_tce in chunk_data:
    #         examples_for_tce = serialize_set_examples_for_tce(data_for_tce)

    #         for example_for_tce in examples_for_tce:
    #             writer.write(example_for_tce)

def plot_flux_window(flux: np.ndarray) -> None:
    pass


def apply_stdnorm_to_img(image: np.ndarray) -> np.ndarray:
    pass


def apply_norm_to_flux(flux: np.ndarray) -> np.ndarray:
    pass


def preprocess_diff_img(image: np.ndarray):
    # apply dataset build preprocessing
    # apply norm
    pass


def preprocess_flux_window(flux_window: np.ndarray, resampling_rate: int = 100):
    # apply dataset build preprocessing
    # apply norm
    pass

def perform_inference_on_target_sector_run(
        model_fp: Path | str,
        target_sector_run_data: Dict,
        log_dir: Path | str,
        trial_transit_durations: List[float | int],
        n_durations_window: int,
        frac_valid_cadences_in_window_thr: float,
        frac_valid_cadences_it_thr: float,
        buffer_time: int,
        gap_width: float,
        resampled_num_points: int,
        # rnd_seed:,
        size_img: List[int, int] | Tuple[int, int],
        f_size: List[int, int] | Tuple[int, int],
        center_target: bool,
        batch_size: int):
    """
    Perform inference on an input target, sector_run pair, by first performing normalization

    Returns:

        {
            "target": f"{tic_id}",
            "sector_run": f"{sector_run}",
            "preds_per_sector": {
                "22: {
                    "trial_transit_durations": {
                        "1.5" : [0, ... 0.42, 0.45, 0.37, ... 0]
                        ...
                        "15" : [0, ... 0.67, 0.7, 0.65, ... 0]
                    }
                }
            }
        }

    """

    # TODO: Update for file logging
    logger = logging.getLogger(f"inference_{target_sector_run_data['tic_id']}-{target_sector_run_data['sector_run']}")
    logger.setLevel(logging.INFO)

    log_path = Path(log_dir) / (f"inference_{target_sector_run_data['tic_id']}-{target_sector_run_data['sector_run']}.log")
    file_handler = logging.FileHandler(log_path)
    logger_formatter = logging.Formatter('%(asctime)s - %(levelname)s- %(message)s')

    file_handler.setFormatter(logger_formatter)
    logger.addHandler(file_handler)
    
    # Load model
    # TODO: add custom model scope?
    model = tf.keras.models.load_model(model_fp) # TODO: or keras.saving.load_model? depends on keras version

    # Return object
    inference_results = {
            "target_id": target_sector_run_data['target_id'],
            "sector_run": target_sector_run_data['sector_run'],
            "preds_per_sector": {} # Contain predictions per sector if multi-sector run
        }

    # Process target_sector_run_data 
    if "-" in target_sector_run_data["sector_run"]:
        start_sector, end_sector = [
            int(sector) for sector in target_sector_run_data["sector_run"].split("-")
        ]
        sector_run_arr = np.arange(start_sector, end_sector + 1)
    else:
        sector_run_arr = [int(target_sector_run_data["sector_run"])]

    # find light curve data for target
    # TODO: handle multi sector run logic
    search_lc_res = lk.search_lightcurve(
        target=f"tic{target_sector_run_data['target_id']}",
        mission="TESS",
        author=("TESS-SPOC", "SPOC"),
        exptime=120,
        cadence="long",
        sector=sector_run_arr,
    )

    # Download lightcurve fits file(s)
    found_sectors = [int(el.split(" ")[-1]) for el in search_lc_res.mission]
    logger.info(f'Found {len(found_sectors)} sectors with light curve data for TIC {target_sector_run_data["tic_id"]}.')
    
    for sector_i, sector in enumerate(found_sectors):
        # Initialize found_sectors results
        inference_results["preds_per_sector"][str(sector)] = {"trial_transit_durations"}
        
        lcf = search_lc_res[sector_i].download(
            download_dir=lc_dir, quality_bitmask="default", flux_column="pdcsap_flux"
        )

        lcf = lk.LightCurve({"time": lcf.time.value, "flux": np.array(lcf.flux.value)})

        raw_time, raw_flux = lcf.time.value, lcf.flux.value

        # Detrend flux
        # NOTE: How should we be handling detrending, if in the original pipeline
        # NOTE: Should we introduce distortion into signal?
        time, flux, trend = detrend_flux_using_sg_filter(
            lc=lcf,
            mask_in_transit=np.zeros_like(raw_time).astype(bool),
            win_len=int(1.2 * 24 * 30),  # TODO: Update win_len
            sigma=5,
            max_poly_order=6,
            penalty_weight=1,
            break_tolerance=5,
        )

        # Find target pixel file for target
        search_tp_res = lk.search_targetpixelfile(
            target=f"tic{target_sector_run_data['tic_id']}",
            mission="TESS",
            author=("TESS-SPOC", "SPOC"),
            exptime=120,
            cadence="long",
            sector=sector,
        )

        # download target pixel file
        tpf = search_tp_res[0].download(
            download_dir=str(tpf_dir), quality_bitmask="default"
        )

        # compute difference image data for each window for target pixel file data
        # compute transit difference image data
        diff_img_data = {
            feature_name: []
            for feature_name in [
                "diff_img",
                "oot_img",
                "snr_img",
                "target_img",
                "target_pos",
            ]
        }

        for trial_transit_duration in trial_transit_durations:
            """
            For a given flux curve and tpf pair:
            1) Detrend flux using SG filter (with params used in pipeline: sigma=5, max_poly_order=6, penalty_weight=1, break_tolerance=5)
                2. For each trial transit duration (from Kepler): 1.5, 2, 2.5, 3, 3.5, 4.5, 5, 6, 7.5, 9, 10, 12, 12.5, and 15 hours
                    1. Calculate window size (in indices corresponding to array) based on trial duration. Ie. 5 * transit_durations
                    2. Slide over flux curve, starting i = 0 and ending at i = N _f - window_size; (N_f: flux curve length)
                        1. At each index ‘i’ extract a window from i to i + window_size
                            1. Resample extracted window to 100 points (used in dataset pipeline)
                            2. Extract diff_img from window using functions already implemented at the provided timestamp (centermost?)
                    3. Either: A) Predict on each individual window OR B) Parallelize predictions for all windows in flux curve, to get prob 0-1
                    4. *TODO: Determine if writing to tfrecord -> predicting or directly using numpy arr/ tf tensor for predictions.
                    5. Use prediction probs to compute average prob for each index in flux_curve. (Such that. All windows covering a given index will be considered in the prob at that index)
                3. ** Compare probability curves from each trial transit duration
                4. *TODO: Look into next steps
            """

            # Perform Dataset Preprocessing

            # Calculate window size (in indices corresponding to array)
            # 5 * trial transit duration hours in seconds // cadence in seconds
            num_it_window_points = (trial_transit_duration * 3600) // 120
            num_window_points = n_durations_window * num_it_window_points

            # TODO: handle edges?

            valid_start_idx = num_window_points // 2 # TODO: update
            valid_end_idx = len(time) - num_window_points // 2

            valid_midpoints = np.arange(valid_start_idx, valid_end_idx)

            window_indices = [
                (midpoint - num_window_points // 2, midpoint + num_window_points // 2)
                for midpoint in valid_midpoints
            ]
            it_window_indices = [
                (
                    midpoint - num_it_window_points // 2,
                    midpoint + num_it_window_points // 2,
                )
                for midpoint in valid_midpoints
            ]

            valid_diff_imgs = []
            valid_flux_windows = []

            for midpoint, (oot_start, oot_end), (it_start, it_end) in zip(
                valid_midpoints, window_indices, it_window_indices
            ):
                # Get detrended flux window
                flux_window = flux[oot_start:oot_end]

                # Get raw flux window and normalize by dividing by average trend from corresponding indices
                raw_it_flux = raw_flux[it_start:it_end]

                it_trend_mean = np.mean(trend[it_start:it_end])

                norm_it_flux = raw_it_flux / it_trend_mean

                # Replace detrended it points with normalized raw it points
                center_idx = len(flux_window) // 2
                flux_window[
                    (center_idx - len(norm_it_flux) // 2) : (
                        center_idx + len(norm_it_flux) // 2
                    )
                ] = norm_it_flux

                valid_flux_windows.append(flux_window)

                plot_flux_window(flux_window)

                try:
                    # Use model to predict
                    diff_imgs = extract_diff_img_data_from_window(
                        tpf.path,
                        midpoint,
                        trial_transit_duration,
                        buffer_time,
                        target_sector_run_data["ra"],
                        target_sector_run_data["dec"],
                        size_img,
                        f_size,
                        sector,
                        center_target,
                        target_sector_run_data["tic_id"],
                        plot_dir=None,
                    )
                except ValueError as error:
                    logger.warning(f"ERROR: Extracting flux windows - {error.args[0]}")
                    continue

                diff_img_data.append(*diff_imgs)
                
                valid_diff_imgs.append(diff_imgs)

            # Write examples to tfrecord per trial transit duration
            tfrec_fp = tfrec_dir / f"inference_shard_{str(target_sector_run_data["tic_id"])}_{str(sector)}_{str(trial_transit_duration)}"
            write_example_to_tfrecord(tfrec_fp, flux, *diff_imgs)

            # TODO: Update to predict on all examples per light curve in batches?
            inference_input_fn = InputFn(
            file_paths=[tfrec_fp]
            batch_size=batch_size,
            mode="PREDICT",
            features_set=config["features_set"]
            # label_map=None,
            # online_preproc_params=None,
            # multiclass=config["config"]["multi_class"],
            # feature_map=config["feature_map"],
            # label_field_name=config["label_field_name"],
            )

            # NOTE: interaction between batch size in InputFn vs predict?
            res_inference = model.predict(
                x=inference_input_fn(),
                # y=None,
                # batch_size=None,
                verbose=config["verbose_model"],
                sample_weight=None,
                steps=None,
                callbacks=None,
                # max_queue_size=10,
                # workers=1,
                # use_multiprocessing=False,
            )
            # TODO: add return for 

            # TODO: validate logic -> ensure match between res_inference & possible splitting?
            lc_pred_sum = np.zeroslike(flux)
            lc_pred_count = np.zeroslike(flux)
            
            for i, res_window in enumerate(res_inference, start=0):
                lc_pred_sum[i:i + num_window_points] += res_window
                lc_pred_count[i:i + num_window_points] += 1
            
            lc_avg_preds = lc_pred_sum / lc_pred_count

            # Record predictions for trial_transit_durations
            inference_results["preds_per_sector"][str(sector)]["trial_transit_durations"][str(trial_transit_duration)] = lc_avg_preds


        """
        2) Generates output predictions for the input samples.
        Computation is done in batches. This method is designed for batch processing of large numbers of inputs. It is not intended for use inside of loops that iterate over your data and process small numbers of inputs at a time.
        For small numbers of inputs that fit in one batch, directly use __call__() for faster execution, e.g., model(x), or model(x, training=False) if you have layers such as BatchNormalization that behave differently during inference.

        2) If x is a tf.data.Dataset and steps is None, predict() will run until the input dataset is exhausted.
        """

    return inference_results

if __name__ == "__main__":
    
    # File path to YAML config file
    config_fp = Path("/Users/jochoa4/Desktop/ExoMiner/exoplanet_dl/transit_detection/inference/config_inference.yaml")

    with open(config_fp, "r") as file:
        config = yaml.unsafe_load(file)

    trial_transit_durations = config["trial_transit_durations"]
    model_fp = config["model_fp"]
    log_dir = config["log_dir"]

    tfrec_dir = config["dataset_fps"]["predict"] # TODO: update?

    lc_dir = config["lc_dir"]
    tpf_dir = config["tpf_dir"]
    
    # Preprocessing params used in dataset build
    preprocessing_config = config["dataset_preprocessing_params"]
    n_durations_window = preprocessing_config["n_durations_window"]
    frac_valid_cadences_in_window_thr = preprocessing_config["frac_valid_cadences_in_window_thr"]
    frac_valid_cadences_it_thr = preprocessing_config["frac_valid_cadences_it_thr"]
    buffer_time = preprocessing_config["buffer_time"] # in minutes, between in-transit cadences and out-of-transit cadences
    # days used to split timeseries into smaller segments if interval between cadences is larger than gap_width
    gap_width = preprocessing_config["gap_width"]
    resampled_num_points = preprocessing_config["resampled_num_points"]  # number of points in the window after resampling
    # rnd_seed = preprocessing_config["rnd_seed"]

    # difference image data parameters
    size_img = preprocessing_config["size_img"]  # resize images to this size
    f_size = preprocessing_config["f_size"]  # enlarge `size_img`; final dimensions are f_size * size_img
    center_target = preprocessing_config["center_target"]  # center target in images

    target_sector_run_data = {  # CP as test
        "tic_id": "1003831",
        "sector_run": "8",
        "ra": float("130.2951647"),
        "dec": float("-16.03628"),
    }

    batch_size = config["batch_size"] # Batch size for predictions

    perform_inference_on_target_sector_run(
        model_fp = model_fp,
        target_sector_run_data = target_sector_run_data,
        log_dir = log_dir,
        trial_transit_durations = trial_transit_durations,
        n_durations_window = n_durations_window,
        frac_valid_cadences_in_window_thr = frac_valid_cadences_in_window_thr,
        frac_valid_cadences_it_thr = frac_valid_cadences_it_thr,
        buffer_time = buffer_time,
        gap_width = gap_width,
        resampled_num_points = resampled_num_points,
        # rnd_seed:,
        size_img = size_img,
        f_size = f_size,
        center_target = center_target,
        batch_size = batch_size)
