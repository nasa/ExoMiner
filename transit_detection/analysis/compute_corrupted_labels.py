"""
Compute corrupted negative label statistics based on weak secondary transits for entire target, sector_run pair
"""

# 3rd Party
from pathlib import Path
import glob
import tensorflow as tf
import pandas as pd
import numpy as np

# Local
from transit_detection.utils_local_fits_processing import (
    search_and_read_tess_lightcurve,
)

from src_preprocessing.lc_preprocessing.utils_ephemeris import (
    find_first_epoch_after_this_time,
)


def is_neg_example_corrupted_by_weak_secondary(
    ex_midpoint: float,
    ex_duration: float,
    ex_n_durations_window: float,
    possible_ws_tce_data: list,
    ws_n_durations_window: float,
):
    "Checks if a negative example is corrupted by a weak secondary transit using ephemerides information from itself"
    ex_duration /= 24  # convert duration to days

    oot_window_start = ex_midpoint - (ex_n_durations_window * 0.5) * ex_duration
    oot_window_end = ex_midpoint + (ex_n_durations_window * 0.5) * ex_duration

    # Process each tce
    for tce_data in possible_ws_tce_data:
        duration = tce_data["tce_duration"]
        period = tce_data["tce_period"]
        epoch = tce_data["tce_time0bk"]
        maxmes = tce_data["tce_maxmes"]
        maxmesd = tce_data["tce_maxmesd"]

        if maxmes > 7.1:  # threshold
            secondary_epoch = epoch + maxmesd

            first_transit_time = find_first_epoch_after_this_time(
                secondary_epoch, period, oot_window_start
            )

            weak_secondary_midtransit_points = [
                first_transit_time + phase_k * period
                for phase_k in range(
                    int(np.ceil((oot_window_end - oot_window_start) / period))
                )
            ]

            for ws_midpoint in weak_secondary_midtransit_points:
                if not ws_n_durations_window:  # Use center point
                    if oot_window_start <= ws_midpoint <= oot_window_end:
                        return True
                else:
                    # TODO: verify this logic
                    ws_it_window_start = (
                        ws_midpoint - (ws_n_durations_window * 0.5) * duration
                    )
                    ws_it_window_end = (
                        ws_midpoint + (ws_n_durations_window * 0.5) * duration
                    )
                    if (
                        ws_it_window_start <= oot_window_end
                        and ws_it_window_end >= oot_window_start
                    ):
                        return True

    return False


if __name__ == "__main__":

    # NOTE: Implementation assumes example windows were 5 transit durations relative to example tce
    tce_tbl = pd.read_csv(
        "/nobackup/jochoa4/work_dir/data/tables/tess_2min_tces_dv_s1-s68_all_msectors_11-29-2023_2157_newlabels_nebs_npcs_bds_ebsntps_to_unks.csv"
    )

    tce_tbl = tce_tbl.loc[
        tce_tbl["label"].isin(["EB", "KP", "CP", "NTP", "NEB", "NPC"])
    ]  # filter for relevant labels

    tce_tbl.rename(
        columns={"label": "disposition", "label_source": "disposition_source"},
        inplace=True,
    )

    dataset_dir = Path(
        "/nobackupp27/jochoa4/work_dir/data/datasets/TESS_exoplanet_dataset_05-04-2025_split_norm"
    )
    tfrec_dir = dataset_dir / "tfrecords"

    stats_dir = Path(
        "/nobackupp27/jochoa4/work_dir/job_runs/compute_corrupted_labels_05-04-2025_v1"
    )
    stats_dir.mkdir(parents=True, exist_ok=True)

    print(f"Begin processing splits")

    for split in ["test", "train", "val"]:
        print(f"Processing split {split}")

        stats_csv = stats_dir / f"{split}_corrupted_neg_label_analysis.csv"

        split_dir = tfrec_dir / split

        tfrec_pattern = f"norm_{split}_shard_????-????.tfrecord"  # TODO: update to ????
        tfrec_fps = glob.glob(str(split_dir / tfrec_pattern))

        print(f"Found {len(tfrec_fps)} file paths using pattern: {tfrec_pattern}")

        csv_rows = []

        neg_examples = 0
        corrupted_neg_examples = 0
        for tfrec_fp in tfrec_fps:
            print(f"Starting processing tfrecord: {Path(tfrec_fp).name}")
            tfrec_dataset = tf.data.TFRecordDataset(str(tfrec_fp))

            for str_record in tfrec_dataset.as_numpy_iterator():
                example = tf.train.Example()
                example.ParseFromString(str_record)

                ex_uid = (
                    example.features.feature["uid"].bytes_list.value[0].decode("utf-8")
                )
                ex_label = (
                    example.features.feature["label"]
                    .bytes_list.value[0]
                    .decode("utf-8")
                )
                ex_disposition = (
                    example.features.feature["disposition"]
                    .bytes_list.value[0]
                    .decode("utf-8")
                )
                ex_tce_period = example.features.feature["tce_period"].float_list.value[
                    0
                ]

                ex_tce_duration = example.features.feature[
                    "tce_duration"
                ].float_list.value[0]

                ex_tce_maxmes = example.features.feature["tce_maxmes"].float_list.value[
                    0
                ]

                ex_tce_maxmesd = example.features.feature[
                    "tce_maxmesd"
                ].float_list.value[0]

                ex_midpoint = example.features.feature["t"].float_list.value[0]

                # Process corruption based on the target, sector_run associated with it
                ex_tce_uid = ex_uid.split("_")[
                    0
                ]  # ie: 1129033-1-S1-36_t_1412.3449704478298 -> ['1129033-1-S1-36', 't', '1412.3449704478298'] -> '1129033-1-S1-36'

                ex_target_id = int(
                    ex_tce_uid.split("-")[0]
                )  # ie: '1129033-1-S1-36' -> ['1129033', '1', 'S1', '36'] -> int(1129033)

                ex_sector_run = "-".join(
                    ex_tce_uid.split("-")[
                        2:
                    ]  # ie single_sector_run: '1129033-1-S1' -> ['1129033', '1', 'S1'] -> ['S1'] -> 'S1'
                )  # multi_sector_run: '1129033-1-S1-36' -> ['1129033', '1', 'S1', '36'] -> ['S1', '36'] -> 'S1-36'
                ex_sector_run = ex_sector_run.replace(
                    "S", ""
                )  # Reformat to expected tce_tbl format: S1-36 -> 1-36

                assert ex_label in (
                    "0",
                    "1",
                ), f"ERROR: Expected '0' or '1' but got {ex_label}"

                row_entry = {
                    "uid": ex_uid,
                    # "target_id" : ex_target_id,
                    # "sector_run" : ex_sector_run,
                    # "midpoint" : ex_midpoint ,
                    "tce_period": ex_tce_period,
                    "tce_duration": ex_tce_duration,
                    "tce_maxmes": ex_tce_maxmes,
                    "tce_maxmesd": ex_tce_maxmesd,
                    "disposition": ex_disposition,
                    "label": ex_label,
                    "corrupt": 0,
                    "corrupt_ws_0.0_td_ex_1.0_td": 0,
                    "corrupt_ws_0.0_td_ex_2.5_td": 0,
                    "corrupt_ws_0.0_td_ex_5.0_td": 0,
                    "corrupt_ws_3.0_td_ex_1.0_td": 0,
                    "corrupt_ws_3.0_td_ex_2.5_td": 0,
                    "corrupt_ws_3.0_td_ex_5.0_td": 0,
                    "cross_tce_corrupt_ws_0.0_td_ex_1.0_td": 0,
                    "cross_tce_corrupt_ws_0.0_td_ex_2.5_td": 0,
                    "cross_tce_corrupt_ws_0.0_td_ex_5.0_td": 0,
                    "cross_tce_corrupt_ws_3.0_td_ex_1.0_td": 0,
                    "cross_tce_corrupt_ws_3.0_td_ex_2.5_td": 0,
                    "cross_tce_corrupt_ws_3.0_td_ex_5.0_td": 0,
                    "self_corrupt_ws_0.0_td_ex_1.0_td": 0,
                    "self_corrupt_ws_0.0_td_ex_2.5_td": 0,
                    "self_corrupt_ws_0.0_td_ex_5.0_td": 0,
                    "self_corrupt_ws_3.0_td_ex_1.0_td": 0,
                    "self_corrupt_ws_3.0_td_ex_2.5_td": 0,
                    "self_corrupt_ws_3.0_td_ex_5.0_td": 0,
                }

                if ex_label == "0":
                    # Process corruption based on its own ephemerides information
                    neg_examples += 1

                    sector_run_data = tce_tbl[
                        (tce_tbl["target_id"] == ex_target_id)
                        & (tce_tbl["sector_run"] == ex_sector_run)
                    ]

                    target_sector_run_tce_data = []  # should exclude example tce data
                    ex_tce_data = []  #

                    for (
                        tce_i,
                        tce_data,
                    ) in (
                        sector_run_data.iterrows()
                    ):  # get all tces in target, sector_run pair
                        if ex_tce_uid != tce_data["uid"]:
                            target_sector_run_tce_data.append(
                                {
                                    "tce_time0bk": tce_data["tce_time0bk"],
                                    "tce_period": tce_data["tce_period"],
                                    "tce_duration": tce_data["tce_duration"],
                                    "tce_maxmes": tce_data["tce_maxmes"],
                                    "tce_maxmesd": tce_data["tce_maxmesd"],
                                }
                            )
                        else:
                            ex_tce_data.append(
                                {
                                    "tce_time0bk": tce_data["tce_time0bk"],
                                    "tce_period": tce_data["tce_period"],
                                    "tce_duration": tce_data["tce_duration"],
                                    "tce_maxmes": tce_data["tce_maxmes"],
                                    "tce_maxmesd": tce_data["tce_maxmesd"],
                                }
                            )

                    for ex_n_durations_window in [1.0, 2.5, 5.0]:
                        for ws_n_durations_window in [0.0, 3.0]:
                            # CHECKS WITH ONLY EXAMPLE EPHEMERIDES
                            self_ws_corrupted = int(
                                is_neg_example_corrupted_by_weak_secondary(
                                    ex_midpoint=ex_midpoint,
                                    ex_duration=ex_tce_duration,
                                    possible_ws_tce_data=ex_tce_data,  # Use own example tce data
                                    ex_n_durations_window=ex_n_durations_window,
                                    ws_n_durations_window=ws_n_durations_window,  # TODO: FIX
                                )
                            )

                            # CHECKS WITH ONLY SECTOR RUN EXCLUDING EXAMPLE EPHEMERIDES
                            cross_tce_ws_corrupted = int(
                                is_neg_example_corrupted_by_weak_secondary(
                                    ex_midpoint=ex_midpoint,
                                    ex_duration=ex_tce_duration,
                                    possible_ws_tce_data=target_sector_run_tce_data,
                                    ex_n_durations_window=ex_n_durations_window,
                                    ws_n_durations_window=ws_n_durations_window,
                                )
                            )

                            # CHECKS WITH ENTIRE SECTOR RUN INCLUDING EXAMPLE EPHEMERIDES
                            ws_corrupted = int(
                                self_ws_corrupted or cross_tce_ws_corrupted
                            )
                            # TODO: you can skip large n_durations windows if any part of a smaller window inside??

                            self_ws_corrupted_field = f"self_corrupt_ws_{ws_n_durations_window}_td_ex_{ex_n_durations_window}_td"
                            assert (
                                self_ws_corrupted_field in row_entry
                            ), f"ERROR: key {self_ws_corrupted_field} not in row_entry"

                            row_entry[self_ws_corrupted_field] = self_ws_corrupted

                            cross_tce_ws_corrupted_field = f"cross_tce_corrupt_ws_{ws_n_durations_window}_td_ex_{ex_n_durations_window}_td"
                            assert (
                                cross_tce_ws_corrupted_field in row_entry
                            ), f"ERROR: key {cross_tce_ws_corrupted_field} not in row_entry"
                            row_entry[cross_tce_ws_corrupted_field] = (
                                cross_tce_ws_corrupted
                            )

                            ws_corrupted_field = f"corrupt_ws_{ws_n_durations_window}_td_ex_{ex_n_durations_window}_td"
                            assert (
                                ws_corrupted_field in row_entry
                            ), f"ERROR: key {ws_corrupted_field} not in row_entry"
                            row_entry[ws_corrupted_field] = ws_corrupted

                            corrupted_field = "corrupt"
                            assert (
                                corrupted_field in row_entry
                            ), f"ERROR: key {corrupted_field} not in row_entry"
                            row_entry[corrupted_field] |= ws_corrupted

                csv_rows.append(row_entry)
                if row_entry[corrupted_field]:
                    corrupted_neg_examples += 1
            print(f"Finished processing tfrecord: {Path(tfrec_fp).name}")
        print(f"Finished processing split {split}")

        split_df = pd.DataFrame(csv_rows)
        print(
            f"Found {split_df['corrupt'].sum(axis=0)} / {len(split_df)} total examples to be corrupt.\n"
        )
        print(
            f"Found {corrupted_neg_examples} / {neg_examples} negative examples to be corrupt.\n"
        )

        split_df.to_csv(str(stats_csv), index=False)
