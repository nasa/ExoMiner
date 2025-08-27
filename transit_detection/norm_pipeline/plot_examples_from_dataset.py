"""
Plot normalized and unnormalized examples from a split dataset
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import numpy as np
from pathlib import Path
import tensorflow as tf
from collections import defaultdict

from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm

import glob
import random
import traceback


def plot_flux_and_diff_img_data(
    flux_curve,
    diff_img,
    oot_img,
    snr_img,
    target_img,
    save_fp,
    label="",
    norm=False,
    snr=None,
    midpoint=None,
):
    """Plot difference image data for TCE in a given quarter/sector.

    Args:
        diff_img: NumPy array, difference image
        oot_img: NumPy array, out-of-transit image
        snr_img: NumPy array, SNR image
        target_img: NumPy array, target location image
        target_coords: dict, target location 'x' and 'y'
        img_num: str, quarter/sector run
        uid: str, TCE ID
        save_fp: Path, file path to saved plot

    Returns:

    """
    label = "In Transit" if label == 1.0 else "Out of Transit"
    snr = str(round(snr, 2)) if snr else None
    it_idx = np.argmax(target_img)
    row, col = np.unravel_index(
        it_idx, target_img.shape
    )  # Get row and col as if array were flattened
    target_coords = {"x": row, "y": col}

    fig = plt.figure(figsize=(16, 16))

    gs = gridspec.GridSpec(3, 2, figure=fig)

    ax = fig.add_subplot(gs[0, :])
    time = np.linspace(0, 100, 100)
    ax.plot(time, flux_curve, linestyle="None", marker="o", markersize=3, alpha=0.6)
    ax.set_title(
        f"{label} Flux Curve w/ midpoint {round(midpoint,2) if midpoint else None} Over Time"
    )
    ax.set_xlabel("Time")
    ax.set_ylabel("Flux value") if not norm else ax.set_ylabel("Normalized Flux value")

    # Approximate image center

    # diff img
    ax = fig.add_subplot(gs[1, 0])
    im = ax.imshow(diff_img, norm=None)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    ax.scatter(
        target_coords["y"], target_coords["x"], marker="x", color="r", label="Target"
    )
    ax.set_ylabel("Row")
    ax.set_xlabel("Col")
    ax.legend()
    (
        ax.set_title("Difference Flux (e-/cadence)")
        if not norm
        else ax.set_title("Normalized Difference Flux")
    )
    # oot img
    ax = fig.add_subplot(gs[1, 1])
    im = ax.imshow(oot_img, norm=LogNorm() if not norm else None)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    ax.scatter(
        target_coords["y"], target_coords["x"], marker="x", color="r", label="Target"
    )
    ax.set_ylabel("Row")
    ax.set_xlabel("Col")
    ax.legend()
    (
        ax.set_title("Out-of-Transit Flux (e-/cadence)")
        if not norm
        else ax.set_title("Normalized Out-of-Transit Flux")
    )
    # target img
    ax = fig.add_subplot(gs[2, 0])
    ax.imshow(target_img)
    ax.set_ylabel("Row")
    ax.set_xlabel("Col")
    ax.set_title("Target Position")
    # snr img
    ax = fig.add_subplot(gs[2, 1])
    im = ax.imshow(snr_img, norm=LogNorm() if not norm else None)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    ax.scatter(
        target_coords["y"], target_coords["x"], marker="x", color="r", label="Target"
    )
    ax.set_ylabel("Row")
    ax.set_xlabel("Col")
    ax.legend()
    (
        ax.set_title(f"Difference SNR: {snr}")
        if not norm
        else ax.set_title(f"Normalized Difference SNR: {snr}")
    )

    plt.tight_layout()
    try:
        plt.savefig(save_fp)
        plt.close()

        print(f"Plotted to {save_fp}")
    except Exception as e:
        print(f"ERROR plotting: {e}")


def get_diff_imgs_from_shard(
    dest_tfrec_fp: Path,
    diff_img_feature_keys: list[str] = ["diff_img", "oot_img", "snr_img", "target_img"],
) -> list[list[np.array]]:
    diff_imgs = []

    src_tfrecord_dataset = tf.data.TFRecordDataset(dest_tfrec_fp)

    for string_record in src_tfrecord_dataset.as_numpy_iterator():

        example = tf.train.Example()

        example.ParseFromString(string_record)

        img_dims = (33, 33)

        example_diff_imgs = []

        for img_feature in diff_img_feature_keys:

            example_img_feature = tf.reshape(
                tf.io.parse_tensor(
                    example.features.feature[img_feature].bytes_list.value[0],
                    tf.float32,
                ),
                img_dims,
            ).numpy()
            example_diff_imgs.append(example_img_feature)

        diff_imgs.append(example_diff_imgs)

    return diff_imgs


def get_random_examples(tfrec_fp, example_num=10):
    # dataset = tf.data.TFRecordDataset(tfrec_fp)
    # dataset = dataset.shuffle(buffer_size=buffer_size)
    # dataset = dataset.map(parse_fn)
    # return next(iter(dataset))
    raw_dataset = tf.data.TFRecordDataset(tfrec_fp)
    total_examples = sum(1 for _ in raw_dataset)

    print(f"Found {total_examples} in shard")
    if total_examples == 0:
        print(f"Plotting no examples for {tfrec_fp}")
        return []

    random_indices = sorted(random.sample(range(total_examples), example_num))
    print(f"Using random_indices: {random_indices}")

    # Reinitialize dataset, consumed during first iteration
    raw_dataset = tf.data.TFRecordDataset(tfrec_fp)

    examples = []
    idx_iter = iter(random_indices)
    current_target = next(idx_iter, None)

    for i, example in enumerate(raw_dataset):
        if i == current_target:
            examples.append(example)
            current_target = next(idx_iter, None)
        if current_target is None:  # Exhausted Random Indices iter
            break

    # if total_examples == 0:
    #     return None
    # rand_index = random.randint(0, total_examples - 1)
    # return next(iter(tf.data.TFRecordDataset(tfrec_fp).skip(rand_index)))
    return examples


# def parse_fn(example_proto):
#     feature_description = {
#         'diff_img': tf.io.FixedLenFeature([], tf.)
#     }
#     return tf.io.parse_single_example(example_proto, feature_description)


# def get_flux_curves_from_shard(
#     dest_tfrec_fp: Path, flux_feature_key: str = "flux"
# ) -> list[np.array]:
#     flux_curves = []

#     # Load source dataset
#     src_tfrecord_dataset = tf.data.TFRecordDataset(dest_tfrec_fp)

#     for string_record in src_tfrecord_dataset.as_numpy_iterator():

#         example = tf.train.Example()

#         example.ParseFromString(string_record)

#         # normalize flux window
#         example_flux_feature = example.features.feature[
#             flux_feature_key
#         ].float_list.value

#         flux_curves.append(example_flux_feature)

#     return flux_curves


def plot_examples_from_shard_inorder(
    tfrec_fp: Path | str, plot_dir: Path | str, num_examples: int = 1
) -> None:
    raw_dataset = tf.data.TFRecordDataset([tfrec_fp])

    try:
        print(f"Processing {tfrec_fp}")

        norm_flux_window = []
        norm_diff_imgs = []
        unnorm_flux_window = []

        unnorm_diff_imgs = []

        label = None
        disposition = None
        uid = None
        period = None
        snr = None
        timestamp = None

        # random_raw_record = get_random_example(tfrec_fp=tfrec_fp)
        random_raw_records = get_random_examples(tfrec_fp=tfrec_fp)

        if random_raw_records != []:
            for random_raw_record in random_raw_records:

                print(f"Successfully got random_raw_record from {tfrec_fp}")
                # for example_i, raw_record in enumerate(raw_dataset.take(num_examples), start=1):
                example = tf.train.Example()
                example.ParseFromString(random_raw_record.numpy())

                # Get flux_window
                norm_flux_window = example.features.feature[
                    "flux_norm"
                ].float_list.value
                unnorm_flux_window = example.features.feature["flux"].float_list.value

                for img_feature in ["diff_img", "oot_img", "snr_img", "target_img"]:
                    example_img_feature = tf.reshape(
                        tf.io.parse_tensor(
                            example.features.feature[img_feature].bytes_list.value[0],
                            tf.float32,
                        ),
                        (33, 33),
                    ).numpy()
                    unnorm_diff_imgs.append(example_img_feature)

                for img_feature in [
                    "diff_img_stdnorm",
                    "oot_img_stdnorm",
                    "snr_img_stdnorm",
                    "target_img",
                ]:

                    example_img_feature = tf.reshape(
                        tf.io.parse_tensor(
                            example.features.feature[img_feature].bytes_list.value[0],
                            tf.float32,
                        ),
                        (33, 33),
                    ).numpy()
                    norm_diff_imgs.append(example_img_feature)

                label = (
                    example.features.feature["label"]
                    .bytes_list.value[0]
                    .decode("utf-8")
                )

                disposition = (
                    example.features.feature["disposition"]
                    .bytes_list.value[0]
                    .decode("utf-8")
                )
                uid = (
                    example.features.feature["uid"].bytes_list.value[0].decode("utf-8")
                )
                period = example.features.feature["tce_period"].float_list.value[0]
                snr = example.features.feature["tce_model_snr"].float_list.value[0]
                timestamp = example.features.feature["t"].float_list.value[0]

                unnorm_save_dir = Path(plot_dir) / disposition / str(uid)
                norm_save_dir = Path(plot_dir) / disposition / str(uid)

                unnorm_save_dir.mkdir(parents=True, exist_ok=True)
                norm_save_dir.mkdir(parents=True, exist_ok=True)

                unnorm_save_fp = unnorm_save_dir / (
                    "RAW_" + str(label) + "_" + str(Path(tfrec_fp).name) + ".png"
                )
                norm_save_fp = norm_save_dir / (
                    "NORM_" + str(label) + "_" + str(Path(tfrec_fp).name) + ".png"
                )

                # Plot Raw Data
                plot_flux_and_diff_img_data(
                    flux_curve=unnorm_flux_window,
                    diff_img=unnorm_diff_imgs[0],
                    oot_img=unnorm_diff_imgs[1],
                    snr_img=unnorm_diff_imgs[2],
                    target_img=unnorm_diff_imgs[3],
                    save_fp=unnorm_save_fp,
                    norm=False,
                    snr=snr,
                    midpoint=timestamp,
                )

                # Plot Norm Data
                plot_flux_and_diff_img_data(
                    flux_curve=norm_flux_window,
                    diff_img=norm_diff_imgs[0],
                    oot_img=norm_diff_imgs[1],
                    snr_img=norm_diff_imgs[2],
                    target_img=norm_diff_imgs[3],
                    save_fp=norm_save_fp,
                    norm=True,
                    snr=snr,
                    midpoint=timestamp,
                )
                print(
                    f"Finished plotting norm data to {norm_save_fp} and raw data to {unnorm_save_fp}\n"
                )
        else:
            print(f"Could not extract random example from {tfrec_fp}")
    except Exception as e:
        print(f"ERROR with example: {e}, tb: {traceback.print_exc()}")


if __name__ == "__main__":
    random.seed(42)

    plot_dir = Path(
        f"/nobackupp27/jochoa4/work_dir/data/plots/plot_dataset_examples_05-04-2025_split_norm/"
    )
    plot_dir.mkdir(parents=True, exist_ok=True)

    print(f"Processing splits: ")

    for split in ["train", "test", "val"]:
        print(f"Processing split {split}")

        tfrec_pattern = f"/nobackupp27/jochoa4/work_dir/data/datasets/TESS_exoplanet_dataset_05-04-2025_split_norm/tfrecords/{split}/norm_{split}_shard_00??-????.tfrecord"
        print(f"Using pattern: {tfrec_pattern}")

        tfrec_fps = glob.glob(tfrec_pattern)
        print(f"Found fps: {len(tfrec_fps)}")

        """
        get all tfrec_fps
        select 1 example per shard (take first is fine for now) (print type so you can filter after)
        plot 
        
        """
        split_plot_dir = plot_dir / f"{split}"
        split_plot_dir.mkdir(parents=True, exist_ok=True)

        for tfrec_fp in tfrec_fps:
            try: 
                print(f"Processing tfrec_fp: {tfrec_fp}")
                plot_examples_from_shard_inorder(
                    tfrec_fp=tfrec_fp, plot_dir=split_plot_dir, num_examples=1
                )
            except Exception as e:
                print(f"ERROR: {e}")
                continue
