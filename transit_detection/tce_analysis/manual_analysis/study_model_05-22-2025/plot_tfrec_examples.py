import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import numpy as np
from pathlib import Path
import tensorflow as tf
from collections import defaultdict
import glob

from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm


def plot_diff_img_data(
    diff_img, oot_img, snr_img, target_img, save_fp, label="", logscale=True
):
    """Plot difference image data for TCE in a given quarter/sector.

    Args:
        diff_img: NumPy array, difference image
        oot_img: NumPy array, out-of-transit image
        snr_img: NumPy array, SNR image
        target_img: NumPy array, target location image
        target_coords: dict, target location 'x' and 'y'
        qmetric: float, quality metric
        img_num: str, quarter/sector run
        uid: str, TCE ID
        save_fp: Path, file path to saved plot
        logscale: bool, if True images color is set to log scale

    Returns:

    """

    target_coords = {"x": diff_img.shape[1] / 2, "y": diff_img.shape[0] / 2}

    f, ax = plt.subplots(2, 2, figsize=(16, 8))
    # diff img
    im = ax[0, 0].imshow(diff_img, norm=LogNorm() if logscale else None)
    divider = make_axes_locatable(ax[0, 0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    ax[0, 0].scatter(
        target_coords["y"], target_coords["x"], marker="x", color="r", label="Target"
    )
    ax[0, 0].set_ylabel("Row")
    ax[0, 0].set_xlabel("Col")
    ax[0, 0].legend()
    ax[0, 0].set_title(f"{label} Difference Flux (e-/cadence)")
    # oot img
    im = ax[0, 1].imshow(oot_img, norm=LogNorm() if logscale else None)
    divider = make_axes_locatable(ax[0, 1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    ax[0, 1].scatter(
        target_coords["y"], target_coords["x"], marker="x", color="r", label="Target"
    )
    ax[0, 1].set_ylabel("Row")
    ax[0, 1].set_xlabel("Col")
    ax[0, 1].legend()
    ax[0, 1].set_title("Out-of-Transit Flux (e-/cadence)")
    # target img
    ax[1, 0].imshow(target_img)
    ax[1, 0].set_ylabel("Row")
    ax[1, 0].set_xlabel("Col")
    ax[1, 0].set_title("Target Position")
    # snr img
    im = ax[1, 1].imshow(snr_img, norm=LogNorm() if logscale else None)
    divider = make_axes_locatable(ax[1, 1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    ax[1, 1].scatter(
        target_coords["y"], target_coords["x"], marker="x", color="r", label="Target"
    )
    ax[1, 1].set_ylabel("Row")
    ax[1, 1].set_xlabel("Col")
    ax[1, 1].legend()
    ax[1, 1].set_title("Difference SNR")

    f.tight_layout()
    f.savefig(save_fp)
    plt.close()


def plot_flux_and_diff_img_data(
    flux_curve,
    diff_img,
    oot_img,
    snr_img,
    target_img,
    save_fp,
    label="",
    norm=False,
    logscale=True,
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
        logscale: bool, if True images color is set to log scale

    Returns:

    """
    label = "In Transit" if label == 1.0 else "Out of Transit"
    snr = str(round(snr, 2))
    it_idx = np.argmax(target_img)
    row, col = np.unravel_index(
        it_idx, target_img.shape
    )  # Get row and col as if array were flattened
    target_coords = {"x": row, "y": col}

    fig = plt.figure(figsize=(16, 16))

    gs = gridspec.GridSpec(3, 2, figure=fig)

    ax = fig.add_subplot(gs[0, :])
    time = np.linspace(0, 100, 100)
    ax.plot(time, flux_curve, "b-", linewidth=2, marker="o", markersize=3, alpha=0.6)
    ax.set_title(f"{label} Flux Curve w/ midpoint {round(midpoint,2)} Over Time")
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
    plt.savefig(save_fp)
    plt.close()


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


def get_flux_curves_from_shard(
    dest_tfrec_fp: Path, flux_feature_key: str = "flux"
) -> list[np.array]:
    flux_curves = []

    # Load source dataset
    src_tfrecord_dataset = tf.data.TFRecordDataset(dest_tfrec_fp)

    for string_record in src_tfrecord_dataset.as_numpy_iterator():

        example = tf.train.Example()

        example.ParseFromString(string_record)

        # normalize flux window
        example_flux_feature = example.features.feature[
            flux_feature_key
        ].float_list.value

        flux_curves.append(example_flux_feature)

    return flux_curves


def get_timestamps_from_shard(
    dest_tfrec_fp: Path, timestamp_feature_key: str = "t"
) -> list[np.array]:
    timestamps = []

    # Load source dataset
    src_tfrecord_dataset = tf.data.TFRecordDataset(dest_tfrec_fp)

    for string_record in src_tfrecord_dataset.as_numpy_iterator():

        example = tf.train.Example()

        example.ParseFromString(string_record)

        # normalize flux window
        example_flux_feature = example.features.feature[
            timestamp_feature_key
        ].float_list.value

        timestamps.append(example_flux_feature)

    return timestamps


def get_label_from_shard(
    dest_tfrec_fp: Path, label_feature_key: str = "label"
) -> list[np.array]:
    labels = []

    # Load source dataset
    src_tfrecord_dataset = tf.data.TFRecordDataset(dest_tfrec_fp)

    for string_record in src_tfrecord_dataset.as_numpy_iterator():

        example = tf.train.Example()

        example.ParseFromString(string_record)

        example_label_feature = example.features.feature[
            label_feature_key
        ].float_list.value[0]

        labels.append(example_label_feature)

    return labels


def get_uid_from_shard(
    dest_tfrec_fp: Path, uid_feature_key: str = "uid"
) -> list[np.array]:
    uids = []

    # Load source dataset
    src_tfrecord_dataset = tf.data.TFRecordDataset(dest_tfrec_fp)

    for string_record in src_tfrecord_dataset.as_numpy_iterator():

        example = tf.train.Example()

        example.ParseFromString(string_record)

        example_disposition_feature = (
            example.features.feature[uid_feature_key]
            .bytes_list.value[0]
            .decode("utf-8")
        )

        uids.append(example_disposition_feature)

    return uids


def get_snr_from_shard(
    dest_tfrec_fp: Path, snr_feature_key: str = "tce_model_snr"
) -> list[np.array]:
    snrs = []

    # Load source dataset
    src_tfrecord_dataset = tf.data.TFRecordDataset(dest_tfrec_fp)

    for string_record in src_tfrecord_dataset.as_numpy_iterator():

        example = tf.train.Example()

        example.ParseFromString(string_record)

        example_disposition_feature = example.features.feature[
            snr_feature_key
        ].float_list.value[0]

        snrs.append(example_disposition_feature)

    return snrs


def get_disposition_from_shard(
    dest_tfrec_fp: Path, disposition_feature_key: str = "disposition"
) -> list[np.array]:
    dispositions = []

    # Load source dataset
    src_tfrecord_dataset = tf.data.TFRecordDataset(dest_tfrec_fp)

    for string_record in src_tfrecord_dataset.as_numpy_iterator():

        example = tf.train.Example()

        example.ParseFromString(string_record)

        example_disposition_feature = (
            example.features.feature[disposition_feature_key]
            .bytes_list.value[0]
            .decode("utf-8")
        )

        dispositions.append(example_disposition_feature)

    return dispositions


def get_tce_period_from_shard(
    dest_tfrec_fp: Path, tce_period_feature_key: str = "tce_period"
) -> list[np.array]:
    tce_periods = []

    # Load source dataset
    src_tfrecord_dataset = tf.data.TFRecordDataset(dest_tfrec_fp)

    for string_record in src_tfrecord_dataset.as_numpy_iterator():

        example = tf.train.Example()

        example.ParseFromString(string_record)

        example_disposition_feature = example.features.feature[
            tce_period_feature_key
        ].float_list.value[0]

        tce_periods.append(example_disposition_feature)

    return tce_periods


# def get_tce_std_from_shard(
#     dest_tfrec_fp: Path, tce_std_feature_key: str = "std"
# ) -> list[np.array]:
#     tce_stds = []

#     # Load source dataset
#     src_tfrecord_dataset = tf.data.TFRecordDataset(dest_tfrec_fp)

#     for string_record in src_tfrecord_dataset.as_numpy_iterator():

#         example = tf.train.Example()

#         example.ParseFromString(string_record)

#         example_std_feature = example.features.feature[
#             tce_std_feature_key
#         ].float_list.value[0]

#         tce_stds.append(example_std_feature)

#     return tce_stds


if __name__ == "__main__":

    tfrec_dir = Path(
        f"/Users/jochoa4/Desktop/study_transfers/study_model_preds_05-22-2025/tfrecords/"
    )

    tfrec_fp = tfrec_dir / "norm_train_shard_2990-8611.tfrecord"

    plot_dir = Path(
        "/Users/jochoa4/Desktop/studies/study_model_preds_05-22-2025/plots/"
    )

    tfrec_plot_dir = plot_dir / tfrec_fp.name

    tfrec_fp = Path(tfrec_fp)

    tfrec_plot_dir = Path(
        f"/Users/jochoa4/Desktop/studies/study_model_preds_05-22-2025/plots/{tfrec_fp.name}"
    )

    plot_dir.mkdir(parents=True, exist_ok=True)

    flux_windows = get_flux_curves_from_shard(tfrec_fp, flux_feature_key="flux")

    norm_flux_windows = get_flux_curves_from_shard(
        tfrec_fp, flux_feature_key="flux_norm"
    )

    diff_imgs = get_diff_imgs_from_shard(
        tfrec_fp, ["diff_img", "oot_img", "snr_img", "target_img"]
    )

    norm_diff_imgs = get_diff_imgs_from_shard(
        tfrec_fp,
        ["diff_img_stdnorm", "oot_img_stdnorm", "snr_img_stdnorm", "target_img"],
    )

    labels = get_label_from_shard(tfrec_fp, label_feature_key="label")

    dispositions = get_disposition_from_shard(
        tfrec_fp, disposition_feature_key="disposition"
    )

    timestamps = get_timestamps_from_shard(tfrec_fp, timestamp_feature_key="t")

    uids = get_uid_from_shard(tfrec_fp, uid_feature_key="uid")

    periods = get_tce_period_from_shard(tfrec_fp, tce_period_feature_key="tce_period")

    snrs = get_snr_from_shard(tfrec_fp, snr_feature_key="tce_model_snr")

    # stds = get_tce_std_from_shard(tfrec_fp, tce_std_feature_key="std")

    disposition_counter = defaultdict(int)
    uid_counter = defaultdict(int)
    label_counter = defaultdict(int)

    for i, (
        label,
        disposition,
        norm_flux_window,
        flux_window,
        diff_img,
        norm_diff_img,
        uid,
        snr,
        timestamp,
        # std,
    ) in enumerate(
        zip(
            labels,
            dispositions,
            norm_flux_windows,
            flux_windows,
            diff_imgs,
            norm_diff_imgs,
            uids,
            snrs,
            timestamps,
            # stds,
        )
    ):
        timestamp = timestamp[0]
        uid = uid.split("_")[0]

        if (
            disposition_counter[disposition] < 3
            and uid_counter[uid] < 2
            # and label_counter[uid] < 2
        ):
            disposition_counter[disposition] += 1
            uid_counter[uid] += 1
            # label_counter[uid] += 1

            ex_plot_dir = plot_dir / str(disposition) / str(uid)
            ex_plot_dir.mkdir(parents=True, exist_ok=True)

            print(
                f"{tfrec_fp.name}: {disposition}: {uids[i]}, period: {periods[i]}, ex_{i} timestamp: {timestamps[i]}"
            )
            print()

            # Plot unnormalized data example
            plot_fp = ex_plot_dir / f"tce_{uid}_{disposition}_{label}.png"

            plot_flux_and_diff_img_data(
                flux_window,
                *diff_img,
                save_fp=plot_fp,
                norm=False,
                label=label,
                snr=snr,
                midpoint=timestamp,
            )

            # Plot normalized data example
            plot_fp = ex_plot_dir / f"tce_{uid}_{disposition}_norm_{label}.png"
            plot_flux_and_diff_img_data(
                norm_flux_window,
                *norm_diff_img,
                save_fp=plot_fp,
                norm=True,
                label=label,
                snr=snr,
                midpoint=timestamp,
            )
