"""
Utilities for plotting related to transit_detection dataset examples
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from pathlib import Path

from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm


def plot_ex_flux_window_and_diff_imgs(
    flux_window,
    diff_imgs,
    midpoint,
    is_transit_ex,
    plot_fp,
    norm=False,
    snr=None,
):
    """
    Plot flux and difference image data corresponding to a transit dataset example.

    Args:
        flux: NumPy array, flux window
        diff_imgs: list/tuple of NumPy arrays, in order: [diff_img,oot_img,snr_img,target_img]
        midpoint: float, time t corresponding to the midpoint of the flux window example
        is_transit_ex: boolean, True if example corresponds to a transit
        plot_fp: Path, file path to saved plot
        norm: bool, changes plot titles and diff_img plot scales
        snr: float, coresponding to tce_model_snr

    Returns:
        None
    """

    (diff_img, oot_img, snr_img, target_img) = diff_imgs

    label = "In Transit" if is_transit_ex else "Out of Transit"

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
    ax.plot(time, flux_window, linestyle="None", marker="o", markersize=3, alpha=0.6)
    ax.set_title(
        f"{label} Flux Window w/ midpoint {round(midpoint,2) if midpoint else None} Over Time"
    )
    ax.set_xlabel("Time")
    ax.set_ylabel("Flux value") if not norm else ax.set_ylabel("Normalized Flux value")

    # diff imgs
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
        plt.savefig(plot_fp)
        plt.close()

        print(f"Saved plots figures to {plot_fp}")
    except Exception as e:
        print(f"ERROR: plotting: {e}")
