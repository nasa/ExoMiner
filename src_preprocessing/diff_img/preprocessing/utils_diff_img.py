""" Utility functions for processing difference imaging. """

# 3rd party
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.switch_backend('agg')


def plot_diff_img_data(diff_img, oot_img, target_img, target_coords, qmetric, img_num, uid, save_fp):
    """ Plot difference image data for TCE in a given quarter/sector.

    Args:
        diff_img: NumPy array, difference image
        oot_img: NumPy array, out-of-transit image
        target_img: NumPy array, target location image
        target_coords: dict, target location 'x' and 'y'
        qmetric: float, quality metric
        img_num: str, quarter/sector run
        uid: str, TCE ID
        save_fp: Path, file path to saved plot

    Returns:

    """

    f, ax = plt.subplots(1, 3, figsize=(16, 8))
    # diff img
    im = ax[0].imshow(diff_img)
    divider = make_axes_locatable(ax[0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    ax[0].scatter(target_coords['y'], target_coords['x'],
                     marker='x',
                     color='r', label='Target')
    ax[0].set_ylabel('Row')
    ax[0].set_xlabel('Col')
    ax[0].legend()
    ax[0].set_title('Difference Flux (e-/cadence)')
    # oot img
    im = ax[1].imshow(oot_img)
    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    ax[1].scatter(target_coords['y'], target_coords['x'],
                     marker='x',
                     color='r', label='Target')
    ax[1].set_ylabel('Row')
    ax[1].set_xlabel('Col')
    ax[1].legend()
    ax[1].set_title('Out-of-Transit Flux (e-/cadence)')
    # target img
    ax[2].imshow(target_img)
    ax[2].set_ylabel('Row')
    ax[2].set_xlabel('Col')
    ax[2].set_title('Target Position')

    f.suptitle(f'TCE {uid} {img_num} Quality Metric: {qmetric}')
    f.tight_layout()
    f.savefig(save_fp)
    plt.close()
