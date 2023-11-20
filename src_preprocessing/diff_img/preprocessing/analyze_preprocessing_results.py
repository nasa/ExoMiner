"""
Analyze preprocessing results.
"""

# 3rd party
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import ast
from mpl_toolkits.axes_grid1 import make_axes_locatable

# local
from src_preprocessing.diff_img.preprocessing.utils_diff_img import plot_diff_img_data

#%% Load results

save_dir = Path('/Users/msaragoc/Projects/exoplanet_transit_classification/data/fits_files/kepler/q1_q17_dr25/dv/preprocessing_step2/09-07-2023_1515')
info_tbl = pd.read_csv('/Users/msaragoc/Projects/exoplanet_transit_classification/data/fits_files/kepler/q1_q17_dr25/dv/preprocessing_step2/09-07-2023_1515/info_tces.csv',
                       # converters={'sampled_qmetrics': np.array}
                       )
info_tbl['sampled_qmetrics'] = info_tbl['sampled_qmetrics'].apply(lambda x: ast.literal_eval(x) if 'nan' not in x else np.nan * np.ones(5))

#%% Count negative OOT images

info_tbl['neg_oot_imgs_cnts'] = info_tbl['oot_negative_values'].apply(lambda x: str(x).count('1'))

bins = np.arange(0, 17 + 1)

f, ax = plt.subplots()
ax.hist(info_tbl['neg_oot_imgs_cnts'], bins, edgecolor='k')
ax.set_xlabel('Number of quarters')
ax.set_ylabel('Counts (TCEs)')
ax.set_yscale('log')
ax.set_xlim(bins[[0, -1]])
ax.set_xticks(bins)
ax.set_title('Quarters with OOT images with negative values')
f.savefig(save_dir / 'hist_neg_oot_imgs_cnts.svg')

for cat in info_tbl['label'].unique():

    f, ax = plt.subplots()
    ax.hist(info_tbl.loc[info_tbl['label'] == cat, 'neg_oot_imgs_cnts'], bins, edgecolor='k')
    ax.set_xlabel('Number of quarters')
    ax.set_ylabel('Counts (TCEs)')
    ax.set_yscale('log')
    ax.set_xlim(bins[[0, -1]])
    ax.set_xticks(bins)
    ax.set_title(f'Quarters with OOT images with negative values\nCategory: {cat}')
    f.savefig(save_dir / f'hist_neg_oot_imgs_cnts_cat_{cat}.svg')

#%% Count valid quarters

info_tbl['valid_qs_cnts'] = info_tbl['valid_qs'].apply(lambda x: str(x).count('1'))

bins = np.arange(0, 17 + 1)

f, ax = plt.subplots()
ax.hist(info_tbl['valid_qs_cnts'], bins, edgecolor='k')
ax.set_xlabel('Number of quarters')
ax.set_ylabel('Counts (TCEs)')
ax.set_yscale('log')
ax.set_xlim(bins[[0, -1]])
ax.set_xticks(bins)
ax.set_title('Number of Valid Quarters')
f.savefig(save_dir / 'hist_valid_qs_cnts.svg')

for cat in info_tbl['label'].unique():

    f, ax = plt.subplots()
    ax.hist(info_tbl.loc[info_tbl['label'] == cat, 'valid_qs_cnts'], bins, edgecolor='k')
    ax.set_xlabel('Number of Valid Quarters')
    ax.set_ylabel('Counts (TCEs)')
    ax.set_yscale('log')
    ax.set_xlim(bins[[0, -1]])
    ax.set_xticks(bins)
    ax.set_title(f'Category: {cat}')
    f.savefig(save_dir / f'hist_valid_qs_cnts_cat_{cat}.svg')

#%% Check average and std sampled quality metrics

info_tbl['mean_qm'] = info_tbl['sampled_qmetrics'].apply(lambda x: np.nanmean(x))
info_tbl['std_qm'] = info_tbl['sampled_qmetrics'].apply(lambda x: np.nanstd(x, ddof=1))

bins = np.linspace(-1, 1, 21, endpoint=True)

f, ax = plt.subplots(figsize=(12, 8))
ax.hist(info_tbl['mean_qm'], bins, edgecolor='k')
ax.set_xlabel('Mean Sampled Quality Metric')
ax.set_ylabel('Counts (TCEs)')
ax.set_yscale('log')
ax.set_xlim(bins[[0, -1]])
ax.set_xticks(bins)
f.savefig(save_dir / 'hist_mean_sampled_qs_cnts.svg')

for cat in info_tbl['label'].unique():

    f, ax = plt.subplots(figsize=(12, 8))
    ax.hist(info_tbl.loc[info_tbl['label'] == cat, 'mean_qm'], bins, edgecolor='k')
    ax.set_xlabel('Mean Sampled Quality Metric')
    ax.set_ylabel('Counts (TCEs)')
    ax.set_yscale('log')
    ax.set_xlim(bins[[0, -1]])
    ax.set_xticks(bins)
    ax.set_title(f'Category: {cat}')
    f.savefig(save_dir / f'hist_mean_sampled_qs_cnts_cat_{cat}.svg')

bins_pos = [np.round((bin_e - bin_s) / 2 + bin_s, 2) for bin_s, bin_e in zip(bins[:-1], bins[1:])]
for cat in info_tbl['label'].unique():

    data_per_bin = []
    for bin_s, bin_e in zip(bins[:-1], bins[1:]):
        data_per_bin.append(info_tbl.loc[((info_tbl['mean_qm'] >= bin_s) & (info_tbl['mean_qm'] < bin_e) &
                                          (info_tbl['label'] == cat)), 'std_qm'])

    f, ax = plt.subplots(figsize=(12, 8))
    ax.boxplot(data_per_bin)  # , positions=bins_pos)
    ax.set_xlabel('Std Sampled Quality Metric')
    ax.set_ylabel('Counts (TCEs)')
    ax.set_title(f'Category: {cat}')
    ax.set_xticklabels(bins_pos)
    f.savefig(save_dir / f'hist_std_sampled_qs_cnts_cat_{cat}.svg')

#%% Count TCEs as function of target star magnitude

mag_thr = 12
bins = np.linspace(0, 20, 21, endpoint=True)

f, ax = plt.subplots(figsize=(12, 8))
cnts, _, _ = ax.hist(info_tbl['mag'], bins, edgecolor='k')
ax.vlines(x=mag_thr, ymin=0, ymax=np.max(cnts), color='r', linestyles='--')
ax.set_xlabel('Target Magnitude')
ax.set_ylabel('Counts (TCEs)')
ax.set_yscale('log')
ax.set_xlim(bins[[0, -1]])
ax.set_xticks(bins)
f.savefig(save_dir / f'hist_mag.svg')

for cat in info_tbl['label'].unique():

    f, ax = plt.subplots(figsize=(12, 8))
    cnts, _, _ = ax.hist(info_tbl.loc[info_tbl['label'] == cat, 'mag'], bins, edgecolor='k')
    ax.vlines(x=mag_thr, ymin=0, ymax=np.max(cnts), color='r', linestyles='--')
    ax.set_xlabel('Target Magnitude')
    ax.set_ylabel('Counts (TCEs)')
    ax.set_yscale('log')
    ax.set_xlim(bins[[0, -1]])
    ax.set_xticks(bins)
    ax.set_title(f'Category: {cat}')
    f.savefig(save_dir / f'hist_mag_cat_{cat}.svg')
#%% Check processed image data vs extracted

diff_img_fp = Path('/Users/msaragoc/Projects/exoplanet_transit_classification/data/fits_files/kepler/q1_q17_dr25/dv/preprocessing_step2/test_11-07-2023_1152/diffimg_preprocess.npy')
diff_img = np.load(diff_img_fp, allow_pickle=True).item()
plot_dir = diff_img_fp.parent / 'plots_examples'
plot_dir.mkdir(exist_ok=True)

diff_img_ext = np.load('/Users/msaragoc/Projects/exoplanet_transit_classification/data/fits_files/kepler/q1_q17_dr25/dv/preprocessing/8-17-2022_1205/keplerq1q17_dr25_diffimg.npy', allow_pickle=True).item()

#%%

tces_uids = ['5130369-1']  # '3446451-1', '3246984-1', '11465813-1', '5130369-1']

for tce_uid in tces_uids:

    # check preprocessed data
    for img_i in range(len(diff_img[tce_uid]['preprocessed_data']['images']['diff_imgs'])):

        f, ax = plt.subplots(1, 3, figsize=(12, 8))
        im = ax[0].imshow(diff_img[tce_uid]['preprocessed_data']['images']['diff_imgs'][img_i])
        divider = make_axes_locatable(ax[0])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        ax[0].set_ylabel('Row')
        ax[0].set_xlabel('Col')
        ax[0].set_title('Difference image (e-/cadence)')
        # ax[0].scatter(diff_img[tce_uid]['cropped_imgs']['x'][img_i],
        #               diff_img[tce_uid]['cropped_imgs']['y'][img_i],
        #               marker='x',
        #               color='r',
        #               label='Target')

        im = ax[1].imshow(diff_img[tce_uid]['preprocessed_data']['images']['oot_imgs'][img_i])
        divider = make_axes_locatable(ax[1])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        ax[1].set_ylabel('Row')
        ax[1].set_xlabel('Col')
        ax[1].set_title('Out-of-transit image (e-/cadence)')
        # ax[1].scatter(diff_img[tce_uid]['cropped_imgs']['x'][img_i],
        #               diff_img[tce_uid]['cropped_imgs']['y'][img_i],
        #               marker='x',
        #               color='r',
        #               label='Target')

        ax[2].imshow(diff_img[tce_uid]['preprocessed_data']['images']['target_imgs'][img_i])
        ax[2].set_ylabel('Row')
        ax[2].set_xlabel('Col')
        ax[2].set_title('Target pixel image (mask)')

        f.suptitle(f'TCE {tce_uid}\nQuality Metric = {diff_img[tce_uid]["preprocessed_data"]["quality"][img_i]}\n'
                   f'Quarter {diff_img[tce_uid]["preprocessed_data"]["imgs_numbers"][img_i]}\n'
                   f'Subpixel target location: {diff_img[tce_uid]["preprocessed_data"]["target_position"]["subpixel_x"][img_i]:.4f}, '
                   f'{diff_img[tce_uid]["preprocessed_data"]["target_position"]["subpixel_y"][img_i]:.4f}')
        f.tight_layout()
        f.savefig(plot_dir / f'tce_{tce_uid}_q{diff_img[tce_uid]["preprocessed_data"]["imgs_numbers"][img_i]}_preprocessed.svg')
        plt.close()

    # check extracted difference image data
    for img_i in range(len(diff_img_ext[tce_uid]['image_data'])):

        f, ax = plt.subplots(1, 2, figsize=(12, 8))
        im = ax[0].imshow(diff_img_ext[tce_uid]['image_data'][img_i][:, :, 2, 0])
        divider = make_axes_locatable(ax[0])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        ax[0].set_ylabel('Row')
        ax[0].set_xlabel('Col')
        ax[0].set_title('Difference image (e-/cadence)')
        ax[0].scatter(diff_img_ext[tce_uid]['target_ref_centroid'][img_i]['col']['value'],
                      diff_img_ext[tce_uid]['target_ref_centroid'][img_i]['row']['value'],
                      marker='x',
                      color='r',
                      label='Target')

        im = ax[1].imshow(diff_img_ext[tce_uid]['image_data'][img_i][:, :, 1, 0])
        divider = make_axes_locatable(ax[1])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        ax[1].set_ylabel('Row')
        ax[1].set_xlabel('Col')
        ax[1].set_title('Out-of-transit image (e-/cadence)')
        ax[1].scatter(diff_img_ext[tce_uid]['target_ref_centroid'][img_i]['col']['value'],
                      diff_img_ext[tce_uid]['target_ref_centroid'][img_i]['row']['value'],
                      marker='x',
                      color='r',
                      label='Target')

        f.suptitle(f'TCE {tce_uid}\n'
                   f'Pixel target location: {diff_img_ext[tce_uid]["target_ref_centroid"][img_i]["col"]["value"]:.4f}, '
                   f'{diff_img_ext[tce_uid]["target_ref_centroid"][img_i]["row"]["value"]:.4f}')
        f.tight_layout()
        f.savefig(plot_dir / f'tce_{tce_uid}_{img_i}_extracted.svg')
        plt.close()


#%% Check preprocessed data

preproc_data_fp = Path('/Users/msaragoc/Projects/exoplanet_transit_classification/data/fits_files/kepler/q1_q17_dr25/dv/preprocessing_step2/11-17-2023_1205/diffimg_preprocess.npy')
preproc_data = np.load(preproc_data_fp, allow_pickle=True).item()

#%% Plot data for examples

plot_dir = Path(preproc_data_fp.parent / 'check_examples')
plot_dir.mkdir(exist_ok=True)
tces = [
    ('10383429-1', 7),
]

for tce, img_n in tces:

    img_n_idx = preproc_data[tce]['images_numbers'].index(img_n)

    plot_diff_img_data(
        preproc_data[tce]['images']['diff_imgs'][img_n_idx],
        preproc_data[tce]['images']['oot_imgs'][img_n_idx],
        preproc_data[tce]['images']['target_imgs'][img_n_idx],
        {
            'x': preproc_data[tce]['target_position']['pixel_x'][img_n_idx],
            'y': preproc_data[tce]['target_position']['pixel_y'][img_n_idx]
         },
        preproc_data[tce]['quality'][img_n_idx],
        f'{img_n}',
        tce,
        plot_dir / f'{tce}_{img_n}.png'
    )
