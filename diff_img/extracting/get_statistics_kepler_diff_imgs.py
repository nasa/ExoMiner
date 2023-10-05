""" Compute statistics for the difference image data in Kepler. """

# 3rd party
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from mpl_toolkits.axes_grid1 import make_axes_locatable

# from astropy.stats import mad_std

plt.switch_backend('agg')

# %% Choose processing experiment

run_dir = Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/data/fits_files/kepler/q1_q17_dr25/dv/preprocessing/8-17-2022_1205')

# %% Load difference image data

data = np.load(run_dir / 'keplerq1q17_dr25_diffimg.npy', allow_pickle=True).item()

# %% Get number of valid quarters per TCE

tce_quarters = {}
for tce_uid, tce_data in data.items():
    n_quarters_valid = np.sum([1 if target_ref_q['col']['uncertainty'] != -1 else 0
                               for target_ref_q in tce_data['target_ref_centroid']])
    tce_quarters[tce_uid] = n_quarters_valid

# %% Store values into csv file

# quarter_df = pd.DataFrame({'tce_uid': list(tce_quarters.keys()), 'n_valid_quarters': list(tce_quarters.values())})
# quarter_df.to_csv(run_dir / 'tces_n_valid_quarters.csv', index=False)
quarter_df = pd.read_csv(run_dir / 'tces_n_valid_quarters.csv')

# get statistics
quarter_df_stats = quarter_df.describe(percentiles=[0.25, 0.5, 0.75, 0.90, 0.95, 0.99]).to_csv(
    run_dir / 'tces_n_valid_quarters_stats.csv')

# %% Plot histogram of number of valid quarters per TCE

QUARTER_S, QUARTER_E = 0, 17
quarter_bins = np.arange(QUARTER_S, QUARTER_E + 2)
f, ax = plt.subplots()
# ax.hist(list(tce_quarters.values()), quarter_bins, edgecolor='k', align='left')
ax.hist(quarter_df['n_valid_quarters'], quarter_bins, edgecolor='k', align='left', cumulative=True)
ax.set_xlabel('Number of Valid Quarters')
ax.set_ylabel('Cumulative TCE Count')
ax.set_xticks(quarter_bins[:-1])
ax.set_yscale('log')
f.savefig(run_dir / 'cumhist_validquarters_tces_log.png')
plt.close()

# %% Get average image size per TCE

tce_img_size = {}
for tce_uid, tce_data in data.items():

    n_quarters = len(tce_data['image_data'])
    tce_img_size[tce_uid] = {
        'n_pxs': np.nan * np.ones(n_quarters),
        'n_cols': np.nan * np.ones(n_quarters),
        'n_rows': np.nan * np.ones(n_quarters),
        # 'mean_n_pxs': np.nan,
        # 'mean_n_rows': np.nan,
        # 'mean_n_cols': np.nan
    }
    for q_i, target_img_q in enumerate(tce_data['image_data']):
        if tce_data['target_ref_centroid'][q_i]['col']['uncertainty'] != -1:
            tce_img_size[tce_uid]['n_pxs'][q_i] = np.prod(target_img_q.shape[:2])
            tce_img_size[tce_uid]['n_cols'][q_i] = target_img_q.shape[0]
            tce_img_size[tce_uid]['n_rows'][q_i] = target_img_q.shape[1]

    # tce_img_size[tce_uid] = {k: np.nanmean(v) for k, v in tce_img_size[tce_uid].items()}

# %% Store values into csv file

N_QUARTERS_MAX = 17
columns = ['tce_uid']
fields = ['n_pxs', 'n_cols', 'n_rows']
for field in fields:
    columns += [f'mean_{field}', f'std_{field}']
columns += [f'q_{q_i}' for q_i in range(N_QUARTERS_MAX)]
img_size_dict = {col: [] for col in columns}
for tce_uid in tce_img_size:

    img_size_dict['tce_uid'].append(tce_uid)
    # for k in fields:
    #     img_size_dict[k].append(np.nan)
    for q_i in range(N_QUARTERS_MAX):
        img_size_dict[f'q_{q_i}'].append((np.nan, np.nan))

    for k in fields:
        img_size_dict[f'mean_{k}'].append(np.nanmean(tce_img_size[tce_uid][k]))
        img_size_dict[f'std_{k}'].append(np.nanstd(tce_img_size[tce_uid][k]))

    n_quarters_tce = len(tce_img_size[tce_uid]['n_pxs'])
    for q_i in range(n_quarters_tce):
        img_size_dict[f'q_{q_i}'][-1] = (tce_img_size[tce_uid]['n_rows'][q_i],
                                         tce_img_size[tce_uid]['n_cols'][q_i])

img_size_df = pd.DataFrame(img_size_dict)
img_size_df.to_csv(run_dir / 'tces_img_size.csv', index=False)
img_size_df = pd.read_csv(run_dir / 'tces_img_size.csv')

# get statistics
img_size_df_stats = img_size_df[
    ['mean_n_pxs', 'std_n_pxs', 'mean_n_cols', 'std_n_cols', 'mean_n_rows', 'std_n_rows']].describe(
    percentiles=[0.01, 0.1, 0.25, 0.5, 0.75, 0.90, 0.95, 0.99]).to_csv(run_dir / 'tces_img_size_stats.csv')

# %% Plot histogram of number of average image size per TCE

f, ax = plt.subplots()
ax.hist(img_size_df['mean_n_rows'], np.arange(2, 20), edgecolor='k', align='mid')
ax.set_xlabel('Mean Row Size (px)')
ax.set_ylabel('TCE Count')
# ax.set_xticks(quarter_bins[:-1])
ax.set_yscale('log')
f.savefig(run_dir / 'hist_mean_row_size_tces.png')
plt.close()

f, ax = plt.subplots()
ax.hist(img_size_df['mean_n_cols'], np.arange(2, 20), edgecolor='k', align='mid')
ax.set_xlabel('Mean Col Size (px)')
ax.set_ylabel('TCE Count')
# ax.set_xticks(quarter_bins[:-1])
ax.set_yscale('log')
f.savefig(run_dir / 'hist_mean_col_size_tces.png')
plt.close()

f, ax = plt.subplots()
ax.hist(img_size_df['mean_n_pxs'], np.arange(4, 400, 4), edgecolor='k', align='mid')
ax.set_xlabel('Mean Image Size (px^2)')
ax.set_ylabel('TCE Count')
ax.set_yscale('log')
# ax.set_xticks(quarter_bins[:-1])
f.savefig(run_dir / 'hist_mean_img_size_tces.png')
plt.close()

# %% Plot image size as function of Kepler magnitude

tce_tbl = pd.read_csv(
    '/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17_DR25/11-17-2021_1243/q1_q17_dr25_tce_2020.09.28_10.36.22_stellar_koi_cfp_norobovetterlabels_renamedcols_nomissingval_symsecphase_cpkoiperiod_rba_cnt0n_valpc_modelchisqr_ruwe_magcat_uid.csv',
    usecols=['uid', 'mag'])

img_size_df = pd.read_csv(run_dir / 'tces_img_size.csv').rename(columns={'tce_uid': 'uid'})
img_size_df = img_size_df.merge(tce_tbl, on=['uid'], how='left', validate='one_to_one')

f, ax = plt.subplots()
ax.scatter(img_size_df['mag'], img_size_df['mean_n_pxs'], s=8)
ax.set_ylabel('Mean Image Size (px^2)')
ax.set_xlabel('Kepler Magnitude')
ax.set_yscale('log')
f.savefig(run_dir / 'scatter_kep_mag-mean_img_size.png')

f, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].scatter(img_size_df['mag'], img_size_df['mean_n_cols'], s=8)
ax[0].set_ylabel('Mean Col Size (px)')
# ax[0].set_xscale('log')
ax[0].set_yscale('log')
ax[0].set_xlabel('Kepler Magnitude')
ax[1].scatter(img_size_df['mag'], img_size_df['mean_n_rows'], s=8)
ax[1].set_ylabel('Mean Row Size (px)')
ax[1].set_xlabel('Kepler Magnitude')
ax[1].set_yscale('log')
f.savefig(run_dir / 'scatter_kep_mag-mean_col_row_sizes.png')

# %% Compute average offset from KIC pixel to pixel with maximum value in difference image (transit source location)

# img_final_size = [7, 7]
# max_in_dist = np.sqrt(np.sum([((el - 1) / 2) ** 2 for el in img_final_size]))
N_QUARTERS = 17
diff_src_dist = {}
for tce_uid, tce_data in data.items():
    # if tce_uid != '892376-1':
    #     continue
    # else:
    #     aaaa

    # n_quarters = len(tce_data['image_data'])
    diff_src_dist[tce_uid] = np.nan * np.ones(N_QUARTERS)
    for q_i, target_img_q in enumerate(tce_data['image_data']):
        if tce_data['target_ref_centroid'][q_i]['col']['uncertainty'] != -1:
            # aaa
            target_px_trunc = {coord: int(coord_val['value'])
                               for coord, coord_val in tce_data['target_ref_centroid'][q_i].items()}
            max_diff_px = np.unravel_index(np.argmax(target_img_q[:, :, 2, 0], axis=None),
                                           target_img_q[:, :, 2, 0].shape)
            diff_src_dist[tce_uid][q_i] = np.sqrt(np.sum([(target_px_trunc['row'] - max_diff_px[0]) ** 2,
                                                          (target_px_trunc['col'] - max_diff_px[1]) ** 2]))
            # if np.isnan(diff_src_dist[tce_uid][q_i]):
            #     aaa

            # if dist_target_to_max_diff <= max_in_dist:
            #     diff_src_cropped[tce_uid][q_i] = 1
            # else:
            #     diff_src_cropped[tce_uid][q_i] = 0

    # diff_src_cropped[tce_uid] = (diff_src_cropped[tce_uid] == 1).sum() / (~np.isnan(diff_src_cropped[tce_uid])).sum()
    # diff_src_dist[tce_uid] = np.nanmean(diff_src_dist[tce_uid])
    # if np.isnan(diff_src_dist[tce_uid]) and not np.all([tce_data['target_ref_centroid'][q_i]['col']['uncertainty'] == -1 for q_i in range(n_quarters)]):
    #     aaaa

# %%

diff_src_dist_dict = {'tce_uid': [], 'mean_dist_diff': [], 'std_dist_diff': []}
diff_src_dist_dict.update({f'dist_q_{q_i}': [] for q_i in range(N_QUARTERS)})

for tce_uid, src_dist in diff_src_dist.items():
    diff_src_dist_dict['tce_uid'].append(tce_uid)
    diff_src_dist_dict['mean_dist_diff'].append(np.nanmean(src_dist))
    diff_src_dist_dict['std_dist_diff'].append(np.nanstd(src_dist))
    for q_i in range(N_QUARTERS):
        diff_src_dist_dict[f'dist_q_{q_i}'].append(src_dist[q_i])

diff_src_dist_df = pd.DataFrame(diff_src_dist_dict)
diff_src_dist_df.to_csv(run_dir / f'diff_src_mean_dist.csv', index=False)

# %% Plot histogram of average offset from KIC pixel to pixel with maximum value in difference image (transit source location)

diff_src_dist_df = pd.read_csv(run_dir / 'diff_src_mean_dist.csv')

bins = np.linspace(0, 10, 100)
f, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].hist(diff_src_dist_df['mean_dist_diff'], bins, edgecolor='k')
ax[0].set_xlabel('Average Transit Source Offset from KIC pixel (px)')
ax[0].set_ylabel('TCE Count')
ax[0].set_yscale('log')
ax[1].hist(diff_src_dist_df['mean_dist_diff'], bins, edgecolor='k', cumulative=True)
ax[1].set_yscale('log')
ax[1].set_ylabel('Cumulative TCE Count')
ax[1].set_xlabel('Average Transit Source Offset from KIC pixel (px)')
f.savefig(run_dir / 'hist_diff_src_mean_dist.png')

#%% Plot difference image data

plot_dir = run_dir / 'plots_22-2-2023_1129'
plot_dir.mkdir(exist_ok=True)

tce_tbl = pd.read_csv('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/data/ephemeris_tables/kepler/q1-q17_dr25/11-17-2021_1243/q1_q17_dr25_tce_2020.09.28_10.36.22_stellar_koi_cfp_norobovetterlabels_renamedcols_nomissingval_symsecphase_cpkoiperiod_rba_cnt0n_valpc_modelchisqr_ruwe_magcat_uid_rv_posprob.csv')

n_samples = 200
sampled_examples_ids = ['757099-1']  # np.random.choice(list(data.keys()), n_samples)

for sampled_example_id in sampled_examples_ids:

    sampled_example_data = data[sampled_example_id]

    n_qs_avail = len(sampled_example_data['image_data'])
    q_chosen = 2  # np.random.choice(n_qs_avail, 1)[0]

    f, ax = plt.subplots(1, 2, figsize=(12, 8))
    im = ax[0].imshow(sampled_example_data['image_data'][q_chosen][:, :, 2, 0])
    divider = make_axes_locatable(ax[0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    ax[0].scatter(sampled_example_data['target_ref_centroid'][q_chosen]['col']['value'],
                     sampled_example_data['target_ref_centroid'][q_chosen]['row']['value'],
                     marker='x',
                     color='r',
                     label='Target')
    ax[0].set_ylabel('Row')
    ax[0].set_xlabel('Col')
    # ax[0].legend()
    ax[0].set_title('Difference image (e-/cadence)')
    im = ax[1].imshow(sampled_example_data['image_data'][q_chosen][:, :, 1, 0])
    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    ax[1].scatter(sampled_example_data['target_ref_centroid'][q_chosen]['col']['value'],
                     sampled_example_data['target_ref_centroid'][q_chosen]['row']['value'],
                     marker='x',
                     color='r',
                     label='Target')
    ax[1].set_ylabel('Row')
    ax[1].set_xlabel('Col')
    # ax[1].legend()
    ax[1].set_title('Out-of-transit image (e-/cadence)')
    f.suptitle(f'KIC {sampled_example_id} '
               f'{tce_tbl.loc[tce_tbl["uid"] == sampled_example_id, "label"].values[0]}\nQuarter {q_chosen + 1}')
    f.tight_layout()
    f.savefig(plot_dir / f'{sampled_example_id}_diff_img_q{q_chosen + 1}.png')
    aaa
    plt.close()

#%%


data_dir = Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/data/fits_files/kepler/q1_q17_dr25/dv/preprocessing/8-17-2022_1205/data')

for data_dict_fp in data_dir.iterdir():
    data = np.load(data_dict_fp, allow_pickle=True).item()
    if '757099-1' in data:
        break

#%% Plot histogram of difference image quality metric

qual_metric_tbl = pd.read_csv('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/data/fits_files/kepler/q1_q17_dr25/dv/diff_img_quality_metric.csv')
tce_tbl = pd.read_csv('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/data/ephemeris_tables/kepler/q1-q17_dr25/11-17-2021_1243/q1_q17_dr25_tce_2020.09.28_10.36.22_stellar_koi_cfp_norobovetterlabels_renamedcols_nomissingval_symsecphase_cpkoiperiod_rba_cnt0n_valpc_modelchisqr_ruwe_magcat_uid_rv_posprob.csv')
plot_dir = Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/data/fits_files/kepler/q1_q17_dr25/dv/qual_metric_plots')
plot_dir.mkdir(exist_ok=True)

thr = -1


def _count_valid_thr_quarters(x, thr):

    N_QUARTERS = 17
    q_value_cols = [f'q{q}_value' for q in range(1, N_QUARTERS + 1)]
    q_valid_cols = [f'q{q}_valid' for q in range(1, N_QUARTERS + 1)]

    return ((x[q_value_cols] > thr) & (x[q_valid_cols])).sum()


# qual_metric_tbl['valid_quarters'] = qual_metric_tbl.apply(lambda x: (x[q_value_cols] > thr).sum(), axis=1)
qual_metric_tbl['valid_quarters'] = qual_metric_tbl.apply(_count_valid_thr_quarters, axis=1, args=(thr, ))

qual_metric_tbl = qual_metric_tbl.merge(tce_tbl[['uid', 'label', 'tce_rogue_flag']], on='uid', how='left',
                                        validate='one_to_one')
qual_metric_tbl = qual_metric_tbl.loc[qual_metric_tbl['tce_rogue_flag'] == 0]

# plot quality metric for a single quarter
q_chosen = 12
f, ax = plt.subplots()
ax.hist(qual_metric_tbl.loc[qual_metric_tbl[f'q{q_chosen}_valid'], f'q{q_chosen}_value'],
        bins=np.linspace(-1, 1, 100), edgecolor='k')
ax.set_yscale('log')
ax.set_ylabel('Counts')
ax.set_xlabel('Quality Metric Value')
ax.set_title(f'Q{q_chosen} Kepler Q1-Q17 DR25')

bins = np.linspace(0, N_QUARTERS + 1, N_QUARTERS + 2, endpoint=True, dtype='int')

f, ax = plt.subplots(figsize=(12, 8))
ax.hist(qual_metric_tbl['valid_quarters'], bins=bins, edgecolor='k')
ax.set_xlabel(f'Number of valid difference images')
ax.set_ylabel('Count')
ax.set_xticks(bins[:-1] + 0.5)
ax.set_xticklabels(bins[:-1])
ax.set_xlim(bins[[0, -1]])
ax.set_title(f'All Kepler data\n Quality thr={thr}')
f.tight_layout()
f.savefig(plot_dir / f'hist_all_kepler_tces_valid_diff_imgs_qm_thr{thr}.png')
plt.close()

categories = {
    'PC': {'zorder': 3, 'alpha': 0.5},
    'AFP': {'zorder': 2, 'alpha': 0.8},
    'NTP': {'zorder': 1, 'alpha': 1.0},
    # 'UNK': {'zorder': 2, 'alpha': 0.5},
}
f, ax = plt.subplots(2, 1, figsize=(12, 8))
for lbl in categories:
    ax[0].hist(qual_metric_tbl.loc[qual_metric_tbl['label'] == lbl, 'valid_quarters'], bins=bins,
            edgecolor='k', label=lbl, zorder=categories[lbl]['zorder'], alpha=categories[lbl]['alpha'],
               cumulative=False)
    ax[0].legend()
    ax[0].set_ylabel('Count')
    ax[0].set_xticks(bins[:-1] + 0.5)
    ax[0].set_xticklabels(bins[:-1])
    ax[0].set_xlim(bins[[0, -1]])
    ax[0].set_yscale('log')
    ax[1].hist(qual_metric_tbl.loc[qual_metric_tbl['label'] == lbl, 'valid_quarters'], bins=bins,
               edgecolor='k', label=lbl, zorder=categories[lbl]['zorder'], alpha=categories[lbl]['alpha'],
               cumulative=True)
    ax[1].legend()
    ax[1].set_xticks(bins[:-1] + 0.5)
    ax[1].set_xticklabels(bins[:-1])
    ax[1].set_xlim(bins[[0, -1]])
    ax[1].set_xlabel(f'Number of valid difference images')
    ax[1].set_ylabel('Cumulative Count')
    ax[1].set_yscale('log')

f.suptitle(f'Kepler Categories\n Quality thr={thr}')
f.tight_layout()
f.savefig(plot_dir / f'hist_cats_kepler_tces_valid_diff_imgs_qm_thr{thr}.png')
plt.close()

f, ax = plt.subplots(2, 1, figsize=(12, 8))
for lbl in ['UNK']:
    ax[0].hist(qual_metric_tbl.loc[qual_metric_tbl['label'] == lbl, 'valid_quarters'], bins=bins,
            edgecolor='k', label=lbl, zorder=None, alpha=None,
               cumulative=False)
    # ax[0].legend()
    ax[0].set_ylabel('Count')
    ax[0].set_xticks(bins[:-1] + 0.5)
    ax[0].set_xticklabels(bins[:-1])
    ax[0].set_xlim(bins[[0, -1]])
    ax[0].set_yscale('log')
    ax[1].hist(qual_metric_tbl.loc[qual_metric_tbl['label'] == lbl, 'valid_quarters'], bins=bins,
               edgecolor='k', label=lbl, zorder=None, alpha=None,
               cumulative=True)
    # ax[1].legend()
    ax[1].set_xticks(bins[:-1] + 0.5)
    ax[1].set_xticklabels(bins[:-1])
    ax[1].set_xlim(bins[[0, -1]])
    ax[1].set_xlabel(f'Number of valid difference images')
    ax[1].set_ylabel('Cumulative Count')
    ax[1].set_yscale('log')

f.suptitle(f'Kepler UNKs\n Quality thr={thr}')
f.tight_layout()
f.savefig(plot_dir / f'hist_unk_kepler_tces_valid_diff_imgs_qm_thr{thr}.png')
plt.close()

#%% Study variation in quality metric intra-example

qual_metric_tbl = pd.read_csv('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/data/fits_files/kepler/q1_q17_dr25/dv/diff_img_quality_metric.csv')
tce_tbl = pd.read_csv('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/data/ephemeris_tables/kepler/q1-q17_dr25/11-17-2021_1243/q1_q17_dr25_tce_2020.09.28_10.36.22_stellar_koi_cfp_norobovetterlabels_renamedcols_nomissingval_symsecphase_cpkoiperiod_rba_cnt0n_valpc_modelchisqr_ruwe_magcat_uid_rv_posprob.csv')
plot_dir = Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/data/fits_files/kepler/q1_q17_dr25/dv/qual_metric_plots')
plot_dir.mkdir(exist_ok=True)

qual_metric_tbl = qual_metric_tbl.merge(tce_tbl[['uid', 'label', 'tce_rogue_flag']], on='uid', how='left',
                                        validate='one_to_one')
qual_metric_tbl = qual_metric_tbl.loc[qual_metric_tbl['tce_rogue_flag'] == 0]

thr = -1


def _get_valid_thr_quarters_stats(x, thr):

    N_QUARTERS = 17
    # q_value_cols = [f'q{q}_value' for q in range(1, N_QUARTERS + 1)]
    q_valid_cols = [f'q{q}_valid' for q in range(1, N_QUARTERS + 1)]

    q_valid_value_cols = [f'{k.split("_")[0]}_value' for k, v in x[q_valid_cols].items() if v]

    return x[q_valid_value_cols].mean(), x[q_valid_value_cols].std(ddof=1)


# qual_metric_tbl['valid_quarters'] = qual_metric_tbl.apply(lambda x: (x[q_value_cols] > thr).sum(), axis=1)
qual_metric_tbl[['mean', 'std']] = qual_metric_tbl.apply(_get_valid_thr_quarters_stats, axis=1, args=(thr, ),
                                                         result_type='expand')

f, ax = plt.subplots(1, 2, figsize=(16, 8))
ax[0].hist(qual_metric_tbl['std'], bins=np.linspace(0, 2, 100), edgecolor='k')
ax[0].set_ylabel('Counts')
ax[0].set_xlabel('Quality Metric std')
ax[0].set_yscale('log')
ax[0].set_xlim([0, 2])
ax[0].grid(True)
ax[1].hist(qual_metric_tbl['mean'], bins=np.linspace(-1, 1, 100), edgecolor='k')
# ax[0].set_ylabel('Counts')
ax[1].set_xlabel('Quality Metric mean')
ax[1].set_yscale('log')
ax[1].set_xlim([-1, 1])
ax[1].grid(True)
f.suptitle('Q1-Q17 DR25 TCEs')
f.tight_layout()
f.savefig(plot_dir / f'hist_mean_std_qual_metric_allexamples.png')

lbls = [
    'PC',
    'AFP',
    'NTP',
    'UNK',
]
for lbl in lbls:
    qual_metric_tbl_lbl = qual_metric_tbl.loc[qual_metric_tbl['label'] == lbl]

    f, ax = plt.subplots(1, 2, figsize=(16, 8))
    ax[0].hist(qual_metric_tbl_lbl['std'], bins=np.linspace(0, 2, 100), edgecolor='k')
    ax[0].set_ylabel('Counts')
    ax[0].set_xlabel('Quality Metric std')
    ax[0].set_yscale('log')
    ax[0].set_xlim([0, 2])
    ax[0].grid(True)
    ax[1].hist(qual_metric_tbl_lbl['mean'], bins=np.linspace(-1, 1, 100), edgecolor='k')
    # ax[0].set_ylabel('Counts')
    ax[1].set_xlabel('Quality Metric mean')
    ax[1].set_yscale('log')
    ax[1].set_xlim([-1, 1])
    ax[1].grid(True)
    f.suptitle(f'Q1-Q17 DR25 {lbl} TCEs')
    f.tight_layout()
    f.savefig(plot_dir / f'hist_mean_std_qual_metric_{lbl}.png')

#%% check distribution of all quality metric values (not distribution of summary statistics)

qual_metric_tbl = pd.read_csv('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/data/fits_files/kepler/q1_q17_dr25/dv/diff_img_quality_metric.csv')
tce_tbl = pd.read_csv('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/data/ephemeris_tables/kepler/q1-q17_dr25/11-17-2021_1243/q1_q17_dr25_tce_2020.09.28_10.36.22_stellar_koi_cfp_norobovetterlabels_renamedcols_nomissingval_symsecphase_cpkoiperiod_rba_cnt0n_valpc_modelchisqr_ruwe_magcat_uid_rv_posprob.csv')
plot_dir = Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/data/fits_files/kepler/q1_q17_dr25/dv/qual_metric_plots')
plot_dir.mkdir(exist_ok=True)

qual_metric_tbl = qual_metric_tbl.merge(tce_tbl[['uid', 'label', 'tce_rogue_flag']], on='uid', how='left',
                                        validate='one_to_one')
qual_metric_tbl = qual_metric_tbl.loc[qual_metric_tbl['tce_rogue_flag'] == 0]

qm_vals = {
    'all': [],
    'PC': [],
    'AFP': [],
    'NTP': [],
    'UNK': [],
}
N_QUARTERS = 17
q_valid_cols = [f'q{q}_valid' for q in range(1, N_QUARTERS + 1)]
for tce_i, tce in qual_metric_tbl.iterrows():
    q_valid_value_cols = [f'{k.split("_")[0]}_value' for k, v in tce[q_valid_cols].items() if v]
    qm_vals['all'].extend(tce[q_valid_value_cols].to_list())
    qm_vals[tce['label']].extend(tce[q_valid_value_cols].to_list())


f, ax = plt.subplots()
ax.hist(qm_vals['all'], bins=np.linspace(-1, 1, 100), edgecolor='k')
ax.set_ylabel('Counts')
ax.set_xlabel('Quality Metric')
ax.set_yscale('log')
ax.set_xlim([-1, 1])
ax.grid(True)
f.suptitle('Q1-Q17 DR25 TCEs')
f.tight_layout()
f.savefig(plot_dir / f'hist_qual_metric_allexamples.png')

lbls = [
    'PC',
    'AFP',
    'NTP',
    'UNK'
]
for lbl in lbls:

    f, ax = plt.subplots()
    ax.hist(qm_vals[lbl], bins=np.linspace(-1, 1, 100), edgecolor='k')
    ax.set_ylabel('Counts')
    ax.set_xlabel('Quality Metric')
    ax.set_yscale('log')
    ax.set_xlim([-1, 1])
    ax.grid(True)
    f.suptitle(f'Q1-Q17 DR25 {lbl} TCEs')
    f.tight_layout()
    f.savefig(plot_dir / f'hist_qual_metric_{lbl}.png')


#%% check fraction of positive quality metric quarters

qual_metric_tbl = pd.read_csv('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/data/fits_files/kepler/q1_q17_dr25/dv/diff_img_quality_metric.csv')
tce_tbl = pd.read_csv('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/data/ephemeris_tables/kepler/q1-q17_dr25/11-17-2021_1243/q1_q17_dr25_tce_2020.09.28_10.36.22_stellar_koi_cfp_norobovetterlabels_renamedcols_nomissingval_symsecphase_cpkoiperiod_rba_cnt0n_valpc_modelchisqr_ruwe_magcat_uid_rv_posprob.csv')
plot_dir = Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/data/fits_files/kepler/q1_q17_dr25/dv/qual_metric_plots')
plot_dir.mkdir(exist_ok=True)

qual_metric_tbl = qual_metric_tbl.merge(tce_tbl[['uid', 'label', 'tce_rogue_flag']], on='uid', how='left',
                                        validate='one_to_one')
qual_metric_tbl = qual_metric_tbl.loc[qual_metric_tbl['tce_rogue_flag'] == 0]

thr = -1


def _count_frac_thr_quarters(x, thr):

    N_QUARTERS = 17
    q_value_cols = [f'q{q}_value' for q in range(1, N_QUARTERS + 1)]
    q_valid_cols = [f'q{q}_valid' for q in range(1, N_QUARTERS + 1)]

    n_q_thr = ((x[q_value_cols] > thr) & (x[q_valid_cols])).sum()
    n_q = x[q_valid_cols].sum()
    # print(n_q, n_q_thr)

    if n_q == 0:
        return np.nan

    return n_q_thr, n_q_thr / n_q


qual_metric_tbl[['n_qs_pos', 'frac_qs_pos']] = qual_metric_tbl.apply(_count_frac_thr_quarters, axis=1, args=(thr, ),
                                                                     result_type='expand')

f, ax = plt.subplots(1, 2, figsize=(16, 8))
ax[0].hist(qual_metric_tbl['n_qs_pos'], bins=np.linspace(0, 19, 18), edgecolor='k')
ax[0].set_ylabel('Counts')
ax[0].set_xlabel('Number Positive Quarter Metric Values')
ax[0].set_yscale('log')
ax[0].set_xlim([0, 18])
ax[0].grid(True)
ax[1].hist(qual_metric_tbl['frac_qs_pos'], bins=np.linspace(0, 1, 100), edgecolor='k')
# ax[0].set_ylabel('Counts')
ax[1].set_xlabel('Fraction Positive Quarter Metric Values')
ax[1].set_yscale('log')
ax[1].set_xlim([0, 1])
ax[1].grid(True)
f.suptitle('Q1-Q17 DR25 TCEs')
f.tight_layout()
f.savefig(plot_dir / f'hist_npos_qual_metric_allexamples.png')

lbls = [
    'PC',
    'AFP',
    'NTP',
    'UNK',
]
for lbl in lbls:
    qual_metric_tbl_lbl = qual_metric_tbl.loc[qual_metric_tbl['label'] == lbl]

    f, ax = plt.subplots(1, 2, figsize=(16, 8))
    ax[0].hist(qual_metric_tbl_lbl['n_qs_pos'], bins=np.linspace(0, 19, 18), edgecolor='k')
    ax[0].set_ylabel('Counts')
    ax[0].set_xlabel('Number Positive Quarter Metric Values')
    ax[0].set_yscale('log')
    ax[0].set_xlim([0, 18])
    ax[0].grid(True)
    ax[1].hist(qual_metric_tbl_lbl['frac_qs_pos'], bins=np.linspace(0, 1, 100), edgecolor='k')
    # ax[0].set_ylabel('Counts')
    ax[1].set_xlabel('Fraction Positive Quarter Metric Values')
    ax[1].set_yscale('log')
    ax[1].set_xlim([0, 1])
    ax[1].grid(True)
    f.suptitle(f'Q1-Q17 DR25 {lbl} TCEs')
    f.tight_layout()
    f.savefig(plot_dir / f'hist_npos_qual_metric_{lbl}.png')
