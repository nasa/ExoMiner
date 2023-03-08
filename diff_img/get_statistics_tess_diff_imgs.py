""" Compute statistics for the difference image data in Kepler. """

# 3rd party
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

# from astropy.stats import mad_std

plt.switch_backend('agg')

# %% Choose processing experiment

run_dir = Path('/data5/tess_project/Data/TESS_dv_fits/dv_xml/preprocessing/8-17-2022_1611')

# %% Load difference image data

data = {}
for data_fp in (run_dir / 'data').iterdir():
    data.update(np.load(data_fp, allow_pickle=True).item())

# %% Get number of valid sectors per TCE

tce_sectors = {}
for tce_uid, tce_data in data.items():
    n_sectors_valid = np.sum([1 if float(target_ref_q['col']['uncertainty']) != -1 else 0
                              for target_ref_q in tce_data['target_ref_centroid']])
    tce_sectors[tce_uid] = n_sectors_valid

# %% Store values into csv file

sector_df = pd.DataFrame({'tce_uid': list(tce_sectors.keys()), 'n_valid_sectors': list(tce_sectors.values())})
sector_df.to_csv(run_dir / 'tces_n_valid_sectors.csv', index=False)
# sector_df = pd.read_csv(run_dir / 'tces_n_valid_sectors.csv')


# %% Plot histogram of number of valid sectors per TCE and compute statistics

sector_runs_arr = sector_df['tce_uid'].apply(lambda x: '-'.join(x.split('-')[2:])).unique()

plot_dir = run_dir / 'validsectors_tces'
plot_dir.mkdir(exist_ok=True)
sector_runs_stats_df = []
for sector_run in sector_runs_arr:

    if '-' not in sector_run:  # single-sector run
        sector_bins = [0, 1, 2]
    else:
        first_sector, last_sector = sector_run.split('-')
        first_sector, last_sector = int(first_sector[1:]), int(last_sector)
        n_sectors = last_sector - first_sector + 1
        sector_bins = np.arange(0, n_sectors + 2)

    sector_run_df = sector_df.loc[sector_df['tce_uid'].str.endswith(sector_run)]

    # get statistics
    sector_run_stats = sector_run_df.describe(percentiles=[0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 0.75, 0.90, 0.95, 0.99]).T
    sector_run_stats['sector_run'] = sector_run
    sector_run_stats.set_index('sector_run', inplace=True)
    sector_runs_stats_df.append(sector_run_stats)

    f, ax = plt.subplots(figsize=(12, 6))
    ax.hist(sector_run_df['n_valid_sectors'], sector_bins,
            edgecolor='k', align='left')
    ax.set_xlabel('Number of Valid Sectors')
    ax.set_ylabel('TCE Count')
    ax.set_yscale('log')
    ax.set_xticks(sector_bins[:-1])
    f.savefig(plot_dir / f'hist_validquarters_tces_{sector_run}.png')
    plt.close()

sector_run_stats_df = pd.concat(sector_runs_stats_df, axis=0).to_csv(run_dir / f'tces_n_valid_sectors_stats.csv')

# %% Get average image size per TCE

tce_img_size = {}
for tce_uid, tce_data in data.items():

    n_sectors = len(tce_data['image_data'])
    tce_img_size[tce_uid] = {
        'n_pxs': np.nan * np.ones(n_sectors),
        'n_cols': np.nan * np.ones(n_sectors),
        'n_rows': np.nan * np.ones(n_sectors),
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

N_SECTORS_MAX = 51
columns = ['tce_uid']
fields = ['n_pxs', 'n_cols', 'n_rows']
for field in fields:
    columns += [f'mean_{field}', f'std_{field}']
columns += [f's_{s_i}' for s_i in range(N_SECTORS_MAX)]
img_size_dict = {col: [] for col in columns}
for tce_uid in tce_img_size:

    img_size_dict['tce_uid'].append(tce_uid)
    # for k in fields:
    #     img_size_dict[k].append(np.nan)
    for s_i in range(N_SECTORS_MAX):
        img_size_dict[f's_{s_i}'].append((np.nan, np.nan))

    for k in fields:
        img_size_dict[f'mean_{k}'].append(np.nanmean(tce_img_size[tce_uid][k]))
        img_size_dict[f'std_{k}'].append(np.nanstd(tce_img_size[tce_uid][k]))

    n_sectors_tce = len(tce_img_size[tce_uid]['n_pxs'])
    for s_i in range(n_sectors_tce):
        img_size_dict[f's_{s_i}'][-1] = (tce_img_size[tce_uid]['n_rows'][s_i],
                                         tce_img_size[tce_uid]['n_cols'][s_i])

img_size_df = pd.DataFrame(img_size_dict)
img_size_df.to_csv(run_dir / 'tces_img_size.csv', index=False)
img_size_df = pd.read_csv(run_dir / 'tces_img_size.csv')

# img_size_df_multisector = img_size_df.loc[img_size_df['tce_uid'].str.count('-') == 3]
# get statistics
img_size_df_stats = img_size_df[
    ['mean_n_pxs', 'std_n_pxs', 'mean_n_cols', 'std_n_cols', 'mean_n_rows', 'std_n_rows']].describe(
    percentiles=[0.01, 0.1, 0.25, 0.5, 0.75, 0.90, 0.95, 0.99]).to_csv(run_dir / 'tces_img_size_stats.csv')

# %% Plot histogram of number of average image size per TCE

f, ax = plt.subplots()
ax.hist(img_size_df['mean_n_rows'], np.arange(11, 20), edgecolor='k', align='mid')
ax.set_xlabel('Mean Row Size (px)')
ax.set_ylabel('TCE Count')
# ax.set_xticks(quarter_bins[:-1])
ax.set_yscale('log')
f.savefig(run_dir / 'hist_mean_row_size_tces.png')
plt.close()

f, ax = plt.subplots()
ax.hist(img_size_df['mean_n_cols'], np.arange(11, 20), edgecolor='k', align='mid')
ax.set_xlabel('Mean Col Size (px)')
ax.set_ylabel('TCE Count')
# ax.set_xticks(quarter_bins[:-1])
ax.set_yscale('log')
f.savefig(run_dir / 'hist_mean_col_size_tces.png')
plt.close()

f, ax = plt.subplots()
ax.hist(img_size_df['mean_n_pxs'], np.arange(121, 400, 4), edgecolor='k', align='mid')
ax.set_xlabel('Mean Image Size (px^2)')
ax.set_ylabel('TCE Count')
ax.set_yscale('log')
# ax.set_xticks(quarter_bins[:-1])
f.savefig(run_dir / 'hist_mean_img_size_tces.png')
plt.close()

# %% Plot image size as function of TESS magnitude

tce_tbl = pd.read_csv(
    '/data5/tess_project/Data/Ephemeris_tables/TESS/DV_SPOC_mat_files/11-29-2021/tess_tces_s1-s40_11-23-2021_1409_stellarparams_updated_eb_tso_tec_label_modelchisqr_astronet_ruwe_magcat_uid_corrtsoebs_corraltdetfail.csv',
    usecols=['uid', 'mag'])

img_size_df = pd.read_csv(run_dir / 'tces_img_size.csv').rename(columns={'tce_uid': 'uid'})
img_size_df = img_size_df.merge(tce_tbl, on=['uid'], how='left', validate='one_to_one')

f, ax = plt.subplots()
ax.scatter(img_size_df['mag'], img_size_df['mean_n_pxs'], s=8)
ax.set_ylabel('Mean Image Size (px^2)')
ax.set_xlabel('TESS Magnitude')
ax.set_yscale('log')
f.savefig(run_dir / 'scatter_tess_mag-mean_img_size.png')

f, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].scatter(img_size_df['mag'], img_size_df['mean_n_cols'], s=8)
ax[0].set_ylabel('Mean Col Size (px)')
# ax[0].set_xscale('log')
ax[0].set_xlabel('TESS Magnitude')
ax[1].scatter(img_size_df['mag'], img_size_df['mean_n_rows'], s=8)
ax[1].set_ylabel('Mean Row Size (px)')
ax[1].set_xlabel('TESS Magnitude')
# ax[1].set_yscale('log')
f.savefig(run_dir / 'scatter_tess_mag-avg_col_row_sizes.png')

# %% Compute average offset from TIC pixel to pixel with maximum value in difference image (transit source location)

# img_final_size = [7, 7]
# max_in_dist = np.sqrt(np.sum([((el - 1) / 2) ** 2 for el in img_final_size]))
N_SECTORS_MAX = 51
diff_src_dist = {}
for tce_uid, tce_data in data.items():

    diff_src_dist[tce_uid] = np.nan * np.ones(N_SECTORS_MAX)
    for q_i, target_img_q in enumerate(tce_data['image_data']):
        if tce_data['target_ref_centroid'][q_i]['col']['uncertainty'] != -1:
            # aaa
            target_px_trunc = {coord: int(coord_val['value'])
                               for coord, coord_val in tce_data['target_ref_centroid'][q_i].items()}
            max_diff_px = np.unravel_index(np.argmax(target_img_q[:, :, 2, 0], axis=None),
                                           target_img_q[:, :, 2, 0].shape)
            diff_src_dist[tce_uid][q_i] = np.sqrt(np.sum([(target_px_trunc['row'] - max_diff_px[0]) ** 2,
                                                          (target_px_trunc['col'] - max_diff_px[1]) ** 2]))

    # diff_src_cropped[tce_uid] = (diff_src_cropped[tce_uid] == 1).sum() / (~np.isnan(diff_src_cropped[tce_uid])).sum()
    # diff_src_dist[tce_uid] = np.nanmean(diff_src_dist[tce_uid])

diff_src_dist_dict = {'tce_uid': [], 'mean_dist_diff': [], 'std_dist_diff': []}
diff_src_dist_dict.update({f'dist_s_{q_i}': [] for q_i in range(N_SECTORS_MAX)})

for tce_uid, src_dist in diff_src_dist.items():
    diff_src_dist_dict['tce_uid'].append(tce_uid)
    diff_src_dist_dict['mean_dist_diff'].append(np.nanmean(src_dist))
    diff_src_dist_dict['std_dist_diff'].append(np.nanstd(src_dist))
    for q_i in range(N_SECTORS_MAX):
        diff_src_dist_dict[f'dist_s_{q_i}'].append(src_dist[q_i])

diff_src_dist_df = pd.DataFrame(diff_src_dist_dict)
diff_src_dist_df.to_csv(run_dir / f'diff_src_mean_dist.csv', index=False)

# %% Plot histogram of average offset from TIC pixel to pixel with maximum value in difference image (transit source location)

diff_src_dist_df = pd.read_csv(run_dir / 'diff_src_mean_dist.csv')

bins = np.linspace(0, 10, 100)
f, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].hist(diff_src_dist_df['mean_dist_diff'], bins, edgecolor='k')
ax[0].set_xlabel('Average Transit Source Offset from TIC pixel (px)')
ax[0].set_ylabel('TCE Count')
ax[0].set_yscale('log')
ax[1].hist(diff_src_dist_df['mean_dist_diff'], bins, edgecolor='k', cumulative=True)
ax[1].set_yscale('log')
ax[1].set_ylabel('Cumulative TCE Count')
ax[1].set_xlabel('Average Transit Source Offset from TIC pixel (px)')
f.savefig(run_dir / 'hist_diff_src_mean_dist.png')


#%% Plot histogram of difference image quality metric

qual_metric_dir = Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/data/fits_files/tess/dv/diff_img_quality_metric')
thr = 0.7
plot_dir = qual_metric_dir / 'plots'
plot_dir.mkdir(exist_ok=True)
tce_tbl = pd.read_csv('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/data/ephemeris_tables/tess/DV_SPOC_mat_files/10-05-2022_1338/tess_tces_dv_s1-s55_10-05-2022_1338_ticstellar_ruwe_tec_tsoebs_ourmatch_preproc.csv')

categories = {
    'KP': {'zorder': 2, 'alpha': 0.8},
    'CP': {'zorder': 1, 'alpha': 1},
    'EB': {'zorder': 2, 'alpha': 0.8},
    'FP': {'zorder': 1, 'alpha': 1},
    'NTP': {'zorder': 1, 'alpha': 1.0},
    'FA': {'zorder': 3, 'alpha': 0.5},
    # 'UNK': {'zorder': 2, 'alpha': 0.5},
# }

# unlbl_categories = {
    'PC': {'zorder': 3, 'alpha': 0.5},
    'APC': {'zorder': 2, 'alpha': 0.8},
    'UNK': {'zorder': 1, 'alpha': 1.0},
}

for qual_metric_tbl_fp in [fp for fp in qual_metric_dir.iterdir() if fp.suffix == '.csv']:

    qual_metric_tbl = pd.read_csv(qual_metric_tbl_fp)

    sector_id = qual_metric_tbl_fp.stem.split('_')[-1]
    print(f'Generating quality metric histograms for sector run {sector_id}')
    if '-' in sector_id:
        s_sector, e_sector = [int(s) for s in sector_id.split('-')]
    else:
        s_sector, e_sector = [int(sector_id)] * 2

    n_sectors = e_sector - s_sector + 1

    s_value_cols = [f's{s}_value' for s in range(s_sector, e_sector + 1)]
    qual_metric_tbl['valid_sectors'] = qual_metric_tbl.apply(lambda x: (x[s_value_cols] > thr).sum(), axis=1)

    qual_metric_tbl = qual_metric_tbl.merge(tce_tbl[['uid', 'label']], on='uid', how='left', validate='one_to_one')

    bins = np.linspace(0, n_sectors + 1, n_sectors + 2, endpoint=True, dtype='int')

    f, ax = plt.subplots(figsize=(12, 8))
    ax.hist(qual_metric_tbl['valid_sectors'], bins=bins, edgecolor='k')
    ax.set_xlabel(f'Number of valid difference images')
    ax.set_ylabel('Count')
    ax.set_xticks(bins[:-1] + 0.5)
    ax.set_xticklabels(bins[:-1])
    ax.set_xlim(bins[[0, -1]])
    ax.set_title(f'S{sector_id} \nAll TESS data\n Quality thr={thr}')
    f.tight_layout()
    f.savefig(plot_dir / f'hist_s{sector_id}_all_tess_tces_valid_diff_imgs_qm_thr{thr}.png')
    plt.close()

    f, ax = plt.subplots(2, 1, figsize=(12, 8))
    for lbl in ['CP', 'KP']:
        aux_tbl = qual_metric_tbl.loc[qual_metric_tbl['label'] == lbl, 'valid_sectors']
        if len(aux_tbl) == 0:
            continue
        ax[0].hist(aux_tbl, bins=bins,
                edgecolor='k', label=lbl, zorder=categories[lbl]['zorder'], alpha=categories[lbl]['alpha'],
                   cumulative=False)
        ax[1].hist(aux_tbl, bins=bins,
                   edgecolor='k', label=lbl, zorder=categories[lbl]['zorder'], alpha=categories[lbl]['alpha'],
                   cumulative=True)
    ax[0].legend()
    ax[0].set_ylabel('Count')
    ax[0].set_xticks(bins[:-1] + 0.5)
    ax[0].set_xticklabels(bins[:-1])
    ax[0].set_xlim(bins[[0, -1]])
    ax[0].set_yscale('log')
    ax[1].legend()
    ax[1].set_xticks(bins[:-1] + 0.5)
    ax[1].set_xticklabels(bins[:-1])
    ax[1].set_xlim(bins[[0, -1]])
    ax[1].set_xlabel(f'Number of valid difference images')
    ax[1].set_ylabel('Cumulative Count')
    ax[1].set_yscale('log')

    f.suptitle(f'S{sector_id}\nTESS Planets\n Quality thr={thr}')
    # f.tight_layout()
    f.savefig(plot_dir / f'hist_s{sector_id}_planets_tess_tces_valid_diff_imgs_qm_thr{thr}.png')
    plt.close()

    f, ax = plt.subplots(2, 1, figsize=(12, 8))
    for lbl in ['EB', 'FP', 'FA']:
        aux_tbl = qual_metric_tbl.loc[qual_metric_tbl['label'] == lbl, 'valid_sectors']
        if len(aux_tbl) == 0:
            continue
        ax[0].hist(aux_tbl, bins=bins,
                edgecolor='k', label=lbl, zorder=categories[lbl]['zorder'], alpha=categories[lbl]['alpha'],
                   cumulative=False)
        ax[1].hist(aux_tbl, bins=bins,
                   edgecolor='k', label=lbl, zorder=categories[lbl]['zorder'],
                   alpha=categories[lbl]['alpha'],
                   cumulative=True)
    ax[0].set_ylabel('Count')
    ax[0].set_xticks(bins[:-1] + 0.5)
    ax[0].set_xticklabels(bins[:-1])
    ax[0].set_xlim(bins[[0, -1]])
    ax[0].set_yscale('log')
    ax[0].legend()
    ax[1].set_xticks(bins[:-1] + 0.5)
    ax[1].set_xticklabels(bins[:-1])
    ax[1].set_xlim(bins[[0, -1]])
    ax[1].set_xlabel(f'Number of valid difference images')
    ax[1].set_ylabel('Cumulative Count')
    ax[1].set_yscale('log')
    ax[1].legend()
    f.suptitle(f'S{sector_id}\nTESS FPs\n Quality thr={thr}')
    # f.tight_layout()
    f.savefig(plot_dir / f'hist_s{sector_id}_fps_tess_tces_valid_diff_imgs_qm_thr{thr}.png')
    plt.close()

    f, ax = plt.subplots(2, 1, figsize=(12, 8))
    for lbl in ['NTP']:
        aux_tbl = qual_metric_tbl.loc[qual_metric_tbl['label'] == lbl, 'valid_sectors']
        if len(aux_tbl) == 0:
            continue
        ax[0].hist(aux_tbl, bins=bins,
                edgecolor='k', label=lbl, zorder=categories[lbl]['zorder'], alpha=categories[lbl]['alpha'],
                   cumulative=False)
        ax[1].hist(aux_tbl, bins=bins,
                   edgecolor='k', label=lbl, zorder=categories[lbl]['zorder'], alpha=categories[lbl]['alpha'],
                   cumulative=True)
    ax[0].legend()
    ax[0].set_ylabel('Count')
    ax[0].set_xticks(bins[:-1] + 0.5)
    ax[0].set_xticklabels(bins[:-1])
    ax[0].set_xlim(bins[[0, -1]])
    ax[0].set_yscale('log')
    ax[1].legend()
    ax[1].set_xticks(bins[:-1] + 0.5)
    ax[1].set_xticklabels(bins[:-1])
    ax[1].set_xlim(bins[[0, -1]])
    ax[1].set_xlabel(f'Number of valid difference images')
    ax[1].set_ylabel('Cumulative Count')
    ax[1].set_yscale('log')

    f.suptitle(f'S{sector_id}\nTESS NTPs\n Quality thr={thr}')
    # f.tight_layout()
    f.savefig(plot_dir / f'hist_s{sector_id}_ntps_tess_tces_valid_diff_imgs_qm_thr{thr}.png')
    plt.close()

    f, ax = plt.subplots(2, 1, figsize=(12, 8))
    for lbl in ['APC', 'PC', 'UNK']:
        aux_tbl = qual_metric_tbl.loc[qual_metric_tbl['label'] == lbl, 'valid_sectors']
        if len(aux_tbl) == 0:
            continue
        ax[0].hist(aux_tbl, bins=bins,
                edgecolor='k', label=lbl, zorder=categories[lbl]['zorder'], alpha=categories[lbl]['alpha'],
                   cumulative=False)
        ax[1].hist(aux_tbl, bins=bins,
                   edgecolor='k', label=lbl, zorder=categories[lbl]['zorder'],
                   alpha=categories[lbl]['alpha'],
                   cumulative=True)
    ax[0].set_ylabel('Count')
    ax[0].set_xticks(bins[:-1] + 0.5)
    ax[0].set_xticklabels(bins[:-1])
    ax[0].set_xlim(bins[[0, -1]])
    ax[0].set_yscale('log')
    ax[0].legend()
    ax[1].set_xticks(bins[:-1] + 0.5)
    ax[1].set_xticklabels(bins[:-1])
    ax[1].set_xlim(bins[[0, -1]])
    ax[1].set_xlabel(f'Number of valid difference images')
    ax[1].set_ylabel('Cumulative Count')
    ax[1].set_yscale('log')
    ax[1].legend()
    f.suptitle(f'S{sector_id}\nTESS Unused\n Quality thr={thr}')
    # f.tight_layout()
    f.savefig(plot_dir / f'hist_s{sector_id}_unused_tess_tces_valid_diff_imgs_qm_thr{thr}.png')
    plt.close()
