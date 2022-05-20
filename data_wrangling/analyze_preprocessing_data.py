"""
Analyze results of the preprocessing pipeline.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%%

res_dir = Path('/data5/tess_project/Data/tfrecords/TESS/tfrecordstess-dv_g301-l31_spline_nongapped_flux-loe-lwks-centroid-centroid_fdl-scalars_4-23-2021_data/tfrecordstess-dv_g301-l31_spline_nongapped_flux-loe-lwks-centroid-centroid_fdl-scalars_4-23-2021')
res_tbl = pd.read_csv(res_dir / 'merged_shards.csv')
# res_tbl.drop(columns='Unnamed: 0', inplace=True)

tce_tbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/TESS/EXOFOP_TOI_lists/TOI/4-22-2021/exofop_toilists_nomissingpephem_sectors.csv')

# dataset_tbls_dir = Path('/data5/tess_project/Data/tfrecords/Kepler/Q1-Q17_DR25/train-val-test-sets/split_6-1-2020')
# dataset_tbls = {fp.stem[:-3]: pd.read_csv(fp) for fp in dataset_tbls_dir.iterdir() if fp.suffix == '.csv'}
dataset_tbls = {'predict': tce_tbl}

#%% add dispositions from the TCE table

# add_fields = ['label', 'koi_disposition', 'fpwg_disp_status', 'tce_period', 'tce_duration', 'tce_time0bk',
#               'transit_depth', 'tce_max_mult_ev', 'tce_model_snr', 'wst_depth', 'tce_maxmes','tce_fwm_stat',
#               'tce_fwm_srao', 'tce_fwm_sdeco', 'tce_fwm_prao', 'tce_fwm_pdeco', 'tce_dicco_msky', 'tce_dikco_msky',
#               'koi_comment']
add_fields = ['label', 'TESS Disposition', 'tce_period', 'tce_duration', 'tce_time0bk',
              'transit_depth', 'tce_max_mult_ev', 'tce_model_snr', 'wst_depth', 'tce_maxmes',
              'tce_dicco_msky', 'tce_dikco_msky', 'Comments']
for field in add_fields:
    res_tbl[field] = np.nan

for tce_i, tce in res_tbl.iterrows():

    # tce_found = tce_tbl.loc[(tce_tbl['target_id'] == tce['target_id']) &
    #                         (tce_tbl['tce_plnt_num'] == tce['tce_plnt_num'])]
    tce_found = tce_tbl.loc[tce_tbl['oi'] == tce['oi']]

    if len(tce_found) == 1:
        res_tbl.loc[tce_i, add_fields] = tce_found[add_fields].values[0]

#%% add train, val, test indicator

add_fields = ['set']
for field in add_fields:
    res_tbl[field] = np.nan

for tce_i, tce in res_tbl.iterrows():

    for dataset_tbl_name, dataset_tbl in dataset_tbls.items():
        tce_found = dataset_tbl.loc[
            (dataset_tbl['target_id'] == tce['target_id']) & (dataset_tbl['tce_plnt_num'] == tce['tce_plnt_num'])]

        if len(tce_found) == 1:
            res_tbl.loc[tce_i, add_fields] = dataset_tbl_name
            break

# res_tbl['set'] = 'predict'

#%%

# add column for Possible Planet KOIs removed from dataset afterwards
res_tbl['pp_removed'] = np.nan
res_tbl.loc[(res_tbl['fpwg_disp_status'] == 'POSSIBLE PLANET') &
            (res_tbl['koi_disposition'] != 'CONFIRMED'), 'pp_removed'] = 'yes'

# compute offset using fwm centroid offset coordinates from KIC position
res_tbl['tce_fwm_co_peak'] = np.sqrt(res_tbl['tce_fwm_srao'] ** 2 + res_tbl['tce_fwm_sdeco'] ** 2)

# compute offset using fwm centroid offset coordinates from KIC position
res_tbl['tce_fwm_co_kic'] = np.sqrt(res_tbl['tce_fwm_prao'] ** 2 + res_tbl['tce_fwm_pdeco'] ** 2)

res_tbl.to_csv(res_dir / 'merged_shards_disp_set.csv', index=False)

figure_dir = res_dir / 'analysis_plots'
figure_dir.mkdir(exist_ok=True)

#%% analyze odd-even flag

disp_col = 'TESS Disposition'
# dispositions = ['PC', 'AFP', 'NTP']
# dispositions = ['KP', 'CP', 'PC', 'APC', 'FP', 'FA']
dispositions = ['KP', 'CP', 'PC', 'EB', 'O']
f, ax = plt.subplots()
bar_center_pts = np.linspace(0.5, 0.5 * len(dispositions), len(dispositions))
# count_disp = [len(res_tbl.loc[(res_tbl['odd_even_flag'] != 'ok') & (res_tbl[disp_col] == disp)])
#               for disp in dispositions]
count_disp = [len(res_tbl.loc[(res_tbl['odd_even_flag'] != 'replaced bins 0 (odd) 0 (even) 0 (both)') &
                              (res_tbl[disp_col] == disp)])
              for disp in dispositions]
ax.bar(bar_center_pts, count_disp, edgecolor='k', align='center', width=0.5)
ax.set_xticks(bar_center_pts)
ax.set_xticklabels(dispositions)
ax.set_ylabel('Counts')
ax.set_xlabel('Disposition')
ax.set_yscale('log')
f.savefig(figure_dir / 'odd_even_flag_hist.png')

#%% analyze avg out-of-transit centroid offset

# bins = np.linspace(0, 100, 10, endpoint=True)
bins = np.logspace(-2, 2, 20, endpoint=True)
# dispositions = ['PC', 'AFP', 'NTP']
dispositions = ['KP', 'CP', 'PC', 'APC', 'FP', 'FA']
for disp in dispositions:
    f, ax = plt.subplots()
    ax.hist(res_tbl.loc[res_tbl[disp_col] == disp, 'avg_oot_centroid_offset'], bins, edgecolor='k')
    ax.axvline(x=30, c='r')
    ax.set_ylabel('Counts')
    ax.set_xlabel('Avg oot centroid offset from KIC position (arcsec)')
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_title(f'{disp}')
    f.savefig(figure_dir / f'avg_oot_co_kic_hist_{disp}.png')
    plt.close()

for disp in dispositions:
    f, ax = plt.subplots()
    ax.scatter(res_tbl.loc[res_tbl[disp_col] == disp, 'avg_oot_centroid_offset'], res_tbl.loc[res_tbl[disp_col] == disp, 'tce_dikco_msky'], s=8)
    ax.set_yscale('log')
    ax.set_ylim(bottom=1e-3)
    ax.set_ylabel('Difference image centroid offset from KIC position (arcsec)')
    ax.set_xlabel('Preprocessing pipeline avg oot centroid offset from KIC position (arcsec)')
    ax.set_title(f'{disp}')
    f.savefig(figure_dir / f'avg_oot_co_kic_vs_dikco_msky_scatter_{disp}.png')
    plt.close()

for disp in dispositions:
    f, ax = plt.subplots()
    ax.scatter(res_tbl.loc[res_tbl[disp_col] == disp, 'std_oot_centroid_offset'] / res_tbl.loc[res_tbl[disp_col] == disp, 'avg_oot_centroid_offset'],
               res_tbl.loc[res_tbl[disp_col] == disp, 'tce_fwm_stat'], s=8)
    # ax.set_yscale('log')
    # ax.set_ylim(bottom=1e-3)
    ax.set_xscale('log')
    ax.set_ylabel('FWM stat (%)')
    ax.set_xlabel('Preprocessing pipeline avg oot centroid offset from KIC position (arcsec)')
    ax.set_title(f'{disp}')
    ax.set_ylim([0, 100])
    f.savefig(figure_dir / f'avg_oot_co_kic_vs_fwm_stat_scatter_{disp}.png')
    plt.close()

for disp in dispositions:
    f, ax = plt.subplots()
    ax.scatter(res_tbl.loc[res_tbl[disp_col] == disp, 'avg_oot_centroid_offset'],
               res_tbl.loc[res_tbl[disp_col] == disp, 'tce_fwm_co_kic'], s=8)
    ax.set_yscale('log')
    ax.set_ylim(bottom=1e-6)
    ax.set_ylabel('FWM centroid offset from KIC position (arcsec)')
    ax.set_xlabel('Preprocessing pipeline avg oot centroid offset from KIC position (arcsec)')
    ax.set_title(f'{disp}')
    f.savefig(figure_dir / f'avg_oot_co_kic_vs_fwm_co_kic_scatter_{disp}.png')
    plt.close()

for disp in dispositions:
    f, ax = plt.subplots()
    ax.scatter(res_tbl.loc[res_tbl[disp_col] == disp, 'peak_centroid_offset'],
               res_tbl.loc[res_tbl[disp_col] == disp, 'tce_fwm_co_peak'], s=8)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_ylim([1e-3, 1e3])
    ax.set_xlim([1e-5, 1e2])
    ax.set_ylabel('FWM transit source offset (arcsec)')
    ax.set_xlabel('Preprocessing pipeline it centroid offset (arcsec)')
    ax.set_title(f'{disp}')
    f.savefig(figure_dir / f'avg_oot_co_kic_vs_fwm_so_scatter_{disp}.png')
    plt.close()

for disp in dispositions:
    f, ax = plt.subplots()
    ax.scatter(res_tbl.loc[res_tbl[disp_col] == disp, 'tce_fwm_co_peak'],
               res_tbl.loc[res_tbl[disp_col] == disp, 'tce_fwm_stat'], s=8)
    ax.set_yscale('log')
    # ax.set_ylim(bottom=1e-3)
    ax.set_xlim([1e-3, 1e2])
    ax.set_xscale('log')
    ax.set_ylabel('FWM stat (%)')
    ax.set_xlabel('FWM transit source offset (arcsec)')
    ax.set_title(f'{disp}')
    ax.set_ylim([1e-1, 1e5])
    f.savefig(figure_dir / f'fwm_so_vs_fwm_stat_scatter_{disp}.png')
    plt.close()

#%% analyze estimated transit depth vs DV transit depth

# dispositions = ['PC', 'AFP', 'NTP']
for disp in dispositions:
    f, ax = plt.subplots()
    ax.scatter(res_tbl.loc[res_tbl[disp_col] == disp, 'transit_depth_hat'],
               res_tbl.loc[res_tbl[disp_col] == disp, 'transit_depth'], s=8)
    ax.plot([0, max(res_tbl.loc[res_tbl[disp_col] == disp, 'transit_depth'].max(),
                    res_tbl.loc[res_tbl[disp_col] == disp, 'transit_depth_hat'].max())],
            [0, max(res_tbl.loc[res_tbl[disp_col] == disp, 'transit_depth'].max(),
                    res_tbl.loc[res_tbl[disp_col] == disp, 'transit_depth_hat'].max())],
            'r--'
            )
    ax.set_ylabel('DV transit depth (ppm)')
    ax.set_xlabel('Preprocessing pipeline estimated transit depth (ppm)')
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlim([1, 1e5])
    ax.set_ylim([1, 1e5])
    ax.grid(True)
    ax.set_title(f'{disp}')
    f.savefig(figure_dir / f'transit_depth_dv_vs_transit_depth_ourpipeline_{disp}_scatter.png')
    plt.close()

#%% analyze estimated weak secondary transit depth vs DV weak secondary transit depth

# dispositions = ['PC', 'AFP', 'NTP']
for disp in dispositions:
    f, ax = plt.subplots()
    ax.scatter(res_tbl.loc[res_tbl[disp_col] == disp, 'wks_transit_depth_hat'],
               res_tbl.loc[res_tbl[disp_col] == disp, 'wst_depth'], s=8)
    ax.plot([0, max(res_tbl.loc[res_tbl[disp_col] == disp, 'wst_depth'].max(),
                    res_tbl.loc[res_tbl[disp_col] == disp, 'wks_transit_depth_hat'].max())],
            [0, max(res_tbl.loc[res_tbl[disp_col] == disp, 'wst_depth'].max(),
                    res_tbl.loc[res_tbl[disp_col] == disp, 'wks_transit_depth_hat'].max())],
            'r--'
            )
    ax.set_ylabel('DV wks transit depth (ppm)')
    ax.set_xlabel('Preprocessing pipeline estimated wks transit depth (ppm)')
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlim([1, 1e5])
    ax.set_ylim([1, 1e5])
    ax.grid(True)
    ax.set_title(f'{disp}')
    f.savefig(figure_dir / f'wks_transit_depth_dv_vs_wks_transit_depth_ourpipeline_{disp}_scatter.png')
    plt.close()

#%% analyze bin shift

bins = np.linspace(0, 15, 16, endpoint=True)
# dispositions = ['PC', 'AFP', 'NTP']
for disp in dispositions:
    f, ax = plt.subplots()
    ax.hist(np.abs(res_tbl.loc[res_tbl[disp_col] == disp, 'mid_local_flux_shift']), bins, edgecolor='k')
    ax.set_ylabel('Counts')
    ax.set_xlabel('Local flux shift (number of bins)')
    ax.set_title(f'{disp}')
    ax.set_yscale('log')
    ax.set_xlim([bins[0], bins[-1]])
    f.savefig(figure_dir / f'local_flux_shift_hist_{disp}.png')
    plt.close()

bins = np.linspace(0, 150, 31, endpoint=True)
# dispositions = ['PC', 'AFP', 'NTP']
for disp in dispositions:
    f, ax = plt.subplots()
    ax.hist(np.abs(res_tbl.loc[res_tbl[disp_col] == disp, 'mid_global_flux_shift']), bins, edgecolor='k')
    ax.set_ylabel('Counts')
    ax.set_xlabel('Global flux shift (number of bins)')
    ax.set_title(f'{disp}')
    ax.set_yscale('log')
    ax.set_xlim([bins[0], bins[-1]])
    f.savefig(figure_dir / f'global_flux_shift_hist_{disp}.png')
    plt.close()

feature_cols = ['transit_depth', 'tce_max_mult_ev', 'tce_model_snr', 'num_transits_flux']
# dispositions = ['PC', 'AFP', 'NTP']
disp_col = 'label'
for feature_col in feature_cols:
    for disp in dispositions:
        f, ax = plt.subplots()
        ax.scatter(np.abs(res_tbl.loc[res_tbl[disp_col] == disp, 'mid_local_flux_shift']),
                   np.abs(res_tbl.loc[res_tbl[disp_col] == disp, feature_col]), s=8)
        ax.set_yscale('log')
        if feature_col in ['transit_depth']:
            ax.set_ylim([1, 1e5])
        ax.set_xlim([-1, 15])
        ax.set_ylabel(f'{feature_col}')
        ax.set_xlabel('Local flux shift (number of bins)')
        ax.set_title(f'{disp}')
        f.savefig(figure_dir / f'local_flux_shift_vs_{feature_col}_scatter_{disp}.png')
        plt.close()

    for disp in dispositions:
        f, ax = plt.subplots()
        ax.scatter(np.abs(res_tbl.loc[res_tbl[disp_col] == disp, 'mid_global_flux_shift']),
                   np.abs(res_tbl.loc[res_tbl[disp_col] == disp, feature_col]), s=8)
        ax.set_yscale('log')
        if feature_col in ['transit_depth']:
            ax.set_ylim([1, 1e5])
        ax.set_xlim([-1, 150])
        ax.set_ylabel(f'{feature_col}')
        ax.set_xlabel('Global flux shift (number of bins)')
        ax.set_title(f'{disp}')
        f.savefig(figure_dir / f'global_flux_shift_vs_{feature_col}_scatter_{disp}.png')
        plt.close()

for disp in dispositions:
    f, ax = plt.subplots()
    ax.scatter(np.abs(res_tbl.loc[res_tbl[disp_col] == disp, 'mid_local_flux_shift']),
               np.abs(res_tbl.loc[res_tbl[disp_col] == disp, 'mid_global_flux_shift']), s=8)
    ax.set_xlim([-1, 31])
    ax.set_ylim([-1, 151])
    ax.set_ylabel('Global flux shift (number of bins)')
    ax.set_xlabel('local flux shift (number of bins)')
    ax.set_title(f'{disp}')
    f.savefig(figure_dir / f'local_flux_shift_vs_global_flux_shift_scatter_{disp}.png')
    plt.close()

#%% analyze odd and even sigma

dispositions = ['PC', 'AFP', 'NTP']
disp_col = 'label'

for disp in dispositions:
    f, ax = plt.subplots()
    ax.scatter(res_tbl.loc[res_tbl[disp_col] == disp, 'sigma_it_odd'],
               res_tbl.loc[res_tbl[disp_col] == disp, 'sigma_it_even'], s=8)
    # ax.set_xlim([-1, 31])
    # ax.set_ylim([-1, 151])
    ax.set_ylabel('sigma_it_odd')
    ax.set_xlabel('sigma_it_even')
    ax.set_title(f'{disp}')
    f.savefig(figure_dir / f'sigma_it_odd_vs_sigma_it_even_scatter_{disp}.png')
    plt.close()

# bins = np.
for disp in dispositions:
    f, ax = plt.subplots()
    ax.hist(np.abs(res_tbl.loc[res_tbl[disp_col] == disp, 'sigma_it_odd'] -
                   res_tbl.loc[res_tbl[disp_col] == disp, 'sigma_it_even']))
    # ax.set_xlim([-1, 31])
    # ax.set_ylim([-1, 151])
    ax.set_ylabel('Counts')
    ax.set_xlabel('|sigma_it_even-sigma_it_odd|')
    ax.set_title(f'{disp}')
    f.savefig(figure_dir / f'abs_diff_sigma_it_odd-even_hist_{disp}.png')
    plt.close()

for disp in dispositions:
    f, ax = plt.subplots(1, 2, figsize=(8, 6))
    ax[0].hist(np.abs(res_tbl.loc[(res_tbl['koi_comment'].str.contains('DEPTH_ODDEVEN', na=False) & (res_tbl[disp_col] == disp)), 'sigma_it_odd'] -
                      res_tbl.loc[(res_tbl['koi_comment'].str.contains('DEPTH_ODDEVEN', na=False) & (res_tbl[disp_col] == disp)), 'sigma_it_even']))
    ax[0].set_xlabel('|sigma_it_even-sigma_it_odd|')
    ax[0].set_ylabel('Counts')
    ax[0].set_title('DEPTH_ODDEVEN DV FLAG')
    ax[1].hist(np.abs(res_tbl.loc[(~res_tbl['koi_comment'].str.contains('DEPTH_ODDEVEN', na=False) & (res_tbl[disp_col] == disp)), 'sigma_it_odd'] -
                      res_tbl.loc[(~res_tbl['koi_comment'].str.contains('DEPTH_ODDEVEN', na=False) & (res_tbl[disp_col] == disp)), 'sigma_it_even']))
    ax[1].set_xlabel('|sigma_it_even-sigma_it_odd|')
    ax[1].set_title('No DEPTH_ODDEVEN DV FLAG')
    f.suptitle(f'{disp}')
    # f.subplots_adjust(
    #     top=0.925,
    #     bottom=0.098,
    #     left=0.083,
    #     right=0.981,
    #     hspace=0.2,
    #     wspace=0.213
    # )
    ax[0].set_yscale('log')
    ax[1].set_yscale('log')
    f.savefig(figure_dir / f'abs_diff_sigma_it_odd-even_hist_koicomment_{disp}.png')
    plt.close()

#%% analyze sigma it/ot for odd/even

# res_dir = Path('/data5/tess_project/Data/tfrecords/Kepler/Q1-Q17_DR25/tfrecordskeplerdr25-dv_g301-l31_spline_nongapped_flux-loe-lwks-centroid-centroid_fdl-scalars_oereplbins_data/tfrecordskeplerdr25-dv_g301-l31_spline_nongapped_flux-loe-lwks-centroid-centroid_fdl-scalars_oereplbins_caphap_stat_diff')
res_dir = Path('/data5/tess_project/Data/tfrecords/TESS/tfrecordstess-dv_g301-l31_spline_nongapped_flux-loe-lwks-centroid-centroid_fdl-scalars_4-23-2021_data/tfrecordstess-dv_g301-l31_spline_nongapped_flux-loe-lwks-centroid-centroid_fdl-scalars_4-23-2021/')
res_tbl = pd.read_csv(res_dir / 'merged_shards_disp_set.csv')

cols = ['sigma_oot_odd', 'sigma_it_odd', 'sigma_oot_even', 'sigma_it_even']
disp_col = 'label'
dispositions = ['KP', 'APC', 'PC', 'CP', 'FA', 'FP']  # res_tbl[disp_col].unique()
# dispositions = ['PC', 'AFP', 'NTP']

# bins = np.linspace(0, 1e-1, 100)
bins = np.logspace(-8, -1, 100)
for col in cols:
    f, ax = plt.subplots(3, 2, figsize=(16, 12))
    ax = ax.reshape((np.prod(ax.shape),))
    # ax.hist(res_tbl.loc[res_tbl[disp_col] == disp][col], bins, edgecolor='k', label=f'{disp}')
    for disp_i, disp in enumerate(dispositions):
        ax[disp_i].hist(res_tbl.loc[res_tbl[disp_col] == disp][col], bins, edgecolor='k')
    # ax.legend()
        ax[disp_i].set_ylabel('Count')
    # ax[disp_i].set_xlabel(f'{col}')
        ax[disp_i].set_title(f'{disp}')
        # ax[disp_i].set_yscale('log')
        ax[disp_i].set_xscale('log')
    f.suptitle(f'{col}')
    # ax.set_yscale('log')
    # f.subplots_adjust(top=0.96,
# bottom=0.066,
# left=0.044,
# right=0.991,
# hspace=0.376,
# wspace=0.098)
    f.tight_layout()
    f.savefig(res_dir / 'analysis_plots' / f'hist_{col}_disp_linear.png')
    plt.close()

    f, ax = plt.subplots()
    ax.hist(res_tbl[col], bins, edgecolor='k')
    ax.set_ylabel('Count')
    ax.set_title(f'{col}')
    # ax.set_yscale('log')
    ax.set_xscale('log')
    f.tight_layout()
    f.savefig(res_dir / 'analysis_plots' / f'hist_{col}_linear.png')
    plt.close()

#%% analyze number of transits for flux and for odd and even flux

res_dir = Path('/data5/tess_project/Data/tfrecords/Kepler/Q1-Q17_DR25/tfrecordskeplerdr25-dv_g301-l31_spline_nongapped_flux-loe-lwks-centroid-centroid_fdl-scalars_oereplbins_data/tfrecordskeplerdr25-dv_g301-l31_spline_nongapped_flux-loe-lwks-centroid-centroid_fdl-scalars_oereplbins_caphap_stat_diff')
# res_dir = Path('/data5/tess_project/Data/tfrecords/TESS/tfrecordstess-dv_g301-l31_spline_nongapped_flux-loe-lwks-centroid-centroid_fdl-scalars_4-23-2021_data/tfrecordstess-dv_g301-l31_spline_nongapped_flux-loe-lwks-centroid-centroid_fdl-scalars_4-23-2021/')
res_tbl = pd.read_csv(res_dir / 'merged_shards_disp_set.csv')

cols = ['num_transits_flux_odd', 'num_transits_flux_even', 'num_transits_flux']
disp_col = 'label'
# dispositions = ['KP', 'APC', 'PC', 'CP', 'FA', 'FP']  # res_tbl[disp_col].unique()
dispositions = ['PC', 'AFP', 'NTP']

# bins = np.linspace(0, 1, 100)
bins = np.logspace(0, 3, 100)
for col in cols:
    f, ax = plt.subplots(3, 1, figsize=(16, 12))
    ax = ax.reshape((np.prod(ax.shape),))
    # ax.hist(res_tbl.loc[res_tbl[disp_col] == disp][col], bins, edgecolor='k', label=f'{disp}')
    for disp_i, disp in enumerate(dispositions):
        ax[disp_i].hist(res_tbl.loc[res_tbl[disp_col] == disp][col], bins, edgecolor='k')
    # ax.legend()
        ax[disp_i].set_ylabel('Count')
    # ax[disp_i].set_xlabel(f'{col}')
        ax[disp_i].set_title(f'{disp}')
        # ax[disp_i].set_yscale('log')
        ax[disp_i].set_xscale('log')
    f.suptitle(f'{col}')
    # ax.set_yscale('log')
    # f.subplots_adjust(top=0.96,
# bottom=0.066,
# left=0.044,
# right=0.991,
# hspace=0.376,
# wspace=0.098)
    f.tight_layout()
    f.savefig(res_dir / 'analysis_plots' / f'hist_{col}_disp_linear.png')
    plt.close()

    f, ax = plt.subplots()
    ax.hist(res_tbl[col], bins, edgecolor='k')
    ax.set_ylabel('Count')
    ax.set_title(f'{col}')
    # ax.set_yscale('log')
    ax.set_xscale('log')
    f.tight_layout()
    f.savefig(res_dir / 'analysis_plots' / f'hist_{col}_linear.png')
    plt.close()

#%% analyze MAD std it/ot for odd/even

# res_dir = Path('/data5/tess_project/Data/tfrecords/Kepler/Q1-Q17_DR25/tfrecordskeplerdr25-dv_g301-l31_spline_nongapped_flux-loe-lwks-centroid-centroid_fdl-scalars_oereplbins_data/tfrecordskeplerdr25-dv_g301-l31_spline_nongapped_flux-loe-lwks-centroid-centroid_fdl-scalars_oereplbins_caphap_stat_diff')
res_dir = Path('/data5/tess_project/Data/tfrecords/TESS/tfrecordstess-dv_g301-l31_spline_nongapped_flux-loe-lwks-centroid-centroid_fdl-scalars_4-23-2021_data/tfrecordstess-dv_g301-l31_spline_nongapped_flux-loe-lwks-centroid-centroid_fdl-scalars_4-23-2021/')
res_tbl = pd.read_csv(res_dir / 'merged_shards_disp_set.csv')

cols = [
    ('sigma_oot_odd', 'num_transits_flux_odd'),
    ('sigma_it_odd', 'num_transits_flux_odd'),
    ('sigma_oot_even', 'num_transits_flux_even'),
    ('sigma_it_even', 'num_transits_flux_even'),
        ]
disp_col = 'label'
dispositions = ['KP', 'APC', 'PC', 'CP', 'FA', 'FP']  # res_tbl[disp_col].unique()
# dispositions = ['PC', 'AFP', 'NTP']

# bins = np.linspace(0, 1e-1, 100)
bins = np.logspace(-4, 1, 100)
for col in cols:
    f, ax = plt.subplots(3, 2, figsize=(16, 12))
    ax = ax.reshape((np.prod(ax.shape),))
    # ax.hist(res_tbl.loc[res_tbl[disp_col] == disp][col], bins, edgecolor='k', label=f'{disp}')
    for disp_i, disp in enumerate(dispositions):
        ax[disp_i].hist(res_tbl.loc[res_tbl[disp_col] == disp][col[0]] * res_tbl.loc[res_tbl[disp_col] == disp][col[1]] ** 2, bins, edgecolor='k')
    # ax.legend()
        ax[disp_i].set_ylabel('Count')
    # ax[disp_i].set_xlabel(f'{col}')
        ax[disp_i].set_title(f'{disp}')
        # ax[disp_i].set_yscale('log')
        ax[disp_i].set_xscale('log')
    f.suptitle(f'{col[0]} non-normalized')
    # ax.set_yscale('log')
    # f.subplots_adjust(top=0.96,
# bottom=0.066,
# left=0.044,
# right=0.991,
# hspace=0.376,
# wspace=0.098)
    f.tight_layout()
    f.savefig(res_dir / 'analysis_plots' / f'hist_{col[0]}_nonnormalized_disp_linear.png')
    plt.close()

    f, ax = plt.subplots()
    ax.hist(res_tbl[col[0]] * res_tbl[col[1]] ** 2, bins, edgecolor='k')
    ax.set_ylabel('Count')
    ax.set_title(f'{col[0]} non-normalized')
    # ax.set_yscale('log')
    ax.set_xscale('log')
    f.tight_layout()
    f.savefig(res_dir / 'analysis_plots' / f'hist_{col[0]}_nonnormalized_linear.png')
    plt.close()

#%% compile histograms to pdf file

from fpdf import FPDF

plots_dir = Path('/home/msaragoc/Downloads/odd_even_sigma_plots')

imgs_fps = [
"hist_num_transits_flux_linear_kepler.png",
"hist_num_transits_flux_linear_tess.png",
"hist_num_transits_flux_disp_linear_kepler.png",
"hist_num_transits_flux_disp_linear_tess.png",

# "hist_num_transits_flux_even_disp_linear_kepler.png",
# "hist_num_transits_flux_even_disp_linear_tess.png",
# "hist_num_transits_flux_even_linear_kepler.png",
# "hist_num_transits_flux_even_linear_tess.png",

# "hist_num_transits_flux_odd_disp_linear_kepler.png",
# "hist_num_transits_flux_odd_disp_linear_tess.png",
# "hist_num_transits_flux_odd_linear_kepler.png",
# "hist_num_transits_flux_odd_linear_tess.png",

"hist_sigma_it_even_linear_kepler.png",
"hist_sigma_it_even_linear_tess.png",
"hist_sigma_it_even_disp_kepler_linear.png",
# "hist_sigma_it_even_disp_kepler.png",
"hist_sigma_it_even_disp_tess_linear.png",
# "hist_sigma_it_even_disp_tess.png",
# "hist_sigma_it_even_kepler.png",
# "hist_sigma_it_even_tess.png",

"hist_sigma_it_odd_linear_kepler.png",
"hist_sigma_it_odd_linear_tess.png",
# "hist_sigma_it_odd_disp_kepler.png",
"hist_sigma_it_odd_disp_linear_kepler.png",
"hist_sigma_it_odd_disp_linear_tess.png",
# "hist_sigma_it_odd_disp_tess.png",
# "hist_sigma_it_odd_kepler.png",
# "hist_sigma_it_odd_tess.png",

"hist_sigma_oot_even_linear_kepler.png",
"hist_sigma_oot_even_linear_tess.png",
# "hist_sigma_oot_even_disp_kepler.png",
"hist_sigma_oot_even_disp_linear_kepler.png",
"hist_sigma_oot_even_disp_linear_tess.png",
# "hist_sigma_oot_even_disp_tess.png",
# "hist_sigma_oot_even_kepler.png",
# "hist_sigma_oot_even_tess.png",

"hist_sigma_oot_odd_linear_kepler.png",
"hist_sigma_oot_odd_linear_tess.png",
# "hist_sigma_oot_odd_tess.png",
# "hist_sigma_oot_odd_disp_kepler.png",
"hist_sigma_oot_odd_disp_linear_kepler.png",
"hist_sigma_oot_odd_disp_linear_tess.png",
# "hist_sigma_oot_odd_disp_tess.png",
# "hist_sigma_oot_odd_kepler.png",

"hist_sigma_it_even_nonnormalized_linear_kepler.png",
"hist_sigma_it_even_nonnormalized_linear_tess.png",
"hist_sigma_it_even_nonnormalized_disp_linear_kepler.png",
"hist_sigma_it_even_nonnormalized_disp_linear_tess.png",

"hist_sigma_it_odd_nonnormalized_linear_kepler.png",
"hist_sigma_it_odd_nonnormalized_linear_tess.png",
"hist_sigma_it_odd_nonnormalized_disp_linear_kepler.png",
"hist_sigma_it_odd_nonnormalized_disp_linear_tess.png",

"hist_sigma_oot_even_nonnormalized_linear_kepler.png",
"hist_sigma_oot_even_nonnormalized_linear_tess.png",
"hist_sigma_oot_even_nonnormalized_disp_linear_kepler.png",
"hist_sigma_oot_even_nonnormalized_disp_linear_tess.png",

"hist_sigma_oot_odd_nonnormalized_linear_kepler.png",
"hist_sigma_oot_odd_nonnormalized_linear_tess.png",
"hist_sigma_oot_odd_nonnormalized_disp_linear_kepler.png",
"hist_sigma_oot_odd_nonnormalized_disp_linear_tess.png",

]

imgs_fps = [plots_dir / img_fn for img_fn in imgs_fps]

pdf = FPDF()
pdf.set_font('Arial', 'B', 12)
# for img_fp in sorted(plots_dir.iterdir()):
for img_fp in imgs_fps:
    pdf.add_page()
    if 'num_transits' in img_fp.name:
        text = 'Number of transits'
    elif 'nonnormalized' in img_fp.name:
        text = 'MAD std'
    else:
        text = 'Standard error of the mean (MAD std / sqrt(num_transits))'

    if 'disp' in img_fp.name:
        text += ' per disposition'

    if 'kepler' in img_fp.name:
        text += ' - Kepler'
    elif 'tess' in img_fp.name:
        text += ' - TESS'

    pdf.cell(0, 0, text, align='C')
    pdf.image(str(img_fp), 5, 25, 200, 150)
    # break
pdf.output('/home/msaragoc/Downloads/num_transits_sigma_oddeven_keplervstess_4-29-2021.pdf', 'F')

#%% analyze MAD std normalized by transit depth for it/ot for odd/even

res_dir = Path('/data5/tess_project/Data/tfrecords/Kepler/Q1-Q17_DR25/tfrecordskeplerdr25-dv_g301-l31_spline_nongapped_flux-loe-lwks-centroid-centroid_fdl-scalars_oereplbins_data/tfrecordskeplerdr25-dv_g301-l31_spline_nongapped_flux-loe-lwks-centroid-centroid_fdl-scalars_oereplbins_caphap_stat_diff')
# res_dir = Path('/data5/tess_project/Data/tfrecords/TESS/tfrecordstess-dv_g301-l31_spline_nongapped_flux-loe-lwks-centroid-centroid_fdl-scalars_4-23-2021_data/tfrecordstess-dv_g301-l31_spline_nongapped_flux-loe-lwks-centroid-centroid_fdl-scalars_4-23-2021/')
res_tbl = pd.read_csv(res_dir / 'merged_shards_disp_set.csv')

cols = [
    ('sigma_oot_odd', 'num_transits_flux_odd', 'transit_depth_hat'),
    ('sigma_it_odd', 'num_transits_flux_odd', 'transit_depth_hat'),
    ('sigma_oot_even', 'num_transits_flux_even', 'transit_depth_hat'),
    ('sigma_it_even', 'num_transits_flux_even', 'transit_depth_hat'),
        ]
disp_col = 'label'
# dispositions = ['KP', 'APC', 'PC', 'CP', 'FA', 'FP']  # res_tbl[disp_col].unique()
dispositions = ['PC', 'AFP', 'NTP']

# bins = np.linspace(0, 1e-1, 100)
bins = np.logspace(-6, -2, 100)
for col in cols:
    f, ax = plt.subplots(3, 1, figsize=(16, 12))
    ax = ax.reshape((np.prod(ax.shape),))
    # ax.hist(res_tbl.loc[res_tbl[disp_col] == disp][col], bins, edgecolor='k', label=f'{disp}')
    for disp_i, disp in enumerate(dispositions):
        ax[disp_i].hist(res_tbl.loc[res_tbl[disp_col] == disp][col[0]] *
                        (res_tbl.loc[res_tbl[disp_col] == disp][col[1]] ** 2) /
                        res_tbl.loc[res_tbl[disp_col] == disp][col[2]], bins, edgecolor='k')
    # ax.legend()
        ax[disp_i].set_ylabel('Count')
    # ax[disp_i].set_xlabel(f'{col}')
        ax[disp_i].set_title(f'{disp}')
        # ax[disp_i].set_yscale('log')
        ax[disp_i].set_xscale('log')
    f.suptitle(f'{col[0]} MAD normalized by min flux')
    # ax.set_yscale('log')
    # f.subplots_adjust(top=0.96,
# bottom=0.066,
# left=0.044,
# right=0.991,
# hspace=0.376,
# wspace=0.098)
    f.tight_layout()
    f.savefig(res_dir / 'analysis_plots' / f'hist_{col[0]}_nonnormalized_minflux_disp_linear.png')
    plt.close()

    f, ax = plt.subplots()
    ax.hist(res_tbl[col[0]] * (res_tbl[col[1]] ** 2) / res_tbl[col[2]], bins, edgecolor='k')
    ax.set_ylabel('Count')
    ax.set_title(f'{col[0]} MAD normalized by min flux')
    # ax.set_yscale('log')
    ax.set_xscale('log')
    f.tight_layout()
    f.savefig(res_dir / 'analysis_plots' / f'hist_{col[0]}_nonnormalized_minflux_linear.png')
    plt.close()

#%% analyze sigma normalized by transit depth for it/ot for odd/even

res_dir = Path('/data5/tess_project/Data/tfrecords/Kepler/Q1-Q17_DR25/tfrecordskeplerdr25-dv_g301-l31_spline_nongapped_flux-loe-lwks-centroid-centroid_fdl-scalars_oereplbins_data/tfrecordskeplerdr25-dv_g301-l31_spline_nongapped_flux-loe-lwks-centroid-centroid_fdl-scalars_oereplbins_caphap_stat_diff')
# res_dir = Path('/data5/tess_project/Data/tfrecords/TESS/tfrecordstess-dv_g301-l31_spline_nongapped_flux-loe-lwks-centroid-centroid_fdl-scalars_4-23-2021_data/tfrecordstess-dv_g301-l31_spline_nongapped_flux-loe-lwks-centroid-centroid_fdl-scalars_4-23-2021/')
res_tbl = pd.read_csv(res_dir / 'merged_shards_disp_set.csv')

cols = [
    ('sigma_oot_odd', 'transit_depth_hat'),
    ('sigma_it_odd', 'transit_depth_hat'),
    ('sigma_oot_even', 'transit_depth_hat'),
    ('sigma_it_even', 'transit_depth_hat'),
        ]
disp_col = 'label'
# dispositions = ['KP', 'APC', 'PC', 'CP', 'FA', 'FP']  # res_tbl[disp_col].unique()
dispositions = ['PC', 'AFP', 'NTP']

# bins = np.linspace(0, 1e-1, 100)
bins = np.logspace(-9, -3, 100)
for col in cols:
    f, ax = plt.subplots(3, 1, figsize=(16, 12))
    ax = ax.reshape((np.prod(ax.shape),))
    # ax.hist(res_tbl.loc[res_tbl[disp_col] == disp][col], bins, edgecolor='k', label=f'{disp}')
    for disp_i, disp in enumerate(dispositions):
        ax[disp_i].hist(res_tbl.loc[res_tbl[disp_col] == disp][col[0]] /
                        res_tbl.loc[res_tbl[disp_col] == disp][col[1]], bins, edgecolor='k')
    # ax.legend()
        ax[disp_i].set_ylabel('Count')
    # ax[disp_i].set_xlabel(f'{col}')
        ax[disp_i].set_title(f'{disp}')
        # ax[disp_i].set_yscale('log')
        ax[disp_i].set_xscale('log')
    f.suptitle(f'{col[0]} normalized by min flux')
    # ax.set_yscale('log')
    # f.subplots_adjust(top=0.96,
# bottom=0.066,
# left=0.044,
# right=0.991,
# hspace=0.376,
# wspace=0.098)
    f.tight_layout()
    f.savefig(res_dir / 'analysis_plots' / f'hist_{col[0]}_minflux_disp_linear.png')
    plt.close()

    f, ax = plt.subplots()
    ax.hist(res_tbl[col[0]] / res_tbl[col[1]], bins, edgecolor='k')
    ax.set_ylabel('Count')
    ax.set_title(f'{col[0]} normalized by min flux')
    # ax.set_yscale('log')
    ax.set_xscale('log')
    f.tight_layout()
    f.savefig(res_dir / 'analysis_plots' / f'hist_{col[0]}_minflux_linear.png')
    plt.close()
