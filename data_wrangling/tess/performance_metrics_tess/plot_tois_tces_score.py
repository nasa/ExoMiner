"""
Plot scores of all TCEs associated with a given TOI as function of variables such as number of observed transits.
"""

# 3rd party
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

#%% plot score as function of number of observed transits

tce_tbl = pd.read_csv('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/data/ephemeris_tables/tess/DV_SPOC_mat_files/10-05-2022_1338/tess_tces_dv_s1-s55_10-05-2022_1338_ticstellar_ruwe_tec_tsoebs_ourmatch_preproc.csv')
# toi_tbl = pd.read_csv('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/data/ephemeris_tables/tess/EXOFOP_TOI_lists/TOI/10-3-2022/exofop_tess_tois-4.csv')
# toi_tbl = toi_tbl.rename(columns={'TOI': 'matched_toi_our'})

# toi_cols = [
#     'TFOPWG Disposition',
#     'TESS Disposition',
#     'Period (days)',
#     'Duration (hours)',
#     'Depth (ppm)',
#     # 'Epoch (BJD)',
#     'Planet Radius (R_Earth)',
#     'Planet SNR',
#     'Spectroscopy Observations',
#     'Imaging Observations',
#     'Time Series Observations',
#     'Comments',
# ]
# tce_tbl = tce_tbl.drop(columns=toi_cols)
# tce_tbl = tce_tbl.merge(toi_tbl[['matched_toi_our'] + toi_cols], on='matched_toi_our', how='left', validate='many_to_one')
# tce_tbl.to_csv('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/data/ephemeris_tables/tess/DV_SPOC_mat_files/10-05-2022_1338/tess_tces_dv_s1-s55_10-05-2022_1338_ticstellar_ruwe_tec_tsoebs_ourmatch_preproc.csv', index=False)

# tois_tces = tce_tbl.loc[tce_tbl['matched_toi_our'].isin(tois)]

# experiment directory
exp_dir = Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/experiments/cv_kepler-tess_1-24-2023_1126')
plot_dir = exp_dir / 'tois_num_obs_transits_v_score'
plot_dir.mkdir(exist_ok=True)

ranking_tbl = pd.read_csv(exp_dir / 'ensemble_ranked_predictions_allfolds.csv')

ranking_tbl = ranking_tbl.merge(tce_tbl[['uid', 'tce_num_transits_obs', 'matched_toi_our', 'match_corr_coef',
                                         'sector_run', 'tce_model_snr', 'Period (days)', 'Depth (ppm)',
                                         'Planet Radius (R_Earth)', 'Planet SNR', 'tce_steff', 'tce_slogg', 'mag',
                                         'tce_smet',
                                         # 'ruwe',
                                         'Comments']],
                                on='uid', how='left', validate='one_to_one')

tois = ranking_tbl.loc[~ranking_tbl['matched_toi_our'].isna(), 'matched_toi_our'].unique()

cats_plot = {'T-CP': 1, 'T-KP': 1, 'T-FP': 0, 'T-FA': 0}

for cat in cats_plot:

    plot_dir_cat = plot_dir / cat
    plot_dir_cat.mkdir(exist_ok=True)

    ranking_tbl_aux = ranking_tbl.loc[ranking_tbl['original_label'] == cat]

    for toi in tois:

        tces_in_toi_tbl = ranking_tbl_aux.loc[ranking_tbl_aux['matched_toi_our'] == toi]

        if len(tces_in_toi_tbl) <= 1:
            continue

        f, ax = plt.subplots(figsize=(12, 8))
        divider = make_axes_locatable(ax)
        ax.plot(tces_in_toi_tbl['tce_num_transits_obs'], tces_in_toi_tbl['score'], zorder=1)
        im = ax.scatter(tces_in_toi_tbl['tce_num_transits_obs'], tces_in_toi_tbl['score'],
                        c=tces_in_toi_tbl['match_corr_coef'], cmap=plt.cm.coolwarm, zorder=2)
        for tce_i in range(len(tces_in_toi_tbl)):
            plt.text(x=tces_in_toi_tbl['tce_num_transits_obs'].values[tce_i], y=tces_in_toi_tbl['score'].values[tce_i],
                     s=tces_in_toi_tbl['sector_run'].values[tce_i], zorder=3, size='small')
            # plt.text(x=500, y=2.5e-6, s='aaaaa', color='black', size='medium')

        ax.set_xlabel('TCE Number Observed Transits')
        ax.set_ylabel('Model Score')
        ax.set_title(f'TOI {toi} {tces_in_toi_tbl["original_label"].values[0]}\n'
                     f'Period (days)={tces_in_toi_tbl["Period (days)"].values[0]}, '
                     f'Depth (ppm)={tces_in_toi_tbl["Depth (ppm)"].values[0]}, '
                     f'Planet Radius (R_Earth)={tces_in_toi_tbl["Planet Radius (R_Earth)"].values[0]}, '
                     f'Planet SNR={tces_in_toi_tbl["Planet SNR"].values[0]}\n'
                     f'TIC {tces_in_toi_tbl["target_id"].values[0]}  '
                     f'Teff={tces_in_toi_tbl["tce_steff"].values[0]}, '
                     f'Met={tces_in_toi_tbl["tce_smet"].values[0]}, '
                     f'Logg={tces_in_toi_tbl["tce_slogg"].values[0]} '
                     f'TMag={tces_in_toi_tbl["mag"].values[0]} '
                     f'RUWE={tces_in_toi_tbl["ruwe"].values[0]}\n'
                     f'Comment:{tces_in_toi_tbl["Comments"].values[0]}')
        ax.set_xlim(right=tces_in_toi_tbl['tce_num_transits_obs'].max()*1.03)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        f.colorbar(im, cax)
        # plt.text(x=500, y=2.5e-6, s='aaaaa', color='black', size='medium')
        # aaa
        f.savefig(plot_dir_cat / f'toi{toi}_num_obs_transits-v-score.png')
        plt.close()

