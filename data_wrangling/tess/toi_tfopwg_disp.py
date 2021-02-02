""" Script used to add TFOPWG disposition to rankings."""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

#%%

toi_exofop_tbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/TESS/NASA_Exoplanet_Archive_TOI_lists/'
                              'TOI_2021.01.12_15.08.41.csv', header=71)
matching_tbl = pd.read_csv('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Analysis/toi_tce_matching/'
                           'tois_matchedtces_ephmerismatching_thrinf_samplint1e-05_1-8-2021.csv')

ranking_tbl_fp = Path('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Kepler_planet_finder/results_ensemble/'
                          'tess_g301-l31_6tr_spline_nongapped_spoctois_configK_wsphase/'
                          'ensemble_ranked_predictions_predictset.csv')
ranking_tbl = pd.read_csv(ranking_tbl_fp)

ranking_tbl['tfopwg_disp'] = np.nan
ranking_tbl['matching_dist'] = np.nan

for toi_i, toi in ranking_tbl.iterrows():

    toi_id = str(np.round(toi['oi'], 2)).split('.')
    toi_id = float(f'{toi_id[0]}.{toi_id[1]}')

    toi_found_tfopwg = toi_exofop_tbl.loc[toi_exofop_tbl['toi'] == toi_id]
    if len(toi_found_tfopwg) == 1:
        ranking_tbl.loc[toi_i, 'tfopwg_disp'] = toi_found_tfopwg['tfopwg_disp'].values

    toi_found_match = matching_tbl.loc[matching_tbl['Full TOI ID'] == toi_id]
    if len(toi_found_match) == 1:
        ranking_tbl.loc[toi_i, 'matching_dist'] = toi_found_match['matching_dist_0'].values

ranking_tbl['oi'] = ranking_tbl['oi'].apply(lambda x: str(np.round(x, 2)))
ranking_tbl.to_csv(ranking_tbl_fp.parent / f'{ranking_tbl_fp.stem}_tfopwg_disp_matchingdist.csv', index=False)

#%%

ranking_tbl = ranking_tbl.loc[ranking_tbl['original_label'] == 'KP']

bins = np.linspace(0, 1, 21)

f, ax = plt.subplots()
ax.hist(ranking_tbl.loc[ranking_tbl['predicted class'] == 1, 'matching_dist'], bins, label=f'Correctly classified',
        edgecolor='k', zorder=1)
ax.hist(ranking_tbl.loc[ranking_tbl['predicted class'] == 0, 'matching_dist'], bins, label=f'Misclassified',
        edgecolor='k', alpha=0.3, zorder=2)
ax.set_xlabel('Matching distance')
ax.set_ylabel('Counts')
ax.set_yscale('log')
ax.set_xlim([0, 1])
ax.legend()
ax.set_title('Known Planets')
f.savefig(ranking_tbl_fp.parent / 'hist_KP_matchingdist_class.png')
