# 3rd party
from pathlib import Path

import numpy as np
import pandas as pd

# %% Assigning TESS TCEs as NTPs based on TEC flux triage tables

tce_tbl_fp = Path(
    '/data5/tess_project/Data/Ephemeris_tables/TESS/DV_SPOC_mat_files/11-29-2021/tess_tces_s1-s40_11-23-2021_1409_stellarparams_updated.csv')
# tce_tbl_cols = ['target_id', 'tce_plnt_num', 'sector_run', 'tce_period', 'tce_time0bk', 'tce_duration', 'match_dist',
#                 'TFOPWG Disposition', 'TESS Disposition']
tce_tbl = pd.read_csv(tce_tbl_fp)  # [tce_tbl_cols]
# tce_tbl.to_csv(res_dir / tce_tbl_fp.name, index=False)

tec_tbl = pd.read_csv(
    '/data5/tess_project/Data/Ephemeris_tables/TESS/TEC_SPOC/tec_tbl_fluxtriage_s1-s41_10-29-2021.csv')

tce_tbl_tec = tce_tbl.merge(tec_tbl, on=['target_id', 'tce_plnt_num', 'sector_run'], how='left', validate='one_to_one')

# # assignment rule: 1) TCE did not pass flux triage; AND 2) the matching distance is larger than 0.3
# tce_tbl_tec.loc[(tce_tbl_tec['tec_fluxtriage_pass'] == 0) &
#                 ((tce_tbl_tec['match_dist'] > 0.3) | (tce_tbl_tec['match_dist'].isna())), 'label'] = 'NTP'

tce_tbl_tec.to_csv(tce_tbl_fp.parent / f'{tce_tbl_fp.stem}_tecfluxtriage.csv', index=False)

# %% Add EB matching distance to TCE table

tce_tbl_fp = Path(
    '/data5/tess_project/Data/Ephemeris_tables/TESS/DV_SPOC_mat_files/11-29-2021/tess_tces_s1-s40_11-23-2021_1409_stellarparams_updated_tecfluxtriage.csv')
tce_tbl = pd.read_csv(tce_tbl_fp)

matching_eb_tbl = pd.read_csv(
    '/home/msaragoc/Projects/Kepler-TESS_exoplanet/Analysis/tess_tce_eb_match/11-29-2021_1457/matching_tbl_thr_0.2_sameperiod_smallestplntnum.csv')
eb_final_cols = ['eb_signal_id', 'eb_bjd0', 'eb_period', 'tce_eb_period_multiple_int', 'tce_eb_period_ratio',
                 'eb_match_dist']
matching_eb_tbl = matching_eb_tbl[['target_id', 'tce_plnt_num', 'sector_run'] + eb_final_cols]

tce_tbl_eb = tce_tbl.merge(matching_eb_tbl, on=['target_id', 'tce_plnt_num', 'sector_run'], how='left',
                           validate='one_to_one')

tce_tbl_eb.to_csv(tce_tbl_fp.parent / f'{tce_tbl_fp.stem}_eb.csv', index=False)

# %% Decide about disposition based on different sources of labels

tce_tbl_fp = Path(
    '/data5/tess_project/Data/Ephemeris_tables/TESS/DV_SPOC_mat_files/11-29-2021/tess_tces_s1-s40_11-23-2021_1409_stellarparams_updated_tecfluxtriage_eb.csv')
tce_tbl = pd.read_csv(tce_tbl_fp)

matching_thr = 0.2
matching_thr_ntp = 1
tce_tbl['label'] = 'UNK'
tce_tbl['label_comment'] = 'N/A'
tce_tbl.loc[tce_tbl['match_dist'].isna(), 'match_dist'] = np.inf
tce_tbl.loc[tce_tbl['eb_match_dist'].isna(), 'eb_match_dist'] = np.inf

for tce_i, tce in tce_tbl.iterrows():

    if tce['match_dist'] <= tce['eb_match_dist'] and tce['match_dist'] <= matching_thr:
        tce_tbl.loc[tce_i, 'label'] = tce['TFOPWG Disposition']
        tce_tbl.loc[tce_i, 'label_comment'] = 'label from TFOPWG'
    elif tce['match_dist'] > tce['eb_match_dist'] and tce['eb_match_dist'] <= matching_thr and tce[
        'TFOPWG Disposition'] not in ['KP', 'CP']:
        tce_tbl.loc[tce_i, 'label'] = 'FP'
        tce_tbl.loc[tce_i, 'label_comment'] = 'label from EB'
    elif tce['tec_fluxtriage_comment'] in ['AltDetFail', 'ses2MesFail', 'ChasesFail', 'newMesBelowThresh',
                                           'newMesShrink', 'SesStatFail'] and tce['match_dist'] >= matching_thr_ntp and \
            tce['eb_match_dist'] >= matching_thr_ntp:
        tce_tbl.loc[tce_i, 'label'] = 'NTP'
        tce_tbl.loc[tce_i, 'label_comment'] = 'label from TEC'

tce_tbl.to_csv(tce_tbl_fp.parent / f'{tce_tbl_fp.stem}_label.csv', index=False)
