"""
Combining different sources of dispositions and information that can be used to define the labels of TCEs.
Sources of dispositions:
1) TSO EB catalog
2) EB catalog shared by Jon
3) TEC flux triage table
"""

# 3rd party
import logging
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

#%% Adding TSO EBs to TCE table

tce_tbl_fp = Path('/Users/msaragoc/OneDrive - NASA/Projects/exoplanet_transit_classification/data/ephemeris_tables/tess/DV_SPOC_mat_files/11-29-2021/tess_tces_s1-s40_11-23-2021_1409_stellarparams_updated.csv')
tce_tbl = pd.read_csv(tce_tbl_fp)

eb_cat = pd.read_csv('/Users/msaragoc/OneDrive - NASA/Projects/exoplanet_transit_classification/data/ephemeris_tables/tess/eb_catalogs/eb_catalog_tso/eb_list_tso_12-16-2021_processed.csv')
eb_cat = eb_cat[['tic_id', 'candidate_n', 'sector_run']]
eb_cat = eb_cat.rename(columns={'tic_id': 'target_id', 'candidate_n': 'tce_plnt_num'})
eb_cat['tso_eb'] = 'yes'

tce_tbl = tce_tbl.merge(eb_cat, on=['target_id', 'tce_plnt_num', 'sector_run'], how='left', validate='one_to_one')
tce_tbl.loc[tce_tbl['tso_eb'].isna(), 'tso_eb'] = 'no'

# tce_tbl.to_csv(tce_tbl_fp.parent / f'{tce_tbl_fp.stem}_tsoebs.csv', index=False)

#%% Adding EBs from Jon's catalog to TCE table

# tce_tbl_fp = Path('/Users/msaragoc/OneDrive - NASA/Projects/exoplanet_transit_classification/data/ephemeris_tables/tess/DV_SPOC_mat_files/11-29-2021/tess_tces_s1-s40_11-23-2021_1409_stellarparams_updated.csv')
# tce_tbl = pd.read_csv(tce_tbl_fp)

eb_matching_tbl = pd.read_csv('/Users/msaragoc/OneDrive - NASA/Projects/exoplanet_transit_classification/Analysis/tess_tce_eb_match/01-04-2022_0658/matching_tbl_ephemthr_inf_similarperiodthr_0.05_smallestplntnum.csv')
# filter TCEs that are not potentially matched to any EB
eb_matching_tbl = eb_matching_tbl.loc[~eb_matching_tbl['eb_signal_id'].isna()]
eb_matching_tbl = eb_matching_tbl[['target_id', 'tce_plnt_num', 'sector_run', 'eb_bjd0', 'eb_period', 'eb_match_dist']]
# eb_matching_tbl['jon_eb'] = 'yes'

tce_tbl = tce_tbl.merge(eb_matching_tbl, on=['target_id', 'tce_plnt_num', 'sector_run'], how='left', validate='one_to_one')
# tce_tbl.loc[tce_tbl['jon_eb'].isna()] = 'no'

# tce_tbl.to_csv(tce_tbl_fp.parent / f'{tce_tbl_fp.stem}_ebs.csv', index=False)

# %% Adding TECs flux triage to TCE table

# tce_tbl_fp = Path('/data5/tess_project/Data/Ephemeris_tables/TESS/DV_SPOC_mat_files/11-29-2021/tess_tces_s1-s40_11-23-2021_1409_stellarparams_updated.csv')
# tce_tbl_cols = ['target_id', 'tce_plnt_num', 'sector_run', 'tce_period', 'tce_time0bk', 'tce_duration', 'match_dist',
#                 'TFOPWG Disposition', 'TESS Disposition']
# tce_tbl = pd.read_csv(tce_tbl_fp)  # [tce_tbl_cols]

tec_tbl = pd.read_csv('/Users/msaragoc/OneDrive - NASA/Projects/exoplanet_transit_classification/data/ephemeris_tables/tess/tec/tec_tbl_fluxtriage_s1-s41_10-29-2021.csv')

tce_tbl = tce_tbl.merge(tec_tbl, on=['target_id', 'tce_plnt_num', 'sector_run'], how='left', validate='one_to_one')

# # assignment rule: 1) TCE did not pass flux triage; AND 2) the matching distance is larger than 0.3
# tce_tbl.loc[(tce_tbl['tec_fluxtriage_pass'] == 0) &
#                 ((tce_tbl['match_dist'] > 0.3) | (tce_tbl['match_dist'].isna())), 'label'] = 'NTP'

# tce_tbl.to_csv(tce_tbl_fp.parent / f'{tce_tbl_fp.stem}_tecfluxtriage.csv', index=False)

# %%

tce_tbl.to_csv(tce_tbl_fp.parent / f'{tce_tbl_fp.stem}_eb_tso_tec.csv', index=False)

#%% Decide about disposition based on different sources of labels

# load TCE table
tce_tbl_fp = Path('/Users/msaragoc/OneDrive - NASA/Projects/exoplanet_transit_classification/data/'
                  'ephemeris_tables/tess/DV_SPOC_mat_files/11-29-2021/'
                  'tess_tces_s1-s40_11-23-2021_1409_stellarparams_updated_eb_tso_tec.csv')
tce_tbl = pd.read_csv(tce_tbl_fp)

# res_dir = Path(tce_tbl_fp.parent / f'{datetime.now().strftime("%m-%d-%Y_%H%M")}')
# res_dir.mkdir(exist_ok=True)
#
# logger = logging.getLogger(name=f'disposition_to_label')
# logger_handler = logging.FileHandler(filename=res_dir / f'disposition_to_label.log', mode='w')
# logger_formatter = logging.Formatter('%(asctime)s - %(message)s')
# logger.setLevel(logging.INFO)
# logger_handler.setFormatter(logger_formatter)
# logger.addHandler(logger_handler)
# logger.info(f'Setting labels of TCEs in {str(res_dir)}')
# logger.info(f'Using as TCE table: {str(tce_tbl_fp)}')

# initialize label field
tce_tbl['label'] = 'UNK'
tce_tbl['label_source'] = 'N/A'

matching_thr = 0.25  # threshold used for ephemeris matching

# 1) match TCEs to TOIs
tce_tbl.loc[tce_tbl['match_dist'] < matching_thr, ['label', 'label_source']] = \
    tce_tbl.loc[tce_tbl['match_dist'] < matching_thr, 'TFOPWG Disposition'], 'TFOPWG Disposition'

# 2) match TCEs to TSO EBs
tce_tbl.loc[(tce_tbl['tso_eb'] == 'yes') & (tce_tbl['label'] == 'UNK'), ['label', 'label_source']] = 'EB', 'TSO EB'

# 3) match TCEs to Jon's EBs
tce_tbl.loc[(tce_tbl['eb_match_dist'] < matching_thr) &
            (tce_tbl['label'] == 'UNK'), ['label', 'label_source']] = 'EB', 'Jon\'s EBs'

# 4) create NTPs based on TEC flux triage
tce_tbl.loc[(tce_tbl['tec_fluxtriage_pass'] == 0) & (~tec_tbl['tec_fluxtriage_comment'].str.contains('SecondaryOfPN')) &
            (tce_tbl['label'] == 'UNK'), ['label', 'label_source']] = 'NTP', 'TEC flux triage'

print(tce_tbl['label'].value_counts())
print(tce_tbl['label_source'].value_counts())

# matching_thr_ntp = 1
#
# tce_tbl.loc[tce_tbl['match_dist'].isna(), 'match_dist'] = np.inf
# tce_tbl.loc[tce_tbl['eb_match_dist'].isna(), 'eb_match_dist'] = np.inf
#
# for tce_i, tce in tce_tbl.iterrows():
#
#     if tce['match_dist'] <= tce['eb_match_dist'] and tce['match_dist'] <= matching_thr:
#         tce_tbl.loc[tce_i, 'label'] = tce['TFOPWG Disposition']
#         tce_tbl.loc[tce_i, 'label_comment'] = 'label from TFOPWG'
#     elif tce['match_dist'] > tce['eb_match_dist'] and tce['eb_match_dist'] <= matching_thr and tce[
#         'TFOPWG Disposition'] not in ['KP', 'CP']:
#         tce_tbl.loc[tce_i, 'label'] = 'FP'
#         tce_tbl.loc[tce_i, 'label_comment'] = 'label from EB'
#     elif tce['tec_fluxtriage_comment'] in ['AltDetFail', 'ses2MesFail', 'ChasesFail', 'newMesBelowThresh',
#                                            'newMesShrink', 'SesStatFail'] and tce['match_dist'] >= matching_thr_ntp and \
#             tce['eb_match_dist'] >= matching_thr_ntp:
#         tce_tbl.loc[tce_i, 'label'] = 'NTP'
#         tce_tbl.loc[tce_i, 'label_comment'] = 'label from TEC'

tce_tbl.to_csv(tce_tbl_fp.parent / f'{tce_tbl_fp.stem}_label.csv', index=False)
