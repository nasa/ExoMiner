""" Update TOI labels for TCEs based on TOI-TCE DV matching. """

# 3rd party
from pathlib import Path
import numpy as np
import pandas as pd
import logging

#%%

save_dir = Path('/Users/msaragoc/Downloads/toi-tce_matching_dv')
save_dir.mkdir(exist_ok=True)

# set up logger
logger = logging.getLogger()
logger_handler = logging.FileHandler(filename=save_dir / f'update_tce_tbl_toi_label.log', mode='w')
logger_formatter = logging.Formatter('%(asctime)s - %(message)s')
logger.setLevel(logging.INFO)
logger_handler.setFormatter(logger_formatter)
logger.addHandler(logger_handler)
logger.info(f'Starting run...')

tce_tbl_fp = Path('/Users/msaragoc/Downloads/toi-tce_matching_dv/tess_tces_s1-s40_11-23-2021_1409_stellarparams_updated_eb_tso_tec_label_modelchisqr_astronet_ruwe_magcat_uid_corrtsoebs_corraltdetfail.csv')
tce_tbl = pd.read_csv(tce_tbl_fp)
logger.info(f'Using TCE table: {tce_tbl_fp}')

tce_tbl.loc[tce_tbl['label_source'] == 'TFOPWG Disposition', 'label_source'] = np.nan

logger.info(f'TCE disposition counts before update:\n {tce_tbl["label"].value_counts()}')

toi_tbl_fp = Path('/Users/msaragoc/Downloads/toi-tce_matching_dv/exofop_toi_cat_9-7-2022.csv')
toi_tbl = pd.read_csv(toi_tbl_fp)
logger.info(f'Using TOI catalog: {toi_tbl_fp}')

cnt_updates = 0
cnt_unks_updated = 0
for toi_i, toi in toi_tbl.iterrows():
    tces_found = tce_tbl['toi_dv'] == toi['TOI']
    if tces_found.sum() == 0:
        continue

    cnt_updates += (tce_tbl.loc[tces_found, 'label'] != toi['TFOPWG Disposition']).sum()
    cnt_unks_updated += ((tce_tbl.loc[tces_found, 'label'] != toi['TFOPWG Disposition']) &
                         (tce_tbl.loc[tces_found, 'label'] == 'UNK')).sum()

    tce_tbl.loc[tces_found, ['label', 'TFOPWG Disposition']] = toi['TFOPWG Disposition']
    tce_tbl.loc[tces_found, 'label_source'] = 'TFOPWG Disposition'

logger.info(f'Number of unlabeled TCEs that changed their disposition: {cnt_unks_updated}')
logger.info(f'Number of TCEs that changed their disposition: {cnt_updates}')

tce_tbl.loc[tce_tbl['label_source'].isna(), ['label', 'label_source', 'TFOPWG Disposition']] = 'UNK', 'N/A', np.nan
logger.info(f'TCE disposition counts after update:\n {tce_tbl["label"].value_counts()}')

tce_tbl.to_csv(save_dir / f'{tce_tbl_fp.stem}_toidv.csv', index=False)

#%% update TCE labels from other sources of dispositions

# set up logger
logger = logging.getLogger()
logger_handler = logging.FileHandler(filename=save_dir / f'update_tce_tbl_other_sources.log', mode='w')
logger_formatter = logging.Formatter('%(asctime)s - %(message)s')
logger.setLevel(logging.INFO)
logger_handler.setFormatter(logger_formatter)
logger.addHandler(logger_handler)
logger.info(f'Starting run...')

tce_tbl_fp = Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/data/ephemeris_matching/toi-tce_matching_dv/tess_tces_s1-s40_11-23-2021_1409_stellarparams_updated_eb_tso_tec_label_modelchisqr_astronet_ruwe_magcat_uid_corrtsoebs_corraltdetfail_toidv.csv')
tce_tbl = pd.read_csv(tce_tbl_fp)
logger.info(f'Using TCE table: {tce_tbl_fp}')

tce_tbl.loc[tce_tbl['label_source'] != 'TFOPWG Disposition', ['label', 'label_source']] = 'UNK', 'N/A'

logger.info(f'TCE disposition counts before update:\n {tce_tbl["label"].value_counts()}')
logger.info(f'TCE label source counts before update:\n {tce_tbl["label_source"].value_counts()}')

# 2) match TCEs to TSO EBs
# tce_tbl.loc[(tce_tbl['tso_eb'] == 'yes') & (tce_tbl['label'] == 'UNK'), ['label', 'label_source']] = 'EB', 'TSO EB'
tce_tbl.loc[(tce_tbl['in_jon_spoc_ebs'] == 'yes') &
            (tce_tbl['label'] == 'UNK'), ['label', 'label_source']] = 'EB', 'TSO EB'

# # 3) match TCEs to Jon's EBs
# tce_tbl.loc[(tce_tbl['eb_match_dist'] < matching_thr) &
#             (tce_tbl['label'] == 'UNK'), ['label', 'label_source']] = 'EB', 'Jon\'s EBs'

# 4) create NTPs based on TEC flux triage
tce_tbl.loc[(tce_tbl['tec_fluxtriage_pass'] == 0) &
            (~tce_tbl['tec_fluxtriage_comment'].str.contains('SecondaryOfPN', na=False)) &
            (tce_tbl['label'] == 'UNK'), ['label', 'label_source']] = 'NTP', 'TEC flux triage'
# set to UNK those TCEs that did not pass the TEC flux triage because they failed AltDet and their period is less or
# equal to 0.3 days
tce_tbl.loc[(tce_tbl['tec_fluxtriage_pass'] == 0) &
            (tce_tbl['tec_fluxtriage_comment'] == 'AltDetFail') &
            (tce_tbl['label_source'] == 'TEC flux triage') & (tce_tbl['tce_period'] <= 0.3),
            ['label', 'label_source']] = 'UNK', 'N/A'

logger.info(f'TCE disposition counts after update:\n {tce_tbl["label"].value_counts()}')
logger.info(f'TCE label source counts after update:\n {tce_tbl["label_source"].value_counts()}')

tce_tbl.to_csv(save_dir / f'{tce_tbl_fp.stem}_final.csv', index=False)
