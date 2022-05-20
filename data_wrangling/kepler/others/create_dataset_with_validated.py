""" Script used to add validated planets by Armstrong et al. and us (ExoMiner 2021) to the working dataset. """

# 3rd party
from pathlib import Path
import pandas as pd
import logging
from datetime import datetime

res_dir = Path(f'/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17_DR25/'
               f'{datetime.now().strftime("%m-%d-%Y_%H%M")}/')
res_dir.mkdir(exist_ok=True)

# set up logger
logger = logging.getLogger(name='create_dataset')
logger_handler = logging.FileHandler(filename=res_dir / f'create_dataset.log', mode='w')
logger_formatter = logging.Formatter('%(asctime)s - %(message)s')
logger.setLevel(logging.INFO)
logger_handler.setFormatter(logger_formatter)
logger.addHandler(logger_handler)
logger.info(f'Starting run...')

# %%
# TCE table with all 34032 TCEs for Q1-Q17 DR25
tce_tbl_fp = Path('/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17_DR25/'
                  '19-08-21_07:21/q1_q17_dr25_tce_2020.09.28_10.36.22_stellar_koi_cfp_norobovetterlabels_renamedcols_nomissingval_symsecphase_cpkoiperiod_rba_cnt0n.csv')
tce_tbl = pd.read_csv(tce_tbl_fp)
logger.info(f'TCE table: {str(tce_tbl_fp)}')

# table with results that contains exoplanets validated by Armstrong et al and us
results_tbl_fp = Path(
    '/home/msaragoc/Projects/Kepler-TESS_exoplanet/Analysis/kepler_multiplicity_candidate_11-8-2021/ranking_comparison_with_paper_12-18-2020_merged_ra_dec_prad_CV_v15.csv')
results_tbl = pd.read_csv(results_tbl_fp)
logger.info(f'Table used for validated exoplanets by Armstrong et al and us: {str(results_tbl_fp)}')

# start by setting labels of TCE table to those used in the paper
# set unused KOIs dispositions to unknown and label id -1
results_tbl.loc[results_tbl['fold'] == -1, ['original_label', 'label']] = 'UNK', -1
logger.info(f'Original TCE disposition count:\n {results_tbl["original_label"].value_counts()}')
logger.info(f'Original TCE label id count:\n {results_tbl["label"].value_counts()}')
tce_tbl = tce_tbl.merge(results_tbl[['target_id', 'tce_plnt_num', 'label', 'original_label']],
                        on=['target_id', 'tce_plnt_num'], how='left', validate='one_to_one')
tce_tbl.drop(columns='label_x', inplace=True)
tce_tbl.rename(columns={'label_y': 'label'}, inplace=True)

# filter for validated planets by Armstrong et al and us
logger.info(f'Number of PCs before adding those validated by Armstrong et al and us: {(tce_tbl["label"] == 1).sum()}')
val_armstrong_tbl = results_tbl[results_tbl['sublist'] == 'in_50_exoplnt'].copy()
val_armstrong_tbl['validated_by'] = 'armstrong_et_al'
val_plnt_us_tbl = results_tbl[(results_tbl['ExoMiner_posterior'] > 0.99) & (results_tbl['MES'] > 10.5) &
                              (results_tbl['sublist'] != 'in_50_exoplnt') & (results_tbl['tce_fold'] == -1) &
                              (results_tbl['saterne_label'] != 'BD')].copy()
val_plnt_us_tbl['validated_by'] = 'us_exominer_2021'
val_plnt_tbl = pd.concat([val_armstrong_tbl, val_plnt_us_tbl], axis=0)
logger.info(f'Number of planets validated by Armstrong et al. and us: {len(val_plnt_tbl)}')
logger.info(f'\n{val_plnt_tbl["validated_by"].value_counts()}')

tce_tbl = tce_tbl.merge(val_plnt_tbl[['target_id', 'tce_plnt_num', 'validated_by']], on=['target_id', 'tce_plnt_num'],
                        how='left', validate='one_to_one')
# setting validated planets to PC and label id 1
tce_tbl.loc[
    tce_tbl['validated_by'].isin(['us_exominer_2021', 'armstrong_et_al']), ['original_label', 'label']] = 'PC', 1
logger.info(f'Number of PCs after adding those validated by Armstrong et al and us: {(tce_tbl["label"] == 1).sum()}')

logger.info(f'TCE disposition count:\n {tce_tbl["original_label"].value_counts()}')
logger.info(f'TCE label id count:\n {tce_tbl["label"].value_counts()}')

new_tce_tbl_fp = res_dir / f'{tce_tbl_fp.stem}_valpc.csv'
logger.info(f'Saving dataset table: {new_tce_tbl_fp}')
tce_tbl.to_csv(new_tce_tbl_fp, index=False)
