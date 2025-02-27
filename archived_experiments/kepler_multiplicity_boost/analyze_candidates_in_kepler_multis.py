""" Analyzing number of candidates in multi-planet systems in Kepler. """

# 3rd party
import pandas as pd
from pathlib import Path

res_dir = Path('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Analysis/kepler_multiplicity_candidate_11-8-2021/')

ranking_tbl_fp = Path(res_dir / 'ranking_comparison_with_paper_12-18-2020_merged_ra_dec_prad_CV_v15.csv')
ranking_tbl = pd.read_csv(ranking_tbl_fp)
ranking_tbl.drop(columns=['Unnamed: 0', 'Unnamed: 0.1', 'Unnamed: 0.1.1', 'Unnamed: 0.1.1.1', 'Unnamed: 0.1.1.1.1'],
                 inplace=True)
tbls = []

# these numbers are estimates for the dataset we are using, which is a subset of  Q1-Q17 DR25 TCEs (< 34k TCEs), and
# KOIs only associated with Q1-Q17 DR25 run
# number of TCEs in target
cnt_tces_target = ranking_tbl['target_id'].value_counts().to_frame(name='num_tces_target').reset_index().rename(
    columns={'index': 'target_id'})
tbls.append(cnt_tces_target)
# number of Candidates KOIs in target
cnt_candidates_target = ranking_tbl.loc[
    ranking_tbl['koi_disposition'] == 'CANDIDATE', 'target_id'].value_counts().to_frame(
    name='num_candidates_target').reset_index().rename(columns={'index': 'target_id'})
tbls.append(cnt_candidates_target)
# number of Confirmed KOIs in target
cnt_confirmed_target = ranking_tbl.loc[
    ranking_tbl['koi_disposition'] == 'CONFIRMED', 'target_id'].value_counts().to_frame(
    name='num_confirmed_target').reset_index().rename(columns={'index': 'target_id'})
tbls.append(cnt_confirmed_target)
# number of AFPs (CFPs aka FP KOIs in our evaluation dataset) in target
cnt_afps_target = ranking_tbl.loc[ranking_tbl['original_label'] == 'AFP', 'target_id'].value_counts().to_frame(
    name='num_afps_target').reset_index().rename(columns={'index': 'target_id'})
tbls.append(cnt_afps_target)
# number of KOIs in target
cnt_kois_target = ranking_tbl.loc[~ranking_tbl['kepoi_name'].isna(), 'target_id'].value_counts().to_frame(
    name='num_kois_target').reset_index().rename(columns={'index': 'target_id'})
tbls.append(cnt_kois_target)
# number of unlabeled KOIs in target
cnt_uknkois_target = ranking_tbl.loc[ranking_tbl['fold'] == -1, 'target_id'].value_counts().to_frame(
    name='num_unkkois_target').reset_index().rename(columns={'index': 'target_id'})
tbls.append(cnt_uknkois_target)

for tbl in tbls:
    # aaa
    ranking_tbl = ranking_tbl.merge(tbl, on=['target_id'], how='left')
    # aaa

# ranking_tbl.to_csv(ranking_tbl_fp.parent / f'{ranking_tbl_fp.stem}_targetcnts.csv', index=False)

# %% add kp, kic centroid offset and uncertainty, model snr

tce_tbl_columns = ['target_id', 'tce_plnt_num', 'tce_model_snr', 'tce_prad', 'mag', 'tce_dikco_msky',
                   'tce_dikco_msky_err', 'koi_model_snr', 'koi_max_mult_ev', 'koi_num_transits', 'koi_count',
                   'koi_prad']  # the koi_count might be larger than the number of KOIs counted in our dataset
tce_tbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17_DR25/'
                      '19-08-21_07:21/q1_q17_dr25_tce_2020.09.28_10.36.22_stellar_koi_cfp_norobovetterlabels_renamedcols_nomissingval_symsecphase_cpkoiperiod_rba_cnt0n.csv')[
    tce_tbl_columns]

ranking_tbl = ranking_tbl.merge(tce_tbl, on=['target_id', 'tce_plnt_num'], how='left')

ranking_tbl.to_csv(ranking_tbl_fp.parent / f'{ranking_tbl_fp.stem}_targetcnts.csv', index=False)
