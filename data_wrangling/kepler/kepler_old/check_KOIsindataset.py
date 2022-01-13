""" Sanity check for which KOIs are in the train, validation and test sets. """

# 3rd party
from pathlib import Path
import pandas as pd

experiment_dir = Path('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Kepler_planet_finder/results_ensemble/keplerdr25-dv_g301-l31_6tr_spline_nongapped_norobovetterkois_starshuffle_configK_secsymphase_wksnormmaxflux-wks_corrprimgap_ptempstat_albedostat_wstdepth_fwmstat_nopps_ckoiper_secparams_prad_per/')
ranking_datasets_tbl = {dataset: pd.read_csv(experiment_dir / f'ensemble_ranked_predictions_{dataset}set.csv')
                        for dataset in ['train', 'val', 'test']}

koi_tbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17_DR25/q1_q17_dr25_tce_2020.09.28_10.36.22_stellar_koi_cfp.csv')
koi_tbl = koi_tbl.loc[(koi_tbl['fpwg_disp_status'].isin(['NOT EXAMINED', 'DATA INCONCLUSIVE'])) &
                      (koi_tbl['koi_disposition'] != 'CONFIRMED') ]
koi_tbl['in_set'] = 'NA'
koi_tbl['label'] = 'NA'
koi_tbl['score'] = -1

for koi_i, koi in koi_tbl.iterrows():

    for dataset in ranking_datasets_tbl:
        koi_found = ranking_datasets_tbl[dataset].loc[(ranking_datasets_tbl[dataset]['target_id'] == koi['kepid'])
                                                      & (ranking_datasets_tbl[dataset]['tce_plnt_num'] ==
                                                         koi['tce_plnt_num'])]

        if len(koi_found) == 1:
            koi_tbl.loc[koi_i, ['in_set']] = dataset
            koi_tbl.loc[koi_i, ['label']] = koi_found['label']
            koi_tbl.loc[koi_i, ['score']] = koi_found['score']
            break

koi_tbl[['kepid', 'tce_plnt_num', 'tce_rogue_flag', 'koi_disposition', 'fpwg_disp_status', 'in_set', 'score', 'label']].to_csv('/home/msaragoc/Downloads/aaa.csv', index=False)