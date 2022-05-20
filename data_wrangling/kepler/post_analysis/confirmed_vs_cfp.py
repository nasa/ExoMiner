""" Check scores produced by the model for those KOIs that are dispositioned as both Confirmed and CFP KOI. """

from pathlib import Path
import pandas as pd

experiment_dir = Path('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Kepler_planet_finder/results_ensemble/keplerdr25-dv_g301-l31_6tr_spline_nongapped_norobovetterkois_starshuffle_configK_secsymphase_wksnormmaxflux-wks_corrprimgap_ptempstat_albedostat_wstdepth_fwmstat_nopps_ckoiper_secparams_prad_per')

tce_tbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17_DR25/q1_q17_dr25_tce_2020.09.28_10.36.22_stellar_koi_cfp_norobovetterlabels_renamedcols_nomissingval_symsecphase_confirmedkoiperiod_sec.csv')

fpwg_tbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/kois_tables/fpwg_2021.03.02_12.09.58.csv', header=75)

# keep only confirmed kois that are also dispositioned as CFP
tce_tbl = tce_tbl.loc[(tce_tbl['koi_disposition'] == 'CONFIRMED') & (tce_tbl['fpwg_disp_status'] == 'CERTIFIED FP')]
print(f'Number of TCE dispositioned as both Confirmed KOI and CFP KOI: {len(tce_tbl)}')

datasets = ['train', 'val', 'test']

ranking_tbl = {dataset: pd.read_csv(experiment_dir / f'ensemble_ranked_predictions_{dataset}set.csv') for dataset in datasets}

new_tbl = pd.DataFrame(columns=ranking_tbl[datasets[0]].columns)
for col in ['dataset', 'koi_comment', 'koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec', 'fpwg_comment', 'fpwg_disp_source', 'fpwg_disp_eb', 'fpwg_disp_offst', 'fpwg_disp_perep', 'fpwg_disp_other']:
    new_tbl[col] = ''

for dataset in datasets:

    for tce_i, tce in ranking_tbl[dataset].iterrows():
        tce_found = tce_tbl.loc[(tce_tbl['target_id'] == tce['target_id']) & (tce_tbl['tce_plnt_num'] == tce['tce_plnt_num'])].reset_index()

        if len(tce_found) == 1:
            tce['dataset'] = dataset
            tce = tce.append(tce_found.loc[0, ['koi_comment', 'koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec']])
            tce_found_fpwg = fpwg_tbl.loc[fpwg_tbl['kepoi_name'] == tce_found['kepoi_name'].values[0]].reset_index()
            tce = tce.append(tce_found_fpwg.loc[0, ['fpwg_comment', 'fpwg_disp_source', 'fpwg_disp_eb', 'fpwg_disp_offst', 'fpwg_disp_perep', 'fpwg_disp_other']])
            new_tbl = new_tbl.append(tce, ignore_index=True)

new_tbl.to_csv(experiment_dir / 'ranked_predicitons_confirmed-cfp_kois.csv', index=False)
