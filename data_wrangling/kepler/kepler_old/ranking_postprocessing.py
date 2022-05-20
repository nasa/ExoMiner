import pandas as pd
import os
import subprocess

#%% Add KOI fields to ranking

rankingTceTbl = pd.read_csv('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Kepler_planet_finder/results_ensemble/'
                            'dr25tcert_spline_gapped_glflux-lcentr-loe-6stellar_glfluxconfig/ranked_predictions_testset',
                            header=0)

keplerTceTbl = pd.read_csv('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Kepler_planet_finder/koi_ephemeris_matching/'
                           'q1_q17_dr25_tce_2019.03.12_updt_tcert_extendedtceparams_updt_normstellarparamswitherrors_koi_processed.csv',
                           header=0)

# fields to be added to the ranking table
addedFields = ['kepoi_name', 'kepler_name', 'koi_disposition', 'koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co',
               'koi_fpflag_ec', 'koi_comment', 'koi_datalink_dvr', 'koi_datalink_dvs']

# instantiate these fields as NaN for all TCEs
rankingTceTbl = pd.concat([rankingTceTbl, pd.DataFrame(columns=addedFields)])

for tce_i, tce in rankingTceTbl.iterrows():
    rankingTceTbl.iloc[tce_i, addedFields] = keplerTceTbl.loc[(keplerTceTbl['target_id'] == tce.kepid) &
                                                              (keplerTceTbl['tce_plnt_num'] == tce.tce_n)][addedFields]

rankingTceTbl.to_csv('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Kepler_planet_finder/results_ensemble/'
                     'dr25tcert_spline_gapped_glflux-lcentr-loe-6stellar_glfluxconfig/ranked_predictions_testset_koi',
                     index=False)

#%% Check where the 19 interesting non-TCEs end up in the new ranking using Config E + all features (8-12-2020)

rankingTbl = pd.read_csv('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Kepler_planet_finder/results_ensemble/'
                         'keplerdr25_nontces_g2001-l201_spline_gbal_nongapped_norobovetterkois_starshuffle_configE_glflux-glcentr-loe-6stellar_prelu/'
                         'ensemble_ranked_predictions_predictset.csv')

nonTces = [int(folderName.split('-')[0])
           for folderName in os.listdir('/data5/tess_project/Data/DV/Promising_TCE_for_TESS_Team/'
                                        '19_interesting_tces(9_20_2019)')]

dfNonTces = pd.DataFrame(columns=rankingTbl.columns)
for nonTce in nonTces:
    dfNonTces = pd.concat([dfNonTces, rankingTbl.loc[rankingTbl['target_id'] == nonTce]], axis=0)

dfNonTces.to_csv('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Analysis/19_interesting_nontces_newranking/'
                 '19interestingnontces_rankingconfigE.csv')

#%% Adding score to the final views plot for non-TCEs

plotsDir = '/home/msaragoc/Downloads/top30_nontces_configE_glflux-glcentr-loe-6stellar/'

rankingTbl = pd.read_csv('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Kepler_planet_finder/results_ensemble/'
                         'keplerdr25_nontces_g2001-l201_spline_gbal_nongapped_norobovetterkois_starshuffle_configE_glflux-glcentr-loe-6stellar_prelu/'
                         'ensemble_ranked_predictions_predictset.csv')

rankingTblTop = rankingTbl[0:30]['target_id'].values

for nonTce in rankingTblTop:
    process = subprocess.Popen(['cd',
                                '{}'.format(plotsDir),
                                'mv',
                                '{}_1_NTP_8_final_views_aug0.png'.format(nonTce),
                                '{}_{}_1_NTP_8_final_views_aug0.png'.format(nonTce,
                                                                            rankingTbl.loc[rankingTbl['target_id'] ==
                                                                                           nonTce]['score'].values[0])],
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    print(stdout)
    print(stderr)
    aaaa
