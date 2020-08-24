import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#%% Analyzing weak secondary

rankingTbl = pd.read_csv('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Kepler_planet_finder/results_ensemble/'
                         'keplerdr25_g2001-l201_spline_gapped_glfluxbohb_norobovetterkois_starshuffle_glflux-lwks_selfnorm/'
                         'ensemble_ranked_predictions_testset')
# rankingTbl = pd.read_csv('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Kepler_planet_finder/results_ensemble/'
#                          'bohb_keplerdr25_g2001-l201_spline_gapped_glflux_norobovetterkois_starshuffle/'
#                          'ensemble_ranked_predictions_testset')
rankingTblAFPmisclf = rankingTbl.loc[(rankingTbl['original_label'] == 'AFP') & (rankingTbl['score'] >= 0.5)]
rankingTblAFP = rankingTbl.loc[rankingTbl['original_label'] == 'AFP']
# rankingTblAFPmisclf = rankingTbl.loc[(rankingTbl['original_label'] == 'PC') & (rankingTbl['score'] < 0.5)]
# rankingTblAFP = rankingTbl.loc[rankingTbl['original_label'] == 'PC']

bins = np.linspace(-3, 3, 10, endpoint=True)
f, ax = plt.subplots()
ax.hist(rankingTblAFPmisclf['wst_robstat'].values, color='b', bins=bins, label='Misclassified AFP', zorder=10, edgecolor='k')
ax.hist(rankingTblAFP['wst_robstat'].values, color='r', bins=bins, label='AFP', zorder=1, edgecolor='k')
ax.set_title('Test set')
ax.set_ylabel('Counts')
ax.set_xlabel('wst_robstat')
ax.set_yscale('log')
ax.legend()

f, ax = plt.subplots()
ax.scatter(rankingTblAFPmisclf['wst_robstat'].values, rankingTblAFPmisclf['score'].values, c='b', s=15, label='Misclassfied AFP', zorder=10, marker='*')
ax.scatter(rankingTblAFP['wst_robstat'].values, rankingTblAFP['score'].values, c='r', s=15, label='AFP', zorder=1)
ax.set_ylabel('Score')
ax.set_xlabel('wst_robstat')
ax.set_title('Test set')
ax.legend()
ax.set_xlim([-3, 20])
# ax.set_xscale('log')
ax.set_ylim([0, 1])

rankingDir = '/home/msaragoc/Projects/Kepler-TESS_exoplanet/Kepler_planet_finder/results_ensemble/' \
             'bohb_keplerdr25_g2001-l201_spline_gapped_glflux_norobovetterkois_starshuffle/'
dispositions = ['PC', 'AFP', 'NTP']
datasets = ['train', 'val', 'test']
clfThr = 0.5

for dataset in datasets:
    rankingTbl = pd.read_csv(os.path.join(rankingDir, 'ensemble_ranked_predictions_{}set'.format(dataset)))
    for disposition in dispositions:
        if disposition == 'PC':
            numCorrect = len(rankingTbl.loc[(rankingTbl['original_label'] == disposition) &
                                              (rankingTbl['score'] >= clfThr)])
        else:
            numCorrect = len(rankingTbl.loc[(rankingTbl['original_label'] == disposition) &
                                              (rankingTbl['score'] < clfThr)])

        numTotal = len(rankingTbl.loc[rankingTbl['original_label'] == disposition])

        print('Dataset {}'.format(dataset))
        print('Disposition {}: {}/{} ({})'.format(disposition, numCorrect, numTotal, numCorrect / numTotal * 100))

#%% Analyzing odd-even

# rankingTbl = pd.read_csv('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Kepler_planet_finder/results_ensemble/'
#                          'keplerdr25_g2001-l201_spline_gapped_glfluxbohb_norobovetterkois_starshuffle_glflux-loe/'
#                          'ensemble_ranked_predictions_testset')
rankingTbl = pd.read_csv('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Kepler_planet_finder/results_ensemble/'
                         'bohb_keplerdr25_g2001-l201_spline_gapped_glflux_norobovetterkois_starshuffle/'
                         'ensemble_ranked_predictions_testset')
rankingTblAFPmisclf = rankingTbl.loc[(rankingTbl['original_label'] == 'AFP') & (rankingTbl['score'] >= 0.5)]
rankingTblAFP = rankingTbl.loc[rankingTbl['original_label'] == 'AFP']
# rankingTblAFPmisclf = rankingTbl.loc[(rankingTbl['original_label'] == 'PC') & (rankingTbl['score'] < 0.5)]
# rankingTblAFP = rankingTbl.loc[rankingTbl['original_label'] == 'PC']

bins = np.linspace(0, 2.764e5, 100, endpoint=True)
f, ax = plt.subplots()
ax.hist(rankingTblAFPmisclf['tce_bin_oedp_stat'].values, color='b', bins=bins, label='Misclassified AFP', zorder=10, edgecolor='k')
ax.hist(rankingTblAFP['tce_bin_oedp_stat'].values, color='r', bins=bins, label='AFP', zorder=1, edgecolor='k')
ax.set_title('Test set')
ax.set_ylabel('Counts')
ax.set_xlabel('tce_bin_oedp_stat')
# ax.set_yscale('log')
ax.legend()

f, ax = plt.subplots()
ax.scatter(rankingTblAFPmisclf['tce_bin_oedp_stat'].values, rankingTblAFPmisclf['score'].values, c='b', s=15, label='Misclassfied AFP', zorder=10, marker='*')
ax.scatter(rankingTblAFP['tce_bin_oedp_stat'].values, rankingTblAFP['score'].values, c='r', s=15, label='AFP', zorder=1)
ax.set_ylabel('Score')
ax.set_xlabel('tce_bin_oedp_stat')
ax.set_title('Test set')
ax.legend()
ax.set_xlim([0, 10])
# ax.set_xscale('log')
ax.set_ylim([0, 1])

rankingDir = '/home/msaragoc/Projects/Kepler-TESS_exoplanet/Kepler_planet_finder/results_ensemble/' \
             'keplerdr25_g2001-l201_spline_gapped_glfluxbohb_norobovetterkois_starshuffle_glflux-loe/'
dispositions = ['PC', 'AFP', 'NTP']
datasets = ['train', 'val', 'test']
clfThr = 0.5

for dataset in datasets:
    rankingTbl = pd.read_csv(os.path.join(rankingDir, 'ensemble_ranked_predictions_{}set'.format(dataset)))
    for disposition in dispositions:
        if disposition == 'PC':
            numCorrect = len(rankingTbl.loc[(rankingTbl['original_label'] == disposition) &
                                              (rankingTbl['score'] >= clfThr)])
        else:
            numCorrect = len(rankingTbl.loc[(rankingTbl['original_label'] == disposition) &
                                              (rankingTbl['score'] < clfThr)])

        numTotal = len(rankingTbl.loc[rankingTbl['original_label'] == disposition])

        print('Dataset {}'.format(dataset))
        print('Disposition {}: {}/{} ({})'.format(disposition, numCorrect, numTotal, numCorrect / numTotal * 100))

#%%

tceTbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17_DR25/'
                     'q1_q17_dr25_tce_2020.04.15_23.19.10_cumkoi_2020.02.21_shuffledstar_noroguetces_noRobovetterKOIs.csv')

studyRoot = '/home/msaragoc/Projects/Kepler-TESS_exoplanet/Kepler_planet_finder/results_ensemble'
studyDir = 'keplerdr25_g2001-l201_spline_gbal_nongapped_norobovetterkois_starshuffle_configE_glflux-glcentrmedcmaxn-loe-lwks-6stellar-bfap-ghost-rollingband_prelu'
studyName = 'configE-all'
dataset = 'train'
rankingTbl = pd.read_csv(os.path.join(studyRoot, studyDir, 'ensemble_ranked_predictions_{}set.csv'.format(dataset)))

koiColumns = ['koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec', 'koi_comment']
for col in koiColumns:
    rankingTbl[col] = np.nan

for tce_i, tce in rankingTbl.iterrows():

    validTce = tceTbl.loc[(tceTbl['target_id'] == tce['target_id']) &
                          (tceTbl['tce_plnt_num'] == tce['tce_plnt_num'])]

    if len(validTce) == 1:
        rankingTbl.loc[tce_i, koiColumns] = validTce[koiColumns].values[0]

saveDir = '/home/msaragoc/Projects/Kepler-TESS_exoplanet/Analysis/koi_comment_flag_analysis'
rankingTbl.to_csv(os.path.join(saveDir, 'ranking_{}set_study{}.csv').format(dataset, studyName), index=False)

rankingTblmisclf = rankingTbl.loc[((rankingTbl['predicted class'] == 1) & (rankingTbl['label'] == 0)) |
                                  ((rankingTbl['predicted class'] == 0) & (rankingTbl['label'] == 1))]
rankingTblmisclf = rankingTblmisclf[['target_id', 'tce_plnt_num', 'original_label', 'label', 'predicted class', 'score']
                                    + koiColumns]

rankingTblmisclf.to_csv(os.path.join(saveDir, 'ranking_{}set_study{}_misclf.csv').format(dataset, studyName), index=False)