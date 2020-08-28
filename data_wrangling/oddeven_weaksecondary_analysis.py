import pandas as pd
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

#%%

tceTbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17_DR25/'
                     'q1_q17_dr25_tce_2020.04.15_23.19.10_cumkoi_2020.02.21_shuffledstar_noroguetces_noRobovetterKOIs.csv')

studyRoot = '/home/msaragoc/Projects/Kepler-TESS_exoplanet/Kepler_planet_finder/results_ensemble/'
studyDir = 'keplerdr25_g2001-l201_spline_gbal_nongapped_norobovetterkois_starshuffle_configD_glflux-lwks_prelu/'
studyName = 'lwks'
dataset = 'train'
rankingTbl = pd.read_csv(os.path.join(studyRoot, studyDir, 'ensemble_ranked_predictions_{}set.csv'.format(dataset)))

comment = 'HAS_SEC_TCE'  # 'DEPTH_ODDEVEN'

# filter TCE table based on comment
tceTblComment = tceTbl.loc[tceTbl['koi_comment'].str.contains(comment, na=False)]

print('There are {} KOIs with comment {}.'.format(len(tceTblComment), comment))

rankingTblComment = pd.DataFrame(columns=rankingTbl.columns)

for tce_i, tce in rankingTbl.iterrows():

    validTce = tceTblComment.loc[(tceTblComment['target_id'] == tce['target_id']) &
                                 (tceTblComment['tce_plnt_num'] == tce['tce_plnt_num'])]

    if len(validTce) == 1:
        rankingTblComment = rankingTblComment.append(tce, ignore_index=True)

print('There are {} TCEs that have comment {} in the {} set.'.format(len(rankingTblComment), comment, dataset))

saveDir = '/home/msaragoc/Projects/Kepler-TESS_exoplanet/Analysis/oddeven_wks_analysis'

rankingTblComment.to_csv(os.path.join(saveDir, 'ranking_{}set_{}_study{}.csv').format(dataset, comment, studyName),
                         index=False)

#%% Compare baseline to local odd-even test

datasets = ['test', 'val', 'train']
studies = ['baseline', 'loe']

rankingTblComment = {}
for dataset in datasets:
    rankingTblComment[dataset] = {}
    for study in studies:
        rankingTblComment[dataset][study] = pd.read_csv(os.path.join(saveDir, 'ranking_{}set_{}_study{}.csv').format(dataset, comment, study))

scores = {}
for dataset in datasets:
    scores[dataset] = {}
    for study in studies:
        scores[dataset][study] = rankingTblComment[dataset][study]['score'].values

# colorMap = {'baseline': 'r', 'loe': 'b'}
markerMap = {'train': 'o', 'test': '*', 'val': 'x'}

f, ax = plt.subplots()
for dataset in datasets:
    ax.scatter(scores[dataset]['baseline'], scores[dataset]['loe'], marker=markerMap[dataset], label=dataset)
ax.legend()
ax.set_xlabel('Baseline score')
ax.set_ylabel('L odd-even score')
# ax.set_ylim([0, 1])
# ax.set_xlim([0, 1])
ax.grid(True)
ax.plot([0, 1], [0, 1], 'r', linestyle='--')
ax.set_yscale('log')
ax.set_xscale('log')
ax.axhline(y=0.5, color='k', linestyle='-.')
f.savefig(os.path.join(saveDir, 'scatter_baselinevsloe_{}.png').format(comment))

scoresSetStudy = {}
for dataset in datasets:
    for study in studies:
        scoresSetStudy['{}_{}'.format(dataset, study)] = scores[dataset][study]

f, ax = plt.subplots()
ax.boxplot(list(scoresSetStudy.values()))
ax.set_xticklabels(scoresSetStudy.keys())
ax.set_ylabel('Score')
ax.set_xlabel('Study')
ax.set_yscale('log')
f.savefig(os.path.join(saveDir, 'boxplot_baselinevsloe_{}.png').format(comment))

#%% Compare baseline to local weak secondary test

datasets = ['test', 'val', 'train']
studies = ['baseline', 'lwks']

rankingTblComment = {}
for dataset in datasets:
    rankingTblComment[dataset] = {}
    for study in studies:
        rankingTblComment[dataset][study] = pd.read_csv(os.path.join(saveDir, 'ranking_{}set_{}_study{}.csv').format(dataset, comment, study))

scores = {}
for dataset in datasets:
    scores[dataset] = {}
    for study in studies:
        scores[dataset][study] = rankingTblComment[dataset][study]['score'].values

# colorMap = {'baseline': 'r', 'loe': 'b'}
markerMap = {'train': 'o', 'test': '*', 'val': 'x'}

f, ax = plt.subplots()
for dataset in datasets:
    ax.scatter(scores[dataset]['baseline'], scores[dataset]['lwks'], marker=markerMap[dataset], label=dataset)
ax.legend()
ax.set_xlabel('Baseline score')
ax.set_ylabel('L Weak Secondary score')
# ax.set_ylim([0, 1])
# ax.set_xlim([0, 1])
ax.grid(True)
ax.plot([0, 1], [0, 1], 'r', linestyle='--')
# ax.set_yscale('log')
# ax.set_xscale('log')
ax.axhline(y=0.5, color='k', linestyle='-.')
f.savefig(os.path.join(saveDir, 'scatter_baselinevslwks_{}.png').format(comment))

scoresSetStudy = {}
for dataset in datasets:
    for study in studies:
        scoresSetStudy['{}_{}'.format(dataset, study)] = scores[dataset][study]

f, ax = plt.subplots()
ax.boxplot(list(scoresSetStudy.values()))
ax.set_xticklabels(scoresSetStudy.keys())
ax.set_ylabel('Score')
ax.set_xlabel('Study')
ax.set_yscale('log')
f.savefig(os.path.join(saveDir, 'boxplot_baselinevslwks_{}.png').format(comment))

#%% Analyze the minimum values for the normalize odd and even local views in the dataset

saveDir = '/home/msaragoc/Projects/Kepler-TESS_exoplanet/Analysis/odd-even_min_comparison'

tfrecDir = '/data5/tess_project/Data/tfrecords/Kepler/Q1-Q17_DR25/' \
           'tfrecordskeplerdr25_g2001-l201_gbal_spline_nongapped_flux-centroid-oddeven-wks-6stellar-ghost-bfap-rollingband_data/' \
           'tfrecordskeplerdr25_g2001-l201_gbal_spline_nongapped_flux-centroid-oddeven-wks-6stellar-ghost-bfap-rollingband_starshuffle_experiment-labels'

tfrecFp = [os.path.join(tfrecDir, fileName) for fileName in os.listdir(tfrecDir) if 'shard' in fileName]

columnsDf = ['target_id', 'tce_plnt_num', 'label', 'odd_min', 'even_min']
oddevenDf = pd.DataFrame(np.nan, index=np.arange(30945), columns=columnsDf)

tceIdentifier = 'tce_plnt_num'
tce_i = 0
for tfrecFile in tfrecFp:

    record_iterator = tf.python_io.tf_record_iterator(path=tfrecFile)

    for string_i, string_record in enumerate(record_iterator):
        example = tf.train.Example()
        example.ParseFromString(string_record)

        tce = {col: np.nan for col in columnsDf}

        tce['target_id'] = int(example.features.feature['target_id'].int64_list.value[0])
        tce[tceIdentifier] = int(example.features.feature[tceIdentifier].int64_list.value[0])
        tce['label'] = example.features.feature['label'].bytes_list.value[0].decode("utf-8")
        tce['odd_min'] = np.min(np.array(example.features.feature['local_flux_odd_view'].float_list.value))
        tce['even_min'] = np.min(np.array(example.features.feature['local_flux_even_view'].float_list.value))

        oddevenDf.ix[tce_i, columnsDf] = list(tce.values())

        tce_i += 1

# drop rows not used
oddevenDf = oddevenDf.loc[~oddevenDf['target_id'].isna()]

oddevenDf.to_csv(os.path.join(saveDir, 'keplerq1q17dr25_norobovetterkois_normalized_oddeven_views.csv'), index=False)

#%% Scatter plot of the odd and even views minimum values

saveDir = '/home/msaragoc/Projects/Kepler-TESS_exoplanet/Analysis/odd-even_min_comparison'

oddevenDf = pd.read_csv(os.path.join(saveDir, 'keplerq1q17dr25_norobovetterkois_normalized_oddeven_views_diff_koicomment_koidisp_fpwgdisp.csv'))

labels = ['PC', 'AFP', 'NTP']

for label in labels:
    f, ax = plt.subplots()
    oddevenDfLabel = oddevenDf.loc[oddevenDf['label'] == label]
    ax.scatter(oddevenDfLabel['odd_min'], oddevenDfLabel['even_min'], s=5)
    ax.set_ylabel('Even Min')
    ax.set_xlabel('Odd Min')
    ax.set_xlim([-2.5, 0])
    ax.set_ylim([-2.5, 0])
    ax.grid(True)
    ax.set_title('{}'.format(label))
    f.savefig(os.path.join(
        saveDir, 'scatter_zoom_{}_keplerq1q17dr25_norobovetterkois_normalized_oddeven_views.png'.format(label)))


f, ax = plt.subplots()
oddevenDfPC = oddevenDf.loc[oddevenDf['label'] == 'PC']
ax.scatter(oddevenDfPC['odd_min'], oddevenDfPC['even_min'], s=5, c='b', label='PC', alpha=0.3, zorder=100)
# oddevenDfAFP = oddevenDf.loc[oddevenDf['label'] == 'AFP']
oddevenDfAFP = oddevenDf.loc[(oddevenDf['label'] == 'AFP') & (oddevenDf['koi_comment'].str.contains('DEPTH_ODDEVEN', na=False))]
ax.scatter(oddevenDfAFP['odd_min'], oddevenDfAFP['even_min'], s=5, c='r', label='AFP', zorder=10)
ax.set_ylabel('Even Min')
ax.set_xlabel('Odd Min')
ax.set_xlim([-2.5, 0])
ax.set_ylim([-2.5, 0])
ax.grid(True)
ax.set_title('{}'.format('PC | AFP'))
ax.legend()
f.savefig(os.path.join(saveDir, 'scatter_zoom_PC-AFP_DEPTHODDEVEN_keplerq1q17dr25_norobovetterkois_normalized_oddeven_views.png'))

oddevenDf.loc[(oddevenDf['label'] == 'AFP') &
              (oddevenDf['koi_comment'].str.contains('DEPTH_ODDEVEN'))].to_csv(
    os.path.join(saveDir,
                 'keplerq1q17dr25_norobovetterkois_normalized_oddeven_views_diff_koicomment_koidisp_fpwgdisp_AFP-DEPTH_ODDEVEN_FLAG.csv'),
    index=False)

#%%

saveDir = '/home/msaragoc/Projects/Kepler-TESS_exoplanet/Analysis/odd-even_min_comparison'

oddevenDf = pd.read_csv(os.path.join(saveDir, 'keplerq1q17dr25_norobovetterkois_normalized_oddeven_views.csv'))

oddevenDf['diff'] = oddevenDf[['odd_min', 'even_min']].apply(lambda x: np.abs(x['odd_min'] - x['even_min']), axis=1)

tceTbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17_DR25/'
                     'q1_q17_dr25_tce_2020.04.15_23.19.10_cumkoi_2020.02.21_shuffledstar_noroguetces_noRobovetterKOIs.csv')

oddevenDf['koi_comment'] = ''
oddevenDf['fpwg_disp_status'] = ''
oddevenDf['koi_disposition'] = ''
for tce_i, tce in oddevenDf.iterrows():

    tceFound = tceTbl.loc[(tceTbl['target_id'] == tce['target_id']) & (tceTbl['tce_plnt_num'] == tce['tce_plnt_num'])]

    if len(tceFound) > 0:
        oddevenDf.loc[tce_i, ['koi_comment', 'fpwg_disp_status', 'koi_disposition']] = tceFound[['koi_comment', 'fpwg_disp_status', 'koi_disposition']].values[0]

oddevenDf.sort_values('diff', ascending=True, inplace=True)

oddevenDf.to_csv(os.path.join(saveDir, 'keplerq1q17dr25_norobovetterkois_normalized_oddeven_views_diff_koicomment_koidisp_fpwgdisp.csv'),
                 index=False)

oddevenDfDepthOeComment = oddevenDf.loc[oddevenDf['koi_comment'].str.contains('DEPTH_ODDEVEN', na=False)]
oddevenDfNoDepthOeComment = oddevenDf.loc[~oddevenDf['koi_comment'].str.contains('DEPTH_ODDEVEN', na=False)]
oddevenDfPC = oddevenDf.loc[oddevenDf['label'] == 'PC']
oddevenDfAFP = oddevenDf.loc[oddevenDf['label'] == 'AFP']

# bins = np.logspace(-6, 0, 100)
bins = np.linspace(1e-6, 1, 100)
f, ax = plt.subplots()
ax.hist(oddevenDfDepthOeComment['diff'].values, bins=bins, label='DEPTH_ODDEVEN', zorder=10)
ax.hist(oddevenDfNoDepthOeComment['diff'].values, bins=bins, label='NO DEPTH_ODDEVEN', zorder=1)
ax.hist(oddevenDfPC['diff'].values, bins=bins, label='PC', zorder=3)
ax.hist(oddevenDfAFP['diff'].values, bins=bins, label='AFP', zorder=2, alpha=0.5)
ax.legend()
ax.set_ylabel('Counts')
ax.set_yscale('log')
ax.set_xlabel('Odd-even min abs diff')
f.savefig(os.path.join(saveDir, 'hist_oe-min_absdiff_keplerq1q17dr25_norobovetterkois_normalized_oddeven_views.png'))
