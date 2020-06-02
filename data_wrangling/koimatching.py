import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

from src_preprocessing.get_dv_files import get_dv_files

#%% Matching KOI with TCEs to add KOI fields such as FP flags using the TCE planet number scrapped from the
# 'koi_datalink_dvs' field and updating the labels using the KOI dispositions from the Cumulative KOI list

# logging.basicConfig(filename='/home/msaragoc/Projects/Kepler-TESS_exoplanet/Kepler_planet_finder/'
#                              'koi_ephemeris_matching/koi_ephemeris_matching.log', filemode='a', format='%(message)s',
#                     level=logging.INFO)

# Q1-Q17 DR25 TCE list
# keplerTceTable = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/final_tce_tables/'
#                              'q1_q17_dr25_tce_2019.03.12_updt_tcert_extendedtceparams_updt_normstellarparamswitherrors_'
#                              'processed.csv',
#                              header=0)
# keplerTceTable = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/final_tce_tables/'
#                              'q1_q17_dr25_tce_2019.03.12_updt_tcert_extendedtceparams_updt_normstellarparamswitherrors_'
#                              'koidatalink_processedlinks.csv', header=0)
keplerTceTable = pd.read_csv('/Users/msaragoc/Projects/Kepler-TESS_exoplanet/data/ephemeris_tables/Kepler/'
                             'final_tce_tables/q1_q17_dr25_tce_2019.03.12_updt_tcert_extendedtceparams_updt_'
                             'normstellarparamswitherrors_koidatalink_processedlinks.csv', header=0)

# Cumulative KOI list
# koiCumTable = pd.read_csv('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Kepler_planet_finder/koi_ephemeris_matching/'
#                           'oldvsnewkoidispositions.csv', header=0)
# koiCumTable = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/koi_table/'
#                           'cumulative_2020.02.21_10.29.22.csv', header=90)
koiCumTable = pd.read_csv('/Users/msaragoc/Projects/Kepler-TESS_exoplanet/data/ephemeris_tables/Kepler/koi_table/'
                          'cumulative_2020.02.21_10.29.22.csv', header=90)

# # filter KOIs that do not come from Q1-Q17 DR25 TCE list
koiCumTable = koiCumTable.loc[koiCumTable['koi_tce_delivname'] == 'q1_q17_dr25_tce']
# koiCumTable = koiCumTable.loc[~koiCumTable['koi_datalink_dvr'].isna()]

koiColumnNames = np.array(koiCumTable.columns.values.tolist())[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, -25, -24, -21,
                                                                -20]]
# koiColumnNames = koiCumTable.columns.values.tolist()

# initialize new columns
keplerTceTable = pd.concat([keplerTceTable, pd.DataFrame(columns=koiColumnNames)])

numKois = len(koiCumTable)

koiNotMatched, koiNoTceInTarget = [], []

# reset label column
keplerTceTable['label'] = 'NTP'

# iterate through the Kepler IDs in the KOI table
for koi_i, koi in koiCumTable.iterrows():

    # add KOI parameters to the matched TCE
    tce_match = (keplerTceTable['target_id'] == koi.kepid) & \
                (keplerTceTable['tce_plnt_num'] == int(koi.koi_datalink_dvs.split('-')[1]))
    keplerTceTable.loc[tce_match, koiColumnNames] = koi[koiColumnNames].values

    # replace label by new disposition
    if koi.koi_disposition in ['CONFIRMED', 'CANDIDATE']:
        keplerTceTable.loc[tce_match, 'label'] = 'PC'
    elif koi.koi_disposition == 'FALSE POSITIVE':
        keplerTceTable.loc[tce_match, 'label'] = 'AFP'
    else:
        print('It is already NTP. This should not be printed.')
    #     keplerTceTable.loc[tce_match, 'label'] = 'NTP'

# # print('Total number of KOI not matched = {}'.format(len(koiNotMatched)))
# logging.info('Total number of KOI not matched = {}'.format(len(koiNotMatched)))
# logging.info('Number of KOI not matched to any TCE in the same Kepler ID = {}'.format(len(koiNotMatched) -
#                                                                                       len(koiNoTceInTarget)))
# logging.info('Number of KOI without any TCE in the same Kepler ID = {}'.format(len(koiNoTceInTarget)))

# save updated TCE table with KOI parameters
# keplerTceTable.to_csv('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Kepler_planet_finder/koi_ephemeris_matching/'
#                       'q1_q17_dr25_tce_2019.03.12_updt_tcert_extendedtceparams_updt_normstellarparamswitherrors_'
#                       'koi_processed.csv',
#                       index=False)
keplerTceTable.to_csv('/Users/msaragoc/Projects/Kepler-TESS_exoplanet/data/ephemeris_tables/Kepler/'
                      'final_tce_tables/q1_q17_dr25_tce_cumkoi2020.02.21.csv',
                      index=False)

#%% Check label of KOIs matched against labels of the TCEs in the Q1-Q17 DR25 TCE list (updated by Laurent on March
# 2019)

keplerTceTable = pd.read_csv('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Kepler_planet_finder/'
                             'koi_ephemeris_matching/'
                             'q1_q17_dr25_tce_2019.03.12_updt_tcert_extendedtceparams_updt_normstellarparamswitherrors_'
                             'koidatalinkoldKOIlist_processed.csv')

keplerTceTable = keplerTceTable[['tce_period', 'tce_time0bk', 'tce_duration', 'target_id', 'tce_plnt_num', 'label',
                                 'kepoi_name', 'koi_disposition']]

# keplerTceTable.dropna(axis=0, subset=['koi_disposition'], inplace=True)

afp_matched = 0
pc_matched = 0
ntp_matched = 0
nonpc_matched = 0
notMatched = {}
for tce_i, tce in keplerTceTable.iterrows():

    # count PC matches to CONFIRMED and CANDIDATE
    if tce.label == 'PC' and tce.koi_disposition in ['CONFIRMED', 'CANDIDATE']:
    # if tce.label == 'PC' and tce.koi_disposition in ['CONFIRMED']:
        pc_matched += 1
        continue

    # count AFP matches to FALSE POSITIVE
    if tce.label == 'AFP' and tce.koi_disposition == 'FALSE POSITIVE':
        afp_matched += 1
        nonpc_matched += 1
        continue

    # count NTP matches to empty (NaN)
    if tce.label == 'NTP' and pd.isna(tce.koi_disposition):
    # if tce.label == 'NTP' and (pd.isna(tce.koi_disposition) or tce.koi_disposition == 'CANDIDATE'):
        ntp_matched += 1
        nonpc_matched += 1
        continue

    # count non-PC matches to FALSE POSITIVE or empty (NaN)
    if tce.label in ['AFP', 'NTP'] and (pd.isna(tce.koi_disposition) or tce.koi_disposition == 'FALSE POSITIVE'):
    # if tce.label in ['AFP', 'NTP'] and (pd.isna(tce.koi_disposition) or tce.koi_disposition == 'FALSE POSITIVE' or tce.koi_disposition == 'CANDIDATE'):
        nonpc_matched += 1

    else:
        notMatched[(tce.target_id, tce.tce_plnt_num)] = {'ephemeris': [tce.tce_period, tce.tce_time0bk,
                                                                       tce.tce_duration],
                                                         'label': tce.label,
                                                         'koi_label': tce.koi_disposition}

print('Number of labels matched: {} out of {}'.format(pc_matched + nonpc_matched, len(keplerTceTable)))
print('Number of PC matched: {} out of {}'.format(pc_matched, len(keplerTceTable.loc[keplerTceTable['label'] == 'PC'])))
print('Number of non-PC matched: {} out of {}'.format(nonpc_matched, len(keplerTceTable.loc[keplerTceTable['label'] !=
                                                                                            'PC'])))
print('Number of AFP matched: {} out of {}'.format(afp_matched, len(keplerTceTable.loc[keplerTceTable['label'] ==
                                                                                       'AFP'])))
print('Number of NTP matched: {} out of {}'.format(ntp_matched, len(keplerTceTable.loc[keplerTceTable['label'] ==
                                                                                       'NTP'])))

#%% Compare KOI dispositions between the KOI cumulative list used by Laurent to update the TCE list on March 2019 and
# the one I downloaded on February 2020

oldKoiTbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/dr25_koi.csv', header=18)
newKoiTbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/koi_table/cumulative_2020.02.21_10.29.22.csv',
                        header=90)

# filter KOIs that do not come from Q1-Q17 DR25 TCE list
newKoiTbl = newKoiTbl.loc[newKoiTbl['koi_tce_delivname'] == 'q1_q17_dr25_tce']

oldKoiTbl['new_koi_disposition'] = np.nan
oldKoiTbl['kepoi_name'] = np.nan
oldKoiTbl['koi_datalink_dvs'] = np.nan
for oldKoi_i, oldKoi in oldKoiTbl.iterrows():

    newKoi = newKoiTbl.loc[(newKoiTbl['kepid'] == oldKoi.kepid) & (newKoiTbl['koi_tce_plnt_num'] ==
                                                                   oldKoi.koi_tce_plnt_num)]
    oldKoiTbl.loc[oldKoi_i, ['new_koi_disposition']] = newKoi.koi_disposition.values
    oldKoiTbl.loc[oldKoi_i, ['kepoi_name']] = newKoi.kepoi_name.values
    oldKoiTbl.loc[oldKoi_i, ['koi_datalink_dvs']] = newKoi.koi_datalink_dvs.values

oldKoiTbl.to_csv('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Kepler_planet_finder/koi_ephemeris_matching/'
                 'oldvsnewkoidispositions.csv', index=False)

confirmed_matched = 0
candidate_matched = 0
c_matched = 0
fp_matched = 0
notMatched = {}
for koi_i, koi in oldKoiTbl.iterrows():

    # count CONFIRMED matches
    if koi.koi_disposition == 'CONFIRMED' and koi.new_koi_disposition == 'CONFIRMED':
        confirmed_matched += 1
        continue

    # count CANDIDATE matches
    if koi.koi_disposition == 'CANDIDATE' and koi.new_koi_disposition == 'CANDIDATE':
        candidate_matched += 1
        continue

    # count FALSE POSITIVE matches
    if koi.koi_disposition == 'FALSE POSITIVE' and koi.new_koi_disposition == 'FALSE POSITIVE':
        fp_matched += 1
        continue

    # count non-PC matches to FALSE POSITIVE or empty (NaN)
    if koi.koi_disposition in ['CANDIDATE', 'CONFIRMED'] and koi.new_koi_disposition in ['CANDIDATE', 'CONFIRMED']:
        c_matched += 1

    else:
        notMatched[(koi.kepid, koi.koi_tce_plnt_num)] = {'ephemeris': [koi.koi_period,
                                                                       koi.koi_time0bk,
                                                                       koi.koi_duration],
                                                         'label': koi.koi_disposition,
                                                         'koi_label': koi.new_koi_disposition}

print('Number of labels matched: {} out of {}'.format(c_matched + confirmed_matched + candidate_matched + fp_matched,
                                                      len(oldKoiTbl)))
print('Number of CONFIRMED matched: {} out of {}'.format(confirmed_matched,
                                                  len(oldKoiTbl.loc[oldKoiTbl['koi_disposition'] == 'CONFIRMED'])))
print('Number of CANDIDATE matched: {} out of {}'.format(candidate_matched,
                                                      len(oldKoiTbl.loc[oldKoiTbl['koi_disposition'] == 'CANDIDATE'])))
print('Number of CANDIDATE-CONFIRMED matched: {} out of {}'.format(c_matched + candidate_matched + confirmed_matched,
                                                   len(oldKoiTbl.loc[oldKoiTbl['koi_disposition'].isin(['CONFIRMED',
                                                                                                        'CANDIDATE'])]))
      )
print('Number of FALSE POSITIVE matched: {} out of {}'.format(fp_matched,
                                                              len(oldKoiTbl.loc[oldKoiTbl['koi_disposition'] ==
                                                                                'FALSE POSITIVE'])))

#%% get TCEs for the KOIs that changed their disposition from the old KOI cumulative list to the new one

columnsKoi = ['kepid', 'kepoi_name', 'koi_disposition', 'koi_fpflag_nt', 'koi_fpflag_co', 'koi_fpflag_ec',
              'koi_period', 'koi_time0bk', 'koi_duration']
columnsDf = ['kepid', 'tce_plnt_num', 'kepoi_name', 'koi_disposition', 'koi_fpflag_nt', 'koi_fpflag_co',
             'koi_fpflag_ec', 'koi_period', 'koi_time0bk', 'koi_duration', 'old_koi_disposition', 'label', 'old_label']
dispositionToLabelMap = {'CONFIRMED': 'PC', 'CANDIDATE': 'PC', 'FALSE POSITIVE': 'AFP'}
koisToBeAdded = []
for koi_i, koi in newKoiTbl.iterrows():

    if (koi.kepid, koi.koi_tce_plnt_num) in notMatched.keys():
        tce_plnt_num = int(koi.koi_datalink_dvs.split('-')[1])
        koisToBeAdded.append(np.insert(koi[columnsKoi].values, 1, tce_plnt_num))
        oldKoiDisposition = oldKoiTbl.loc[(oldKoiTbl['kepid'] == koi.kepid) &
                                          (oldKoiTbl['koi_tce_plnt_num'] ==
                                           koi.koi_tce_plnt_num)]['koi_disposition'].values[0]
        oldKoiLabel = dispositionToLabelMap[oldKoiDisposition]
        newKoiLabel = dispositionToLabelMap[koi.koi_disposition]
        koisToBeAdded[-1] = np.append(koisToBeAdded[-1], [oldKoiDisposition, newKoiLabel, oldKoiLabel])

updtKoisTbl = pd.concat([pd.DataFrame([koisToBeAdded[koi_i]], columns=columnsDf)
                         for koi_i in range(len(koisToBeAdded))])
updtKoisTbl.to_csv('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Kepler_planet_finder/koi_ephemeris_matching/'
                   'updatedKOIsdisposition.csv', index=False)

#%% Check model predictions against KOI updated dispositions

resultsDir = '/home/msaragoc/Projects/Kepler-TESS_exoplanet/Kepler_planet_finder/results_ensemble/' \
             'dr25tcert_spline_gapped_glflux-lcentr-loe-6stellar_glfluxconfig/'
rankingSet = 'ranked_predictions_testset'
rankingTbl = pd.read_csv(os.path.join(resultsDir, rankingSet))

updtKoisTbl = pd.read_csv('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Kepler_planet_finder/koi_ephemeris_matching/'
                          'updatedKOIsdisposition.csv')

columnsToBeAdded = ['koi_disposition', 'koi_fpflag_nt', 'koi_fpflag_co', 'koi_fpflag_ec', 'new_label']
columnsKoi = ['koi_disposition', 'koi_fpflag_nt', 'koi_fpflag_co', 'koi_fpflag_ec', 'label']
for column in columnsToBeAdded:
    rankingTbl[column] = np.nan
for tce_i, tce in rankingTbl.iterrows():

    koi = updtKoisTbl.loc[(updtKoisTbl['kepid'] == tce.kepid) &
                          (updtKoisTbl['tce_plnt_num'] == tce.tce_n)]

    if len(koi) > 0:
        for i in range(len(columnsToBeAdded)):
            rankingTbl.loc[tce_i, columnsToBeAdded[i]] = koi[columnsKoi[i]].values

rankingTbl.to_csv('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Kepler_planet_finder/results_ensemble/'
                  'dr25tcert_spline_gapped_glflux-lcentr-loe-6stellar_glfluxconfig/'
                  '{}updtkoi.csv'.format(rankingSet), index=False)

# filter only KOIs with updated dispositions
rankingTblUpdatedKois = rankingTbl.loc[~rankingTbl['koi_disposition'].isna()]
rankingTblUpdatedKois.to_csv('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Kepler_planet_finder/results_ensemble/'
                             'dr25tcert_spline_gapped_glflux-lcentr-loe-6stellar_glfluxconfig/'
                             '{}koisupdatedonly.csv'.format(rankingSet), index=False)

# predicted PCs that were not PCs according to old label but now are
newPCsDetected = len(rankingTbl.loc[(rankingTbl['predicted class'] == 1) &
                                    (rankingTbl['new_label'] == 'PC') &
                                    (rankingTbl['label'] == 0)])
numNewPcs = len(rankingTbl.loc[(rankingTbl['new_label'] == 'PC') & (rankingTbl['label'] == 0)])

# predicted non-PCs that were PCs according to old label but now are
newNonPcsDetected = len(rankingTbl.loc[(rankingTbl['predicted class'] == 0) &
                                       (rankingTbl['new_label'] == 'AFP') &
                                       (rankingTbl['label'] == 1)])
numNewNonPcs = len(rankingTbl.loc[(rankingTbl['new_label'] == 'AFP') &
                                  (rankingTbl['label'] == 1)])

accNewPcs = newPCsDetected / numNewPcs
accNewNonPcs = newNonPcsDetected / numNewNonPcs

with open(os.path.join(resultsDir, "res_{}_updatedKOIs.txt".format(rankingSet)), "w") as res_file:
    res_file.write('Performance for the set of updated KOIs\n')
    res_file.write('Number of new PCs detected (Total number of new PCs): {} ({}) | Acc = {}\n'.format(newPCsDetected,
                                                                                                       numNewPcs,
                                                                                                       accNewPcs))
    res_file.write(
        'Number of new non-PCs detected (Total number of new non-PCs): {} ({}) | Acc = {}\n'.format(newNonPcsDetected,
                                                                                                    numNewNonPcs,
                                                                                                    accNewNonPcs))

#%% Check model predictions against KOI dispositions

# tceTbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/final_tce_tables/'
#                      'q1_q17_dr25_tce_2019.03.12_updt_tcert_extendedtceparams_updt_normstellarparamswitherrors_'
#                      'koidatalink_processed.csv')
tceTbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/final_tce_tables/'
                     'q1_q17_dr25_tce_2019.03.12_updt_tcert_extendedtceparams_updt_normstellarparamswitherrors_'
                     'koidatalink_processedlinks.csv')

resultsDir = '/home/msaragoc/Projects/Kepler-TESS_exoplanet/Kepler_planet_finder/results_ensemble/' \
             'dr25tcert_spline_gapped_glflux-glcentr-loe-6stellar_glfluxconfig_updtKOIs/'

rankingSet = 'ranked_predictions_testset'
rankingTbl = pd.read_csv(os.path.join(resultsDir, rankingSet))

rankingTbl['koi_disposition'] = np.nan
for tce_i, tce in rankingTbl.iterrows():

    rankingTbl.loc[tce_i, 'koi_disposition'] = tceTbl.loc[(tceTbl['target_id'] == tce.kepid) &
                                                          (tceTbl['tce_plnt_num'] == tce.tce_n)]['koi_disposition'].values

rankingTbl.to_csv(os.path.join(resultsDir, '{}kois.csv'.format(rankingSet)), index=False)

koiDispositions = ['CONFIRMED', 'CANDIDATE', 'FALSE POSITIVE']
output_cl = {koiDisposition: rankingTbl.loc[rankingTbl['koi_disposition'] == koiDisposition]['output'].values
             for koiDisposition in koiDispositions}
koiDispositions.append('NTP')
output_cl['NTP'] = rankingTbl.loc[rankingTbl['koi_disposition'].isna()]['output'].values

bins = np.linspace(0, 1, 11, True)
# normalizedToClass = False

for normalizedToClass in [True, False]:
    hist, bin_edges = {}, {}
    for class_label in output_cl:
        counts_cl = list(np.histogram(output_cl[class_label], bins, density=False, range=(0, 1)))
        if normalizedToClass:
            counts_cl[0] = counts_cl[0] / max(len(output_cl[class_label]), 1e-7)
        hist[class_label] = counts_cl[0]
        bin_edges[class_label] = counts_cl[1]

    bins_multicl = np.linspace(0, 1, len(output_cl) * 10 + 1, True)
    bin_width = bins_multicl[1] - bins_multicl[0]
    bins_cl = {}
    for i, class_label in enumerate(output_cl):
        bins_cl[class_label] = [(bins_multicl[idx] + bins_multicl[idx + 1]) / 2
                                for idx in range(i, len(bins_multicl) - 1, len(output_cl))]

    f, ax = plt.subplots()
    for class_label in output_cl:
        ax.bar(bins_cl[class_label], hist[class_label], bin_width, label=class_label, edgecolor='k')
    # if dataset == 'predict':
    #     ax.set_ylabel('Dataset fraction')
    # else:
    if normalizedToClass:
        ax.set_ylabel('KOI disposition fraction')
    else:
        ax.set_ylabel('Number of samples')
    ax.set_yscale('log')
    ax.set_xlabel('Predicted output')
    if normalizedToClass:
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
    ax.set_title('Output distribution - {}'.format(rankingSet))
    ax.set_xticks(np.linspace(0, 1, 11, True))
    # if dataset != 'predict':
    ax.legend()
    plt.savefig(os.path.join(resultsDir, 'koi_predoutput_distribution_{}_{}.svg'.format(rankingSet, ('Relative', 'Normalized')[normalizedToClass])))
    plt.close()

cl_thr = 0.5
results = {koiDisposition: {'total': None, 'detected': None, 'accuracy': None} for koiDisposition in koiDispositions}
for koiDisposition in koiDispositions:

    results[koiDisposition]['total'] = len(output_cl[koiDisposition])
    if koiDisposition in ['CONFIRMED', 'CANDIDATE']:
        results[koiDisposition]['detected'] = len(np.where(output_cl[koiDisposition] >= cl_thr)[0])
    else:
        results[koiDisposition]['detected'] = len(np.where(output_cl[koiDisposition] < cl_thr)[0])

    results[koiDisposition]['accuracy'] = results[koiDisposition]['detected'] / results[koiDisposition]['total']

with open(os.path.join(resultsDir, "res_{}_KOIs.txt".format(rankingSet)), "w") as res_file:
    res_file.write('Performance for the set of KOIs\n')

    for koiDisposition in koiDispositions:

        res_file.write('Number of detected {} (Total number of): {} ({}) | Acc = {}\n'.format(koiDisposition,
                                                                                              results[koiDisposition]['detected'],
                                                                                              results[koiDisposition]['total'],
                                                                                              results[koiDisposition]['accuracy']))

#%% Get NTPs in the ranking above the threshold

# tceTbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/final_tce_tables/'
#                      'q1_q17_dr25_tce_2019.03.12_updt_tcert_extendedtceparams_updt_normstellarparamswitherrors_koidatalink_processed.csv')
#
# linksTbl = pd.read_csv('/home/msaragoc/Downloads/q1_q17_dr25_tce_2020.03.03_15.13.55.csv', header=34)
#
# tceTbl['tce_datalink_dvs'] = np.nan
# tceTbl['tce_datalink_dvr'] = np.nan
#
# for tce_i, tce in tceTbl.iterrows():
#
#     tceTbl.loc[tce_i, ['tce_datalink_dvs',
#                        'tce_datalink_dvr']] = linksTbl.loc[(linksTbl['kepid'] == tce.target_id) &
#                                                            (linksTbl['tce_plnt_num'] == tce.tce_plnt_num)][['tce_datalink_dvs',
#                                                                                                            'tce_datalink_dvr']].values
#
# tceTbl.to_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/final_tce_tables/'
#                      'q1_q17_dr25_tce_2019.03.12_updt_tcert_extendedtceparams_updt_normstellarparamswitherrors_koidatalink_processedlinks.csv',
#               index=False)

# resultsDir = '/home/msaragoc/Projects/Kepler-TESS_exoplanet/Kepler_planet_finder/results_ensemble/' \
#              'dr25tcert_spline_gapped_glflux-glcentr-loe-6stellar_glfluxconfig_updtKOIs/'
resultsDir = '/Users/msaragoc/Projects/Kepler-TESS_exoplanet/results_ensemble/' \
             'dr25tcert_spline_gapped_glflux-glcentr-loe-6stellar_glfluxconfig_updtKOIs'

rankingSet = 'ranked_predictions_trainset.csv'
rankingTbl = pd.read_csv(os.path.join(resultsDir, rankingSet))

# filter misclassified TCEs
rankingTbl = rankingTbl.loc[(rankingTbl['original label'] == 'PC') & (rankingTbl['output'] < 0.5)]

print(len(rankingTbl))

# tceTbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/final_tce_tables/'
#                      'q1_q17_dr25_tce_2019.03.12_updt_tcert_extendedtceparams_updt_normstellarparamswitherrors_'
#                      'koidatalink_processedlinks.csv')
# tceTbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/final_tce_tables/'
#                      'q1_q17_dr25_tce_2019.03.12_updt_tcert_extendedtceparams_updt_normstellarparamswitherrors_'
#                      'koidatalink_processedlinks.csv')
tceTbl = pd.read_csv('/Users/msaragoc/Projects/Kepler-TESS_exoplanet/data/ephemeris_tables/Kepler/final_tce_tables/'
                     'q1_q17_dr25_tce_2019.03.12_updt_tcert_extendedtceparams_updt_normstellarparamswitherrors_'
                     'koidatalink_processedlinks.csv')

# add links to DV reports and summaries
rankingTbl['tce_datalink_dvs'] = np.nan
rankingTbl['tce_datalink_dvr'] = np.nan
rankingTbl['koi_disposition'] = np.nan
for tce_i, tce in rankingTbl.iterrows():

    # rankingTbl.loc[tce_i, ['tce_datalink_dvs', 'tce_datalink_dvr']] = \
    #     tceTbl.loc[(tceTbl['target_id'] == tce.kepid) &
    #                (tceTbl['tce_plnt_num'] == tce.tce_n)][['tce_datalink_dvs', 'tce_datalink_dvr']].values

    rankingTbl.loc[tce_i, ['tce_datalink_dvs', 'tce_datalink_dvr', 'koi_disposition']] = \
        tceTbl.loc[(tceTbl['target_id'] == tce.kepid) &
                   (tceTbl['tce_plnt_num'] == tce.tce_n)][['tce_datalink_dvs', 'tce_datalink_dvr',
                                                           'koi_disposition']].values[0]

rankingTbl.to_csv(os.path.join(resultsDir, '{}_NTPabovethr.csv'.format(rankingSet.split('.')[0])), index=False)
rankingTbl = pd.read_csv(os.path.join(resultsDir, '{}_NTPabovethr.csv'.format(rankingSet.split('.')[0])))

rootUrl = 'https://exoplanetarchive.ipac.caltech.edu/data/KeplerData/'
# downloadUrls = [subUrl[0][2:-2] for subUrl in list(rankingTbl['tce_datalink_dvr'].values)]
downloadUrls = [subUrl[4:-4] for subUrl in list(rankingTbl['tce_datalink_dvr'].values)]
downloadUrls = [rootUrl + url for url in downloadUrls]
downloadDir = '/home/msaragoc/Projects/Kepler-TESS_exoplanet/Kepler_planet_finder/results_ensemble/' \
              'dr25tcert_spline_gapped_glflux-glcentr-loe-6stellar_glfluxconfig_updtKOIs/NTPabovethr_reports'

get_dv_files(downloadUrls, downloadDir)

# downloadUrls = [subUrl[0][2:-2] for subUrl in list(rankingTbl['tce_datalink_dvs'].values)]
downloadUrls = [subUrl[4:-4] for subUrl in list(rankingTbl['tce_datalink_dvs'].values)]
downloadUrls = [rootUrl + url for url in downloadUrls]
get_dv_files(downloadUrls, downloadDir)

#%% Check the model performance in the subset of KOIs in the Certified False Positive List

# add certified false positive disposition to the TCE list
# load TCE list
tceTbl = pd.read_csv('/Users/msaragoc/Projects/Kepler-TESS_exoplanet/data/ephemeris_tables/Kepler/final_tce_tables/'
                     'q1_q17_dr25_tce_2019.03.12_updt_tcert_extendedtceparams_updt_normstellarparamswitherrors_'
                     'koidatalink_processedlinks.csv')

# load FPWG Certified False Positive list
koiFpTbl = pd.read_csv('/Users/msaragoc/Projects/Kepler-TESS_exoplanet/data/ephemeris_tables/Kepler/koi_table/'
                       'fpwg_2020.03.13_11.37.49.csv', header=13)

# instantiate column in the TCE list for the FPWG dispositions
tceTbl['fpwg_disp_status'] = np.nan

for koi_i, koi in koiFpTbl.iterrows():
    koiInTceTbl = tceTbl['kepoi_name'] == koi.kepoi_name
    if len(koiInTceTbl) > 0:
        tceTbl.loc[koiInTceTbl, 'fpwg_disp_status'] = koi.fpwg_disp_status

# save the TCE list with the added CFP list dispositions for the KOIs
tceTbl.to_csv('/Users/msaragoc/Projects/Kepler-TESS_exoplanet/koi_matching/'
              'certified_false_positive/q1_q17_dr25_tce_2019.03.12_KOIsDR25_cfp.csv', index=False)

# results directory
resultsDir = '/Users/msaragoc/Projects/Kepler-TESS_exoplanet/results_ensemble/' \
             'dr25tcert_spline_gapped_glflux-glcentr-loe-6stellar_glfluxconfig_updtKOIs/Certified False Positive'

# add to the ranking the cfp dispositions
rankingSet = 'ranked_predictions_valset'
rankingTbl = pd.read_csv(os.path.join(os.path.dirname(resultsDir), rankingSet))

# instantiate CFP disposition column in the ranking
rankingTbl['fpwg_disp_status'] = np.nan
for tce_i, tce in rankingTbl.iterrows():

    rankingTbl.loc[tce_i, 'fpwg_disp_status'] = \
        tceTbl.loc[(tceTbl['target_id'] == tce.kepid) &
                   (tceTbl['tce_plnt_num'] == tce.tce_n)]['fpwg_disp_status'].values

# save the ranking with the added CFP disposition column
rankingTbl.to_csv(os.path.join(resultsDir, '{}cfp.csv'.format(rankingSet)), index=False)

# get score distribution for each disposition in the CFP list
koiDispositions = ['CERTIFIED FP', 'CERTIFIED FA', 'POSSIBLE PLANET', 'NOT EXAMINED', 'DATA INCONCLUSIVE']
output_cl = {koiDisposition: rankingTbl.loc[rankingTbl['fpwg_disp_status'] == koiDisposition]['output'].values
             for koiDisposition in koiDispositions}
# koiDispositions.append('NOT EXAMINED')
# output_cl['NOT EXAMINED'] = rankingTbl.loc[rankingTbl['fpwg_disp_status'].isna()]['output'].values

bins = np.linspace(0, 1, 11, True)
# normalizedToClass = False

for normalizedToClass in [True, False]:
    hist, bin_edges = {}, {}
    for class_label in output_cl:
        counts_cl = list(np.histogram(output_cl[class_label], bins, density=False, range=(0, 1)))
        if normalizedToClass:
            counts_cl[0] = counts_cl[0] / max(len(output_cl[class_label]), 1e-7)
        hist[class_label] = counts_cl[0]
        bin_edges[class_label] = counts_cl[1]

    bins_multicl = np.linspace(0, 1, len(output_cl) * 10 + 1, True)
    bin_width = bins_multicl[1] - bins_multicl[0]
    bins_cl = {}
    for i, class_label in enumerate(output_cl):
        bins_cl[class_label] = [(bins_multicl[idx] + bins_multicl[idx + 1]) / 2
                                for idx in range(i, len(bins_multicl) - 1, len(output_cl))]

    f, ax = plt.subplots()
    for class_label in output_cl:
        ax.bar(bins_cl[class_label], hist[class_label], bin_width, label=class_label, edgecolor='k')
    # if dataset == 'predict':
    #     ax.set_ylabel('Dataset fraction')
    # else:
    if normalizedToClass:
        ax.set_ylabel('Disposition fraction')
    else:
        ax.set_ylabel('Number of samples')
    ax.set_yscale('log')
    ax.set_xlabel('Predicted output')
    if normalizedToClass:
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
    ax.set_title('FPWG disposition\nOutput distribution - {}'.format(rankingSet))
    ax.set_xticks(np.linspace(0, 1, 11, True))
    # if dataset != 'predict':
    ax.legend()
    plt.savefig(os.path.join(resultsDir,
                             'cfpkoi_predoutput_distribution_{}_{}.png'.format(rankingSet,
                                                                               ('Relative',
                                                                                'Normalized')[normalizedToClass])))
    plt.close()

cl_thr = 0.5
results = {koiDisposition: {'total': None, 'detected': None, 'accuracy': None} for koiDisposition in
           ['CERTIFIED FP', 'CERTIFIED FA', 'POSSIBLE PLANET']}
for koiDisposition in ['CERTIFIED FP', 'CERTIFIED FA', 'POSSIBLE PLANET']:

    results[koiDisposition]['total'] = len(output_cl[koiDisposition])
    if koiDisposition == 'POSSIBLE PLANET':
        results[koiDisposition]['detected'] = len(np.where(output_cl[koiDisposition] >= cl_thr)[0])
    else:
        results[koiDisposition]['detected'] = len(np.where(output_cl[koiDisposition] < cl_thr)[0])

    results[koiDisposition]['accuracy'] = results[koiDisposition]['detected'] / results[koiDisposition]['total']

with open(os.path.join(resultsDir, "res_{}_cfpKOIs.txt".format(rankingSet)), "w") as res_file:
    res_file.write('Performance for the set of KOIs\n')

    for koiDisposition in ['CERTIFIED FP', 'CERTIFIED FA', 'POSSIBLE PLANET']:

        res_file.write('Number of detected {} (Total number of): '
                       '{} ({}) | Acc = {}\n'.format(koiDisposition,
                                                     results[koiDisposition]['detected'],
                                                     results[koiDisposition]['total'],
                                                     results[koiDisposition]['accuracy']))

#%% Check the model performance against the Kepler Names list (Kepler confirmed and validated planets) that show up in
# the KOI lists

keplerNamesTable = pd.read_csv('/Users/msaragoc/Projects/Kepler-TESS_exoplanet/data/ephemeris_tables/Kepler/'
                               'kepler_names_tables/keplernames_2020.03.17_11.49.15.csv', header=17)

# filter out Kepler Names that are not in the KOI lists
keplerNamesTable = keplerNamesTable.loc[keplerNamesTable['koi_list_flag'] == 'YES']

# add certified false positive disposition to the TCE list
tceTbl = pd.read_csv('/Users/msaragoc/Projects/Kepler-TESS_exoplanet/data/ephemeris_tables/Kepler/final_tce_tables/'
                     'q1_q17_dr25_tce_2019.03.12_updt_tcert_extendedtceparams_updt_normstellarparamswitherrors_'
                     'koidatalink_processedlinks.csv')

tceTbl['kepler_name'] = np.nan

for keplerName_i, keplerName in keplerNamesTable.iterrows():
    keplerNameInTceTbl = tceTbl['kepoi_name'] == keplerName.kepoi_name
    if len(keplerNameInTceTbl) > 0:
        tceTbl.loc[keplerNameInTceTbl, 'kepler_name'] = 'YES'

# save the TCE list with the added Kepler Names column
tceTbl.to_csv('/Users/msaragoc/Projects/Kepler-TESS_exoplanet/koi_matching/'
              'kepler_names/q1_q17_dr25_tce_2019.03.12_KOIsDR25_keplernames.csv', index=False)

resultsDir = '/Users/msaragoc/Projects/Kepler-TESS_exoplanet/results_ensemble/' \
             'dr25tcert_spline_gapped_glflux-glcentr-loe-6stellar_glfluxconfig_updtKOIs'

# add to the ranking the cfp dispositions
rankingSet = 'ranked_predictions_valset'
rankingTbl = pd.read_csv(os.path.join(resultsDir, rankingSet))

rankingTbl['kepler_name'] = np.nan
for tce_i, tce in rankingTbl.iterrows():

    rankingTbl.loc[tce_i, 'kepler_name'] = tceTbl.loc[(tceTbl['target_id'] == tce.kepid) &
                                                      (tceTbl['tce_plnt_num'] == tce.tce_n)]['kepler_name'].values

rankingTbl.to_csv(os.path.join(resultsDir, '{}keplernames.csv'.format(rankingSet)), index=False)


output_cl = {'Kepler Names': rankingTbl.loc[rankingTbl['kepler_name'] == 'YES']['output'].values}

bins = np.linspace(0, 1, 11, True)

for normalizedToClass in [True, False]:
    hist, bin_edges = {}, {}
    for class_label in output_cl:
        counts_cl = list(np.histogram(output_cl[class_label], bins, density=False, range=(0, 1)))
        if normalizedToClass:
            counts_cl[0] = counts_cl[0] / max(len(output_cl[class_label]), 1e-7)
        hist[class_label] = counts_cl[0]
        bin_edges[class_label] = counts_cl[1]

    bins_multicl = np.linspace(0, 1, len(output_cl) * 10 + 1, True)
    bin_width = bins_multicl[1] - bins_multicl[0]
    bins_cl = {}
    for i, class_label in enumerate(output_cl):
        bins_cl[class_label] = [(bins_multicl[idx] + bins_multicl[idx + 1]) / 2
                                for idx in range(i, len(bins_multicl) - 1, len(output_cl))]

    f, ax = plt.subplots()
    for class_label in output_cl:
        ax.bar(bins_cl[class_label], hist[class_label], bin_width, label=class_label, edgecolor='k')
    # if dataset == 'predict':
    #     ax.set_ylabel('Dataset fraction')
    # else:
    if normalizedToClass:
        ax.set_ylabel('Disposition fraction')
    else:
        ax.set_ylabel('Number of samples')
    ax.set_yscale('log')
    ax.set_xlabel('Predicted output')
    if normalizedToClass:
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
    ax.set_title('Kepler Names\nOutput distribution - {}'.format(rankingSet))
    ax.set_xticks(np.linspace(0, 1, 11, True))
    # if dataset != 'predict':
    ax.legend()
    plt.savefig(os.path.join(resultsDir,
                             'keplernames_predoutput_distribution_{}_{}.png'.format(rankingSet,
                                                                                    ('Relative',
                                                                                     'Normalized')[normalizedToClass])))
    plt.close()

cl_thr = 0.5
results = {'Kepler Names': {'total': None, 'detected': None, 'accuracy': None}}

results['Kepler Names']['total'] = len(output_cl['Kepler Names'])
results['Kepler Names']['detected'] = len(np.where(output_cl['Kepler Names'] >= cl_thr)[0])
results['Kepler Names']['accuracy'] = results['Kepler Names']['detected'] / results['Kepler Names']['total']

with open(os.path.join(resultsDir, "res_{}_keplernames.txt".format(rankingSet)), "w") as res_file:
    res_file.write('Performance for the set of Kepler Names\n')

    res_file.write('Number of detected {} (Total number of): {} ({}) | '
                   'Acc = {}\n'.format('Kepler Names',
                                       results['Kepler Names']['detected'],
                                       results['Kepler Names']['total'],
                                       results['Kepler Names']['accuracy']))

#%% Check the model performance in the subset of KOIs in the Certified False Positive List that have disposition of
# POSSIBLE PLANET against the PC label from the Cumulative KOI list and the overlap between the two dispositions

# results directory
resultsDir = '/Users/msaragoc/Projects/Kepler-TESS_exoplanet/results_ensemble/' \
             'dr25tcert_spline_gapped_glflux-glcentr-loe-6stellar_glfluxconfig_updtKOIs/Certified False Positive'

# load the ranking with the cfp dispositions for a given dataset
rankingSet = 'ranked_predictions_testset'
rankingTbl = pd.read_csv(os.path.join(resultsDir, '{}cfp.csv'.format(rankingSet)))

# get scores for each disposition
dispositions = ['PC', 'POSSIBLE PLANET', 'PC-POSSIBLE PLANET']
output_cl = {}
for disposition in dispositions:
    if disposition == 'PC':
        output_cl[disposition] = rankingTbl.loc[rankingTbl['original label'] == disposition]['output'].values
    elif disposition == 'POSSIBLE PLANET':
        output_cl[disposition] = rankingTbl.loc[rankingTbl['fpwg_disp_status'] == disposition]['output'].values
    else:  # scores for overlap
        output_cl['PC-POSSIBLE PLANET'] = \
            rankingTbl.loc[(rankingTbl['original label'] == 'PC') &
                           (rankingTbl['fpwg_disp_status'] == 'POSSIBLE PLANET')]['output'].values

# plot score distribution for each disposition
bins = np.linspace(0, 1, 11, True)
# normalizedToClass = False

for normalizedToClass in [True, False]:
    hist, bin_edges = {}, {}
    for class_label in output_cl:
        counts_cl = list(np.histogram(output_cl[class_label], bins, density=False, range=(0, 1)))
        if normalizedToClass:
            counts_cl[0] = counts_cl[0] / max(len(output_cl[class_label]), 1e-7)
        hist[class_label] = counts_cl[0]
        bin_edges[class_label] = counts_cl[1]

    bins_multicl = np.linspace(0, 1, len(output_cl) * 10 + 1, True)
    bin_width = bins_multicl[1] - bins_multicl[0]
    bins_cl = {}
    for i, class_label in enumerate(output_cl):
        bins_cl[class_label] = [(bins_multicl[idx] + bins_multicl[idx + 1]) / 2
                                for idx in range(i, len(bins_multicl) - 1, len(output_cl))]

    f, ax = plt.subplots()
    for class_label in output_cl:
        ax.bar(bins_cl[class_label], hist[class_label], bin_width, label=class_label, edgecolor='k')
    # if dataset == 'predict':
    #     ax.set_ylabel('Dataset fraction')
    # else:
    if normalizedToClass:
        ax.set_ylabel('Disposition fraction')
    else:
        ax.set_ylabel('Number of samples')
    ax.set_yscale('log')
    ax.set_xlabel('Predicted output')
    if normalizedToClass:
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
    ax.set_title('FPWG disposition\nOutput distribution - {}'.format(rankingSet))
    ax.set_xticks(np.linspace(0, 1, 11, True))
    ax.grid(True)
    # if dataset != 'predict':
    ax.legend()
    plt.savefig(os.path.join(resultsDir,
                             'pc-pp_predoutput_distribution_{}_{}.png'.format(rankingSet,
                                                                              ('Relative',
                                                                               'Normalized')[normalizedToClass])))
    plt.close()

# compute accuracy for each disposition
cl_thr = 0.5  # classification threshold
results = {disposition: {'total': None, 'detected': None, 'accuracy': None} for disposition in
           dispositions}
for disposition in dispositions:
    results[disposition]['total'] = len(output_cl[disposition])
    results[disposition]['detected'] = len(np.where(output_cl[disposition] >= cl_thr)[0])
    results[disposition]['accuracy'] = results[disposition]['detected'] / results[disposition]['total']

# save results to a file
with open(os.path.join(resultsDir, "res_{}_pc-pp.txt".format(rankingSet)), "w") as res_file:
    res_file.write('Performance for the set of KOIs\n')

    for disposition in results:
        res_file.write('Number of detected {} (Total number of): '
                       '{} ({}) | Acc = {}\n'.format(disposition,
                                                     results[disposition]['detected'],
                                                     results[disposition]['total'],
                                                     results[disposition]['accuracy']))

# table with the PC-POSSIBLE PLANET misclassified
misclf_pc_pp = rankingTbl.loc[(rankingTbl['original label'] == 'PC') &
                              (rankingTbl['fpwg_disp_status'] == 'POSSIBLE PLANET') &
                              (rankingTbl['label'] == 1) &
                              (rankingTbl['predicted class'] == 0)]

# add the URLs to the misclassfied PC-POSSIBLE PLANET table
# load the TCE table with those URLs
tceTbl = pd.read_csv('/Users/msaragoc/Projects/Kepler-TESS_exoplanet/data/ephemeris_tables/Kepler/final_tce_tables/'
                     'q1_q17_dr25_tce_2019.03.12_updt_tcert_extendedtceparams_updt_normstellarparamswitherrors_'
                     'koidatalink_processedlinks.csv')
misclf_pc_pp['tce_datalink_dvs'] = np.nan
misclf_pc_pp['tce_datalink_dvr'] = np.nan
# add them to the misclassified table
for tce_i, tce in misclf_pc_pp.iterrows():
    misclf_pc_pp.loc[tce_i, ['tce_datalink_dvs', 'tce_datalink_dvr']] = \
        tceTbl.loc[(tceTbl['target_id'] == tce.kepid) &
                   (tceTbl['tce_plnt_num'] == tce.tce_n)][['tce_datalink_dvs', 'tce_datalink_dvr']].values

# download reports for PC-POSSIBLE PLANET that were misclassified
rootUrl = 'https://exoplanetarchive.ipac.caltech.edu/data/KeplerData/'  # base URL
downloadDir = os.path.join(resultsDir, 'reports_missclf_pc-pp')  # save directory

# DV report
downloadUrls = [subUrl[0][2:-2] for subUrl in list(misclf_pc_pp['tce_datalink_dvr'].values)]
downloadUrls = [rootUrl + url for url in downloadUrls]
get_dv_files(downloadUrls, downloadDir)

# DV summary
downloadUrls = [subUrl[0][2:-2] for subUrl in misclf_pc_pp['tce_datalink_dvs'].values]
downloadUrls = [rootUrl + url for url in downloadUrls]
get_dv_files(downloadUrls, downloadDir)

#%% Compute accuracy for classes PC, AFP, NTP

# results directory
resultsDir = '/Users/msaragoc/Projects/Kepler-TESS_exoplanet/results_ensemble/' \
             'dr25tcert_spline_gapped_glflux-glcentr-loe-6stellar_glfluxconfig_updtKOIs/'

# load the ranking with the cfp dispositions for a given dataset
rankingSet = 'ranked_predictions_valset'
rankingTbl = pd.read_csv(os.path.join(resultsDir, '{}'.format(rankingSet)))

# get scores for each disposition
dispositions = ['PC', 'AFP', 'NTP']
output_cl = {}
for disposition in dispositions:
    output_cl[disposition] = rankingTbl.loc[rankingTbl['original label'] == disposition]['output'].values

# compute accuracy for each disposition
cl_thr = 0.5  # classification threshold
results = {disposition: {'total': None, 'detected': None, 'accuracy': None} for disposition in
           dispositions}
for disposition in dispositions:
    results[disposition]['total'] = len(output_cl[disposition])
    if disposition == 'PC':
        results[disposition]['detected'] = len(np.where(output_cl[disposition] >= cl_thr)[0])
    else:
        results[disposition]['detected'] = len(np.where(output_cl[disposition] < cl_thr)[0])
    results[disposition]['accuracy'] = results[disposition]['detected'] / results[disposition]['total']

# save results to a file
with open(os.path.join(resultsDir, "res_{}_pc-afp-ntp.txt".format(rankingSet)), "w") as res_file:
    res_file.write('Performance for each category: PC, AFP, NTP\n')

    for disposition in results:
        res_file.write('Number of detected {} (Total number of): '
                       '{} ({}) | Acc = {}\n'.format(disposition,
                                                     results[disposition]['detected'],
                                                     results[disposition]['total'],
                                                     results[disposition]['accuracy']))

#%% Compute accuracy for classes PC, AFP, NTP

# results directory
resultsDir = '/home/msaragoc/Projects/Kepler-TESS_exoplanet/Kepler_planet_finder/results_ensemble/' \
             'bohb_dr25_spline_gapped_hvb_glflux/'

# load the ranking with the cfp dispositions for a given dataset
rankingSet = 'ensemble_ranked_predictions_trainset'
rankingTbl = pd.read_csv(os.path.join(resultsDir, '{}'.format(rankingSet)))

# get scores for each disposition
dispositions = ['PC', 'AFP', 'NTP']
output_cl = {}
for disposition in dispositions:
    output_cl[disposition] = rankingTbl.loc[rankingTbl['original_label'] == disposition]['score'].values

# compute accuracy for each disposition
cl_thr = 0.5  # classification threshold
results = {disposition: {'total': None, 'detected': None, 'accuracy': None} for disposition in
           dispositions}
for disposition in dispositions:
    results[disposition]['total'] = len(output_cl[disposition])
    if disposition == 'PC':
        results[disposition]['detected'] = len(np.where(output_cl[disposition] >= cl_thr)[0])
    else:
        results[disposition]['detected'] = len(np.where(output_cl[disposition] < cl_thr)[0])
    results[disposition]['accuracy'] = results[disposition]['detected'] / results[disposition]['total']

# save results to a file
with open(os.path.join(resultsDir, "res_{}_pc-afp-ntp.txt".format(rankingSet)), "w") as res_file:
    res_file.write('Performance for each category: PC, AFP, NTP\n')

    for disposition in results:
        res_file.write('Number of detected {} (Total number of): '
                       '{} ({}) | Acc = {}\n'.format(disposition,
                                                     results[disposition]['detected'],
                                                     results[disposition]['total'],
                                                     results[disposition]['accuracy']))

#%% Compute accuracy for TCEs associated with CANDIDATE, FALSE POSITIVE and DATA INCONCLUSIVE KOIs

# results directory
resultsDir = '/home/msaragoc/Projects/Kepler-TESS_exoplanet/Kepler_planet_finder/results_ensemble/' \
             'bohb_dr25_spline_gapped_hvb_glflux/'

tceTbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17 DR25/'
                     'q1_q17_dr25_tce_2020.04.15_23.19.10_cumkoi_2020.02.21_shuffled_noroguetces_norm_bhv.csv')

# remove TCEs that do not have any of these dispositions
tceTbl = tceTbl.loc[(tceTbl['koi_disposition'].isin(['CANDIDATE', 'FALSE POSITIVE'])) |
                    (tceTbl['fpwg_disp_status'] == 'DATA INCONCLUSIVE')]

# load the ranking with the cfp dispositions for a given dataset
rankingSet = 'ensemble_ranked_predictions_testset'
rankingTbl = pd.read_csv(os.path.join(resultsDir, '{}'.format(rankingSet)))

# filter out non-NTPs
rankingTbl = rankingTbl.loc[rankingTbl['original_label'] == 'NTP']
rankingTbl['koi_disposition'] = ''
rankingTbl['fpwg_disp_status'] = ''

for tce_i, tce in rankingTbl.iterrows():

    validTces = tceTbl.loc[(tceTbl['target_id'] == tce.target_id) & (tceTbl['tce_plnt_num'] == tce.tce_plnt_num) &
                           ((tceTbl['koi_disposition'].isin(['FALSE POSITIVE', 'CANDIDATE'])) |
                           (tceTbl['fpwg_disp_status'] == 'DATA INCONCLUSIVE'))][['koi_disposition',
                                                                                 'fpwg_disp_status']]

    if len(validTces) == 1:
        # print('dasd')
        rankingTbl.loc[tce_i, ['koi_disposition', 'fpwg_disp_status']] = validTces.values[0]
    # elif len(validTces) > 1:
    #     print(len(validTces))

# get scores for each disposition
dispositions = ['CANDIDATE', 'FALSE POSITIVE', 'DATA INCONCLUSIVE']
output_cl = {}
for disposition in dispositions:
    if disposition in ['CANDIDATE', 'FALSE POSITIVE']:
        output_cl[disposition] = rankingTbl.loc[rankingTbl['koi_disposition'] == disposition]['score'].values
    else:
        output_cl[disposition] = rankingTbl.loc[rankingTbl['fpwg_disp_status'] == disposition]['score'].values

# compute accuracy for each disposition
cl_thr = 0.5  # classification threshold
results = {disposition: {'total': None, 'detected': None, 'accuracy': None} for disposition in
           dispositions}
for disposition in dispositions:
    results[disposition]['total'] = len(output_cl[disposition])
    if disposition == 'CANDIDATE':
        results[disposition]['detected'] = len(np.where(output_cl[disposition] >= cl_thr)[0])
    else:
        results[disposition]['detected'] = len(np.where(output_cl[disposition] < cl_thr)[0])
    results[disposition]['accuracy'] = results[disposition]['detected'] / results[disposition]['total']

# save results to a file
with open(os.path.join(resultsDir, "res_{}_cand-fp-datainconcl.txt".format(rankingSet)), "w") as res_file:
    res_file.write('Performance for each category: CANDIDATE, FALSE POSITIVE, DATA INCONCLUSIVE\n')

    for disposition in results:
        res_file.write('Number of detected {} (Total number of): '
                       '{} ({}) | Acc = {}\n'.format(disposition,
                                                     results[disposition]['detected'],
                                                     results[disposition]['total'],
                                                     results[disposition]['accuracy']))

#%% Compute accuracy for CANDIDATE and FALSE POSITIVE KOIs which are not dispositioned as CFP, CFP or PP in the CFP list
# compute accuracy for CANDIDATE and FALSE POSITIVE KOIs which were labeled as NTP in the 'hvb' dataset

tceTbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17 DR25/'
                     'q1_q17_dr25_tce_2020.04.15_23.19.10_cumkoi_2020.02.21_shuffled_noroguetces_norm.csv')

# add FPWG disposition to the TCE table
cfpTbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/kois_tables/'
                     'cumulative_2020.02.21_10.29.22_fpwgdisp.csv')

tceTbl['fpwg_disp_status'] = ''
for tce_i, tce in tceTbl.iterrows():
    validKoi = cfpTbl.loc[cfpTbl['kepoi_name'] == tce.kepoi_name]

    assert len(validKoi) <= 1

    if len(validKoi) == 1:
        tceTbl.loc[tce_i, ['fpwg_disp_status']] = validKoi.fpwg_disp_status.values[0]

# filter out CONFIRMED, CFP, CFA, PP KOIs
tceTbl = tceTbl.loc[tceTbl['koi_disposition'] != 'CONFIRMED']
tceTbl = tceTbl.loc[~tceTbl['fpwg_disp_status'].isin(['CERTIFIED FA', 'CERTIFIED FP', 'POSSIBLE PLANET'])]

# remove TCEs that do not have any of these dispositions
tceTbl = tceTbl.loc[(tceTbl['koi_disposition'].isin(['CANDIDATE', 'FALSE POSITIVE'])) |
                    (tceTbl['fpwg_disp_status'] == 'DATA INCONCLUSIVE')]

#%% Experiment 3 (Cumulative KOI disposition dataset)

# results directory
resultsDir = '/home/msaragoc/Projects/Kepler-TESS_exoplanet/Kepler_planet_finder/results_ensemble/' \
             'dr25_spline_gapped_glflux_cumkoidisp_glfluxhvbconfig'

# load the ranking with the cfp dispositions for a given dataset
rankingSet = 'ensemble_ranked_predictions_testset'
rankingTbl = pd.read_csv(os.path.join(resultsDir, '{}'.format(rankingSet)))

rankingTbl['koi_disposition_og'] = ''

for tce_i, tce in rankingTbl.iterrows():

    validTces = tceTbl.loc[(tceTbl['target_id'] == tce.target_id) &
                           (tceTbl['tce_plnt_num'] == tce.tce_plnt_num)][['koi_disposition']]

    if len(validTces) == 1:
        # print('dasd')
        # aaaa
        rankingTbl.loc[tce_i, ['koi_disposition_og']] = validTces.values[0]
    # elif len(validTces) > 1:
    #     print(len(validTces))

# filter TCEs that did not match
rankingTbl = rankingTbl.loc[rankingTbl['koi_disposition_og'] != '']

# get scores for each disposition
dispositions = ['CANDIDATE', 'FALSE POSITIVE']
output_cl = {}
for disposition in dispositions:
    output_cl[disposition] = rankingTbl.loc[rankingTbl['koi_disposition_og'] == disposition]['score'].values

# compute accuracy for each disposition
cl_thr = 0.5  # classification threshold
results = {disposition: {'total': None, 'detected': None, 'accuracy': None} for disposition in
           dispositions}
for disposition in dispositions:
    results[disposition]['total'] = len(output_cl[disposition])
    if disposition == 'CANDIDATE':
        results[disposition]['detected'] = len(np.where(output_cl[disposition] >= cl_thr)[0])
    else:
        results[disposition]['detected'] = len(np.where(output_cl[disposition] < cl_thr)[0])
    results[disposition]['accuracy'] = results[disposition]['detected'] / results[disposition]['total']

# save results to a file
with open(os.path.join(resultsDir, "res_{}_cand-fp_tontps.txt".format(rankingSet)), "w") as res_file:
    res_file.write('Performance for each category: CANDIDATE, FALSE POSITIVE\n')

    for disposition in results:
        res_file.write('Number of detected {} (Total number of): '
                       '{} ({}) | Acc = {}\n'.format(disposition,
                                                     results[disposition]['detected'],
                                                     results[disposition]['total'],
                                                     results[disposition]['accuracy']))

#%% Experiment 2 - compute accuracy for CANDIDATE and FALSE POSITIVE KOIs which were labeled as NTP in the 'hvb' dataset

# results directory
resultsDir = '/home/msaragoc/Projects/Kepler-TESS_exoplanet/Kepler_planet_finder/results_ensemble/' \
             'dr25_spline_gapped_glflux_norobovetterkois_glfluxhvbconfig_cumkois_cand_fp_added'

# load the ranking with the cfp dispositions for a given dataset
rankingSet = 'ensemble_ranked_predictions_testset'
rankingTbl = pd.read_csv(os.path.join(resultsDir, '{}'.format(rankingSet)))

rankingTbl['koi_disposition_og'] = ''

for tce_i, tce in rankingTbl.iterrows():

    validTces = tceTbl.loc[(tceTbl['target_id'] == tce.target_id) &
                           (tceTbl['tce_plnt_num'] == tce.tce_plnt_num)][['koi_disposition']]

    if len(validTces) == 1:
        # print('dasd')
        # aaaa
        rankingTbl.loc[tce_i, ['koi_disposition_og']] = validTces.values[0]
    # elif len(validTces) > 1:
    #     print(len(validTces))

# filter TCEs that did not match
rankingTbl = rankingTbl.loc[rankingTbl['koi_disposition_og'] != '']

# get scores for each disposition
dispositions = ['CANDIDATE', 'FALSE POSITIVE']
output_cl = {}
for disposition in dispositions:
    output_cl[disposition] = rankingTbl.loc[rankingTbl['koi_disposition_og'] == disposition]['score'].values

# compute accuracy for each disposition
cl_thr = 0.335  # classification threshold
results = {disposition: {'total': None, 'detected': None, 'accuracy': None} for disposition in
           dispositions}
for disposition in dispositions:
    results[disposition]['total'] = len(output_cl[disposition])
    if disposition == 'CANDIDATE':
        results[disposition]['detected'] = len(np.where(output_cl[disposition] >= cl_thr)[0])
    else:
        results[disposition]['detected'] = len(np.where(output_cl[disposition] < cl_thr)[0])
    results[disposition]['accuracy'] = results[disposition]['detected'] / results[disposition]['total']

# save results to a file
with open(os.path.join(resultsDir, "res_{}_cand-fp_tontps_clfthr{}.txt".format(rankingSet, cl_thr)), "w") as res_file:
    res_file.write('Performance for each category: CANDIDATE, FALSE POSITIVE\n')

    for disposition in results:
        res_file.write('Number of detected {} (Total number of): '
                       '{} ({}) | Acc = {}\n'.format(disposition,
                                                     results[disposition]['detected'],
                                                     results[disposition]['total'],
                                                     results[disposition]['accuracy']))

#%% Experiment 2 - prediction using classification threshold of 0.35
# compute accuracy for CANDIDATE and FALSE POSITIVE KOIs which were labeled as NTP in the 'hvb' dataset

# results directory
resultsDir = '/home/msaragoc/Projects/Kepler-TESS_exoplanet/Kepler_planet_finder/results_ensemble/' \
             'dr25_spline_gapped_glflux_norobovetterkois_glfluxhvbconfig_cumkois_cand_fp_added_thr0.35'

# load the ranking with the cfp dispositions for a given dataset
rankingSet = 'ensemble_ranked_predictions_testset'
rankingTbl = pd.read_csv(os.path.join(resultsDir, '{}'.format(rankingSet)))

rankingTbl['koi_disposition_og'] = ''

for tce_i, tce in rankingTbl.iterrows():

    validTces = tceTbl.loc[(tceTbl['target_id'] == tce.target_id) &
                           (tceTbl['tce_plnt_num'] == tce.tce_plnt_num)][['koi_disposition']]

    if len(validTces) == 1:
        # print('dasd')
        # aaaa
        rankingTbl.loc[tce_i, ['koi_disposition_og']] = validTces.values[0]
    # elif len(validTces) > 1:
    #     print(len(validTces))

# filter TCEs that did not match
rankingTbl = rankingTbl.loc[rankingTbl['koi_disposition_og'] != '']

# get scores for each disposition
dispositions = ['CANDIDATE', 'FALSE POSITIVE']
output_cl = {}
for disposition in dispositions:
    output_cl[disposition] = rankingTbl.loc[rankingTbl['koi_disposition_og'] == disposition]['score'].values

# compute accuracy for each disposition
cl_thr = 0.35  # classification threshold
results = {disposition: {'total': None, 'detected': None, 'accuracy': None} for disposition in
           dispositions}
for disposition in dispositions:
    results[disposition]['total'] = len(output_cl[disposition])
    if disposition == 'CANDIDATE':
        results[disposition]['detected'] = len(np.where(output_cl[disposition] >= cl_thr)[0])
    else:
        results[disposition]['detected'] = len(np.where(output_cl[disposition] < cl_thr)[0])
    results[disposition]['accuracy'] = results[disposition]['detected'] / results[disposition]['total']

# save results to a file
with open(os.path.join(resultsDir, "res_{}_cand-fp_tontps_clfthr{}.txt".format(rankingSet, cl_thr)), "w") as res_file:
    res_file.write('Performance for each category: CANDIDATE, FALSE POSITIVE\n')

    for disposition in results:
        res_file.write('Number of detected {} (Total number of): '
                       '{} ({}) | Acc = {}\n'.format(disposition,
                                                     results[disposition]['detected'],
                                                     results[disposition]['total'],
                                                     results[disposition]['accuracy']))