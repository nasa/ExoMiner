"""
Script used to analyse the Robovetter results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%% Update Cumulative KOI list with the dispositions from the Certified False Positive list

cumKoiTbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/kois_tables/'
                        'cumulative_2020.02.21_10.29.22.csv', header=90)
cfpTbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/kois_tables/fpwg_2020.03.13_11.37.49.csv',
                     header=13)

# initialize fpwg disposition column in the Cumulative KOI table
cumKoiTbl['fpwg_disp_status'] = ''

countUpdated = 0
for koi_i, koi in cfpTbl.iterrows():

    # validKoi = cumKoiTbl.loc[cumKoiTbl['kepoi_name'] == koi.kepoi_name]

    # assert len(validKoi) <= 1

    # if len(validKoi) == 1:
    cumKoiTbl.loc[cumKoiTbl['kepoi_name'] == koi.kepoi_name, 'fpwg_disp_status'] = koi['fpwg_disp_status']
    countUpdated += 1

print('Number of KOIs in the Cumulative KOI list which were matched to a KOI in the Certified False Positive list: '
      '{}'.format(countUpdated))

cumKoiTbl.to_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/kois_tables/cumulative_2020.02.21_10.29.22_'
                 'fpwgdisp.csv', index=False)

#%% Plot histograms of Robovetter score for each disposition

# filter out KOIs not associated with TCEs from Q1-Q17 DR25
cumKoiTbl = cumKoiTbl.loc[cumKoiTbl['koi_tce_delivname'] == 'q1_q17_dr25_tce']

cumKoiDisp = cumKoiTbl['koi_disposition'].unique()
cfpKoiDisp = cumKoiTbl['fpwg_disp_status'].unique()

bins = np.linspace(0, 1, num=11, endpoint=True)
fontSize = 15
# ymax = 5000
for disp in np.concatenate([cumKoiDisp, cfpKoiDisp]):
# for disp in ['CERTIFIED FP']:

    if disp in cumKoiDisp:
        scores = cumKoiTbl.loc[cumKoiTbl['koi_disposition'] == disp]['koi_score']
    else:
        # if disp == 'CERTIFIED FP':
        #     scores = cumKoiTbl.loc[cumKoiTbl['fpwg_disp_status'].isin(['CERTIFIED FP', 'CERTIFIED FA'])]['koi_score']
        # elif disp == 'CERTIFIED FA':
        #     continue
        # else:
        scores = cumKoiTbl.loc[cumKoiTbl['fpwg_disp_status'] == disp]['koi_score']

    f, ax = plt.subplots(figsize=(12, 8))
    ax.hist(scores, bins, edgecolor='k')
    ax.set_xlabel('Robovetter score', size=fontSize)
    ax.set_ylabel('Number of KOIs', size=fontSize)
    ax.set_yscale('log')
    ax.set_xlim([0, 1])
    ax.set_xticks(np.linspace(0, 1, num=11, endpoint=True))
    ax.tick_params(labelsize=15)
    # ax.set_ylim(top=ymax)
    # if disp == 'CERTIFIED FP':
    #     ax.set_title('{}'.format('CERTIFIED FP+FA'), size=fontSize)
    # else:
    ax.set_title('{}'.format(disp), size=fontSize)
    ax.grid(True)
    # aaa
    # if disp == 'CERTIFIED FP':
    #     f.savefig('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Analysis/robovetter_scores_analysis/'
    #               'hist_robovetterscore_{}.png'.format('CERTIFIED FP+FA'))
    # else:
    f.savefig('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Analysis/robovetter_scores_analysis/'
              'hist_robovetterscore_{}.png'.format(disp))
    plt.close()
