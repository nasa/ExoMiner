"""
Post-analysis of ranking results:
1) compute and plot classification error rate as function of period and  planet radius;
2) compute and plot PC rate as function of period.
"""

# 3rd party
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

#%%

tceTbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17_DR25/q1_q17_dr25_tce_2020.04.15_23.19.10_cumkoi_2020.02.21_shuffledstar.csv')
studyDir = '/home/msaragoc/Projects/Kepler-TESS_exoplanet/Kepler_planet_finder/results_ensemble/keplerdr25_g2001-l201_spline_nongapped_norobovetterkois_configB_starshuffle_glflux-glcentrmedcmaxn-loe-lwks-6stellar-bfap-ghost/'
dataset = 'train'
rankingTbl = pd.read_csv(os.path.join(studyDir, 'ensemble_ranked_predictions_{}set'.format(dataset)))

rankingTbl = rankingTbl.loc[rankingTbl['label'] == 0]
bins = 20

min_period, max_period = tceTbl['tce_period'].min(), tceTbl['tce_period'].max()

# binsParam = np.linspace(0, 500, 40, endpoint=True)  # orbital period
binsOrbitPeriod = np.logspace(np.log10(min_period), np.log10(max_period), num=bins, endpoint=True, base=10)

paramInterval = zip(binsOrbitPeriod[:-1], binsOrbitPeriod[1:])
rankingTblBinned = []
for interval in paramInterval:
    rankingTblBinned.append(rankingTbl.loc[(rankingTbl['tce_period'] >= interval[0]) &
                                           (rankingTbl['tce_period'] < interval[1])])

#%% compute classification error per bin

clfErrorPerBinOPer = []
for bin_i in range(len(rankingTblBinned)):
    numTotalBin = len(rankingTblBinned[bin_i])
    if numTotalBin == 0:
        clfErrorPerBinOPer.append(0)
    else:
        numClfBin = (rankingTblBinned[bin_i]['label'] != rankingTblBinned[bin_i]['predicted class']).sum()
        clfErrorPerBinOPer.append(numClfBin / numTotalBin * 100)

f, ax = plt.subplots()
# ax.bar(binsParam[:-1], clfErrorPerBin, align='edge', width=np.diff(binsParam), fill=False)
ax.plot(binsOrbitPeriod, [clfErrorPerBinOPer[0]] + clfErrorPerBinOPer, ls='steps')
# ax.hist(clfErrorPerBin, bins=binsParam, fill=False, histtype='step')
ax.set_ylabel('Classification Error (%)')
ax.set_xlabel('Orbital Period (days)')
ax.set_xscale('log')  # orbital period
# ax.set_xticks(np.linspace(0, 500, num=6, endpoint=True))  # orbital period
ax.set_xticks([0.4, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0])  # orbital period
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.get_xaxis().set_tick_params(which='minor', size=0)
ax.get_xaxis().set_tick_params(which='minor', width=0)
# ax.set_xlim([0, 500])
f.savefig(os.path.join(studyDir, 'clferrorrate-orbitalperiod_{}set.svg'.format(dataset)))

rankingTbl['tce_prad'] = np.nan
for tce_i, tce in rankingTbl.iterrows():

    rankingTbl.loc[tce_i, ['tce_prad']] = tceTbl.loc[(tceTbl['target_id'] == tce['target_id']) &
                                                     ((tceTbl['tce_plnt_num'] == tce['tce_plnt_num']))]['tce_prad'].values

min_prad, max_prad = tceTbl['tce_prad'].min(), tceTbl['tce_prad'].max()
binsPrad = np.logspace(np.log10(min_prad), np.log10(30), num=bins, endpoint=True, base=10)

paramInterval = zip(binsPrad[:-1], binsPrad[1:])
rankingTblBinned = []
for interval in paramInterval:
    rankingTblBinned.append(rankingTbl.loc[(rankingTbl['tce_prad'] >= interval[0]) &
                                           (rankingTbl['tce_prad'] < interval[1])])

# compute classification error per bin
clfErrorPerBinPRad = []
for bin_i in range(len(rankingTblBinned)):
    numTotalBin = len(rankingTblBinned[bin_i])
    if numTotalBin == 0:
        clfErrorPerBinPRad.append(0)
    else:
        numClfBin = (rankingTblBinned[bin_i]['label'] != rankingTblBinned[bin_i]['predicted class']).sum()
        clfErrorPerBinPRad.append(numClfBin / numTotalBin * 100)

# plot classification error rate based orbital period and planet radius
f, ax = plt.subplots(2, 2, figsize=(12, 8), gridspec_kw={'width_ratios': [2, 1], 'height_ratios': [1, 2]})
# ax.bar(binsParam[:-1], clfErrorPerBin, align='edge', width=np.diff(binsParam), fill=False)
ax[0, 0].plot(binsOrbitPeriod, [clfErrorPerBinOPer[0]] + clfErrorPerBinOPer, ls='steps')
# ax.hist(clfErrorPerBin, bins=binsParam, fill=False, histtype='step')
ax[0, 0].set_ylabel('Classification Error (%)')
ax[0, 0].set_xscale('log')  # orbital period
# ax.set_xticks(np.linspace(0, 500, num=6, endpoint=True))  # orbital period
ax[0, 0].set_xticks([0.4, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0])  # orbital period
ax[0, 0].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax[0, 0].get_xaxis().set_tick_params(which='minor', size=0)
ax[0, 0].get_xaxis().set_tick_params(which='minor', width=0)
ax[0, 1].set_visible(False)
ax[1, 0].set_xlabel('Orbital Period (days)')
ax[1, 0].set_ylabel('Rp (R{})'.format(r"$_\bigoplus$"))
a = ax[1, 0].scatter(rankingTbl['tce_period'], rankingTbl['tce_prad'], s=10, cmap='viridis', c=rankingTbl['score'],
                     vmin=0, vmax=1)
ax[1, 0].set_xscale('log')  # orbital period
ax[1, 0].set_yscale('log')  # orbital period
ax[1, 0].set_yticks([0.4, 1.0, 2.0, 3.0, 10.0, 30.0])  # planet radius
ax[1, 0].set_ylim(top=30)
ax[1, 0].set_xticks([0.4, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0])  # orbital period
ax[1, 0].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax[1, 0].get_xaxis().set_tick_params(which='minor', size=0)
ax[1, 0].get_xaxis().set_tick_params(which='minor', width=0)
ax[1, 0].get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax[1, 0].get_yaxis().set_tick_params(which='minor', size=0)
ax[1, 0].get_yaxis().set_tick_params(which='minor', width=0)
ax[1, 0].get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax[1, 0].get_yaxis().set_tick_params(which='minor', size=0)
ax[1, 0].get_yaxis().set_tick_params(which='minor', width=0)
ax[1, 1].plot([clfErrorPerBinPRad[0]] + clfErrorPerBinPRad, binsPrad, ls='steps')
# ax[1, 1].barh(y=binsPrad[:-1], width=clfErrorPerBinPRad, align='edge', height=np.diff(binsPrad), fill=False)
ax[1, 1].set_yscale('log')
ax[1, 1].set_yticks([0.4, 1.0, 2.0, 3.0, 10.0, 30.0])  # planet radius
ax[1, 1].set_ylim([0.4, 30])
ax[1, 1].get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax[1, 1].get_yaxis().set_tick_params(which='minor', size=0)
ax[1, 1].get_yaxis().set_tick_params(which='minor', width=0)
ax[1, 1].set_xlabel('Classification Error (%)')
plt.colorbar(a, ax=ax[1, 1])
ax[1, 1].set_ylabel('Score', labelpad=50)
ax[1, 1].yaxis.set_label_position('right')
# f.colorbar()
f.savefig(os.path.join(studyDir, 'clferrorrate-orbitalperiod-plntrad_{}set.svg'.format(dataset)))

#%% PC rate as function of orbital period

tceTbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17_DR25/q1_q17_dr25_tce_2020.04.15_23.19.10_cumkoi_2020.02.21_shuffledstar.csv')
rankingTbl = pd.read_csv('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Kepler_planet_finder/results_ensemble/bohb_keplerdr25_g2001-l201_spline_nongapped_norobovetterkois_starshuffle_glflux-glcentrmedcmaxn-loe-lwks-6stellar-bfap-ghost/ensemble_ranked_predictions_testset')

# rankingTbl = rankingTbl.loc[rankingTbl['label'] == 1]
bins = 20

numTotalPCs = (rankingTbl['label'] == 1).sum()

min_period, max_period = tceTbl['tce_period'].min(), tceTbl['tce_period'].max()

# binsParam = np.linspace(0, 500, 40, endpoint=True)  # orbital period
binsOrbitPeriod = np.logspace(np.log10(min_period), np.log10(max_period), num=bins, endpoint=True, base=10)

paramInterval = zip(binsOrbitPeriod[:-1], binsOrbitPeriod[1:])
rankingTblBinned = []
for interval in paramInterval:
    rankingTblBinned.append(rankingTbl.loc[(rankingTbl['tce_period'] >= interval[0]) &
                                           (rankingTbl['tce_period'] < interval[1])])

# compute rate per bin
ratePerBinOPer = []
for bin_i in range(len(rankingTblBinned)):
    numBin = (rankingTblBinned[bin_i]['label'] == 1).sum()
    if numBin == 0:
        ratePerBinOPer.append(0)
    else:
        ratePerBinOPer.append(numBin / numTotalPCs * 100)

f, ax = plt.subplots()
# ax.bar(binsParam[:-1], clfErrorPerBin, align='edge', width=np.diff(binsParam), fill=False)
ax.plot(binsOrbitPeriod, [ratePerBinOPer[0]] + ratePerBinOPer, ls='steps')
# ax.hist(clfErrorPerBin, bins=binsParam, fill=False, histtype='step')
ax.set_ylabel('Rate (%)')
ax.set_xlabel('Orbital Period (days)')
ax.set_xscale('log')  # orbital period
# ax.set_xticks(np.linspace(0, 500, num=6, endpoint=True))  # orbital period
ax.set_xticks([0.4, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0])  # orbital period
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.get_xaxis().set_tick_params(which='minor', size=0)
ax.get_xaxis().set_tick_params(which='minor', width=0)
# ax.set_xlim([0, 500])
