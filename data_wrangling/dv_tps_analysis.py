"""
Compare the TCE parameters between TPS and DV.
"""

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

#%% plot DV vs TPS TCE parameters

# load DV TCE table
dvTbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17_DR25/'
                    'q1_q17_dr25_tce_2020.04.15_23.19.10_cumkoi_2020.02.21_shuffled.csv')

# filter out rogue TCEs and TCEs with TCE planet number above 1
dvTbl = dvTbl.loc[dvTbl['tce_plnt_num'] == 1]

# load TPS TCE table
tpsTbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/TPS_tables/Q1-Q17 DR25/'
                     'keplerTPS_KSOP2536_tces.csv')

# make sure the number of TCEs is the same
assert len(dvTbl) == len(tpsTbl)

# list of parameters to analyze
params = ['tce_period', 'tce_duration', 'tce_time0bk', 'transit_depth']

# initialize data dictionaries
dv_arr, tps_arr = {param: [] for param in params}, {param: [] for param in params}

# get TCE parameters from both TCE tables
for tce_i, tce in dvTbl.iterrows():  # iterate throught the DV TCE table
    foundTce = tpsTbl.loc[(tpsTbl['target_id'] == tce['target_id']) &
                          (tpsTbl['tce_plnt_num'] == tce['tce_plnt_num'])][params]

    assert len(foundTce) == 1  # it must find the TCE in the TPS TCE table

    for param in params:
        dv_arr[param].append(tce[param])
        tps_arr[param].append(foundTce[param].values[0])

# scatter plot for each parameter
for param in params:

    f, ax = plt.subplots()
    if param == 'transit_depth':
        ax.scatter(dv_arr[param], np.array(tps_arr[param]) * 1e6, c='b', s=10)
    else:
        ax.scatter(dv_arr[param], tps_arr[param], c='b', s=10)
    # ax.plot(np.arange(0, 1e7, 10), np.arange(0, 1e7, 10), color='r', linestyle='dashed')  # transit depth
    ax.plot(np.arange(0, 1e3, 10), np.arange(0, 1e3, 10), color='r', linestyle='dashed')  # transit depth

    # ax.set_title(param)
    # ax.set_ylim([1e-1, 1e7])  # transit depth
    # ax.set_xlim([1e-6, 1e7])
    # ax.set_xlim([0, 750])  # orbital period
    # ax.set_ylim([0, 750])
    # ax.set_xlim([0, 160])  # transit duration
    # ax.set_ylim([0, 16])
    ax.set_xlim([130, 630])
    ax.set_ylim([130, 630])
    ax.set_ylabel('TPS Epoch (day)', size=14)
    ax.set_xlabel('DV Epoch (day)', size=14)
    # ax.set_yscale('log')  # transit depth
    # ax.set_xscale('log')
    # ax.set_xticks(np.logspace(-6, 8, num=14, endpoint=False))  # transit depth
    # ax.set_yticks(np.linspace(0, 16, 17, endpoint=True))  # transit duration
    # ax.set_xticks(np.linspace(0, 750, num=16, endpoint=True))  # orbital period
    # ax.set_yticks(np.linspace(0, 750, num=16, endpoint=True))
    ax.set_xticks(np.linspace(130, 630, num=11, endpoint=True))  # epoch
    ax.set_yticks(np.linspace(130, 630, num=11, endpoint=True))
    ax.grid(True)
    f.savefig('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Analysis/tps_dv_tceparams/{}.png'.format(param))
    plt.close()
