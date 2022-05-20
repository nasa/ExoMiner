import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

#%% create table from the single-sector runs that Jeff provided
# TODO: ephemeris cross-matching; TCEs belonging to the same target but in different sectors can have their planet id
#       changed from one sector to the other

# 4 TCEs in sector 3 change label from b01 to b02
# 1 TCE in sector 8 changes label from b01 to b02

toi_csv_dir = '/data5/tess_project/Data/Ephemeris_tables/TESS/Dispositions_TOI_list_from_jeff(9-3-2019)/' \
              'single-sector runs'
toi_csv_files = [os.path.join(toi_csv_dir, toi_csv_file) for toi_csv_file in os.listdir(toi_csv_dir)
                 if 'csv-file' in toi_csv_file]

toi_tbl = {'tic': np.array([], dtype='uint64'), 'tce_plnt_num': np.array([], dtype='uint8'),
           'disposition': np.array([], dtype='<U3'), 'sector': np.array([], dtype='uint8')}

for toi_csv_file in toi_csv_files:

    print('Reading {}'.format(toi_csv_file))

    ephem_tbl_filename = toi_csv_file.split('/')[-1]
    # assuming filename aaa-aaa-aaaa-aa-aa-aaaa-sNN-....csv where NN = {01, 02, ..., 11, ...}
    sector = int(ephem_tbl_filename.split('-')[6][1:])

    print('Sector {}'.format(sector))

    toi_df = pd.read_csv(toi_csv_file)
    # aaa
    for i in range(len(toi_df)):

        # if toi_df.loc[i, 'cid'] != 1 or str(toi_df.loc[i, 'disposition']) == 'nan':
        if str(toi_df.loc[i, 'disposition']) == 'nan':
            # print('Entry ignored: ', toi_df.loc[i, ['tic', 'cid', 'disposition']])
            continue

        idx = np.where(toi_tbl['tic'] == toi_df.loc[i, 'tic'])[0]
        idx2 = np.where(toi_tbl['sector'] == sector)[0]
        idx_intersect = np.intersect1d(idx, idx2)
        idx3 = np.where(toi_tbl['tce_plnt_num'] == toi_df.loc[i, 'cid'])[0]
        idx_intersect = np.intersect1d(idx_intersect, idx3)
        # if toi_df.loc[i, 'tic'] in toi_tbl['tic']:
        if len(idx_intersect) > 0:  # and toi_tbl['disposition'][idx_intersect] != toi_df.loc[i, 'disposition']:
            print('#' * 100)
            print('TCE TIC {} CID {} Sector {} '
                  'was already added to the table.'.format(toi_df.loc[i, 'tic'], toi_df.loc[i, 'cid'], sector))
            print('Already added: ', toi_tbl['tic'][idx_intersect], toi_tbl['tce_plnt_num'][idx_intersect], toi_tbl['disposition'][idx_intersect],
                  toi_tbl['sector'][idx_intersect])
            print('Current entry: ', toi_df.loc[i, ['tic', 'cid', 'disposition']],
                  'sector {}'.format(sector))
            if toi_tbl['disposition'][idx_intersect] != toi_df.loc[i, 'disposition']:
                print('DIFFERENT LABELS!!!')
            # aaaaa
        else:
            toi_tbl['tic'] = np.r_[toi_tbl['tic'], toi_df.loc[i, 'tic']]
            toi_tbl['tce_plnt_num'] = np.r_[toi_tbl['tce_plnt_num'], toi_df.loc[i, 'cid']]
            toi_tbl['disposition'] = np.concatenate((toi_tbl['disposition'], [toi_df.loc[i, 'disposition']]))
            toi_tbl['sector'] = np.r_[toi_tbl['sector'], sector]
            # toi_tbl['sector'] = np.concatenate((toi_tbl['sector'], [toi_csv_file.split('/')[-1]]))
            # print('TIC added to the table.', toi_tbl['tic'][-1], toi_tbl['disposition'][-1])

    # aaa

new_toi_df = pd.DataFrame(toi_tbl)

#%% add ephemeris per sector to the TCE table with dispositions

ephem_tbl_dir = '/data5/tess_project/Data/Ephemeris_tables/TESS/DV_ephemeris/single-sector runs(9-11-2019)'
ephem_tbl_files = [os.path.join(ephem_tbl_dir, ephem_tbl) for ephem_tbl in os.listdir(ephem_tbl_dir)
                   if 'tcestats' in ephem_tbl]

# fields to be extracted from the ephemeris tables
fields = ['transitDurationHours', 'orbitalPeriodDays', 'transitEpochBtjd']
add_fields = ['starTeffKelvin', 'planetRadiusEarthRadii', 'starRadiusSolarRadii', 'transitDepthPpm', 'mes',
              'InsolationFlux']

fields += add_fields

# create new columns for those fields
for field in fields:
    new_toi_df[field] = np.nan

for ephem_tbl_file in ephem_tbl_files:

    # read ephemeris csv file
    ephem_tbl = pd.read_csv(ephem_tbl_file, header=6)

    # get sector
    ephem_tbl_filename = ephem_tbl_file.split('/')[-1]
    sector = int(ephem_tbl_filename.split('-')[1][1:])

    # create new column for the TCE planet number using the TCE ID (TOI ID - PLNT NUMBER: 01, 02, ...)
    ephem_tbl['tce_plnt_num'] = ephem_tbl['tceid']
    ephem_tbl['tce_plnt_num'] = ephem_tbl['tce_plnt_num'].apply(lambda x: int(x.split('-')[1]))

    print('Iterating over ephemeris table {}'.format(ephem_tbl_file))

    print('Sector {}'.format(sector))

    for i in range(len(ephem_tbl)):
        # if i == 59:
        #     print(ephem_tbl.iloc[i]['ticid'])
        #     aaa
        new_toi_df.loc[(new_toi_df['sector'] == sector) &
                       (new_toi_df['tic'] == ephem_tbl.iloc[i]['ticid']) &
                       (new_toi_df['tce_plnt_num'] == ephem_tbl.iloc[i]['tce_plnt_num']),
                       fields] = ephem_tbl.iloc[i][fields].values


# new_toi_df = pd.DataFrame(toi_tbl)
new_toi_df.to_csv('/home/msaragoc/Downloads/toi_list.csv', index=False)

# # count how many with disposition + ephemeris
# # 923, 661 (TCE 1)
# j = 0
# for i in range(len(new_toi_df)):
#     # if '_1' in tceid[i]:
#     if ~np.isnan(new_toi_df.iloc[i]['transitDurationHours']):  # and new_toi_df.iloc[i]['tce_plnt_num'] == 1:
#         j += 1

# filter TCEs without disposition + ephemeris
new_toi_df_dispEphem = new_toi_df.loc[~np.isnan(new_toi_df['transitDurationHours'])]
new_toi_df_dispEphem.to_csv('/home/msaragoc/Downloads/toi_list_dispEphem.csv', index=False)

#%% check tev.mit.edu and NASA Exoplanet Archive TCE lists

toi_tbl_tevmit = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/toi-plus-2019-08-30.csv')

toi_dict = {}

for i in range(len(toi_tbl_tevmit)):
    if toi_tbl_tevmit.loc[i, 'tic_id'] not in toi_dict.keys():
        toi_dict[toi_tbl_tevmit.loc[i, 'tic_id']] = [toi_tbl_tevmit.loc[i, 'toi_id']]
    else:
        toi_dict[toi_tbl_tevmit.loc[i, 'tic_id']].append(toi_tbl_tevmit.loc[i, 'toi_id'])

toi_tbl_nasaexoarch = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/TOI_2019.08.30_11.37.22.csv', header=72)

toi_dict = {}

for i in range(len(toi_tbl_nasaexoarch)):
    if toi_tbl_nasaexoarch.loc[i, 'tid'] not in toi_dict.keys():
        toi_dict[toi_tbl_nasaexoarch.loc[i, 'tid']] = [toi_tbl_nasaexoarch.loc[i, 'toi']]
    else:
        toi_dict[toi_tbl_nasaexoarch.loc[i, 'tid']].append(toi_tbl_nasaexoarch.loc[i, 'toi'])

#%%% Counting how many TCEs are in different TESS sectors

toi_tbl = '/home/msaragoc/Downloads/toi_list_dispEphem.csv'

toi_tbl = pd.read_csv(toi_tbl)

tceid = toi_tbl[['tic', 'tce_plnt_num']].apply(lambda x: '{:d}_{:d}'.format(int(x['tic']), int(x['tce_plnt_num'])),

                                               axis=1)
# j = 0
# for i in range(len(tceid)):
#     if '_1' in tceid[i]:
#         j += 1

toi_dict = {}
for i in range(len(toi_tbl)):

    if '_1' in tceid[i]:  # only TCEs coming from TPS
        if tceid.iloc[i] not in toi_dict:
            # toi_dict[tceid.iloc[i]] = {'sector': [], 'disposition': []}
            # toi_dict[tceid.iloc[i]]['sector'] = [toi_tbl.iloc[i]['sector']]
            # toi_dict[tceid.iloc[i]]['disposition'] = [toi_tbl.iloc[i]['disposition']]

            toi_dict[tceid.iloc[i]] = [toi_tbl.iloc[i]['sector']]
        else:
            # # add new entry if TCE shows up in a new sector with a disposition that was not seen so far
            # if toi_tbl.iloc[i]['disposition'] not in toi_dict[tceid.iloc[i]]['disposition']:
            #     toi_dict[tceid.iloc[i]]['disposition'].append(toi_tbl.iloc[i]['disposition'])
            #     toi_dict[tceid.iloc[i]]['sector'].append(toi_tbl.iloc[i]['sector'])

            toi_dict[tceid.iloc[i]].append(toi_tbl.iloc[i]['sector'])

sector_counts = [len(toi_dict[key]) for key in toi_dict]
# sector_counts = [len(toi_dict[key]['sector']) for key in toi_dict]
# sector_counts /= np.sum(sector_counts)
hist, bins = np.histogram(sector_counts, bins=np.arange(0, 14, 1))
# hist = hist / np.sum(sector_counts)

f, ax = plt.subplots()
# ax.hist(sector_counts)
# bins = np.arange(0, len(hist), 1)
ax.bar(bins[:-1], hist, width=1, align='edge')
# ax.set_xscale("log", nonposx='clip')
ax.set_yscale("log", nonposy='clip')
# plt.hist()
ax.set_xlabel('Number of sectors in which a TCE shows up')
# ax.set_xlabel('Number of unique dispositions for the TCEs across sectors')
ax.set_ylabel('Number of TCEs')
ax.set_xticks(bins)
ax.set_xlim(left=1)
ax.set_ylim(ymin=0.1)

#%% Plot histograms of the TCE period in the Ephemeris tables

ephem_tbl_file = '/data5/tess_project/Data/Ephemeris_tables/TESS/toi_list_ssectors_dvephemeris.csv'
ephem_tbl = pd.read_csv(ephem_tbl_file)

range_period = [0, 0.75]
f, ax = plt.subplots()
ax.hist(ephem_tbl['orbitalPeriodDays'], bins='auto', range=range_period)
ax.set_ylabel('Number of TCEs')
ax.set_xlabel('Bins TCE period')
ax.set_xlim(range_period)
ax.set_xlim(left=0)
ax.set_title('Histogram TCE period')
f.savefig('/home/msaragoc/Downloads/tce_period_hist_TESS-ssectors1-12_all.svg')
# plt.show()
