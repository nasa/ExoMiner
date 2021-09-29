"""
Auxiliary script used to visualize inputs (check raw data, preprocessing results, ...)
"""

# 3rd party
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# local
from old.src_old.estimator_util import get_data_from_tfrecords

#%% Visualize input channels

tfrec_dir = '/home/msaragoc/Projects/Kepler-TESS_exoplanet/Kepler_planet_finder/src_preprocessing/tfrecords/' \
            'tfrecordkepler_dr25_flux_centroidnonnormalized_nonwhitened_gapped_2001-201'

dataset = 'train'

tfrec_filenames = [os.path.join(tfrec_dir, file)
                   for file in os.listdir(tfrec_dir) if dataset in file]

# data_fields = ['global_view', 'local_view', 'global_view_centr', 'local_view_centr', 'global_view_even',
#                'global_view_odd', 'local_view_even', 'local_view_odd', 'label']
data_fields = ['global_view_centr', 'local_view_centr', 'global_view', 'local_view']

data_dict = get_data_from_tfrecords(tfrec_filenames, data_fields, label_map=None, filt=None, coupled=False)

ntces = len(data_dict['global_view'])
chosen_idxs = np.random.choice(ntces, 340)  # np.arange(ntces)
print('Number of TCEs = {}'.format(ntces))

# print('Computing quantities...')
# stats = np.zeros((5, 4, ntces), dtype='float')
# stats = np.zeros((5, 8, len(chosen_idxs)), dtype='float')
# for i in range(ntces):
for ei, i in enumerate(chosen_idxs):
    # for j, field in enumerate(data_fields[:-1]):
        # stats[0, j, ei] = np.min(data_dict[field][i])
        # stats[1, j, ei] = np.max(data_dict[field][i])
        # stats[2, j, ei] = np.mean(data_dict[field][i])
        # stats[3, j, ei] = np.median(data_dict[field][i])
        # stats[4, j, ei] = np.std(data_dict[field][i])

    # f, ax = plt.subplots(2, 2, figsize=(12, 10))
    # ax[0, 0].plot(data_dict['global_view'][i])
    # ax[0, 0].set_title('{}\nMin={:.2f}, Max={:.2f}, Mean={:.2f}, Med={:.2f}, Std={:.2f}'.format('global_view',
    #                                                                                             stats[0, 0, ei],
    #                                                                         stats[1, 0, ei], stats[2, 0, ei],
    #                                                                         stats[3, 0, ei], stats[4, 0, ei]))
    # ax[0, 0].set_ylabel('Normalized Brightness')
    # ax[1, 0].plot(data_dict['local_view'][i])
    # ax[1, 0].set_title('{}\nMin={:.2f}, Max={:.2f}, Mean={:.2f}, Med={:.2f}, Std={:.2f}'.format('local_view',
    #                                                                                             stats[0, 1, ei],
    #                                                                         stats[1, 1, ei], stats[2, 1, ei],
    #                                                                         stats[3, 1, ei], stats[4, 1, ei]))
    # ax[1, 0].set_xlabel('Bin Number')
    # ax[1, 0].set_ylabel('Normalized Brightness')
    # ax[0, 1].plot(data_dict['global_view_centr'][i])
    # ax[0, 1].set_title('{}\nMin={:.2f}, Max={:.2f}, Mean={:.2f}, Med={:.2f}, Std={:.2f}'.format('global_view_centr',
    #                                                                                             stats[0, 2, ei],
    #                                                                         stats[1, 2, ei], stats[2, 2, ei],
    #                                                                         stats[3, 2, ei], stats[4, 2, ei]))
    # ax[0, 1].set_ylabel('Normalized Brightness')
    # ax[1, 1].plot(data_dict['local_view_centr'][i])
    # ax[1, 1].set_title('{}\nMin={:.2f}, Max={:.2f}, Mean={:.2f}, Med={:.2f}, Std={:.2f}'.format('local_view_centr',
    #                                                                                             stats[0, 3, ei],
    #                                                                         stats[1, 3, ei], stats[2, 3, ei],
    #                                                                         stats[3, 3, ei], stats[4, 3, ei]))
    # ax[1, 1].set_xlabel('Bin Number')
    # ax[1, 1].set_ylabel('Normalized Brightness')
    # f.suptitle('Label: {}\n'.format(data_dict['label'][i]))
    # f.savefig('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Kepler_planet_finder/visualize_inputs/time_series_plots/'
    #           '{}_{}.svg'.format(data_dict['label'][i], ei))
    # plt.close()

    # f, ax = plt.subplots(2, 4, figsize=(16, 11))
    # f, ax = plt.subplots(2, 3, figsize=(16, 11))
    f, ax = plt.subplots(2, 2)
    # ax[0, 0].plot(data_dict['global_view'][i])
    # ax[0, 0].set_title('{}\nMin={:.2f}, Max={:.2f}, Mean={:.2f}, \nMed={:.2f}, Std={:.2f}'.format('global_view',
    #                                                                                             stats[0, 0, ei],
    #                                                                         stats[1, 0, ei], stats[2, 0, ei],
    #                                                                         stats[3, 0, ei], stats[4, 0, ei]))
    # ax[0, 0].set_ylabel('Normalized Brightness')
    # ax[1, 0].plot(data_dict['local_view'][i])
    # ax[1, 0].set_title('{}\nMin={:.2f}, Max={:.2f}, Mean={:.2f}, \nMed={:.2f}, Std={:.2f}'.format('local_view',
    #                                                                                             stats[0, 1, ei],
    #                                                                         stats[1, 1, ei], stats[2, 1, ei],
    #                                                                         stats[3, 1, ei], stats[4, 1, ei]))
    # ax[1, 0].set_xlabel('Bin Number')
    # ax[1, 0].set_ylabel('Normalized Brightness')
    # ax[0, 1].plot(data_dict['global_view_centr'][i])
    # ax[0, 1].set_title('{}\nMin={:.2f}, Max={:.2f}, Mean={:.2f}, \nMed={:.2f}, Std={:.2f}'.format('global_view_centr',
    #                                                                                             stats[0, 2, ei],
    #                                                                         stats[1, 2, ei], stats[2, 2, ei],
    #                                                                         stats[3, 2, ei], stats[4, 2, ei]))
    # ax[1, 1].plot(data_dict['local_view_centr'][i])
    # ax[1, 1].set_title('{}\nMin={:.2f}, Max={:.2f}, Mean={:.2f}, \nMed={:.2f}, Std={:.2f}'.format('local_view_centr',
    #                                                                                             stats[0, 3, ei],
    #                                                                         stats[1, 3, ei], stats[2, 3, ei],
    #                                                                         stats[3, 3, ei], stats[4, 3, ei]))
    # ax[1, 1].set_xlabel('Bin Number')
    # ax[0, 2].plot(data_dict['global_view_even'][i])
    # ax[0, 2].set_title('{}'.format('global_view_even-odd'))
    # # ax[0, 2].set_title('{}\nMin={:.2f}, Max={:.2f}, Mean={:.2f}, \nMed={:.2f}, Std={:.2f}'.format('global_view_even',
    # #                                                                                               stats[0, 4, ei],
    # #                                                                                               stats[1, 4, ei],
    # #                                                                                               stats[2, 4, ei],
    # #                                                                                               stats[3, 4, ei],
    # #                                                                                               stats[4, 4, ei]))
    # ax[1, 2].plot(data_dict['local_view_even'][i])
    # ax[1, 2].set_title('{}'.format('local_view_even-odd'))
    # # ax[1, 2].set_title('{}\nMin={:.2f}, Max={:.2f}, Mean={:.2f}, \nMed={:.2f}, Std={:.2f}'.format('local_view_even',
    # #                                                                                             stats[0, 5, ei],
    # #                                                                         stats[1, 5, ei], stats[2, 5, ei],
    # #                                                                         stats[3, 5, ei], stats[4, 5, ei]))
    # ax[1, 2].set_xlabel('Bin Number')
    # # ax[0, 3].plot(data_dict['global_view_odd'][i])
    # # ax[0, 3].set_title('{}\nMin={:.2f}, Max={:.2f}, Mean={:.2f}, \nMed={:.2f}, Std={:.2f}'.format('global_view_odd',
    # #                                                                                             stats[0, 6, ei],
    # #                                                                         stats[1, 6, ei], stats[2, 6, ei],
    # #                                                                         stats[3, 6, ei], stats[4, 6, ei]))
    # # ax[1, 3].plot(data_dict['local_view_odd'][i])
    # # ax[1, 3].set_title('{}\nMin={:.2f}, Max={:.2f}, Mean={:.2f}, \nMed={:.2f}, Std={:.2f}'.format('local_view_odd',
    # #                                                                                             stats[0, 7, ei],
    # #                                                                                             stats[1, 7, ei],
    # #                                                                                             stats[2, 7, ei],
    # #                                                                                             stats[3, 7, ei],
    # #                                                                                             stats[4, 7, ei]))
    # # ax[1, 3].set_xlabel('Bin Number')
    #
    # ax[0, 2].plot(data_dict['global_view_odd'][i], 'r--')
    # ax[1, 2].plot(data_dict['local_view_odd'][i], 'r--')

    ax[0, 0].plot(data_dict['global_view'][i])
    ax[0, 0].set_title('Global view')
    ax[1, 0].set_xlabel('Bin Number')
    ax[1, 0].set_ylabel('Amplitude')
    ax[1, 0].plot(data_dict['global_view_centr'][i])

    ax[0, 1].plot(data_dict['local_view'][i])
    ax[0, 1].set_title('Local view')
    ax[1, 1].set_xlabel('Bin Number')
    # ax[0, 1].set_ylabel('Brightness')
    ax[1, 1].plot(data_dict['local_view_centr'][i])


    # f.subplots_adjust(top=0.88, bottom=0.11, left=0.125, right=0.9, hspace=0.245, wspace=0.2)
    # f.suptitle('Label: {}\n'.format(data_dict['label'][i]))
    f.suptitle('Non-normalized flux and centroid views')
    # f.savefig('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Kepler_planet_finder/visualize_inputs/'
    #           'time_series_plots_oddeven_normed/{}_{}.svg'.format(data_dict['label'][i], ei))
    f.savefig('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Kepler_planet_finder/visualize_inputs/'
              'non-normalized_fluxandcentroid/{}.svg'.format(ei))
    # aaaa
    plt.close()

# quantities = ['Min', 'Max', 'Mean', "Median", "Std"]
# idxs_tces = np.arange(ntces)  # np.where(np.array(data_dict['label']) == 'NTP')[0]
# print('Number of TCEs used = {}'.format(len(idxs_tces)))
# for quantity_i in range(len(stats)):
#     for j, field in enumerate(data_fields[:-1]):
#         print('Plotting histogram for quantity {} for input {}'.format(quantities[quantity_i], field))
#         plt.figure()
#         plt.hist(stats[quantity_i, j, idxs_tces], bins=np.arange(-1.5, 1.5, 0.1))
#         quantity_mean, quantity_std = np.mean(stats[quantity_i, j, idxs_tces]), np.std(stats[quantity_i, j, idxs_tces])
#         plt.title('{} {}\nMean={:.2f}, Std={:.2f}'.format(quantities[quantity_i], field, quantity_mean, quantity_std))
#         plt.ylabel('Number of TCEs')
#         plt.xlabel('Normalized Amplitude')
#         plt.savefig('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Kepler_planet_finder/visualize_inputs/testset/'
#                     '{}_{}.svg'.format(field, quantities[quantity_i]))
#         plt.close()

#%% Plot histogram for each class in each dataset (train, val and test) as a function of MES

save_path = '/home/msaragoc/Projects/Kepler-TESS_exoplanet/Kepler_planet_finder/visualize_inputs/histogram_classes_mes/'

tfrec_dir = '/data5/tess_project/Data/tfrecords/dr25_koilabels/' \
            'tfrecord_dr25_manual_2dkeplernonwhitened_gapped_oddeven_centroid'
# tfrec_dir = '/home/msaragoc/Projects/Kepler-TESS_exoplanet/Data/' \
#                                        'tfrecord_dr25_manual_2dkeplernonwhitened_2001-201'
upd_tce_table = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/'
                            'q1_q17_dr25_tce_2019.08.07_16.23.32_updt_tcert.csv')

datasets = ['train', 'val', 'test']
tfrec_filenames = {dataset: [os.path.join(tfrec_dir, file) for file in os.listdir(tfrec_dir) if dataset in file] for
                   dataset in datasets}
# data_fields = ['label', 'MES']
data_fields = ['label', 'kepid', 'tce_n']

data_dict = {dataset: get_data_from_tfrecords(tfrec_filenames[dataset], data_fields, label_map=None, filt=None,
                                              coupled=False) for dataset in datasets}

# for dataset in datasets:
#     print('Min and Max MES values for dataset {}'.format(dataset), np.min(data_dict[dataset]['MES']),
#           np.max(data_dict[dataset]['MES']))

# nsamples_class = {len(data_dict[dataset]['label']) for dataset in datasets}
# # nsamplestotal = np.sum([len(data_dict[dataset]['label']) for dataset in datasets])
# nsamplestotal = np.sum(data_dict.values())

classes = ['AFP', 'NTP', 'PC']

mes_bins = np.arange(0, 12, 0.5)
for dataset in datasets:
    for class_i in classes:

        # get data pertaining to the given glass
        idxs_class_i = np.where(np.array(data_dict[dataset]['label']) == class_i)[0]
        data_aux = {field: np.array(data_dict[dataset][field])[idxs_class_i]for field in data_dict[dataset]}

        # get MES for those examples
        mes_vec = []
        for i in range(len(data_aux['kepid'])):
            mes_vec.append(
                upd_tce_table.loc[(upd_tce_table['kepid'] == data_aux['kepid'][i]) &
                                  (upd_tce_table['tce_plnt_num'] == data_aux['tce_n'][i])]['tce_max_mult_ev'].values[0])

        print('MES range:', min(mes_vec), max(mes_vec))
        # hist, bin_edges = np.histogram(data_aux['MES'], mes_bins, density=False, range=None)
        hist, bin_edges = np.histogram(mes_vec, mes_bins, density=False, range=None)

        f, ax = plt.subplots()
        ax.bar([(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(len(bin_edges) - 1)],
               hist, mes_bins[1] - mes_bins[0], edgecolor='k')
        ax.set_ylabel('Number of TCEs')
        ax.set_xlabel('MES')
        ax.set_xlim([0, mes_bins[-1]])
        # ax.set_ylim([0, 1])
        ax.set_title('Class {} - Dataset {}'.format(class_i, dataset))
        # ax.set_xticks(np.linspace(0, 1, 11, True))
        ax.set_xticks(np.arange(0, 13, 1))
        # ax.legend()
        plt.savefig(save_path + 'hist_class{}_mes_{}.svg'.format(class_i, dataset))
        plt.close()
        # aaaa
