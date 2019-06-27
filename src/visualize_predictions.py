# 3rd party
import os
import numpy as np
import matplotlib.pyplot as plt

# local
import paths
import src.config
from src.estimator_util import get_data_from_tfrecords

pathres = '/home/msaragoc/Projects/Kepler-TESS_exoplanet/Kepler_planet_finder/visualize_lc_modelpred/'

satellite = 'kepler'
multi_class = False
label_map = src.config.label_map[satellite][multi_class]
study = 'study_bohb_dr25_tcert_spline2'

# tfrecord directory
tfrec_dir = paths.tfrec_dir['DR25']['spline']['TCERT']

# choose dataset of interest
dataset = 'train'  # 'test'
# choose data extracted from the tfrecords
data_fields = ['kepid', 'global_view', 'local_view', 'labels', 'tce_n']  # 'epoch', 'tce_duration', 'tce_period', 'MES', 'global_view_centr', 'local_view_centr']
tfrec_filenames = [os.path.join(tfrec_dir, file) for file in os.listdir(tfrec_dir) if dataset in file]
data_dict = get_data_from_tfrecords(tfrec_filenames, data_fields, label_map=label_map)

# load predictions
predictions = np.load(paths.pathsaveres_get_pcprobs + study + '/predictions_per_dataset.npy').item()[dataset]

idxs_labels = np.where(np.array(data_dict['labels'], dtype='int') == label_map['AFP'])
thr_predout = 0.9
idxs_predictions = np.where(predictions > thr_predout)
idxs_interest = np.intersect1d(idxs_labels[0], idxs_predictions[0], assume_unique=True)
# idxs_interest = np.random.choice(idxs_interest, size=50)
print('Number of TCEs of interest: {}'.format(len(idxs_interest)))

plotname = 'fp_above_.9'
# plot_fields = ['global_view', 'local_view']
for idx in idxs_interest:
    f, ax = plt.subplots(2, 1, figsize=(10, 7))
    ax[0].plot(data_dict['global_view'][int(idx)])
    ax[0].set_title('{}'.format('global_view'))
    ax[0].set_ylabel('Normalized Brightness')
    ax[1].plot(data_dict['local_view'][int(idx)])
    ax[1].set_title('{}'.format('local_view'))
    ax[1].set_xlabel('Bin Number')
    ax[1].set_ylabel('Normalized Brightness')
    f.suptitle('Kepler ID: {} | TCE PLNT: {}\n '
               'Period=  Duration=  Epoch='.format(data_dict['kepid'][int(idx)],
                                                                 data_dict['tce_n'][int(idx)],
                                                                 # data_dict['tce_period'][int(idx)],
                                                                 # data_dict['tce_duration'][int(idx)],
                                                                 # data_dict['epoch'][int(idx)]
                                                         ))
    f.savefig('{}{}_kepid{}_tcen{}.png'.format(pathres, plotname, data_dict['kepid'][int(idx)],
                                               data_dict['tce_n'][int(idx)]))
    # aa
    plt.close()

    # f, ax = plt.subplots(2, 2, figsize=(10, 7))
    # ax[0, 0].plot(data_dict['global_view'][int(idx)])
    # ax[0, 0].set_title('{}'.format('global_view'))
    # ax[0, 0].set_ylabel('Normalized Brightness')
    # ax[1, 0].plot(data_dict['local_view'][int(idx)])
    # ax[1, 0].set_title('{}'.format('local_view'))
    # ax[1, 0].set_xlabel('Bin Number')
    # ax[1, 0].set_ylabel('Normalized Brightness')
    # ax[0, 1].plot(data_dict['global_view_centr'][int(idx)])
    # ax[0, 1].set_title('{}'.format('global_view_centr'))
    # ax[0, 1].set_ylabel('Normalized Brightness')
    # ax[1, 1].plot(data_dict['local_view_centr'][int(idx)])
    # ax[1, 1].set_title('{}'.format('local_view_centr'))
    # ax[1, 1].set_xlabel('Bin Number')
    # ax[1, 1].set_ylabel('Normalized Brightness')
    # f.suptitle('Kepler ID: {}'.format(data_dict['kepid'][int(idx)]))
    # f.savefig('{}{}_kepid{}.png'.format(pathres, plotname, data_dict['kepid'][int(idx)]))
    # aaaa
    # plt.close()


#%% get common observations that are present in both datasets - DV and TPS ephemeris based

kepids_data = {ephemeris_src: {dataset: [] for dataset in ['train', 'val', 'test']} for ephemeris_src in ['dv', 'tps']}

tfrec_dirs = {'dv': '/data5/tess_project/Data/tfrecords/dr25_koilabels/tfrecord_vanilla',
              'tps': '/data5/tess_project/Data/tfrecords/dr25_koilabels/tfrecord_vanilla_tps'}

data_fields = ['kepid', 'tce_n']
for tfrec_dir in tfrec_dirs:
    for dataset in ['train', 'val', 'test']:
        tfrec_filenames = [os.path.join(tfrec_dirs[tfrec_dir], file)
                           for file in os.listdir(tfrec_dirs[tfrec_dir]) if dataset in file]

        data = get_data_from_tfrecords(tfrec_filenames, data_fields, label_map=label_map)
        print(tfrec_dir, dataset, len(data['kepid']), len(data['tce_n']))
        idxs_tce1 = np.where(np.array(data['tce_n'], dtype='uint64') == 1)
        kepids_data[tfrec_dir][dataset] = np.array(data['kepid'], dtype='int')[idxs_tce1]


# get the common ones
cmmn_kepids = {dataset: np.intersect1d(kepids_data['tps'][dataset], kepids_data['dv'][dataset])
               for dataset in ['train', 'val', 'test']}

# np.save('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Kepler_planet_finder/cmmn_kepids', cmmn_kepids)

# %%

tfrec_dirs = {'dv': paths.tfrec_dir['DR25']['spline']['TCERT'],
              'tps': '/data5/tess_project/Data/tfrecords/dr25_koilabels/tfrecord_vanilla_tps/'}
satellite = 'kepler'
multi_class = False
cmmn_kepids = np.load('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Kepler_planet_finder/cmmn_kepids.npy').item()
data_fields = ['kepid']  # ['kepid', 'global_view', 'local_view', 'labels', 'tce_n']  # 'epoch', 'tce_duration', 'tce_period', 'MES', 'global_view_centr', 'local_view_centr']
thr_clf = 0.5

label_map = src.config.label_map[satellite][multi_class]

for tfrec_dir in tfrec_dirs:
    for dataset in ['train', 'val', 'test']:
        tfrec_filenames = [os.path.join(tfrec_dirs[tfrec_dir], file)
                           for file in os.listdir(tfrec_dirs[tfrec_dir]) if dataset in file]

        data_dict = get_data_from_tfrecords(tfrec_filenames, data_fields, label_map=label_map,
                                            filt_by_kepids=cmmn_kepids[dataset])
        print(tfrec_dir, dataset, len(cmmn_kepids[dataset]), len(data_dict['kepid']))

        # aaaaa
        #
        # predictions = {'dv': np.load(paths.pathsaveres_get_pcprobs + study +
        #                              '/dv_ephemeris/predictions_per_dataset.npy').item()[dataset],
        #                'tps': np.load(paths.pathsaveres_get_pcprobs + study +
        #                               '/tps_ephemeris/predictions_per_dataset.npy').item()[dataset]}
        #
        # predictions_clf = {'dv': np.zeros(predictions['dv'].shape, dtype='uint8'),
        #                    'tps': np.zeros(predictions['dv'].shape, dtype='uint8')}
        #
        # predictions_clf['dv'][np.where(predictions['dv'] >= 0.5)] = 1
        # predictions_clf['tps'][np.where(predictions['tps'] >= 0.5)] = 1
        #
        # idxs_tps_miss = np.where(predictions_clf['tps'] == data_dict['labels'])
        #
        #
        # thr_predout = 0.9
        # idxs_predictions = np.where(predictions > thr_predout)
        # idxs_interest = np.intersect1d(idxs_labels[0], idxs_predictions[0], assume_unique=True)
        # # idxs_interest = np.random.choice(idxs_interest, size=50)
        # print('Number of TCEs of interest: {}'.format(len(idxs_interest)))
        #
        # for idx in idxs_interest:
        #     f, ax = plt.subplots(2, 1, figsize=(10, 7))
        #     ax[0].plot(data_dict['global_view'][int(idx)])
        #     ax[0].set_title('{}'.format('global_view'))
        #     ax[0].set_ylabel('Normalized Brightness')
        #     ax[1].plot(data_dict['local_view'][int(idx)])
        #     ax[1].set_title('{}'.format('local_view'))
        #     ax[1].set_xlabel('Bin Number')
        #     ax[1].set_ylabel('Normalized Brightness')
        #     f.suptitle('Kepler ID: {} | TCE PLNT: {}\n '
        #                'Period=  Duration=  Epoch='.format(data_dict['kepid'][int(idx)],
        #                                                    data_dict['tce_n'][int(idx)],
        #                                                    # data_dict['tce_period'][int(idx)],
        #                                                    # data_dict['tce_duration'][int(idx)],
        #                                                    # data_dict['epoch'][int(idx)]
        #                                                    ))
        #     f.savefig('{}{}_kepid{}_tcen{}.png'.format(pathres, plotname, data_dict['kepid'][int(idx)],
        #                                                data_dict['tce_n'][int(idx)]))
        #     # aa
        #     plt.close()