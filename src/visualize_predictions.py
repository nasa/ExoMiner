# 3rd party
import os
import numpy as np
import matplotlib.pyplot as plt

# local
import paths
import src.config
from src.estimator_util import get_data_from_tfrecords

#%%

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

import pandas as pd

src = ['spline', 'whitened']
kepids_data = {data_src: {dataset: None for dataset in ['train', 'val', 'test']} for data_src in src}

tfrec_dirs = {src[0]: '/home/msaragoc/Projects/Kepler-TESS_exoplanet/Data/tfrecord_dr25_manual_2dkeplernonwhitened_2001-201',
              src[1]: '/home/msaragoc/Projects/Kepler-TESS_exoplanet/Data/tfrecord_dr25_manual_2dkeplerwhitened_2001-201'}

data_fields = ['kepid', 'tce_n']
for tfrec_dir in tfrec_dirs:
    for dataset in ['train', 'val', 'test']:
        tfrec_filenames = [os.path.join(tfrec_dirs[tfrec_dir], file)
                           for file in os.listdir(tfrec_dirs[tfrec_dir]) if dataset in file]

        data = get_data_from_tfrecords(tfrec_filenames, data_fields, label_map=label_map, filt=None)
        print(tfrec_dir, dataset, len(data['kepid']), len(data['tce_n']))
        # get only the first TCEs
        # idxs_tce1 = np.where(np.array(data['tce_n'], dtype='uint64') == 1)
        # kepids_data[tfrec_dir][dataset] = np.array(data['kepid'], dtype='int')[idxs_tce1]
        # kepids_data[tfrec_dir][dataset] = np.array(data['kepid'], dtype='int')
        # kepids_data[tfrec_dir][dataset] = pd.DataFrame(data)
        kepids_data[tfrec_dir][dataset] = ['{}_{}'.format(data['kepid'][i], data['tce_n'][i])
                                           for i in range(len(data['kepid']))]

# a.to_csv('/home/msaragoc/Downloads/testpd', index=False)
# b = pd.read_csv('/home/msaragoc/Downloads/testpd')
# get the common ones
cmmn_ids = {dataset: {'kepid+tce_n': np.intersect1d(kepids_data[src[0]][dataset], kepids_data[src[1]][dataset])}
               for dataset in ['train', 'val', 'test']}

np.save('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Kepler_planet_finder/cmmn_kepids_spline-whitened', cmmn_ids)

# %%

pathres = '/home/msaragoc/Projects/Kepler-TESS_exoplanet/Kepler_planet_finder/visualize_lc_modelpred/' \
          'succclf_dv_vs_missclf_tps/'
tfrec_dirs = {'dv': '/data5/tess_project/Data/tfrecords/dr25_koilabels/tfrecord_vanilla',
              'tps': '/data5/tess_project/Data/tfrecords/dr25_koilabels/tfrecord_vanilla_tps'}
satellite = 'kepler'
multi_class = False
cmmn_kepids = np.load('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Kepler_planet_finder/cmmn_kepids.npy').item()
data_fields = ['kepid', 'global_view', 'local_view', 'labels']  # 'epoch', 'tce_duration', 'tce_period', 'MES', 'global_view_centr', 'local_view_centr']
thr_clf = 0.5
study = 'study_bohb_dr25_tcert_spline2'

label_map = src.config.label_map[satellite][multi_class]
kepids_data = {ephemeris_src: {dataset: [] for dataset in ['train', 'val', 'test']} for ephemeris_src in ['dv', 'tps']}
for dataset in ['train', 'val', 'test']:

    print('Dataset {}'.format(dataset))

    print('Thresholding predictions and getting the indexes of interest...')
    predictions = {'dv': np.load(paths.pathsaveres_get_pcprobs + study +
                                 '/dv_ephemeris/predictions_per_dataset.npy').item()[dataset],
                   'tps': np.load(paths.pathsaveres_get_pcprobs + study +
                                  '/tps_ephemeris/predictions_per_dataset.npy').item()[dataset]}

    predictions_clf = {'dv': np.zeros(predictions['dv'].shape, dtype='uint8'),
                       'tps': np.zeros(predictions['dv'].shape, dtype='uint8')}

    for tfrec_dir in tfrec_dirs:
        tfrec_filenames = [os.path.join(tfrec_dirs[tfrec_dir], file)
                           for file in os.listdir(tfrec_dirs[tfrec_dir]) if dataset in file]

        # get labels and respective Kepler ids
        data_dict = get_data_from_tfrecords(tfrec_filenames, ['kepid', 'labels'], label_map=label_map,
                                           filt_by_kepids=cmmn_kepids[dataset])

        kepids_data[tfrec_dir][dataset] = np.array(data_dict['kepid'], dtype='uint64')
        # print(tfrec_dir, dataset, len(cmmn_kepids[dataset]), len(data_dict['kepid']), data_dict['kepid'][:5])

        # threshold the predictions
        predictions_clf[tfrec_dir][np.where(predictions[tfrec_dir] >= thr_clf)] = 1

        # find the examples of interest on the two datasets
        if 'tps' == tfrec_dir:  # missclassified examples in the TPS dataset
            # idxs_tps_miss = np.nonzero(predictions_clf[tfrec_dir] - np.array(data_dict['labels'], dtype='uint8'))
            idxs_tps_miss = np.where((predictions_clf[tfrec_dir] - np.array(data_dict['labels'], dtype='uint8')) == 0)
        elif 'dv' == tfrec_dir:  # correctly classified examples in the DV dataset
            # idxs_dv_correct = np.where((predictions_clf[tfrec_dir] - np.array(data_dict['labels'], dtype='uint8')) == 0)
            idxs_dv_correct = np.nonzero(predictions_clf[tfrec_dir] - np.array(data_dict['labels'], dtype='uint8'))

        del data_dict

    # find mapping between examples in the two datasets
    print('Finding the common indexes...')
    map_kepids = np.zeros(len(kepids_data['dv'][dataset]), dtype='uint64')
    for i, kepid in enumerate(kepids_data['dv'][dataset]):
        # entry i in DV dataset corresponds to entry map_idx in TPS dataset
        map_idx = np.where(kepids_data['tps'][dataset] == kepid)[0]
        assert len(map_idx) != 0
        map_kepids[i] = map_idx

    # find the indexes of interest after DV indexes are mapped to TPS indexes
    commtps = np.intersect1d(map_kepids[idxs_dv_correct], idxs_tps_miss)
    commdv = np.array([np.where(map_kepids == idx)[0][0] for idx in commtps], dtype='uint64')
    commtps = np.array(commtps, dtype='uint64')

    print('Number of TCEs of interest: {}'.format(len(commtps)))

    # extract data pertaining to the examples of interest
    tfrec_filenames = [os.path.join(tfrec_dirs['tps'], file)
                       for file in os.listdir(tfrec_dirs['tps']) if dataset in file]
    data_tps = get_data_from_tfrecords(tfrec_filenames, data_fields, label_map=label_map,
                                       filt_by_kepids=cmmn_kepids[dataset])

    tfrec_filenames = [os.path.join(tfrec_dirs['dv'], file)
                       for file in os.listdir(tfrec_dirs['dv']) if dataset in file]
    data_dv = get_data_from_tfrecords(tfrec_filenames, data_fields, label_map=label_map,
                                      filt_by_kepids=cmmn_kepids[dataset])

    for i in range(len(commtps)):
        f, ax = plt.subplots(2, 2, figsize=(10, 7))
        ax[0, 0].plot(data_tps['global_view'][commtps[i]])
        ax[0, 0].set_title('{}'.format('TPS\nglobal_view'))
        ax[0, 0].set_ylabel('Normalized Brightness')
        ax[1, 0].plot(data_tps['local_view'][commtps[i]])
        ax[1, 0].set_title('{}'.format('local_view'))
        ax[1, 0].set_xlabel('Bin Number')
        ax[1, 0].set_ylabel('Normalized Brightness')
        ax[0, 1].plot(data_dv['global_view'][commdv[i]])
        ax[0, 1].set_title('{}'.format('DV\nglobal_view'))
        # ax[0, 1].set_ylabel('Normalized Brightness')
        ax[1, 1].plot(data_dv['local_view'][commdv[i]])
        ax[1, 1].set_title('{}'.format('local_view'))
        ax[1, 1].set_xlabel('Bin Number')
        # ax[1, 1].set_ylabel('Normalized Brightness')
        f.suptitle('Kepler ID: {} | Label: {}\nPrediction: {}({:.3f}) | {}({:.3f})'.format(data_dv['kepid'][commdv[i]],
                                                                                           data_dv['labels'][commdv[i]],
                                                                                           predictions_clf['tps'][commtps[i]],
                                                                                           predictions['tps'][commtps[i]],
                                                                                           predictions_clf['dv'][commdv[i]],
                                                                                           predictions['dv'][commdv[i]]))
        f.savefig('{}{}/tpsvdv_kepid{}.svg'.format(pathres, dataset, data_dv['kepid'][commdv[i]]))
        plt.close()
        # aaaa

    del data_tps, data_dv
