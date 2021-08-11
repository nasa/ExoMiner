#%% Checking normalized data in a CV iteration

# TFRecord directory
tfrecDir = Path('/data5/tess_project/Data/tfrecords/Kepler/Q1-Q17_DR25/cv/cv_01-26-2021_16-33/test_cv_kepler/cv_iter_0/norm_data/')
# tfrecDir = Path('/data5/tess_project/Data/tfrecords/Kepler/Q1-Q17_DR25/cv/cv_01-26-2021_16-33/tfrecords/')

# get filepaths to TFRecord files
# tfrecFiles = [file for file in tfrecDir.iterdir() if 'shard' in file.stem]
tfrecFiles = [tfrecDir / 'shard-0001']

tceIdentifier = 'tce_plnt_num'

# set views to be plotted
views = [
    'global_flux_view',
    'local_flux_view',
    'global_flux_view_fluxnorm',
    'local_flux_view_fluxnorm',
    # 'global_flux_odd_view',
    'local_flux_odd_view',
    'local_flux_odd_view_fluxnorm',
    # 'global_flux_even_view',
    'local_flux_even_view',
    'local_flux_even_view_fluxnorm',
    # 'local_flux_oddeven_view_diff',
    # 'local_flux_oddeven_view_diff_dir',
    # 'global_weak_secondary_view',
    # 'local_weak_secondary_view',
    # 'local_weak_secondary_view_selfnorm',
    # 'local_weak_secondary_view_fluxnorm',
    'local_weak_secondary_view_max_flux-wks_norm',
    # centroid
    'global_centr_view',
    'local_centr_view',
    # 'global_centr_view_std_clip',
    # 'local_centr_view_std_clip',
    'global_centr_view_std_noclip',
    'local_centr_view_std_noclip',
    # 'global_centr_view_medind_std',
    # 'local_centr_view_medind_std',
    # 'global_centr_view_medcmaxn',
    # 'local_centr_view_medcmaxn',
    # 'global_centr_view_medcmaxn_dir',
    # 'local_centr_view_medcmaxn_dir',
    # 'global_centr_view_medn',
    # 'local_centr_view_medn',
    'global_centr_fdl_view',
    'local_centr_fdl_view',
    'global_centr_fdl_view_norm',
    'local_centr_fdl_view_norm',
]

# set scalar parameter values to be extracted
scalarParams = [
    # stellar parameters
    'tce_steff',
    'tce_slogg',
    'tce_smet',
    'tce_sradius',
    'tce_smass',
    'tce_sdens',
    # secondary
    'wst_depth',
    'tce_maxmes',
    'tce_albedo_stat',
    # 'tce_albedo',
    'tce_ptemp_stat',
    # 'tce_ptemp',
    # 'wst_robstat',
    # odd-even
    # 'tce_bin_oedp_stat',
    # other parameters
    'boot_fap',
    'tce_cap_stat',
    'tce_hap_stat',
    'tce_rb_tcount0',
    # centroid
    'tce_fwm_stat',
    'tce_dikco_msky',
    'tce_dikco_msky_err',
    'tce_dicco_msky',
    'tce_dicco_msky_err',
    # flux
    # 'tce_max_mult_ev',
    # 'tce_depth_err',
    # 'tce_duration_err',
    # 'tce_period_err',
    'transit_depth',
    'tce_prad',
    'tce_period'
]

# set this to get the normalized scalar parameters
tceOfInterest = '465.01'  # '9773869-1'  # (6500206, 2)  # (9773869, 1) AFP with clear in-transit shift, (8937762, 1) nice PC, (8750094, 1)  # (8611832, 1)
scheme = (3, 6)
basename = 'all_views'  # basename for figures
# probPlot = 1.01  # probability threshold for plotting
count_i = 0
count_label = {label: 0 for label in ['NTP', 'PC', 'AFP']}
for tfrecFile in tfrecFiles:
    tfrecord_dataset = tf.data.TFRecordDataset(str(tfrecFile))

    for string_record in tfrecord_dataset.as_numpy_iterator():

        # if np.random.random() > probPlot:
        #     continue

        example = tf.train.Example()
        example.ParseFromString(string_record)

        targetIdTfrec = example.features.feature['target_id'].int64_list.value[0]

        if tceIdentifier == 'oi':
            tceIdentifierTfrec = example.features.feature[tceIdentifier].float_list.value[0]
            tceid = str(tceIdentifierTfrec).split('.')
            tceid = f'{tceid[0]}.{tceid[1][:2]}'
        else:
            tceIdentifierTfrec = example.features.feature[tceIdentifier].int64_list.value[0]
            tceid = f'{targetIdTfrec}-{tceIdentifierTfrec}'

        count_i += 1
        # if tceid != tceOfInterest:
        #     continue


        # get label
        labelTfrec = example.features.feature['label'].bytes_list.value[0].decode("utf-8")

        print(f'TCE {tceid} {labelTfrec}')
        count_label[labelTfrec] += 1

        if labelTfrec != 'PC':
            continue

        # get scalar parameters
        scalarParamsStr = ''
        for scalarParam_i in range(len(scalarParams)):
            scalar_param_val_norm = example.features.feature[f'{scalarParams[scalarParam_i]}_norm'].float_list.value[0]
            if scalarParams[scalarParam_i] in ['tce_steff', 'tce_rb_tcount0']:
                scalar_param_val = example.features.feature[scalarParams[scalarParam_i]].int64_list.value[0]
            else:
                scalar_param_val = example.features.feature[scalarParams[scalarParam_i]].float_list.value[0]

            if scalarParam_i % 6 == 0 and scalarParam_i != 0:
                scalarParamsStr += '\n'
            if scalarParams[scalarParam_i] in ['boot_fap']:
                scalarParamsStr += f'{scalarParams[scalarParam_i]}=' \
                                   f'{scalar_param_val_norm:.4f} ({scalar_param_val:.4E})|'
            elif scalarParams[scalarParam_i] in ['tce_steff', 'tce_rb_tcount0']:
                scalarParamsStr += f'{scalarParams[scalarParam_i]}=' \
                                   f'{scalar_param_val_norm:.4f} ({scalar_param_val})|'
            else:
                scalarParamsStr += f'{scalarParams[scalarParam_i]}=' \
                                   f'{scalar_param_val_norm:.4f} ({scalar_param_val:.4f})|'

        # get time series views
        viewsDict = {}
        for view in views:
            viewsDict[view] = np.array(example.features.feature[view].float_list.value)

        # plot features
        plot_features_example(viewsDict, scalarParamsStr, tceid, labelTfrec, tfrecDir, scheme, basename=basename,
                              display=True)
        aaa

print(count_i)
print(count_label)

#%% Testing tensorflow device selection

# import os
import tensorflow as tf
tf.debugging.set_log_device_placement(True)
from tensorflow.keras import losses, optimizers
from pathlib import Path
import multiprocessing
import time

# import src
from src.models_keras import CNN1dPlanetFinderv2
from src import config_keras
from src_hpo import utils_hpo
from src.utils_dataio import InputFnv2 as InputFn
import paths

# tf.compat.v1.disable_eager_execution()
# tf.config.run_functions_eagerly(False)
# import multiprocessing

# @tf.function
# def compute_in_tf():
#     tf.debugging.set_log_device_placement(True)
#     with tf.device('/job:localhost/replica:0/task:0/device:GPU:0'):
#         a = tf.constant([[1]])
#         b = tf.constant([[2]])
#         c = tf.matmul(a, b)
#         # print('aaaa')
#
# compute_in_tf()
# p = multiprocessing.Process(target=compute_in_tf)
# p.start()
# p.join()

# with tf.device('gpu:0'):
#     a = tf.constant([[1]])
#     b = tf.constant([[2]])
#     c = tf.matmul(a, b)
#
# sess =

config = {}

# name of the HPO study from which to get a configuration
hpo_study = 'ConfigK-bohb_keplerdr25-dv_g301-l31_spline_nongapped_starshuffle_norobovetterkois_glflux-glcentr_' \
            'std_noclip-loe-lwks-6stellar-bfap-ghost-rollingband-convscalars_loesubtract'
# set the configuration from an HPO study
if hpo_study is not None:
    hpo_path = Path(paths.path_hpoconfigs) / hpo_study
    res = utils_hpo.logged_results_to_HBS_result(hpo_path, f'_{hpo_study}')

    # get ID to config mapping
    id2config = res.get_id2config_mapping()
    # best config - incumbent
    incumbent = res.get_incumbent_id()
    config_id_hpo = incumbent
    config = id2config[config_id_hpo]['config']

config = config_keras.add_dataset_params(
    'kepler',
    False,
    False,
    {},
    config
)

# add missing parameters in hpo with default values
config = config_keras.add_default_missing_params(config=config)

config['branches'] = [
        'global_flux_view_fluxnorm',
        'local_flux_view_fluxnorm',
        # 'global_centr_fdl_view_norm',
        # 'local_centr_fdl_view_norm',
        'local_flux_oddeven_views',
        'global_centr_view_std_noclip',
        'local_centr_view_std_noclip',
        # 'local_weak_secondary_view_fluxnorm',
        # 'local_weak_secondary_view_selfnorm',
        'local_weak_secondary_view_max_flux-wks_norm'
    ]

features_set = {
        # flux related features
        'global_flux_view_fluxnorm': {'dim': (301, 1), 'dtype': tf.float32},
        'local_flux_view_fluxnorm': {'dim': (31, 1), 'dtype': tf.float32},
        'transit_depth_norm': {'dim': (1,), 'dtype': tf.float32},
        # odd-even views
        'local_flux_odd_view_fluxnorm': {'dim': (31, 1), 'dtype': tf.float32},
        'local_flux_even_view_fluxnorm': {'dim': (31, 1), 'dtype': tf.float32},
        # centroid views
        # 'global_centr_fdl_view_norm': {'dim': (301, 1), 'dtype': tf.float32},
        # 'local_centr_fdl_view_norm': {'dim': (31, 1), 'dtype': tf.float32},
        'global_centr_view_std_noclip': {'dim': (301, 1), 'dtype': tf.float32},
        'local_centr_view_std_noclip': {'dim': (31, 1), 'dtype': tf.float32},
        # secondary related features
        # 'local_weak_secondary_view_fluxnorm': {'dim': (31, 1), 'dtype': tf.float32},
        # 'local_weak_secondary_view_selfnorm': {'dim': (31, 1), 'dtype': tf.float32},
        'local_weak_secondary_view_max_flux-wks_norm': {'dim': (31, 1), 'dtype': tf.float32},
        'tce_maxmes_norm': {'dim': (1,), 'dtype': tf.float32},
        'wst_depth_norm': {'dim': (1,), 'dtype': tf.float32},
        # 'tce_albedo_norm': {'dim': (1,), 'dtype': tf.float32},
        'tce_albedo_stat_norm': {'dim': (1,), 'dtype': tf.float32},
        # 'tce_ptemp_norm': {'dim': (1,), 'dtype': tf.float32},
        'tce_ptemp_stat_norm': {'dim': (1,), 'dtype': tf.float32},
        # centroid related features
        'tce_fwm_stat_norm': {'dim': (1,), 'dtype': tf.float32},
        'tce_dikco_msky_norm': {'dim': (1,), 'dtype': tf.float32},
        'tce_dikco_msky_err_norm': {'dim': (1,), 'dtype': tf.float32},
        'tce_dicco_msky_norm': {'dim': (1,), 'dtype': tf.float32},
        'tce_dicco_msky_err_norm': {'dim': (1,), 'dtype': tf.float32},
        # other diagnostic parameters
        'boot_fap_norm': {'dim': (1,), 'dtype': tf.float32},
        'tce_cap_stat_norm': {'dim': (1,), 'dtype': tf.float32},
        'tce_hap_stat_norm': {'dim': (1,), 'dtype': tf.float32},
        'tce_rb_tcount0_norm': {'dim': (1,), 'dtype': tf.float32},
        # stellar parameters
        'tce_sdens_norm': {'dim': (1,), 'dtype': tf.float32},
        'tce_steff_norm': {'dim': (1,), 'dtype': tf.float32},
        'tce_smet_norm': {'dim': (1,), 'dtype': tf.float32},
        'tce_slogg_norm': {'dim': (1,), 'dtype': tf.float32},
        'tce_smass_norm': {'dim': (1,), 'dtype': tf.float32},
        'tce_sradius_norm': {'dim': (1,), 'dtype': tf.float32},
        # tce parameters
        'tce_prad_norm': {'dim': (1,), 'dtype': tf.float32},
        'tce_period_norm': {'dim': (1,), 'dtype': tf.float32},
    }

online_preproc_params = {'num_bins_global': 301, 'num_bins_local': 31, 'num_transit_dur': 6}

def train():
    # instantiate Keras model
    model = CNN1dPlanetFinderv2(config, features_set).kerasModel

    # setup metrics to be monitored
    metrics_list = ['acc']

    model.compile(optimizer=optimizers.SGD(learning_rate=config['lr'],
                                           momentum=0.1,
                                           nesterov=False,
                                           name='SGD'),  # optimizer
                  # loss function to minimize
                  loss=losses.BinaryCrossentropy(from_logits=False,
                                                 label_smoothing=0,
                                                 name='binary_crossentropy'),
                  # list of metrics to monitor
                  metrics=metrics_list)

    data_dir = '/data5/tess_project/Data/tfrecords/Kepler/Q1-Q17_DR25/tfrecordskeplerdr25-dv_g301-l31_6tr_spline_nongapped_flux-loe-lwks-centroid-centroidfdl-6stellar-bfap-ghost-rollband-stdts_secsymphase_correctprimarygapping_confirmedkoiperiod_data/tfrecordskeplerdr25-dv_g301-l31_6tr_spline_nongapped_flux-loe-lwks-centroid-centroidfdl-6stellar-bfap-ghost-rollband-stdts_secsymphase_correctprimarygapping_confirmedkoiperiod_starshuffle_experiment-labels-norm_nopps_secparams_prad_period'

    # input function for training, validation and test
    train_input_fn = InputFn(file_pattern=data_dir + '/train*',
                             batch_size=config['batch_size'],
                             mode=tf.estimator.ModeKeys.TRAIN,
                             label_map=config['label_map'],
                             data_augmentation=False,
                             online_preproc_params=online_preproc_params,
                             filter_data=None,
                             features_set=features_set)

    # fit the model to the training data
    history = model.fit(x=train_input_fn(),
                        epochs=1,
                        verbose=1,
                        )


with tf.device('/gpu:0'):
    train()
time.sleep(10)
with tf.device('/gpu:0'):
    print('2nd')
    train()
   # p = multiprocessing.Process(target=train)
   # p.start()
   # p.join()

#%% aggregate test set rankings in the CV experiment

cv_experiment_dir = Path('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Kepler_planet_finder/cv_experiments/cv_kepler_1-28-2021')

trainset_tbl = pd.read_csv('/data5/tess_project/Data/tfrecords/Kepler/Q1-Q17_DR25/train-val-test-sets/split_6-1-2020/trainset.csv')
trainset_tbl['UID'] = trainset_tbl[['target_id', 'tce_plnt_num']].apply(lambda x: '{}-{}'.format(x['target_id'],
                                                                                                 x['tce_plnt_num']),
                                                                        axis=1)
valset_tbl = pd.read_csv('/data5/tess_project/Data/tfrecords/Kepler/Q1-Q17_DR25/train-val-test-sets/split_6-1-2020/valset.csv')
valset_tbl['UID'] = valset_tbl[['target_id', 'tce_plnt_num']].apply(lambda x: '{}-{}'.format(x['target_id'],
                                                                                             x['tce_plnt_num']),
                                                                    axis=1)

cv_iter_dirs = [fp for fp in cv_experiment_dir.iterdir() if fp.stem.startswith('cv_iter_') and fp.is_dir()]

ranking_tbls_cv_iter = [{'iter': fp.stem.split('_')[-1],
                         'tbl': pd.read_csv(fp / 'ensemble_ranked_predictions_testset.csv')} for fp in cv_iter_dirs]

add_fields = ['iter', 'in_dataset']
ranking_tbl = []

for ranking_tbl_cv_iter_dict in ranking_tbls_cv_iter:

    ranking_tbl_cv_iter  = ranking_tbl_cv_iter_dict['tbl']

    ranking_tbl_cv_iter['in_dataset'] = '-'
    ranking_tbl_cv_iter['iter'] = ranking_tbl_cv_iter_dict['iter']

    ranking_tbl_cv_iter['UID'] = \
        ranking_tbl_cv_iter[['target_id', 'tce_plnt_num']].apply(lambda x: '{}-{}'.format(x['target_id'],
                                                                                          x['tce_plnt_num']),
                                                                 axis=1)

    ranking_tbl_cv_iter.loc[ranking_tbl_cv_iter['UID'].isin(trainset_tbl['UID']), 'in_dataset'] = 'train'
    ranking_tbl_cv_iter.loc[ranking_tbl_cv_iter['UID'].isin(valset_tbl['UID']), 'in_dataset'] = 'val'

    ranking_tbl_cv_iter.drop(columns='UID', inplace=True)

    ranking_tbl.append(ranking_tbl_cv_iter)

ranking_tbl = pd.concat(ranking_tbl, axis=0)

ranking_tbl.to_csv(cv_experiment_dir / 'ensemble_ranked_predictions_testsets_agg.csv', index=False)

#%% count number of PCs for computing k max

cv_iter_dirs = [fp for fp in cv_experiment_dir.iterdir() if fp.stem.startswith('cv_iter_') and fp.is_dir()]

ranking_tbls_cv_iter = [(fp.stem.split('_')[-1], pd.read_csv(fp / 'ensemble_ranked_predictions_trainset.csv'))
                        for fp in cv_iter_dirs]

count_pcs = [len(ranking_tbl[1].loc[ranking_tbl[1]['original_label'] == 'PC']) for ranking_tbl in ranking_tbls_cv_iter]

kmax = min(count_pcs)

print(f'Minimum number of PCs in fold: {kmax}')

precatmax_tbl = pd.DataFrame(columns=['cv_iter', 'precatmax'])

for ranking_tbl in ranking_tbls_cv_iter:
    precatmax_tbl = pd.concat([precatmax_tbl, pd.DataFrame(data={'cv_iter': [ranking_tbl[0]],
                                                                 'precatmax': [ranking_tbl[1][:kmax]['label'].sum() /
                                                                              kmax]})])

#%% aggregate performance metrics, compute mean and std,


def add_study_to_restbl(cv_iter, cv_results_fp, mapDatasetMetric):
    """ Add study results to table.

    :param studyName: str, name to give the study in the results study table.
    :param studyDir: str, name of the study directory.
    :param studyRootDir: str, filepath to root directory containing the study.
    :param mapDatasetMetric: dict, mapping between study result keys and column names in the result study table.
    :param resTblFp: str, filepath to results study table
    :return:
    """

    resTbl = pd.DataFrame(columns=['CV iteration'] + list(mapDatasetMetric.values()))

    resStudy = np.load(cv_results_fp, allow_pickle=True).item()

    dataDict = {}
    for metric in mapDatasetMetric:
        dataDict[mapDatasetMetric[metric]] = [resStudy[metric]]

    studyTbl = pd.DataFrame(data=dataDict)

    studyTbl.insert(0, 'CV iteration', cv_iter)

    resTbl = pd.concat([resTbl, studyTbl], axis=0)

    return resTbl


datasetsMetrics = np.load('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Kepler_planet_finder/results_ensemble/'
                          'map_datasets_metrics.npy', allow_pickle=True).item()

cv_iter_results = [(fp.stem.split('_')[-1], fp / 'results_ensemble.npy') for fp in cv_experiment_dir.iterdir()
                   if fp.stem.startswith('cv_iter_') and fp.is_dir()]

# select values of k for which precision at k was computed
# topk = {'train': [100, 1000, 2084], 'val': [50, 150, 257], 'test': [50, 150, 283]}
topk = {'train': [100, 1000, 1818], 'val': [50, 150, 222], 'test': [50, 150, 251]}  # No PPs

# if saveFP is not None, create a new metric mapping and save it
saveFp = None
# saveFp = '/home/msaragoc/Projects/Kepler-TESS_exoplanet/Kepler_planet_finder/results_ensemble/map_datasets_metrics.npy'
# datasetsMetrics = np.load('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Kepler_planet_finder/results_ensemble/'
#                           'map_datasets_metrics.npy', allow_pickle=True).item()

resTbl = []
for cv_iter, cv_result in cv_iter_results:

    datasetsMetrics = create_metric_mapping(topk, saveFp)

    resTbl.append(add_study_to_restbl(cv_iter, cv_result, datasetsMetrics))

resTbl = pd.concat(resTbl, axis=0)
resTbl.set_index(keys='CV iteration', inplace=True)
resTbl = pd.concat([resTbl,
                    pd.DataFrame([resTbl.mean(),
                                  resTbl.std(ddof=1)],
                                 index=['mean', 'std']).rename_axis('CV iteration')],
                   axis=0)
resTbl.reset_index(inplace=True)

resTbl.to_csv(cv_experiment_dir / f'results_studies_{datetime.now().date()}.csv', index=False)

#%% compute precision at k

def compute_prcurve_roc_auc(study_dir, dataset, num_thresholds, basename, plot=False, verbose=False):
    """ Compute PR curve and ROC (also AUC).

    :param study_dir: Path, study directory
    :param dataset: str, dataset
    :param num_thresholds: int, number of thresholds to be used between 0 and 1
    :param basename: str, name to be added to the figure name
    :param plot: bool, if True plots figures
    :param verbose: bool, if True prints data into std out.
    :return:
    """

    save_dir = study_dir / basename
    save_dir.mkdir(exist_ok=True)

    dataset_tbl = pd.read_csv(study_dir / f'ensemble_ranked_predictions_{dataset}set.csv')

    threshold_range = list(np.linspace(0, 1, num=num_thresholds, endpoint=False))

    # compute metrics
    auc_pr = AUC(num_thresholds=num_thresholds,
                 summation_method='interpolation',
                 curve='PR',
                 name='auc_pr')
    auc_roc = AUC(num_thresholds=num_thresholds,
                  summation_method='interpolation',
                  curve='ROC',
                  name='auc_roc')

    _ = auc_pr.update_state(dataset_tbl['label'].tolist(), dataset_tbl['score'].tolist())
    auc_pr = auc_pr.result().numpy()
    _ = auc_roc.update_state(dataset_tbl['label'].tolist(), dataset_tbl['score'].tolist())
    auc_roc = auc_roc.result().numpy()

    # compute precision and recall thresholds for the PR curve
    precision_thr = Precision(thresholds=threshold_range, top_k=None, name='prec_thr')
    recall_thr = Recall(thresholds=threshold_range, top_k=None, name='rec_thr')
    _ = precision_thr.update_state(dataset_tbl['label'].tolist(), dataset_tbl['score'].tolist())
    precision_thr_arr = precision_thr.result().numpy()
    _ = recall_thr.update_state(dataset_tbl['label'].tolist(), dataset_tbl['score'].tolist())
    recall_thr_arr = recall_thr.result().numpy()

    # compute FPR for the ROC
    false_pos_thr = FalsePositives(thresholds=threshold_range, name='prec_thr')
    _ = false_pos_thr.update_state(dataset_tbl['label'].tolist(), dataset_tbl['score'].tolist())
    false_pos_thr_arr = false_pos_thr.result().numpy()
    fpr_thr_arr = false_pos_thr_arr / len(dataset_tbl.loc[dataset_tbl['label'] == 0])

    metrics = {
        'auc_roc': auc_roc,
        'auc_pr': auc_pr,
        'prec_thr': precision_thr_arr,
        'rec_thr': recall_thr_arr,
        'fpr_thr': fpr_thr_arr
    }

    if verbose:
        print(f'Dataset {dataset}: {metrics}')

    np.save(save_dir / f'metrics_{dataset}.npy', metrics)

    if plot:
        # plot PR curve
        f, ax = plt.subplots(figsize=(10, 8))
        ax.plot(recall_thr_arr, precision_thr_arr)
        # ax.scatter(recall_thr_arr, precision_thr_arr, c='r')
        ax.set_ylabel('Precision')
        ax.set_xlabel('Recall')
        ax.grid(True)
        ax.set_xticks(np.linspace(0, 1, 11))
        ax.set_yticks(np.linspace(0, 1, 11))
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.text(0.8, 0.1, f'AUC={auc_pr:.3}', bbox={'facecolor': 'gray', 'alpha': 0.2, 'pad': 10})
        f.savefig(save_dir / f'precision-recall_curve_{dataset}.svg')
        plt.close()

        # plot PR curve zoomed in
        f, ax = plt.subplots(figsize=(10, 8))
        ax.plot(recall_thr_arr, precision_thr_arr)
        # ax.scatter(recall_thr_arr, precision_thr_arr, c='r')
        ax.set_ylabel('Precision')
        ax.set_xlabel('Recall')
        ax.grid(True)
        ax.set_xticks(np.linspace(0, 1, 21))
        ax.set_yticks(np.linspace(0, 1, 21))
        ax.set_xlim([0.5, 1])
        ax.set_ylim([0.5, 1])
        ax.text(0.8, 0.6, f'AUC={auc_pr:.3}', bbox={'facecolor': 'gray', 'alpha': 0.2, 'pad': 10})
        f.savefig(save_dir / f'precision-recall_curve_zoom_{dataset}.svg')
        plt.close()

        # plot ROC
        f, ax = plt.subplots(figsize=(10, 8))
        ax.plot(fpr_thr_arr, recall_thr_arr)
        ax.set_ylabel('True Positive Rate')
        ax.set_xlabel('False Positive Rate')
        ax.grid(True)
        ax.set_xticks(np.linspace(0, 1, 11))
        ax.set_yticks(np.linspace(0, 1, 11))
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.text(0.8, 0.1, 'AUC={:.3}'.format(auc_roc), bbox={'facecolor': 'gray', 'alpha': 0.2, 'pad': 10})
        f.savefig(save_dir / f'roc_{dataset}.svg')
        plt.close()

        # plot ROC zoomed in
        f, ax = plt.subplots(figsize=(10, 8))
        ax.plot(fpr_thr_arr, recall_thr_arr)
        ax.set_ylabel('True Positive Rate')
        ax.set_xlabel('False Positive Rate')
        ax.grid(True)
        ax.set_xticks(np.linspace(0, 1, 21))
        ax.set_yticks(np.linspace(0, 1, 21))
        ax.set_xlim([0, 0.5])
        ax.set_ylim([0.5, 1])
        ax.text(0.3, 0.6, 'AUC={:.3}'.format(auc_roc), bbox={'facecolor': 'gray', 'alpha': 0.2, 'pad': 10})
        f.savefig(save_dir / f'roc_zoom_{dataset}.svg')
        plt.close()


def compute_precision_at_k(study, dataset, rootDir, k_arr, k_curve_arr, k_curve_arr_plot, basename, plot=False,
                           verbose=False):
    """ Compute precision-at-k and plot related curves.

    :param study: str, study name
    :param dataset: str, dataset. Either 'train', 'val' and 'test'
    :param rootDir: Path, root directory with the studies
    :param k_arr: list with k values for which to compute precision-at-k
    :param k_curve_arr: list with k values for which to compute precision-at-k curve
    :param k_curve_arr_plot: list with values for which to draw xticks (k values)
    :param plot: bool, if True plots precision-at-k and misclassified-at-k curves
    :param verbose: bool, if True print precision-at-k values
    :return:
    """

    save_dir = rootDir / study / basename
    save_dir.mkdir(exist_ok=True)

    rankingTbl = pd.read_csv(rootDir / study / f'ensemble_ranked_predictions_{dataset}set.csv')

    # order by ascending score
    rankingTblOrd = rankingTbl.sort_values('score', axis=0, ascending=True)

    # compute precision at k
    precision_at_k = {k: np.nan for k in k_arr}
    for k_i in range(len(k_arr)):
        if len(rankingTblOrd) < k_arr[k_i]:
            precision_at_k[k_arr[k_i]] = np.nan
        else:
            precision_at_k[k_arr[k_i]] = \
                np.sum(rankingTblOrd['label'][-k_arr[k_i]:]) / k_arr[k_i]

    np.save(save_dir / f'precision_at_k_{dataset}.npy', precision_at_k)

    if verbose:
        print(f'Dataset {dataset}: {precision_at_k}')

    # compute precision at k curve
    precision_at_k = {k: np.nan for k in k_curve_arr}
    for k_i in range(len(k_curve_arr)):
        if len(rankingTblOrd) < k_curve_arr[k_i]:
            precision_at_k[k_curve_arr[k_i]] = np.nan
        else:
            precision_at_k[k_curve_arr[k_i]] = \
                np.sum(rankingTblOrd['label'][-k_curve_arr[k_i]:]) / k_curve_arr[k_i]

    np.save(save_dir / f'precision_at_k_curve_{dataset}.npy', precision_at_k)

    if plot:
        # precision at k curve
        f, ax = plt.subplots(figsize=(10, 8))
        ax.plot(list(precision_at_k.keys()), list(precision_at_k.values()))
        ax.set_ylabel('Precision')
        ax.set_xlabel('Top-K')
        ax.grid(True)
        ax.set_xticks(k_curve_arr_plot)
        ax.set_xlim([k_curve_arr[0], k_curve_arr[-1]])
        # ax.set_ylim(top=1)
        ax.set_ylim([-0.01, 1.01])
        f.savefig(save_dir / f'precision_at_k_{dataset}.svg')
        plt.close()

        # misclassified examples at k curve
        f, ax = plt.subplots(figsize=(10, 8))
        kvalues = np.array(list(precision_at_k.keys()))
        precvalues = np.array(list(precision_at_k.values()))
        num_misclf_examples = kvalues - kvalues * precvalues
        ax.plot(kvalues, num_misclf_examples)
        ax.set_ylabel('Number Misclassfied TCEs')
        ax.set_xlabel('Top-K')
        ax.grid(True)
        ax.set_xticks(k_curve_arr_plot)
        ax.set_xlim([k_curve_arr[0], k_curve_arr[-1]])
        ax.set_ylim(bottom=-0.01)
        f.savefig(save_dir / f'misclassified_at_k_{dataset}.svg')
        plt.close()

datasets = ['train', 'val', 'test']

k_arr = {'train': [100, 1000, 1677], 'val': [50, 150, 227], 'test': [50, 150, 177]}

k_curve_arr = {
    'train': np.linspace(100, 1600, 16, endpoint=True, dtype='int'),
    'val': np.linspace(10, 220, 22, endpoint=True, dtype='int'),
    'test': np.linspace(10, 170, 17, endpoint=True, dtype='int')
}
k_curve_arr_plot = {
    'train': np.linspace(100, 1600, 16, endpoint=True, dtype='int'),
    'val': np.linspace(20, 220, 11, endpoint=True, dtype='int'),
    'test': np.linspace(20, 170, 16, endpoint=True, dtype='int')
}

num_thresholds = 1000

rootDir = Path('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Kepler_planet_finder/cv_experiments/cv_kepler_1-28-2021')
cv_iter_dirs = [fp.stem for fp in cv_experiment_dir.iterdir() if fp.stem.startswith('cv_iter_') and fp.is_dir()]

for cv_iter_dir in cv_iter_dirs:
    print(f'Running for study {cv_iter_dir}')
    for dataset in datasets:
        compute_precision_at_k(cv_iter_dir, dataset, rootDir, k_arr[dataset], k_curve_arr[dataset],
                               k_curve_arr_plot[dataset],
                               f'thr{num_thresholds}', plot=True, verbose=True)
        compute_prcurve_roc_auc(rootDir / cv_iter_dir, dataset, num_thresholds, f'thr{num_thresholds}', plot=True)

#%% generate ROC, PR curve, precision at k curve, misclassified at k curve with mean and mean +- std

metrics_list = ['prec_thr', 'rec_thr', 'fpr_thr']
precatk_cv = {dataset: {'k': k_curve_arr[dataset], 'precision': {'iters': [], 'mean': [], 'std': []}}
              for dataset in datasets}
misclfatk_cv = {dataset: {'k': k_curve_arr[dataset], 'num_misclf': {'iters': [], 'mean': [], 'std': []}}
                for dataset in datasets}
metrics_cv = {dataset: {metric: {'iters': [], 'mean': [], 'std': []} for metric in metrics_list}
              for dataset in datasets}
thr_dir = 'thr1000'

for dataset in datasets:
    for cv_iter_dir in cv_iter_dirs:
        metrics_aux = np.load(rootDir / cv_iter_dir / thr_dir / f'metrics_{dataset}.npy', allow_pickle=True).item()
        for metric in metrics_list:
            metrics_cv[dataset][metric]['iters'].append(metrics_aux[metric])

        precatk_aux = np.load(rootDir / cv_iter_dir / thr_dir / f'precision_at_k_curve_{dataset}.npy',
                              allow_pickle=True).item()
        precatk_cv[dataset]['precision']['iters'].append(list(precatk_aux.values()))
        misclfatk_cv[dataset]['num_misclf']['iters'].append(precatk_cv[dataset]['k'] - precatk_cv[dataset]['k'] *
                                                            precatk_cv[dataset]['precision']['iters'][-1])

    for metric in metrics_list:
        metrics_cv[dataset][metric]['mean'] = np.mean(metrics_cv[dataset][metric]['iters'], axis=0)
        metrics_cv[dataset][metric]['std'] = np.std(metrics_cv[dataset][metric]['iters'], axis=0, ddof=1)

    precatk_cv[dataset]['precision']['mean'] = np.mean(precatk_cv[dataset]['precision']['iters'], axis=0)
    precatk_cv[dataset]['precision']['std'] = np.std(precatk_cv[dataset]['precision']['iters'], axis=0, ddof=1)

    misclfatk_cv[dataset]['num_misclf']['mean'] = np.mean(misclfatk_cv[dataset]['num_misclf']['iters'], axis=0)
    misclfatk_cv[dataset]['num_misclf']['std'] = np.std(misclfatk_cv[dataset]['num_misclf']['iters'], axis=0, ddof=1)

# plot precision at k curve
for dataset in datasets:

    metrics_study = np.load(f'/home/msaragoc/Projects/Kepler-TESS_exoplanet/Kepler_planet_finder/results_ensemble/keplerdr25-dv_g301-l31_6tr_spline_nongapped_norobovetterkois_starshuffle_configK_secsymphase_wksnormmaxflux-wks_corrprimgap_ptempstat_albedostat_wstdepth_fwmstat_nopps_ckoiper_secparams_prad_per/thr1000/metrics_{dataset}.npy', allow_pickle=True).item()
    precatk_study = np.load(
        f'/home/msaragoc/Projects/Kepler-TESS_exoplanet/Kepler_planet_finder/results_ensemble/keplerdr25-dv_g301-l31_6tr_spline_nongapped_norobovetterkois_starshuffle_configK_secsymphase_wksnormmaxflux-wks_corrprimgap_ptempstat_albedostat_wstdepth_fwmstat_nopps_ckoiper_secparams_prad_per/thr1000/precision_at_k_curve_{dataset}.npy',
        allow_pickle=True).item()

    f, ax = plt.subplots(figsize=(10, 8))
    ax.plot(precatk_cv[dataset]['k'], precatk_cv[dataset]['precision']['mean'], 'r')
    ax.plot(precatk_cv[dataset]['k'],
            precatk_cv[dataset]['precision']['mean'] + precatk_cv[dataset]['precision']['std'], 'r--')
    ax.plot(precatk_cv[dataset]['k'],
            precatk_cv[dataset]['precision']['mean'] - precatk_cv[dataset]['precision']['mean'], 'r--')
    for cv_iter_i in range(len(precatk_cv[dataset]['precision']['iters'])):
        ax.plot(precatk_cv[dataset]['k'], precatk_cv[dataset]['precision']['iters'][cv_iter_i], 'b', alpha=0.3)
    ax.plot(list(precatk_study.keys()), list(precatk_study.values()), 'g')
    ax.set_ylabel('Precision')
    ax.set_xlabel('Top-K')
    ax.grid(True)
    ax.set_xticks(k_curve_arr_plot[dataset])
    ax.set_yticks(np.linspace(0.9, 1, 11))
    ax.set_xlim([k_curve_arr[dataset][0], k_curve_arr[dataset][-1]])
    ax.set_ylim(top=1.001, bottom=0.9)
    # ax.legend(loc=3, fontsize=7)
    f.savefig(rootDir / f'precision_at_k_{dataset}.svg')
    plt.close()

    f, ax = plt.subplots(figsize=(10, 8))
    ax.plot(misclfatk_cv[dataset]['k'], misclfatk_cv[dataset]['num_misclf']['mean'], 'r')
    ax.plot(misclfatk_cv[dataset]['k'],
            misclfatk_cv[dataset]['num_misclf']['mean'] + misclfatk_cv[dataset]['num_misclf']['std'], 'r--')
    ax.plot(misclfatk_cv[dataset]['k'],
            misclfatk_cv[dataset]['num_misclf']['mean'] - misclfatk_cv[dataset]['num_misclf']['mean'], 'r--')
    for cv_iter_i in range(len(misclfatk_cv[dataset]['num_misclf']['iters'])):
        ax.plot(misclfatk_cv[dataset]['k'], misclfatk_cv[dataset]['num_misclf']['iters'][cv_iter_i], 'b', alpha=0.3)
    misclfatk_study = np.array(list(precatk_study.keys())) - np.array(list(precatk_study.keys())) * \
                      np.array(list(precatk_study.values()))
    ax.plot(list(precatk_study.keys()), misclfatk_study, 'g')
    ax.set_ylabel('Number Misclassfied TCEs')
    ax.set_xlabel('Top-K')
    ax.grid(True)
    ax.set_xticks(k_curve_arr_plot[dataset])
    ax.set_xlim([k_curve_arr_plot[dataset][0], k_curve_arr_plot[dataset][-1]])
    # ax.legend(loc=2, fontsize=7)
    f.savefig(rootDir / f'misclassified_at_k_{dataset}.svg')
    plt.close()

# plot PR curve
for dataset in datasets:

    f, ax = plt.subplots(figsize=(10, 8))
    ax.plot(metrics_cv[dataset]['rec_thr']['mean'], metrics_cv[dataset]['prec_thr']['mean'], 'r')
    ax.plot(metrics_cv[dataset]['rec_thr']['mean'] + metrics_cv[dataset]['rec_thr']['std'],
            metrics_cv[dataset]['prec_thr']['mean'] + metrics_cv[dataset]['prec_thr']['std'], 'r--')
    ax.plot(metrics_cv[dataset]['rec_thr']['mean'] - metrics_cv[dataset]['rec_thr']['std'],
            metrics_cv[dataset]['prec_thr']['mean'] - metrics_cv[dataset]['prec_thr']['std'], 'r--')
    for cv_iter_i in range(len(metrics_cv[dataset]['fpr_thr']['iters'])):
        ax.plot(metrics_cv[dataset]['rec_thr']['iters'][cv_iter_i],
                metrics_cv[dataset]['prec_thr']['iters'][cv_iter_i], 'b', alpha=0.3)
    ax.plot(metrics_study['rec_thr'], metrics_study['prec_thr'], 'g')
    ax.set_ylabel('Precision')
    ax.set_xlabel('Recall')
    ax.grid(True)
    ax.set_xticks(np.linspace(0.5, 1, 11, endpoint=True))
    ax.set_yticks(np.linspace(0.5, 1, 11, endpoint=True))
    ax.set_xlim([0.5, 1])
    ax.set_ylim([0.5, 1])
    # ax.legend(loc=3, fontsize=7)
    f.savefig(rootDir / f'precision_recall_curve_{dataset}.svg')
    ax.set_xlim([0.85, 1])
    ax.set_ylim([0.7, 1])
    f.savefig(rootDir / f'precision_recall_curve_{dataset}_zoomin.svg')
    plt.close()

# plot ROC
for dataset in datasets:

    f, ax = plt.subplots(figsize=(10, 8))

    ax.plot(metrics_cv[dataset]['fpr_thr']['mean'], metrics_cv[dataset]['rec_thr']['mean'], 'r')
    ax.plot(metrics_cv[dataset]['fpr_thr']['mean'] + metrics_cv[dataset]['fpr_thr']['std'],
            metrics_cv[dataset]['rec_thr']['mean'] + metrics_cv[dataset]['rec_thr']['std'], 'r--')
    ax.plot(metrics_cv[dataset]['fpr_thr']['mean'] - metrics_cv[dataset]['fpr_thr']['std'],
            metrics_cv[dataset]['rec_thr']['mean'] -  metrics_cv[dataset]['rec_thr']['std'], 'r--')
    for cv_iter_i in range(len(metrics_cv[dataset]['fpr_thr']['iters'])):
        ax.plot(metrics_cv[dataset]['fpr_thr']['iters'][cv_iter_i],
                metrics_cv[dataset]['rec_thr']['iters'][cv_iter_i], 'b', alpha=0.3)
    ax.plot(metrics_study['fpr_thr'], metrics_study['rec_thr'], 'g')
    ax.set_ylabel('True Positive Rate')
    ax.set_xlabel('False Positive Rate')
    ax.grid(True)
    ax.set_xticks(np.linspace(0, 0.5, 11, endpoint=True))
    ax.set_yticks(np.linspace(0.75, 1, 11, endpoint=True))
    ax.set_xlim([0, 0.5])
    ax.set_ylim([0.75, 1])
    # ax.legend(loc=4, fontsize=7)
    f.savefig(rootDir / f'roc_{dataset}.svg')
    ax.set_xlim([0, 0.5])
    ax.set_ylim([0.95, 1])
    f.savefig(rootDir / f'roc_{dataset}_zoomin.svg')
    plt.close()
