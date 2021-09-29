import os
import tensorflow as tf


from old.src_old.estimator_util import InputFn, ModelFn, CNN1dPlanetFinderv1
import paths
from src_hpo import utils_hpo

base_model = CNN1dPlanetFinderv1

tfrec_dir = '/home/msaragoc/Projects/Kepler-TESS_exoplanet/Data/tfrecords/Kepler/' \
            'tfrecordskeplerdr25_g2001-l201_spline_gapped_flux-centroid_selfnormalized-oddeven_updtkois_shuffled_' \
            'nonwhitened_gapped_2001-201'
res_dir = '/home/msaragoc/Downloads/'
model_dir = '/home/msaragoc/Projects/Kepler-TESS_exoplanet/Kepler_planet_finder/trained_models/' \
            'dr25tcert_spline_gapped_glflux-glcentr-loe-6stellar_glfluxconfig_updtKOIs_shuffled/models/' \
            'tmpwgmpp_w0_test'

n_epochs = 100

hpo_study = 'bohb_dr25tcert_spline_gapped_glflux'
res_hpo = utils_hpo.logged_results_to_HBS_result(os.path.join(paths.path_hpoconfigs, hpo_study)
                                             , '_{}'.format(hpo_study))
# get ID to config mapping
id2config = res_hpo.get_id2config_mapping()
# best config - incumbent
incumbent = res_hpo.get_incumbent_id()
config = id2config[incumbent]['config']

multi_class = False  # multiclass classification
ce_weights_args = {'tfrec_dir': tfrec_dir, 'datasets': ['train'], 'label_fieldname': 'label',
                   'verbose': False}
use_kepler_ce = False  # use weighted CE loss based on the class proportions in the training set
satellite = 'kepler'

# add dataset parameters
config = old.old.config.add_dataset_params(satellite, multi_class, use_kepler_ce, config, ce_weights_args)

# add missing parameters in hpo with default values
config = old.old.config.add_default_missing_params(config=config)

# features to be extracted from the dataset
features_names = ['global_view', 'local_view']
features_dim = {feature_name: 2001 if 'global' in feature_name else 201 for feature_name in features_names}
features_names.append('scalar_params')
features_dim['scalar_params'] = 6
# choose indexes of scalar parameters to be extracted as features; None to get all of them in the TFRecords
scalar_params_idxs = None  # [1, 2]
features_dtypes = {feature_name: tf.float32 for feature_name in features_names}
features_set = {feature_name: {'dim': features_dim[feature_name], 'dtype': features_dtypes[feature_name]}
                for feature_name in features_names}

data_augmentation = False  # if True, uses data augmentation in the training set

patience = 20

sess_config = tf.ConfigProto(log_device_placement=False)

# instantiate the estimator using the TF Estimator API
classifier = tf.estimator.Estimator(ModelFn(base_model, config),
                                    config=tf.estimator.RunConfig(keep_checkpoint_max=1 if patience == -1
                                    else patience + 1,
                                                                  session_config=sess_config,
                                                                  tf_random_seed=None),
                                    model_dir=model_dir
                                    )

# input function for training on the training set
train_input_fn = InputFn(file_pattern=tfrec_dir + '/train*', batch_size=config['batch_size'],
                         mode=tf.estimator.ModeKeys.TRAIN, label_map=config['label_map'],
                         data_augmentation=data_augmentation,
                         filter_data=None,
                         features_set=features_set,
                         scalar_params_idxs=scalar_params_idxs)

# input functions for evaluation on the training, validation and test sets
traineval_input_fn = InputFn(file_pattern=tfrec_dir + '/train*', batch_size=config['batch_size'],
                             mode=tf.estimator.ModeKeys.EVAL, label_map=config['label_map'],
                             filter_data=None,
                             features_set=features_set,
                             scalar_params_idxs=scalar_params_idxs)
val_input_fn = InputFn(file_pattern=tfrec_dir + '/val*', batch_size=config['batch_size'],
                       mode=tf.estimator.ModeKeys.EVAL, label_map=config['label_map'],
                       filter_data=None, features_set=features_set,
                       scalar_params_idxs=scalar_params_idxs)
test_input_fn = InputFn(file_pattern=tfrec_dir + '/test*', batch_size=config['batch_size'],
                        mode=tf.estimator.ModeKeys.EVAL, label_map=config['label_map'],
                        filter_data=None, features_set=features_set,
                        scalar_params_idxs=scalar_params_idxs)

# METRIC LIST DEPENDS ON THE METRICS COMPUTED FOR THE ESTIMATOR - CHECK create_metrics method of class ModelFn in
# estimator_util.py
metrics_list = ['loss', 'accuracy', 'pr auc', 'precision', 'recall', 'roc auc', 'prec thr', 'rec thr']

dataset_ids = ['training', 'validation', 'test']

res = {dataset: {metric: [] for metric in metrics_list} for dataset in dataset_ids}

for epoch_i in range(1, n_epochs + 1):  # Train and evaluate the model for n_epochs

    print('\n\x1b[0;33;33m' + "Starting epoch %d of %d for %s (%s)" %
          (epoch_i, n_epochs, res_dir.split('/')[-1], classifier.model_dir.split('/')[-1]) + '\x1b[0m\n')

    # train model
    _ = classifier.train(train_input_fn)

    # evaluate model on given datasets
    print('\n\x1b[0;33;33m' + "Evaluating" + '\x1b[0m\n')
    res_i = {'training': classifier.evaluate(traineval_input_fn, name='training set'),
             'validation': classifier.evaluate(val_input_fn, name='validation set'),
             'test': classifier.evaluate(test_input_fn, name='test set')}

    for dataset in dataset_ids:
        for metric in metrics_list:
            res[dataset][metric].append(res_i[dataset][metric])