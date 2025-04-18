"""
Plot features of example.
"""

# 3rd party
from pathlib import Path
import tensorflow as tf
import yaml
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# local
from src.utils.utils_dataio import set_tf_data_type_for_features

def example_parser(serialized_example, features_set):
    """Parses a single tf.Example into feature and label tensors.

    :param serialized_example: a single tf.Example
    :return:
        tuple, feature and label tensors
    """

    # get features names, shapes and data types to be extracted from the TFRecords
    data_fields = {}
    for feature_name, feature_info in features_set.items():
        if len(feature_info['dim']) > 1 and feature_info['dim'][-1] > 1:  # N-D feature, N > 1
            data_fields[feature_name] = tf.io.FixedLenFeature(1, tf.string)
        else:
            data_fields[feature_name] = tf.io.FixedLenFeature(feature_info['dim'], feature_info['dtype'])

    # parse the features
    parsed_features = tf.io.parse_single_example(serialized=serialized_example, features=data_fields)

    # initialize feature output
    output = {}
    for feature_name, value in parsed_features.items():

        feature_info = features_set[feature_name]
        if len(feature_info['dim']) > 1 and feature_info['dim'][-1] > 1:  # parse tensors
            value = tf.io.parse_tensor(serialized=value[0], out_type=features_set[feature_name]['dtype'])
            value = tf.reshape(value, features_set[feature_name]['dim'])

        output[feature_name] = value

    return output


def filter_examples_uid(x, tce_uid):

    return tf.squeeze(tf.equal(x['uid'], tce_uid))

#%%

cv_folds_fp = Path('/u/msaragoc/work_dir/Kepler-TESS_exoplanet/data/tfrecords/TESS/tfrecords_tess_spoc_ffi_s36-s72_multisector_s56-s69_1-3-2025_1157_data/cv_tfrecords_tess_spoc_ffi_s36-s72_multisector_s56-s69_1-6-2025_1132/tfrecords/eval_with_2mindata_transferlearning/cv_2min_ffi_combined/cv_iterations.yaml')
config_fp = Path('/nobackupp19/msaragoc/work_dir/Kepler-TESS_exoplanet/codebase/tess_spoc_ffi/diff_img/config_cv_train.yaml')
dataset = 'train'
tce_uid = '382069441-1-S12'
plot_dir = Path('/u/msaragoc/work_dir/Kepler-TESS_exoplanet/experiments/tess_spoc_ffi/diff_img/plot_examples_4-15-2025_1649')

plot_dir.mkdir(exist_ok=True)

with open(config_fp, 'r') as f:
    config = yaml.unsafe_load(f)

with open(cv_folds_fp, 'r') as f:
    cv_folds = yaml.unsafe_load(f)


cv_iter_dataset = cv_folds[0]

config['datasets_fps'] = cv_iter_dataset

config['features_set']['uid'] = { 'dim': [ 1, ], 'dtype': 'string' }
config['features_set']['label'] = { 'dim': [ 1, ], 'dtype': 'string' }
config['features_set']['mag'] = { 'dim': [ 1, ], 'dtype': 'float' }
config['features_set']['tce_dikco_msky'] = { 'dim': [ 1, ], 'dtype': 'float' }
config['features_set']['tce_dikco_msky_err'] = { 'dim': [ 1, ], 'dtype': 'float' }
config['features_set']['diff_imgs'] = { 'dim': [ 5, 33, 33 ], 'dtype': 'float' }
config['features_set']['oot_imgs'] = { 'dim': [ 5, 33, 33 ], 'dtype': 'float' }

config['features_set'] = set_tf_data_type_for_features(config['features_set'])

# predict_input_fn = InputFn(
#     file_paths=config['datasets_fps'][dataset],
#     batch_size=config['inference']['batch_size'],
#     mode='PREDICT',
#     label_map=config['label_map'],
#     features_set=config['features_set'],
#     multiclass=config['config']['multi_class'],
#     feature_map=config['feature_map'],
#     label_field_name=config['label_field_name'],
# )

filenames = [str(fp) for fp in config['datasets_fps'][dataset]]
filename_dataset = tf.data.Dataset.from_tensor_slices(filenames)

# map a TFRecordDataset object to each tfrecord filepath
dataset = filename_dataset.flat_map(tf.data.TFRecordDataset)

dataset = dataset.map(lambda string_example: example_parser(string_example, config['features_set']),
                      num_parallel_calls=tf.data.AUTOTUNE,
                      deterministic=True)

dataset = dataset.filter(lambda parsed_example: filter_examples_uid(parsed_example, tce_uid))

for example in dataset:
    continue
    # diff_img_data = example['diff_imgs_std_trainset']
    # print(diff_img_data.numpy())

# print(example)

for sector in range(5):
    gs = gridspec.GridSpec(3, 2)
    f = plt.figure(figsize=(14, 10))

    ax = plt.subplot(gs[0, 0])
    im = ax.imshow(example['diff_imgs_std_trainset'].numpy()[sector], cmap=plt.cm.viridis, origin='lower', aspect=0.7)
    cbar_im = plt.colorbar(im, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
    cbar_im.set_label(r'Normalized Flux')
    # cbar_im.ax.set_position([cbar_im.ax.get_position().x1 - 0.02,
    #                          cbar_im.ax.get_position().y0,
    #                          cbar_im.ax.get_position().width,
    #                          cbar_im.ax.get_position().height])
    ax.set_ylabel('Row')
    ax.set_xlabel('Col', labelpad=10)
    ax.set_title('Difference Image', pad=50)

    ax = plt.subplot(gs[0, 1])
    im = ax.imshow(example['diff_imgs'].numpy()[sector], cmap=plt.cm.viridis, origin='lower', aspect=0.7)
    cbar_im = plt.colorbar(im, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
    cbar_im.set_label(r'Flux [$e^-/cadence$]')
    # cbar_im.ax.set_position([cbar_im.ax.get_position().x1 - 0.02,
    #                          cbar_im.ax.get_position().y0,
    #                          cbar_im.ax.get_position().width,
    #                          cbar_im.ax.get_position().height])
    ax.set_ylabel('Row')
    ax.set_xlabel('Col', labelpad=10)

    ax = plt.subplot(gs[1, 0])
    im = ax.imshow(example['oot_imgs_std_trainset'].numpy()[sector], cmap=plt.cm.viridis, origin='lower', aspect=0.7)
    cbar_im = plt.colorbar(im, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
    cbar_im.set_label(r'Normalized Flux')
    # cbar_im.ax.set_position([cbar_im.ax.get_position().x1 - 0.02,
    #                          cbar_im.ax.get_position().y0,
    #                          cbar_im.ax.get_position().width,
    #                          cbar_im.ax.get_position().height])
    ax.set_ylabel('Row')
    ax.set_xlabel('Col', labelpad=10)
    ax.set_title('Out-of-transit Image', pad=50)

    ax = plt.subplot(gs[1, 1])
    im = ax.imshow(example['oot_imgs'].numpy()[sector], cmap=plt.cm.viridis, origin='lower', aspect=0.7)
    cbar_im = plt.colorbar(im, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
    cbar_im.set_label(r'Flux [$e^-/cadence$]')
    # cbar_im.ax.set_position([cbar_im.ax.get_position().x1 - 0.02,
    #                          cbar_im.ax.get_position().y0,
    #                          cbar_im.ax.get_position().width,
    #                          cbar_im.ax.get_position().height])

    ax = plt.subplot(gs[2, :])
    ax.imshow(example['target_imgs'].numpy()[sector], origin='lower', aspect=0.7)
    ax.set_title('Target Location Image', pad=50)

    f.suptitle(f'TMag={example["mag"].numpy()[0]:.3f} | tce_dikco_msky={example["tce_dikco_msky"].numpy()[0]:.3f} | '
               f'tce_dikco_msky_err={example["tce_dikco_msky_err"].numpy()[0]:.3f} | '
               f'QMetrics={example["quality"].numpy()[sector][0]:.3f}')
    f.tight_layout()
    f.savefig(plot_dir /
              f'diff_img_data_tce_{example["uid"].numpy()[0].decode("utf-8")}_sector{sector}_{example["label"].numpy()[0].decode("utf-8")}')
