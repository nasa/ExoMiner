"""

"""

# 3rd party
import tensorflow as tf
import yaml
from pathlib import Path
from tensorflow.keras.utils import custom_object_scope
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
# from matplotlib.colors import LogNorm

# local
from src.utils.utils_dataio import InputFnv2 as InputFn, set_tf_data_type_for_features
from models.models_keras import Time2Vec, SplitLayer


def filter_by_uid(parsed_example, target_uid):

    return tf.squeeze(parsed_example['uid'] == target_uid)


res_dir = Path('/home6/msaragoc/work_dir/Kepler-TESS_exoplanet/experiments/tess_spoc_ffi/study_neighbors/')
config_fp = Path('/nobackupp19/msaragoc/work_dir/Kepler-TESS_exoplanet/experiments/tess_spoc_ffi/cv_exominernew_tess-spoc-ffi-s36-s72_5-31-2025_1127/cv_iter_0/ensemble_model/config_cv.yaml')
model_fp = Path('/nobackupp19/msaragoc/work_dir/Kepler-TESS_exoplanet/experiments/tess_spoc_ffi/cv_exominernew_tess-spoc-ffi-s36-s72_5-31-2025_1127/cv_iter_0/ensemble_model/ensemble_avg_model.keras')
norm_stats_diffimg_fp = Path('/home6/msaragoc/work_dir/Kepler-TESS_exoplanet/data/tfrecords/TESS/tfrecords_tess_spoc_ffi_s36-s72_4-23-2025_1709_data/tfrecords_tess_spoc_ffi_s36-s72_4-23-2025_1709_agg_bdslabels_diffimg_cv_5-22-2025_1038/tfrecords/eval_normalized/cv_iter_0/norm_stats/train_diffimg_norm_stats.csv')
norm_stats_scalars_fp = Path('/home6/msaragoc/work_dir/Kepler-TESS_exoplanet/data/tfrecords/TESS/tfrecords_tess_spoc_ffi_s36-s72_4-23-2025_1709_data/tfrecords_tess_spoc_ffi_s36-s72_4-23-2025_1709_agg_bdslabels_diffimg_cv_5-22-2025_1038/tfrecords/eval_normalized/cv_iter_0/norm_stats/train_scalarparam_norm_stats.csv')

norm_stats_diffimg_df = pd.read_csv(norm_stats_diffimg_fp)
norm_stats_scalars_df = pd.read_csv(norm_stats_scalars_fp)

# tce_uid = '83053699-1-S57'  # NEB
tce_uid = '307210830-1-S38'  # CP

res_dir.mkdir(parents=True, exist_ok=True)

# load configuration parameters
with open(config_fp, 'r') as file:
    config = yaml.unsafe_load(file)

# concatenate filepaths for the TFRecord files of all datasets
datasets_fps = []
for dataset in config['datasets']:
    datasets_fps += config['datasets_fps'][dataset]
config['datasets_fps'] = datasets_fps

# add uid to features to extract from TFRecord dataset
config['features_set']['uid'] = {'dim': [1,], 'dtype': 'string'}
config['features_set']['mag'] = {'dim': [1,], 'dtype': 'float'}
config['features_set'] = set_tf_data_type_for_features(config['features_set'])

# load model
custom_objects = {"Time2Vec": Time2Vec, 'SplitLayer': SplitLayer}
with custom_object_scope(custom_objects):
    model = load_model(filepath=model_fp, compile=False)

# create input function
predict_input_fn = InputFn(
            file_paths=config['datasets_fps'],
            batch_size=config['inference']['batch_size'],
            mode='PREDICT',
            label_map=config['label_map'],
            features_set=config['features_set'],
            multiclass=config['config']['multi_class'],
            feature_map=config['feature_map'],
            label_field_name=config['label_field_name'],
            filter_fn=lambda x: filter_by_uid(x, target_uid=tce_uid)
        )

# get example from dataset using input function
dataset = predict_input_fn()
example_batch = next(iter(dataset))
example_np = {k: v.numpy() for k, v in example_batch.items()}

example_score = model(example_batch, training=False).numpy().ravel()[0]
print(f'Model score before transformation: {example_score:.3f}')

# plot data before transformation
f, ax = plt.subplots()
im = ax.imshow(example_np['neighbors_imgs_tc_std_trainset'][0, 0])
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(im, cax=cax)
cbar.set_label(label=fr'Target-to-Neighbor Magnitude\nNormalized', fontsize=12)

ax.set_title(f'Model score before transformation: {example_score:.3f}')
f.savefig(res_dir / f'tce_{tce_uid}_neighbors_imgs_tc_std_trainset-original.png')

# create new example from original
# new_example_np = {k: np.array(v) for k, v in example_np.items()}

# # transform data to become less close to background offset
# # set neighbors to zero
# new_example_np['neighbors_imgs_tc_std_trainset'][:] = (0 - norm_stats_diffimg_df['neighbors_imgs_tc_median'] ) / (norm_stats_diffimg_df['neighbors_imgs_tc_std'] + 1e-12)
# # set difference image centroid offset stats to zero
# new_example_np['tce_dikco_msky_err_norm'] = (0 - norm_stats_scalars_df['tce_dikco_msky_err_median'] ) / (norm_stats_scalars_df['tce_dikco_msky_err_mad_std'] + 1e-12)
# new_example_np['tce_dikco_msky_norm'] = (0 - norm_stats_scalars_df['tce_dikco_msky_median'] ) / (norm_stats_scalars_df['tce_dikco_msky_mad_std'] + 1e-12)

# transform data to become more like a background offset
# new_example_np['neighbors_imgs_tc_std_trainset'][:] = (1 - norm_stats_diffimg_df['neighbors_imgs_tc_median'] ) / (norm_stats_diffimg_df['neighbors_imgs_tc_std'] + 1e-12)
# new_example_np['global_flux_view_fluxnorm'][:] = 0
# new_example_np['local_flux_view_fluxnorm'][:] = 0
# for feature_name in config['features_set']:
#     if feature_name == 'uid':
#         continue
#
#     new_example_np[feature_name][:] = 0

# exp 1: add neighbour on same pixel as target
center_px_coords = [55//2] * 2
neighbor_mag_range = np.arange(1, 30)
data_dict = {'neighbors_mag': [], 'model_score': []}
for neighbor_mag in neighbor_mag_range:

    new_example_np = {k: np.array(v) for k, v in example_np.items()}
    target_to_neighbor_mag_ratio = new_example_np['mag'][0][0] / neighbor_mag
    target_to_neighbor_mag_ratio_norm = ((target_to_neighbor_mag_ratio -
                                                           norm_stats_diffimg_df['neighbors_imgs_tc_median']) /
                                                           (norm_stats_diffimg_df['neighbors_imgs_tc_std'] + 1e-12))
    new_example_np['neighbors_imgs_tc_std_trainset'][:, :, center_px_coords[0], center_px_coords[1]] = (
        target_to_neighbor_mag_ratio_norm)

    # run predict on transformed example
    new_example_tf = {k: tf.convert_to_tensor(v) for k, v in new_example_np.items()}
    new_example_score = model(new_example_tf, training=False).numpy().ravel()[0]
    print(f'Model score after transformation (neighbor TMag {neighbor_mag} (Ratio: {target_to_neighbor_mag_ratio})): '
          f'{new_example_score:.3f}')

    data_dict['neighbors_mag'].append(neighbor_mag)
    data_dict['model_score'].append(new_example_score)

data_df = pd.DataFrame(data_dict)
data_df.to_csv(res_dir / f'{tce_uid}_neighbor_on_centertarget_magrange.csv', index=False)

img_indices_pairs = np.indices((55, 55)).reshape(2, -1).T
data_dict = {'px_indices': [], 'model_score': []}
for img_indices_pair in img_indices_pairs:

    new_example_np = {k: np.array(v) for k, v in example_np.items()}
    target_to_neighbor_mag_ratio = 1
    target_to_neighbor_mag_ratio_norm = ((target_to_neighbor_mag_ratio -
                                                           norm_stats_diffimg_df['neighbors_imgs_tc_median']) /
                                                           (norm_stats_diffimg_df['neighbors_imgs_tc_std'] + 1e-12))
    new_example_np['neighbors_imgs_tc_std_trainset'][:, :, img_indices_pair[0], img_indices_pair[1]] = (
        target_to_neighbor_mag_ratio_norm)

    # run predict on transformed example
    new_example_tf = {k: tf.convert_to_tensor(v) for k, v in new_example_np.items()}
    new_example_score = model(new_example_tf, training=False).numpy().ravel()[0]
    print(f'Model score after transformation (neighbor in pixel {img_indices_pair} '
          f'(TMag Ratio: {target_to_neighbor_mag_ratio})): '
          f'{new_example_score:.3f}')

    data_dict['px_indices'].append(img_indices_pair)
    data_dict['model_score'].append(new_example_score)

data_df = pd.DataFrame(data_dict)
data_df.to_csv(res_dir / f'{tce_uid}_neighbor_same_tmag_pixelrange.csv', index=False)

# plot data after transformation
f, ax = plt.subplots()
im = ax.imshow(new_example_np['neighbors_imgs_tc_std_trainset'][0, 0])
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(im, cax=cax)
cbar.set_label(label=fr'Target-to-Neighbor Magnitude\nNormalized', fontsize=12)
ax.set_title(f'Model score after transformation: {example_score:.3f}')
f.savefig(res_dir / f'tce_{tce_uid}_neighbors_imgs_tc_std_trainset-set-to-zero_testfluxzero.png')
