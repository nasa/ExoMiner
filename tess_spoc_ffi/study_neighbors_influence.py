"""
Study .
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
config_fp = Path('/nobackupp19/msaragoc/work_dir/Kepler-TESS_exoplanet/experiments/tess_spoc_ffi/cv_exominernew_tess-spoc-ffi-s36-s72_5-31-2025_1127/cv_iter_4/ensemble_model/config_cv.yaml')
model_fp = Path('/nobackupp19/msaragoc/work_dir/Kepler-TESS_exoplanet/experiments/tess_spoc_ffi/cv_exominernew_tess-spoc-ffi-s36-s72_5-31-2025_1127/cv_iter_4/ensemble_model/ensemble_avg_model.keras')
norm_stats_diffimg_fp = Path('/home6/msaragoc/work_dir/Kepler-TESS_exoplanet/data/tfrecords/TESS/tfrecords_tess_spoc_ffi_s36-s72_4-23-2025_1709_data/tfrecords_tess_spoc_ffi_s36-s72_4-23-2025_1709_agg_bdslabels_diffimg_cv_5-22-2025_1038/tfrecords/eval_normalized/cv_iter_4/norm_stats/train_diffimg_norm_stats.csv')
norm_stats_scalars_fp = Path('/home6/msaragoc/work_dir/Kepler-TESS_exoplanet/data/tfrecords/TESS/tfrecords_tess_spoc_ffi_s36-s72_4-23-2025_1709_data/tfrecords_tess_spoc_ffi_s36-s72_4-23-2025_1709_agg_bdslabels_diffimg_cv_5-22-2025_1038/tfrecords/eval_normalized/cv_iter_4/norm_stats/train_scalarparam_norm_stats.csv')

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

# # exp 1: add neighbour on same pixel as target
# center_px_coords = [55//2] * 2
# neighbor_mag_range = np.arange(1, 30)
# data_dict = {'neighbors_mag': [], 'model_score': []}
# for neighbor_mag in neighbor_mag_range:
#
#     new_example_np = {k: np.array(v) for k, v in example_np.items()}
#     target_to_neighbor_mag_ratio = new_example_np['mag'][0][0] / neighbor_mag
#     target_to_neighbor_mag_ratio_norm = ((target_to_neighbor_mag_ratio -
#                                                            norm_stats_diffimg_df['neighbors_imgs_tc_median']) /
#                                                            (norm_stats_diffimg_df['neighbors_imgs_tc_std'] + 1e-12))
#     new_example_np['neighbors_imgs_tc_std_trainset'][:, :, center_px_coords[0], center_px_coords[1]] = (
#         target_to_neighbor_mag_ratio_norm)
#
#     # run predict on transformed example
#     new_example_tf = {k: tf.convert_to_tensor(v) for k, v in new_example_np.items()}
#     new_example_score = model(new_example_tf, training=False).numpy().ravel()[0]
#     print(f'Model score after transformation (neighbor TMag {neighbor_mag} (Ratio: {target_to_neighbor_mag_ratio})): '
#           f'{new_example_score:.3f}')
#
#     data_dict['neighbors_mag'].append(neighbor_mag)
#     data_dict['model_score'].append(new_example_score)
#
# data_df = pd.DataFrame(data_dict)
# data_df.to_csv(res_dir / f'{tce_uid}_neighbor_on_centertarget_magrange.csv', index=False)
#
# # exp 2: add neighbor with same magnitude as target over all pixels in the image
# img_indices_pairs = np.indices((55, 55)).reshape(2, -1).T
# data_dict = {'px_indices': [], 'model_score': []}
# for img_indices_pair in img_indices_pairs:
#
#     new_example_np = {k: np.array(v) for k, v in example_np.items()}
#     target_to_neighbor_mag_ratio = 1
#     target_to_neighbor_mag_ratio_norm = ((target_to_neighbor_mag_ratio -
#                                                            norm_stats_diffimg_df['neighbors_imgs_tc_median']) /
#                                                            (norm_stats_diffimg_df['neighbors_imgs_tc_std'] + 1e-12))
#     new_example_np['neighbors_imgs_tc_std_trainset'][:, :, img_indices_pair[0], img_indices_pair[1]] = (
#         target_to_neighbor_mag_ratio_norm)
#
#     # run predict on transformed example
#     new_example_tf = {k: tf.convert_to_tensor(v) for k, v in new_example_np.items()}
#     new_example_score = model(new_example_tf, training=False).numpy().ravel()[0]
#     print(f'Model score after transformation (neighbor in pixel {img_indices_pair} '
#           f'(TMag Ratio: {target_to_neighbor_mag_ratio})): '
#           f'{new_example_score:.3f}')
#
#     data_dict['px_indices'].append(img_indices_pair)
#     data_dict['model_score'].append(new_example_score)
#
# data_df = pd.DataFrame(data_dict)
# data_df.to_csv(res_dir / f'{tce_uid}_neighbor_same_tmag_pixelrange.csv', index=False)

# exp 3: sample location and magnitude of new neighbor

exp_dir = res_dir / 'sampled_mag_pixel_location_6-23-2025_1301'
exp_dir.mkdir(parents=True, exist_ok=True)

# create new example from original
new_example_np = {k: np.array(v) for k, v in example_np.items()}

mag_range = np.arange(1, 30)
img_indices_pairs = np.indices((55, 55)).reshape(2, -1).T

sampled_arr = []
for img_indices_pair in img_indices_pairs:
    for mag in mag_range:
        sampled_arr.append((mag, img_indices_pair))
n_samples = len(sampled_arr)

# n_samples = 10000
# rng = np.random.default_rng(seed=21)
# sampled_mags = rng.choice(mag_range, n_samples)
# sampled_img_idxs = rng.choice(img_indices_pairs, n_samples)

fields = [
    'neighbors_mag',
    'neighbors_pixel_row',
    'neighbors_pixel_col',
    'model_score',
    'target_to_neighbors_mag_ratio'
]
data_dict = {field: [] for field in fields}
# for sample_i, (sampled_mag, sampled_img_idx) in enumerate(zip(sampled_mags, sampled_img_idxs)):
new_examples_list = []
for sample_i, (sampled_mag, sampled_img_idx) in enumerate(sampled_arr):

    if sample_i % 50 == 0:
        print(f'Iterating on sample {sample_i + 1}/({n_samples}): {sampled_mag}, {sampled_img_idx}...')

    new_example_np = {k: np.array(v) for k, v in example_np.items()}

    # if (sampled_mag in data_dict['neighbors_mag'] and sampled_img_idx[0] in data_dict['neighbors_pixel_row'] and
    #         sampled_img_idx[1] in data_dict['neighbors_pixel_col']):
    #     print(f'Sampled pair ({sampled_mag}, {sampled_img_idx}) was already used. Skipping.')

    target_to_neighbor_mag_ratio = example_np['mag'][0][0] / sampled_mag

    target_to_neighbor_mag_ratio_norm = ((target_to_neighbor_mag_ratio -
                                                           norm_stats_diffimg_df['neighbors_imgs_tc_median']) /
                                                           (norm_stats_diffimg_df['neighbors_imgs_tc_std'] + 1e-12))

    # new_example_np['neighbors_imgs_tc_std_trainset'] = np.array(example_np['neighbors_imgs_tc_std_trainset'])
    new_example_np['neighbors_imgs_tc_std_trainset'][:, :, sampled_img_idx[0], sampled_img_idx[1]] = (
        target_to_neighbor_mag_ratio_norm)

    # # run predict on transformed example
    # new_example_tf = {k: tf.convert_to_tensor(v) for k, v in new_example_np.items()}

    # Add the transformed example to the list
    new_examples_list.append(new_example_np)

    # new_example_score = model(new_example_tf, training=False).numpy().ravel()[0]

    # print(f'Model score after transformation (neighbor in pixel {sampled_img_idx} with TMag {sampled_mag} | '
    #       f'ratio {target_to_neighbor_mag_ratio:.3f}): '
    #       f'(TMag Ratio: {target_to_neighbor_mag_ratio})): '
    #       f'{new_example_score:.3f}')

    data_dict['neighbors_mag'].append(sampled_mag)
    data_dict['neighbors_pixel_row'].append(sampled_img_idx[0])
    data_dict['neighbors_pixel_col'].append(sampled_img_idx[1])
    data_dict['target_to_neighbors_mag_ratio'].append(target_to_neighbor_mag_ratio)
    # data_dict['model_score'].append(new_example_score)

# After the loop, convert the list of examples to tensors
new_examples_tensor = {k: tf.convert_to_tensor([example[k] for example in new_examples_list])
                       for k in new_example_np.keys()}
# Run the model on the batch of transformed examples
print('Predicting on new examples...')
new_examples_scores = model(new_examples_tensor, training=False).numpy()
# If you want to get the score for each example, you can access it like this
new_example_scores_flat = new_examples_scores.ravel()  # If needed, flatten the scores

data_dict['model_score'] = new_example_scores_flat

data_df = pd.DataFrame(data_dict)
data_df.to_csv(exp_dir / f'{tce_uid}_sampled_neighbors.csv', index=False)

# plot data after transformation
print(f'Plotting data...')
for sample_i in range(n_samples):
    f, ax = plt.subplots()
    im = ax.imshow(new_examples_list[sample_i]['neighbors_imgs_tc_std_trainset'][0, 0])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label(label=f'Target-to-Neighbor Magnitude\nNormalized', fontsize=12)
    ax.set_title(f'Model score: {data_dict["model_score"][sample_i]:.3f}\n Neighbor TMag: {data_dict["neighbors_mag"][sample_i]} '
                 f'(ratio: {data_dict["target_to_neighbors_mag_ratio"][sample_i]:.3f}) | '
                 f'Row {data_dict["neighbors_pixel_row"][sample_i]}, Col {data_dict["neighbors_pixel_col"][sample_i]} px')
    f.savefig(exp_dir / f'tce_{tce_uid}_sampled_neighbors_mag{data_dict["neighbors_mag"][sample_i]}_row{data_dict["neighbors_pixel_row"][sample_i]}_'
                        f'col{data_dict["neighbors_pixel_col"][sample_i]}.png')
    plt.close()

# # plot data after transformation
# f, ax = plt.subplots()
# im = ax.imshow(new_example_np['neighbors_imgs_tc_std_trainset'][0, 0])
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="5%", pad=0.05)
# cbar = plt.colorbar(im, cax=cax)
# cbar.set_label(label=fr'Target-to-Neighbor Magnitude\nNormalized', fontsize=12)
# ax.set_title(f'Model score after transformation: {example_score:.3f}')
# f.savefig(res_dir / f'tce_{tce_uid}_neighbors_imgs_tc_std_trainset-set-to-zero_testfluxzero.png')
