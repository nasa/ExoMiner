"""
Train a  model.
"""

# 3rd party
from tensorflow.keras.utils import plot_model
import tensorflow as tf
from tensorflow.keras import callbacks  # TODO: remove python^
import numpy as np
import yaml
from pathlib import Path
import logging
from functools import partial
import csv

from collections import Counter
import pandas as pd
import uuid
import random

# local
from transit_detection.keras_model.utils_dataio import InputFnv2 as InputFn
from src.utils.utils_metrics import get_metrics
from models.utils_models import compile_model
from transit_detection.keras_model import test_model_keras as model_keras
from src.utils.utils_dataio import set_tf_data_type_for_features

import random

# from transit_detection.norm_pipeline.plot_examples_from_dataset import plot_flux_and_diff_img_data

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict

from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm

counter = 0

def increment_counter():
    global counter
    counter += 1

def get_counter_val():
    global counter
    return counter

def plot_flux_and_diff_img_data(
    label,
    flux_curve,
    diff_img,
    oot_img,
    snr_img,
    target_img,
    save_fp,
    norm=True,
    snr=None,
    midpoint=None,
):
    """Plot difference image data for TCE in a given quarter/sector.

    Args:
        diff_img: NumPy array, difference image
        oot_img: NumPy array, out-of-transit image
        snr_img: NumPy array, SNR image
        target_img: NumPy array, target location image
        target_coords: dict, target location 'x' and 'y'
        img_num: str, quarter/sector run
        uid: str, TCE ID
        save_fp: Path, file path to saved plot

    Returns:

    """
    label = "In Transit" if label == 1.0 else "Out of Transit"
    snr = str(round(snr, 2)) if snr else None
    it_idx = np.argmax(target_img)
    row, col = np.unravel_index(
        it_idx, target_img.shape
    )  # Get row and col as if array were flattened
    target_coords = {"x": row, "y": col}

    fig = plt.figure(figsize=(16, 16))

    gs = gridspec.GridSpec(3, 2, figure=fig)

    ax = fig.add_subplot(gs[0, :])
    time = np.linspace(0, 100, 100)
    ax.plot(time, flux_curve, "b-", linewidth=2, marker="o", markersize=3, alpha=0.6)
    ax.set_title(f"{label} Flux Curve w/ midpoint {midpoint if midpoint else None} Over Time")
    ax.set_xlabel("Time")
    ax.set_ylabel("Flux value") if not norm else ax.set_ylabel("Normalized Flux value")

    # Approximate image center

    # diff img
    ax = fig.add_subplot(gs[1, 0])
    im = ax.imshow(diff_img, norm=None)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    ax.scatter(
        target_coords["y"], target_coords["x"], marker="x", color="r", label="Target"
    )
    ax.set_ylabel("Row")
    ax.set_xlabel("Col")
    ax.legend()
    (
        ax.set_title("Difference Flux (e-/cadence)")
        if not norm
        else ax.set_title("Normalized Difference Flux")
    )
    # oot img
    ax = fig.add_subplot(gs[1, 1])
    im = ax.imshow(oot_img, norm=LogNorm() if not norm else None)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    ax.scatter(
        target_coords["y"], target_coords["x"], marker="x", color="r", label="Target"
    )
    ax.set_ylabel("Row")
    ax.set_xlabel("Col")
    ax.legend()
    (
        ax.set_title("Out-of-Transit Flux (e-/cadence)")
        if not norm
        else ax.set_title("Normalized Out-of-Transit Flux")
    )
    # target img
    ax = fig.add_subplot(gs[2, 0])
    ax.imshow(target_img)
    ax.set_ylabel("Row")
    ax.set_xlabel("Col")
    ax.set_title("Target Position")
    # snr img
    ax = fig.add_subplot(gs[2, 1])
    im = ax.imshow(snr_img, norm=LogNorm() if not norm else None)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    ax.scatter(
        target_coords["y"], target_coords["x"], marker="x", color="r", label="Target"
    )
    ax.set_ylabel("Row")
    ax.set_xlabel("Col")
    ax.legend()
    (
        ax.set_title(f"Difference SNR: {snr}")
        if not norm
        else ax.set_title(f"Normalized Difference SNR: {snr}")
    )

    plt.tight_layout()
    try: 
        plt.savefig(save_fp)
        plt.close()

        print(f"Plotted to {save_fp}")
    except Exception as e:
        print(f"ERROR plotting: {e}")


# from transit_detection.keras_model.utils_train import filter_examples_tfrecord_tce_model_snr
class InputLoggerCallback(callbacks.Callback):
    def on_train_batch_begin(self, batch, logs=None):
        x, y = self.model._training_data_adapter._current_input_data
        if isinstance(x, dict):
            print(f"\n Batch {batch} Inputs::")


def plot_from_tensor(batch_i, batch, batch_size):
    def _py_plot(flux, diff_img, oot_img, snr_img, target_img, disposition, midpoint, label, batch_i, ex_i):
        
        # Clean single values
        disposition = disposition[0].decode('utf-8')
        midpoint = midpoint[0]

        tag = f"label_{label}_batch_{batch_i}_ex_{ex_i}_disp_{disposition}_{uuid.uuid4().hex[:6]}"
        save_fp = "/nobackupp27/jochoa4/work_dir/job_runs/visualize_train_input_v2/" + tag + '.png'
        print(f"Saving to {save_fp} with tag: {tag}")
        plot_flux_and_diff_img_data(label, flux, diff_img, oot_img, snr_img, target_img, save_fp, True, None, midpoint)
        return np.int64(100)

    inputs, labels = batch

    ex_i = random.randint(0, batch_size)

    tensors = []
    # flux_tensor = inputs['flux_norm'][i]
    # diff_img_tensor = inputs['diff_img_stdnorm'][i]
    # oot_img_tensor = inputs['oot_img_stdnorm'][i]
    # snr_img_tensor = inputs['snr_img_stdnorm'][i]
    # target_img_tensor = inputs['target_img'][i]
    # disposition_tensor = inputs['disposition'][i]
    # midpoint_tensor = inputs['t'][i]
    tensors.append(inputs['flux_norm'][ex_i])
    tensors.append(inputs['diff_img_stdnorm'][ex_i])
    tensors.append(inputs['oot_img_stdnorm'][ex_i])
    tensors.append(inputs['snr_img_stdnorm'][ex_i])
    tensors.append(inputs['target_img'][ex_i])
    tensors.append(inputs['disposition'][ex_i])
    tensors.append(inputs['t'][ex_i])

    label = labels[ex_i] if labels is not None else tf.constant(-1)

    print(f"Tensors: {tensors}")


    # label_tensor = labels[i] if labels is not None else tf.constant(-1)


    tf.numpy_function(
        func=_py_plot,
        inp=[*tensors, label, batch_i, ex_i],
        Tout=tf.int64
    )

    return inputs, labels
    

def visualize_train_model_input(config, model_dir, plot_dir, logger=None):

    # set tensorflow data type for features in the feature set
    config["features_set"] = set_tf_data_type_for_features(config["features_set"])
    print(f"Feature Set: {config['features_set']}")

    base_model = getattr(model_keras, config["model_architecture"])

    model = base_model(config, config["features_set"]).kerasModel

    # get model summary
    with open(model_dir / "model_summary.txt", "w") as f:
        model.summary(print_fn=lambda x: f.write(x + "\n"))

    # setup metrics to be monitored
    metrics_list = get_metrics(
        clf_threshold=config["metrics"]["clf_thr"],
        num_thresholds=config["metrics"]["num_thr"],
    )

    # compile model - set optimizer, loss and metrics
    model = compile_model(model, config, metrics_list)  # binary crossentropy

    # input function for training, validation and test
    train_input_fn = InputFn(
        file_paths=config["datasets_fps"]["train"],
        batch_size=config["training"]["batch_size"],
        mode="TRAIN",
        label_map=config["label_map"],
        data_augmentation=config["training"]["data_augmentation"],  # ??? not neccesary
        online_preproc_params=config["training"]["online_preprocessing_params"],  # ???
        features_set=config["features_set"],
        category_weights=config["training"]["category_weights"],  # ???
        multiclass=config["config"]["multi_class"],
        feature_map=config["feature_map"],
        shuffle_buffer_size=config["training"]["shuffle_buffer_size"],  # ???
        label_field_name=config["label_field_name"],
        # filter_fn=partial(filter_examples_tfrecord_tce_model_snr, snr_threshold=20),
    )
    if "val" in config["datasets"]:
        val_input_fn = InputFn(
            file_paths=config["datasets_fps"]["val"],
            batch_size=config["training"]["batch_size"],
            mode="EVAL",
            label_map=config["label_map"],
            features_set=config["features_set"],
            multiclass=config["config"]["multi_class"],
            feature_map=config["feature_map"],
            label_field_name=config["label_field_name"],
            # filter_fn=partial(filter_examples_tfrecord_tce_model_snr, snr_threshold=20),
        )
    else:
        val_input_fn = None

    print(f"Succesfully set up input functions.")

    # keep early stopping?
    # early stopping callback

    config["callbacks_list"] = {
        "train": [
            callbacks.EarlyStopping(**config["callbacks"]["early_stopping"]),
            callbacks.BackupAndRestore(**config["callbacks"]["backup_and_restore"]),
            callbacks.ModelCheckpoint(**config["callbacks"]["model_checkpoint"]),
            callbacks.CSVLogger(**config["callbacks"]["csv_logger"]),
        ],
    }

    # fit the model to the training data
    if logger is None:
        print("Training model...")
    else:
        logger.info("Training model...")

    print(f"Processing training dataset...")

    dataset = train_input_fn()
    for i, batch in enumerate(dataset):
        batch_size = len(batch[1]) # labels
        plot_from_tensor(i, batch, batch_size)
    # visualized_dataset = dataset.enumerate().map(plot_from_tensor)
    # print(f"DATASET: {type(dataset)}")

    # batch_stats = []

    # for batch_i, (x, y) in enumerate(dataset, start=1): # Process a single batch
        
    #     flux_windows = x['flux_norm']

    #     diff_imgs = x['diff_img_stdnorm']
    #     oot_imgs = x['oot_img_stdnorm']
    #     snr_imgs = x['snr_img_stdnorm']
    #     target_imgs = x['target_img']

    #     dispositions = x['disposition']


    #     uids = x['uid']
    #     midpoints = x['t']

    #     print(f"Disposition: {x['disposition']}")

    #     labels = y
    #     label_counts = Counter(labels.numpy())
    #     disp_counts = Counter([d.numpy()[0].decode('utf-8') for d in dispositions])

    #     row = {'batch_index': batch_i}
        
    #     for disp in ['NTP', 'EB', 'CP', 'KP']:
    #         row[disp] = disp_counts.get(disp,0)

    #     row[f"batch_it_num"] = sum(labels.numpy())
    
    #     batch_stats.append(row)

    # history = model.fit(    
    #     x=visualized_dataset,
    #     y=None,
    #     batch_size=None,
    #     epochs=config["training"]["n_epochs"],
    #     verbose=config["verbose_model"],
    #     callbacks=config["callbacks_list"]["train"],
    #     validation_split=0.0,
    #     validation_data=val_input_fn() if val_input_fn is not None else None,
    #     shuffle=True,  # does the input function shuffle for every epoch?
    #     class_weight=None,
    #     sample_weight=None,
    #     initial_epoch=0,
    #     steps_per_epoch=None,
    #     validation_steps=None,
    #     max_queue_size=10,  # does not matter when using input function with tf.data API
    #     workers=1,  # same
    #     use_multiprocessing=False,  # same
    # )



        
        # for example_i, (flux_window, diff_img, oot_img, snr_img, target_img, label, disposition, uid, midpoint) in enumerate(zip(flux_windows, diff_imgs, oot_imgs, snr_imgs, target_imgs, labels, dispositions, uids, midpoints), start=1):
        #     save_fp = plot_dir / f"{'it' if label == 1.0 else 'oot'}_example_{example_i * batch_i}_{disposition.numpy()[0].decode('utf-8')}_norm.png"
        #     plot_flux_and_diff_img_data(flux_curve=flux_window, diff_img=diff_img, oot_img=oot_img, snr_img=snr_img, target_img=target_img, save_fp=save_fp, label=label, norm=True)
    # df = pd.DataFrame(batch_stats).fillna(0).astype(int)
    # df.to_csv('label_distribution_per_batch.csv', index=False)
        
        


        
    # for i, example in enumerate(dataset):
    #     print(f"Example {i} in dataset: ")
    #     print(f"    {type(example)}")
    #     print(f"          {example}")

if __name__ == "__main__":

    # TODO: update file paths
    plot_dir = Path("/nobackupp27/jochoa4/work_dir/job_runs/visualize_train_input_v2/")
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    # output directory
    model_dir_fp = Path(
        "/nobackupp27/jochoa4/work_dir/job_runs/test_keras_model_inputs/"
    )
    model_dir_fp.mkdir(parents=True, exist_ok=True)

    # YAML configuration
    config_fp = Path(
        "/nobackupp27/jochoa4/work_dir/exoplanet_dl/transit_detection/keras_model/test_config_train.yaml"
    )

    with open(config_fp, "r") as file:
        train_config = yaml.unsafe_load(file)

    visualize_train_model_input(train_config, model_dir_fp, plot_dir, logger=None)
