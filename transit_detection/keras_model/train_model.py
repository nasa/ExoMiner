"""
Train a  model.
"""

# 3rd party
from tensorflow.keras.utils import plot_model
from tensorflow.keras import callbacks  # TODO: remove python^
import numpy as np
import yaml
from pathlib import Path
import logging
from functools import partial

# local
from transit_detection.keras_model.utils_dataio import InputFnv2 as InputFn
from src.utils.utils_metrics import get_metrics
from models.utils_models import compile_model
from transit_detection.keras_model import model_keras
from src.utils.utils_dataio import set_tf_data_type_for_features

# from transit_detection.keras_model.utils_train import filter_examples_tfrecord_tce_model_snr


def train_model(config, model_dir, logger=None):

    # set tensorflow data type for features in the feature set
    config["features_set"] = set_tf_data_type_for_features(config["features_set"])

    base_model = getattr(model_keras, config["model_architecture"])

    model = base_model(config, config["features_set"]).kerasModel

    if config["plot_model"]:  # currently true, look @ model_dir
        # save plot of model
        plot_model(
            model,
            to_file=model_dir / "model.png",
            show_shapes=True,
            show_layer_names=True,
            rankdir="TB",
            expand_nested=False,
            dpi=48,
        )

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
    history = model.fit(
        x=train_input_fn(),
        y=None,
        batch_size=None,
        epochs=config["training"]["n_epochs"],
        verbose=config["verbose_model"],
        callbacks=config["callbacks_list"]["train"],
        validation_split=0.0,
        validation_data=val_input_fn() if val_input_fn is not None else None,
        shuffle=True,  # does the input function shuffle for every epoch?
        class_weight=None,
        sample_weight=None,
        initial_epoch=0,
        steps_per_epoch=None,
        validation_steps=None,
        # max_queue_size=10,  # does not matter when using input function with tf.data API
        # workers=1,  # same
        # use_multiprocessing=False,  # same
    )

    if logger is None:
        print("Saving model...")
    else:
        logger.info("Saving model...")

    # save model
    model.save(model_dir / f"model.keras")

    res = history.history

    np.save(
        model_dir / "res_train.npy", res
    )  # will contain monitored metrics for train/val, but not for test set


if __name__ == "__main__":

    # TODO: update file paths

    # output directory
    model_dir_fp = Path(
        "/nobackupp27/jochoa4/work_dir/job_runs/train_keras_model_05-04-2025_v2/"  # v3 using str labels
    )
    model_dir_fp.mkdir(parents=True, exist_ok=True)

    # YAML configuration
    config_fp = Path(
        "/nobackupp27/jochoa4/work_dir/exoplanet_dl/transit_detection/keras_model/config_train.yaml"
    )

    with open(config_fp, "r") as file:
        train_config = yaml.unsafe_load(file)

    # set up logger
    train_config["logger"] = logging.getLogger(name=f"train_model")
    logger_handler = logging.FileHandler(
        filename=model_dir_fp / "train_model.log", mode="w"
    )
    logger_formatter = logging.Formatter("%(asctime)s - %(message)s")
    train_config["logger"].setLevel(logging.INFO)
    logger_handler.setFormatter(logger_formatter)
    train_config["logger"].addHandler(logger_handler)
    train_config["logger"].info(f"Starting training model in {model_dir_fp}")

    train_model(train_config, model_dir_fp, logger=train_config["logger"])
