"""
Predict model.
"""

# 3rd party
from tensorflow.keras.utils import plot_model, custom_object_scope
from tensorflow.keras.models import load_model
from tensorflow.keras import callbacks
import numpy as np
import yaml
from pathlib import Path
import logging
from functools import partial
import pandas as pd

# temp
import os
import psutil

# local
from transit_detection.keras_model.utils_dataio import InputFnv2 as InputFn
from src.utils.utils_metrics import get_metrics
from models.utils_models import compile_model
from transit_detection.keras_model import model_keras
from src.utils.utils_dataio import set_tf_data_type_for_features
from models.models_keras import Time2Vec


def predict_model(config, model_path, res_dir, logger=None):

    config["features_set"] = set_tf_data_type_for_features(config["features_set"])

    # load models
    if logger is None:
        print("Loading model...")
    else:
        logger.info("Loading model...")
    custom_objects = {"Time2Vec": Time2Vec}

    with custom_object_scope(custom_objects):
        model = load_model(filepath=model_path, compile=False)

    if config["write_model_summary"]:
        with open(res_dir / "model_summary.txt", "w") as f:
            model.summary(print_fn=lambda x: f.write(x + "\n"))

    # plot model and save the figure
    if config["plot_model"]:
        plot_model(
            model,
            to_file=res_dir / "model.png",
            show_shapes=False,
            show_layer_names=True,
            rankdir="TB",
            expand_nested=False,
            dpi=96,
        )

    # set up metrics to be monitored
    if not config["config"]["multi_class"]:
        metrics_list = get_metrics(
            clf_threshold=config["metrics"]["clf_thr"],
            num_thresholds=config["metrics"]["num_thr"],
        )

    # compile model - loss and metrics
    model = compile_model(model, config, metrics_list, train=False)

    # initialize results dictionary for the predicted datasets
    res = {}
    for dataset in config["datasets"]:

        if dataset == "predict":
            continue

        if logger is None:
            print(f"Evaluating on dataset {dataset}")
        else:
            logger.info(f"Evaluating on dataset {dataset}")

        # input function for evaluating on each dataset
        pred_input_fn = InputFn(
            file_paths=config["datasets_fps"][
                dataset
            ],  # list of train/val/test path, calls eval
            batch_size=config["evaluation"]["batch_size"],
            mode="TRAIN",
            label_map=config["label_map"],
            features_set=config["features_set"],
            online_preproc_params=None,
            multiclass=config["config"]["multi_class"],
            feature_map=config["feature_map"],
            label_field_name=config["label_field_name"],
            # category_weights=config["training"]["category_weights"]
        )

        # predict model in the given dataset
        tf_dataset = pred_input_fn()

        uids, dispositions, times = [], [], []
        labels, preds = [], []
        tw_flags = []

        model_input_keys = set(model.input_names)

        print(f"model_input_keys: {list(model_input_keys)}")
        print(f"Starting processing batch_features")
        # process = psutil.Process(os.getpid())
        # print(f"Memory used: {process.memory_info().rss / 1024 ** 2:.2f} MB")

        for batch_features, batch_labels in tf_dataset:
            # process = psutil.Process(os.getpid())
            # print(f"Memory used: {process.memory_info().rss / 1024 ** 2:.2f} MB")

            # Predict
            batch_preds = model.predict(batch_features, verbose=0)

            batch_preds = np.squeeze(batch_preds)

            preds.extend(batch_preds.tolist())
            labels.extend(batch_labels.numpy().tolist())

            uids.extend([x[0].decode("utf-8") for x in batch_features["uid"].numpy()])
            dispositions.extend(
                [x[0].decode("utf-8") for x in batch_features["disposition"].numpy()]
            )
            times.extend(t[0] for t in batch_features["t"].numpy().tolist())
            tw_flags.extend(
                [
                    x[0].decode("utf-8")
                    for x in batch_features["transit_example"].numpy()
                ]
            )

        print(f"Finished processing batch_features")
        df = pd.DataFrame(
            {
                "uid": uids,
                "disposition": dispositions,
                "time": times,
                "label": labels,
                "pred_label": preds,
                "pred_prob": preds,
                "tw_flag": tw_flags,
            }
        )

        if model.output_shape[-1] == 1:
            df["pred_label"] = (df["pred_label"] > 0.5).astype(int)

        print(f"Saving to csv")
        df.to_csv(res_dir / f"preds_{dataset}.csv", index=False)


if __name__ == "__main__":
    model_name = "TESS_exoplanet_dataset_07-24-2025_no_detrend_split_norm_filt_3sig_it_EB_bal_ntp_low_lr"

    config_fp = Path(
        "/nobackupp27/jochoa4/work_dir/exoplanet_dl/transit_detection/keras_model/config_train.yaml"
    )  # File path to YAML config file

    model_fp = Path(
        f"/nobackupp27/jochoa4/work_dir/job_runs/train_model_{model_name}/model.keras"
    )  # Model file path.

    output_dir = Path(
        f"/nobackupp27/jochoa4/work_dir/job_runs/predict_model_{model_name}"
    )  # Output directory path

    output_dir.mkdir(parents=True, exist_ok=True)

    with open(config_fp, "r") as file:
        eval_config = yaml.unsafe_load(file)

    # set up logger
    eval_config["logger"] = logging.getLogger(name=f"predict_model")
    logger_handler = logging.FileHandler(
        filename=output_dir / "predict_model.log", mode="w"
    )
    logger_formatter = logging.Formatter("%(asctime)s - %(message)s")
    eval_config["logger"].setLevel(logging.INFO)
    logger_handler.setFormatter(logger_formatter)
    eval_config["logger"].addHandler(logger_handler)
    eval_config["logger"].info(f"Starting evaluating model {model_fp} in {output_dir}")

    predict_model(eval_config, model_fp, output_dir, logger=eval_config["logger"])
