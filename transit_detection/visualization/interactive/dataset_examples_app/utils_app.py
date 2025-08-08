from pathlib import Path
from typing import Union
import tensorflow as tf
from collections import defaultdict
import json
import lightkurve as lk
import numpy as np
import plotly.graph_objects as go
from functools import lru_cache
import io
import base64


def fig_to_base64(fig):
    buffer = io.BytesIO()
    fig.write_image(buffer, format="png")  # Requires kaleido
    encoded = base64.b64encode(buffer.getvalue()).decode()
    return "data:image/png;base64," + encoded


def _sector_dd():
    return {"lightcurve": None, "examples_by_tce": defaultdict(list)}


def load_index(index_json_fp: Union[Path, str]):
    with open(index_json_fp, "r") as fp:
        return json.load(fp)


def build_index(
    tfrec_dir: Union[Path, str],
    index_json_fp: Union[Path, str],
    tfrec_glob_pattern: str = "*.tfrecord",
):
    tfrec_index = {}
    base = Path(tfrec_dir)
    assert base.exists() and base.is_dir(), f"Invalid TFRecord dir: {base}"

    # Walk *all* subdirectories for any .tfrecord
    for tfrec_fp in base.rglob(tfrec_glob_pattern):
        if not tfrec_fp.is_file():
            continue

        # keep track of where it came from
        rel_subdir = tfrec_fp.parent.relative_to(base)  # e.g. "train/run1"
        try:
            for raw in tf.data.TFRecordDataset(tfrec_fp).as_numpy_iterator():
                ex = tf.train.Example()
                ex.ParseFromString(raw)

                uid = ex.features.feature["uid"].bytes_list.value[0].decode()
                tce_uid = uid.split("_")[0]
                target_id = tce_uid.split("-")[0]
                sector_run = tce_uid.split("S")[-1]

                # store both path and subdir
                tfrec_index.setdefault(target_id, {})[sector_run] = {
                    "path": str(tfrec_fp),
                    "subdir": str(rel_subdir),
                }
        except Exception as e:
            print(f"Skipping {tfrec_fp}: {e}")

    if not tfrec_index:
        raise ValueError("No TFRecords found!")

    with open(index_json_fp, "w") as fp:
        json.dump(tfrec_index, fp, indent=2)


def get_tfrec_fp_from_index(index: dict, target: str, sector_run: str):
    """
    Now returns a dict {path: ..., subdir: ...}, so in your app you can do:

        entry = index[target][sector_run]
        tfrec_path = entry["path"]
        came_from = entry["subdir"]
    """
    return index.get(target, {}).get(sector_run, {})


def load_t_sr_data(target: str, sector_run: str, tfrec_fp: Union[str, Path]):
    tfrec_fp = Path(tfrec_fp)
    assert (
        tfrec_fp.exists() and tfrec_fp.is_file()
    ), f"TFRecord path invalid: {tfrec_fp}"

    t_sr_data = {"tces": [], "sectors": defaultdict(_sector_dd)}

    try:
        for str_record in tf.data.TFRecordDataset(tfrec_fp).as_numpy_iterator():
            example = tf.train.Example()
            example.ParseFromString(str_record)

            uid = example.features.feature["uid"].bytes_list.value[0].decode("utf-8")
            tce_uid = uid.split("_")[0]
            _target = tce_uid.split("-")[0]
            _sector_run = tce_uid.split("S")[-1]

            if _target != target or _sector_run != sector_run:
                continue

            sector = (
                example.features.feature["sector"].bytes_list.value[0].decode("utf-8")
            )
            disposition = (
                example.features.feature["disposition"]
                .bytes_list.value[0]
                .decode("utf-8")
            )
            tce_time0bk = example.features.feature["tce_time0bk"].float_list.value[0]
            tce_period = example.features.feature["tce_period"].float_list.value[0]
            tce_duration = example.features.feature["tce_duration"].float_list.value[0]
            t = example.features.feature["t"].float_list.value[0]
            label = (
                example.features.feature["label"].bytes_list.value[0].decode("utf-8")
            )
            assert label in ("0", "1"), f"Invalid label: {label}"

            if not any(tce["tce_uid"] == tce_uid for tce in t_sr_data["tces"]):
                t_sr_data["tces"].append(
                    {
                        "tce_uid": tce_uid,
                        "tce_time0bk": tce_time0bk,
                        "tce_period": tce_period,
                        "tce_duration": tce_duration,
                        "disposition": disposition,
                    }
                )
            # TODO: UPDATE
            flux = list(
                example.features.feature["flux"].float_list.value
            )  # Ensures JSON-serializable
            flux_norm = list(
                example.features.feature["flux_norm"].float_list.value
            )  # Ensures JSON-serializable
            flux_quality = list(
                example.features.feature["flux_quality"].float_list.value
            )  # Ensures JSON-serializable

            diff_imgs = {}
            for img_feature in ["diff_img", "oot_img", "snr_img", "target_img"]:
                tensor_bytes = example.features.feature[img_feature].bytes_list.value[0]
                parsed_tensor = tf.io.parse_tensor(tensor_bytes, out_type=tf.float32)
                reshaped = tf.reshape(parsed_tensor, (33, 33))
                diff_imgs[img_feature] = (
                    reshaped.numpy().tolist()
                )  # List of lists (33x33)

            # Store in examples_by_tce
            t_sr_data["sectors"][sector]["examples_by_tce"][tce_uid].append(
                {
                    "uid": str(uid),
                    "t": float(t),
                    "flux": flux,
                    "flux_norm": flux_norm,
                    "flux_quality": flux_quality,
                    "diff_imgs": diff_imgs,  # dict of 33x33 list-of-lists
                    "label": str(label),
                }
            )

    except Exception as e:
        raise RuntimeError(f"Failed parsing TFRecord: {e}")

    for sector, sector_data in t_sr_data["sectors"].items():
        try:
            search_lc_res = lk.search_lightcurve(
                target=f"tic{target}",
                mission="TESS",
                author=("TESS-SPOC", "SPOC"),
                exptime=120,
                cadence="long",
                sector=[int(sector)],
            )
            lcf = search_lc_res[0].download(
                download_dir=None, quality_bitmask="default", flux_column="pdcsap_flux"
            )
            time = np.array(lcf.time.value)
            flux = np.array(lcf.flux.value)
            sector_data["lightcurve"] = {
                "time": time.tolist(),
                "flux": flux.tolist(),
                # "base_fig": go.Figure(
                #     go.Scatter(x=time, y=flux, mode="lines", name="Flux")
                # ),
            }
            # sector_data["lightcurve"]["base_fig"].update_layout(
            #     title=f"Sector {sector}", xaxis_title="Time (BTJD)", yaxis_title="Flux"
            # )
        except Exception as e:
            raise RuntimeError(f"Failed to load lightcurve for sector {sector}: {e}")

    return t_sr_data


def flux_preview_to_base64(time, flux):
    fig = go.Figure(go.Scatter(x=time, y=flux, mode="lines"))
    fig.update_layout(
        width=200,
        height=100,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(showticklabels=False),
        yaxis=dict(showticklabels=False),
    )
    buffer = io.BytesIO()
    fig.write_image(buffer, format="png")  # Requires kaleido installed
    encoded = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{encoded}"


import io
import base64
import plotly.graph_objects as go
import numpy as np


def gen_ex_hovertemplate(ex_data, tce_data, n_durations_window=5):
    t = ex_data["t"]
    flux = np.array(ex_data["flux"])
    duration = float(tce_data["tce_duration"])  # hours
    duration /= 24  # days

    label = ex_data.get("label", "?")

    time = np.linspace(
        t - (n_durations_window * duration) / 2,
        t + (n_durations_window * duration) / 2,
        len(flux),
    )

    # Plot preview
    fig = go.Figure(go.Scatter(x=time, y=flux, mode="lines"))
    fig.update_layout(
        width=220,
        height=120,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(showticklabels=False),
        yaxis=dict(showticklabels=False),
    )

    # Encode to base64
    buffer = io.BytesIO()
    fig.write_image(buffer, format="png")
    encoded = base64.b64encode(buffer.getvalue()).decode()
    img_url = f"data:image/png;base64,{encoded}"

    # Hovertemplate HTML
    hovertemplate = f"""
    Duration = ahhhh<br>
    <b>Label {label}</b><br>
    t = {t:.4f}<br>
    Duration = {duration:.2f}h<br>
    <img src="{img_url}" width="200"/><extra></extra>
    """
    return hovertemplate


@lru_cache(maxsize=128)
def get_preview_fig(tce_uid: str, t: float, flux: tuple, duration: float):
    # print(f"building preview for {tce_uid} @ {t}")
    time = np.linspace(t - 5 * duration, t + 5 * duration, len(flux))
    fig = go.Figure(go.Scatter(x=time, y=np.array(flux), mode="markers"))
    fig.update_layout(
        title=f"Preview: TCE {tce_uid} @ t = {t:.4f} Flux",
        height=200,
        margin=dict(l=20, r=20, t=30, b=30),
    )
    return fig


# # helpers
# def _sector_dd():  # needed for pickling w/ lambda
#     return {"lightcurve": None, "examples_by_tce": defaultdict(list)}


# # primary utils


# def search_index_as_json_stream(
#     index_json_fp: Union[Path, str], target_id: str, sector_run: str
# ) -> Path:
#     with open(index_json_fp, "r") as file:
#         # Parse the JSON objects one by one
#         parser = ijson.items(file, "item")

#         for item in parser:
#             # Process each JSON object as needed
#             print(item)


# def load_index(index_json_fp: Union[Path, str]):
#     with open(index_json_fp, "r") as fp:
#         index = json.load(fp)
#     return index


# def get_tfrec_fp_from_index(index: dict, target: str, sector_run: str):
#     return index.get(target, "").get(sector_run, "")


# def build_tfrec_cache():

#     pass


# def build_index(
#     tfrec_dir: Union[Path, str],
#     index_json_fp: Union[Path, str],
#     tfrec_glob_pattern: str = "*.tfrecord",
# ):
#     tfrec_index = {}

#     if isinstance(tfrec_dir, str):
#         tfrec_dir = Path(tfrec_dir)
#     assert isinstance(
#         tfrec_dir, Path
#     ), "ERROR: could not convert input type of tfrec_dir to Path"

#     tfrec_fp_gen = tfrec_dir.glob(tfrec_glob_pattern)

#     for tfrec_fp in tfrec_fp_gen:
#         try:
#             tfrec_dataset = tf.data.TFRecordDataset(tfrec_fp)
#             for str_record in tfrec_dataset.as_numpy_iterator():
#                 example = tf.train.Example()
#                 example.ParseFromString(str_record)

#                 # get example info
#                 uid = (
#                     example.features.feature["uid"].bytes_list.value[0].decode("utf-8")
#                 )

#                 tce_uid = uid.split("_")[0]
#                 target_id = tce_uid.split("-")[0]
#                 sector_run = tce_uid.split("S")[-1]

#                 if target_id not in tfrec_index:
#                     tfrec_index[target_id] = {}
#                 if sector_run not in tfrec_index[target_id]:
#                     tfrec_index[target_id][sector_run] = str(tfrec_fp)
#         except Exception:
#             continue

#     with open(index_json_fp, "w") as fp:
#         json.dump(tfrec_index, fp)


# def load_t_sr_data(
#     target: str,
#     sector_run: str,
#     tfrec_fp: Union[str | Path],
#     # lc_download_dir: Union[str | Path],
# ):
#     """
#     NOTE: time and flux from lightcurve non interpolated or populated for now.
#         also, not detrended for now. does not exactly match pipeline, but is sufficient
#         for test visualization purposes

#     """
#     if isinstance(tfrec_fp, str):
#         tfrec_fp = Path(tfrec_fp)
#     assert isinstance(
#         tfrec_fp, Path
#     ), f"ERROR: could not convert input type of tfrec_fp - {tfrec_fp} - to Path"
#     assert tfrec_fp.exists(), f"ERROR: tfrec_fp - {tfrec_fp} - does not exist"

#     t_sr_data = {"tces": [], "sectors": defaultdict(_sector_dd)}

#     tfrec_dataset = tf.data.TFRecordDataset(tfrec_fp)

#     for str_record in tfrec_dataset.as_numpy_iterator():
#         example = tf.train.Example()
#         example.ParseFromString(str_record)

#         # get example info
#         uid = example.features.feature["uid"].bytes_list.value[0].decode("utf-8")

#         tce_uid = uid.split("_")[0]
#         _target = tce_uid.split("-")[0]
#         _sector_run = tce_uid.split("S")[-1]

#         if _target != target or _sector_run != sector_run:  # skip other t, sr pairs
#             continue

#         sector = example.features.feature["sector"].bytes_list.value[0].decode("utf-8")
#         disposition = (
#             example.features.feature["disposition"].bytes_list.value[0].decode("utf-8")
#         )
#         tce_time0bk = example.features.feature["tce_time0bk"].float_list.value[0]
#         tce_period = example.features.feature["tce_period"].float_list.value[0]
#         tce_duration = example.features.feature["tce_duration"].float_list.value[0]
#         t = example.features.feature["t"].float_list.value[0]
#         label = example.features.feature["label"].bytes_list.value[0].decode("utf-8")
#         assert label in (
#             "0",
#             "1",
#         ), f"ERROR: label: {label} of unexpected type in tfrecord"

#         if not any(tce_info["tce_uid"] == tce_uid for tce_info in t_sr_data["tces"]):
#             t_sr_data["tces"].append(
#                 {
#                     "tce_uid": tce_uid,
#                     "tce_time0bk": tce_time0bk,
#                     "tce_period": tce_period,
#                     "tce_duration": tce_duration,
#                     "disposition": disposition,
#                 }
#             )

#         # flux = np.array(example.features.feature["flux"].float_list.value).tolist()

#         # diff_imgs = {}
#         # for img_feature in ["diff_img", "oot_img", "snr_img", "target_img"]:
#         #     example_img_feature = tf.reshape(
#         #         tf.io.parse_tensor(
#         #             example.features.feature[img_feature].bytes_list.value[0],
#         #             tf.float32,
#         #         ),
#         #         (33, 33),
#         #     ).numpy()
#         #     diff_imgs[img_feature] = example_img_feature.tolist()

#         flux = None
#         diff_imgs = {}

#         t_sr_data["sectors"][sector]["examples_by_tce"][tce_uid].append(
#             {"uid": uid, "t": t, "flux": flux, "diff_imgs": diff_imgs, "label": label}
#         )

#     #  add sector lightcurve and basefig
#     for sector, sector_data in t_sr_data["sectors"].items():
#         search_lc_res = lk.search_lightcurve(
#             target=f"tic{target}",
#             mission="TESS",
#             author=("TESS-SPOC", "SPOC"),
#             exptime=120,
#             cadence="long",
#             sector=[int(sector)],
#         )
#         lcf = search_lc_res[0].download(
#             download_dir=None, quality_bitmask="default", flux_column="pdcsap_flux"
#         )
#         time = np.array(lcf.time.value)
#         flux = np.array(lcf.flux.value)
#         t_sr_data["sectors"][sector]["lightcurve"] = {
#             "time": time.tolist(),
#             "flux": flux.tolist(),
#         }

#         # Precompute base figure
#         base_fig = go.Figure(go.Scatter(x=time, y=flux, mode="lines", name="Flux"))
#         base_fig.update_layout(
#             title=f"Sector {sector}", xaxis_title="Time (BTJD)", yaxis_title="Flux"
#         )
#         t_sr_data["sectors"][sector]["lightcurve"]["base_fig"] = base_fig

#     return t_sr_data


# def load_t_sr_tce_list_from_tfrec(
#     tfrec_fp: Union[str | Path], target: str, sector_run: str
# ):
#     if isinstance(tfrec_fp, str):
#         tfrec_fp = Path(tfrec_fp)
#     assert isinstance(
#         tfrec_fp, Path
#     ), f"ERROR: could not convert input type of tfrec_fp - {tfrec_fp} - to Path"
#     assert tfrec_fp.exists(), f"ERROR: tfrec_fp - {tfrec_fp} - does not exist"

#     tce_list = []

#     tfrec_dataset = tf.data.TFRecordDataset(tfrec_fp)
#     for str_record in tfrec_dataset.as_numpy_iterator():
#         example = tf.train.Example()
#         example.ParseFromString(str_record)

#         # get example info
#         uid = example.features.feature["uid"].bytes_list.value[0].decode("utf-8")

#         tce_uid = uid.split("_")[0]
#         t = tce_uid.split("-")[0]
#         sr = tce_uid.split("S")[-1]

#         if t == target and sr == sector_run:
#             if tce_uid not in tce_list:
#                 tce_list.append(tce_uid)

#     return tce_list


# def encode_tfrec_bytes_img_as_base_64(bytes_img):
#     base64_img = base64.b64encode(bytes_img).decode("utf-8")


# def decode_base64_img(base64_img, dimensions):
#     pass


# # def generate_preview_plot(target, sector_run, storage_dir, )


def detect_set(path_str: str) -> str | None:
    """
    Returns 'train', 'val', or 'test' if one of those substrings
    appears in path_str (case-insensitive), else None.
    """
    lower = path_str.lower()
    for set in ("train", "val", "test"):
        if set in lower:
            return set
    return None


# ─────────────── Styling ───────────────
SECTOR_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]


def get_sector_color(sector):
    try:
        return SECTOR_COLORS[int(sector) % len(SECTOR_COLORS)]
    except:
        return "#000000"


def get_color(label):
    return "green" if label == "1" else "red"
