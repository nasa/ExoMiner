from pathlib import Path
from typing import Union
import tensorflow as tf
from collections import defaultdict
import json
import ijson
import lightkurve as lk
import numpy as np
import base64
import plotly.graph_objects as go


# helpers
def _sector_dd():  # needed for pickling w/ lambda
    return {"lightcurve": None, "examples_by_tce": defaultdict(list)}


# primary utils


def search_index_as_json_stream(
    index_json_fp: Union[Path, str], target_id: str, sector_run: str
) -> Path:
    with open(index_json_fp, "r") as file:
        # Parse the JSON objects one by one
        parser = ijson.items(file, "item")

        for item in parser:
            # Process each JSON object as needed
            print(item)


def load_index(index_json_fp: Union[Path, str]):
    with open(index_json_fp, "r") as fp:
        index = json.load(fp)
    return index


def get_tfrec_fp_from_index(index: dict, target: str, sector_run: str):
    return index.get(target, "").get(sector_run, "")


def build_tfrec_cache():

    pass


def build_index(
    tfrec_dir: Union[Path, str],
    index_json_fp: Union[Path, str],
    tfrec_glob_pattern: str = "*.tfrecord",
):
    tfrec_index = {}

    if isinstance(tfrec_dir, str):
        tfrec_dir = Path(tfrec_dir)
    assert isinstance(
        tfrec_dir, Path
    ), "ERROR: could not convert input type of tfrec_dir to Path"

    tfrec_fp_gen = tfrec_dir.glob(tfrec_glob_pattern)

    for tfrec_fp in tfrec_fp_gen:
        try:
            tfrec_dataset = tf.data.TFRecordDataset(tfrec_fp)
            for str_record in tfrec_dataset.as_numpy_iterator():
                example = tf.train.Example()
                example.ParseFromString(str_record)

                # get example info
                uid = (
                    example.features.feature["uid"].bytes_list.value[0].decode("utf-8")
                )

                tce_uid = uid.split("_")[0]
                target_id = tce_uid.split("-")[0]
                sector_run = tce_uid.split("S")[-1]

                if target_id not in tfrec_index:
                    tfrec_index[target_id] = {}
                if sector_run not in tfrec_index[target_id]:
                    tfrec_index[target_id][sector_run] = str(tfrec_fp)
        except Exception:
            continue

    with open(index_json_fp, "w") as fp:
        json.dump(tfrec_index, fp)


def load_t_sr_data(
    target: str,
    sector_run: str,
    tfrec_fp: Union[str | Path],
    # lc_download_dir: Union[str | Path],
):
    """
    NOTE: time and flux from lightcurve non interpolated or populated for now.
        also, not detrended for now. does not exactly match pipeline, but is sufficient
        for test visualization purposes

    """
    if isinstance(tfrec_fp, str):
        tfrec_fp = Path(tfrec_fp)
    assert isinstance(
        tfrec_fp, Path
    ), f"ERROR: could not convert input type of tfrec_fp - {tfrec_fp} - to Path"
    assert tfrec_fp.exists(), f"ERROR: tfrec_fp - {tfrec_fp} - does not exist"

    t_sr_data = {"tces": [], "sectors": defaultdict(_sector_dd)}

    tfrec_dataset = tf.data.TFRecordDataset(tfrec_fp)

    for str_record in tfrec_dataset.as_numpy_iterator():
        example = tf.train.Example()
        example.ParseFromString(str_record)

        # get example info
        uid = example.features.feature["uid"].bytes_list.value[0].decode("utf-8")

        tce_uid = uid.split("_")[0]
        _target = tce_uid.split("-")[0]
        _sector_run = tce_uid.split("S")[-1]

        if _target != target or _sector_run != sector_run:  # skip other t, sr pairs
            continue

        sector = example.features.feature["sector"].bytes_list.value[0].decode("utf-8")
        disposition = (
            example.features.feature["disposition"].bytes_list.value[0].decode("utf-8")
        )
        tce_time0bk = example.features.feature["tce_time0bk"].float_list.value[0]
        tce_period = example.features.feature["tce_period"].float_list.value[0]
        tce_duration = example.features.feature["tce_duration"].float_list.value[0]
        t = example.features.feature["t"].float_list.value[0]

        if not any(tce_info["tce_uid"] == tce_uid for tce_info in t_sr_data["tces"]):
            t_sr_data["tces"].append(
                {
                    "tce_uid": tce_uid,
                    "tce_time0bk": tce_time0bk,
                    "tce_period": tce_period,
                    "tce_duration": tce_duration,
                    "disposition": disposition,
                }
            )

        # flux = np.array(example.features.feature["flux"].float_list.value).tolist()

        # diff_imgs = {}
        # for img_feature in ["diff_img", "oot_img", "snr_img", "target_img"]:
        #     example_img_feature = tf.reshape(
        #         tf.io.parse_tensor(
        #             example.features.feature[img_feature].bytes_list.value[0],
        #             tf.float32,
        #         ),
        #         (33, 33),
        #     ).numpy()
        #     diff_imgs[img_feature] = example_img_feature.tolist()

        flux = None
        diff_imgs = {}

        t_sr_data["sectors"][sector]["examples_by_tce"][tce_uid].append(
            {"uid": uid, "t": t, "flux": flux, "diff_imgs": diff_imgs}
        )

    #  add sector lightcurve and basefig
    for sector, sector_data in t_sr_data["sectors"].items():
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
        t_sr_data["sectors"][sector]["lightcurve"] = {
            "time": time.tolist(),
            "flux": flux.tolist(),
        }

        # Precompute base figure
        base_fig = go.Figure(go.Scatter(x=time, y=flux, mode="lines", name="Flux"))
        base_fig.update_layout(
            title=f"Sector {sector}", xaxis_title="Time (BTJD)", yaxis_title="Flux"
        )
        t_sr_data["sectors"][sector]["lightcurve"]["base_fig"] = base_fig

    return t_sr_data


def load_t_sr_tce_list_from_tfrec(
    tfrec_fp: Union[str | Path], target: str, sector_run: str
):
    if isinstance(tfrec_fp, str):
        tfrec_fp = Path(tfrec_fp)
    assert isinstance(
        tfrec_fp, Path
    ), f"ERROR: could not convert input type of tfrec_fp - {tfrec_fp} - to Path"
    assert tfrec_fp.exists(), f"ERROR: tfrec_fp - {tfrec_fp} - does not exist"

    tce_list = []

    tfrec_dataset = tf.data.TFRecordDataset(tfrec_fp)
    for str_record in tfrec_dataset.as_numpy_iterator():
        example = tf.train.Example()
        example.ParseFromString(str_record)

        # get example info
        uid = example.features.feature["uid"].bytes_list.value[0].decode("utf-8")

        tce_uid = uid.split("_")[0]
        t = tce_uid.split("-")[0]
        sr = tce_uid.split("S")[-1]

        if t == target and sr == sector_run:
            if tce_uid not in tce_list:
                tce_list.append(tce_uid)

    return tce_list


def encode_tfrec_bytes_img_as_base_64(bytes_img):
    base64_img = base64.b64encode(bytes_img).decode("utf-8")


def decode_base64_img(base64_img, dimensions):
    pass
