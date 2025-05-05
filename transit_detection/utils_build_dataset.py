"""
Utility functions used to build the dataset.
"""

# 3rd party
import tensorflow as tf
import pandas as pd
import numpy as np

# local
from src_preprocessing.tf_util import example_util


INT64_FEATURES = [
    "target_id",
    "tce_plnt_num",
    "numberOfPlanets",
    "tce_num_transits",
    "tce_num_transits_obs",
    "mag_cat",
    "mission",
]
FLOAT_FEATURES = [
    "tce_max_sngle_ev",
    "tce_max_mult_ev",
    "tce_model_chisq",
    "tce_depth",
    "tce_period",
    "tce_time0bk",
    "tce_duration",
    "tce_prad",
    "tce_sma",
    "tce_impact",
    "boot_fap",
    "tce_ror",
    "tce_dor",
    "tce_model_snr",
    "tce_dikco_msky_err",
    "tce_dikco_msky",
    "tce_dicco_msky",
    "tce_dicco_msky_err",
    "mag",
    "tce_sdens",
    "tce_sradius",
    "tce_steff",
    "tce_slogg",
    "tce_smet",
    "tce_eqt",
    "tce_robstat",
    "tce_maxmes",
    "tce_maxmesd",
    "tce_bin_oedp_stat",
    "tce_shortper_stat",
    "tce_longper_stat",
    "planetToiId",
    "tce_incl",
    "tce_eccen",
    "tce_longp",
    "ra",
    "dec",
    "tce_albedo",
    "tce_albedo_stat",
    "tce_ptemp",
    "tce_ptemp_stat",
    "tce_insol",
    "tce_cap_stat",
    "tce_hap_stat",
    "wst_depth",
    "ruwe",
    "mag_shift",
]
BYTES_FEATURES = ["disposition", "disposition_source", "matched_object"]

FEATURES_TO_TBL = [
    "target_id",
    "tce_plnt_num",
    "numberOfPlanets",
    "tce_num_transits",
    "tce_num_transits_obs",
    "tce_max_sngle_ev",
    "tce_max_mult_ev",
    "tce_model_chisq",
    "tce_depth",
    "tce_period",
    "tce_time0bk",
    "tce_duration",
    "tce_prad",
    "tce_sma",
    "tce_impact",
    "boot_fap",
    "tce_ror",
    "tce_dor",
    "tce_model_snr",
    "tce_dikco_msky_err",
    "tce_dikco_msky",
    "tce_dicco_msky",
    "tce_dicco_msky_err",
    "mag",
    "tce_sdens",
    "tce_sradius",
    "tce_steff",
    "tce_slogg",
    "tce_smet",
    "tce_eqt",
    "tce_robstat",
    "tce_maxmes",
    "tce_maxmesd",
    "tce_bin_oedp_stat",
    "tce_shortper_stat",
    "tce_longper_stat",
    "planetToiId",
    "tce_incl",
    "tce_eccen",
    "tce_longp",
    "ra",
    "dec",
    "tce_albedo",
    "tce_albedo_stat",
    "tce_ptemp",
    "tce_ptemp_stat",
    "tce_insol",
    "tce_cap_stat",
    "tce_hap_stat",
    "wst_depth",
    "ruwe",
    "mag_shift",
    "disposition",
    "disposition_source",
    "matched_object",
]

POSITIVE_LABELS = ["KP", "CP", "EB"]


def serialize_set_examples_for_tce(data_for_tce):
    """Serializes data of a set of examples for a TCE into an example that can be added to a TFRecord dataset.

    Args:
        data_for_tce: dict, data for a TCE. Keys are "tce_uid", "tce_data", and "sector", and map to the unique TCE ID,
        TCE data from DV table, and preprocessed flux and difference image data, respectively. "sectors" maps to a
        list of dictionaries with keys "sector", "transit_examples" and "not_transit_examples", the latter two holding
        data for those two cases. For each subdictionary of examples, there are keyes "sector", "flux", "t",
        "diff_img", "it_img", "oot_img", "snr_img", "target_img", and "target_pos" with the respective data for these
        examples.

    Returns:
        examples_for_tce, serialized string of data for examples of a TCE
    """

    examples_for_tce = []
    for data_for_sector in data_for_tce["sectors"]:  # iterate over sector data

        for example_type in ["transit_examples", "not_transit_examples"]:

            # iterate over examples
            for example_i in range(len(data_for_sector[example_type]["t"])):

                example = tf.train.Example()

                # set example unique id
                example_util.set_bytes_feature(
                    example,
                    "uid",
                    [
                        f'{data_for_tce["tce_uid"]}_t_{data_for_sector[example_type]["t"][example_i]}'
                    ],
                )

                # set label
                example_util.set_bytes_feature(
                    example,
                    "label",
                    [
                        (
                            "1"
                            if data_for_tce["disposition"] in POSITIVE_LABELS
                            and example_type == "transit_examples"
                            else "0"
                        )
                    ],
                )
                # set auxiliary data
                example_util.set_bytes_feature(
                    example, "sector", [data_for_sector["sector"]]
                )
                for feature_name in INT64_FEATURES:
                    example_util.set_int64_feature(
                        example, feature_name, [data_for_tce["tce_info"][feature_name]]
                    )
                for feature_name in FLOAT_FEATURES:
                    example_util.set_float_feature(
                        example, feature_name, [data_for_tce["tce_info"][feature_name]]
                    )
                for feature_name in BYTES_FEATURES:
                    example_util.set_bytes_feature(
                        example, feature_name, [data_for_tce["tce_info"][feature_name]]
                    )

                # set features for example
                example_util.set_float_feature(
                    example, "t", [data_for_sector[example_type]["t"][example_i]]
                )
                example_util.set_float_feature(
                    example, "flux", data_for_sector[example_type]["flux"][example_i]
                )

                for feature_name in ["oot_img", "diff_img", "snr_img", "target_img"]:
                    example_util.set_tensor_feature(
                        example,
                        feature_name,
                        data_for_sector[example_type][feature_name][example_i],
                    )

                # set "target_pos feature" to
                example_util.set_int64_feature(
                    example,
                    "target_pos",
                    data_for_sector[example_type]["target_pos"][example_i],
                )

                examples_for_tce.append(example.SerializeToString())

    return examples_for_tce


def write_data_to_auxiliary_tbl(data_to_tfrec, tfrec_fp):
    """Writes auxiliary data for the TCEs with examples in TFRecord `tfrec_fp`.

    Args:
        data_to_tfrec: list, each item is a dictionary of data for a TCE
        tfrec_fp: Path, TFRecord file path

    Returns: data_tbl, pandas DataFrame with auxiliary data for the TCE with examples processed into the TFRecord in
    `tfrec_fp`

    """

    data_to_tbl = {
        field_name: []
        for field_name in [
            "tce_uid",
            "n_transit_examples",
            "n_not_transit_examples",
            "tfrec_fn",
        ]
    }
    data_to_tbl.update({field_name: [] for field_name in FEATURES_TO_TBL})
    for data_for_tce in data_to_tfrec:

        data_to_tbl["tce_uid"].append(data_for_tce["tce_uid"])
        data_to_tbl["n_transit_examples"].append(
            np.sum(
                [
                    len(data_for_sector["transit_examples"]["t"])
                    for data_for_sector in data_for_tce["sectors"]
                ]
            )
        )
        data_to_tbl["n_not_transit_examples"].append(
            np.sum(
                [
                    len(data_for_sector["not_transit_examples"]["t"])
                    for data_for_sector in data_for_tce["sectors"]
                ]
            )
        )

        for feature_name in FEATURES_TO_TBL:
            data_to_tbl[feature_name].append(data_for_tce["tce_info"][feature_name])

        data_to_tbl["tfrec_fn"].append(tfrec_fp.name)

    data_tbl = pd.DataFrame(data_to_tbl)

    return data_tbl
