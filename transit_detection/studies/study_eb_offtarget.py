"""
Investigate how the model scores of EBs change with a proxy/estimate of transit source offset.
"""

# 3rd party
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import numpy as np

# %%

tce_tbl = pd.read_csv(
    "/Users/msaragoc/Projects/exoplanet_transit_classification/data/ephemeris_tables/tess/tess_spoc_2min/tess_2min_tces_dv_s1-s88_3-27-2025_1316_label.csv"
)

# %%

pred_dir = Path(
    "/Users/msaragoc/Projects/exoplanet_transit_classification/experiments/transit_detection/eb_disp_offtarget_7-22-2025/predict_model_TESS_exoplanet_dataset_07-11-2025_no_ntp_no_detrend_split_norm_filtered"
)
pred_tbls_fps = pred_dir.glob("preds_*.csv")

pred_tbls_lst = []
for pred_tbl_fp in pred_tbls_fps:

    pred_tbl = pd.read_csv(pred_tbl_fp)
    pred_tbl["dataset"] = pred_tbl_fp.stem.split("_")[1]
    pred_tbls_lst.append(pred_tbl)

pred_df = pd.concat(pred_tbls_lst, axis=0, ignore_index=True)

tce_cols = [
    "tce_dikco_msky",
    "tce_dikco_msky_err",
    "tce_dikco_msky_original",
    "tce_dikco_msky_err_original",
]

pred_df = pred_df.rename(columns={"uid": "example_uid"})
pred_df["uid"] = pred_df.apply(lambda row: row["example_uid"].split("_")[0], axis=1)

pred_df = pred_df.merge(
    tce_tbl[["uid"] + tce_cols], on=["uid"], how="left", validate="many_to_one"
)

pred_df.to_csv(pred_dir.parent / "preds_dataset.csv", index=False)

# %%

pred_df_ebs = pred_df.loc[pred_df["disposition"] == "EB"]
pred_df_ebs = pred_df_ebs.loc[pred_df["label"] == 1]

tce_feature_name = "tce_dikco_msky_original"

pred_df_ebs[f"{tce_feature_name}_ratio"] = pred_df_ebs.apply(
    lambda row: row["tce_dikco_msky_err_original"]
    / (row["tce_dikco_msky_original"] + 1e-12),
    axis=1,
)
unc_thr = 1 / 3
pred_df_ebs = pred_df_ebs.loc[pred_df_ebs[f"{tce_feature_name}_ratio"] < unc_thr]

bins_offset = np.linspace(0, 42, 43)

tr_off_thr = 0.2 * 21

f, ax = plt.subplots(figsize=(12, 6))
ax.hist(pred_df_ebs[tce_feature_name], bins=bins_offset, edgecolor="k")
ax.set_xticks(bins_offset)
ax.set_xlabel(f"{tce_feature_name} [arcsec]")
ax.set_ylabel("Count")
ax.set_yscale("log")
ax.set_xticks(bins_offset)
ax.set_xlim(bins_offset[[0, -1]])
ax.axvline(tr_off_thr, color="r", linestyle="dashed")
ax.set_title(
    "Number of transit-window EB examples after cut-off\n"
    rf"Transit Source offset and error thresholds: $thr={tr_off_thr}$ arcsec $\pm$ $error={unc_thr:.3f}\sigma$"
    f'\n{(pred_df_ebs["tce_dikco_msky_err_original"] < tr_off_thr).sum()} examples'
)
f.tight_layout()
f.savefig(
    pred_dir.parent
    / f"hist_{tce_feature_name}_{unc_thr:.3f}_scores_EB_pos_examples.png"
)

# %%

pred_df_ebs = pred_df.loc[pred_df["disposition"] == "EB"]
pred_df_ebs = pred_df_ebs.loc[pred_df["label"] == 1]

f, ax = plt.subplots()
ax.scatter(pred_df_ebs[tce_feature_name], pred_df_ebs["pred_prob"], s=8)
ax.set_xlabel(f"{tce_feature_name}")
ax.set_ylabel("Model Score")
f.savefig(pred_dir.parent / f"scatter_{tce_feature_name}_scores_EB_pos_examples.png")

scores_bins = np.linspace(0, 1, 21)
mean_values, max_values, min_values, median_values, std_values = [], [], [], [], []
for s_bin, e_bin in zip(scores_bins[:-1], scores_bins[1:]):
    aux_df = pred_df_ebs.loc[
        ((pred_df_ebs["pred_prob"] > s_bin) & (pred_df_ebs["pred_prob"] < e_bin))
    ]
    mean_values.append(aux_df[tce_feature_name].mean())
    median_values.append(aux_df[tce_feature_name].median())
    max_values.append(aux_df[tce_feature_name].max())
    min_values.append(aux_df[tce_feature_name].min())
    std_values.append(aux_df[tce_feature_name].std())

f, ax = plt.subplots(figsize=(12, 6))
ax.bar(
    scores_bins[:-1],
    mean_values,
    width=np.diff(scores_bins),
    yerr=std_values,
    edgecolor="green",
    align="edge",
    facecolor="none",
    label="mean",
    linewidth=2,
    bottom=0,
)
# ax.bar(scores_bins[:-1], std_values, width=np.diff(scores_bins), edgecolor='green', align='edge', facecolor='none', label='mean+-std', linewidth=2)
ax.bar(
    scores_bins[:-1],
    median_values,
    width=np.diff(scores_bins),
    edgecolor="blue",
    align="edge",
    facecolor="none",
    label="median",
    linewidth=2,
)
# ax.bar(scores_bins[:-1], min_values, width=np.diff(scores_bins), edgecolor='red', align='edge', facecolor='none', label='min', linewidth=2)
# ax.bar(scores_bins[:-1], max_values, width=np.diff(scores_bins), edgecolor='k', align='edge', facecolor='none', label='max', linewidth=2)
ax.set_ylabel(f"{tce_feature_name} [arcsec]")
ax.set_xlabel("Model Score")
ax.set_xticks(scores_bins)
ax.set_xlim(scores_bins[[0, -1]])
# ax.set_yscale('log')
ax.set_ylim(bottom=0)
ax.legend()
f.tight_layout()
f.savefig(pred_dir.parent / f"bar_stats_{tce_feature_name}_scores_EB_pos_examples.png")
