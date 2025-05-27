"""
Check whether there is a trend in the scores.
"""

# 3rd party
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import lightkurve as lk

# %%

plot_dir = Path(
    "/Users/msaragoc/Projects/exoplanet_transit_classification/experiments/transit_detection/study_model_scores_05-20-2025_0959"
)

scores_tbls_dir = Path("/Users/msaragoc/Downloads/model_predictions_20percent_dataset")
scores_tbl_fp = scores_tbls_dir / "preds_train.csv"

tce_tbl_fp = Path(
    ""
    "/nobackup/jochoa4/work_dir/data/tables/tess_2min_tces_dv_s1-s68_all_msectors_11-29-2023_2157_newlabels_nebs_npcs_bds_ebsntps_to_unks.csv"
)

# tce_tbl_fp = Path(
#     "/Users/msaragoc/Projects/exoplanet_transit_classification/data/ephemeris_tables/tess/DV_SPOC_mat_files/preprocessing_tce_tables/09-25-2023_1608/tess_2min_tces_dv_s1-s68_all_msectors_11-29-2023_2157_newlabels_nebs_npcs_bds_ebsntps_to_unks_sg1master_allephemmatches_exofoptois.csv"
# )

plot_dir.mkdir(parents=True, exist_ok=True)

tce_tbl = pd.read_csv(tce_tbl_fp)
tce_tbl = tce_tbl.rename(columns={"uid": "tce_uid", "label": "disposition"})

train_tbl = pd.read_csv(scores_tbls_dir / "preds_train.csv")


def _get_tce_uid_from_examples(x):

    tce_uid = x["uid"].split("_")[0]

    return tce_uid


train_tbl["tce_uid"] = train_tbl.apply(_get_tce_uid_from_examples, axis=1)

train_tbl = train_tbl.merge(
    tce_tbl[["tce_uid", "tce_max_mult_ev"]],
    on="tce_uid",
    how="left",
    validate="many_to_one",
)

# %% plot histogram scores

bins_scores = np.linspace(0, 1, 11)

for disp in train_tbl["disposition"].unique():

    train_tbl_disp = train_tbl.loc[train_tbl["disposition"] == disp]

    transit_window_examples = train_tbl_disp.loc[train_tbl_disp["label"] == 1]
    not_transit_window_examples = train_tbl_disp.loc[train_tbl_disp["label"] == 0]

    f, ax = plt.subplots()

    if len(transit_window_examples) > 0:
        ax.hist(
            transit_window_examples["raw_pred"],
            bins_scores,
            histtype="step",
            label="Transit Windows",
            zorder=2,
            linestyle="dashed",
            color="#1f77b4",
        )
    if len(not_transit_window_examples) > 0:
        ax.hist(
            not_transit_window_examples["raw_pred"],
            bins_scores,
            histtype="step",
            label="Not-Transit Windows",
            zorder=1,
            color="#ff7f0e",
        )

    ax.set_ylabel("Example Count")
    ax.set_xlabel("Model Score")
    ax.set_yscale("log")
    ax.set_title(f"{disp}")
    ax.legend()
    ax.set_xlim(bins_scores[[0, -1]])
    f.savefig(plot_dir / f"hist_model_scores_{disp}.png")
    plt.show()

# %% plot score as function of MES

bins_mes = np.logspace(np.log10(7.1), 4, 50)
bins_scores = np.linspace(0, 1, 11)

for disp in train_tbl["disposition"].unique():
    for label in train_tbl["label"].unique():

        train_tbl_disp = train_tbl.loc[
            ((train_tbl["disposition"] == disp) & (train_tbl["label"] == label))
        ]
        if len(train_tbl_disp) == 0:
            continue

        f, ax = plt.subplots()

        # ax.scatter(train_tbl_disp['tce_max_mult_ev'], train_tbl_disp['raw_pred'], s=8, alpha=0.01)
        # ax.set_ylim([0, 1])
        # ax.set_xlim(left=7.1)

        im = ax.hist2d(
            train_tbl_disp["tce_max_mult_ev"],
            train_tbl_disp["raw_pred"],
            bins=[bins_mes, bins_scores],
            cmap="jet",
            norm=LogNorm(),
        )[3]
        ax.set_ylim([0, 1])
        ax.set_xlim(left=7.1)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax, label="Example Count")

        ax.set_xscale("log")

        ax.set_ylabel("Model Score")
        ax.set_xlabel("TCE MES")
        ax.set_title(f"Disposition {disp} | Label {label}")
        f.savefig(plot_dir / f"hist2d_model_scores_tcemes_{disp}.png")
        plt.show()

# %% create table that averages scores for same TCE

avg_tce_score = (
    train_tbl[["tce_uid", "raw_pred", "label"]]
    .groupby(["tce_uid", "label"])
    .median()
    .reset_index()
)
avg_tce_score = avg_tce_score.merge(
    tce_tbl[["tce_uid", "disposition", "tce_max_mult_ev"]],
    how="left",
    on="tce_uid",
    validate="many_to_one",
)

# %% plot same hist2d for average score

bins_mes = np.logspace(np.log10(7.1), 4, 50)
bins_scores = np.linspace(0, 1, 11)

for disp in avg_tce_score["disposition"].unique():
    for label in avg_tce_score["label"].unique():

        avg_tce_score_disp = avg_tce_score.loc[
            ((avg_tce_score["disposition"] == disp) & (avg_tce_score["label"] == label))
        ]
        if len(avg_tce_score_disp) == 0:
            continue

        f, ax = plt.subplots()

        # ax.scatter(train_tbl_disp['tce_max_mult_ev'], train_tbl_disp['raw_pred'], s=8, alpha=0.01)
        # ax.set_ylim([0, 1])
        # ax.set_xlim(left=7.1)

        im = ax.hist2d(
            avg_tce_score_disp["tce_max_mult_ev"],
            avg_tce_score_disp["raw_pred"],
            bins=[bins_mes, bins_scores],
            cmap="jet",
            norm=LogNorm(),
        )[3]
        ax.set_ylim([0, 1])
        ax.set_xlim(left=7.1)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax, label="TCE Count")

        ax.set_xscale("log")

        ax.set_ylabel("TCE Median Model Score")
        ax.set_xlabel("TCE MES")
        ax.set_title(f"Disposition {disp} | Label {label}")
        f.savefig(plot_dir / f"hist2d_model_scores_tcemes_mediantces_{disp}.png")
        plt.show()

# %% check scores for a given TCE

tce_uid = "198408416-1-S14-60"  # '425064757-1-S1-65'   # '198408416-1-S14-60'

examples_tce = train_tbl.loc[train_tbl["tce_uid"] == tce_uid]
disp_tce = examples_tce["disposition"].values[0]
mes_tce = examples_tce["tce_max_mult_ev"].values[0]

transit_window_examples = examples_tce.loc[examples_tce["label"] == 1]
not_transit_window_examples = examples_tce.loc[examples_tce["label"] == 0]

f, ax = plt.subplots(figsize=(10, 5))
ax.scatter(
    transit_window_examples["time"],
    transit_window_examples["raw_pred"],
    s=8,
    alpha=0.3,
    edgecolors="k",
    label="Transit Window Examples",
)
ax.scatter(
    not_transit_window_examples["time"],
    not_transit_window_examples["raw_pred"],
    s=8,
    alpha=0.3,
    edgecolors="k",
    label="Not-Transit Window Examples",
)
ax.set_ylabel("Model Score")
ax.set_xlabel("Timestamp [BTJD]")
ax.set_ylim(bins_scores[[0, -1]])
ax.legend()
ax.set_title(
    f"TCE {tce_uid}\nDisposition {disp_tce}\nNumber of examples"
    f" {len(examples_tce)} | TCE Max MES {mes_tce:.3f}"
)
f.tight_layout()
f.savefig(
    plot_dir / f"scatter_transit_nottransit_examples_scores_{tce_uid}_{disp_tce}.png"
)
plt.show()


bins_scores = np.linspace(0, 1, 11)

f, ax = plt.subplots()
ax.hist(
    transit_window_examples["raw_pred"],
    bins_scores,
    histtype="step",
    label="Transit Window Examples",
)
ax.hist(
    not_transit_window_examples["raw_pred"],
    bins_scores,
    histtype="step",
    label="Not-Transit Window Examples",
)
ax.set_xlabel("Model Score")
ax.set_ylabel("Example Count")
ax.set_xlim(bins_scores[[0, -1]])
ax.legend()
ax.set_title(
    f"TCE {tce_uid}\nDisposition {disp_tce}\nNumber of examples"
    f" {len(examples_tce)} | TCE Max MES {mes_tce:.3f}"
)
f.tight_layout()
f.savefig(
    plot_dir / f"hist_transit_nottransit_examples_scores_{tce_uid}_{disp_tce}.png"
)
plt.show()

# %% check extracted windows for a TCE

tce_uid = "198408416-1-S14-60"  # '425064757-1-S1-65'   # '198408416-1-S14-60'
lc_dir = Path("/Users/msaragoc/Downloads/")
sector_arr = [40, 41]

tce = tce_tbl.loc[tce_tbl["tce_uid"] == tce_uid]

# find light curve data for target
search_lc_res = lk.search_lightcurve(
    target=f"tic{tce['target_id'].values[0]}",
    mission="TESS",
    author=("TESS-SPOC", "SPOC"),
    exptime=120,
    cadence="long",
    sector=sector_arr,
)

lcf = search_lc_res.download_all(
    download_dir=str(lc_dir), quality_bitmask="default", flux_column="pdcsap_flux"
)
lcf = lcf.stitch()

# %%

t0, win_label = 2425.47, 0  # 2425.47, 0  # 2424.57, 1

dur_f = 5
win_len = tce["tce_duration"].values[0] / 24 * dur_f

t_start, t_end = t0 - win_len / 2, t0 + win_len / 2

f, ax = plt.subplots()
lcf.plot(ax=ax)
ax.set_xlim([t_start, t_end])
ax.set_title(f"TCE {tce_uid} | Disposition {disp_tce}\nt0={t0} | Label {win_label}")
f.savefig(plot_dir / f"plot_{tce_uid}_{disp_tce}_timestamp{t0}_label{win_label}.png")
