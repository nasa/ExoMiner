# %%
# 3rd party
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import lightkurve as lk
from collections import defaultdict

# %%
for set in ["train", "val"]:
    for label_key in ["label", "tw_flag"]:
        for feature_key in ["tce_max_mult_ev", "tce_depth"]:

            plot_dir = Path(
                f"/Users/jochoa4/Desktop/studies/study_model_preds_07-02-2025/by_{label_key}/{set}/{feature_key}"
            )
            plot_dir.mkdir(parents=True, exist_ok=True)

            tce_tbl_fp = Path(
                "/Users/jochoa4/Projects/exoplanet_transit_classification/ephemeris_tables/preprocessing_tce_tables/tess_2min_tces_dv_s1-s68_all_msectors_11-29-2023_2157_newlabels_nebs_npcs_bds_ebsntps_to_unks.csv"
            )

            tce_tbl = pd.read_csv(tce_tbl_fp)
            tce_tbl = tce_tbl.rename(columns={"uid": "tce_uid", "label": "disposition"})

            scores_tbls_dir = Path(
                "/Users/jochoa4/Desktop/pfe_transfers/predict_model_dataset_06-20-2025"
            )
            scores_tbl_fp = scores_tbls_dir / f"preds_{set}.csv"
            train_tbl = pd.read_csv(scores_tbl_fp)

            def _get_tce_uid_from_examples(x):

                tce_uid = x["uid"].split("_")[0]

                return tce_uid

            train_tbl["tce_uid"] = train_tbl.apply(_get_tce_uid_from_examples, axis=1)

            train_tbl = train_tbl.merge(
                tce_tbl[["tce_uid", feature_key]],
                on="tce_uid",
                how="left",
                validate="many_to_one",
            )

            # %%
            for disp, type in zip(
                ["EB", "CP", "KP", "NTP", "NPC", "NEB"], ["p", "p", "p", "n", "n", "n"]
            ):
                disp_tbl = train_tbl[train_tbl["disposition"] == disp]
                if type == "p":
                    assert len(disp_tbl[disp_tbl["label"] == 0]) == len(
                        disp_tbl[disp_tbl["tw_flag"] == 0]
                    ), f"ERROR: {disp} num of label 0 does not match num of tw_flag 0"
                    assert len(disp_tbl[disp_tbl["label"] == 1]) == len(
                        disp_tbl[disp_tbl["tw_flag"] == 1]
                    ), f"ERROR: {disp} num of label 1 does not match num of tw_flag 1"
                else:
                    assert len(disp_tbl[disp_tbl["label"] == 0]) == len(
                        disp_tbl["tw_flag"]
                    ), f"ERROR: {disp} num of label 0 does not match num of tw_flag 0 + 1"

            # %%

            bins_scores = np.linspace(0, 1, 11)

            for disp in train_tbl["disposition"].unique():

                train_tbl_disp = train_tbl.loc[train_tbl["disposition"] == disp]

                transit_window_examples = train_tbl_disp.loc[
                    train_tbl_disp[label_key] == 1
                ]
                not_transit_window_examples = train_tbl_disp.loc[
                    train_tbl_disp[label_key] == 0
                ]

                f, ax = plt.subplots()

                if len(transit_window_examples) > 0:
                    ax.hist(
                        transit_window_examples["pred_prob"],
                        bins_scores,
                        histtype="step",
                        label=(
                            "Transit Windows"
                            if label_key == "tw_flag"
                            else "Positive Label Examples"
                        ),
                        zorder=2,
                        linestyle="dashed",
                        color="#1f77b4",
                    )
                if len(not_transit_window_examples) > 0:
                    ax.hist(
                        not_transit_window_examples["pred_prob"],
                        bins_scores,
                        histtype="step",
                        label=(
                            "Not-Transit Windows"
                            if label_key == "tw_flag"
                            else "Negative Label Examples"
                        ),
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
                plt.close()

            # %%

            if "depth" in feature_key:
                bins_feature = np.logspace(np.log10(1.6), 6, 50)
            else:
                bins_feature = np.logspace(np.log10(7.1), 4, 50)

            bins_scores = np.linspace(0, 1, 11)

            for disp in train_tbl["disposition"].unique():
                for label in train_tbl[label_key].unique():

                    train_tbl_disp = train_tbl.loc[
                        (
                            (train_tbl["disposition"] == disp)
                            & (train_tbl[label_key] == label)
                        )
                    ]
                    if len(train_tbl_disp) == 0:
                        continue

                    f, ax = plt.subplots()

                    # ax.scatter(train_tbl_disp['tce_max_mult_ev'], train_tbl_disp['pred_prob'], s=8, alpha=0.01)
                    # ax.set_ylim([0, 1])
                    # ax.set_xlim(left=7.1)

                    im = ax.hist2d(
                        train_tbl_disp[feature_key],
                        train_tbl_disp["pred_prob"],
                        bins=[bins_feature, bins_scores],
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
                    ax.set_xlabel(f"TCE {'DEPTH' if 'depth' in feature_key else 'mes'}")
                    ax.set_title(
                        f"Disposition {disp} | {'Transit Window Flag' if label_key =='tw_flag' else 'Label'} {label}"
                    )
                    f.savefig(
                        plot_dir
                        / f"hist2d_model_scores_tce{'depth' if 'depth' in feature_key else 'mes'}_{disp}_{label}.png"
                    )
                    plt.close()

            # %%

            avg_tce_score = (
                train_tbl[
                    [
                        "tce_uid",
                        "pred_prob",
                        f"{label_key}",
                    ]
                ]
                .groupby(["tce_uid", f"{label_key}"])
                .median()
                .reset_index()
            )
            avg_tce_score = avg_tce_score.merge(
                tce_tbl[["tce_uid", "disposition", feature_key]],
                how="left",
                on="tce_uid",
                validate="many_to_one",
            )

            # %%

            # %%

            # print(train_tbl["tce_depth"].max(), train_tbl["tce_max_mult_ev"].max())
            # print(train_tbl["tce_depth"].min(), train_tbl["tce_max_mult_ev"].min())
            # print(train_tbl["tce_depth"].quantile(0.999),train_tbl["tce_max_mult_ev"].quantile(0.999) )
            # print(train_tbl["tce_depth"].quantile(0.005),train_tbl["tce_max_mult_ev"].quantile(0.001) )
            # print(np.log10(train_tbl["tce_depth"].quantile(0.999)),train_tbl["tce_max_mult_ev"].quantile(0.999) )
            # print(np.log10(train_tbl["tce_depth"].quantile(0.005)),train_tbl["tce_max_mult_ev"].quantile(0.001) )

            if "depth" in feature_key:
                bins_feature = np.logspace(np.log10(1.6), 6, 50)
            else:
                bins_feature = np.logspace(np.log10(7.1), 4, 50)

            bins_scores = np.linspace(0, 1, 11)

            for disp in avg_tce_score["disposition"].unique():
                for label in avg_tce_score[label_key].unique():

                    avg_tce_score_disp = avg_tce_score.loc[
                        (
                            (avg_tce_score["disposition"] == disp)
                            & (avg_tce_score[label_key] == label)
                        )
                    ]
                    if len(avg_tce_score_disp) == 0:
                        continue

                    f, ax = plt.subplots()

                    # ax.scatter(train_tbl_disp['tce_max_mult_ev'], train_tbl_disp['pred_prob'], s=8, alpha=0.01)
                    # ax.set_ylim([0, 1])
                    # ax.set_xlim(left=7.1)

                    im = ax.hist2d(
                        avg_tce_score_disp[feature_key],
                        avg_tce_score_disp["pred_prob"],
                        bins=[bins_feature, bins_scores],
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
                    ax.set_xlabel(f"TCE {'DEPTH' if 'depth' in feature_key else 'mes'}")
                    ax.set_title(
                        f"Disposition {disp} | {'Transit Window Flag' if label_key =='tw_flag' else 'Label'} {label}"
                    )
                    f.savefig(
                        plot_dir
                        / f"hist2d_model_scores_tce{'depth' if 'depth' in feature_key else 'mes'}_mediantces_{disp}_{label}.png"
                    )
                    plt.close()

# %%
