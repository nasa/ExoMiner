import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# data = pd.DataFrame(
#     {
#         "1TD": [0, 0, 0, 0],
#         "3TD": [0, 0, 0, 0],
#         "5TD": [0, 0, 0, 0],
#     },
#     index=["self-center", "self-3TD", "other-center", "other-3TD"],
# )


if __name__ == "__main__":

    plot_dir = Path(
        "/nobackupp27/jochoa4/work_dir/data/plots/compute_corrupted_labels_05-04-2025_v1"
    )
    csv_dir = Path(
        "/nobackupp27/jochoa4/work_dir/job_runs/compute_corrupted_labels_05-04-2025_v1"
    )

    for split in ["train", "test", "val"]:
        csv_fp = csv_dir / f"{split}_corrupted_neg_label_analysis.csv"
        split_df = pd.read_csv(str(csv_fp))

        neg_df = split_df[split_df["label"] == 0].copy()  # Filter negative labels

        split_plot_dir = plot_dir / split
        split_plot_dir.mkdir(parents=True, exist_ok=True)

        for disposition in ["ALL", "EB", "KP", "CP", "NTP", "NEB", "NPC"]:

            if disposition != "ALL":
                disp_df = neg_df[neg_df["disposition"] == disposition].copy()
            else:
                disp_df = neg_df.copy()

            ws_windows = [0.0, 3.0]
            ex_windows = [1.0, 2.5, 5.0]

            corruption_sources = ["self_", "cross_tce_", ""]

            corruption_matrix = pd.DataFrame(
                0.0,
                index=[
                    f"{src}ws_{ws}" for src in corruption_sources for ws in ws_windows
                ],
                columns=ex_windows,
            )

            annot_matrix = pd.DataFrame(
                "",
                index=corruption_matrix.index,
                columns=corruption_matrix.columns,
            )

            for src in corruption_sources:
                for ws in ws_windows:
                    for ex in ex_windows:
                        colname = f"{src}corrupt_ws_{ws}_td_ex_{ex}_td"
                        rowname = f"{src}ws_{ws}"

                        assert (
                            colname in disp_df.columns
                        ), f"ERROR: missing column {colname}"

                        total = len(disp_df)
                        valid = disp_df[colname].count()

                        assert valid == total, f"valid != total"

                        count = disp_df[colname].sum()
                        rate = count / valid if valid > 0 else 0.0

                        corruption_matrix.loc[rowname, ex] = rate

                        annot_matrix.loc[rowname, ex] = (
                            f"{int(count)} / {valid}\n{rate:.1%}"
                        )

            pretty_labels = {
                "self_ws_0.0": "Same-TCE Secondary (Center Only)",
                "self_ws_3.0": "Same-TCE Secondary (± 1.5 TD)",
                "cross_tce_ws_0.0": "Cross-TCE Secondary (Center Only)",
                "cross_tce_ws_3.0": "Cross-TCE Secondary (± 1.5 TD)",
                "ws_0.0": "Any Secondary (Center Only)",
                "ws_3.0": "Any Secondary (± 1.5 TD)",
            }
            corruption_matrix.rename(index=pretty_labels, inplace=True)
            annot_matrix.rename(index=pretty_labels, inplace=True)

            corruption_matrix.columns = ["± 0.5 TD", "± 1.25 TD", "±2.5 TD"]
            annot_matrix.columns = ["± 0.5 TD", "± 1.25 TD", "±2.5 TD"]

            plt.figure(figsize=(10, 6))
            sns.heatmap(
                corruption_matrix * 100,
                annot=annot_matrix,
                fmt="",
                cmap="viridis",
                cbar_kws={"label": "Contamination Rate (%)"},
            )
            plt.title(
                f"Fraction of {disposition} Negative Examples Affected by Strong Weak Secondary Transit Contamination (> 7.1 maxmes)"
            )

            plt.xticks(fontsize=10)
            plt.yticks(fontsize=10)
            plt.xlabel("Primary Window Size of Negative Examples (x TD)")
            plt.ylabel("Contamination Source and Weak Secondary Window Size (x TD)")
            print(f"Saving fig")
            plt.tight_layout()

            plt.savefig(
                str(split_plot_dir / f"{disposition}_neg_label_ws_contamination.png"),
                dpi=300,
                bbox_inches="tight",
            )
