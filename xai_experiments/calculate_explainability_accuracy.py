import pandas as pd
import numpy as np
import os
import ast

csv_folder = "csv/primary_branch_approaches"

folders = ["max_jump", "top_4_branches", "threshold_0.09", "threshold_0.12", "threshold_0.15"]

csv_files = {
    "hubs": "ranked_predictions_with_hubs.csv",
    "authorities": "ranked_predictions_with_authorities.csv",
    "attn": "ranked_predictions_testset_attn-mean-query.csv",
    "shap": "ranked_predictions_testset_shap.csv"
}

correct_map = {
    "EB": ["local_weak_secondary", "local_odd_even", "global_flux"],
    "FP-EB": ["local_weak_secondary", "local_odd_even", "global_flux"],
    "NTP": ["global_flux", "local_flux", "dv_tce_fit"],
    "FP-NPC": ["diff_img", "local_centroid"],
    "FP-NEB": ["diff_img", "local_centroid"]
}

incorrect_map = {
    "EB": ["diff_img", "local_centroid"],
    "FP-EB": ["diff_img", "local_centroid"],
    "NTP": ["local_weak_secondary", "local_odd_even", "diff_img", "local_centroid"],
    "FP-NPC": ["local_weak_secondary", "local_odd_even", "global_flux", "local_flux", "flux_periodogram", "local_unfolded_flux", "dv_tce_fit"],
    "FP-NEB": ["local_weak_secondary", "local_odd_even", "global_flux", "local_flux", "flux_periodogram", "local_unfolded_flux", "dv_tce_fit"]
}

combined_labels = {
    "EB/FP-EB":     ["EB", "FP-EB"],
    "NTP":          ["NTP"],
    "FP-NEB/FP-NPC":["FP-NEB", "FP-NPC"]
}
def compute_combined_stats_csv(approach_folder):
    base = os.path.join(csv_folder, approach_folder)
    for file_key, fname in csv_files.items():
        path = os.path.join(base, fname)
        if not os.path.exists(path):
            continue

        df = pd.read_csv(path)
        branch_col = "shap_primary_branch" if file_key=="shap" else "primary_branch"
        df[branch_col] = df[branch_col].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )

        # accumulate rows for this table
        out = []
        overall = {"total":0, "corr":0, "inc":0, "both":0}

        for grp, members in combined_labels.items():
            sub = df[df["clarified_label"].isin(members)]
            tot = len(sub)

            # build unioned sets
            corr_set = set().union(*(correct_map[m] for m in members))
            inc_set  = set().union(*(incorrect_map[m] for m in members))

            c = sub[branch_col].apply(lambda bl: bool(set(bl)&corr_set)).sum()
            i = sub[branch_col].apply(lambda bl: bool(set(bl)&inc_set)).sum()
            b = sub[branch_col].apply(
                    lambda bl: bool(set(bl)&corr_set) and bool(set(bl)&inc_set)
                ).sum()

            pct = lambda x: f"{x/ tot * 100:.1f}%" if tot>0 else "0.0%"

            out.append({
                "Group":    grp,
                "Correct":   f"{c} / {tot} ({pct(c)})",
                "Incorrect": f"{i} / {tot} ({pct(i)})",
                "Both":      f"{b} / {tot} ({pct(b)})"
            })

            overall["total"] += tot
            overall["corr"]  += c
            overall["inc"]   += i
            overall["both"]  += b

        # append the grand total row
        T   = overall["total"]
        Ct  = overall["corr"]
        It  = overall["inc"]
        Bt  = overall["both"]
        pct = lambda x: f"{x/ T * 100:.1f}%" if T>0 else "0.0%"

        out.append({
            "Group":    "Total",
            "Correct":   f"{Ct} / {T} ({pct(Ct)})",
            "Incorrect": f"{It} / {T} ({pct(It)})",
            "Both":      f"{Bt} / {T} ({pct(Bt)})"
        })

        df_out = pd.DataFrame(out)
        print(f"\n# {approach_folder} & {file_key}")
        print(df_out.to_csv(index=False))


# example run
for folder in folders:
    compute_combined_stats_csv(folder)