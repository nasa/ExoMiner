"""Match objects based on period aloone.

CSV-only ephemeris matching between two TESS catalogs using TIC+period.

Required columns in both CSVs:
- uid       (string; enforced)
- target_id (TIC; numeric preferred; matched as string if not)
- epoch     (BTJD; carried through, not used for matching)
- period    (days; float)

Matching:
1) Inner-join on target_id.
2) Compare periods under user-selected tolerance mode:
   - absolute: |p1 - p2| <= tol
   - relative: |p1 - p2| / max(p1, p2) <= tol
   - ppm:      (|p1 - p2| / REF) * 1e6 <= tol_ppm, where REF ∈ {max, p1, p2}

Optional harmonic checks (½×, 2×, 3× …).

Output: CSV with match details.
"""

# 3rd party
from __future__ import annotations
import argparse
import math
import sys
from pathlib import Path
from typing import Iterable, List, Tuple, Dict
from datetime import datetime, timezone
import pandas as pd


# ------------------------------ I/O ------------------------------

def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _enforce_schema(df: pd.DataFrame, label: str) -> pd.DataFrame:
    required = ["uid", "target_id", "epoch", "period"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"[{label}] Missing columns: {missing}")

    df = df.copy()

    # uid as string
    df["uid"] = df["uid"].astype(str)

    # epoch and period to float
    df["epoch"]  = pd.to_numeric(df["epoch"],  errors="coerce").astype("float64")
    df["period"] = pd.to_numeric(df["period"], errors="coerce").astype("float64")

    # target_id: try numeric int64; if it fails, keep as string and warn
    try:
        df["target_id"] = pd.to_numeric(df["target_id"], errors="raise").astype("int64")
    except Exception:
        sys.stderr.write(f"[WARN] {label}.target_id could not be parsed to integer; "
                         f"keeping as string for matching.\n")
        df["target_id"] = df["target_id"].astype(str)

    # drop rows missing key fields
    n0 = len(df)
    df = df.dropna(subset=["target_id", "period"])
    if len(df) < n0:
        sys.stderr.write(f"[INFO] Dropped {n0 - len(df)} rows with NaN target_id/period in {label}.\n")
    return df


# ------------------------------ Period deltas ------------------------------

def _ppm_delta(p1: float, p2: float, ref: str = "max") -> float:
    """
    ppm delta = |p1 - p2| / REF * 1e6, REF ∈ {'max','p1','p2'}:
      - 'max': REF = max(p1, p2)  (symmetric, conservative)
      - 'p1' : REF = p1           (asymmetric, normalize to left)
      - 'p2' : REF = p2           (asymmetric, normalize to right)
    """
    if ref == "max":
        r = max(p1, p2)
    elif ref == "p1":
        r = p1
    elif ref == "p2":
        r = p2
    else:
        raise ValueError("ppm-ref must be one of {'max','p1','p2'}")
    return abs(p1 - p2) / r * 1e6


def _period_match(p1: float, p2: float, mode: str, tol: float, ppm_ref: str = "max") -> Tuple[bool, float]:
    if mode == "absolute":
        delta = abs(p1 - p2)
        return (delta <= tol, delta)
    elif mode == "relative":
        denom = max(p1, p2)
        delta = abs(p1 - p2) / denom if denom > 0 else math.inf
        return (delta <= tol, delta)
    elif mode == "ppm":
        delta = _ppm_delta(p1, p2, ppm_ref)
        return (delta <= tol, delta)
    else:
        raise ValueError("mode must be one of {'absolute','relative','ppm'}")


def _harmonic_periods(p: float, max_factor: int) -> List[float]:
    downs = [p / k for k in range(2, max_factor + 1)]
    ups   = [p * k for k in range(2, max_factor + 1)]
    return downs + ups


# ------------------------------ Matching core ------------------------------

def match_catalogs(
    left: pd.DataFrame,
    right: pd.DataFrame,
    mode: str,
    tol: float,
    ppm_ref: str = "max",
    check_harmonics: bool = False,
    max_harmonic: int = 3,
    keep_all_candidates: bool = False,
) -> pd.DataFrame:
    """
    Returns row-level matches (one left↔right pair per row).
    """
    
    # normalize join keys to strings to avoid dtype conflicts
    left_key  = left["target_id"].astype(str)
    right_key = right["target_id"].astype(str)
    left  = left.assign(_tic_key=left_key)
    right = right.assign(_tic_key=right_key)

    merged = left.merge(right, on="_tic_key", suffixes=("_l", "_r"), how="inner")
    if merged.empty:
        return pd.DataFrame(columns=[
            "uid_l","uid_r","target_id","period_l","period_r",
            "epoch_l","epoch_r","mode","delta",
            "matched_via_harmonic","harmonic_used",
            # "tfopwg_disposition_r"
        ])

    # # detect if the TOI table actually has TFOPWG column
    # tfop_col_present = "TFOPWG Disposition" in merged.columns

    rows = []
    for _, row in merged.iterrows():
        p1, p2 = float(row["period_l"]), float(row["period_r"])

        ok, d = _period_match(p1, p2, mode=mode, tol=tol, ppm_ref=ppm_ref)
        # tfop_raw = row["TFOPWG Disposition"] if tfop_col_present else None

        if ok:
            rows.append({
                "uid_l": row["uid_l"], "uid_r": row["uid_r"],
                "target_id": row["_tic_key"],
                "period_l": p1, "period_r": p2,
                "epoch_l": row["epoch_l"], "epoch_r": row["epoch_r"],
                "mode": mode, "delta": d,
                "matched_via_harmonic": False, "harmonic_used": 1.0,
                # "tfopwg_disposition_r": tfop_raw,
            })
            continue

        if check_harmonics:
            best = None
            best_factor = None
            for h in _harmonic_periods(p1, max_harmonic):
                okh, dh = _period_match(h, p2, mode=mode, tol=tol, ppm_ref=ppm_ref)
                if okh and (best is None or dh < best):
                    best = dh
                    best_factor = h / p1
            if best is not None:
                rows.append({
                    "uid_l": row["uid_l"], "uid_r": row["uid_r"],
                    "target_id": row["_tic_key"],
                    "period_l": p1, "period_r": p2,
                    "epoch_l": row["epoch_l"], "epoch_r": row["epoch_r"],
                    "mode": mode, "delta": best,
                    "matched_via_harmonic": True, "harmonic_used": best_factor,
                    # "tfopwg_disposition_r": tfop_raw,
                })

    result = pd.DataFrame(rows)
    if result.empty:
        return result

    if not keep_all_candidates:
        result = (result
                  .sort_values(["uid_l", "delta"], ascending=[True, True])
                  .groupby("uid_l", as_index=False)
                  .first())

    # cast TIC back to int if possible
    try:
        result["target_id"] = result["target_id"].astype("int64")
    except Exception:
        pass

    # order columns
    cols = ["uid_l","uid_r","target_id","period_l","period_r","epoch_l","epoch_r",
            "mode","delta","matched_via_harmonic","harmonic_used"]
    return result[cols]


# ------------------------------ Aggregation (optional) ------------------------------

def aggregate_by_left(matches: pd.DataFrame) -> pd.DataFrame:
    """
    Build per-left (uid_l) summary:
      - target_id
      - n_toi_matches
      - toi_uids
      - toi_periods
      - harmonic_used_list
      - min_delta, max_delta
    """
    if matches.empty:
        return pd.DataFrame(columns=[
            "uid_l","target_id","n_toi_matches","toi_uids","toi_periods",
            "harmonic_used_list","min_delta","max_delta"
        ])

    def join_str(series):
        return ",".join([str(x) for x in series if pd.notna(x)])

    def join_floats(series):
        return ",".join([f"{float(x):.8f}" for x in series if pd.notna(x)])

    grouped = (matches
               .sort_values(["uid_l","delta"], ascending=[True, True])
               .groupby(["uid_l","target_id"], as_index=False)
               .agg({
                   "uid_r": join_str,
                   "period_r": join_floats,
                   "harmonic_used": lambda s: ",".join([f"{float(x):g}" for x in s if pd.notna(x)]),
                   "delta": ["min","max"]
               }))

    # flatten multiindex columns from agg
    grouped.columns = [
        "uid_l","target_id","toi_uids","toi_periods",
        "harmonic_used_list","min_delta","max_delta"
    ]

    grouped["n_toi_matches"] = grouped["toi_uids"].apply(lambda s: 0 if s == "" else len(s.split(",")))
    cols = ["uid_l","target_id","n_toi_matches","toi_uids","toi_periods",
            "harmonic_used_list","min_delta","max_delta"]
    return grouped[cols]


# ------------------------------ Metadata / CSV writing ------------------------------

def _collect_stats(
    left: pd.DataFrame, right: pd.DataFrame, merged_matches: pd.DataFrame
) -> Dict[str, int]:
    # TIC overlap estimate (unique keys intersection)
    left_tics  = set(left["target_id"].astype(str).unique())
    right_tics = set(right["target_id"].astype(str).unique())
    tic_overlap = len(left_tics & right_tics)

    # harmonic stats
    n_harm = int((merged_matches["matched_via_harmonic"] == True).sum()) if not merged_matches.empty else 0
    n_direct = int((merged_matches["matched_via_harmonic"] == False).sum()) if not merged_matches.empty else 0

    # # dispositions available
    # disp_col = "tfopwg_disposition_r" in merged_matches.columns
    # n_disp_rows = int(merged_matches["tfopwg_disposition_r"].notna().sum()) if disp_col else 0

    return {
        "left_rows": len(left),
        "right_rows": len(right),
        "tic_overlap": tic_overlap,
        "matches": len(merged_matches),
        "matches_direct": n_direct,
        "matches_harmonic": n_harm,
        # "tfop_dispositions_rows": n_disp_rows,
    }


def _write_csv_with_header(
    output_path: Path,
    df: pd.DataFrame,
    metadata_lines: List[str]
) -> None:
    """
    Write comment metadata lines first, then the CSV table.
    """
    with open(output_path, "w", encoding="utf-8") as f:
        for line in metadata_lines:
            if not line.startswith("#"):
                line = "# " + line
            f.write(line.rstrip() + "\n")
        df.to_csv(f, index=False)


def _build_metadata_lines(
    created_iso: str,
    params: argparse.Namespace,
    paths: Dict[str, Path],
    stats: Dict[str, int]
) -> List[str]:
    """
    Compose the header block you'll see at the top of the CSV.
    """
    lines = [
        f"# Ephemeris match output",
        f"# Created: {created_iso}",
        f"# Mode: {params.mode}",
        f"# Tolerance: {params.tol}",
    ]
    if params.mode == "ppm":
        lines.append(f"# ppm-ref: {params.ppm_ref}")
    lines += [
        f"# Harmonics enabled: {bool(params.harmonics)}",
        f"# Max harmonic: {params.max_harmonic}",
        f"# Keep all candidates: {bool(params.keep_all)}",
        f"# Left catalog: {paths['left']}",
        f"# Right catalog: {paths['right']}",
        f"# Left rows: {stats['left_rows']}",
        f"# Right rows: {stats['right_rows']}",
        f"# TIC overlap (unique): {stats['tic_overlap']}",
        f"# Matches: {stats['matches']}",
        f"# Matches (direct): {stats['matches_direct']}",
        f"# Matches (harmonic): {stats['matches_harmonic']}",
        # f"# TFOPWG dispositions included (row-level): {stats['tfop_dispositions_rows']}",
        f"# Columns: uid_l, uid_r, target_id, period_l, period_r, epoch_l, epoch_r, mode, delta, matched_via_harmonic, harmonic_used", # , tfopwg_disposition_r",
    ]
    return lines


# ------------------------------ CLI ------------------------------

def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Match two TESS CSV catalogs on TIC and period (threshold-based) and write a CSV with metadata header."
    )
    ap.add_argument("left",  type=Path, help="Path to first CSV (e.g., Prša EBs)")
    ap.add_argument("right", type=Path, help="Path to second CSV (e.g., ExoFOP TOIs)")
    ap.add_argument("-o","--output", type=Path, required=True, help="Output CSV path.")

    ap.add_argument("--mode", choices=["absolute","relative","ppm"], default="relative",
                    help="Tolerance mode: absolute (days), relative (fraction), ppm.")
    ap.add_argument("--tol", type=float, required=True,
                    help="Tolerance value (interpreted per --mode).")
    ap.add_argument("--ppm-ref", choices=["max","p1","p2"], default="max",
                    help="Reference period used in ppm mode.")

    ap.add_argument("--harmonics", action="store_true",
                    help="Enable harmonic checks (½×, 2×, 3×... up to --max-harmonic).")
    ap.add_argument("--max-harmonic", type=int, default=3,
                    help="Maximum harmonic factor to consider (default=3).")
    ap.add_argument("--all", dest="keep_all", action="store_true",
                    help="Keep all matches; default keeps best per left.uid.")

    ap.add_argument("--aggregate-output", type=Path, default=None,
                    help="Optional path for left-level aggregation CSV (lists all matched right objects).")

    return ap.parse_args(argv)


def main(argv: Iterable[str] = None) -> int:
    args = parse_args(argv or sys.argv[1:])

    left_df  = _enforce_schema(_read_csv(args.left),  label="left")
    right_df = _enforce_schema(_read_csv(args.right), label="right")

    result = match_catalogs(
        left=left_df, right=right_df,
        mode=args.mode, tol=args.tol, ppm_ref=args.ppm_ref,
        check_harmonics=args.harmonics, max_harmonic=args.max_harmonic,
        keep_all_candidates=args.keep_all,
    )

    # Build metadata
    created_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    stats = _collect_stats(left_df, right_df, result)
    meta_lines = _build_metadata_lines(
        created_iso=created_iso,
        params=args,
        paths={"left": args.left, "right": args.right},
        stats=stats
    )

    # Write main CSV
    _write_csv_with_header(args.output, result, meta_lines)
    print(f"Wrote {len(result)} matched rows to {args.output}")

    # Optional: write left-level aggregation CSV
    if args.aggregate_output is not None:
        agg = aggregate_by_left(result)
        agg_meta = [
            "# left-level aggregation",
            f"# Created: {created_iso}",
            f"# Source match file: {args.output}",
            f"# Rows: {len(agg)}",
            "# Columns: uid_l, target_id, n_toi_matches, toi_uids, toi_periods, harmonic_used_list, min_delta, max_delta",
        ]
        _write_csv_with_header(args.aggregate_output, agg, agg_meta)
        print(f"Wrote left-level aggregation for {len(agg)} objects to {args.aggregate_output}")

    return 0


if __name__ == "__main__":
    
    raise SystemExit(main())

# #%% load catalogs

# psra_ebs = Path('/u/msaragoc/work_dir/Kepler-TESS_exoplanet/data/Ephemeris_tables/TESS/hlsp_tess-ebs_tess_lcf-ffi_s0001-s0026_tess_v1.0_cat_processed.csv')
# tois = Path('/u/msaragoc/work_dir/Kepler-TESS_exoplanet/data/Ephemeris_tables/TESS/exofop_tois/exofop_tois_9-11-2025_processed_ephem_matching.csv')

# python /home6/msaragoc/work_dir/Kepler-TESS_exoplanet/codebase_aux_loss_source_offset/data_wrangling/prototyping/ephemeris_matching/match_prsaebs_tois.py ~/work_dir/Kepler-TESS_exoplanet/experiments/ephemeris_matching/ephem-match-period_kostov-ebs_tess-spoc-tces-ffi/tess-spoc-ffi-tces-dv_s36-s72_multisector-s56s69_10-8-2025_exofop-sg1-tois_3-2-2026.csv ~/work_dir/Kepler-TESS_exoplanet/data/Ephemeris_tables/TESS/ebs_kostov_7.9k_catalog.csv -o ~/work_dir/K
# epler-TESS_exoplanet/experiments/ephemeris_matching/ephem-match-period_kostov-ebs_tess-spoc-tces-ffi/period_matching_tess-spoc-tces-ffi_kostov-ebs_3-2-2026_1407.csv --mode relative --tol 0.005 
# #%%
