"""Match objects based on period aloone.

CSV-only ephemeris matching between two TESS catalogs using TIC+period.

Required columns in both CSVs:
- uid       (string; enforced)
- target_id (TIC; numeric preferred; matched as string if not)
- epoch     (BTJD; carried through, not used for matching)
- period    (days; float)
- duration  (hours; float)

Matching:
1) Inner-join on target_id.
2) Compare periods and durations under user-selected tolerance mode:
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
    required = ["uid", "target_id", "epoch", "period", "duration"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"[{label}] Missing columns: {missing}")

    df = df.copy()

    # uid as string
    df["uid"] = df["uid"].astype(str)

    # epoch, period, and transit to float
    df["epoch"]  = pd.to_numeric(df["epoch"],  errors="coerce").astype("float64")
    df["period"] = pd.to_numeric(df["period"], errors="coerce").astype("float64")
    df["duration"] = pd.to_numeric(df["duration"], errors="coerce").astype("float64")

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

def _duration_match(dur1: float, dur2: float, mode: str, tol: float, ppm_ref: str = "max") -> Tuple[bool, float]:
    if mode == "absolute":
        delta = abs(dur1 - dur2)
        return (delta <= tol, delta)
    elif mode == "relative":
        denom = max(dur1, dur2)
        delta = abs(dur1 - dur2) / denom if denom > 0 else math.inf
        return (delta <= tol, delta)
    elif mode == "ppm":
        delta = _ppm_delta(dur1, dur2, ppm_ref)
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
    tol_period: float,
    tol_duration: float,
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
            "uid_l","uid_r","target_id","period_l","period_r", "duration_l","duration_r",
            "epoch_l","epoch_r","mode","delta_period", "delta_duration",
            "matched_via_harmonic","harmonic_used",
        ])

    rows = []
    for _, row in merged.iterrows():

        p1, p2 = float(row["period_l"]), float(row["period_r"])
        
        dur1, dur2 = float(row["duration_l"]), float(row["duration_r"])

        ok_p, delta_p = _period_match(p1, p2, mode=mode, tol=tol_period, ppm_ref=ppm_ref)
        
        ok_dur, delt_dur = _duration_match(dur1, dur2, mode=mode, tol=tol_duration, ppm_ref=ppm_ref)

        if ok_p and ok_dur:
            rows.append({
                "uid_l": row["uid_l"], "uid_r": row["uid_r"],
                "target_id": row["_tic_key"],
                "period_l": p1, "period_r": p2,
                "duration_l": dur1, "duration_r": dur2,
                "epoch_l": row["epoch_l"], "epoch_r": row["epoch_r"],
                "mode": mode, "delta_period": delta_p, "delta_duration": delt_dur,
                "matched_via_harmonic": False, "harmonic_used": 1.0,
            })
            continue

        if check_harmonics and ok_dur:
            best = None
            best_factor = None
            for h in _harmonic_periods(p1, max_harmonic):
                okh, dh = _period_match(h, p2, mode=mode, tol=tol_period, ppm_ref=ppm_ref)
                if okh and (best is None or dh < best):
                    best = dh
                    best_factor = h / p1
            if best is not None:
                rows.append({
                    "uid_l": row["uid_l"], "uid_r": row["uid_r"],
                    "target_id": row["_tic_key"],
                    "period_l": p1, "period_r": p2,
                    "duration_l": dur1, "duration_r": dur2,
                    "epoch_l": row["epoch_l"], "epoch_r": row["epoch_r"],
                    "mode": mode, "delta_period": best, "delta_duration": delt_dur,
                    "matched_via_harmonic": True, "harmonic_used": best_factor,
                })

    result = pd.DataFrame(rows)
    if result.empty:
        return result

    if not keep_all_candidates:
        result = (result
                  .sort_values(["uid_l", "delta_period", "delta_duration"], ascending=[True, True, True])
                  .groupby("uid_l", as_index=False)
                  .first())

    # cast TIC back to int if possible
    try:
        result["target_id"] = result["target_id"].astype("int64")
    except Exception:
        pass

    # order columns
    cols = ["uid_l","uid_r","target_id","period_l","period_r","duration_l","duration_r", "epoch_l","epoch_r",
            "mode","delta_period","delta_duration","matched_via_harmonic","harmonic_used"]
    return result[cols]


# ------------------------------ Aggregation (optional) ------------------------------

def aggregate_by_left(matches: pd.DataFrame) -> pd.DataFrame:
    """
    Build per-left (uid_l) summary:
      - target_id
      - n_obj_matches
      - obj_uids
      - obj_periods
      - obj_durations
      - harmonic_used_list
      - min_delta, max_delta
    """
    if matches.empty:
        return pd.DataFrame(columns=[
            "uid_l","target_id","n_obj_matches","obj_uids","obj_periods","obj_durations",
            "harmonic_used_list","min_period_delta","max_period_delta","min_duration_delta","max_duration_delta"
        ])

    def join_str(series):
        return ",".join([str(x) for x in series if pd.notna(x)])

    def join_floats(series):
        return ",".join([f"{float(x):.8f}" for x in series if pd.notna(x)])

    grouped = (matches
               .sort_values(["uid_l","delta_period","delta_duration"], ascending=[True, True, True])
               .groupby(["uid_l","target_id"], as_index=False)
               .agg({
                   "uid_r": join_str,
                   "period_r": join_floats,
                   "duration_r": join_floats,
                   "harmonic_used": lambda s: ",".join([f"{float(x):g}" for x in s if pd.notna(x)]),
                   "delta_period": ["min","max"],
                   "delta_duration": ["min","max"]
               }))

    # flatten multiindex columns from agg
    grouped.columns = [
        "uid_l","target_id","obj_uids","obj_periods",
        "harmonic_used_list","min_period_delta","max_period_delta","min_duration_delta","max_duration_delta"
    ]

    grouped["n_obj_matches"] = grouped["obj_uids"].apply(lambda s: 0 if s == "" else len(s.split(",")))
    cols = ["uid_l","target_id","n_obj_matches","obj_uids","obj_periods",
            "harmonic_used_list","min_period_delta","max_period_delta","min_duration_delta","max_duration_delta"]
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

    return {
        "left_rows": len(left),
        "right_rows": len(right),
        "tic_overlap": tic_overlap,
        "matches": len(merged_matches),
        "matches_direct": n_direct,
        "matches_harmonic": n_harm,
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
        f"# Tolerance period: {params.tol_period}",
        f"# Tolerance duration: {params.tol_duration}",
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
        f"# Columns: uid_l, uid_r, target_id, period_l, period_r, duration_l, duration_r, epoch_l, epoch_r, mode, delta, matched_via_harmonic, harmonic_used",
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
    ap.add_argument("--tol_period", type=float, required=True,
                    help="Tolerance value for period (interpreted per --mode).")
    ap.add_argument("--tol_duration", type=float, required=True,
                    help="Tolerance value for duration (interpreted per --mode).")
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
        mode=args.mode, tol_period=args.tol_period, tol_duration=args.tol_duration, ppm_ref=args.ppm_ref,
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
            "# Columns: uid_l, target_id, n_obj_matches, obj_uids, obj_periods, harmonic_used_list, min_period_delta, max_period_delta, min_duration_delta, max_duration_delta",
        ]
        _write_csv_with_header(args.aggregate_output, agg, agg_meta)
        print(f"Wrote left-level aggregation for {len(agg)} objects to {args.aggregate_output}")

    return 0


if __name__ == "__main__":
    
    raise SystemExit(main())


# python /path/to/script.py /home6/msaragoc/work_dir/Kepler-TESS_exoplanet/data/Ephemeris_tables/TESS/tess_spoc_2min/tess-spoc-2min-tces-dv_s89-s98_s1s92_2-13-2026_1010/tess_spoc_2min_tces_dv_s89-s98_s1s92_2-13-2026_1010_stellartic8_ruwegaiadr2_preproc_ruwe_tic8stellar_for-ephem-matching.csv /home6/msaragoc/work_dir/Kepler-TESS_exoplanet/data/Ephemeris_tables/TESS/exofop_tois/tois_3-2-2026_processed_ephem_matching.csv -o /home6/msaragoc/work_dir/Kepler-TESS_exoplanet/experiments/ephemeris_matching/period-duration-matching_tess-spoc-2min-tces-dv_s89-s98_s1s92_exofop-tois_3-11-2026_1125.csv --mode relative --tol_period 0.01 --tol_duration 0.1 
