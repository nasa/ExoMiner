#!/usr/bin/env python3
"""
Create a CSV of TESS sector start/end timestamps in BTJD, without downloading any FITS files.

Approach:
  - Query MAST via Astroquery (Observations) for TESS time-series metadata.
  - Aggregate per sector: start = min(t_min), end = max(t_max).
  - Convert UTC MJD → JD(TDB) via astropy.Time, then BTJD = JD - 2457000.0.
    (Use `.jd`, not `.bjd`; we do not supply sky coords, and sector windows do not need object-dependent BJD.) 

Output CSV:
  sector,start_btjd,end_btjd

Tips:
  - First run may take a couple of minutes while Astroquery/Astropy populate caches (IERS leap seconds, etc.).
  - Use `--min-sector/--max-sector` to limit ranges (e.g., S43+), and `--no-cloud` to compare MAST vs AWS mirror.
"""

import argparse
import time
import pandas as pd
from astropy.time import Time
from astroquery.mast import Observations


def fetch_sector_batched(min_sector: int, max_sector: int, use_cloud: bool,
                         max_retries: int = 4, sleep_base: int = 2) -> pd.DataFrame:
    """
    Query Astroquery MAST per sector and return a table:
        sector, start_btjd, end_btjd

    We batch by sector to:
      - Give clearer progress / logging
      - Retry a single sector independently on transient errors
    """
    if use_cloud:
        # Use the S3 public mirror to reduce load on MAST (you noted this helped for SPOC runs). [1](https://teams.microsoft.com/l/meeting/details?eventId=AAMkAGRjM2FhMTRhLWQ0NjEtNGQyMy05YjdmLWY5NjhjZjM1MzE5NwFRAAgI3pBKyI9AAEYAAAAAB3jk5oENEUyynJVKNHXYBQcAvrXwDcFjl0OV9roQhHAmGwAAAGnnRwAAtmoGGxtzOkGuB541-_e6pwAEz9EL4QAAEA%3d%3d)
        Observations.enable_cloud_dataset()

    rows = []
    total = max_sector - min_sector + 1
    print(f"[INFO] Fetching sectors {min_sector}–{max_sector} in {total} batch(es)…")

    for sector in range(min_sector, max_sector + 1):
        # Retry this sector independently
        for attempt in range(max_retries):
            try:
                print(f"  - Sector {sector} … (attempt {attempt+1})")
                obs = Observations.query_criteria(
                    obs_collection="TESS",
                    dataproduct_type="timeseries",
                    sequence_number=sector
                )
                break
            except Exception as e:
                wait = sleep_base ** attempt
                print(f"    [WARN] sector {sector}: {e}; retrying in {wait}s")
                time.sleep(wait)
        else:
            print(f"    [ERROR] sector {sector}: failed after {max_retries} attempts; skipping")
            continue

        if len(obs) == 0:
            # No rows for this sector (beyond current mission or indexing gap)
            continue

        df = obs.to_pandas()
        # Expect t_min/t_max (UTC MJD)
        if not {"t_min", "t_max"}.issubset(df.columns):
            print(f"    [ERROR] sector {sector}: missing t_min/t_max; skipping")
            continue

        df["t_min"] = pd.to_numeric(df["t_min"], errors="coerce")
        df["t_max"] = pd.to_numeric(df["t_max"], errors="coerce")
        df = df.dropna(subset=["t_min", "t_max"])
        if df.empty:
            continue

        # Aggregate per sector
        start_mjd = df["t_min"].min()
        end_mjd   = df["t_max"].max()

        # UTC MJD → TDB → JD(TDB) → BTJD
        t_start_tdb = Time(start_mjd, format="mjd", scale="utc").tdb
        t_end_tdb   = Time(end_mjd,   format="mjd", scale="utc").tdb
        start_btjd  = float(t_start_tdb.jd - 2457000.0)
        end_btjd    = float(t_end_tdb.jd   - 2457000.0)

        rows.append({"sector": sector, "start_btjd": start_btjd, "end_btjd": end_btjd})

    return pd.DataFrame(rows).sort_values("sector")


def main():
    ap = argparse.ArgumentParser(description="Create TESS sector start/end BTJD table via Astroquery (batched).")
    ap.add_argument("--out", default="tess_sector_times.csv", help="Output CSV filename")
    ap.add_argument("--min-sector", type=int, default=1, help="First sector to include")
    ap.add_argument("--max-sector", type=int, default=120, help="Last sector to include (upper bound)")
    ap.add_argument("--no-cloud", action="store_true", help="Disable AWS cloud mirror")
    ap.add_argument("--retries", type=int, default=4, help="Max per-sector retries")
    args = ap.parse_args()

    print("[INFO] Querying MAST via Astroquery (TESS, timeseries)…")
    df = fetch_sector_batched(
        min_sector=args.min_sector,
        max_sector=args.max_sector,
        use_cloud=not args.no_cloud,
        max_retries=args.retries
    )

    df.to_csv(args.out, index=False)
    print(f"[DONE] Wrote {len(df)} sectors to {args.out}")


if __name__ == "__main__":
    main()