#!/usr/bin/env python3
"""
cleaner.py - Zomato Bangalore: raw CSV -> one validated row per outlet.

    python cleaner.py                                  # zomato.csv -> zomato_cleaned.parquet + zomato_rag_ready.jsonl
    python cleaner.py --input data/zomato.csv --parquet out/outlets.parquet --jsonl out/outlets.jsonl
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pandera.pandas as pa

KEY = ["name", "address"]
CAT_COLS = ["listed_intype", "listed_incity"]

SCHEMA = pa.DataFrameSchema({
    "name": pa.Column(str, nullable=False),
    "address": pa.Column(str, nullable=False),
    "rate": pa.Column(float, pa.Check.in_range(0.0, 5.0), nullable=True),
    "approx_cost": pa.Column(float, pa.Check.greater_than(0.0), nullable=True),
    "is_new": pa.Column(bool),
})


def _require(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"column(s) not found after standardizing names: {missing}; have: {list(df.columns)}")


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (df.columns.str.lower()
                  .str.replace(" ", "_", regex=False)
                  .str.replace(r"[^a-z0-9_]", "", regex=True))
    return df.rename(columns={"approx_costfor_two_people": "approx_cost"})


def count_sentinels(df: pd.DataFrame) -> int:
    if "rate" not in df.columns:
        return 0
    raw = df["rate"].astype(str).str.strip()
    return int((raw == "NEW").sum() + (raw == "-").sum() + df["rate"].isna().sum())


def collapse_outlets(df: pd.DataFrame) -> pd.DataFrame:
    _require(df, KEY + CAT_COLS + ["votes"])
    df = df.copy()
    df["votes"] = pd.to_numeric(df["votes"], errors="coerce").fillna(0)

    cats = (df.groupby(KEY, sort=False)[CAT_COLS]
              .agg(lambda s: sorted(s.dropna().unique()))
              .reset_index())

    best = (df.sort_values("votes", ascending=False, kind="stable")
              .drop_duplicates(subset=KEY, keep="first")
              .drop(columns=CAT_COLS))

    if "location" in df.columns:
        loc = (df.groupby(KEY, sort=False)["location"]
                 .agg(lambda s: s.mode().iat[0] if s.notna().any() else None)
                 .rename("location").reset_index())
        best = best.drop(columns="location").merge(loc, on=KEY, how="left")

    return best.merge(cats, on=KEY, how="left")


def clean_rate_and_cost(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "rate" in df.columns:
        df["is_new"] = df["rate"].astype(str).str.strip().eq("NEW")
        df["rate"] = df["rate"].astype(str).str.replace("/5", "", regex=False).str.strip()
        df = df.replace(["NEW", "-"], np.nan)
        df["rate"] = pd.to_numeric(df["rate"], errors="coerce").astype(float)
    if "approx_cost" in df.columns:
        df["approx_cost"] = df["approx_cost"].astype(str).str.replace(",", "", regex=False)
        df["approx_cost"] = pd.to_numeric(df["approx_cost"], errors="coerce").astype(float)
    return df


def validate(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    # Pandera ko schema-level error crash se rokne ke liye required nulls pehle drop karein
    start_len = len(df)
    df = df.dropna(subset=["name", "address"])
    dropped = start_len - len(df)
    
    try:
        return SCHEMA.validate(df, lazy=True), dropped
    except pa.errors.SchemaErrors as err:
        idx = err.failure_cases["index"]
        if idx.isna().any():
            raise
        failed = idx.unique()
        return df.drop(index=failed), dropped + len(failed)


def clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    stats: dict = {"raw_rows": len(raw)}
    df = standardize_columns(raw)
    stats["sentinel_values"] = count_sentinels(df)
    stats["votes_missing_set_to_zero"] = (
        int(pd.to_numeric(df["votes"], errors="coerce").isna().sum()) if "votes" in df.columns else 0)
    df = collapse_outlets(df)
    stats["redundant_rows_removed"] = stats["raw_rows"] - len(df)
    df = clean_rate_and_cost(df)
    df, stats["invalid_rows_dropped"] = validate(df)
    stats["final_rows"] = len(df)
    return df, stats


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, default=Path("zomato.csv"))
    ap.add_argument("--parquet", type=Path, default=Path("zomato_cleaned.parquet"))
    ap.add_argument("--jsonl", type=Path, default=Path("zomato_rag_ready.jsonl"))
    a = ap.parse_args(argv)

    print(f"Loading {a.input} ...")
    df, s = clean(pd.read_csv(a.input))

    df.to_parquet(a.parquet, index=False)
    df.to_json(a.jsonl, orient="records", lines=True, force_ascii=False)

    print()
    print("=" * 50)
    print(f"Raw rows:                    {s['raw_rows']}")
    print(f"  Redundant rows collapsed:  {s['redundant_rows_removed']}")
    print(f"  Sentinel ratings handled:  {s['sentinel_values']}")
    print(f"  Missing votes set to 0:    {s['votes_missing_set_to_zero']}")
    print(f"  Invalid rows dropped:      {s['invalid_rows_dropped']}")
    print(f"Unique validated outlets:    {s['final_rows']}")
    print(f"Wrote {a.parquet} and {a.jsonl}")
    print("=" * 50)


if __name__ == "__main__":
    main()