#!/usr/bin/env python3
"""
diagnose_outlets.py -- why is the composite outlet id not unique?

    python diagnose_outlets.py zomato_cleaned.parquet --name name --address address --url url --votes votes

Read-only. Does NOT use outlet_key.py: it applies a plain lower-case + whitespace-collapse to name/address,
so you can compare its counts with what build_outlet_id produced. Changes nothing on disk.
"""
from __future__ import annotations

import argparse
import re
import sys

import pandas as pd

WS = re.compile(r"\s+")


def _norm(v) -> str:
    return WS.sub(" ", str(v)).strip().lower()


def diagnose(df: pd.DataFrame, name: str, address: str, url: str | None = None,
             votes: str | None = None, expect: int = 12_499) -> dict:
    out: dict = {"rows": len(df)}
    print(f"rows                         : {len(df):,}   (Day-1 outlet table should be ~{expect:,})")
    print(f"columns                      : {list(df.columns)}")
    for c in (name, address):
        if c not in df.columns:
            raise SystemExit(f"column {c!r} not in file")
    print(f"null name / null address     : {int(df[name].isna().sum())} / {int(df[address].isna().sum())}")

    raw_pairs = len(df[[name, address]].astype(str).drop_duplicates())
    ids = df[name].map(_norm) + "\x1f" + df[address].map(_norm)
    out.update(unique_raw_pairs=raw_pairs, unique_norm_ids=int(ids.nunique()))
    print(f"unique (name,address) raw    : {raw_pairs:,}")
    print(f"unique normalised ids        : {ids.nunique():,}")
    if url and url in df.columns:
        out["unique_urls"] = int(df[url].nunique())
        print(f"unique {url:<22}: {df[url].nunique():,}")

    dup = ids.duplicated(keep=False)
    n_groups = int(ids[dup].nunique())
    out.update(dup_rows=int(dup.sum()), dup_groups=n_groups)
    print(f"rows sharing an id           : {int(dup.sum()):,}  in {n_groups:,} groups")
    if not dup.any():
        print("-> ids ARE unique with plain normalisation; the problem is how build_outlet_id is being called.")
        return out

    d = df[dup].assign(_id=ids[dup]).copy()
    # which columns differ inside duplicate groups? (list/array cells compared as text)
    share = {}
    for c in df.columns:
        s = d[c].astype(str)
        share[c] = float((s.groupby(d["_id"]).nunique() > 1).mean())
    differing = {c: round(v, 3) for c, v in sorted(share.items(), key=lambda kv: -kv[1]) if v > 0}
    out["differing_columns_share"] = differing
    print("\nshare of duplicate groups in which each column DIFFERS (0 = identical copies):")
    for c, v in differing.items():
        print(f"  {c:<32}{v:6.1%}")
    if not differing:
        print("  (none: the duplicate rows are exact copies)")

    print("\nfirst 3 duplicate groups:")
    show = [c for c in (name, address, url, votes) if c and c in df.columns]
    for gid in list(d["_id"].drop_duplicates())[:3]:
        print(d.loc[d["_id"] == gid, show].to_string(index=False), "\n")
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("file")
    p.add_argument("--name", default="name")
    p.add_argument("--address", default="address")
    p.add_argument("--url", default="url")
    p.add_argument("--votes", default="votes")
    p.add_argument("--expect", type=int, default=12_499)
    a = p.parse_args(argv)
    df = pd.read_parquet(a.file) if a.file.endswith(".parquet") else pd.read_csv(a.file)
    diagnose(df, a.name, a.address, a.url, a.votes, a.expect)
    return 0


if __name__ == "__main__":
    sys.exit(main())