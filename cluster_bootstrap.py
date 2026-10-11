#!/usr/bin/env python3
"""Outlet-level (cluster) bootstrap and sign-flip test for comparing two systems.

Why: several gold queries target the same outlet, so queries are not independent.
Resampling whole outlets gives honest confidence intervals for the paired difference.

Input: a per-query CSV (default reports/per_query.csv), one row per (system, query):
    query_id, system, outlet_key, hit1, hit5, rr
  hit1 / hit5 are 0/1 and rr is the reciprocal rank (0 on a miss). They must come
  from the same code that produced eval_report.json, so the means reproduce it.
  outlet_key is the query's gold outlet (the cluster).

Output: a table on screen and reports/cluster_bootstrap.json.
  - naive CI:   resamples queries (assumes independence; shown for comparison)
  - cluster CI: resamples whole outlets (the honest one)
  - sign-flip p: flips the sign of each outlet's total difference; exact when there
                 are <= 16 outlets, Monte Carlo otherwise

Usage:
    python cluster_bootstrap.py
    python cluster_bootstrap.py --treat "B|exact" --base "A800|exact" --n-boot 20000
"""
import argparse
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd

METRICS = {"Hit@1": "hit1", "Hit@5": "hit5", "MRR": "rr"}
REQUIRED = ["query_id", "system", "outlet_key", "hit1", "hit5", "rr"]
EXACT_MAX_CLUSTERS = 16
MC_SIGN_DRAWS = 200_000


def load_pair(path, treat, base):
    df = pd.read_csv(path, dtype={"query_id": str, "system": str, "outlet_key": str})
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        sys.exit(f"{path} is missing columns {missing}; expected {REQUIRED}")
    systems = sorted(df["system"].unique())
    for s in (treat, base):
        if s not in systems:
            sys.exit(f"system {s!r} not found in {path}. Available: {systems}")

    parts = {}
    for name in (treat, base):
        sub = df[df["system"] == name]
        if sub["query_id"].duplicated().any():
            sys.exit(f"duplicate query_id rows for system {name!r}")
        parts[name] = sub.set_index("query_id")
    t, b = parts[treat], parts[base]
    if set(t.index) != set(b.index):
        sys.exit("the two systems cover different query_ids; a paired comparison needs identical sets")
    qids = sorted(t.index)
    t, b = t.loc[qids], b.loc[qids]
    if not (t["outlet_key"] == b["outlet_key"]).all():
        sys.exit("outlet_key differs between the systems for the same query_id")
    return t, b


def percentile_ci(x):
    lo, hi = np.percentile(x, [2.5, 97.5])
    return [float(lo), float(hi)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", default="reports/per_query.csv")
    ap.add_argument("--treat", default="B|exact", help="system being tested")
    ap.add_argument("--base", default="A800|exact", help="baseline system")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="reports/cluster_bootstrap.json")
    args = ap.parse_args()

    t, b = load_pair(args.file, args.treat, args.base)
    n = len(t)
    codes, outlets = pd.factorize(t["outlet_key"])
    c = len(outlets)
    if c < 2:
        sys.exit("need at least 2 outlets for a cluster bootstrap")
    counts = np.bincount(codes, minlength=c).astype(float)

    rng = np.random.default_rng(args.seed)
    cluster_idx = rng.integers(0, c, size=(args.n_boot, c))   # same draws for every metric
    query_idx = rng.integers(0, n, size=(args.n_boot, n))

    if c <= EXACT_MAX_CLUSTERS:
        signs = np.array(list(itertools.product((-1.0, 1.0), repeat=c)))
        perm_kind = f"exact, {len(signs)} sign patterns"
    else:
        signs = rng.choice((-1.0, 1.0), size=(MC_SIGN_DRAWS, c))
        perm_kind = f"Monte Carlo, {MC_SIGN_DRAWS} sign draws"

    results = {}
    for label, col in METRICS.items():
        tv, bv = t[col].to_numpy(float), b[col].to_numpy(float)
        d = tv - bv
        observed = float(d.mean())
        sums = np.bincount(codes, weights=d, minlength=c)  # per-outlet total difference

        naive = d[query_idx].mean(axis=1)
        cluster = sums[cluster_idx].sum(axis=1) / counts[cluster_idx].sum(axis=1)
        flipped = (signs @ sums) / n
        p = float(np.mean(np.abs(flipped) >= abs(observed) - 1e-12))

        results[label] = {
            "treat_mean": float(tv.mean()),
            "base_mean": float(bv.mean()),
            "diff": observed,
            "naive_ci": percentile_ci(naive),
            "cluster_ci": percentile_ci(cluster),
            "signflip_p": p,
        }

    print(f"{args.treat}  minus  {args.base}   |   {n} queries, {c} outlets, "
          f"{args.n_boot} resamples, seed {args.seed}")
    print(f"{'metric':<7} {'treat':>6} {'base':>6} {'diff':>7}   {'naive 95% CI':<18} "
          f"{'outlet-cluster 95% CI':<24} {'sign-flip p':>11}")
    for label, r in results.items():
        nlo, nhi = r["naive_ci"]
        clo, chi = r["cluster_ci"]
        verdict = "excludes 0" if (clo > 0 or chi < 0) else "includes 0"
        print(f"{label:<7} {r['treat_mean']:>6.3f} {r['base_mean']:>6.3f} {r['diff']:>+7.3f}   "
              f"[{nlo:+.3f}, {nhi:+.3f}]    [{clo:+.3f}, {chi:+.3f}] {verdict:<10} {r['signflip_p']:>8.4f}")
    print(f"sign-flip test: {perm_kind}")
    print("Check: treat/base means should match eval_report.json to 3 decimals.")

    payload = {
        "treat": args.treat, "base": args.base,
        "n_queries": n, "n_outlets": c,
        "n_boot": args.n_boot, "seed": args.seed,
        "signflip": perm_kind, "metrics": results,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()