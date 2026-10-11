#!/usr/bin/env python3
"""
parity_test.py -- Phase 3, step 2b: does Chroma's approximate (HNSW) search return what exact search returns?

    python parity_test.py --variants B A800

Queries are random chunk vectors from the variant itself (no model needed). For each query both engines fetch the
top-K; then, identically on both sides:
  * chunk level : the query's own chunk is dropped; recall@10 / recall@50 of Chroma vs exact
  * outlet level: the query's own OUTLET is dropped, the rest is collapsed to unique outlets in rank order,
                  and the top-5 outlets are compared. This is the unit Hit@5 / MRR are scored on, so it is the
                  number that decides whether ANN error can contaminate the A-vs-B comparison.
Exit code 3 if a gate fails (report is still written to --report-dir). Same gates for every variant.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from embed_chunks import VARIANT_FILES, sha256_file
from exact_search import collapse_to_outlets, exact_topk, overlap_at_k
from load_chroma import verify_alignment


def infer_metric(E, qi, ch_rows, ch_dist, n_queries: int = 20, tol: float = 1e-3) -> str:
    """Chroma reports 1-dot for cosine/ip and squared L2 (2-2*dot) for l2, on unit vectors. Ranking is identical
    for all three on normalised embeddings; this only tells you which one the collection actually uses."""
    dots, ds = [], []
    for q in range(min(n_queries, len(qi))):
        for r, d in zip(ch_rows[q][:5], ch_dist[q][:5]):
            dots.append(float(E[qi[q]] @ E[r]))
            ds.append(float(d))
    dots, ds = np.array(dots), np.array(ds)
    if np.allclose(ds, 1 - dots, atol=tol):
        return "cosine_or_ip"
    if np.allclose(ds, 2 - 2 * dots, atol=tol):
        return "l2_squared"
    return "unknown"


def run_parity(search_fn, E: np.ndarray, outlet_of, n_queries: int = 500, k_fetch: int = 150, seed: int = 0,
               n_outlets: int = 5) -> dict:
    """search_fn(Q, k) -> (rows[list[list[int]]], distances[list[list[float]]]) for the approximate engine."""
    rng = np.random.default_rng(seed)
    qi = np.sort(rng.choice(len(E), size=min(n_queries, len(E)), replace=False))
    Q = E[qi]

    t = time.perf_counter()
    ex_idx, _ = exact_topk(E, Q, k_fetch + 1)
    exact_ms = (time.perf_counter() - t) * 1000 / len(qi)
    t = time.perf_counter()
    ch_rows, ch_dist = search_fn(Q, k_fetch + 1)
    ann_ms = (time.perf_counter() - t) * 1000 / len(qi)

    r10, r50, top1, o_ov, o_top1, short = [], [], [], [], [], 0
    for q, row in enumerate(qi):
        own = outlet_of[row]
        ex = [int(r) for r in ex_idx[q] if r != row]
        ch = [int(r) for r in ch_rows[q] if r != row]
        r10.append(overlap_at_k(ex, ch, 10))
        r50.append(overlap_at_k(ex, ch, 50))
        top1.append(float(ex[:1] == ch[:1]))
        ex_o = collapse_to_outlets([r for r in ex if outlet_of[r] != own], outlet_of, n_outlets)
        ch_o = collapse_to_outlets([r for r in ch if outlet_of[r] != own], outlet_of, n_outlets)
        short += len(ch_o) < n_outlets
        o_ov.append(len(set(ex_o) & set(ch_o)) / n_outlets)
        o_top1.append(float(ex_o[:1] == ch_o[:1]))

    return {
        "n_queries": int(len(qi)), "k_fetch": k_fetch,
        "chunk_recall@10_mean": float(np.mean(r10)), "chunk_recall@10_min": float(np.min(r10)),
        "chunk_recall@50_mean": float(np.mean(r50)), "chunk_top1_agreement": float(np.mean(top1)),
        f"outlet_top{n_outlets}_overlap_mean": float(np.mean(o_ov)),
        f"outlet_top{n_outlets}_overlap_min": float(np.min(o_ov)),
        "outlet_top1_agreement": float(np.mean(o_top1)),
        "queries_with_short_ann_outlet_list": int(short),
        "metric_inferred": infer_metric(E, qi, ch_rows, ch_dist),
        "exact_ms_per_query": round(exact_ms, 2), "ann_ms_per_query": round(ann_ms, 2),
    }


def evaluate_gates(res: dict, min_recall10: float, min_outlet_overlap: float, n_outlets: int = 5) -> dict:
    checks = {
        f"chunk_recall@10_mean >= {min_recall10}": res["chunk_recall@10_mean"] >= min_recall10,
        f"outlet_top{n_outlets}_overlap_mean >= {min_outlet_overlap}":
            res[f"outlet_top{n_outlets}_overlap_mean"] >= min_outlet_overlap,
        "no short ANN outlet lists": res["queries_with_short_ann_outlet_list"] == 0,
    }
    return {"checks": checks, "passed": all(checks.values())}


def chroma_search_fn(col, id_to_row: dict, block: int = 100):
    def fn(Q, k):
        rows, dists = [], []
        for lo in range(0, len(Q), block):
            res = col.query(query_embeddings=Q[lo:lo + block].tolist(), n_results=k, include=["distances"])
            for ids, ds in zip(res["ids"], res["distances"]):
                rows.append([id_to_row[i] for i in ids])
                dists.append(ds)
        return rows, dists
    return fn


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--chunks-dir", type=Path, default=Path("data/processed"))
    p.add_argument("--emb-dir", type=Path, default=Path("data/embeddings"))
    p.add_argument("--chroma-dir", type=Path, default=Path("data/chroma"))
    p.add_argument("--report-dir", type=Path, default=Path("reports"))
    p.add_argument("--variants", nargs="+", default=["B", "A800"], choices=[*VARIANT_FILES, "all"])
    p.add_argument("--n-queries", type=int, default=500)
    p.add_argument("--k-fetch", type=int, default=150, help="results fetched per query (needs ef_search > this)")
    p.add_argument("--min-recall10", type=float, default=0.98)
    p.add_argument("--min-outlet-overlap", type=float, default=0.98)
    return p.parse_args(argv)


def main(argv=None) -> int:
    import chromadb

    a = parse_args(argv)
    variants = list(VARIANT_FILES) if "all" in a.variants else a.variants
    a.report_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(a.chroma_dir))
    all_ok = True

    for v in variants:
        chunk_path = a.chunks_dir / VARIANT_FILES[v]
        manifest = json.loads((a.emb_dir / f"{v}.manifest.json").read_text())
        df = pd.read_parquet(chunk_path, columns=["chunk_id", "outlet_key"])
        E = np.load(a.emb_dir / f"{v}.npy")
        verify_alignment(df, E, manifest, sha256_file(chunk_path))

        col = client.get_collection(name=f"zomato_{v}", embedding_function=None)
        if col.count() != len(df):
            raise SystemExit(f"[{v}] collection holds {col.count()} rows, expected {len(df)}. Run load_chroma.py --rebuild.")
        load_info = {}
        lp = a.emb_dir / f"{v}.load.json"
        if lp.exists():
            load_info = json.loads(lp.read_text())
            ef = load_info.get("hnsw", {}).get("ef_search", 0)
            if ef <= a.k_fetch:
                print(f"[{v}] WARNING: ef_search={ef} <= k_fetch={a.k_fetch}; Chroma may error or truncate results.",
                      file=sys.stderr)

        id_to_row = {cid: i for i, cid in enumerate(df["chunk_id"])}
        res = run_parity(chroma_search_fn(col, id_to_row), E, df["outlet_key"].to_numpy(), a.n_queries, a.k_fetch)
        gates = evaluate_gates(res, a.min_recall10, a.min_outlet_overlap)
        report = {"variant": v, "rows": len(df), "hnsw": load_info.get("hnsw"),
                  "chroma_version": load_info.get("chroma_version"), **res, **gates,
                  "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        (a.report_dir / f"parity_{v}.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
        all_ok &= gates["passed"]

    if not all_ok:
        print("PARITY GATE FAILED: raise --ef-search / --ef-construction in load_chroma.py (--rebuild), identically "
              "for every variant, and re-run.", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())