#!/usr/bin/env python3
"""
evaluate.py -- Phase 4: Strategy A vs Strategy B, scored at OUTLET level (Hit@k, MRR)

    python evaluate.py --gold eval/gold_queries.jsonl --variants B A800

Gold file (JSON lines, one query per line):
    {"query_id": "q001", "query": "quiet place for a date night, slow service though",
     "query_type": "review_evidence",                 # review_evidence | profile_fact | mixed
     "gold_outlet_keys": ["a1b2c3d4e5f6"],            # outlet_key(s) from chunks_*.parquet / index_input.parquet; NEVER url
     "source_review_id": "0123456789abcdef"}          # optional: review the query was written from

What this enforces (aborts with the full problem list if the gold file breaks a rule):
  * every gold outlet has indexed reviews (otherwise Strategy A cannot retrieve it: unfair by construction)
  * source reviews exist in the shared input frame and are not residual-mojibake
  * unique ids, known query types, no duplicate queries, at least --min-queries queries
What it measures:
  * PRIMARY (fixed in advance): B vs A800, exact engine, ΔHit@5, cluster-bootstrap 95% CI. Everything else is exploratory.
  * exact numpy search is the oracle; Chroma (if its collection exists) is scored the same way and the two engines are
    compared; Chroma never decides a verdict.
  * control: B_restricted = B with outlets that have no indexed reviews removed from results (A cannot return them).
  * leakage: queries sharing a --leak-ngram-word run with a gold outlet's text are flagged and re-scored without them.
  * uncertainty: bootstrap over OUTLET clusters (queries about the same outlet are not independent), exact McNemar on
    Hit@5 discordant pairs, per-type breakdown (CI only for n >= 10).
Requires: pandas, numpy, pyarrow, sentence-transformers (query embedding), chromadb (optional engine)
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from embed_chunks import VARIANT_FILES, STEmbedder, sha256_file
from exact_search import exact_topk
from load_chroma import verify_alignment
from parity_test import chroma_search_fn

QUERY_TYPES = ("review_evidence", "profile_fact", "mixed")
HEX12 = re.compile(r"^[0-9a-f]{12}$")
METRICS = {"hit1": "Hit@1", "hit5": "Hit@5", "hit10": "Hit@10", "rr": "MRR", "rr10": "MRR@10"}


@dataclass
class VariantData:
    name: str
    E: np.ndarray
    outlet_of: np.ndarray
    chunk_type: np.ndarray
    chunk_id: np.ndarray
    text: np.ndarray | None = None


# ------------------------------------------------------------------------- gold file
def load_gold(path: Path) -> pd.DataFrame:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return pd.DataFrame(rows)


def validate_gold(gold: pd.DataFrame, outlets_with_reviews: set, reviews: pd.DataFrame | None = None,
                  min_queries: int = 30) -> list[str]:
    """-> list of human-readable problems (empty list == gold set is acceptable)."""
    need = {"query_id", "query", "query_type", "gold_outlet_keys"}
    if need - set(gold.columns):
        return [f"missing columns: {sorted(need - set(gold.columns))}"]
    probs: list[str] = []
    if len(gold) < min_queries:
        probs.append(f"only {len(gold)} queries (< {min_queries}): differences would be noise")
    for qid in gold["query_id"][gold["query_id"].duplicated()].unique():
        probs.append(f"duplicate query_id {qid!r}")
    norm = gold["query"].astype(str).str.lower().str.replace(r"\W+", " ", regex=True).str.strip()
    for qid in gold.loc[norm.duplicated(keep=False), "query_id"]:
        probs.append(f"{qid}: duplicate query text")
    has_src = "source_review_id" in gold.columns
    for r in gold.itertuples(index=False):
        qid = r.query_id
        if not isinstance(r.query, str) or len(r.query.split()) < 3:
            probs.append(f"{qid}: query is empty or shorter than 3 words")
        if r.query_type not in QUERY_TYPES:
            probs.append(f"{qid}: query_type {r.query_type!r} not in {QUERY_TYPES}")
        keys = r.gold_outlet_keys
        if not isinstance(keys, (list, tuple)) or not keys or len(set(keys)) != len(keys):
            probs.append(f"{qid}: gold_outlet_keys must be a non-empty list without duplicates")
            continue
        for k in keys:
            if not isinstance(k, str) or not HEX12.match(k):
                probs.append(f"{qid}: gold key {k!r} is not a 12-hex outlet_key (url / outlet_id are not allowed)")
            elif k not in outlets_with_reviews:
                probs.append(f"{qid}: gold outlet {k} has no indexed reviews (Strategy A cannot retrieve it) -> relabel")
        src = getattr(r, "source_review_id", None) if has_src else None
        if isinstance(src, str) and src and reviews is not None:
            if src not in reviews.index:
                probs.append(f"{qid}: source_review_id {src} not in the shared input frame (dropped as spam, or typo)")
            else:
                row = reviews.loc[src]
                if row["outlet_key"] not in keys:
                    probs.append(f"{qid}: source review belongs to {row['outlet_key']}, not a gold outlet")
                if bool(row["mojibake_residual"]):
                    probs.append(f"{qid}: source review is residual-mojibake (excluded from gold by rule)")
    return probs


def _words(s: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", s.lower())


def _ngrams(words: list[str], n: int) -> set:
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def leaky_flags(gold: pd.DataFrame, corpus_by_outlet: dict, n: int = 5) -> np.ndarray:
    """True where the query shares an n-word run with any text of a gold outlet (copy-paste / near copy)."""
    cache: dict = {}
    flags = []
    for q, keys in zip(gold["query"], gold["gold_outlet_keys"]):
        grams = _ngrams(_words(q), n)
        leak = False
        for k in keys:
            if k not in cache:
                cache[k] = set().union(*[_ngrams(_words(t), n) for t in corpus_by_outlet.get(k, [])]) \
                    if corpus_by_outlet.get(k) else set()
            if grams & cache[k]:
                leak = True
                break
        flags.append(leak)
    return np.array(flags, dtype=bool)


# ---------------------------------------------------------------------------- scoring
def rank_outlets(rows, outlet_of, allowed=None) -> tuple[list, list]:
    """Chunk rows in rank order -> (unique outlets in rank order, the best-ranked chunk row of each)."""
    seen, outs, best = set(), [], []
    for r in rows:
        o = outlet_of[r]
        if o in seen or (allowed is not None and o not in allowed):
            continue
        seen.add(o)
        outs.append(o)
        best.append(int(r))
    return outs, best


def score_lists(row_lists, outlet_of, gold_sets, allowed=None, depth: int | None = None) -> dict:
    m = len(gold_sets)
    s = {k: np.zeros(m) for k in METRICS}
    s.update(n_outlets=np.zeros(m, dtype=int), rank=np.full(m, np.nan),
             first_gold_row=np.full(m, -1, dtype=int), top1_row=np.full(m, -1, dtype=int))
    for q, rows in enumerate(row_lists):
        rows = list(rows)[:depth] if depth else list(rows)
        outs, best = rank_outlets(rows, outlet_of, allowed)
        s["n_outlets"][q] = len(outs)
        if best:
            s["top1_row"][q] = best[0]
        r = next((i + 1 for i, o in enumerate(outs) if o in gold_sets[q]), None)
        if r is not None:
            s["rank"][q], s["first_gold_row"][q] = r, best[r - 1]
            s["hit1"][q], s["hit5"][q], s["hit10"][q] = r == 1, r <= 5, r <= 10
            s["rr"][q], s["rr10"][q] = 1.0 / r, (1.0 / r if r <= 10 else 0.0)
    return s


# ----------------------------------------------------------------------- statistics
def make_weights(cluster_ids, n_boot: int = 10_000, seed: int = 0) -> np.ndarray:
    """Cluster-bootstrap weights (n_boot, n_queries): resample OUTLETS with replacement; every query of a drawn
    outlet comes along. Same weights for every metric/system, so paired differences stay paired."""
    uniq, inv = np.unique(np.asarray(cluster_ids), return_inverse=True)
    nc, rng = len(uniq), np.random.default_rng(seed)
    counts = np.stack([np.bincount(rng.integers(0, nc, nc), minlength=nc) for _ in range(n_boot)])
    return counts[:, inv].astype(np.float64)


def mean_ci(W: np.ndarray, x, mask=None, with_ci: bool = True) -> dict:
    x = np.asarray(x, dtype=float)
    if mask is not None:
        W, x = W[:, mask], x[mask]
    n = len(x)
    if n == 0:
        return {"mean": None, "ci": None, "n": 0}
    out = {"mean": float(x.mean()), "ci": None, "n": int(n)}
    if with_ci:
        den = W.sum(axis=1)
        ok = den > 0
        lo, hi = np.percentile((W[ok] @ x) / den[ok], [2.5, 97.5])
        out["ci"] = [float(lo), float(hi)]
    return out


def mcnemar_exact_p(b: int, c: int) -> float:
    """Two-sided exact McNemar (binomial) p-value from the discordant counts."""
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def summarize(s: dict, W: np.ndarray, mask=None, with_ci: bool = True) -> dict:
    m = np.ones(len(s["hit5"]), bool) if mask is None else mask
    out = {name: mean_ci(W, s[key], m, with_ci) for key, name in METRICS.items()}
    out["short_list_rate_lt5_outlets"] = float((s["n_outlets"][m] < 5).mean()) if m.any() else None
    out["outlets_returned_mean"] = float(s["n_outlets"][m].mean()) if m.any() else None
    return out


def compare(sa: dict, sb: dict, W: np.ndarray, mask=None) -> dict:
    """b minus a (a = baseline)."""
    m = np.ones(len(sa["hit5"]), bool) if mask is None else mask
    ha, hb, ra, rb = sa["hit5"][m], sb["hit5"][m], sa["rr"][m], sb["rr"][m]
    b_only, a_only = int(((hb == 1) & (ha == 0)).sum()), int(((ha == 1) & (hb == 0)).sum())
    return {
        "n": int(m.sum()),
        "hit5_diff": mean_ci(W, sb["hit5"] - sa["hit5"], m),
        "mrr_diff": mean_ci(W, sb["rr"] - sa["rr"], m),
        "hit5_discordant": {"b_only": b_only, "a_only": a_only, "mcnemar_exact_p": mcnemar_exact_p(b_only, a_only)},
        "rr_wins_losses_ties": {"b_better": int((rb > ra).sum()), "a_better": int((ra > rb).sum()),
                                "tie": int((ra == rb).sum())},
    }


def verdict(diff: dict, a: str = "A800", b: str = "B") -> str:
    if diff["ci"] is None:
        return "no data"
    lo, hi = diff["ci"]
    if lo > 0:
        return f"{b} > {a}: 95% CI excludes 0"
    if hi < 0:
        return f"{a} > {b}: 95% CI excludes 0"
    return "inconclusive: 95% CI includes 0"


# ----------------------------------------------------------------------- core (pure)
def evaluate(variants: dict, Q: np.ndarray, gold: pd.DataFrame, leaky: np.ndarray, outlets_with_reviews: set,
             chroma_rows: dict | None = None, k_fetch: int = 150, sens_depths=(50, 100), n_boot: int = 10_000,
             seed: int = 0) -> tuple[dict, pd.DataFrame]:
    gold_sets = [set(k) for k in gold["gold_outlet_keys"]]
    qtype = gold["query_type"].to_numpy()
    W = make_weights([k[0] for k in gold["gold_outlet_keys"]], n_boot, seed)

    systems: dict = {}  # name -> dict(scores, rows, variant, allowed)
    for vn, v in variants.items():
        ex = exact_topk(v.E, Q, k_fetch)[0]
        systems[f"{vn}|exact"] = {"rows": ex, "v": v, "allowed": None}
        if chroma_rows and vn in chroma_rows:
            systems[f"{vn}|chroma"] = {"rows": chroma_rows[vn], "v": v, "allowed": None}
        if vn == "B" and "A800" in variants:
            systems["B_restricted|exact"] = {"rows": ex, "v": v, "allowed": outlets_with_reviews}
    for s in systems.values():
        s["scores"] = score_lists(s["rows"], s["v"].outlet_of, gold_sets, s["allowed"])

    report: dict = {
        "meta": {"n_queries": len(gold), "n_outlet_clusters": int(len({k[0] for k in gold["gold_outlet_keys"]})),
                 "query_type_counts": {t: int((qtype == t).sum()) for t in QUERY_TYPES},
                 "leaky_queries": int(leaky.sum()), "k_fetch_chunks": k_fetch, "n_boot": n_boot, "seed": seed,
                 "systems": list(systems)},
        "systems": {n: summarize(s["scores"], W) for n, s in systems.items()},
        "by_type": {}, "exploratory": {}, "engine_agreement": {},
    }
    for t in QUERY_TYPES:
        mt = qtype == t
        if mt.any():
            report["by_type"][t] = {n: summarize(s["scores"], W, mt, with_ci=bool(mt.sum() >= 10))
                                    for n, s in systems.items()}

    sc = {n: s["scores"] for n, s in systems.items()}
    if "A800|exact" in sc and "B|exact" in sc:
        prim = compare(sc["A800|exact"], sc["B|exact"], W)
        prim["verdict_hit5"] = verdict(prim["hit5_diff"])
        prim["verdict_mrr"] = verdict(prim["mrr_diff"])
        report["primary"] = {"definition": "B minus A800, exact engine, all queries", **prim}
        ex = report["exploratory"]
        if "B_restricted|exact" in sc:
            ex["B_restricted_vs_A800"] = compare(sc["A800|exact"], sc["B_restricted|exact"], W)
        if leaky.any():
            ex["B_vs_A800_excluding_leaky"] = compare(sc["A800|exact"], sc["B|exact"], W, ~leaky)
        ex["B_vs_A800_by_type"] = {t: compare(sc["A800|exact"], sc["B|exact"], W, qtype == t)
                                   for t in QUERY_TYPES if (qtype == t).any()}
        ex["depth_sensitivity"] = {}
        for d in sens_depths:
            sa = score_lists(systems["A800|exact"]["rows"], systems["A800|exact"]["v"].outlet_of, gold_sets, depth=d)
            sb = score_lists(systems["B|exact"]["rows"], systems["B|exact"]["v"].outlet_of, gold_sets, depth=d)
            ex["depth_sensitivity"][f"top{d}_chunks"] = compare(sa, sb, W)
    for vn in variants:
        if f"{vn}|chroma" in sc:
            a, b = sc[f"{vn}|exact"], sc[f"{vn}|chroma"]
            report["engine_agreement"][vn] = {
                "queries_where_hit5_differs": int((a["hit5"] != b["hit5"]).sum()),
                "hit5_exact": float(a["hit5"].mean()), "hit5_chroma": float(b["hit5"].mean()),
                "mrr_exact": float(a["rr"].mean()), "mrr_chroma": float(b["rr"].mean())}
    report["caveats"] = [
        "Gold outlets are restricted to outlets with indexed reviews; outlets without reviews exist only in B.",
        "Queries are written by one person: results are about this query set, not about users in general.",
        "Metrics are micro-averaged over queries; check by_type before quoting a single headline number.",
        "McNemar ignores outlet clustering; the cluster bootstrap CI is the primary uncertainty statement.",
        "Hit@k needs k distinct outlets: see short_list_rate_lt5_outlets (retrieval depth is a real cost of clustering).",
    ]

    recs = []
    for n, s in systems.items():
        v, sc_ = s["v"], s["scores"]
        for q in range(len(gold)):
            fg, t1 = int(sc_["first_gold_row"][q]), int(sc_["top1_row"][q])
            recs.append({
                "query_id": gold["query_id"].iat[q], "query_type": qtype[q], "leaky": bool(leaky[q]), "system": n,
                "rank": None if np.isnan(sc_["rank"][q]) else int(sc_["rank"][q]), "hit5": int(sc_["hit5"][q]),
                "rr": float(sc_["rr"][q]), "n_outlets": int(sc_["n_outlets"][q]),
                "first_gold_chunk_type": v.chunk_type[fg] if fg >= 0 else None,
                "first_gold_chunk_id": v.chunk_id[fg] if fg >= 0 else None,
                "top1_chunk_id": v.chunk_id[t1] if t1 >= 0 else None,
                "top1_is_gold": bool(t1 >= 0 and v.outlet_of[t1] in gold_sets[q]),
                "top1_snippet": (v.text[t1][:200] if (v.text is not None and t1 >= 0) else None),
            })
    return report, pd.DataFrame(recs)


def format_summary(report: dict) -> str:
    def cell(d):
        return "n/a" if d["mean"] is None else (f"{d['mean']:.3f} [{d['ci'][0]:.3f}, {d['ci'][1]:.3f}]" if d["ci"]
                                                else f"{d['mean']:.3f}")
    m = report["meta"]
    lines = [f"queries={m['n_queries']} (outlet clusters={m['n_outlet_clusters']}, leaky={m['leaky_queries']}), "
             f"types={m['query_type_counts']}", ""]
    lines.append(f"{'system':<20}{'Hit@1':<26}{'Hit@5':<26}{'MRR':<26}short<5")
    for n, s in report["systems"].items():
        lines.append(f"{n:<20}{cell(s['Hit@1']):<26}{cell(s['Hit@5']):<26}{cell(s['MRR']):<26}"
                     f"{s['short_list_rate_lt5_outlets']:.2f}")
    p = report.get("primary")
    if p:
        d, r = p["hit5_diff"], p["mrr_diff"]
        lines += ["", "PRIMARY (B - A800, exact engine):",
                  f"  dHit@5 = {d['mean']:+.3f} [{d['ci'][0]:+.3f}, {d['ci'][1]:+.3f}]  -> {p['verdict_hit5']}"
                  f"   (McNemar p={p['hit5_discordant']['mcnemar_exact_p']:.4f}, "
                  f"B-only {p['hit5_discordant']['b_only']} / A-only {p['hit5_discordant']['a_only']})",
                  f"  dMRR   = {r['mean']:+.3f} [{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}]  -> {p['verdict_mrr']}"]
    return "\n".join(lines)


# --------------------------------------------------------------------------------- CLI
def check_manifests(manifests: dict) -> None:
    keys = ["model", "model_commit_hash", "max_seq_length", "passage_prefix", "query_prefix", "dim", "normalize"]
    ref_name, ref = next(iter(manifests.items()))
    for n, m in manifests.items():
        diff = {k: (ref.get(k), m.get(k)) for k in keys if ref.get(k) != m.get(k)}
        if diff:
            raise SystemExit(f"Embeddings of {ref_name} and {n} were not built identically: {diff}")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gold", type=Path, default=Path("eval/gold_queries.jsonl"))
    p.add_argument("--chunks-dir", type=Path, default=Path("data/processed"))
    p.add_argument("--index-input", type=Path, default=Path("data/processed/index_input.parquet"))
    p.add_argument("--emb-dir", type=Path, default=Path("data/embeddings"))
    p.add_argument("--chroma-dir", type=Path, default=Path("data/chroma"))
    p.add_argument("--report-dir", type=Path, default=Path("reports"))
    p.add_argument("--run-dir", type=Path, default=Path("data/eval_runs"),
                   help="per-query CSV incl. dataset text snippets (keep out of git)")
    p.add_argument("--variants", nargs="+", default=["B", "A800"], choices=list(VARIANT_FILES))
    p.add_argument("--k-fetch", type=int, default=150, help="chunks retrieved per query before collapsing to outlets")
    p.add_argument("--sens-depths", type=int, nargs="*", default=[50, 100])
    p.add_argument("--leak-ngram", type=int, default=5)
    p.add_argument("--min-queries", type=int, default=30)
    p.add_argument("--n-boot", type=int, default=10_000)
    p.add_argument("--skip-chroma", action="store_true")
    p.add_argument("--device", default=None)
    p.add_argument("--threads", type=int, default=None)
    return p.parse_args(argv)


def main(argv=None) -> int:
    a = parse_args(argv)
    gold = load_gold(a.gold)
    idx = pd.read_parquet(a.index_input, columns=["review_id", "outlet_key", "text", "mojibake_residual"])
    outlets_with_reviews = set(idx["outlet_key"])
    problems = validate_gold(gold, outlets_with_reviews, idx.set_index("review_id"), a.min_queries)
    if problems:
        print(f"GOLD SET REJECTED ({len(problems)} problems):", file=sys.stderr)
        print("\n".join(f"  - {p}" for p in problems), file=sys.stderr)
        return 2

    variants, manifests = {}, {}
    for vn in a.variants:
        path = a.chunks_dir / VARIANT_FILES[vn]
        manifests[vn] = json.loads((a.emb_dir / f"{vn}.manifest.json").read_text())
        df = pd.read_parquet(path, columns=["chunk_id", "outlet_key", "chunk_type", "text"])
        E = np.load(a.emb_dir / f"{vn}.npy")
        verify_alignment(df, E, manifests[vn], sha256_file(path))
        variants[vn] = VariantData(vn, E, df["outlet_key"].to_numpy(), df["chunk_type"].to_numpy(),
                                   df["chunk_id"].to_numpy(), df["text"].to_numpy())
    check_manifests(manifests)

    gold_keys = {k for ks in gold["gold_outlet_keys"] for k in ks}
    corpus = {k: g["text"].tolist() for k, g in idx[idx["outlet_key"].isin(gold_keys)].groupby("outlet_key")}
    if "B" in variants:
        v = variants["B"]
        prof = (v.chunk_type == "profile") & np.isin(v.outlet_of, list(gold_keys))
        for k, t in zip(v.outlet_of[prof], v.text[prof]):
            corpus.setdefault(k, []).append(t)
    leaky = leaky_flags(gold, corpus, a.leak_ngram)

    ref = next(iter(manifests.values()))
    emb = STEmbedder(ref["model"], ref.get("model_commit_hash"), a.device, None, a.threads)
    Q = emb.encode([ref["query_prefix"] + q for q in gold["query"]])

    chroma_rows = {}
    if not a.skip_chroma:
        import chromadb
        client = chromadb.PersistentClient(path=str(a.chroma_dir))
        for vn, v in variants.items():
            try:
                col = client.get_collection(name=f"zomato_{vn}", embedding_function=None)
            except Exception:  # noqa: BLE001
                print(f"[{vn}] no Chroma collection -> exact engine only", file=sys.stderr)
                continue
            if col.count() != len(v.chunk_id):
                raise SystemExit(f"[{vn}] collection has {col.count()} rows, expected {len(v.chunk_id)}.")
            id_to_row = {cid: i for i, cid in enumerate(v.chunk_id)}
            chroma_rows[vn], _ = chroma_search_fn(col, id_to_row)(Q, a.k_fetch)

    report, per_query = evaluate(variants, Q, gold, leaky, outlets_with_reviews, chroma_rows, a.k_fetch,
                                 a.sens_depths, a.n_boot)
    report["meta"].update(gold_file=str(a.gold), gold_sha256=sha256_file(a.gold),
                          model=ref["model"], model_commit_hash=ref.get("model_commit_hash"),
                          created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    a.report_dir.mkdir(parents=True, exist_ok=True)
    a.run_dir.mkdir(parents=True, exist_ok=True)
    (a.report_dir / "eval_report.json").write_text(json.dumps(report, indent=2))
    per_query.to_csv(a.run_dir / "eval_per_query.csv", index=False, encoding="utf-8-sig")
    print(format_summary(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())