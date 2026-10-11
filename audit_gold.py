#!/usr/bin/env python3
"""
audit_gold.py -- human audit of an LLM-generated gold set, BLIND to retrieval results

    python audit_gold.py make        # -> data/eval_work/gold_audit.csv  (sample of queries + the context needed to judge them)
    #  open in Excel/LibreOffice, fill answerable / ambiguity / copies_source_wording, save as CSV UTF-8
    python audit_gold.py summarize   # -> reports/gold_audit_summary.json  (aggregates only, safe to commit)

Rules
  * Do the audit BEFORE looking at any per-query evaluation result. Never edit or drop a query because of how it scored.
  * For each query judge only: can the gold outlet's profile/reviews answer it, would MANY other restaurants satisfy it
    equally well, and does it copy the wording of its source review.
  * The audit sheet contains dataset text (profile + review): keep it out of git (data/ is ignored); the summary has none.
"""
from __future__ import annotations

import argparse
import io
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ALLOWED = {"answerable": {"y", "n"}, "ambiguity": {"low", "medium", "high"}, "copies_source_wording": {"y", "n"}}
AUDIT_COLS = ["query_id", "query_type", "query", "answerable", "ambiguity", "copies_source_wording", "notes",
              "gold_profile_text", "source_review_text"]


def sample_queries(gold: pd.DataFrame, n: int, seed: int = 0) -> pd.DataFrame:
    """Stratified (by query_type) deterministic sample of n queries; at least one per type when n allows."""
    rng = np.random.default_rng(seed)
    n = min(n, len(gold))
    counts = gold["query_type"].value_counts()
    alloc = {t: min(int(c), max(1, round(n * c / len(gold)))) for t, c in counts.items()}
    while sum(alloc.values()) > n:
        alloc[max(alloc, key=alloc.get)] -= 1
    while sum(alloc.values()) < n:
        room = [t for t in alloc if alloc[t] < counts[t]]
        if not room:
            break
        alloc[max(room, key=lambda t: counts[t] - alloc[t])] += 1
    picked: list[int] = []
    for t, k in alloc.items():
        pool = gold.index[gold["query_type"] == t].to_numpy()
        picked += rng.choice(pool, size=k, replace=False).tolist()
    return gold.loc[sorted(picked)]


def make_sheet(gold: pd.DataFrame, profiles: dict, review_text: dict, n: int = 20, seed: int = 0) -> pd.DataFrame:
    s = sample_queries(gold, n, seed)
    has_src = "source_review_id" in s.columns
    rows = []
    for r in s.itertuples(index=False):
        src = getattr(r, "source_review_id", None) if has_src else None
        rows.append({
            "query_id": r.query_id, "query_type": r.query_type, "query": r.query,
            "answerable": "", "ambiguity": "", "copies_source_wording": "", "notes": "",
            "gold_profile_text": " || ".join(profiles.get(k, "") for k in r.gold_outlet_keys),
            "source_review_text": review_text.get(src, "") if isinstance(src, str) else "",
        })
    return pd.DataFrame(rows, columns=AUDIT_COLS)


def read_sheet_csv(path: Path) -> pd.DataFrame:
    """Excel-proof reader: UTF-8 (BOM) first, then cp1252; comma, semicolon or tab separated; everything as text."""
    raw = Path(path).read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        print("WARNING: sheet is not UTF-8 (Excel 'CSV' default); decoding as cp1252. Save as 'CSV UTF-8'.", file=sys.stderr)
        text = raw.decode("cp1252", errors="replace")
    for sep in (",", ";", "\t"):
        df = pd.read_csv(io.StringIO(text), sep=sep, dtype=str, keep_default_na=False)
        if "query_id" in df.columns:
            return df
    raise SystemExit("Could not find a 'query_id' column in the sheet (wrong file or separator).")


def wilson(k: int, n: int, z: float = 1.96) -> list | None:
    if n == 0:
        return None
    p, d = k / n, 1 + z * z / n
    centre, half = p + z * z / (2 * n), z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [round((centre - half) / d, 4), round((centre + half) / d, 4)]


def summarize(sheet: pd.DataFrame) -> dict:
    df = sheet.copy()
    for c in ALLOWED:
        df[c] = df[c].astype(str).str.strip().str.lower()
    filled = (df[list(ALLOWED)] != "").any(axis=1)
    done = df[filled]
    if done.empty:
        raise SystemExit("No audited rows found: fill answerable / ambiguity / copies_source_wording first.")
    errors = []
    for r in done.itertuples(index=False):
        for c, allowed in ALLOWED.items():
            v = getattr(r, c)
            if v not in allowed:
                errors.append(f"{r.query_id}: {c}={v!r} (allowed: {sorted(allowed)})")
    if errors:
        raise SystemExit("Audit sheet problems:\n  " + "\n  ".join(errors))
    n = len(done)

    def rate(mask) -> dict:
        k = int(mask.sum())
        return {"count": k, "rate": round(k / n, 4), "wilson95": wilson(k, n)}

    flagged = done[(done["answerable"] == "n") | (done["ambiguity"] == "high") | (done["copies_source_wording"] == "y")]
    return {
        "n_audited": n, "n_in_sheet": int(len(df)),
        "not_answerable": rate(done["answerable"] == "n"),
        "ambiguity_high": rate(done["ambiguity"] == "high"),
        "ambiguity_medium": rate(done["ambiguity"] == "medium"),
        "copies_source_wording": rate(done["copies_source_wording"] == "y"),
        "any_flag": rate(df.loc[done.index].query_id.isin(flagged.query_id)),
        "flagged_query_ids": flagged["query_id"].tolist(),
        "by_type": {t: int(c) for t, c in done["query_type"].value_counts().items()},
        "note": "n is small: read the Wilson intervals, not the point rates.",
    }


def load_jsonl(path: Path) -> pd.DataFrame:
    return pd.DataFrame([json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()])


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("--gold", type=Path, default=Path("eval/gold_queries.jsonl"))
    m.add_argument("--index-input", type=Path, default=Path("data/processed/index_input.parquet"))
    m.add_argument("--b-chunks", type=Path, default=Path("data/processed/chunks_B.parquet"))
    m.add_argument("--n", type=int, default=20)
    m.add_argument("--seed", type=int, default=0)
    m.add_argument("--out", type=Path, default=Path("data/eval_work/gold_audit.csv"))
    m.add_argument("--force", action="store_true", help="overwrite an existing sheet (destroys your answers!)")
    s = sub.add_parser("summarize")
    s.add_argument("--sheet", type=Path, default=Path("data/eval_work/gold_audit.csv"))
    s.add_argument("--out", type=Path, default=Path("reports/gold_audit_summary.json"))
    return p.parse_args(argv)


def main(argv=None) -> int:
    a = parse_args(argv)
    if a.cmd == "make":
        if a.out.exists() and not a.force:
            raise SystemExit(f"{a.out} exists; refusing to overwrite your audit answers (use --force).")
        gold = load_jsonl(a.gold)
        idx = pd.read_parquet(a.index_input, columns=["review_id", "text"])
        b = pd.read_parquet(a.b_chunks, columns=["outlet_key", "chunk_type", "text"])
        profiles = dict(zip(b.loc[b["chunk_type"] == "profile", "outlet_key"], b.loc[b["chunk_type"] == "profile", "text"]))
        need = set(gold.get("source_review_id", pd.Series(dtype=str)).dropna())
        review_text = dict(zip(idx.loc[idx["review_id"].isin(need), "review_id"], idx.loc[idx["review_id"].isin(need), "text"]))
        sheet = make_sheet(gold, profiles, review_text, a.n, a.seed)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        sheet.to_csv(a.out, index=False, encoding="utf-8-sig")
        print(f"wrote {a.out} ({len(sheet)} queries). Fill the 3 judgement columns, save as CSV UTF-8, then run `summarize`.")
        return 0
    result = summarize(read_sheet_csv(a.sheet))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())