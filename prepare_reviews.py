#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import sys
from functools import lru_cache
from pathlib import Path

import ftfy
import pandas as pd
from ftfy.badness import is_bad

# کلاؤڈ کی الگ کی گئی فائل سے امپورٹ
from outlet_key import build_outlet_id

RATED_PREFIX = re.compile(r"^\s*RATED\b\s*")
CTRL_CHARS = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
WS = re.compile(r"\s+")
RATING_NUM = re.compile(r"\d+(?:\.\d+)?")
OVERSIZED_CHARS = 20_000

def parse_reviews_list(raw: str) -> tuple[list[tuple[str | None, str]], str | None]:
    raw = raw.strip()
    if raw in ("", "[]"): return [], None
    try: obj = ast.literal_eval(raw)
    except (ValueError, SyntaxError, MemoryError, RecursionError) as exc: return [], f"literal_eval:{type(exc).__name__}"
    if not isinstance(obj, (list, tuple)): return [], "not_a_list"
    items, bad = [], 0
    for it in obj:
        if isinstance(it, (list, tuple)) and len(it) == 2 and isinstance(it[1], str):
            items.append((it[0] if isinstance(it[0], str) else None, it[1]))
        else: bad += 1
    return items, (f"bad_items:{bad}" if bad else None)

def parse_rating(s: str | None) -> float:
    if not isinstance(s, str): return float("nan")
    m = RATING_NUM.search(s)
    return float(m.group()) if m else float("nan")

@lru_cache(maxsize=None)
def clean_text(raw: str) -> tuple[str, bool, bool]:
    if raw.isascii(): t, before, after = raw, False, False
    else:
        before = is_bad(raw) or "\ufffd" in raw
        t = ftfy.fix_text(raw)
        after = is_bad(t) or "\ufffd" in t
        t = t.encode("utf-8", "replace").decode("utf-8")
    t = RATED_PREFIX.sub("", t, count=1)
    t = CTRL_CHARS.sub(" ", t)
    t = WS.sub(" ", t).strip()
    return t, before, after

def explode_and_dedup(df: pd.DataFrame, key: str) -> tuple[pd.DataFrame, list[dict], dict]:
    funnel: dict = {"raw_rows": len(df), "raw_rows_null_reviews": int(df["reviews_list"].isna().sum())}
    df = df.assign(reviews_list=df["reviews_list"].fillna("[]"))
    funnel["outlets_in_raw"] = int(df[key].nunique())
    pairs = df.groupby([key, "reviews_list"], sort=False).size().rename("n_rows").reset_index()
    funnel["distinct_outlet_reviewlist_pairs"] = len(pairs)
    rejects: list[dict] = []
    recs: list[tuple] = []
    for pair_id, (oid, raw, n_rows) in enumerate(pairs.itertuples(index=False, name=None)):
        items, err = parse_reviews_list(raw)
        if err: rejects.append({"outlet_id": oid, "error": err, "raw_len": len(raw), "raw_head": raw[:300]})
        recs.extend((oid, pair_id, n_rows, pos, r, t) for pos, (r, t) in enumerate(items))
    funnel["parse_error_pairs"] = len(rejects)
    if not recs: raise SystemExit("No reviews parsed at all -- wrong column or wrong file?")
    ex = pd.DataFrame(recs, columns=["outlet_id", "pair_id", "n_rows", "pos", "rating_raw", "text_raw"])
    ex["seq"] = range(len(ex))
    funnel["reviews_exploded"] = len(ex)
    cleaned = [clean_text(t) for t in ex["text_raw"]]
    ex["text"] = [c[0] for c in cleaned]
    ex["moji_before"] = [c[1] for c in cleaned]
    ex["moji_after"] = [c[2] for c in cleaned]
    ex["rating"] = [parse_rating(r) for r in ex["rating_raw"]]
    ex = ex.drop(columns=["rating_raw", "text_raw"])
    empty = ex["text"].str.len() == 0
    funnel["dropped_empty_text"] = int(empty.sum())
    ex = ex[~empty].copy()
    ex["text_key"] = ex["text"].str.casefold()
    ex = ex.drop_duplicates(["outlet_id", "text_key", "pair_id"])
    out = (ex.groupby(["outlet_id", "text_key"], sort=False).agg(
        text=("text", "first"), rating=("rating", "first"), rating_nuniq=("rating", "nunique"),
        source_rows=("n_rows", "sum"), mojibake_before=("moji_before", "max"),
        mojibake_residual=("moji_after", "max"), seq=("seq", "min")
    ).reset_index())
    out["rating_conflict"] = out.pop("rating_nuniq") > 1
    out["review_order"] = out.groupby("outlet_id")["seq"].rank(method="first").astype("int64") - 1
    out["n_chars"] = out["text"].str.len().astype("int64")
    out["n_words"] = out["text"].str.split().str.len().astype("int64")
    out["is_oversized"] = out["n_chars"] > OVERSIZED_CHARS
    out["n_outlets_same_text"] = out.groupby("text_key")["outlet_id"].transform("nunique").astype("int64")
    out["review_id"] = [hashlib.sha1(f"{o}\x1f{k}".encode("utf-8")).hexdigest()[:16] for o, k in zip(out["outlet_id"], out["text_key"])]
    cols = ["review_id", "outlet_id", "review_order", "rating", "text", "n_chars", "n_words",
            "source_rows", "n_outlets_same_text", "rating_conflict", "mojibake_before", "mojibake_residual", "is_oversized"]
    out = out.sort_values("seq", kind="stable")[cols].reset_index(drop=True)
    q = out["n_chars"].quantile([0.5, 0.9, 0.99, 0.999])
    funnel.update(
        reviews_after_dedup=len(out), duplicates_removed=funnel["reviews_exploded"] - funnel["dropped_empty_text"] - len(out),
        outlets_with_reviews=int(out["outlet_id"].nunique()), mojibake_before_share=round(float(out["mojibake_before"].mean()), 4),
        mojibake_residual_share=round(float(out["mojibake_residual"].mean()), 4), rating_conflicts=int(out["rating_conflict"].sum()),
        oversized=int(out["is_oversized"].sum()), cross_outlet_dup_reviews=int((out["n_outlets_same_text"] > 1).sum()),
        n_chars_quantiles={f"p{round(k * 1000) / 10:g}": int(v) for k, v in q.items()}, n_chars_max=int(out["n_chars"].max()),
    )
    return out, rejects, funnel

def validate(df: pd.DataFrame, failures_path: Path) -> pd.DataFrame:
    try: import pandera.pandas as pa
    except ImportError: import pandera as pa
    schema = pa.DataFrameSchema({
        "review_id": pa.Column(str, pa.Check.str_matches(r"^[0-9a-f]{16}$"), unique=True),
        "outlet_id": pa.Column(str, pa.Check.str_length(min_value=1)), "review_order": pa.Column(int, pa.Check.ge(0)),
        "rating": pa.Column(float, pa.Check.in_range(0.0, 5.0), nullable=True),
        "text": pa.Column(str, checks=[pa.Check(lambda s: ~s.str.match(r"^\s*RATED\b"), error="RATED prefix leaked"), pa.Check(lambda s: ~s.str.contains(CTRL_CHARS.pattern), error="control chars left"), pa.Check(lambda s: s == s.str.strip(), error="untrimmed text"), pa.Check(lambda s: ~s.str.contains(r"\s{2,}"), error="whitespace not collapsed"), pa.Check(lambda s: s.str.len() > 0, error="empty text")]),
        "n_chars": pa.Column(int, pa.Check.ge(1)), "n_words": pa.Column(int, pa.Check.ge(1)),
        "source_rows": pa.Column(int, pa.Check.ge(1)), "n_outlets_same_text": pa.Column(int, pa.Check.ge(1)),
        "rating_conflict": pa.Column(bool), "mojibake_before": pa.Column(bool), "mojibake_residual": pa.Column(bool), "is_oversized": pa.Column(bool),
    }, checks=[pa.Check(lambda d: d["n_words"] <= d["n_chars"], error="n_words > n_chars"), pa.Check(lambda d: d["n_chars"] == d["text"].str.len(), error="n_chars stale")], unique=["outlet_id", "review_order"], strict=True, coerce=True)
    try: return schema.validate(df, lazy=True)
    except pa.errors.SchemaErrors as exc:
        exc.failure_cases.to_csv(failures_path, index=False)
        raise SystemExit(f"Pandera validation FAILED: {len(exc.failure_cases)} failure cases -> {failures_path}\n{exc.failure_cases.groupby(['column', 'check']).size().to_string()}")

def check_outlets(reviews: pd.DataFrame, path: Path, keys: list[str]) -> dict:
    ref_df = pd.read_parquet(path, columns=keys)
    ref_ids = set(build_outlet_id(ref_df, keys).dropna())
    have = set(reviews["outlet_id"])
    n_blank = reviews["outlet_id"].isna().sum()
    if n_blank: raise SystemExit(f"{n_blank} reviews have a blank key column; refusing to guess")
    orphans = have - ref_ids
    if orphans: raise SystemExit(f"{len(orphans)} outlet_ids in reviews are not in Day-1 outlets (e.g. {sorted(orphans)[:3]!r})")
    return {"day1_outlets": len(ref_ids), "day1_outlets_without_reviews": len(ref_ids - have)}

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw", required=True, type=Path)
    p.add_argument("--key", default="name,address", type=lambda s: [c.strip() for c in s.split(",") if c.strip()])
    p.add_argument("--outlets", type=Path)
    p.add_argument("--encoding", default="utf-8")
    p.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    p.add_argument("--max-parse-error-rate", type=float, default=0.001)
    return p.parse_args(argv)

def main(argv=None) -> int:
    a = parse_args(argv)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    use_cols = a.key + ["reviews_list"]
    raw = pd.read_csv(a.raw, usecols=use_cols, dtype=str, encoding=a.encoding)
    
    raw["outlet_id"] = build_outlet_id(raw, a.key)
    n_null_key = int(raw["outlet_id"].isna().sum())
    raw = raw.dropna(subset=["outlet_id"])

    reviews, rejects, funnel = explode_and_dedup(raw, "outlet_id")
    funnel["raw_rows_null_key_dropped"] = n_null_key
    if rejects: pd.DataFrame(rejects).to_parquet(a.out_dir / "reviews_rejects.parquet", index=False)
    err_rate = funnel["parse_error_pairs"] / funnel["distinct_outlet_reviewlist_pairs"]
    if err_rate > a.max_parse_error_rate: return 2

    reviews = validate(reviews, a.out_dir / "reviews_validation_failures.csv")
    if a.outlets: funnel.update(check_outlets(reviews, a.outlets, a.key))

    reviews.to_parquet(a.out_dir / "reviews_clean.parquet", index=False, compression="zstd")
    reviews.to_json(a.out_dir / "reviews_clean.jsonl", orient="records", lines=True, force_ascii=False)
    (a.out_dir / "reviews_prep_report.json").write_text(json.dumps(funnel, indent=2))
    print(json.dumps(funnel, indent=2))
    return 0

if __name__ == "__main__":
    sys.exit(main())