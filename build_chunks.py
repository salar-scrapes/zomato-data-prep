#!/usr/bin/env python3
"""
build_chunks.py -- Day 2, step 2

reviews_clean.parquet + Day-1 outlets.parquet  ->  index_input.parquet
                                               ->  chunks_A.parquet        (800-char windows, 120 overlap)
                                               ->  chunks_A_w400.parquet   (sensitivity, 60 overlap)
                                               ->  chunks_A_w1600.parquet  (sensitivity, 240 overlap)
                                               ->  chunks_B.parquet        (profile chunk + review chunks)
                                               ->  outlet_collisions.csv   (audit of identity collapses)

    python build_chunks.py --outlets data/processed/outlets.parquet

Rules baked in (do not change after you have seen eval results):
  * One shared input frame: cross-outlet duplicate reviews (n_outlets_same_text > 1) are dropped
    for BOTH strategies. Residual-mojibake rows stay in the corpus (they are excluded from GOLD, not from the index).
  * Sizes are measured with the embedding model's own tokenizer, never in characters.
  * Strategy B never exceeds the model limit; Strategy A may, and the overflow is measured and reported
    (silent truncation by the embedder is a real cost of naive chunking).

Outlet identity comes from outlet_key.build_outlet_id (same function prepare_reviews.py used);
it is NOT re-implemented here.

Requires: pandas, pyarrow, pandera, transformers (tokenizer only), and outlet_key.py on PYTHONPATH
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from outlet_key import build_outlet_id  # single source of truth for outlet identity

# size -> overlap (15% of the window; the 800/120 pair is the primary Strategy A)
A_VARIANTS = {800: 120, 400: 60, 1600: 240}
PRIMARY_A = 800

SPLIT_HEADROOM = 30  # split windows target (max_content - 30) tokens: margin for re-tokenisation drift
HARD_SPLIT_OVERLAP = 40  # tokens of overlap when a single sentence must be cut mid-sentence
SENT_SPLIT = re.compile(r"(?<=[.!?\u2026])\s+")

CHUNK_COLS = [
    "chunk_id", "strategy", "outlet_id", "outlet_key", "chunk_type", "review_id", "chunk_idx", "text",
]

# logical field -> candidate physical column names in the Day-1 outlets table (first match wins).
# Override with --map logical=physical.
CANDIDATES = {
    "name": ["name"],
    "location": ["location", "locality"],
    "rest_type": ["rest_type"],
    "cuisines": ["cuisines"],
    "dish_liked": ["dish_liked"],
    "cost": ["cost_for_two", "approx_cost", "approx_cost_for_two", "approx_cost(for two people)"],
    "rating": ["rate", "rating", "rate_num"],
    "votes": ["votes"],
    "online_order": ["online_order"],
    "book_table": ["book_table"],
}
REQUIRED = {"name", "location", "cuisines"}


# ------------------------------------------------------------------ tokenizer wrapper
class HFCounter:
    """Thin wrapper so the chunking logic only needs count() and offsets() (and is testable without HF)."""

    def __init__(self, name: str):
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(name, use_fast=True)
        if not tok.is_fast:
            raise SystemExit("Need a fast tokenizer (offset mapping is used for run-on sentences).")
        tok.model_max_length = int(1e9)  # we enforce limits ourselves; silences length warnings
        self.tok = tok

    def count(self, texts: list[str], batch: int = 10_000) -> np.ndarray:
        out = np.empty(len(texts), dtype="int64")
        for i in range(0, len(texts), batch):
            enc = self.tok(texts[i:i + batch], add_special_tokens=False,
                           return_attention_mask=False, return_token_type_ids=False)
            out[i:i + batch] = [len(x) for x in enc["input_ids"]]
        return out

    def offsets(self, text: str) -> list[tuple[int, int]]:
        return self.tok(text, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]


# ------------------------------------------------------------------- small helpers
def short_hash(outlet_id: str) -> str:
    return hashlib.sha1(outlet_id.encode("utf-8")).hexdigest()[:12]


def _missing(v) -> bool:
    return v is None or v is pd.NA or (isinstance(v, float) and np.isnan(v))


def clean_str(v, max_items: int | None = None) -> str | None:
    if _missing(v):
        return None
    if isinstance(v, (list, tuple, np.ndarray)):
        items = [str(x).strip() for x in v if not _missing(x) and str(x).strip()]
        if max_items:
            items = items[:max_items]
        s = ", ".join(items)
    else:
        parts = [p.strip() for p in str(v).split(",") if p.strip()]
        s = ", ".join(parts[:max_items] if max_items else parts)
    return s or None


def to_float(v) -> float | None:
    """4.1 / '4.1/5' / '1,200' -> float ; 'NEW', '-', NaN -> None."""
    if _missing(v):
        return None
    if isinstance(v, (int, float, np.integer, np.floating)):
        return float(v)
    m = re.search(r"\d+(?:\.\d+)?", str(v).replace(",", ""))
    return float(m.group()) if m else None


def yes_no(v) -> str | None:
    if _missing(v):
        return None
    s = str(v).strip().lower()
    if s in {"yes", "true", "1", "y"}:
        return "yes"
    if s in {"no", "false", "0", "n"}:
        return "no"
    return None


# ------------------------------------------------------------------- outlets / profile
def resolve_columns(columns, overrides: list[str]) -> dict[str, str]:
    cols = set(columns)
    mapping: dict[str, str] = {}
    for logical, cands in CANDIDATES.items():
        hit = next((c for c in cands if c in cols), None)
        if hit:
            mapping[logical] = hit
    for item in overrides or []:
        logical, _, phys = item.partition("=")
        if logical not in CANDIDATES or phys not in cols:
            raise SystemExit(f"--map {item!r}: unknown logical field or missing column. "
                             f"Logical fields: {sorted(CANDIDATES)}")
        mapping[logical] = phys
    missing = REQUIRED - set(mapping)
    if missing:
        raise SystemExit(f"Outlets table lacks required fields {sorted(missing)}. "
                         f"Columns present: {sorted(cols)}. Use --map logical=physical.")
    print("outlet column mapping:", json.dumps(mapping), file=sys.stderr)
    return mapping


PROFILE_FIELDS = ("rest_type", "cuisines", "dish_liked", "cost", "rating", "votes", "online_order", "book_table")
COLLISION_LOG_COLS = ["outlet_key", "outlet", "kept", "deciding_rule", "votes", "completeness", "tiebreak", "address"]


def compute_outlet_ids(df: pd.DataFrame, name_col: str, address_col: str, id_fn=build_outlet_id) -> list[str]:
    """Day-1 rows -> outlet_id through the shared build_outlet_id.

    CONTRACT (as used in the run that worked): id_fn(df, [name_col, address_col]) -> Series/list of str,
    one per row, same order as df.
    """
    for c in (name_col, address_col):
        if c not in df.columns:
            raise SystemExit(f"Outlets table has no column {c!r} (needed for build_outlet_id). "
                             f"Columns: {sorted(df.columns)}")
    bad = df[name_col].isna() | df[address_col].isna()
    if bad.any():
        raise SystemExit(f"{int(bad.sum())} outlets have a null name/address; cannot build outlet_id.")
    ids = list(id_fn(df, [name_col, address_col]))
    if len(ids) != len(df) or not all(isinstance(i, str) and i for i in ids):
        raise SystemExit("build_outlet_id must return one non-empty string per row -- signature mismatch?")
    return ids


def collapse_identity_collisions(df: pd.DataFrame, ids: list[str], cols: dict[str, str], address_col: str,
                                 tiebreak_col: str = "url") -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    """Day 1 collapsed on url; Day 2 identity is the normalised (name, address). Rows that now share an id
    are the same outlet listed under several urls -> keep exactly one, deterministically:

        most votes (the Day-1 rule)  ->  most complete profile  ->  smallest `tiebreak_col`  ->  file order

    Returns (kept_df, kept_ids, audit_log). Every decision is logged; nothing is dropped silently.
    """
    work = pd.DataFrame({"outlet_id": ids}, index=df.index)
    if not work["outlet_id"].duplicated().any():
        return df, ids, pd.DataFrame(columns=COLLISION_LOG_COLS)

    votes = [to_float(v) for v in df[cols["votes"]]] if "votes" in cols else [None] * len(df)
    work["_votes"] = [-1.0 if v is None else v for v in votes]  # unknown votes lose to any known votes
    present = [cols[f] for f in PROFILE_FIELDS if f in cols]
    work["_complete"] = [sum(clean_str(v) is not None for v in row)
                         for row in df[present].itertuples(index=False, name=None)]
    work["_tb"] = df[tiebreak_col].astype(str).to_numpy() if tiebreak_col in df.columns else ""
    work["_pos"] = np.arange(len(df))

    ranked = work.sort_values(["outlet_id", "_votes", "_complete", "_tb", "_pos"],
                              ascending=[True, False, False, True, True], kind="mergesort")
    ranked["_rank"] = ranked.groupby("outlet_id", sort=False).cumcount()

    dup_ids = work.loc[work["outlet_id"].duplicated(keep=False), "outlet_id"].unique()
    log = []
    for oid, g in ranked[ranked["outlet_id"].isin(dup_ids)].groupby("outlet_id", sort=False):
        win, runner = g.iloc[0], g.iloc[1]
        if win["_votes"] != runner["_votes"]:
            rule = "votes"
        elif win["_complete"] != runner["_complete"]:
            rule = "completeness"
        elif win["_tb"] != runner["_tb"]:
            rule = f"tiebreak:{tiebreak_col}"
        else:
            rule = "file_order"
        for rank, (_, r) in enumerate(g.iterrows()):
            log.append({
                "outlet_key": short_hash(oid), "outlet": oid.replace("\x1f", " | "), "kept": rank == 0,
                "deciding_rule": rule, "votes": None if r["_votes"] < 0 else r["_votes"],
                "completeness": int(r["_complete"]), "tiebreak": r["_tb"],
                "address": str(df.iloc[int(r["_pos"])][address_col]),
            })

    keep_pos = set(ranked.loc[ranked["_rank"] == 0, "_pos"])
    mask = work["_pos"].isin(keep_pos).to_numpy()
    return df.loc[mask].reset_index(drop=True), [i for i, m in zip(ids, mask) if m], pd.DataFrame(log)


def prep_outlets(df: pd.DataFrame, cols: dict[str, str], address_col: str = "address",
                 id_fn=build_outlet_id, tiebreak_col: str = "url",
                 max_collapse_share: float = 0.005) -> tuple[pd.DataFrame, pd.DataFrame]:
    """-> (outlets, collision_log). One row per composite outlet_id."""
    ids = compute_outlet_ids(df, cols["name"], address_col, id_fn)
    kept, ids, log = collapse_identity_collisions(df, ids, cols, address_col, tiebreak_col)
    dropped = len(df) - len(kept)
    if dropped / len(df) > max_collapse_share:
        raise SystemExit(f"Identity collapse would drop {dropped} of {len(df)} rows "
                         f"(> {max_collapse_share:.2%}). That is not 'a few typo variants' -- check build_outlet_id.")

    out = pd.DataFrame({"outlet_id": ids})
    if not out["outlet_id"].is_unique:  # cannot happen after the collapse; guard stays
        raise SystemExit("outlet_id still not unique after identity collapse.")
    out["outlet_key"] = [short_hash(x) for x in out["outlet_id"]]
    if not out["outlet_key"].is_unique:
        raise SystemExit("short_hash collision on outlet_key (extend the hash length).")
    for logical, phys in cols.items():
        out[logical] = pd.Series(kept[phys].to_numpy(dtype=object), index=out.index, dtype=object)
    out["rating_num"] = [to_float(v) for v in out["rating"]] if "rating" in out else np.nan
    out["cost_num"] = [to_float(v) for v in out["cost"]] if "cost" in out else np.nan
    return out, log


def profile_text(r: dict, max_dish: int = 15) -> str:
    """Templated, deterministic profile text. Numbers are kept coarse; filtering is not what is being tested."""
    head = clean_str(r.get("name")) or ""
    if (kind := clean_str(r.get("rest_type"))):
        head += f" - {kind}"
    if (loc := clean_str(r.get("location"))):
        head += f" in {loc}"
    parts = [head + "."]
    if (c := clean_str(r.get("cuisines"))):
        parts.append(f"Cuisines: {c}.")
    if (d := clean_str(r.get("dish_liked"), max_items=max_dish)):
        parts.append(f"Known for: {d}.")
    if (cost := to_float(r.get("cost"))) is not None:
        parts.append(f"Cost for two: about {int(cost)} rupees.")
    if (rt := to_float(r.get("rating"))) is not None:
        votes = to_float(r.get("votes"))
        parts.append(f"Rated {rt:.1f} out of 5" + (f" by {int(votes)} people." if votes is not None else "."))
    if (o := yes_no(r.get("online_order"))):
        parts.append(f"Online ordering: {o}.")
    if (b := yes_no(r.get("book_table"))):
        parts.append(f"Table booking: {b}.")
    return " ".join(parts)


# ------------------------------------------------------------------- shared input frame
def build_index_input(reviews: pd.DataFrame, outlets: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    rep = {"reviews_in": len(reviews)}
    keep = reviews[reviews["n_outlets_same_text"] == 1]
    rep["dropped_cross_outlet_spam"] = len(reviews) - len(keep)
    keep = keep.merge(outlets[["outlet_id", "outlet_key"]], on="outlet_id", how="left", validate="m:1")
    if keep["outlet_key"].isna().any():
        miss = keep.loc[keep["outlet_key"].isna(), "outlet_id"].unique()
        raise SystemExit(
            f"{len(miss)} outlet_ids in reviews do not exist in the outlets table built via build_outlet_id.\n"
            f"  review ids : {[repr(x) for x in miss[:3]]}\n"
            f"  outlets ids: {[repr(x) for x in outlets['outlet_id'].head(3)]}\n"
            f"Compare the two: different normalisation/version of build_outlet_id, or stale reviews_clean.parquet."
        )
    idx = keep[["review_id", "outlet_id", "outlet_key", "review_order", "text", "rating",
                "n_chars", "mojibake_residual"]].reset_index(drop=True)
    rep.update(reviews_indexed=len(idx), outlets_with_reviews=int(idx["outlet_id"].nunique()),
               outlets_total=len(outlets))
    return idx, rep


# ------------------------------------------------------------------ Strategy A (naive)
def window_spans(n: int, size: int, overlap: int) -> list[tuple[int, int]]:
    """Fixed character windows. Last window ends exactly at n (no tail-only duplicate chunk)."""
    if size <= overlap:
        raise ValueError("size must exceed overlap")
    spans, start = [], 0
    while n > 0:
        end = min(start + size, n)
        spans.append((start, end))
        if end == n:
            break
        start += size - overlap
    return spans


def finalize(df: pd.DataFrame, counter, max_content: int) -> pd.DataFrame:
    df = df.reset_index(drop=True)
    df["n_chars"] = df["text"].str.len().astype("int64")
    df["n_tokens"] = counter.count(df["text"].tolist())
    df["exceeds_model_limit"] = df["n_tokens"] > max_content
    return df


def build_chunks_A(idx: pd.DataFrame, size: int, overlap: int, counter, max_content: int) -> pd.DataFrame:
    """Per outlet: concatenate reviews in union order, then cut fixed windows. Blind to review boundaries."""
    blobs = (
        idx.sort_values(["outlet_id", "review_order"], kind="stable")
        .groupby("outlet_id", sort=False)
        .agg(outlet_key=("outlet_key", "first"), blob=("text", " ".join))
    )
    tag = f"A{size}"
    rows = []
    for oid, okey, blob in zip(blobs.index, blobs["outlet_key"], blobs["blob"]):
        for n, (a, b) in enumerate(window_spans(len(blob), size, overlap)):
            t = blob[a:b].strip()
            if t:
                rows.append((f"{okey}:w{size}:{n}", tag, oid, okey, "window", None, n, t))
    return finalize(pd.DataFrame(rows, columns=CHUNK_COLS), counter, max_content)


# --------------------------------------------------------------- Strategy B (profile + review)
def split_long_review(text: str, counter, target: int) -> list[tuple[int, int]]:
    """Sentence-aware split of ONE over-long review -> char spans, each <= `target` tokens.

    Greedy sentence packing with a one-sentence overlap (only when that sentence is short).
    A single run-on 'sentence' longer than `target` is cut by token offsets with HARD_SPLIT_OVERLAP.
    """
    cuts = [0] + [m.end() for m in SENT_SPLIT.finditer(text)] + [len(text)]
    sents = [(a, b) for a, b in zip(cuts, cuts[1:]) if b > a]
    toks = counter.count([text[a:b] for a, b in sents])

    units: list[tuple[int, int, int]] = []  # (start, end, n_tokens), every unit <= target
    for (a, b), t in zip(sents, toks):
        if t <= target:
            units.append((a, b, int(t)))
            continue
        offs = counter.offsets(text[a:b])
        n, w, stride = len(offs), 0, max(target - HARD_SPLIT_OVERLAP, target // 2, 1)
        while True:
            e = min(w + target, n)
            # The tokenizer's normaliser silently drops some characters (zero-width joiners, variation selectors,
            # U+FFFD, control chars), so token offsets can skip them. Anchor the first window to the sentence
            # start and the last to the sentence end; middle windows overlap, so their gaps are covered too.
            ua = a if w == 0 else a + offs[w][0]
            ub = b if e == n else a + offs[e - 1][1]
            units.append((ua, ub, e - w))
            if e == n:
                break
            w += stride

    spans, i, k = [], 0, len(units)
    while i < k:
        j, total = i, 0
        while j < k and total + units[j][2] <= target:
            total += units[j][2]
            j += 1
        spans.append((units[i][0], units[j - 1][1]))
        if j >= k:
            break
        i = j - 1 if (j - 1 > i and units[j - 1][2] <= target // 4) else j
    return spans


def uncovered_nonspace(text: str, spans: list[tuple[int, int]]) -> list[int]:
    """Positions of non-whitespace characters of `text` that lie in no span (empty list == lossless split)."""
    covered = np.zeros(len(text), dtype=bool)
    for a, b in spans:
        covered[a:b] = True
    nonspace = np.fromiter((not ch.isspace() for ch in text), dtype=bool, count=len(text))
    return np.flatnonzero(nonspace & ~covered).tolist()


def covers_all_nonspace(text: str, spans: list[tuple[int, int]]) -> bool:
    return not uncovered_nonspace(text, spans)


def build_chunks_B(idx: pd.DataFrame, outlets: pd.DataFrame, counter, max_content: int,
                   target: int, max_dish: int = 15) -> tuple[pd.DataFrame, dict]:
    # 1) one profile chunk per outlet (ALL outlets, including those with no reviews)
    prof = [
        (f"{r['outlet_key']}:p", "B", r["outlet_id"], r["outlet_key"], "profile", None, 0,
         profile_text(r, max_dish))
        for r in outlets.to_dict("records")
    ]

    # 2) one chunk per review; only reviews over the model limit are split
    n_tok = counter.count(idx["text"].tolist())
    fits = idx[n_tok <= max_content]
    long = idx[n_tok > max_content]
    rev = list(zip(
        (fits["review_id"] + ":0").tolist(), ["B"] * len(fits), fits["outlet_id"], fits["outlet_key"],
        ["review"] * len(fits), fits["review_id"], [0] * len(fits), fits["text"],
    ))
    n_split_chunks = 0
    for rid, oid, okey, text in zip(long["review_id"], long["outlet_id"], long["outlet_key"], long["text"]):
        spans = split_long_review(text, counter, target)
        lost = uncovered_nonspace(text, spans)
        if lost:
            detail = [(i, f"U+{ord(text[i]):04X}", unicodedata.name(text[i], "?")) for i in lost[:10]]
            raise SystemExit(f"Lossy split on review {rid} ({len(text)} chars, {len(spans)} spans): "
                             f"{len(lost)} non-space chars uncovered, first 10 (pos, codepoint, name): {detail}. "
                             f"Splitter bug -- refusing to continue.")
        for n, (a, b) in enumerate(spans):
            rev.append((f"{rid}:{n}", "B", oid, okey, "review", rid, n, text[a:b].strip()))
        n_split_chunks += len(spans)

    df = pd.DataFrame(prof + rev, columns=CHUNK_COLS)
    meta = outlets[["outlet_id", "location", "cuisines", "rating_num", "cost_num"]].rename(
        columns={"rating_num": "rating", "cost_num": "cost_for_two"})
    meta["location"] = meta["location"].map(clean_str)
    meta["cuisines"] = meta["cuisines"].map(clean_str)
    df = df.merge(meta, on="outlet_id", how="left", validate="m:1")
    df["rating"] = df["rating"].astype("float64")
    df["cost_for_two"] = df["cost_for_two"].astype("float64")
    rep = {"reviews_split": len(long), "chunks_from_split_reviews": n_split_chunks}
    return finalize(df, counter, max_content), rep


# ------------------------------------------------------------------------- validation
def _pa():
    try:
        import pandera.pandas as pa
    except ImportError:  # pandera < 0.24
        import pandera as pa
    return pa


def validate_chunks(df: pd.DataFrame, kind: str, expected_strategy: str, max_content: int,
                    failures_path: Path) -> pd.DataFrame:
    pa = _pa()
    cols = {
        "chunk_id": pa.Column(str, pa.Check.str_length(min_value=1), unique=True),
        "strategy": pa.Column(str, pa.Check.equal_to(expected_strategy)),
        "outlet_id": pa.Column(str, pa.Check.str_length(min_value=1)),
        "outlet_key": pa.Column(str, pa.Check.str_matches(r"^[0-9a-f]{12}$")),
        "chunk_type": pa.Column(str, pa.Check.isin(["window"] if kind == "A" else ["profile", "review"])),
        "review_id": pa.Column(str, nullable=True),
        "chunk_idx": pa.Column(int, pa.Check.ge(0)),
        "text": pa.Column(str, checks=[
            pa.Check(lambda s: s.str.len() > 0, error="empty text"),
            pa.Check(lambda s: s == s.str.strip(), error="untrimmed text"),
        ]),
        "n_chars": pa.Column(int, pa.Check.ge(1)),
        "n_tokens": pa.Column(int, pa.Check.ge(1)),
        "exceeds_model_limit": pa.Column(bool),
    }
    dfchecks = [
        pa.Check(lambda d: bool((d["exceeds_model_limit"] == (d["n_tokens"] > max_content)).all()),
                 error="exceeds_model_limit inconsistent with n_tokens"),
    ]
    if kind == "A":
        dfchecks.append(pa.Check(lambda d: bool(d["review_id"].isna().all()), error="A must carry no review_id"))
    else:
        cols["n_tokens"] = pa.Column(int, [pa.Check.ge(1), pa.Check.le(max_content)])  # B: hard limit
        cols["location"] = pa.Column(str, nullable=True)
        cols["cuisines"] = pa.Column(str, nullable=True)
        cols["rating"] = pa.Column(float, pa.Check.in_range(0.0, 5.0), nullable=True)
        cols["cost_for_two"] = pa.Column(float, pa.Check.ge(0), nullable=True)
        dfchecks += [
            pa.Check(lambda d: bool(((d["chunk_type"] == "review") == d["review_id"].notna()).all()),
                     error="review_id must be set exactly on review chunks"),
            pa.Check(lambda d: bool(d.loc[d["chunk_type"] == "profile", "outlet_key"].is_unique),
                     error="more than one profile chunk per outlet"),
        ]
    schema = pa.DataFrameSchema(cols, checks=dfchecks, strict=True, coerce=True)
    try:
        return schema.validate(df, lazy=True)
    except pa.errors.SchemaErrors as exc:
        exc.failure_cases.to_csv(failures_path, index=False)
        raise SystemExit(f"Pandera FAILED for {expected_strategy}: {len(exc.failure_cases)} failure cases "
                         f"-> {failures_path}\n"
                         f"{exc.failure_cases.groupby(['column', 'check']).size().to_string()}")


def validate_index_input(df: pd.DataFrame, failures_path: Path) -> pd.DataFrame:
    pa = _pa()
    schema = pa.DataFrameSchema(
        {
            "review_id": pa.Column(str, unique=True),
            "outlet_id": pa.Column(str),
            "outlet_key": pa.Column(str, pa.Check.str_matches(r"^[0-9a-f]{12}$")),
            "review_order": pa.Column(int, pa.Check.ge(0)),
            "text": pa.Column(str, pa.Check.str_length(min_value=1)),
            "rating": pa.Column(float, pa.Check.in_range(0.0, 5.0), nullable=True),
            "n_chars": pa.Column(int, pa.Check.ge(1)),
            "mojibake_residual": pa.Column(bool),
        },
        unique=["outlet_id", "review_order"], strict=True, coerce=True,
    )
    try:
        return schema.validate(df, lazy=True)
    except pa.errors.SchemaErrors as exc:
        exc.failure_cases.to_csv(failures_path, index=False)
        raise SystemExit(f"Pandera FAILED for index_input -> {failures_path}")


def frame_stats(df: pd.DataFrame) -> dict:
    q = df["n_tokens"].quantile([0.5, 0.9, 0.99]).to_dict()
    return {
        "chunks": len(df),
        "outlets": int(df["outlet_id"].nunique()),
        "chunks_per_outlet_mean": round(len(df) / max(df["outlet_id"].nunique(), 1), 2),
        "n_tokens_p50": int(q[0.5]), "n_tokens_p90": int(q[0.9]), "n_tokens_p99": int(q[0.99]),
        "n_tokens_max": int(df["n_tokens"].max()),
        "exceeds_model_limit": int(df["exceeds_model_limit"].sum()),
        "exceeds_model_limit_share": round(float(df["exceeds_model_limit"].mean()), 5),
    }


# ----------------------------------------------------------------------------------- CLI
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--reviews", type=Path, default=Path("data/processed/reviews_clean.parquet"))
    p.add_argument("--outlets", type=Path, required=True, help="Day-1 outlets Parquet")
    p.add_argument("--address-col", default="address",
                   help="address column in the outlets table (identity = build_outlet_id over name + address)")
    p.add_argument("--tiebreak-col", default="url",
                   help="final tie-breaker (ascending) when colliding outlets tie on votes and completeness")
    p.add_argument("--max-collapse-share", type=float, default=0.005,
                   help="abort if identity collapse drops more than this share of Day-1 rows")
    p.add_argument("--map", action="append", default=[], metavar="LOGICAL=COLUMN",
                   help=f"override outlet column mapping; logical fields: {sorted(CANDIDATES)}")
    p.add_argument("--tokenizer", default="BAAI/bge-small-en-v1.5")
    p.add_argument("--max-seq-len", type=int, default=512, help="embedder limit incl. [CLS]/[SEP]")
    p.add_argument("--max-dish", type=int, default=15)
    p.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    return p.parse_args(argv)


def main(argv=None) -> int:
    a = parse_args(argv)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    max_content = a.max_seq_len - 2
    target = max_content - SPLIT_HEADROOM
    counter = HFCounter(a.tokenizer)

    reviews = pd.read_parquet(a.reviews)
    outlets_raw = pd.read_parquet(a.outlets)
    cols = resolve_columns(outlets_raw.columns, a.map)
    outlets, collisions = prep_outlets(outlets_raw, cols, a.address_col, tiebreak_col=a.tiebreak_col,
                                       max_collapse_share=a.max_collapse_share)
    if len(collisions):
        collisions.to_csv(a.out_dir / "outlet_collisions.csv", index=False, encoding="utf-8-sig")

    idx, report = build_index_input(reviews, outlets)
    report["identity"] = {
        "day1_rows": len(outlets_raw), "outlets_after_identity_collapse": len(outlets),
        "rows_collapsed": len(outlets_raw) - len(outlets),
        "collision_groups": int(collisions["outlet_key"].nunique()) if len(collisions) else 0,
        "deciding_rules": collisions.loc[collisions["kept"], "deciding_rule"].value_counts().to_dict()
        if len(collisions) else {},
    }
    idx = validate_index_input(idx, a.out_dir / "index_input_failures.csv")
    idx.to_parquet(a.out_dir / "index_input.parquet", index=False, compression="zstd")
    report["limits"] = {"max_seq_len": a.max_seq_len, "max_content_tokens": max_content, "split_target": target}

    # Strategy A: primary + sensitivity runs
    for size, overlap in A_VARIANTS.items():
        tag = f"A{size}"
        df = build_chunks_A(idx, size, overlap, counter, max_content)
        df = validate_chunks(df, "A", tag, max_content, a.out_dir / f"chunks_{tag}_failures.csv")
        name = "chunks_A.parquet" if size == PRIMARY_A else f"chunks_A_w{size}.parquet"
        df.to_parquet(a.out_dir / name, index=False, compression="zstd")
        report[tag] = {"file": name, "window": size, "overlap": overlap, **frame_stats(df)}

    # Strategy B
    B, b_rep = build_chunks_B(idx, outlets, counter, max_content, target, a.max_dish)
    B = validate_chunks(B, "B", "B", max_content, a.out_dir / "chunks_B_failures.csv")

    # Lineage invariants Pandera cannot express
    rev = B[B["chunk_type"] == "review"]
    if set(rev["review_id"]) != set(idx["review_id"]):
        raise SystemExit("Strategy B lost or invented reviews relative to index_input.")
    if int((B["chunk_type"] == "profile").sum()) != len(outlets):
        raise SystemExit("Strategy B profile chunk count != number of outlets.")

    B.to_parquet(a.out_dir / "chunks_B.parquet", index=False, compression="zstd")
    report["B"] = {"file": "chunks_B.parquet", **b_rep, **frame_stats(B),
                   "profile_chunks": len(outlets), "review_chunks": len(rev)}

    (a.out_dir / "chunks_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())