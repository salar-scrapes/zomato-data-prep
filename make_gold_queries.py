#!/usr/bin/env python3
"""Convert the annotated draft CSV into eval/gold_queries.jsonl.

Input: data/processed/eval_sample_draft.csv with `query` and `query_type` filled in
(see fill_queries.py). Each row's gold outlet is looked up by review_id in
data/processed/index_input.parquet, so gold_outlet_keys holds the 12-hex outlet_key
that evaluate.py expects. The CSV's own outlet_id column (the long name+address
string) is not used.

Output schema (one JSON object per line):
  query_id, query, query_type, gold_outlet_keys (list of one 12-hex key), source_review_id

Rows with a blank `query` are skipped. Nothing is written if any check fails.
"""
import json
import os
import re
import sys

import pandas as pd

DRAFT = "data/processed/eval_sample_draft.csv"
INDEX = "data/processed/index_input.parquet"
OUT = "eval/gold_queries.jsonl"
MIN_QUERIES = 30  # same threshold as validate_gold
KEY_RE = re.compile(r"[0-9a-fA-F]{12}")


def main():
    # Everything is read as text so IDs can't be turned into floats or lose characters.
    df = pd.read_csv(DRAFT, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    for col in ("review_id", "query", "query_type"):
        if col not in df.columns:
            sys.exit(f"Draft CSV has no '{col}' column (columns: {list(df.columns)}).")

    df["query"] = df["query"].str.strip()
    df["query_type"] = df["query_type"].str.strip()
    gold = df[df["query"] != ""].copy().reset_index(drop=True)

    # review_id -> outlet_key, from the same table the index is built from.
    # drop_duplicates() so a chunk-level table (several rows per review) is fine.
    idx = pd.read_parquet(INDEX, columns=["review_id", "outlet_key"]).astype(str).drop_duplicates()
    ambiguous = idx.loc[idx["review_id"].duplicated(keep=False), "review_id"].unique()
    if len(ambiguous):
        sys.exit(f"{len(ambiguous)} review_ids map to more than one outlet_key in {INDEX}, "
                 f"e.g. {list(ambiguous[:3])}. Nothing written.")
    key_of = dict(zip(idx["review_id"], idx["outlet_key"]))
    gold["outlet_key"] = gold["review_id"].map(key_of)

    # Local pre-flight checks that mirror validate_gold.
    problems = []
    if len(gold) < MIN_QUERIES:
        problems.append(f"only {len(gold)} queries (< {MIN_QUERIES})")

    no_type = gold[gold["query_type"] == ""]
    if len(no_type):
        problems.append(f"{len(no_type)} rows have a query but no query_type "
                        f"(first: review_id {no_type['review_id'].iloc[0]})")

    norm = gold["query"].str.lower().str.replace(r"\W+", " ", regex=True).str.strip()
    for q in gold.loc[norm.duplicated(keep=False), "query"].unique():
        problems.append(f"duplicate query text: {q!r}")

    not_indexed = gold[gold["outlet_key"].isna()]
    if len(not_indexed):
        problems.append(f"{len(not_indexed)} review_ids not found in {INDEX}, e.g. "
                        f"{not_indexed['review_id'].head(3).tolist()} "
                        "(spreadsheet apps can mangle IDs; re-export the CSV as text)")

    bad_key = gold[gold["outlet_key"].notna() & ~gold["outlet_key"].fillna("").str.fullmatch(KEY_RE)]
    if len(bad_key):
        problems.append(f"{len(bad_key)} outlet_keys are not 12-hex, e.g. "
                        f"{bad_key['outlet_key'].head(2).tolist()}")

    if problems:
        print("Not writing gold file. Fix these first:")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        rows = zip(gold["outlet_key"], gold["review_id"], gold["query"], gold["query_type"])
        for i, (key, review_id, query, qtype) in enumerate(rows, 1):
            f.write(json.dumps({
                "query_id": f"q{i:03d}",
                "query": query,
                "query_type": qtype,
                "gold_outlet_keys": [key],
                "source_review_id": review_id,
            }, ensure_ascii=False) + "\n")

    print(f"Wrote {len(gold)} queries across {gold['outlet_key'].nunique()} outlets to {OUT}")
    print(gold["query_type"].value_counts().to_string())


if __name__ == "__main__":
    main()