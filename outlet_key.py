"""
outlet_key.py - the ONE definition of outlet identity.
"""
from __future__ import annotations
from typing import Sequence
import pandas as pd

SEP = "\x1f"

def _clean(s: pd.Series) -> pd.Series:
    out = (
        s.astype("string")
        .str.normalize("NFKC")
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
        .str.casefold()
    )
    return out.mask(out == "")

def build_outlet_id(df: pd.DataFrame, cols: Sequence[str]) -> pd.Series:
    cols = list(cols)
    if not cols:
        raise ValueError("need at least one key column")
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"key column(s) not in data: {missing}; have: {list(df.columns)}")
    out = _clean(df[cols[0]])
    for c in cols[1:]:
        out = out + SEP + _clean(df[c])
    return out