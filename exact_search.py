"""
exact_search.py -- brute-force ground truth for retrieval (shared by parity_test.py and, later, evaluate.py)

Embeddings are L2-normalised float32, so cosine == dot product. ~150k x 384 floats is a ~230 MB matrix:
exact search is a matrix multiply, no index and no approximation error.
"""
from __future__ import annotations

import numpy as np


def exact_topk(E: np.ndarray, Q: np.ndarray, k: int, block: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """Exact top-k rows of E for each query row of Q -> (row_idx[m, k], scores[m, k]), best first.

    Ties inside the top-k are ordered by row index. (A tie exactly at the k-th cut-off is resolved arbitrarily
    but deterministically.)
    """
    n = E.shape[0]
    k = min(k, n)
    idx = np.empty((len(Q), k), dtype=np.int64)
    sc = np.empty((len(Q), k), dtype=np.float32)
    for lo in range(0, len(Q), block):
        S = Q[lo:lo + block] @ E.T
        for r in range(S.shape[0]):
            s = S[r]
            cand = np.arange(n) if k >= n else np.argpartition(-s, k - 1)[:k]
            cand = cand[np.lexsort((cand, -s[cand]))]
            idx[lo + r], sc[lo + r] = cand, s[cand]
    return idx, sc


def collapse_to_outlets(rows, outlet_of, n: int) -> list:
    """Chunk rows in rank order -> first `n` unique outlets in rank order (the unit Hit@5 / MRR are scored on)."""
    seen, out = set(), []
    for r in rows:
        o = outlet_of[r]
        if o not in seen:
            seen.add(o)
            out.append(o)
            if len(out) == n:
                break
    return out


def overlap_at_k(a, b, k: int) -> float:
    """|top-k(a) ∩ top-k(b)| / k."""
    return len(set(a[:k]) & set(b[:k])) / k