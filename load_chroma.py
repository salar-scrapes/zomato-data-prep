#!/usr/bin/env python3
"""
load_chroma.py -- Phase 3, step 2

chunks_*.parquet + <variant>.npy + <variant>.manifest.json  ->  Chroma collection zomato_<variant>

    python load_chroma.py --variants B A800
    python load_chroma.py --variants B --rebuild --ef-search 400

Rules
  * Chroma is a SERVING index rebuilt from the .npy + manifest; it is never the source of truth.
  * Alignment is verified before anything is written: row count, ordered chunk_id hash, source-file hash, unit norms.
  * Embeddings are passed explicitly and embedding_function=None, so Chroma's default embedder can never run.
  * Identical HNSW settings for every variant (they are recorded in <variant>.load.json) -- a fairness requirement.
  * A collection is either complete and verified, or the run aborts. Partial / stale collections are never reused:
    use --rebuild. Do not run two processes against the same --chroma-dir at once.
  * ef_search must be >= the n_results you query with (parity_test.py asks for 151).

Requires: pandas, pyarrow, numpy, chromadb
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from embed_chunks import VARIANT_FILES, check_embeddings, sha256_file, sha256_ids

META_COLS = ["outlet_key", "chunk_type", "chunk_idx", "review_id", "location", "cuisines", "rating", "cost_for_two"]


@dataclass(frozen=True)
class HnswParams:
    space: str = "cosine"
    ef_construction: int = 200
    ef_search: int = 256
    m: int = 16


# ------------------------------------------------------------------------ alignment
def verify_alignment(df: pd.DataFrame, E: np.ndarray, manifest: dict, source_sha: str | None) -> None:
    """The vectors must belong to exactly this chunk table, row for row."""
    if manifest["rows"] != len(df) or E.shape[0] != len(df):
        raise SystemExit(f"Row count mismatch: manifest {manifest['rows']}, chunk table {len(df)}, vectors {E.shape[0]}.")
    if manifest["ids_sha256"] != sha256_ids(df["chunk_id"].tolist()):
        raise SystemExit("chunk_id order/content differs from the table the embeddings were built from.")
    if source_sha is not None and manifest["source_sha256"] != source_sha:
        raise SystemExit("Chunk file changed after embedding (source hash mismatch). Re-embed.")
    check_embeddings(E, len(df), manifest["dim"])


# ------------------------------------------------------------------------- metadata
def _clean_meta_value(v):
    if v is None or v is pd.NA:
        return None
    if isinstance(v, np.generic):
        v = v.item()  # numpy scalars are rejected by Chroma; plain python only
    if isinstance(v, float) and math.isnan(v):
        return None
    return v


def build_metadatas(df: pd.DataFrame) -> list[dict]:
    """Chroma metadata may not contain None/NaN or numpy types -> drop nulls, convert to python scalars."""
    cols = [c for c in META_COLS if c in df.columns]
    out = []
    for rec in df[cols].to_dict("records"):
        m = {}
        for k, v in rec.items():
            v = _clean_meta_value(v)
            if v is not None:
                m[k] = v
        out.append(m)
    return out


# ------------------------------------------------------------------------ collection
def create_collection(client, name: str, p: HnswParams):
    """Create a collection with explicit HNSW settings. Chroma changed how these are passed between versions,
    so try the newer form first and fall back to the legacy metadata keys. Returns (collection, api_form)."""
    errors = []
    attempts = [
        ("configuration", dict(configuration={"hnsw": {"space": p.space, "ef_construction": p.ef_construction,
                                                       "ef_search": p.ef_search, "max_neighbors": p.m}})),
        ("legacy-metadata", dict(metadata={"hnsw:space": p.space, "hnsw:construction_ef": p.ef_construction,
                                           "hnsw:search_ef": p.ef_search, "hnsw:M": p.m})),
    ]
    for form, kwargs in attempts:
        try:
            return client.create_collection(name=name, embedding_function=None, **kwargs), form
        except Exception as exc:  # noqa: BLE001 -- version differences surface as various exception types
            errors.append(f"{form}: {exc!r}")
            try:
                client.delete_collection(name)  # remove a half-created collection before the next attempt
            except Exception:  # noqa: BLE001
                pass
    raise SystemExit("Could not create the collection with explicit HNSW settings:\n  " + "\n  ".join(errors))


def _get_existing(client, name: str):
    try:
        return client.get_collection(name=name, embedding_function=None)
    except Exception:  # noqa: BLE001 -- "not found" type differs between Chroma versions
        return None


def readback_check(col, df: pd.DataFrame, E: np.ndarray, n_check: int = 200, seed: int = 0) -> None:
    """Fetch random rows back out of Chroma: vectors, documents and outlet_key must match what we put in."""
    rng = np.random.default_rng(seed)
    sel = rng.choice(len(df), size=min(n_check, len(df)), replace=False)
    ids = [df["chunk_id"].iat[int(i)] for i in sel]
    res = col.get(ids=ids, include=["embeddings", "documents", "metadatas"])
    pos = {cid: j for j, cid in enumerate(res["ids"])}
    if len(pos) != len(ids):
        raise SystemExit(f"Read-back returned {len(pos)} of {len(ids)} requested ids.")
    emb = np.asarray(res["embeddings"], dtype=np.float32)
    for i, cid in zip(sel, ids):
        j, i = pos[cid], int(i)
        if not np.allclose(emb[j], E[i], atol=1e-6):
            raise SystemExit(f"Read-back: stored vector differs from the .npy for {cid}.")
        if res["documents"][j] != df["text"].iat[i]:
            raise SystemExit(f"Read-back: stored document differs for {cid}.")
        if res["metadatas"][j].get("outlet_key") != df["outlet_key"].iat[i]:
            raise SystemExit(f"Read-back: stored outlet_key differs for {cid}.")


def load_variant(client, variant: str, df: pd.DataFrame, E: np.ndarray, params: HnswParams,
                 rebuild: bool = False, batch: int = 5000, log=print) -> dict:
    name, n = f"zomato_{variant}", len(df)
    col = _get_existing(client, name)
    if col is not None and rebuild:
        client.delete_collection(name)
        col = None
    if col is not None:
        have = col.count()
        if have != n:
            raise SystemExit(f"[{variant}] collection {name} holds {have} of {n} rows (partial or stale). "
                             f"Re-run with --rebuild.")
        readback_check(col, df, E)
        log(f"[{variant}] {name} already loaded ({n:,} rows) and verified -> skipping")
        return {"collection": name, "rows": n, "skipped": True}

    col, api_form = create_collection(client, name, params)
    ids, docs, metas = df["chunk_id"].tolist(), df["text"].tolist(), build_metadatas(df)
    max_bs = int(getattr(client, "get_max_batch_size", lambda: 5000)())
    bs = max(1, min(batch, max_bs))
    for lo in range(0, n, bs):
        hi = min(n, lo + bs)
        col.upsert(ids=ids[lo:hi], embeddings=E[lo:hi].tolist(), documents=docs[lo:hi], metadatas=metas[lo:hi])
        log(f"[{variant}] upserted {hi:,}/{n:,}")
    if col.count() != n:
        raise SystemExit(f"[{variant}] count after load {col.count()} != {n}.")
    readback_check(col, df, E)
    log(f"[{variant}] {name} loaded and verified ({n:,} rows, HNSW via {api_form})")
    return {"collection": name, "rows": n, "skipped": False, "api_form": api_form, "batch_size": bs}


# ----------------------------------------------------------------------------------- CLI
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--chunks-dir", type=Path, default=Path("data/processed"))
    p.add_argument("--emb-dir", type=Path, default=Path("data/embeddings"))
    p.add_argument("--chroma-dir", type=Path, default=Path("data/chroma"))
    p.add_argument("--variants", nargs="+", default=["B", "A800"], choices=[*VARIANT_FILES, "all"])
    p.add_argument("--rebuild", action="store_true", help="drop and recreate the collection(s)")
    p.add_argument("--space", default="cosine", choices=["cosine", "l2", "ip"])
    p.add_argument("--ef-construction", type=int, default=200)
    p.add_argument("--ef-search", type=int, default=256)
    p.add_argument("--m", type=int, default=16)
    p.add_argument("--batch", type=int, default=5000)
    return p.parse_args(argv)


def main(argv=None) -> int:
    import chromadb

    a = parse_args(argv)
    params = HnswParams(a.space, a.ef_construction, a.ef_search, a.m)
    variants = list(VARIANT_FILES) if "all" in a.variants else a.variants
    a.chroma_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(a.chroma_dir))

    for v in variants:
        chunk_path = a.chunks_dir / VARIANT_FILES[v]
        npy, man = a.emb_dir / f"{v}.npy", a.emb_dir / f"{v}.manifest.json"
        for pth in (chunk_path, npy, man):
            if not pth.exists():
                raise SystemExit(f"{pth} not found (embeddings for {v} not finished?)")
        manifest = json.loads(man.read_text())
        df = pd.read_parquet(chunk_path)
        E = np.load(npy)
        verify_alignment(df, E, manifest, sha256_file(chunk_path))
        info = load_variant(client, v, df, E, params, a.rebuild, a.batch)
        info.update(variant=v, hnsw=asdict(params), chroma_version=getattr(chromadb, "__version__", None),
                    embeddings_ids_sha256=manifest["ids_sha256"],
                    model_commit_hash=manifest.get("model_commit_hash"),
                    loaded_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"))
        if not info.get("skipped"):
            (a.emb_dir / f"{v}.load.json").write_text(json.dumps(info, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())