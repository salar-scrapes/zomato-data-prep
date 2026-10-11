#!/usr/bin/env python3
"""
embed_chunks.py -- Phase 3, step 1

chunks_*.parquet  ->  <out-dir>/<variant>.npy            float32, L2-normalised, row order == chunk table
                      <out-dir>/<variant>.manifest.json  everything needed to trust / reproduce the vectors

    python embed_chunks.py --variants B A800 --bench 2000      # measure speed + ETA, writes nothing
    python embed_chunks.py --variants B A800                    # resumable: Ctrl+C / crash / reboot is safe
    python embed_chunks.py --variants all --device cuda         # on a free GPU (Kaggle / Colab)

Design rules
  * Source of truth = the .npy + manifest. Chroma (next step) is rebuilt from them, never the other way round.
  * Passages are embedded WITHOUT a prefix; queries use QUERY_PREFIX (recorded in the manifest so evaluate.py
    reads it from there -- one definition for A and B).
  * Work is sharded; a shard is written atomically; resuming refuses to mix shards from a different model
    revision / source file / row order / shard size.
  * Gates before the final file is written: right shape + float32, no NaN/inf, unit norms, and a re-embed
    self-check on random rows. Chunks longer than the model limit are counted (the embedder truncates silently).

Requires: pandas, pyarrow, numpy, torch, sentence-transformers
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

MODEL = "BAAI/bge-small-en-v1.5"
PASSAGE_PREFIX = ""
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
VARIANT_FILES = {
    "B": "chunks_B.parquet",
    "A800": "chunks_A.parquet",
    "A400": "chunks_A_w400.parquet",
    "A1600": "chunks_A_w1600.parquet",
}
IDENTITY_KEYS = ["model", "model_commit_hash", "max_seq_length", "passage_prefix", "normalize", "dtype",
                 "dim", "source_sha256", "ids_sha256", "shard_size", "rows"]
SELFCHECK_MIN_COS = 0.999


# ------------------------------------------------------------------------- embedder
class STEmbedder:
    """sentence-transformers wrapper. Anything with .encode/.dim/.max_seq_length/.device/.info works (tests)."""

    def __init__(self, model_name: str, revision: str | None, device: str | None,
                 batch_size: int | None, threads: int | None):
        import sentence_transformers
        import torch
        from sentence_transformers import SentenceTransformer

        if threads:
            torch.set_num_threads(threads)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(model_name, device=self.device, revision=revision)
        self.max_seq_length = int(self.model.max_seq_length)
        if self.max_seq_length < 512:
            raise SystemExit(f"max_seq_length={self.max_seq_length} < 512: B chunks (up to 510 tokens) would be "
                             f"silently truncated. Refusing.")
        self.dim = int(self.model.get_sentence_embedding_dimension())
        self.batch_size = batch_size or (256 if self.device == "cuda" else 64)
        commit = getattr(getattr(self.model[0].auto_model, "config", None), "_commit_hash", None)
        self.info = {
            "model": model_name, "model_commit_hash": commit, "max_seq_length": self.max_seq_length,
            "dim": self.dim, "device": self.device, "batch_size": self.batch_size,
            "torch": torch.__version__, "sentence_transformers": sentence_transformers.__version__,
            "torch_threads": torch.get_num_threads(),
        }

    def encode(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True,
                                 convert_to_numpy=True, show_progress_bar=False).astype("float32")


# --------------------------------------------------------------------------- helpers
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha256_ids(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def check_embeddings(E: np.ndarray, rows: int, dim: int) -> None:
    if E.shape != (rows, dim) or E.dtype != np.float32:
        raise SystemExit(f"Embedding matrix has shape/dtype {E.shape}/{E.dtype}, expected {(rows, dim)}/float32.")
    if not np.isfinite(E).all():
        raise SystemExit("Embeddings contain NaN/inf.")
    norms = np.linalg.norm(E, axis=1)
    if (np.abs(norms - 1.0) > 1e-3).any():
        raise SystemExit(f"Embeddings are not unit-norm (min {norms.min():.5f}, max {norms.max():.5f}).")


def _atomic_save(path: Path, arr: np.ndarray) -> None:
    tmp = path.with_name(path.stem + ".tmp.npy")
    np.save(tmp, arr)
    os.replace(tmp, path)


def effective_tokens(df: pd.DataFrame, max_seq_length: int) -> np.ndarray:
    """Tokens the model actually reads: content + [CLS]/[SEP], capped at the model limit."""
    return np.minimum(df["n_tokens"].to_numpy(dtype="int64") + 2, max_seq_length)


def _fmt_eta(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    return f"{h}h{rem // 60:02d}m"


# ------------------------------------------------------------------------ core logic
def embed_table(df: pd.DataFrame, emb, out_dir: Path, name: str, source_sha: str, shard_size: int = 5000,
                selfcheck_rows: int = 200, force: bool = False, log=print) -> dict:
    """Embed one chunk table (columns: chunk_id, text, n_tokens) -> <name>.npy + manifest. Resumable."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ids = df["chunk_id"].tolist()
    texts = df["text"].tolist()
    n = len(df)
    if n == 0 or len(set(ids)) != n:
        raise SystemExit(f"[{name}] chunk table empty or chunk_id not unique.")
    if any((not isinstance(t, str)) or not t.strip() for t in texts):
        raise SystemExit(f"[{name}] empty/non-string text in chunk table.")

    identity = {
        "model": emb.info["model"], "model_commit_hash": emb.info.get("model_commit_hash"),
        "max_seq_length": emb.max_seq_length, "passage_prefix": PASSAGE_PREFIX, "normalize": True,
        "dtype": "float32", "dim": emb.dim, "source_sha256": source_sha, "ids_sha256": sha256_ids(ids),
        "shard_size": shard_size, "rows": n,
    }
    final_path, manifest_path = out_dir / f"{name}.npy", out_dir / f"{name}.manifest.json"

    if manifest_path.exists() and final_path.exists() and not force:
        old = json.loads(manifest_path.read_text())
        if all(old.get(k) == identity[k] for k in IDENTITY_KEYS):
            log(f"[{name}] already complete and identical -> skipping")
            return old
        raise SystemExit(f"[{name}] existing output was built from a different identity "
                         f"(model/source/rows changed). Re-run with --force to rebuild.")

    shard_dir = out_dir / f"{name}.shards"
    shard_dir.mkdir(exist_ok=True)
    meta_path = shard_dir / "meta.json"
    devices = [emb.device]
    if meta_path.exists():
        old = json.loads(meta_path.read_text())
        diff = {k: (old.get(k), identity[k]) for k in IDENTITY_KEYS if old.get(k) != identity[k]}
        if diff:
            raise SystemExit(f"[{name}] refusing to resume: run identity changed {diff}. "
                             f"Delete {shard_dir} to restart from scratch.")
        devices = sorted(set(old.get("devices", [])) | {emb.device})
    meta_path.write_text(json.dumps({**identity, "devices": devices}, indent=2))

    n_shards = math.ceil(n / shard_size)
    eff = effective_tokens(df, emb.max_seq_length)
    todo = [k for k in range(n_shards) if not (shard_dir / f"{k:05d}.npy").exists()]
    todo_tokens = int(sum(eff[k * shard_size:(k + 1) * shard_size].sum() for k in todo))
    log(f"[{name}] {n:,} chunks, {n_shards} shards ({n_shards - len(todo)} already done), "
        f"{todo_tokens / 1e6:.1f}M tokens to embed on {emb.device}")

    t0, done_tokens = time.time(), 0
    for i, k in enumerate(todo):
        lo, hi = k * shard_size, min(n, (k + 1) * shard_size)
        E = emb.encode(texts[lo:hi])
        if E.shape != (hi - lo, emb.dim) or E.dtype != np.float32:
            raise SystemExit(f"[{name}] shard {k}: bad shape/dtype {E.shape}/{E.dtype}")
        _atomic_save(shard_dir / f"{k:05d}.npy", E)
        done_tokens += int(eff[lo:hi].sum())
        el = time.time() - t0
        tps = done_tokens / max(el, 1e-9)
        log(f"[{name}] shard {k + 1}/{n_shards} | {(hi - lo) * (i + 1) / max(el, 1e-9):.1f} chunks/s (avg) | "
            f"{tps:,.0f} tok/s | ETA {_fmt_eta((todo_tokens - done_tokens) / max(tps, 1e-9))}")

    E = np.concatenate([np.load(shard_dir / f"{k:05d}.npy") for k in range(n_shards)]).astype("float32", copy=False)
    check_embeddings(E, n, emb.dim)

    # self-check: re-embed random rows, they must reproduce the stored vectors
    rng = np.random.default_rng(0)
    sel = np.sort(rng.choice(n, size=min(selfcheck_rows, n), replace=False))
    E2 = emb.encode([texts[i] for i in sel])
    cos = (E[sel] * E2).sum(axis=1)
    if cos.min() < SELFCHECK_MIN_COS:
        raise SystemExit(f"[{name}] self-check FAILED: min cosine {cos.min():.5f} < {SELFCHECK_MIN_COS} "
                         f"between stored and re-embedded rows (non-deterministic or mixed-model shards).")

    _atomic_save(final_path, E)
    manifest = {
        **identity, "variant": name, "devices": devices, "model_info": emb.info,
        "query_prefix": QUERY_PREFIX, "n_truncated_at_embedding": int((df["n_tokens"] + 2 > emb.max_seq_length).sum()),
        "selfcheck_min_cosine": float(cos.min()), "selfcheck_rows": int(len(sel)),
        "norm_min": float(np.linalg.norm(E, axis=1).min()), "norm_max": float(np.linalg.norm(E, axis=1).max()),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    log(f"[{name}] DONE -> {final_path.name} | truncated at embedding: {manifest['n_truncated_at_embedding']} "
        f"| self-check min cosine {cos.min():.6f}")
    return manifest


def bench_table(df: pd.DataFrame, emb, name: str, n: int, log=print) -> dict:
    """Embed a random sample, report throughput and a full-variant ETA. Writes nothing."""
    sample = df.sample(min(n, len(df)), random_state=0)
    t = time.time()
    emb.encode(sample["text"].tolist())
    dt = time.time() - t
    tok_s = effective_tokens(sample, emb.max_seq_length).sum() / dt
    total = effective_tokens(df, emb.max_seq_length).sum()
    res = {"variant": name, "chunks_per_s": len(sample) / dt, "tokens_per_s": float(tok_s),
           "total_tokens_M": total / 1e6, "eta_s": float(total / tok_s)}
    log(f"[{name}] bench: {res['chunks_per_s']:.1f} chunks/s | {tok_s:,.0f} tok/s | "
        f"{res['total_tokens_M']:.1f}M tokens total -> full-variant ETA {_fmt_eta(res['eta_s'])}")
    return res


# ----------------------------------------------------------------------------------- CLI
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--chunks-dir", type=Path, default=Path("data/processed"))
    p.add_argument("--out-dir", type=Path, default=Path("data/embeddings"))
    p.add_argument("--variants", nargs="+", default=["B", "A800"], choices=[*VARIANT_FILES, "all"],
                   help="default: the primary comparison only (B, A800)")
    p.add_argument("--model", default=MODEL)
    p.add_argument("--revision", default=None, help="pin a HF commit hash (taken from an earlier manifest)")
    p.add_argument("--device", default=None, help="cpu | cuda (default: auto)")
    p.add_argument("--batch-size", type=int, default=None, help="default 64 on CPU, 256 on GPU")
    p.add_argument("--threads", type=int, default=None, help="torch CPU threads (default: torch's choice)")
    p.add_argument("--shard-size", type=int, default=5000)
    p.add_argument("--bench", type=int, default=0, metavar="N", help="benchmark N random chunks per variant, write nothing")
    p.add_argument("--force", action="store_true", help="rebuild even if a complete output exists")
    return p.parse_args(argv)


def main(argv=None) -> int:
    a = parse_args(argv)
    variants = list(VARIANT_FILES) if "all" in a.variants else a.variants
    emb = STEmbedder(a.model, a.revision, a.device, a.batch_size, a.threads)
    print(json.dumps(emb.info), file=sys.stderr)

    for v in variants:
        path = a.chunks_dir / VARIANT_FILES[v]
        if not path.exists():
            raise SystemExit(f"{path} not found")
        df = pd.read_parquet(path, columns=["chunk_id", "text", "n_tokens"])
        if a.bench:
            bench_table(df, emb, v, a.bench)
        else:
            embed_table(df, emb, a.out_dir, v, sha256_file(path), a.shard_size, force=a.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())