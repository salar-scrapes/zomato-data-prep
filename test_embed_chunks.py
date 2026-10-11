"""pytest -q test_embed_chunks.py   (no model download: deterministic fake embedder)"""
import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import embed_chunks as ec

DIM = 16


class FakeEmbedder:
    """Deterministic per-text unit vectors. Counts encode() calls/rows so tests can prove what was re-computed."""

    def __init__(self, commit="abc123", fail_after_calls=None, nan=False, nondeterministic=False, max_seq=512):
        self.device, self.dim, self.max_seq_length = "cpu", DIM, max_seq
        self.info = {"model": "fake", "model_commit_hash": commit}
        self.calls, self.rows = 0, 0
        self.fail_after_calls, self.nan, self.nondeterministic = fail_after_calls, nan, nondeterministic

    def _vec(self, t):
        seed = int(hashlib.sha1(t.encode()).hexdigest()[:8], 16)
        v = np.random.default_rng(seed).normal(size=DIM)
        return v / np.linalg.norm(v)

    def encode(self, texts):
        if self.fail_after_calls is not None and self.calls >= self.fail_after_calls:
            raise RuntimeError("simulated crash")
        self.calls += 1
        self.rows += len(texts)
        if self.nondeterministic:
            v = np.random.default_rng().normal(size=(len(texts), DIM))
            return (v / np.linalg.norm(v, axis=1, keepdims=True)).astype("float32")
        E = np.stack([self._vec(t) for t in texts]).astype("float32")
        if self.nan:
            E[0, 0] = np.nan
        return E


def _df(n=23, tokens=50):
    return pd.DataFrame({"chunk_id": [f"c{i}" for i in range(n)], "text": [f"chunk number {i}" for i in range(n)],
                         "n_tokens": [tokens] * n})


def _quiet(*a, **k):
    pass


def _run(df, emb, out, **kw):
    kw.setdefault("shard_size", 10)
    kw.setdefault("source_sha", "src1")
    return ec.embed_table(df, emb, Path(out), "T", log=_quiet, **kw)


def test_full_run_alignment_manifest_and_gates():
    with tempfile.TemporaryDirectory() as d:
        df, emb = _df(), FakeEmbedder()
        m = _run(df, emb, d)
        E = np.load(Path(d) / "T.npy")
        assert E.shape == (23, DIM) and E.dtype == np.float32
        fresh = FakeEmbedder()
        for i in (0, 9, 10, 22):  # across shard boundaries: row i must be the embedding of text i
            assert np.allclose(E[i], fresh.encode([df.text[i]])[0], atol=1e-6)
        assert m["rows"] == 23 and m["ids_sha256"] == ec.sha256_ids(df.chunk_id.tolist())
        assert m["query_prefix"] == ec.QUERY_PREFIX and m["passage_prefix"] == ""
        assert m["selfcheck_min_cosine"] > 0.9999
        assert json.loads((Path(d) / "T.manifest.json").read_text())["n_truncated_at_embedding"] == 0


def test_resume_after_crash_recomputes_only_missing_shards():
    with tempfile.TemporaryDirectory() as d:
        df = _df()
        try:
            _run(df, FakeEmbedder(fail_after_calls=2), d)  # shards 0,1 written, crash on shard 2
        except RuntimeError:
            pass
        else:
            raise AssertionError("expected simulated crash")
        assert sorted(p.name for p in (Path(d) / "T.shards").glob("0*.npy")) == ["00000.npy", "00001.npy"]
        emb = FakeEmbedder()
        _run(df, emb, d)
        ref = tempfile.TemporaryDirectory()
        _run(df, FakeEmbedder(), ref.name)
        assert np.array_equal(np.load(Path(d) / "T.npy"), np.load(Path(ref.name) / "T.npy"))
        ref.cleanup()


def test_resume_encodes_only_remaining_rows():
    with tempfile.TemporaryDirectory() as d:
        df = _df()
        try:
            _run(df, FakeEmbedder(fail_after_calls=2), d, selfcheck_rows=5)
        except RuntimeError:
            pass
        emb = FakeEmbedder()
        _run(df, emb, d, selfcheck_rows=5)
        assert emb.rows == 3 + 5  # only shard 2 (3 rows) + the 5-row self-check


def test_second_run_is_a_noop():
    with tempfile.TemporaryDirectory() as d:
        df = _df()
        _run(df, FakeEmbedder(), d)
        emb = FakeEmbedder()
        _run(df, emb, d)
        assert emb.calls == 0


def test_resume_refuses_different_model_revision():
    with tempfile.TemporaryDirectory() as d:
        df = _df()
        try:
            _run(df, FakeEmbedder(commit="aaa", fail_after_calls=1), d)
        except RuntimeError:
            pass
        try:
            _run(df, FakeEmbedder(commit="bbb"), d)
        except SystemExit as e:
            assert "refusing to resume" in str(e) and "model_commit_hash" in str(e)
        else:
            raise AssertionError("must not mix shards from different model revisions")


def test_resume_refuses_reordered_or_changed_chunk_table():
    with tempfile.TemporaryDirectory() as d:
        df = _df()
        try:
            _run(df, FakeEmbedder(fail_after_calls=1), d)
        except RuntimeError:
            pass
        shuffled = df.sample(frac=1, random_state=1).reset_index(drop=True)
        try:
            _run(shuffled, FakeEmbedder(), d)
        except SystemExit as e:
            assert "ids_sha256" in str(e)
        else:
            raise AssertionError("row order change must be detected")
        try:
            _run(df, FakeEmbedder(), d, source_sha="different-file")
        except SystemExit as e:
            assert "source_sha256" in str(e)
        else:
            raise AssertionError("source file change must be detected")


def test_nan_embeddings_abort():
    with tempfile.TemporaryDirectory() as d:
        try:
            _run(_df(), FakeEmbedder(nan=True), d)
        except SystemExit as e:
            assert "NaN" in str(e)
        else:
            raise AssertionError("NaN must abort")
        assert not (Path(d) / "T.npy").exists()  # nothing final is written on failure


def test_nondeterministic_embedder_fails_selfcheck():
    with tempfile.TemporaryDirectory() as d:
        try:
            _run(_df(), FakeEmbedder(nondeterministic=True), d)
        except SystemExit as e:
            assert "self-check FAILED" in str(e)
        else:
            raise AssertionError("self-check must catch non-reproducible vectors")
        assert not (Path(d) / "T.npy").exists()


def test_truncation_is_counted():
    with tempfile.TemporaryDirectory() as d:
        df = _df(12)
        df.loc[[0, 5, 7], "n_tokens"] = 600  # > 510 content tokens
        m = _run(df, FakeEmbedder(), d)
        assert m["n_truncated_at_embedding"] == 3
        assert ec.effective_tokens(df, 512).max() == 512


def test_bad_chunk_tables_rejected():
    with tempfile.TemporaryDirectory() as d:
        dup = _df(5)
        dup.loc[1, "chunk_id"] = "c0"
        for bad, msg in ((dup, "not unique"), (_df(5).assign(text=["ok", " ", "x", "y", "z"]), "empty")):
            try:
                _run(bad, FakeEmbedder(), d)
            except SystemExit as e:
                assert msg in str(e)
            else:
                raise AssertionError(f"expected abort: {msg}")


def test_bench_reports_throughput_and_eta_without_writing():
    with tempfile.TemporaryDirectory() as d:
        res = ec.bench_table(_df(50), FakeEmbedder(), "T", 20, log=_quiet)
        assert res["chunks_per_s"] > 0 and res["eta_s"] > 0 and res["total_tokens_M"] > 0
        assert list(Path(d).iterdir()) == []