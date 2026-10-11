"""pytest -q test_evaluate.py   (no model, no Chroma: synthetic worlds where the right answer is known)"""
import numpy as np
import pandas as pd

import evaluate as ev
import exact_search as xs

D = 16


def _unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


# ------------------------------------------------------------------------ exact_search
def test_exact_topk_matches_bruteforce():
    rng = np.random.default_rng(0)
    E, Q = _unit(rng.normal(size=(300, D))).astype("float32"), _unit(rng.normal(size=(7, D))).astype("float32")
    idx, sc = xs.exact_topk(E, Q, 10, block=3)
    for q in range(len(Q)):
        full = Q[q] @ E.T
        assert idx[q].tolist() == np.argsort(-full, kind="stable")[:10].tolist()
        assert np.allclose(sc[q], np.sort(full)[::-1][:10], atol=1e-6)


def test_collapse_and_overlap():
    outlet_of = ["a", "a", "b", "c", "b", "d"]
    assert xs.collapse_to_outlets([0, 1, 2, 3, 4, 5], outlet_of, 3) == ["a", "b", "c"]
    assert xs.collapse_to_outlets([1, 0], outlet_of, 5) == ["a"]
    assert xs.overlap_at_k([1, 2, 3, 4], [4, 3, 9, 8], 4) == 0.5


# ------------------------------------------------------------------------- scoring
def test_rank_outlets_allowed_filter_and_best_row():
    outlet_of = np.array(["a", "a", "b", "c", "b"])
    outs, best = ev.rank_outlets([0, 1, 2, 3, 4], outlet_of)
    assert outs == ["a", "b", "c"] and best == [0, 2, 3]
    outs, best = ev.rank_outlets([0, 1, 2, 3, 4], outlet_of, allowed={"b", "c"})
    assert outs == ["b", "c"] and best == [2, 3]


def test_score_lists_known_answers():
    outlet_of = np.array(["a", "b", "c", "d", "e", "f", "g"])
    rows = [[0, 1, 2, 3, 4, 5, 6],       # gold 'c' -> rank 3
            [6, 5, 4, 3, 2, 1, 0],       # gold 'a' -> rank 7
            [1, 2, 3, 4, 5, 6, 0],       # gold 'a' or 'b' (multi-gold) -> rank 1
            [3, 4]]                      # gold 'a' -> miss, and a short list
    gold = [{"c"}, {"a"}, {"a", "b"}, {"a"}]
    s = ev.score_lists(rows, outlet_of, gold)
    assert s["rank"][:3].tolist() == [3, 7, 1] and np.isnan(s["rank"][3])
    assert s["hit1"].tolist() == [0, 0, 1, 0] and s["hit5"].tolist() == [1, 0, 1, 0] and s["hit10"].tolist() == [1, 1, 1, 0]
    assert np.allclose(s["rr"], [1 / 3, 1 / 7, 1.0, 0.0]) and np.allclose(s["rr10"], [1 / 3, 1 / 7, 1.0, 0.0])
    assert s["n_outlets"].tolist() == [7, 7, 7, 2]
    d = ev.score_lists(rows, outlet_of, gold, depth=2)  # depth limits CHUNKS before collapsing
    assert d["hit5"].tolist() == [0, 0, 1, 0]


# ---------------------------------------------------------------------- statistics
def test_mcnemar_exact_p():
    assert ev.mcnemar_exact_p(0, 0) == 1.0
    assert abs(ev.mcnemar_exact_p(10, 0) - 2 / 1024) < 1e-12
    assert ev.mcnemar_exact_p(5, 5) == 1.0


def test_mcnemar_value_check():
    # n=10, min=2: 2 * (C(10,0)+C(10,1)+C(10,2)) / 1024 = 2*56/1024
    assert abs(ev.mcnemar_exact_p(8, 2) - 112 / 1024) < 1e-12


def test_cluster_bootstrap_is_reproducible_and_clustered():
    clusters = ["o1", "o1", "o1", "o2", "o3", "o3"]
    W1, W2 = ev.make_weights(clusters, 500, seed=3), ev.make_weights(clusters, 500, seed=3)
    assert W1.shape == (500, 6) and np.array_equal(W1, W2)
    assert (W1[:, 0] == W1[:, 1]).all() and (W1[:, 1] == W1[:, 2]).all()  # queries of one outlet travel together
    x = np.array([1, 1, 1, 0, 0, 0.0])
    ci = ev.mean_ci(W1, x)
    assert ci["mean"] == 0.5 and ci["ci"][0] <= 0.5 <= ci["ci"][1] and ci["n"] == 6
    const = ev.mean_ci(W1, np.ones(6))
    assert const["ci"] == [1.0, 1.0]


def test_mean_ci_mask_and_no_ci_for_tiny_groups():
    W = ev.make_weights(list("abcdef"), 200, seed=0)
    m = np.array([1, 1, 1, 0, 0, 0], bool)
    r = ev.mean_ci(W, np.array([1, 0, 1, 0, 0, 0.0]), m, with_ci=False)
    assert r["n"] == 3 and abs(r["mean"] - 2 / 3) < 1e-12 and r["ci"] is None


def test_verdict_wording_is_conservative():
    assert ev.verdict({"ci": [0.01, 0.2]}) == "B > A800: 95% CI excludes 0"
    assert ev.verdict({"ci": [-0.2, -0.01]}) == "A800 > B: 95% CI excludes 0"
    assert ev.verdict({"ci": [-0.05, 0.2]}).startswith("inconclusive")


# --------------------------------------------------------------------------- gold file
def _gold(**over):
    rows = [{"query_id": f"q{i:02d}", "query": f"some natural query number {i} please",
             "query_type": ev.QUERY_TYPES[i % 3], "gold_outlet_keys": [f"{i % 5:012x}"]} for i in range(40)]
    df = pd.DataFrame(rows)
    for k, v in over.items():
        df[k] = v
    return df


def test_validate_gold_accepts_a_clean_set():
    ok = {f"{i:012x}" for i in range(5)}
    assert ev.validate_gold(_gold(), ok) == []


def test_validate_gold_catches_every_rule():
    ok = {f"{i:012x}" for i in range(4)}  # outlet 4 has NO indexed reviews
    reviews = pd.DataFrame({"outlet_key": ["000000000000", "000000000001"], "mojibake_residual": [False, True]},
                           index=["r_ok", "r_bad"])
    g = _gold()
    g.loc[1, "query_id"] = "q00"  # duplicate id
    g.loc[2, "query"] = g.loc[3, "query"]  # duplicate text
    g.loc[4, "query_type"] = "typo_type"
    g.at[5, "gold_outlet_keys"] = ["https://zomato.com/x"]  # url instead of outlet_key
    g.at[6, "gold_outlet_keys"] = ["000000000000", "000000000000"]  # duplicate gold key
    g.loc[7, "query"] = "too short"
    g["source_review_id"] = None
    g.loc[8, "source_review_id"] = "r_bad"  # residual mojibake source
    g.loc[9, "source_review_id"] = "r_missing"  # not in shared frame
    text = "\n".join(ev.validate_gold(g, ok, reviews, min_queries=30))
    for needle in ("duplicate query_id 'q00'", "duplicate query text", "typo_type", "not a 12-hex outlet_key",
                   "without duplicates", "shorter than 3 words", "residual-mojibake", "not in the shared input frame",
                   "has no indexed reviews"):
        assert needle in text, needle
    assert any("only 5 queries" in p for p in ev.validate_gold(_gold().head(5), ok))
    assert "missing columns" in ev.validate_gold(_gold().drop(columns=["query_type"]), ok)[0]


def test_leaky_flags_catch_copy_paste_not_paraphrase():
    gold = pd.DataFrame({"query": ["the biryani here was absolutely amazing and fresh",
                                   "great rice dish, really tasty and fresh"],
                         "gold_outlet_keys": [["k1"], ["k1"]]})
    corpus = {"k1": ["Honestly the biryani here was absolutely amazing and so fresh!"]}
    assert ev.leaky_flags(gold, corpus, 5).tolist() == [True, False]


# --------------------------------------------------------------- end-to-end synthetic world
def _world(seed=0, n_out=40, with_profile_advantage=True):
    """B has an extra 'profile' chunk per outlet very close to the outlet direction; A only has noisy window chunks.
    Queries sit near their outlet's direction -> B should find the gold outlet more often, by construction."""
    rng = np.random.default_rng(seed)
    keys = [f"{i:012x}" for i in range(n_out)]
    U = _unit(rng.normal(size=(n_out, D)))

    def build(name, per_outlet, profile):
        vecs, outlet, ctype, cid = [], [], [], []
        for i, k in enumerate(keys):
            if profile:
                vecs.append(_unit(U[i] + 0.25 * rng.normal(size=D)))
                outlet.append(k); ctype.append("profile"); cid.append(f"{k}:p")
            for j in range(per_outlet):
                vecs.append(_unit(U[i] + 1.1 * rng.normal(size=D)))
                outlet.append(k); ctype.append("review" if profile else "window"); cid.append(f"{k}:{j}")
        return ev.VariantData(name, np.array(vecs, dtype="float32"), np.array(outlet), np.array(ctype), np.array(cid))

    variants = {"A800": build("A800", 4, False), "B": build("B", 4, with_profile_advantage)}
    owners = [int(x) for x in rng.integers(0, n_out, 60)]
    gold = pd.DataFrame({"query_id": [f"q{i}" for i in range(60)], "query": [f"query text number {i} here" for i in range(60)],
                         "query_type": [ev.QUERY_TYPES[i % 3] for i in range(60)],
                         "gold_outlet_keys": [[keys[o]] for o in owners]})
    Q = _unit(np.array([U[o] + 0.5 * rng.normal(size=D) for o in owners])).astype("float32")
    return variants, Q, gold, set(keys)


def test_evaluate_end_to_end_direction_and_structure():
    variants, Q, gold, owr = _world()
    leaky = np.zeros(len(gold), bool)
    leaky[:3] = True
    exact_rows = {v: xs.exact_topk(d.E, Q, 150)[0] for v, d in variants.items()}
    report, pq = ev.evaluate(variants, Q, gold, leaky, owr, chroma_rows=exact_rows, n_boot=300)

    names = set(report["systems"])
    assert names == {"A800|exact", "A800|chroma", "B|exact", "B|chroma", "B_restricted|exact", "B_reviews_only|exact"}
    assert len(pq) == len(names) * len(gold)
    # B was built to be better: point estimate must favour it, and its 'profile' chunk must be what surfaces the gold outlet
    assert report["systems"]["B|exact"]["Hit@1"]["mean"] > report["systems"]["A800|exact"]["Hit@1"]["mean"]
    assert report["primary"]["hit5_diff"]["mean"] >= 0
    # bookkeeping check: the reported chunk type of the first gold hit must equal a direct recomputation
    hits = pq[(pq.system == "B|exact") & pq["rank"].notna()].set_index("query_id")
    v, ex = variants["B"], xs.exact_topk(variants["B"].E, Q, 150)[0]
    direct = []
    for q in range(len(gold)):
        rows = [r for r in ex[q] if v.outlet_of[r] == gold["gold_outlet_keys"].iat[q][0]]
        if rows:
            direct.append((gold["query_id"].iat[q], v.chunk_type[rows[0]]))
    assert {qid: ct for qid, ct in direct} == hits["first_gold_chunk_type"].to_dict()
    assert (hits["first_gold_chunk_type"] == "profile").mean() > 1 / 5  # chance level: 1 profile among 5 chunks
    # identical rows through the 'chroma' path -> engines agree exactly
    assert all(v["queries_where_hit5_differs"] == 0 for v in report["engine_agreement"].values())
    # structure of the report
    assert set(report["by_type"]) == set(ev.QUERY_TYPES)
    assert "B_vs_A800_excluding_leaky" in report["exploratory"] and report["exploratory"]["B_vs_A800_excluding_leaky"]["n"] == 57
    assert set(report["exploratory"]["depth_sensitivity"]) == {"top50_chunks", "top100_chunks"}
    assert report["primary"]["rr_wins_losses_ties"]["b_better"] + report["primary"]["rr_wins_losses_ties"]["a_better"] \
        + report["primary"]["rr_wins_losses_ties"]["tie"] == 60
    assert "inconclusive" in ev.verdict({"ci": [-1, 1]}) and isinstance(ev.format_summary(report), str)


def test_no_difference_world_is_not_declared_a_win():
    variants, Q, gold, owr = _world(seed=1, with_profile_advantage=False)  # B == A in distribution
    report, _ = ev.evaluate(variants, Q, gold, np.zeros(len(gold), bool), owr, n_boot=400)
    d = report["primary"]["hit5_diff"]
    assert d["ci"][0] <= 0.0 <= d["ci"][1]  # CI must straddle 0 when there is nothing to find
    assert report["primary"]["verdict_hit5"].startswith("inconclusive")


def test_engine_disagreement_is_counted():
    variants, Q, gold, owr = _world()
    exact = {v: xs.exact_topk(d.E, Q, 150)[0] for v, d in variants.items()}
    broken = {v: [list(reversed(r.tolist())) for r in rows] for v, rows in exact.items()}  # worst-first = garbage ANN
    report, _ = ev.evaluate(variants, Q, gold, np.zeros(len(gold), bool), owr, chroma_rows=broken, n_boot=100)
    assert report["engine_agreement"]["B"]["queries_where_hit5_differs"] > 0
    assert report["engine_agreement"]["B"]["hit5_chroma"] < report["engine_agreement"]["B"]["hit5_exact"]


def test_b_restricted_removes_review_less_outlets_from_results():
    variants, Q, gold, owr = _world()
    allowed = set(sorted(owr)[:20])  # pretend half the outlets have no indexed reviews
    report, pq = ev.evaluate(variants, Q, gold, np.zeros(len(gold), bool), allowed, n_boot=50)
    s = ev.score_lists(xs.exact_topk(variants["B"].E, Q, 150)[0], variants["B"].outlet_of,
                       [set(k) for k in gold["gold_outlet_keys"]], allowed=allowed)
    outs, _ = ev.rank_outlets(xs.exact_topk(variants["B"].E, Q, 150)[0][0], variants["B"].outlet_of, allowed)
    assert set(outs) <= allowed and s["n_outlets"].max() <= 20


def test_manifest_consistency_guard():
    base = {"model": "m", "model_commit_hash": "h", "max_seq_length": 512, "passage_prefix": "", "query_prefix": "Q: ",
            "dim": 384, "normalize": True}
    ev.check_manifests({"A": base, "B": dict(base)})
    try:
        ev.check_manifests({"A": base, "B": {**base, "model_commit_hash": "other"}})
    except SystemExit as e:
        assert "model_commit_hash" in str(e)
    else:
        raise AssertionError("different model revisions must abort")