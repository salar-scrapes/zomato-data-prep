"""pytest -q test_build_chunks.py   (no model download: uses a whitespace 'tokenizer')"""
import random
import re
import unicodedata

import numpy as np
import pandas as pd

import build_chunks as bc


class FakeCounter:
    """1 token == 1 whitespace-separated word. Same interface as HFCounter."""

    def count(self, texts, batch=0):
        return np.array([len(t.split()) for t in texts], dtype="int64")

    def offsets(self, text):
        return [m.span() for m in re.finditer(r"\S+", text)]


C = FakeCounter()


class BertLikeCounter:
    """Emulates how a BERT-uncased fast tokenizer treats odd characters: control/format chars (Cc/Cf, e.g. ZWSP, ZWJ),
    U+FFFD and combining marks (Mn, e.g. emoji variation selector U+FE0F) are DROPPED -- no token, no offset.
    Punctuation is its own token; whitespace separates words."""

    def _toks(self, text):
        toks, cur = [], None
        for i, ch in enumerate(text):
            cat = unicodedata.category(ch)
            if ch == "\ufffd" or (cat.startswith("C") and ch not in "\t\n\r") or cat == "Mn":
                continue
            if ch.isspace():
                cur = None
                continue
            if cat.startswith("P") or (ord(ch) < 128 and not ch.isalnum()):
                toks.append([i, i + 1])
                cur = None
                continue
            if cur is None:
                cur = [i, i + 1]
                toks.append(cur)
            else:
                cur[1] = i + 1
        return [tuple(t) for t in toks]

    def count(self, texts, batch=0):
        return np.array([len(self._toks(t)) for t in texts], dtype="int64")

    def offsets(self, text):
        return self._toks(text)


BC = BertLikeCounter()


# ---------------------------------------------------------------- Strategy A windows
def test_window_spans_overlap_and_tail():
    spans = bc.window_spans(25, 10, 3)
    assert spans == [(0, 10), (7, 17), (14, 24), (21, 25)]
    assert all(spans[i][1] - spans[i + 1][0] == 3 for i in range(2))  # exact overlap
    assert bc.window_spans(10, 10, 3) == [(0, 10)]  # no tail-only duplicate
    assert bc.window_spans(5, 10, 3) == [(0, 5)]
    assert bc.window_spans(0, 10, 3) == []


def _idx():
    rows = []
    for o in ("u1", "u2"):
        for i in range(4):
            rows.append((f"{o}r{i}", o, bc.short_hash(o), i, f"review {i} of {o} " + "word " * 20, 4.0, 90, False))
    df = pd.DataFrame(rows, columns=["review_id", "outlet_id", "outlet_key", "review_order", "text",
                                     "rating", "n_chars", "mojibake_residual"])
    df["text"] = df["text"].str.strip()
    return df


def test_chunks_A_cover_blob_and_respect_size():
    idx = _idx()
    A = bc.build_chunks_A(idx, 100, 15, C, max_content=510)
    assert (A["n_chars"] <= 100).all()
    assert A["chunk_id"].is_unique and (A["strategy"] == "A100").all()
    for oid, g in A.groupby("outlet_id"):
        blob = " ".join(idx[idx.outlet_id == oid].sort_values("review_order")["text"])
        assert g.sort_values("chunk_idx")["text"].iloc[0] == blob[:100].strip()
        assert blob[-20:].strip() in g.sort_values("chunk_idx")["text"].iloc[-1]  # tail reached


# ------------------------------------------------------ Strategy B splitting (lossless)
def test_split_sentence_aware_with_one_sentence_overlap():
    text = " ".join(f"alpha{i} beta{i} gamma{i}." for i in range(30))  # 3 tokens per sentence
    spans = bc.split_long_review(text, C, target=20)
    assert len(spans) > 1
    assert all(C.count([text[a:b]])[0] <= 20 for a, b in spans)
    assert bc.covers_all_nonspace(text, spans)
    for (a1, b1), (a2, b2) in zip(spans, spans[1:]):
        last_sentence = text[a1:b1].split()[-3:]
        assert all(tok in text[a2:b2] for tok in last_sentence)  # overlap = last sentence of previous chunk
        assert text[a2:b2].startswith(last_sentence[0])


def test_split_runon_without_punctuation():
    text = " ".join(f"w{i}" for i in range(100))
    spans = bc.split_long_review(text, C, target=20)
    assert all(C.count([text[a:b]])[0] <= 20 for a, b in spans)
    assert bc.covers_all_nonspace(text, spans)


def test_runon_with_dropped_chars_at_both_ends_is_lossless():
    """Regression: 'Lossy split on review 93dff10e65630ecb'. A run-on 'sentence' (no '. ') is cut by token offsets;
    the tokenizer drops ZWSP/ZWJ/variation-selector/U+FFFD, which used to fall outside the first/last span."""
    text = "\u200b\u200d" + " ".join(f"w{i}" for i in range(600)) + " \u2764\ufe0f\ufffd"
    spans = bc.split_long_review(text, BC, target=100)
    assert len(spans) > 1
    assert bc.uncovered_nonspace(text, spans) == []
    assert all(BC.count([text[a:b]])[0] <= 100 for a, b in spans)


def test_fuzz_split_is_lossless_and_within_budget():
    rng = random.Random(7)
    pool = ["good", "food", "w1", "x2", "Great", "service", "\u2764\ufe0f", "\U0001F44D", "\u200d", "\u200b", "\ufffd",
            "????", "!!!", "...", "\u0928\u092e\u0938\u094d\u0924\u0947", "\u4e2d\u6587", "caf\u00e9", "\x07", "a\u0301"]
    seps = [" ", " ", " ", ". ", "! ", "? ", ", ", ".\n", "  "]
    for _ in range(400):
        target = rng.randint(12, 60)
        n = rng.randint(1, 900)
        sep_pool = [" "] if rng.random() < 0.4 else seps  # 40%: run-on text with no sentence breaks
        text = "".join(rng.choice(pool) + rng.choice(sep_pool) for _ in range(n)).strip()
        if not text:
            continue
        spans = bc.split_long_review(text, BC, target)
        assert bc.uncovered_nonspace(text, spans) == [], text[:80]
        assert all(BC.count([text[a:b]])[0] <= target for a, b in spans), text[:80]


def test_loss_report_names_the_characters():
    assert bc.uncovered_nonspace("abc \u200b def", [(0, 3), (6, 9)]) == [4]


def test_coverage_detector_catches_loss():
    text = "one two three four"
    assert not bc.covers_all_nonspace(text, [(0, 7)])


# ----------------------------------------------------------------------- profile text
def test_profile_text_formats_and_skips_missing():
    t = bc.profile_text({"name": "Jalsa", "rest_type": "Casual Dining", "location": "Banashankari",
                         "cuisines": "North Indian, Mughlai", "dish_liked": ["Pasta", "Buffet"],
                         "cost": "1,200", "rating": "4.1/5", "votes": "775",
                         "online_order": "Yes", "book_table": "No"})
    assert t == ("Jalsa - Casual Dining in Banashankari. Cuisines: North Indian, Mughlai. "
                 "Known for: Pasta, Buffet. Cost for two: about 1200 rupees. Rated 4.1 out of 5 by 775 people. "
                 "Online ordering: yes. Table booking: no.")
    sparse = bc.profile_text({"name": "X", "location": "Y", "cuisines": "Cafe", "rating": "NEW", "cost": None,
                              "dish_liked": float("nan")})
    assert sparse == "X in Y. Cuisines: Cafe."


# ---------------------------------------------------------------------- identity + full B build
def fake_id(df, cols):
    """Stand-in for outlet_key.build_outlet_id(df, cols): lower-case + whitespace-collapse, joined by \x1f."""
    norm = lambda s: s.astype(str).str.replace(r"\s+", " ", regex=True).str.strip().str.lower()
    return norm(df[cols[0]]) + "\x1f" + norm(df[cols[1]])


def _outlets():
    return pd.DataFrame({
        "name": ["Jalsa", "B", "C"], "address": ["942, 21st Main Road", "addr b", "addr c"],
        "location": ["L1", "L2", "L3"], "cuisines": ["Cafe", "Pizza", "Thai"], "rate": ["4.0/5", "3.5/5", "NEW"],
    })


def _prep(df, **kw):
    kw.setdefault("max_collapse_share", 1.0)  # fixtures collapse far more than the 0.5% production guard
    return bc.prep_outlets(df, bc.resolve_columns(df.columns, []), "address", id_fn=fake_id, **kw)


def test_outlet_id_is_composite_and_unique():
    out, log = _prep(_outlets())
    assert out["outlet_id"].iloc[0] == "jalsa\x1f942, 21st main road"
    assert out["outlet_id"].is_unique and out["outlet_key"].is_unique and log.empty


def test_missing_address_column_fails_loudly():
    o = _outlets().drop(columns=["address"])
    try:
        _prep(o)
    except SystemExit as e:
        assert "address" in str(e)
    else:
        raise AssertionError("missing address column must abort")


def _collision_frame():
    return pd.DataFrame({
        "name": ["Mahendra", "Mahendra", "Cafe X", "Cafe X", "Zed", "Zed", "Solo"],
        "address": ["Behind Mahendra showroom", "Behind  Mahendra Showroom", "1 MG Rd", "1 mg rd",
                    "2 Park St", "2 PARK ST", "9 Lake View"],
        "location": ["L"] * 7, "cuisines": ["Cafe"] * 7,
        "votes": ["10", "20", "50", "50", "7", "7", "3"],
        "dish_liked": [None, None, None, "Pasta", None, None, None],  # Cafe X: 2nd row is more complete
        "url": ["u_a", "u_b", "u_c", "u_d", "u_f", "u_e", "u_g"],     # Zed: full tie -> smaller url wins
    })


def test_collision_resolution_votes_then_completeness_then_url():
    out, log = _prep(_collision_frame())
    assert len(out) == 4 and out["outlet_id"].is_unique
    kept = log[log.kept].set_index(log[log.kept]["outlet"].str.split(" \\| ").str[0])
    assert kept.loc["mahendra", "deciding_rule"] == "votes" and kept.loc["mahendra", "votes"] == 20.0
    assert kept.loc["cafe x", "deciding_rule"] == "completeness"
    assert kept.loc["zed", "deciding_rule"] == "tiebreak:url" and kept.loc["zed", "tiebreak"] == "u_e"
    assert len(log) == 6 and log.groupby("outlet_key")["kept"].sum().eq(1).all()  # 1 winner per group
    # winner's profile fields really come from the winning row
    assert out.loc[out.name == "Cafe X", "dish_liked"].iloc[0] == "Pasta"


def test_collision_resolution_is_row_order_independent():
    f = _collision_frame()
    a, _ = _prep(f)
    b, _ = _prep(f.sample(frac=1, random_state=3).reset_index(drop=True))
    cols = ["outlet_id", "votes", "dish_liked"]
    assert a.sort_values("outlet_id")[cols].reset_index(drop=True).equals(
        b.sort_values("outlet_id")[cols].reset_index(drop=True))


def test_collapse_guard_aborts_on_mass_merge():
    try:
        _prep(_collision_frame(), max_collapse_share=0.01)  # 3 of 7 rows collapse
    except SystemExit as e:
        assert "Identity collapse" in str(e)
    else:
        raise AssertionError("mass collapse must abort")


def test_reviews_with_foreign_ids_fail_with_diagnostics():
    outlets, _ = _prep(_outlets())
    reviews = _idx().drop(columns=["outlet_key"])  # outlet_id values 'u1'/'u2' -> not composite ids
    reviews["n_outlets_same_text"] = 1
    try:
        bc.build_index_input(reviews, outlets)
    except SystemExit as e:
        msg = str(e)
        assert "review ids" in msg and "outlets ids" in msg
    else:
        raise AssertionError("foreign outlet ids must abort")


def test_build_B_and_index_input():
    out, _ = _prep(_outlets())
    ids = out["outlet_id"].tolist()

    reviews = _idx().drop(columns=["outlet_key"])
    reviews["outlet_id"] = reviews["outlet_id"].map({"u1": ids[0], "u2": ids[1]})  # reviews carry composite ids
    reviews["n_outlets_same_text"] = 1
    reviews.loc[0, "n_outlets_same_text"] = 2  # spam -> must be dropped
    reviews.loc[1, "text"] = " ".join(f"s{i} x y." for i in range(40))  # 120 tokens -> must split
    idx, rep = bc.build_index_input(reviews, out)
    assert rep["dropped_cross_outlet_spam"] == 1 and len(idx) == len(reviews) - 1

    B, brep = bc.build_chunks_B(idx, out, C, max_content=60, target=40)
    assert (B.chunk_type == "profile").sum() == 3  # incl. outlet C with zero reviews
    assert set(B[B.chunk_type == "review"].review_id) == set(idx.review_id)  # nothing lost
    assert (B.n_tokens <= 60).all() and not B.exceeds_model_limit.any()
    assert brep["reviews_split"] == 1 and brep["chunks_from_split_reviews"] > 1
    assert B.chunk_id.is_unique
    assert B.loc[B.outlet_id == ids[2], "rating"].isna().all()  # 'NEW' -> NaN, not a crash