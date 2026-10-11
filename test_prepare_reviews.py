"""pytest -q test_prepare_reviews.py"""
import numpy as np
import pandas as pd
import pytest

import prepare_reviews as pr


def garble(s: str, times: int = 1) -> str:
    """Simulate the upstream bug: UTF-8 bytes decoded as cp1252 (optionally repeatedly)."""
    for _ in range(times):
        s = s.encode("utf-8").decode("cp1252", errors="replace")
    return s


# --------------------------------------------------------------------- mojibake
# (garbled input, expected). ftfy uncurls curly quotes by default -> straight apostrophe.
MOJIBAKE = [
    ("curly apostrophe", garble("It\u2019s a great place"), "It's a great place"),
    ("en-dash", garble("Food \u2013 amazing"), "Food \u2013 amazing"),
    ("ellipsis", garble("Wait\u2026 what"), "Wait\u2026 what"),
    ("emoji", garble("Loved it \U0001F60B"), "Loved it \U0001F60B"),
    ("latin1-safe e-acute", "Caf\u00c3\u00a9 Coffee Day", "Caf\u00e9 Coffee Day"),
    ("double-encoded", garble("It\u2019s great", 2), "It's great"),
    ("mixed clean + garbled", "Caf\u00e9 " + garble("\u2013 loved it"), "Caf\u00e9 \u2013 loved it"),
]


@pytest.mark.parametrize("name,garbled,want", MOJIBAKE, ids=[m[0] for m in MOJIBAKE])
def test_mojibake_repaired(name, garbled, want):
    text, _before, after = pr.clean_text(garbled)
    assert text == want
    assert after is False


@pytest.mark.parametrize("clean", ["Cr\u00e8me br\u00fbl\u00e9e", "Na\u00efve but tasty", "Plain ascii"])
def test_clean_text_untouched(clean):
    text, before, after = pr.clean_text(clean)
    assert text == clean and not before and not after


def test_rated_prefix_stripped_but_not_real_reviews():
    assert pr.clean_text("RATED\n  Great food")[0] == "Great food"
    assert pr.clean_text("Rated this place highly")[0] == "Rated this place highly"


# ------------------------------------------------------------------------ parsing
def test_parse_handles_quotes_and_newlines():
    raw = r"""[('Rated 4.0', 'RATED\n  It\'s great'), ('Rated 1.0', "RATED\n  Don't go")]"""
    items, err = pr.parse_reviews_list(raw)
    assert err is None and len(items) == 2
    assert "It's great" in items[0][1]


def test_parse_reports_malformed():
    items, err = pr.parse_reviews_list("[('Rated 4.0', 'oops'")
    assert items == [] and err.startswith("literal_eval")


def test_rating_parse():
    assert pr.parse_rating("Rated 4.5") == 4.5
    assert np.isnan(pr.parse_rating(None))


# ------------------------------------------------------------- explode + dedup
def _lst(*pairs):
    return repr(list(pairs))


@pytest.fixture
def raw():
    great = ("Rated 4.0", "RATED\n  Great food.")
    slow = ("Rated 3.0", "RATED\n  Slow service.")
    rows = [
        ("A", _lst(great, slow)),
        ("A", _lst(great, ("Rated 2.0", "RATED\n  Slow service."), ("Rated 5.0", "RATED\n  Loved it"))),
        ("A", _lst(great, slow)),  # exact duplicate raw row
        ("B", "[]"),
        ("C", None),
        ("D", _lst(("Rated 4.0", "RATED\n  great   FOOD."))),  # same text as A, different case/spacing
        ("E", "[('Rated 4.0', 'oops'"),  # malformed
        ("F", _lst(("Rated 4.0", "RATED\n  "), ("Rated 4.0", "RATED\n  x"), ("Rated 4.0", "RATED\n  x"))),
    ]
    return pd.DataFrame(rows, columns=["url", "reviews_list"])


def test_pipeline(raw):
    out, rejects, funnel = pr.explode_and_dedup(raw, "url")

    a = out[out.outlet_id == "A"].set_index("text")
    assert len(a) == 3  # union across rows, duplicates collapsed
    assert a.loc["Great food.", "source_rows"] == 3
    assert a.loc["Loved it", "source_rows"] == 1
    assert a.loc["Slow service.", "rating_conflict"]  # 3.0 vs 2.0 across rows
    assert a.loc["Slow service.", "rating"] == 3.0  # first-seen wins
    assert sorted(out[out.outlet_id == "A"].review_order) == [0, 1, 2]

    assert (out[out.text.str.casefold() == "great food."].n_outlets_same_text == 2).all()
    assert [r["outlet_id"] for r in rejects] == ["E"]
    assert funnel["dropped_empty_text"] == 1  # F's blank review
    assert (out.outlet_id == "F").sum() == 1  # F's duplicated "x" collapsed to one
    assert set(out.outlet_id) == {"A", "D", "F"}  # B, C, E have no reviews
    assert out.review_id.is_unique