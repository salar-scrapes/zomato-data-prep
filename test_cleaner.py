"""pytest -q test_cleaner.py   (small synthetic data, no Kaggle file needed)"""
import numpy as np
import pandas as pd
import pytest

import cleaner as cl

RAW_COLUMNS = [
    "url", "address", "name", "online_order", "book_table", "rate", "votes", "phone", "location",
    "rest_type", "dish_liked", "cuisines", "approx_cost(for two people)", "reviews_list",
    "menu_item", "listed_in(type)", "listed_in(city)",
]


def row(**kw) -> dict:
    base = {
        "url": "u0", "address": "1 Main Road", "name": "Cafe A", "online_order": "Yes", "book_table": "No",
        "rate": "4.0/5", "votes": "10", "phone": "123", "location": "Koramangala", "rest_type": "Cafe",
        "dish_liked": "Coffee", "cuisines": "Cafe", "approx_cost(for two people)": "500",
        "reviews_list": "[]", "menu_item": "[]", "listed_in(type)": "Delivery", "listed_in(city)": "BTM",
    }
    base.update(kw)
    return base


def raw_frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=RAW_COLUMNS)


def test_standardize_columns():
    out = cl.standardize_columns(raw_frame([row()]))
    assert "approx_cost" in out.columns
    assert "listed_intype" in out.columns and "listed_incity" in out.columns
    assert "approx_costfor_two_people" not in out.columns


def test_rate_and_cost_parsing():
    df = pd.DataFrame({
        "rate": ["3.9/5", "3.9 /5", "NEW", "-", None, "4.5/5"],
        "approx_cost": ["1,200", "800", "800", "800", "800", None],
    })
    out = cl.clean_rate_and_cost(df)
    assert out["rate"].iloc[0] == 3.9 and out["rate"].iloc[1] == 3.9
    assert out["rate"].iloc[[2, 3, 4]].isna().all()
    assert out["rate"].iloc[5] == 4.5
    assert out["is_new"].tolist() == [False, False, True, False, False, False]
    assert out["approx_cost"].iloc[0] == 1200.0
    assert np.isnan(out["approx_cost"].iloc[5])


def test_sentinels_counted_on_raw_rows():
    df = cl.standardize_columns(raw_frame([
        row(url="1", name="A", rate="NEW"),
        row(url="2", name="B", rate="-"),
        row(url="3", name="C", rate=None),
        row(url="4", name="D", rate="4.1/5"),
    ]))
    assert cl.count_sentinels(df) == 3


def test_collapse_keeps_highest_votes_and_merges_lists():
    df = cl.standardize_columns(raw_frame([
        row(url="1", votes="10", rate="3.0/5", location="BTM", **{"listed_in(type)": "Buffet"}),
        row(url="2", votes="50", rate="4.4/5", location="Koramangala", **{"listed_in(type)": "Delivery"}),
        row(url="3", votes="20", rate="3.5/5", location="Koramangala", **{"listed_in(type)": "Delivery"}),
    ]))
    out = cl.collapse_outlets(df)
    assert len(out) == 1
    r = out.iloc[0]
    assert r["votes"] == 50 and r["url"] == "2" and r["rate"] == "4.4/5"
    assert r["listed_intype"] == ["Buffet", "Delivery"]
    assert r["location"] == "Koramangala"  # most frequent label


def test_same_name_different_address_stay_separate():
    df = cl.standardize_columns(raw_frame([
        row(url="1", address="1 Main Road"),
        row(url="2", address="9 Other Street"),
    ]))
    assert len(cl.collapse_outlets(df)) == 2


def test_missing_votes_are_counted_and_lose():
    out, stats = cl.clean(raw_frame([
        row(url="1", votes=None, rate="2.0/5"),
        row(url="2", votes="5", rate="4.0/5"),
    ]))
    assert stats["votes_missing_set_to_zero"] == 1
    assert out.iloc[0]["rate"] == 4.0


def test_full_pipeline_stats_and_invalid_rows_dropped():
    raw = raw_frame([
        row(url="1", name="Good", votes="30"),
        row(url="2", name="Good", votes="10", **{"listed_in(type)": "Buffet"}),   # same outlet -> collapsed
        row(url="3", name="New place", rate="NEW", votes="0"),
        row(url="4", name="Bad rate", rate="7.0/5"),                              # rate > 5 -> invalid
        row(url="5", name="Bad cost", **{"approx_cost(for two people)": "0"}),   # cost must be > 0
        row(url="6", name=None),                                                  # missing name -> invalid
    ])
    out, stats = cl.clean(raw)
    assert stats["raw_rows"] == 6
    assert stats["redundant_rows_removed"] == 1
    assert stats["invalid_rows_dropped"] == 3
    assert stats["final_rows"] == 2 == len(out)
    assert set(out["name"]) == {"Good", "New place"}
    by_name = out.set_index("name")
    assert by_name.loc["New place", "is_new"]
    assert by_name.loc["Good", "listed_intype"] == ["Buffet", "Delivery"]
    assert out["rate"].dropna().between(0, 5).all()


def test_missing_required_column_raises_clear_error():
    raw = raw_frame([row()]).drop(columns=["votes"])
    with pytest.raises(KeyError, match="votes"):
        cl.clean(raw)


def test_cli_writes_parquet_and_jsonl(tmp_path):
    src = tmp_path / "z.csv"
    raw_frame([row(url="1", name="A"), row(url="2", name="A", votes="99"), row(url="3", name="B")]).to_csv(src, index=False)
    pq, js = tmp_path / "o.parquet", tmp_path / "o.jsonl"
    cl.main(["--input", str(src), "--parquet", str(pq), "--jsonl", str(js)])
    out = pd.read_parquet(pq)
    assert len(out) == 2
    assert len(js.read_text(encoding="utf-8").strip().splitlines()) == 2