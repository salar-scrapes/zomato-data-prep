import pandas as pd
import pytest
from outlet_key import build_outlet_id

KEY = ["name", "address"]

def test_dine_in_and_delivery_urls_resolve_to_the_day1_outlet():
    outlets = pd.DataFrame({"name": ["Cafe Zoe"], "address": ["12 Main St"]})
    reviews = pd.DataFrame({
        "url": ["https://example.com/cafe-zoe/info", "https://example.com/cafe-zoe/order"],
        "name": ["Cafe Zoe", "CAFE ZOE "],
        "address": ["12 Main St", "12  Main St"],
    })
    review_ids = build_outlet_id(reviews, KEY)
    assert review_ids.nunique() == 1
    assert not (set(review_ids) - set(build_outlet_id(outlets, KEY)))

def test_whitespace_and_case_noise_does_not_split_an_outlet():
    df = pd.DataFrame({"name": ["Cafe  Zoe", " cafe zoe\u00a0"], "address": ["12 Main St", "12  main st "]})
    assert build_outlet_id(df, KEY).nunique() == 1

def test_same_name_different_address_stays_two_outlets():
    df = pd.DataFrame({"name": ["Biryani House"] * 2, "address": ["North Rd", "South Rd"]})
    assert build_outlet_id(df, KEY).nunique() == 2

def test_missing_address_is_na_not_the_string_nan():
    df = pd.DataFrame({"name": ["Cafe Zoe", "Cafe Zoe", "Cafe Zoe"], "address": [None, float("nan"), "   "]})
    ids = build_outlet_id(df, KEY)
    assert ids.isna().all()

def test_separator_prevents_boundary_collisions():
    df = pd.DataFrame({"name": ["ab", "a"], "address": ["c", "bc"]})
    assert build_outlet_id(df, KEY).nunique() == 2

def test_missing_key_column_fails_loudly():
    with pytest.raises(KeyError):
        build_outlet_id(pd.DataFrame({"name": ["x"]}), KEY)