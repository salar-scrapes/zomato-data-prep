# Zomato Bangalore: Data Prep Pipeline

Collapses the Kaggle Zomato Bangalore Restaurants dataset to one record per outlet, cleans `rate` and `approx_cost`, validates the result with Pandera, and exports Parquet and JSONL for analytics or downstream LLM ingestion.

## Finding
Exact-row `drop_duplicates()` removes 0 rows because every row has a unique `url` (51,717 unique values in 51,717 rows). The same outlet repeats across many rows (different `listed_in(type)` values, vote counts and URLs), so the 51,717 rows describe 12,499 outlets when keyed on `name` + `address`.

After normalizing formatting (`3.9/5` vs `3.9 /5`, newlines in phone numbers), rows of the same outlet still disagree on `rate`, `votes` or `approx_cost` for 4,277 of the 12,499 outlets. The cause was not verified.

## Pipeline
1. Standardize column names
2. Collapse to one row per outlet (key: `name` + `address`): keep the highest-vote row, set `location` to the most frequent label, keep `listed_intype` / `listed_incity` as lists
3. Add `is_new` flag; convert `NEW` / `-` ratings to null
4. Parse `rate` and `approx_cost` to numeric
5. Validate with Pandera
6. Export Parquet and JSONL

## Results (this dataset)
| Metric | Value |
| --- | --- |
| Raw rows | 51,717 |
| Rows collapsed into outlets | 39,218 |
| Unique outlets | 12,499 |
| Raw rows with missing, `NEW` or `-` rating | 10,052 |
| Rows failing schema | 0 |

Outlets whose rows disagree, after formatting normalization (`python diagnostics.py`):

| Column | Outlets |
| --- | --- |
| `votes` | 3,936 |
| `rate` | 1,412 |
| `approx_cost` | 344 |
| `location` | 28 |
| any of `rate` / `votes` / `approx_cost` | 4,277 |

## Schema rules
`rate` between 0 and 5, `approx_cost` > 0, non-null `name` and `address`.

## Run
```
pip install -r requirements.txt
# download zomato.csv from Kaggle into this folder
python cleaner.py
python diagnostics.py
```

## Limitations
- `name` + `address` is a heuristic key; the same outlet with a differently spelled address would not merge.
- Only the highest-vote row is kept; its `rate` (and so `is_new`) is used even if a sibling row has a different or non-missing rating.
- `location` differs across rows for 28 outlets; the most frequent label is kept.
- `phone` differs across rows for some outlets and is not normalized.
- Only `rate`, `approx_cost` and `votes` are cleaned; `phone`, `cuisines`, `dish_liked`, `reviews_list`, `menu_item` and other columns are carried over as-is.
- JSONL is not chunked or embedded; it is input to a RAG pipeline, not a finished one.
- The CSV and the generated Parquet/JSONL files are not committed.

Data: Kaggle, Zomato Bangalore Restaurants.