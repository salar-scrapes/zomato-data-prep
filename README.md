# Zomato Bangalore: Data Prep Pipeline and RAG Chunking Experiment

This repository has two parts built on the Kaggle Zomato Bangalore Restaurants dataset:

1. **Data prep** (`cleaner.py`, `prepare_reviews.py`): collapse 51,717 rows to one record per outlet, clean `rate` and `approx_cost`, validate with Pandera, and export Parquet and JSONL.
2. **Retrieval experiment** (`build_chunks.py` onward): chunk outlet profiles and reviews two ways, embed them, load them into Chroma, and measure which chunking finds the right outlet more often (Hit@k, MRR) on 50 gold queries.

```
zomato.csv --cleaner.py--> zomato_cleaned.parquet --+
     |                                              +--build_chunks.py--> chunks_*.parquet
     +--prepare_reviews.py--> reviews_clean.parquet-+          |
                                                     embed_chunks.py --> load_chroma.py --> evaluate.py
```

## Run

```
pip install -r requirements.txt
# download zomato.csv from Kaggle into this folder

# Part 1: data prep
python cleaner.py                                   # -> zomato_cleaned.parquet, zomato_rag_ready.jsonl
python diagnostics.py                               # outlets whose rows disagree (table below)

# Part 2: retrieval experiment
python prepare_reviews.py --raw zomato.csv          # -> data/processed/reviews_clean.parquet
python build_chunks.py --outlets zomato_cleaned.parquet
python embed_chunks.py --variants B A800
python load_chroma.py --variants B A800
python parity_test.py --variants B A800             # Chroma vs exact search check

# gold queries (50 queries, drafted with an LLM from sampled reviews; see Limitations)
python generate_eval_sample.py                      # -> data/processed/eval_sample_draft.csv
python fill_queries.py
python make_gold_queries.py                         # -> eval/gold_queries.jsonl
python evaluate.py --gold eval/gold_queries.jsonl --variants B A800

# tests
pytest -q
```

## Finding
Exact-row `drop_duplicates()` removes 0 rows because every row has a unique `url` (51,717 unique values in 51,717 rows). The same outlet repeats across many rows (different `listed_in(type)` values, vote counts and URLs), so the 51,717 rows describe 12,499 outlets when keyed on `name` + `address`.

After normalizing formatting (`3.9/5` vs `3.9 /5`, newlines in phone numbers), rows of the same outlet still disagree on `rate`, `votes` or `approx_cost` for 4,277 of the 12,499 outlets. The cause was not verified.

## Pipeline (`cleaner.py`)
1. Standardize column names
2. Collapse to one row per outlet (key: `name` + `address`): keep the highest-vote row, set `location` to the most frequent label, keep `listed_intype` / `listed_incity` as lists
3. Add `is_new` flag; convert `NEW` / `-` ratings to null
4. Parse `rate` and `approx_cost` to numeric
5. Validate with Pandera
6. Export Parquet and JSONL

`build_chunks.py` later applies the normalised outlet identity from `outlet_key.py` and logs every extra merge it makes.

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

## Limitations (data prep)
- `name` + `address` is a heuristic key; the same outlet with a differently spelled address would not merge in `cleaner.py` (`build_chunks.py` merges some of these using the normalised key).
- Only the highest-vote row is kept; its `rate` (and so `is_new`) is used even if a sibling row has a different or non-missing rating.
- Missing `votes` are treated as 0 when choosing the highest-vote row (the count is printed by `cleaner.py`).
- `location` differs across rows for 28 outlets; the most frequent label is kept.
- `phone` differs across rows for some outlets and is not normalized.
- Only `rate`, `approx_cost` and `votes` are cleaned; `phone`, `cuisines`, `dish_liked`, `reviews_list`, `menu_item` and other columns are carried over as-is.
- The CSV and the generated Parquet/JSONL files are not committed.

Data: Kaggle, Zomato Bangalore Restaurants.

## Conclusion

**On the primary metric (Hit@5), Strategy B is ahead of fixed 800-character windows (A800) by +0.14, but the 95% interval [-0.019, +0.275] includes zero, so the result is inconclusive at 50 queries. On MRR (exploratory) B is ahead by +0.139 [+0.072, +0.211]. Removing B's outlet-profile chunks did not hurt (reviews-only B: Hit@5 0.420, MRR 0.346), so on this gold set the gain comes from chunking reviews individually, not from profile metadata.**

### Strategies compared

- **A800 (baseline):** per outlet, all reviews are joined in order and cut into fixed 800-character windows with 120 characters of overlap. Windows ignore review boundaries.
- **B:** one chunk per review (a review longer than the embedding model's token limit is split at sentence boundaries), plus one templated profile chunk per outlet built from fields such as name, type, location, cuisines, dishes and cost.
- **B_restricted (control):** B with outlets that have no indexed reviews removed from the results, so both strategies draw from the same pool of outlets (A800 cannot return outlets without reviews).
- **B_reviews_only (control):** B without its profile chunks.

### Results

50 gold queries, each with one gold outlet. The 95% confidence intervals in brackets are outlet-level cluster bootstrap intervals (10,000 resamples, percentile method), because queries about the same outlet are not independent. McNemar's exact test ignores that clustering and is shown for reference only.

| System | Hit@1 | Hit@5 | MRR |
|---|---|---|---|
| B (exact) | 0.240 [0.154, 0.327] | 0.400 [0.294, 0.510] | 0.322 [0.237, 0.405] |
| B (Chroma) | 0.220 [0.122, 0.314] | 0.380 [0.255, 0.500] | 0.303 [0.206, 0.396] |
| B_restricted (exact) | 0.240 [0.154, 0.327] | 0.400 [0.294, 0.510] | 0.323 [0.238, 0.406] |
| B_reviews_only (exact) | 0.260 [0.157, 0.380] | 0.420 [0.314, 0.531] | 0.346 [0.245, 0.454] |
| A800 (exact) | 0.100 [0.020, 0.196] | 0.260 [0.104, 0.431] | 0.183 [0.082, 0.294] |
| A800 (Chroma) | 0.100 [0.020, 0.196] | 0.260 [0.128, 0.400] | 0.179 [0.084, 0.283] |

Primary comparison (B - A800, exact engine):

- **Hit@5 (primary): +0.140 [-0.019, +0.275].** Inconclusive, because the interval includes 0 (McNemar p = 0.065). B alone found the gold outlet in the top 5 on 9 queries, A800 alone on 2.
- **MRR (exploratory): +0.139 [+0.072, +0.211].** The interval excludes 0, which points to B ranking the right outlet higher.

### Takeaways

- **Ranking quality (exploratory):** B places the correct outlet higher on average than A800 (MRR 0.32 vs 0.18).
- **Source of the gain:** removing B's profile chunks (B_reviews_only) did not hurt (Hit@5 0.420, MRR 0.346), so on this gold set the gain comes from chunking reviews individually rather than from profile metadata.
- **Vector store:** Chroma's approximate index tracks brute-force exact search closely (differences of 0.02 or less on Hit@1, Hit@5 and MRR for both strategies), so the comparison between strategies is not an artefact of the vector store.

### Limitations

- **Small, narrow gold set:** 50 queries drawn from 13 outlets. Each query has a single gold outlet, so a query that other outlets could also answer counts as a miss, which makes the recall figures conservative.
- **Gold-set selection:** the queries cover the first 13 outlets, alphabetically, of a 50-outlet random sample (queries were written for the first 50 reviews of the sorted sample). They are not a random draw across the sample.
- **LLM-drafted queries:** The queries were drafted with an LLM from reviews that are in the index, so they may favour per-review chunking and may not match how real users phrase questions.
- **Modest absolute recall:** Even B surfaces the gold outlet in the top 5 for only 40% of queries.
- **Profile chunks:** B_reviews_only (B without its profile chunks) scores Hit@5 0.420 and MRR 0.346 versus 0.400 and 0.322 for B, so the profile chunks show no measurable benefit. All 50 queries were drafted from reviews, so this gold set cannot reward profile information. The result does not show that profile chunks are useless in general.

### Next steps

- Grow the gold set to about 150 queries spread across all sampled outlets. That would narrow the Hit@5 interval by roughly 40%.
- Add queries drafted from profile fields (cuisine, cost, location) to test the profile chunks fairly.
- Add a keyword (BM25) hybrid or a reranker and re-run the same harness.