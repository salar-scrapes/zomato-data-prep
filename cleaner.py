import pandas as pd
import pandera.pandas as pa
import numpy as np

print("Loading raw data...")
df = pd.read_csv('zomato.csv')
initial_rows = len(df)

print("1. Standardizing column names...")
df.columns = df.columns.str.lower().str.replace(' ', '_').str.replace(r'[^a-z0-9_]', '', regex=True)

if 'approx_costfor_two_people' in df.columns:
    df.rename(columns={'approx_costfor_two_people': 'approx_cost'}, inplace=True)

print("2. Counting Sentinel Values on RAW data...")
if 'rate' in df.columns:
    raw_rate = df['rate'].astype(str).str.strip()
    sentinel_new = (raw_rate == 'NEW').sum()
    sentinel_dash = (raw_rate == '-').sum()
    sentinel_null = df['rate'].isna().sum()
    total_sentinels = sentinel_new + sentinel_dash + sentinel_null
else:
    total_sentinels = 0

print("3. Handling Redundant Rows (Highest-Vote & Location Mode Logic)...")
key = ['name', 'address']
cat_cols = ['listed_intype', 'listed_incity']

if 'votes' in df.columns:
    df['votes'] = pd.to_numeric(df['votes'], errors='coerce').fillna(0)

cats = (df.groupby(key, sort=False)[cat_cols]
          .agg(lambda s: sorted(s.dropna().unique()))
          .reset_index())

best = (df.sort_values('votes', ascending=False, kind='stable')
          .drop_duplicates(subset=key, keep='first')
          .drop(columns=cat_cols))


if 'location' in df.columns:
    loc = (df.groupby(key, sort=False)['location']
             .agg(lambda s: s.mode().iat[0] if s.notna().any() else None)
             .rename('location').reset_index())
    best = best.drop(columns='location').merge(loc, on=key, how='left')

df = best.merge(cats, on=key, how='left')
redundant_rows_removed = initial_rows - len(df)

print("4. Feature Engineering & Cleaning...")
if 'rate' in df.columns:
    df['is_new'] = df['rate'].astype(str).str.strip().eq('NEW')
    df['rate'] = df['rate'].astype(str).str.replace('/5', '', regex=False).str.strip()
    df.replace(['NEW', '-'], np.nan, inplace=True)
    df['rate'] = pd.to_numeric(df['rate'], errors='coerce')

if 'approx_cost' in df.columns:
    df['approx_cost'] = df['approx_cost'].astype(str).str.replace(',', '', regex=False)
    df['approx_cost'] = pd.to_numeric(df['approx_cost'], errors='coerce')

print("5. Validation with Pandera...")
schema = pa.DataFrameSchema({
    "name": pa.Column(str, nullable=False),
    "address": pa.Column(str, nullable=False),
    "rate": pa.Column(float, pa.Check.in_range(0.0, 5.0), nullable=True),
    "approx_cost": pa.Column(float, pa.Check.greater_than(0.0), nullable=True),
    "is_new": pa.Column(bool)
})

try:
    validated_df = schema.validate(df, lazy=True)
    failed_validation = 0
except pa.errors.SchemaErrors as err:
    failed_index = err.failure_cases['index'].dropna().unique()
    failed_validation = len(failed_index)
    validated_df = df.drop(index=failed_index)

print("6. Saving to Parquet & JSONL...")
validated_df.to_parquet('zomato_cleaned.parquet', index=False)
validated_df.to_json('zomato_rag_ready.jsonl', orient='records', lines=True, force_ascii=False)

final_rows = len(validated_df)

print("\n" + "="*50)
print("🎯 MISSION ACCOMPLISHED! ALL FINAL NUMBERS:")
print(f"[X] Raw Rows: {initial_rows}")
print(f"    - Redundant Rows Collapsed: {redundant_rows_removed}")
print(f"    - Sentinel Values Handled: {total_sentinels}")
print(f"    - Invalid Rows Dropped: {failed_validation}")
print(f"[Y] Unique Validated Outlets: {final_rows}")
print("="*50 + "\n")