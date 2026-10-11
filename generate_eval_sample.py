#!/usr/bin/env python3
import pandas as pd
import numpy as np

def main():
    print("Loading clean reviews...")
    df = pd.read_parquet("data/processed/reviews_clean.parquet")
    
    # 1. Strict Filtering (No garbage, no spam, no one-liners)
    mask = (
        (df["n_outlets_same_text"] == 1) &  # Exclude copy-paste spam
        (~df["mojibake_residual"]) &        # Exclude unrecoverable text
        (df["n_chars"] >= 50) &             # Must have some meat to answer a query
        (df["n_chars"] <= 1500)             # Exclude massive reviews for eval simplicity
    )
    clean_pool = df[mask].copy()
    
    # 2. Stratified Sampling (Pick 50 random outlets that have at least 3 valid reviews)
    outlet_counts = clean_pool["outlet_id"].value_counts()
    valid_outlets = outlet_counts[outlet_counts >= 3].index
    
    # Set seed for reproducibility (so Claude can reproduce it tomorrow if he wants)
    np.random.seed(42) 
    sampled_outlets = np.random.choice(valid_outlets, size=50, replace=False)
    
    # 3. Pick up to 4 reviews per sampled outlet
    eval_sample = clean_pool[clean_pool["outlet_id"].isin(sampled_outlets)]
    eval_sample = eval_sample.groupby("outlet_id").head(4).reset_index(drop=True)
    
    # 4. Prepare output for manual query drafting
    out_cols = ["outlet_id", "review_id", "rating", "text"]
    eval_sample = eval_sample[out_cols].sort_values(["outlet_id", "review_id"])
    
    # Save as CSV so you can open it in Excel/Google Sheets to add queries
    out_path = "data/processed/eval_sample_draft.csv"
    eval_sample.to_csv(out_path, index=False)
    
    print(f"Sampling complete! Selected {len(eval_sample)} reviews across {len(sampled_outlets)} outlets.")
    print(f"Saved to {out_path}")
    print("Next Step: Open the CSV, add a 'query' column, and write realistic user questions for each review.")

if __name__ == "__main__":
    main()