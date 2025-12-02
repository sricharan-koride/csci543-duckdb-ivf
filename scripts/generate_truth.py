import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import os

BASE_FILE = "data/wiki_base_full.parquet"
QUERY_FILE = "data/wiki_query_full.parquet"
OUTPUT_DIR = "data"
K = 100

def compute_truth(base_df, query_df, name):
    print(f"\n--- Computing Truth for: {name} ---")
    print(f"Base size: {len(base_df)}")
    
    base_vecs = np.stack(base_df['vec'].values)
    query_vecs = np.stack(query_df['vec'].values)
    base_ids = base_df['id'].values

    nbrs = NearestNeighbors(n_neighbors=K, algorithm='brute', metric='euclidean').fit(base_vecs)
    distances, indices = nbrs.kneighbors(query_vecs)

    truth_ids = []
    for i in range(len(indices)):
        row_ids = [base_ids[idx] for idx in indices[i]]
        truth_ids.append(row_ids)

    out_file = f"{OUTPUT_DIR}/wiki_truth_{name}.parquet"
    pd.DataFrame({'query_id': query_df['id'], 'truth_ids': truth_ids}).to_parquet(out_file)
    print(f"✅ Saved {out_file}")

def main():
    print("Loading full dataset...")
    full_df = pd.read_parquet(BASE_FILE)
    queries = pd.read_parquet(QUERY_FILE)

    # 1. Global Truth (No filter)
    compute_truth(full_df, queries, "global")

    # 2. Region='US' Truth
    print("\nFiltering for Region = 'US'...")
    us_df = full_df[full_df['region'] == 'US']
    compute_truth(us_df, queries, "region_US")

    # 3. Category='Tech' Truth
    print("\nFiltering for Category = 'Tech'...")
    tech_df = full_df[full_df['category'] == 'Tech']
    compute_truth(tech_df, queries, "category_Tech")

if __name__ == "__main__":
    main()