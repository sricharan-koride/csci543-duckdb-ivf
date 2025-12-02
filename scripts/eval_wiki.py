import duckdb
import time
import numpy as np
import pandas as pd
import os

# ============================================================
# CONFIG
# ============================================================
DB = "wiki_full.db"
EXT = "build/release/extension/ivf/ivf.duckdb_extension"

BASE_TABLE = "wiki_base"
VEC_COL    = "vec"
ID_COL     = "id"

# --- Your Requested Configuration ---
CLUSTER_SWEEP = [1024, 2048, 4096]
NPROBE_SWEEP = [32, 64, 128, 256]

K = 10
TRUTH_FILE = "data/wiki_truth_global.parquet"
QUERY_FILE = "data/wiki_query_full.parquet"

# Hybrid filters
FILTER_CLAUSES = {
    "no_filter": "",
    "region_US": "region = 'US'",
    "category_Tech": "category = 'Tech'"
}

# ============================================================
# HELPERS
# ============================================================

def connect_and_load():
    con = duckdb.connect(DB, config={'allow_unsigned_extensions': 'true'})
    con.execute(f"LOAD '{EXT}'")
    return con

def build_index(con, index_name, num_clusters):
    print(f"\n=== Building index '{index_name}' (clusters={num_clusters}) ===")
    t0 = time.time()
    before_size = os.path.getsize(DB)

    # C++ Signature: (index_name, table, col, num_clusters)
    con.execute(f"""
        SELECT create_ivf_index(
            '{index_name}',
            '{BASE_TABLE}',
            '{VEC_COL}',
            {num_clusters}
        )
    """)

    after_size = os.path.getsize(DB)
    delta_mb = (after_size - before_size) / (1024 * 1024)
    elapsed = time.time() - t0
    print(f"Index build time: {elapsed:.2f} sec")
    print(f"DB size growth: {delta_mb:.2f} MB")
    return elapsed, delta_mb

def ann_search_topk(con, index_name, query_vec, k, nprobe, where_clause=""):
    dim = len(query_vec)
    if where_clause:
        sql = f"""
            SELECT id FROM ann_search(
                '{index_name}', ?::FLOAT[{dim}], {k}, {nprobe}
            ) WHERE {where_clause}
        """
    else:
        sql = f"""
            SELECT id FROM ann_search(
                '{index_name}', ?::FLOAT[{dim}], {k}, {nprobe}
            )
        """
    rows = con.execute(sql, [query_vec]).fetchall()
    return [r[0] for r in rows]

def eval_performance(con, index_name, queries, truth_df, num_clusters, build_time, file_mb):
    results = []
    
    # Iterate through your specific nprobe list
    for nprobe in NPROBE_SWEEP:
        print(f"  Testing nprobe={nprobe}...")
        total_time = 0.0
        total_recall = 0.0
        
        for i, row in queries.iterrows():
            q_vec = row['vec']
            true_ids = set(truth_df.iloc[i]['truth_ids'][:K])
            
            t0 = time.time()
            ann_ids = ann_search_topk(con, index_name, q_vec, K, nprobe)
            total_time += (time.time() - t0)
            
            if len(true_ids) > 0:
                match = len(set(ann_ids).intersection(true_ids))
                total_recall += (match / len(true_ids))
        
        avg_latency = (total_time / len(queries)) * 1000 # ms
        avg_recall = total_recall / len(queries)
        
        print(f"    -> Latency: {avg_latency:.2f} ms | Recall: {avg_recall:.2%}")
        
        results.append({
            "num_clusters": num_clusters,
            "nprobe": nprobe,
            "latency_ms": avg_latency,
            "recall": avg_recall,
            "build_time_s": build_time,
            "index_size_mb": file_mb
        })
    return results

def eval_filters(con, index_name, query_vec, k):
    print(f"  Testing Filter Selectivity...")
    results = []
    
    # Baseline: nprobe=32 (lowest in your new set)
    base_nprobe = 32
    t0 = time.time()
    base_ids = ann_search_topk(con, index_name, query_vec, k, base_nprobe, "")
    base_time = time.time() - t0
    
    results.append({"filter": "none", "time_ms": base_time*1000, "speedup": 1.0})
    
    for name, clause in FILTER_CLAUSES.items():
        if name == "no_filter": continue
        
        t0 = time.time()
        ids = ann_search_topk(con, index_name, query_vec, k, base_nprobe, clause)
        elapsed = time.time() - t0
        
        speedup = base_time / elapsed if elapsed > 0 else 0
        print(f"    Filter '{name}' ({clause}): {elapsed*1000:.2f} ms ({speedup:.1f}x speedup)")
        
        results.append({
            "filter": name, 
            "time_ms": elapsed*1000, 
            "speedup": speedup
        })
    return results

# ============================================================
# MAIN
# ============================================================

def main():
    import os
    if not os.path.exists("data/wiki_truth_global.parquet"):
        print("❌ Error: Ground truth not found. Run scripts/gen_truth_full.py first.")
        return

    con = connect_and_load()
    queries = pd.read_parquet(QUERY_FILE)
    truth = pd.read_parquet(TRUTH_FILE)
    
    all_perf_results = []
    
    # 2. Main Loop
    for n_clusters in CLUSTER_SWEEP:
        idx_name = f"wiki_idx_{n_clusters}"
        
        # Build
        b_time, b_size = build_index(con, idx_name, n_clusters)
        
        # Evaluate Recall/Latency
        perf_res = eval_performance(con, idx_name, queries, truth, n_clusters, b_time, b_size)
        all_perf_results.extend(perf_res)
        
        # Evaluate Filters (using first query)
        eval_filters(con, idx_name, queries.iloc[0]['vec'], K)
        
        # Cleanup
        con.execute(f"DROP TABLE ivf_centroids_{idx_name}")
        con.execute(f"DROP TABLE ivf_lists_{idx_name}")
        con.execute(f"DROP TABLE ivf_pq_codes_{idx_name}")

    # 3. Save Final Report
    df = pd.DataFrame(all_perf_results)
    print("\n=== FINAL RESULTS ===")
    print(df)
    df.to_csv("final_evaluation.csv", index=False)
    print("\nSaved to final_evaluation.csv")

if __name__ == "__main__":
    main()