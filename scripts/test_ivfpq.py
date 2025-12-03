import duckdb
import time
import numpy as np
import pandas as pd
import os


# CONFIG

DB = "sift_data.db"
EXT = "build/release/extension/ivf/ivf.duckdb_extension"

BASE_TABLE = "sift_base"      
VEC_COL    = "vec"            
ID_COL     = "id"

# Cluster sizes to sweep
CLUSTER_SWEEP = [1024, 2048, 4096]

# nprobe to use for main evaluation
DEFAULT_NPROBE = 64

# Number of random query vectors to evaluate for recall
NUM_QUERY_VECS = 100

NPROBE_SWEEP = [32, 64, 128, 256]

# Hybrid filter clauses to test
FILTER_CLAUSES = {
    "no_filter": "",
    "region_US": "region = 'US'",
    "region_EU": "region = 'EU'",
    "category_Tech": "category = 'Tech'",
    "category_Fashion": "category = 'Fashion'"
}



# HELPERS


def connect_and_load():
    con = duckdb.connect(DB, config={'allow_unsigned_extensions': 'true'})
    con.execute(f"LOAD '{EXT}'")
    print("Extension loaded.")
    return con


def check_base_schema(con):
    print("\n--- Checking base table schema ---")
    print(con.execute(f"PRAGMA table_info('{BASE_TABLE}')").df())

def index_exists(con, index_name):
    tables = con.execute("SHOW TABLES").fetchall()
    table_names = [t[0] for t in tables]
    return any(index_name in t for t in table_names)


def build_index(con, index_name, num_clusters, use_pq=True):
    """
    Build either IVFPQ or IVFFlat index, depending on use_pq flag.

    NOTE: This assumes your create_ivf_index(...) function has a `use_pq` or
    similar boolean parameter. If your actual signature differs, adjust the SQL
    below accordingly.
    """
    print(f"\n=== Building index '{index_name}' (clusters={num_clusters}, use_pq={use_pq}) ===")
    t0 = time.time()
    before_size = os.path.getsize(DB)

    if use_pq:
        # IVFPQ 
        con.execute(f"""
            SELECT create_ivf_index(
                '{index_name}',
                '{BASE_TABLE}',
                '{VEC_COL}',
                {num_clusters}
            )
        """)
    else:
        # Plain IVFFlat version 
        # If your function needs an explicit flag, change this to:
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
    print(f"DB size increased by: {delta_mb:.2f} MB")
    return elapsed, delta_mb


def sample_queries(con, num_queries):
    """Sample a set of query vectors from the base table."""
    print(f"\n--- Sampling {num_queries} query vectors ---")
    rows = con.execute(f"""
        SELECT {VEC_COL}
        FROM {BASE_TABLE}
        USING SAMPLE {num_queries} ROWS (RESERVOIR)
    """).fetchall()

    queries = [r[0] for r in rows]
    print(f"Sampled {len(queries)} query vectors.")
    return queries


def exact_search_topk(con, query_vec, k):
    """Run full-scan exact search for a single query."""
    rows = con.execute(f"""
        SELECT {ID_COL}, {VEC_COL} <-> ? AS dist
        FROM {BASE_TABLE}
        ORDER BY dist
        LIMIT {k}
    """, [query_vec]).fetchall()
    return [r[0] for r in rows]


def ann_search_topk(con, index_name, query_vec, k, nprobe, where_clause=""):
    """
    Run ann_search on a single query.
    where_clause is a SQL fragment applied *inside* ann_search via parameter,
    assuming your table function supports a where_clause or filter parameter.
    """
    if where_clause:
        # Escape single quotes inside the where_clause for SQL safety
        escaped_clause = where_clause.replace("'", "''")
        sql = f"""
            SELECT *
            FROM ann_search(
                '{index_name}',
                ?::FLOAT[128],
                {k},
                {nprobe},
                where_clause := '{escaped_clause}'
            )
        """
    else:
        sql = f"""
            SELECT *
            FROM ann_search(
                '{index_name}',
                ?::FLOAT[128],
                {k},
                {nprobe}
            )
        """

    rows = con.execute(sql, [query_vec]).fetchall()
    # Assume first column is ID
    return [r[0] for r in rows]


def eval_recall_latency(con, index_name, queries, k, nprobe, label=""):
    """
    Compute average Recall@k and latency over a batch of query vectors.
    """
    print(f"\n=== Evaluating {label} (index={index_name}, k={k}, nprobe={nprobe}) ===")

    exact_total_time = 0.0
    ann_total_time = 0.0
    total_recall = 0.0

    for i, q in enumerate(queries):
        # Exact
        t0 = time.time()
        exact_ids = exact_search_topk(con, q, k)
        exact_total_time += (time.time() - t0)

        # ANN
        t0 = time.time()
        ann_ids = ann_search_topk(con, index_name, q, k, nprobe)
        ann_total_time += (time.time() - t0)

        # Recall@k
        inter = len(set(exact_ids) & set(ann_ids))
        total_recall += inter / float(len(exact_ids))

        if (i + 1) % max(1, len(queries)//10) == 0:
            print(f"  Processed {i+1}/{len(queries)} queries...")

    avg_exact = exact_total_time / len(queries)
    avg_ann = ann_total_time / len(queries)
    avg_recall = total_recall / len(queries)

    print(f"Average exact latency: {avg_exact*1000:.2f} ms")
    print(f"Average ANN   latency: {avg_ann*1000:.2f} ms")
    print(f"Average Recall@{k}: {avg_recall*100:.2f}%")

    return {
        "index_name": index_name,
        "label": label,
        "k": k,
        "nprobe": nprobe,
        "avg_exact_ms": avg_exact * 1000.0,
        "avg_ann_ms": avg_ann * 1000.0,
        "recall_at_k": avg_recall
    }


def eval_filter_selectivity(con, index_name, query_vec, k, nprobe):
    """
    Evaluate latency under different hybrid filters for a **single** query vector
    just to understand relative speedups.
    """
    print(f"\n=== Hybrid Filter Selectivity (index={index_name}) ===")
    rows = []

    # No filter baseline
    t0 = time.time()
    base_ids = ann_search_topk(con, index_name, query_vec, k, nprobe, where_clause="")
    base_time = time.time() - t0
    base_count = len(base_ids)
    print(f"no_filter: time={base_time:.4f} sec, results={base_count}")

    rows.append({
        "filter": "no_filter",
        "latency_sec": base_time,
        "result_count": base_count,
        "speedup_vs_no_filter": 1.0
    })

    # With filters
    for name, clause in FILTER_CLAUSES.items():
        if name == "no_filter":
            continue
        t0 = time.time()
        ids = ann_search_topk(con, index_name, query_vec, k, nprobe, where_clause=clause)
        t1 = time.time()
        elapsed = t1 - t0
        count = len(ids)
        speedup = base_time / elapsed if elapsed > 0 else float("inf")

        print(f"{name}: WHERE {clause} → time={elapsed:.4f} sec, results={count}, speedup={speedup:.2f}x")

        rows.append({
            "filter": name,
            "latency_sec": elapsed,
            "result_count": count,
            "speedup_vs_no_filter": speedup
        })

    return pd.DataFrame(rows)


def memory_profile(con, index_name):
    """
    Memory/storage profiling:
    1. SHOW TABLES to find all physical tables.
    2. Filter tables whose names contain the index prefix.
    3. Run PRAGMA storage_info('<table>') for each.
    4. Aggregate compressed and uncompressed sizes.
    """
    print(f"\n=== Memory Profile for index '{index_name}' ===")

    # List all tables
    tables = con.execute("SHOW TABLES").fetchall()
    table_names = [t[0] for t in tables]

    # Filter tables belonging to this index
    related = [t for t in table_names if index_name in t]

    if not related:
        print(f"No physical tables found for index prefix '{index_name}'.")
        return pd.DataFrame()

    rows = []

    # Run storage_info for each matching table
    for tbl in related:
        try:
            df = con.execute(f"PRAGMA storage_info('{tbl}')").df()
            if not df.empty:
                if "total_compressed_size" in df.columns and "total_size" in df.columns:
                    # DuckDB >= 1.6.x
                    total_compressed = df["total_compressed_size"].sum()
                    total_uncompressed = df["total_size"].sum()
                else:
                    # DuckDB <= 1.4.x fallback
                    if "segment_size" in df.columns:
                        total_compressed = df["segment_size"].sum()
                        total_uncompressed = df["segment_size"].sum()
                    else:
                        total_compressed = 0
                        total_uncompressed = 0

                rows.append({
                    "table": tbl,
                    "compressed_mb": total_compressed / (1024 * 1024),
                    "uncompressed_mb": total_uncompressed / (1024 * 1024)
                })
        except Exception as e:
            print(f"Error reading storage_info for {tbl}: {e}")

    mem_df = pd.DataFrame(rows)

    print("\nPer-table size (MB):")
    print(mem_df)

    print("\nTotal index size (MB):")
    print(mem_df[["compressed_mb", "uncompressed_mb"]].sum())

    return mem_df



# MAIN


def main():
    con = connect_and_load()
    check_base_schema(con)

    # Sample queries for recall evaluation
    #queries = sample_queries(con, NUM_QUERY_VECS)
    queries = [row[0] for row in con.execute("SELECT vec FROM sift_query LIMIT 1000").fetchall()]

    all_results = []

    filter_query = queries[0] if queries else None

    for num_clusters in CLUSTER_SWEEP:

        
        # IVFPQ
        
        pq_index = f"ivfpq_{num_clusters}"

        if index_exists(con, pq_index):
            print(f"Index '{pq_index}' already exists. Skipping build.")
            pq_build_time = 0.0
            pq_mem_mb = 0.0
        else:
            pq_build_time, pq_mem_mb = build_index(con, pq_index, num_clusters, use_pq=True)
        for nprobe_cap in NPROBE_SWEEP:
            for k_val in [1, 10, 100]:
                res_pq = eval_recall_latency(
                    con,
                    pq_index,
                    queries,
                    k=k_val,
                    nprobe=nprobe_cap,
                    label=f"IVFPQ_{num_clusters}_k{k_val}_nprobe{nprobe_cap}"
                )
                res_pq["num_clusters"] = num_clusters
                res_pq["index_type"] = "IVFPQ"
                res_pq["build_time_sec"] = pq_build_time
                res_pq["file_mb"] = pq_mem_mb
                all_results.append(res_pq)

        if filter_query is not None:
            df_filters_pq = eval_filter_selectivity(con, pq_index, filter_query, k=10, nprobe=DEFAULT_NPROBE)
            print("\nFilter selectivity (IVFPQ) summary:")
            print(df_filters_pq)

    
    # Final summary tables
    
    results_df = pd.DataFrame(all_results)
    print("\n==================== OVERALL ANN RESULTS ====================")
    print(results_df)


    import matplotlib.pyplot as plt

    # Plot Recall vs nprobe for k=1,10,100
    for k_val in [1, 10, 100]:
        plt.figure()
        for num_clusters in CLUSTER_SWEEP:
            sub = results_df[(results_df["k"] == k_val) & (results_df["num_clusters"] == num_clusters)]
            plt.plot(sub["nprobe"], sub["recall_at_k"], marker="o", label=f"clusters={num_clusters}")
        plt.title(f"Recall@{k_val} vs nprobe")
        plt.xlabel("nprobe")
        plt.ylabel("Recall")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"recall_k{k_val}.png")

    # Plot Latency vs nprobe for k=10
    plt.figure()
    for num_clusters in CLUSTER_SWEEP:
        sub = results_df[(results_df["k"] == 10) & (results_df["num_clusters"] == num_clusters)]
        plt.plot(sub["nprobe"], sub["avg_ann_ms"], marker="o", label=f"clusters={num_clusters}")
    plt.title("ANN Latency vs nprobe (k=10)")
    plt.xlabel("nprobe")
    plt.ylabel("ANN Latency (ms)")
    plt.grid(True)
    plt.legend()
    plt.savefig("latency_k10.png")

    print("Saved plots: recall_k1.png, recall_k10.png, recall_k100.png, latency_k10.png")

    # save to CSV for plotting
    results_df.to_csv("ivf_eval_results.csv", index=False)
    print("\nSaved ivf_eval_results.csv and ivf_memory_profile.csv.")


if __name__ == "__main__":
    main()
