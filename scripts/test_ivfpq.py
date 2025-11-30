import duckdb
import time
import numpy as np

DB = "sift_data.db"
EXT = "build/release/extension/ivf/ivf.duckdb_extension"

# -------------------------------
# Connect + Load extension
# -------------------------------
con = duckdb.connect(DB, config={'allow_unsigned_extensions': 'true'})
con.execute(f"LOAD '{EXT}'")
print("Extension loaded.")

# -------------------------------
# Confirm base table exists
# -------------------------------
print("\n--- Checking table schema ---")
print(con.execute("PRAGMA table_info('sift_base')").df())


# -------------------------------
# STEP 1 — Recreate the IVF-PQ index
# -------------------------------
print("\n=== Building IVF-PQ Index ===")
t0 = time.time()

con.execute("""
    SELECT create_ivf_index(
        'my_index',
        'sift_base',
        'vec',
        1024
    )
""")

t1 = time.time()
print(f"Index build time: {t1 - t0:.2f} sec")


# -------------------------------
# STEP 2 — Sanity check: PQ tables exist
# -------------------------------
print("\n--- Checking PQ tables ---")
print(con.execute("SHOW TABLES").df())


# -------------------------------
# STEP 3 — Fetch one query vector
# -------------------------------
query = con.execute("SELECT vec FROM sift_base LIMIT 1").fetchone()[0]
print("\n--- Query vector dimension:", len(query), "---")


# -------------------------------
# STEP 4 — IVF-PQ Search (nprobe sweep + k=100 candidates)
# -------------------------------
print("\n=== IVF-PQ Sweep (k=100 candidates) ===")

nprobes = [16, 32, 64, 128]
sweep_results = []

for npb in nprobes:
    print(f"\n--- nprobe = {npb} ---")
    t0 = time.time()

    pq_rows = con.execute("""
        SELECT *
        FROM ann_search('my_index', ?::FLOAT[128], 100, ?)
    """, [query, npb]).fetchall()

    elapsed = time.time() - t0
    pq_ids_10 = [r[0] for r in pq_rows[:10]]
    sweep_results.append((npb, elapsed, pq_ids_10))

    print(f"IVF-PQ time: {elapsed:.4f} sec")
    print("Top-10 IVF-PQ IDs:", pq_ids_10)

# Store last pq_rows for recall check
results_pq = pq_rows[:10]


# -------------------------------
# STEP 5 — Exact Search for recall check
# -------------------------------
print("\n=== Running Exact Search (Baseline) ===")
t0 = time.time()

results_exact = con.execute("""
    SELECT id, vec <-> ? AS dist
    FROM sift_base
    ORDER BY dist
    LIMIT 10
""", [query]).fetchall()

t1 = time.time()
print(f"Exact search time: {t1 - t0:.4f} sec")


# -------------------------------
# STEP 6 — Recall@K
# -------------------------------
exact_ids = [r[0] for r in results_exact]
pq_ids    = [r[0] for r in results_pq]

recall = len(set(exact_ids) & set(pq_ids)) / len(exact_ids)
print(f"\nRecall@10 = {recall*100:.2f}%")
print("Exact IDs:", exact_ids)
print("PQ IDs:   ", pq_ids)


# -------------------------------
# STEP 7 — Test hybrid WHERE filter
# -------------------------------
print("\n=== Testing Hybrid Filtering ===")
filtered = con.execute("""
    SELECT *
    FROM ann_search(
        'my_index',
        ?::FLOAT[128],
        5, 32,
        where_clause := 'region=''US'''
    )
""", [query]).fetchall()

print("Filtered results:", filtered)
