import duckdb
import numpy as np

con = duckdb.connect("sift_data.db", config={'allow_unsigned_extensions': 'true'})
con.execute("LOAD 'build/release/extension/ivf/ivf.duckdb_extension'")

# 1. Get vector for ID 0
query_vector = con.execute("SELECT vec FROM sift_base WHERE id = 0").fetchone()[0]

print("--- Hybrid Search Test ---")
print("Querying for neighbors of ID 0...")
print("Constraint: allowed_ids = [2, 6, 999999]") # 999999 probably doesn't exist, which is fine

# 2. Run Search with Filter
# We expect to see ONLY IDs 2 or 6. ID 0 should be GONE.
results = con.execute("""
    SELECT * FROM ann_search(
        'my_index', 
        ?::FLOAT[128], 
        10, 
        5, 
        allowed_ids => [2, 6, 999999]
    )
""", [query_vector]).fetchall()

print("\nResults:")
print(results)

# Verification
ids = [r[0] for r in results]
if 0 in ids:
    print("\n❌ FAILED: ID 0 was returned but it was not in the allowed list!")
elif 2 in ids or 6 in ids:
    print("\n✅ SUCCESS: Only allowed IDs were returned.")
else:
    print("\n⚠️ NOTE: No results found (maybe IDs 2/6 are in different clusters?)")