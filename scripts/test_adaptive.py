import duckdb
import numpy as np

con = duckdb.connect("sift_data.db", config={'allow_unsigned_extensions': 'true'})
con.execute("LOAD 'build/release/extension/ivf/ivf.duckdb_extension'")

# 1. Get vector for ID 0 (This vector definitely lives in the closest cluster)
query_vector = con.execute("SELECT vec FROM sift_base WHERE id = 0").fetchone()[0]

print("--- Adaptive Search Test ---")
print("We ask for k=3 results, but we restrict the search to distant ID 800000.")
print("Normal search (nprobe=1) would find NOTHING because ID 800000 is far away.")
print("Adaptive search should keep scanning until it finds it.")

# 2. Run Search
# nprobe = 1 (Start small)
# allowed_ids = [800000] (Force it to keep looking)
results = con.execute("""
    SELECT * FROM ann_search(
        'my_index', 
        ?::FLOAT[128], 
        3, 
        1, 
        allowed_ids => [800000]
    )
""", [query_vector]).fetchall()

print("\nResults:")
print(results)

if len(results) > 0 and results[0][0] == 800000:
    print("\n✅ SUCCESS: The search adapted and expanded to find the distant ID!")
else:
    print("\n❌ FAILED: No results found.")