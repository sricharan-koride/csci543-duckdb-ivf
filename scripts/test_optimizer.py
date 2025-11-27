import duckdb

# Connect (allow unsigned extensions)
con = duckdb.connect("sift_data.db", config={'allow_unsigned_extensions': 'true'})
con.execute("LOAD 'build/release/extension/ivf/ivf.duckdb_extension'")

# Get a query vector (ID 0)
try:
    query_vec = con.execute("SELECT vec FROM sift_base LIMIT 1").fetchone()[0]
except:
    print("❌ Error: Could not fetch query vector. Is sift_base loaded?")
    exit(1)

print("--- Optimizer Integration Test ---")
print("Query: SELECT * FROM ann_search(...) WHERE id IN (2, 6)")
print("Expectation: The C++ Optimizer should detect this and print a rocket 🚀 message.")

# We run a NORMAL SQL query with a WHERE clause.
# We do NOT pass 'allowed_ids' manually. The optimizer should do it for us.
results = con.execute("""
    SELECT * FROM ann_search(
        'my_index', 
        ?::FLOAT[128], 
        10, 
        5
    )
    WHERE id IN (2, 6) 
""", [query_vec]).fetchall()

print("\nResults:")
print(results)

# Verify
ids = [r[0] for r in results]
if len(ids) == 2 and 2 in ids and 6 in ids:
    print("\n✅ SUCCESS: The optimizer pushed the filter down! Only IDs 2 and 6 were searched.")
else:
    print("\n❌ FAILURE: The filter was applied AFTER search (or not at all).")