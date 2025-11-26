import duckdb
import numpy as np

# Connect to your existing database
con = duckdb.connect("sift_data.db", config={'allow_unsigned_extensions': 'true'})

# Load your extension
con.execute("LOAD 'build/release/extension/ivf/ivf.duckdb_extension'")

# 1. Get a real query vector from your table
# We fetch it as a Python object (list of floats)
query_vector = con.execute("SELECT vec FROM sift_base LIMIT 1").fetchone()[0]

print(f"Query vector retrieved (Dimension: {len(query_vector)})")

# 2. Run your ANN Search function
# We pass the 'query_vector' list directly into the SQL as a parameter (?)
try:
    results = con.execute(
        "SELECT * FROM ann_search('my_index', ?::FLOAT[128], 10, 5)", 
        [query_vector]
    ).fetchall()

    print("\n--- Search Results ---")
    print(results)
    print("----------------------")
    print("✅ Success! The C++ skeleton is working.")

except Exception as e:
    print("\n❌ Error:")
    print(e)