import duckdb
import os

BASE_FILE = "data/wiki_base.parquet"
QUERY_FILE = "data/wiki_query.parquet"
TEMP_BASE = "data/wiki_base_fixed.parquet"

def main():
    print("Fixing dataset splits...")
    con = duckdb.connect()
    
    # 1. Get total count from the file you already downloaded
    total_rows = con.execute(f"SELECT COUNT(*) FROM '{BASE_FILE}'").fetchone()[0]
    print(f"Total rows available: {total_rows}")
    
    # We will take the last 100 as queries, and the rest as base
    num_queries = 100
    num_base = total_rows - num_queries
    
    print(f"Splitting into: {num_base} Base Vectors + {num_queries} Query Vectors")

    # 2. Save the Query Vectors (The last 100)
    # We use OFFSET to skip the base vectors
    con.execute(f"""
        COPY (SELECT * FROM '{BASE_FILE}' LIMIT {num_queries} OFFSET {num_base}) 
        TO '{QUERY_FILE}' (FORMAT PARQUET)
    """)
    print(f"✅ Saved {QUERY_FILE}")

    # 3. Save the Base Vectors (The first N-100)
    # We save to a temp file first so we don't corrupt the source while reading
    con.execute(f"""
        COPY (SELECT * FROM '{BASE_FILE}' LIMIT {num_base}) 
        TO '{TEMP_BASE}' (FORMAT PARQUET)
    """)
    print(f"✅ Saved {TEMP_BASE}")

    # 4. Overwrite the original base file
    os.replace(TEMP_BASE, BASE_FILE)
    print(f"✅ Replaced original {BASE_FILE}")

if __name__ == "__main__":
    main()