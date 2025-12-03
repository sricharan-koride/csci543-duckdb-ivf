import duckdb
import os

# Config
DB_PATH = "sift_data.db"
EXT_PATH = "build/release/extension/ivf/ivf.duckdb_extension"
INDEX_NAME = "sift_idx"

def main():
    if not os.path.exists(DB_PATH):
        print(f"Error: {DB_PATH} not found. Run your SIFT tests first.")
        return

    print(f"Connecting to {DB_PATH}...")
    con = duckdb.connect(DB_PATH, config={'allow_unsigned_extensions': 'true'})
    con.execute(f"LOAD '{EXT_PATH}'")

    # Calculate Raw Vector Size 
    # SIFT has 1M vectors of 128 dimensions 
    print("Calculating Raw Data Size...")
    res = con.execute("SELECT COUNT(*) FROM sift_base").fetchone()
    num_rows = res[0]
    
    # Each float is 4 bytes
    dim = 128
    raw_size_bytes = num_rows * dim * 4
    
    print(f"  Rows: {num_rows:,}")
    print(f"  Dimensions: {dim}")
    print(f"  Raw Size: {raw_size_bytes / (1024*1024):.2f} MB")

    # Calculate PQ Index Size 
    # We query the internal table 'ivf_pq_codes_my_index' that your extension created
    print("\nCalculating PQ Index Size...")
    pq_table = f"ivf_pq_codes_{INDEX_NAME}"
    
    try:
        # We sum the length of the 'code' blob column
        pq_size_bytes = con.execute(f"SELECT SUM(len(code)) FROM {pq_table}").fetchone()[0]
        
        
        print(f"  PQ Table: {pq_table}")
        print(f"  Compressed Size: {pq_size_bytes / (1024*1024):.2f} MB")
        
        # The Ratio
        ratio = raw_size_bytes / pq_size_bytes
        print(f"\n Compression Ratio: {ratio:.1f}x")
        
    except Exception as e:
        print(f"Error: Could not find PQ table '{pq_table}'. Did you build the index?")
        print(e)

if __name__ == "__main__":
    main()