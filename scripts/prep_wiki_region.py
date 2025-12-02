import time
import os
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset

# --- Config ---
DATASET_NAME = "Cohere/wikipedia-22-12-simple-embeddings"
OUTPUT_DIR = "data"
BASE_OUTPUT = f"{OUTPUT_DIR}/wiki_base_full.parquet"
QUERY_OUTPUT = f"{OUTPUT_DIR}/wiki_query_full.parquet"
BATCH_SIZE = 100000

# Metadata Logic (Matching your SQL snippet)
REGIONS = ["US", "EU", "ASIA"]
CATEGORIES = ["Tech", "Fashion", "Sports", "Health"]

def get_region(row_id):
    return REGIONS[row_id % 3]

def get_category(row_id):
    return CATEGORIES[row_id % 4]

def main():
    print(f"--- Downloading Full Dataset & Adding Metadata ---")
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

    # Define Schema (now with Region and Category)
    schema = pa.schema([
        ('id', pa.int64()),
        ('vec', pa.list_(pa.float32())),
        ('region', pa.string()),
        ('category', pa.string())
    ])

    dataset = load_dataset(DATASET_NAME, split="train", streaming=True)
    dataset_iter = iter(dataset)
    writer = pq.ParquetWriter(BASE_OUTPUT, schema)
    
    count = 0
    batch_data = {'id': [], 'vec': [], 'region': [], 'category': []}
    
    # 1. Process Base Vectors (All available)
    start_time = time.time()
    try:
        for row in dataset_iter:
            # Skip the first 100 to save them for queries (optional, or just take from end)
            # Let's download EVERYTHING first.
            
            row_id = int(row['id'])
            
            batch_data['id'].append(row_id)
            batch_data['vec'].append(row['emb'])
            batch_data['region'].append(get_region(row_id))
            batch_data['category'].append(get_category(row_id))
            
            count += 1
            
            if len(batch_data['id']) >= BATCH_SIZE:
                table = pa.Table.from_pydict(batch_data, schema=schema)
                writer.write_table(table)
                print(f"  -> Flushed {count} vectors...")
                batch_data = {'id': [], 'vec': [], 'region': [], 'category': []}

        # Flush remaining
        if batch_data['id']:
            table = pa.Table.from_pydict(batch_data, schema=schema)
            writer.write_table(table)

    finally:
        writer.close()
        print(f"✅ Saved {count} vectors to {BASE_OUTPUT}")

    # 2. Create Query Set (Take last 100 IDs for testing)
    # We will read from the file we just created to be safe
    print("Extracting queries from the end of the dataset...")
    
    # Use DuckDB to efficiently grab the last 100 rows
    import duckdb
    con = duckdb.connect()
    con.execute(f"COPY (SELECT * FROM '{BASE_OUTPUT}' ORDER BY id DESC LIMIT 100) TO '{QUERY_OUTPUT}' (FORMAT PARQUET)")
    print(f"✅ Saved query vectors to {QUERY_OUTPUT}")

if __name__ == "__main__":
    main()