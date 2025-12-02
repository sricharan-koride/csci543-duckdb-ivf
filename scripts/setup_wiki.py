import duckdb
import os

DB_PATH = "wiki_full.db"
EXT_PATH = "build/release/extension/ivf/ivf.duckdb_extension"
BASE_PARQUET = "data/wiki_base_full.parquet"

if os.path.exists(DB_PATH): os.remove(DB_PATH)
if os.path.exists(DB_PATH + ".wal"): os.remove(DB_PATH + ".wal")

print(f"Creating database {DB_PATH}...")
con = duckdb.connect(DB_PATH, config={'allow_unsigned_extensions': 'true'})
con.execute(f"LOAD '{EXT_PATH}'")

# Get Dimension
dim = len(con.execute(f"SELECT vec FROM read_parquet('{BASE_PARQUET}') LIMIT 1").fetchone()[0])
print(f"Vector dim: {dim}")

print("Creating table 'wiki_base'...")
# Note: We keep the region/category columns!
con.execute(f"""
    CREATE TABLE wiki_base AS 
    SELECT id, vec::FLOAT[{dim}] AS vec, region, category
    FROM read_parquet('{BASE_PARQUET}')
""")

print("✅ Database ready.")