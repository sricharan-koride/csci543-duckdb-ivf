#!/usr/bin/env python3
"""
Simple smoke test for predicate pushdown support.
- Creates a small DB `pushdown_test.db` with table `items(id, vec, region)`
- Builds an IVF index via duckdb CLI (falls back to local duckdb if build path missing)
- Calls `ann_search` via CLI with the named parameter `where_clause=>'region = ''US'''`
- Verifies that returned ids are a subset of rows with region='US'

This test uses the CLI fallback exclusively for extension operations to avoid
Python ABI extension-load issues.
"""

import os
import duckdb
import random
import sys

DB = 'sift_data.db'  # use the same DB as test_search.py when possible
DIM = 128
N = 12


def setup_db():
    con = duckdb.connect(DB)
    # If a table named 'items' already exists in this DB (e.g. from other tests), reuse it.
    tbls = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()] if False else []
    # DuckDB doesn't have sqlite_master; use a safe existence check
    try:
        exists = con.execute("SELECT 1 FROM information_schema.tables WHERE table_name='items'").fetchone()
    except Exception:
        exists = None

    if not exists:
        con.execute("CREATE OR REPLACE TABLE items(id BIGINT, vec FLOAT[], region VARCHAR)")
        for i in range(N):
            vec = [random.random() for _ in range(DIM)]
            region = 'US' if (i % 3 != 0) else 'EU'  # roughly 2/3 US, 1/3 EU
            con.execute('INSERT INTO items VALUES (?, ?, ?)', [i, vec, region])
        print('Created sample table items with', N, 'rows in', DB)
    else:
        print("Table 'items' exists in", DB, "— reusing it for the pushdown test")
    con.close()


def cli_create_index():
    con = con = duckdb.connect(DB, config={"allow_unsigned_extensions": "true"})
    con.execute(f"LOAD 'build/release/extension/ivf/ivf.duckdb_extension'")
    con.execute("SELECT create_ivf_index('my_index', 'items', 'vec', 'id')")
    con.close()
    print("Index created via Python API")
    return True


def cli_run_search_and_get_ids():
    con = con = duckdb.connect(DB, config={"allow_unsigned_extensions": "true"})
    con.execute(f"LOAD 'build/release/extension/ivf/ivf.duckdb_extension'")
    row = con.execute("SELECT vec FROM items LIMIT 1").fetchone()
    if not row:
        print('No vectors found in items table')
        con.close()
        sys.exit(4)
    query_vec = row[0]
    dim = len(query_vec)
    ids = [r[0] for r in con.execute(
        f"SELECT id FROM ann_search('my_index', ?::FLOAT[{dim}], 5, 2, where_clause=>'region = ''US''')",
        [query_vec]
    ).fetchall()]
    con.close()
    return ids


if __name__ == '__main__':
    setup_db()
    ok = cli_create_index()
    if not ok:
        print('Index creation failed; aborting test')
        sys.exit(2)

    ids = cli_run_search_and_get_ids()
    print('ANN search returned ids:', ids)

    # verify these ids are subset of items with region='US'
    con = con = duckdb.connect(DB, config={"allow_unsigned_extensions": "true"})
    us_ids = [r[0] for r in con.execute("SELECT id FROM items WHERE region = 'US'").fetchall()]
    con.close()
    print('Expected US ids:', us_ids)
    # Fail if search returned no ids — this is unexpected for the smoke test
    if len(ids) == 0:
        print('TEST FAILED: Search returned no ids. Likely degenerate index (too few samples) or faulty clustering')
        sys.exit(1)

    if not set(ids).issubset(set(us_ids)):
        print('TEST FAILED: Some returned ids do not belong to region=US rows')
        sys.exit(1)

    print('TEST PASSED: All returned ids match region=US and result set was non-empty')
    sys.exit(0)
