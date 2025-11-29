#!/usr/bin/env python3
"""
Test script for Hybrid Analytical Filtering pushdown.
Creates a small table with region/category/price fields, builds an IVF index,
then runs several queries with WHERE predicates to ensure the optimizer
pushes down simple predicates into the inner candidate SQL.

This script attempts to load the extension via the Python duckdb API; if
that fails (ABI mismatch) it falls back to using the built duckdb CLI.

Usage:
  python scripts/test_hybrid_analytical.py
"""

import os
import subprocess
import tempfile
import duckdb
import random
import sys

DB = 'sift_data.db'
EXT_PATH = 'build/release/extension/ivf/ivf.duckdb_extension'
DUCKDB_CLI = os.path.join('build', 'release', 'duckdb.exe')

DIM = 64
N = 50

# We'll detect the actual vector dimensionality after creating/inspecting the table
def detect_dim_from_db(con=None):
    # If a DuckDB connection is provided, use it. Otherwise open one with
    # the same relaxed config to avoid "different configuration" errors.
    own = False
    if con is None:
        con = duckdb.connect(DB, config={'allow_unsigned_extensions': 'true'})
        own = True
    try:
        row = con.execute('SELECT vec FROM sift_base LIMIT 1').fetchone()
        if row and row[0] is not None:
            v = row[0]
            return len(v)
    except Exception:
        pass
    finally:
        if own:
            con.close()
    return DIM


def ensure_sample_db():
    con = duckdb.connect(DB)
    try:
        con.execute('SELECT 1 FROM sift_base LIMIT 1')
        con.close()
        return
    except Exception:
        pass

    print('Creating sample table sift_base with', N, 'vectors')
    con.execute("CREATE OR REPLACE TABLE sift_base (id BIGINT, vec FLOAT[], region VARCHAR, category VARCHAR, price DOUBLE)")
    categories = ['tech', 'home', 'sports']
    for i in range(N):
        vec = [random.random() for _ in range(DIM)]
        region = 'US' if (i % 2 == 0) else 'EU'
        category = categories[i % len(categories)]
        price = random.random() * 100.0
        con.execute('INSERT INTO sift_base VALUES (?, ?, ?, ?, ?)', [i, vec, region, category, price])
    con.close()
    print('Sample DB ready:', DB)


def run_tests_via_python():
    print('\n=== Running analytical pushdown test via Python API ===')
    con = duckdb.connect(DB, config={'allow_unsigned_extensions': 'true'})
    loaded = False
    try:
        con.execute(f"LOAD '{EXT_PATH}'")
        loaded = True
        print('Extension loaded via Python API')
    except Exception as e:
        print('Could not load extension via Python API:', e)

    # Build index (best-effort)
    try:
        print('Calling create_ivf_index...')
        con.execute("SELECT create_ivf_index('my_index', 'sift_base', 'vec', 'id')")
        print('create_ivf_index finished')
    except Exception as e:
        print('create_ivf_index failed (continuing):', e)

    # Sample predicates to test
    predicates = [
        "region = 'US'",
        "category = 'tech'",
        "price > 50",
        "region = 'US' AND price > 20",
        "category IN ('tech','home') AND price < 80",
    ]

    actual_dim = detect_dim_from_db(con)
    for p in predicates:
        print('\n--- EXPLAIN for predicate:', p, '---')
        try:
            explain = con.execute(f"EXPLAIN SELECT id, score FROM ann_search('my_index', ?::FLOAT[{actual_dim}], 10, 5) WHERE {p}", [con.execute('SELECT vec FROM sift_base LIMIT 1').fetchone()[0]]).fetchall()
            for row in explain:
                print(row[0])
        except Exception as e:
            print('EXPLAIN failed:', e)

        print('\n--- RUN for predicate:', p, '---')
        try:
            results = con.execute(f"SELECT id, score FROM ann_search('my_index', ?::FLOAT[{actual_dim}], 10, 5) WHERE {p}", [con.execute('SELECT vec FROM sift_base LIMIT 1').fetchone()[0]]).fetchall()
            print('Results:', results[:10])
        except Exception as e:
            print('Search failed via Python API:', e)

    con.close()
    return loaded


def run_tests_via_cli():
    print('\n=== Running analytical pushdown test via duckdb CLI fallback ===')
    cli = DUCKDB_CLI if os.path.exists(DUCKDB_CLI) else 'duckdb'
    ext = EXT_PATH.replace('\\', '/')
    print('Using CLI:', cli)

    # 1) Create index via CLI
    sql_create = f"LOAD '{ext}'; SELECT create_ivf_index('my_index', 'sift_base', 'vec', 'id');"
    try:
        subprocess.check_call(f"{cli} {DB} -c \"{sql_create}\"", shell=True)
        print('create_ivf_index succeeded (CLI)')
    except subprocess.CalledProcessError as e:
        print('create_ivf_index CLI failed (continuing):', e)

    # Predicates to test
    predicates = [
        "region = 'US'",
        "category = 'tech'",
        "price > 50",
        "region = 'US' AND price > 20",
        "category IN ('tech','home') AND price < 80",
    ]

    # prepare a query vector literal
    tmp_vec = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
    tmp_vec.close()
    try:
        cmd_vec = f"{cli} {DB} -c \"COPY (SELECT array_to_string(vec, ',') FROM sift_base LIMIT 1) TO '{tmp_vec.name.replace('\\','/')}' (FORMAT CSV, HEADER false);\""
        subprocess.check_call(cmd_vec, shell=True)
        with open(tmp_vec.name, 'r') as f:
            vec_line = f.read().strip().splitlines()[-1]
    finally:
        try:
            os.remove(tmp_vec.name)
        except Exception:
            pass

    # Determine actual dimensionality of the extracted vector from the CSV line
    try:
        actual_dim = vec_line.count(',') + 1
    except Exception:
        actual_dim = DIM
    array_literal = f"ARRAY[{vec_line}]::FLOAT[{actual_dim}]"

    for p in predicates:
        explain_sql = f"LOAD '{ext}'; EXPLAIN SELECT id, score FROM ann_search('my_index', {array_literal}, 10, 5) WHERE {p};"
        try:
            print('\n--- EXPLAIN for predicate:', p, '---')
            subprocess.check_call(f"{cli} {DB} -c \"{explain_sql}\"", shell=True)
        except subprocess.CalledProcessError as e:
            print('EXPLAIN via CLI failed:', e)

        tmp_res = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
        tmp_res.close()
        search_sql = f"LOAD '{ext}'; COPY (SELECT id, score FROM ann_search('my_index', {array_literal}, 10, 5) WHERE {p}) TO '{tmp_res.name.replace('\\','/')}' (FORMAT CSV, HEADER false);"
        try:
            print('\n--- RUN for predicate:', p, '---')
            subprocess.check_call(f"{cli} {DB} -c \"{search_sql}\"", shell=True)
            with open(tmp_res.name, 'r') as f:
                out = [l.strip() for l in f.read().splitlines() if l.strip()]
            res = []
            for line in out:
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 2:
                    try:
                        a = int(parts[0]); b = float(parts[1]); res.append((a, b))
                    except Exception:
                        res.append(tuple(parts))
            print('Results (CLI):', res[:10])
        except subprocess.CalledProcessError as e:
            print('Search via CLI failed:', e)
        finally:
            try:
                os.remove(tmp_res.name)
            except Exception:
                pass


if __name__ == '__main__':
    ensure_sample_db()
    loaded = run_tests_via_python()
    if not loaded:
        print('\nPython API path unavailable, running CLI fallback...')
        run_tests_via_cli()
    print('\nAnalytical pushdown test complete.')
