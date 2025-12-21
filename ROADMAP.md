# DuckDB IVF Extension Roadmap

This document outlines the development roadmap to transition the IVF Extension from a working prototype to a complete, production-ready vector search solution for DuckDB.

## Phase 1: Core Vector Capabilities (High Priority)
*Focus: Essential features for LLM/AI workloads and raw performance.*

### 1.1 Support for Cosine Similarity & Inner Product
- **Goal**: Support standard distance metrics required by OpenAI/Cohere/HuggingFace embeddings, not just L2.
- **Tasks**:
    - [x] Refactor `L2SquaredDistance` into a `DistanceMetric` interface/enum.
    - [x] Implement `CosineSimilarity` (normalize vectors + dot product).
    - [x] Implement `InnerProduct` (pure dot product).
    - [x] Update `create_ivf_index` scalar function to accept a `metric` argument (default to 'l2').
    - [x] Persist the chosen metric in `ivf_index_metadata`.

### 1.2 SIMD Optimizations (AVX2/AVX-512/Neon)
- **Goal**: 4x-8x speedup in distance calculations, which is the bottleneck for search.
- **Tasks**:
    - [ ] Add `SimdUtils` class with architecture detection.
    - [ ] Implement AVX2 intrinsics for L2 and Dot Product.
    - [ ] (Optional) Implement AVX-512 or ARM Neon intrinsics.
    - [ ] Benchmark before/after latency.

---

## Phase 2: Production Hardening
*Focus: Stability, correctness, and code quality.*

### 2.1 Error Handling & Logging
- **Goal**: Remove `printf` debugging and use DuckDB's native error propagation.
- **Tasks**:
    - [ ] Replace `printf` with `duckdb::throw_exception` for critical failures.
    - [ ] Use `context.client_context->log_query_progress` or similar for long-running operations like clustering.
    - [ ] Validate inputs (e.g., ensure `nprobe` > 0, `k` > 0).

### 2.2 Cleaner SQL Interface
- **Goal**: Make the extension feel native to SQL users.
- **Tasks**:
    - [ ] Implement a `DROP_IVF_INDEX('index_name')` function to clean up the `ivf_centroids`, `ivf_lists`, `ivf_pq_codes`, and metadata tables automatically.
    - [ ] (Stretch) Investigate `CREATE INDEX` syntax hooks (requires deeper DuckDB parser integration, potentially rewriting as a standard DuckDB Index extension rather than scalar/table functions).

---

## Phase 3: Lifecycle Management
*Focus: Real-world usage where data changes.*

### 3.1 Index Refresh Strategy
- **Goal**: Allow the index to reflect changes in the base table without a full rebuild.
- **Tasks**:
    - [ ] Implement `REFRESH_IVF_INDEX('index_name')`.
    - [ ] **Strategy A (Full Rebuild)**: Simplest. Drop and recreate.
    - [ ] **Strategy B (Incremental)**: Identify new rows (via strictly increasing ID?) and insert them into existing clusters. *Note: Clustering centers will drift over time.*

### 3.2 Metadata Management
- **Goal**: Prevent orphaned tables.
- **Tasks**:
    - [ ] Add foreign key-like checks or dependency tracking to ensure that if the base table is dropped, the index tables are warned about or cleaned up.

---

## Phase 4: Advanced Features (Long Term)

### 4.1 HNSW Integration
- **Goal**: Compare or combine IVF with HNSW (Graph-based) search for higher recall.
- **Strategy**: Potential integration with `duckdb-vss` or implementing a graph layer inside the coarse quantizer.

### 4.2 Binary Quantization (BQ)
- **Goal**: Ultra-fast search for high-dimensional vectors (compression to bits).
- **Tasks**:
    - [ ] Implement bit-packing for float vectors.
    - [ ] Implement Hamming distance.
