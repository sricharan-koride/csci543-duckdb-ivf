#pragma once

#include "duckdb.hpp"
#include <vector>

namespace duckdb {

struct IVFSearchFunctionData : public TableFunctionData {
    string index_name;
    int32_t k;
    int32_t nprobe;
    std::vector<float> query_vector;
};

// --- State object to hold loaded data AND results ---
struct IVFSearchGlobalState : public GlobalTableFunctionState {
    // 1. Loaded Index Data
    std::vector<std::vector<float>> centroids;

    // 2. Search Results Buffer
    // We compute everything in the first call, store it here, 
    // and then stream it out to DuckDB.
    struct SearchResult {
        int64_t id;
        float score;
    };
    std::vector<SearchResult> final_results;
    idx_t current_offset = 0; // Tracks how many we've sent so far
};

unique_ptr<FunctionData> BindIVFSearch(ClientContext &context, TableFunctionBindInput &input,
                                       vector<LogicalType> &return_types, vector<string> &names);

unique_ptr<GlobalTableFunctionState> InitIVFSearch(ClientContext &context, TableFunctionInitInput &input);

void ComputeIVFSearch(ClientContext &context, TableFunctionInput &data_p, DataChunk &output);

} // namespace duckdb